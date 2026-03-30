"""
train.py — Training Pipeline for MR-AFCNN
==========================================
Features:
    - Fold-aware cross-validation (ESC-50: 5-fold, US8K: 10-fold)
    - Mixed precision training (torch.cuda.amp)
    - Cosine annealing LR with warm restarts (CosineAnnealingWarmRestarts)
    - AdamW optimizer with weight decay
    - Mixup augmentation (batch-level, from dataset.py)
    - Gradient clipping
    - Early stopping (patience-based)
    - Model checkpointing (best val acc per fold)
    - TensorBoard logging (optional)
    - Resumable from checkpoint

Usage:
    # Train ESC-50, all 5 folds:
    python train.py --dataset esc50 --root ./ESC-50 --epochs 200

    # Train UrbanSound8K, fold 1 only:
    python train.py --dataset us8k --root ./UrbanSound8K --folds 1 --epochs 150

    # Resume from checkpoint:
    python train.py --dataset esc50 --root ./ESC-50 --resume ./checkpoints/esc50_fold1_best.pt
"""

import os
import time
import argparse
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.optim.swa_utils import AveragedModel, SWALR

from model   import MRAFCNN, MRAFCNNLoss, build_esc50, build_urbansound8k
from dataset import (
    ESC50Dataset, UrbanSound8KDataset,
    get_dataloader, soft_cross_entropy,
    esc50_fold_splits, us8k_fold_splits,
    CFG
)


# ---------------------------------------------------------------------------
# 1.  REPRODUCIBILITY
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For fixed-size convolutions (like 128x128 mels), benchmark=True is much faster
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True
    
    # Enable TF32 for faster training on Ampere/Ada GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True



# ---------------------------------------------------------------------------
# 2.  TRAINING CONFIG
# ---------------------------------------------------------------------------

class TrainConfig:
    # Optimizer
    LR:            float = 3e-4
    WEIGHT_DECAY:  float = 1e-4
    BETAS:         tuple = (0.9, 0.999)

    # LR schedule: CosineAnnealingWarmRestarts
    T0:            int   = 50       # first restart cycle length (epochs)
    T_MULT:        int   = 2        # cycle length multiplier after each restart
    ETA_MIN:       float = 1e-6

    # Warmup
    WARMUP_EPOCHS: int   = 10

    # Training
    BATCH_SIZE:    int   = 32
    NUM_WORKERS:   int   = 0
    EPOCHS:        int   = 300
    GRAD_CLIP:     float = 5.0

    # Mixup
    USE_MIXUP:     bool  = True
    MIXUP_ALPHA:   float = 0.4

    # Loss weights (see MRAFCNNLoss)
    LAMBDA_CE:     float = 0.4
    LAMBDA_LS:     float = 0.3
    LAMBDA_FUSED:  float = 0.3
    LABEL_SMOOTH:  float = 0.1

    # Early stopping
    PATIENCE:      int   = 40       # epochs without val improvement

    # Checkpointing
    CHECKPOINT_DIR: str  = './checkpoints'
    LOG_DIR:        str  = './runs'

    # Misc
    SEED:          int   = 42
    CACHE:         bool  = True     # True if RAM > 16 GB (loads all spectrograms)
    AMP:           bool  = True     # mixed precision (disable if CPU-only)


TCFG = TrainConfig()


# ---------------------------------------------------------------------------
# 3.  WARMUP LR SCHEDULER
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """
    Linear warmup for the first warmup_epochs, then CosineAnnealingWarmRestarts.
    Wraps the PyTorch scheduler for clean integration.
    """
    def __init__(self, optimizer, warmup_epochs: int,
                 cosine_scheduler: optim.lr_scheduler._LRScheduler,
                 base_lr: float):
        self.optimizer        = optimizer
        self.warmup_epochs    = warmup_epochs
        self.cosine_scheduler = cosine_scheduler
        self.base_lr          = base_lr
        self._epoch           = 0

    def step(self):
        self._epoch += 1
        if self._epoch <= self.warmup_epochs:
            lr = self.base_lr * self._epoch / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
        else:
            self.cosine_scheduler.step()

    def get_last_lr(self) -> list:
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ---------------------------------------------------------------------------
# 4.  ONE EPOCH: TRAIN
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    loss_fn:   MRAFCNNLoss,
    scaler:    GradScaler,
    device:    torch.device,
    use_mixup: bool,
    num_classes: int,
    grad_clip: float,
) -> dict:
    """
    Run one training epoch.

    Returns dict with keys: loss, acc
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for batch in loader:
        specs, labels = batch
        specs = specs.to(device, non_blocking=True)

        if use_mixup:
            # labels are soft [B, C] from mixup_collate
            soft_labels = labels.to(device, non_blocking=True)
            hard_labels = soft_labels.argmax(dim=1)
        else:
            hard_labels = labels.to(device, non_blocking=True)
            soft_labels = None

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=scaler.is_enabled()):
            outputs = model(specs)                          # (fused, ce, proj_supcon)
            logits_fused, logits_ce, proj_supcon = outputs

            if use_mixup and soft_labels is not None:
                # Use soft CE for fused/ce heads, but standard SupCon for the projection head using hard majority labels
                loss = (
                    TCFG.LAMBDA_CE    * soft_cross_entropy(logits_ce,    soft_labels)
                  + TCFG.LAMBDA_FUSED * soft_cross_entropy(logits_fused, soft_labels)
                  + TCFG.LAMBDA_LS    * loss_fn.supcon_loss(proj_supcon, hard_labels)
                )
            else:
                loss = loss_fn(outputs, hard_labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * specs.size(0)
        preds       = logits_fused.argmax(dim=1)
        correct    += (preds == hard_labels).sum().item()
        total      += specs.size(0)

    return {
        'loss': total_loss / total,
        'acc':  correct / total * 100,
    }


# ---------------------------------------------------------------------------
# 5.  ONE EPOCH: VALIDATE
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model:  nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: MRAFCNNLoss,
    device: torch.device,
) -> dict:
    """
    Run validation epoch (no augmentation, no mixup).

    Returns dict with keys: loss, acc
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for specs, labels in loader:
        specs  = specs.to(device,  non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(specs)
        logits_fused, logits_ce, proj_supcon = outputs

        loss  = loss_fn(outputs, labels)

        total_loss += loss.item() * specs.size(0)
        preds       = logits_fused.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += specs.size(0)

    return {
        'loss': total_loss / total,
        'acc':  correct / total * 100,
    }


# ---------------------------------------------------------------------------
# 6.  EARLY STOPPING
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 30, min_delta: float = 0.01):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_acc   = 0.0
        self.counter    = 0
        self.should_stop = False

    def step(self, val_acc: float) -> bool:
        """Returns True if training should stop."""
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter  = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ---------------------------------------------------------------------------
# 7.  TRAIN ONE FOLD
# ---------------------------------------------------------------------------

def train_fold(
    fold:        int,
    train_ds,
    test_ds,
    num_classes: int,
    dataset_tag: str,
    args,
) -> dict:
    """
    Full training loop for a single fold.

    Returns: dict with fold result metrics.
    """
    set_seed(TCFG.SEED + fold)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(TCFG.CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(TCFG.CHECKPOINT_DIR,
                             f'{dataset_tag}_fold{fold}_best.pt')

    print(f"\n{'='*60}")
    print(f"  Fold {fold}/{args.total_folds} | {dataset_tag.upper()}")
    print(f"  Train: {len(train_ds)} samples | Test: {len(test_ds)} samples")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # ---- Data loaders ----
    train_dl = get_dataloader(
        train_ds, batch_size=TCFG.BATCH_SIZE,
        shuffle=True,  num_workers=args.num_workers,
        use_mixup=TCFG.USE_MIXUP, mixup_alpha=TCFG.MIXUP_ALPHA
    )
    test_dl = get_dataloader(
        test_ds, batch_size=TCFG.BATCH_SIZE,
        shuffle=False, num_workers=args.num_workers,
        use_mixup=False
    )

    # ---- Model ----
    if dataset_tag == 'esc50':
        model = build_esc50(
            n_mels=CFG.TARGET_MELS, n_frames=CFG.TARGET_FRAMES
        ).to(device)
    else:
        model = build_urbansound8k(
            n_mels=CFG.TARGET_MELS, n_frames=CFG.TARGET_FRAMES
        ).to(device)

    # ---- Loss ----
    loss_fn = MRAFCNNLoss(
        num_classes=num_classes,
        lambda_ce=TCFG.LAMBDA_CE,
        lambda_ls=TCFG.LAMBDA_LS,
        lambda_fused=TCFG.LAMBDA_FUSED,
        smoothing=TCFG.LABEL_SMOOTH,
    )

    # ---- Optimizer ----
    optimizer = optim.AdamW(
        model.parameters(),
        lr=TCFG.LR,
        weight_decay=TCFG.WEIGHT_DECAY,
        betas=TCFG.BETAS,
    )

    # ---- LR Scheduler ----
    cosine_sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=TCFG.T0, T_mult=TCFG.T_MULT, eta_min=TCFG.ETA_MIN
    )
    scheduler = WarmupCosineScheduler(
        optimizer, TCFG.WARMUP_EPOCHS, cosine_sched, TCFG.LR
    )

    # ---- Mixed precision scaler ----
    amp_enabled = TCFG.AMP and device.type == 'cuda'
    scaler = GradScaler(enabled=amp_enabled)

    # ---- TensorBoard ----
    log_dir = os.path.join(TCFG.LOG_DIR, f'{dataset_tag}_fold{fold}')
    writer  = SummaryWriter(log_dir=log_dir)

    # ---- Resume ----
    start_epoch  = 1
    best_val_acc = 0.0
    history      = {'train_loss': [], 'train_acc': [],
                    'val_loss': [],   'val_acc': []}

    if args.resume and os.path.exists(args.resume):
        print(f"  Resuming from {args.resume}")
        ckpt        = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt.get('epoch', 1) + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        history      = ckpt.get('history', history)

    # ---- Early stopping ----
    early_stop = EarlyStopping(patience=TCFG.PATIENCE)

    # ---- Stochastic Weight Averaging ----
    swa_start = int(TCFG.EPOCHS * 0.60)  # Start SWA at 60% of training
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-5, anneal_epochs=5)
    swa_active = False

    # ---- Training loop ----
    for epoch in range(start_epoch, TCFG.EPOCHS + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_dl, optimizer, loss_fn,
            scaler, device,
            use_mixup=TCFG.USE_MIXUP,
            num_classes=num_classes,
            grad_clip=TCFG.GRAD_CLIP,
        )
        val_metrics = validate(model, test_dl, loss_fn, device)
        
        # Switch to SWA after swa_start epochs
        if epoch >= swa_start:
            if not swa_active:
                print(f"\n  \U0001f504 SWA activated at epoch {epoch}")
                swa_active = True
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # Log history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['acc'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['acc'])

        # TensorBoard
        writer.add_scalars('Loss',     {'train': train_metrics['loss'],
                                        'val':   val_metrics['loss']},   epoch)
        writer.add_scalars('Accuracy', {'train': train_metrics['acc'],
                                        'val':   val_metrics['acc']},    epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        # Checkpoint
        is_best = val_metrics['acc'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['acc']
            torch.save({
                'epoch':           epoch,
                'model_state':     model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_val_acc':    best_val_acc,
                'history':         history,
                'fold':            fold,
                'dataset':         dataset_tag,
            }, ckpt_path)

        # Console log
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]
        marker  = " ★" if is_best else ""
        print(
            f"  Epoch {epoch:>3}/{TCFG.EPOCHS} | "
            f"Train: {train_metrics['acc']:5.2f}% (loss {train_metrics['loss']:.4f}) | "
            f"Val: {val_metrics['acc']:5.2f}% (loss {val_metrics['loss']:.4f}) | "
            f"LR: {lr_now:.2e} | {elapsed:.1f}s{marker}"
        )

        # Early stopping
        if early_stop.step(val_metrics['acc']):
            print(f"\n  ⚡ Early stopping at epoch {epoch} "
                  f"(no improvement for {TCFG.PATIENCE} epochs)")
            break

    writer.close()

    # ---- Finalize SWA model ----
    if swa_active:
        print("  \U0001f504 Updating SWA batch normalization statistics...")
        # Update BN stats on entire training set
        swa_bn_loader = get_dataloader(
            train_ds, batch_size=TCFG.BATCH_SIZE,
            shuffle=False, num_workers=args.num_workers,
            use_mixup=False
        )
        torch.optim.swa_utils.update_bn(swa_bn_loader, swa_model, device=device)
        
        # Evaluate SWA model
        swa_val = validate(swa_model, test_dl, loss_fn, device)
        print(f"  \U0001f504 SWA val acc: {swa_val['acc']:.2f}%")
        
        if swa_val['acc'] > best_val_acc:
            best_val_acc = swa_val['acc']
            # Save SWA model
            swa_ckpt_path = os.path.join(TCFG.CHECKPOINT_DIR,
                                         f'{dataset_tag}_fold{fold}_swa.pt')
            torch.save({
                'epoch':        TCFG.EPOCHS,
                'model_state':  swa_model.module.state_dict(),
                'best_val_acc': best_val_acc,
                'history':      history,
                'fold':         fold,
                'dataset':      dataset_tag,
                'swa':          True,
            }, swa_ckpt_path)
            # Also overwrite the best checkpoint
            torch.save({
                'epoch':        TCFG.EPOCHS,
                'model_state':  swa_model.module.state_dict(),
                'best_val_acc': best_val_acc,
                'history':      history,
                'fold':         fold,
                'dataset':      dataset_tag,
                'swa':          True,
            }, ckpt_path)
            print(f"  \u2705 SWA improved best acc to {best_val_acc:.2f}%!")

    print(f"\n  Fold {fold} best val acc: {best_val_acc:.2f}%")

    return {
        'fold':         fold,
        'best_val_acc': best_val_acc,
        'history':      history,
        'checkpoint':   ckpt_path,
    }


# ---------------------------------------------------------------------------
# 8.  FULL CROSS-VALIDATION RUNNER
# ---------------------------------------------------------------------------

def run_cross_validation(args) -> dict:
    """
    Run k-fold cross-validation for the chosen dataset.
    Prints per-fold results and final mean ± std accuracy.
    Saves all results to JSON.
    """
    set_seed(TCFG.SEED)
    results = []

    if args.dataset == 'esc50':
        total_folds   = 5
        dataset_tag   = 'esc50'
        num_classes   = 50
        fold_split_fn = esc50_fold_splits
    else:
        total_folds   = 10
        dataset_tag   = 'us8k'
        num_classes   = 10
        fold_split_fn = us8k_fold_splits

    # Allow training a subset of folds (e.g., --folds 1 2 3)
    folds_to_run = args.folds if args.folds else list(range(1, total_folds + 1))
    args.total_folds = total_folds

    print(f"\n{'#'*60}")
    print(f"  MR-AFCNN Cross-Validation")
    print(f"  Dataset:     {dataset_tag.upper()}")
    print(f"  Folds:       {folds_to_run}")
    print(f"  Epochs/fold: {TCFG.EPOCHS}")
    print(f"  Batch size:  {TCFG.BATCH_SIZE}")
    print(f"  Mixup:       {TCFG.USE_MIXUP} (α={TCFG.MIXUP_ALPHA})")
    print(f"{'#'*60}")

    for fold in folds_to_run:
        train_ds, test_ds = fold_split_fn(
            root=args.root, test_fold=fold, cache=TCFG.CACHE
        )
        fold_result = train_fold(
            fold=fold,
            train_ds=train_ds,
            test_ds=test_ds,
            num_classes=num_classes,
            dataset_tag=dataset_tag,
            args=args,
        )
        results.append(fold_result)

    # Summary
    accs = [r['best_val_acc'] for r in results]
    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)

    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION RESULTS — {dataset_tag.upper()}")
    print(f"{'='*60}")
    for r in results:
        print(f"  Fold {r['fold']}: {r['best_val_acc']:.2f}%")
    print(f"  {'─'*40}")
    print(f"  Mean:  {mean_acc:.2f}%")
    print(f"  Std:   {std_acc:.2f}%")
    print(f"  Best:  {max(accs):.2f}%")
    print(f"{'='*60}\n")

    # Save to JSON
    summary = {
        'dataset':    dataset_tag,
        'mean_acc':   mean_acc,
        'std_acc':    std_acc,
        'best_acc':   max(accs),
        'per_fold':   [{'fold': r['fold'], 'val_acc': r['best_val_acc']}
                       for r in results],
    }
    os.makedirs(TCFG.CHECKPOINT_DIR, exist_ok=True)
    json_path = os.path.join(TCFG.CHECKPOINT_DIR,
                             f'{dataset_tag}_cv_results.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Results saved to {json_path}")

    return summary


# ---------------------------------------------------------------------------
# 9.  CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Train MR-AFCNN')
    p.add_argument('--dataset', type=str, default='esc50',
                   choices=['esc50', 'us8k'],
                   help='Dataset to train on')
    p.add_argument('--root', type=str, required=True,
                   help='Path to dataset root directory')
    p.add_argument('--folds', type=int, nargs='+', default=None,
                   help='Specific folds to train (default: all)')
    p.add_argument('--epochs', type=int, default=TCFG.EPOCHS,
                   help=f'Max epochs per fold (default: {TCFG.EPOCHS})')
    p.add_argument('--batch_size', type=int, default=TCFG.BATCH_SIZE,
                   help=f'Batch size (default: {TCFG.BATCH_SIZE})')
    p.add_argument('--lr', type=float, default=TCFG.LR,
                   help=f'Learning rate (default: {TCFG.LR})')
    p.add_argument('--base_channels', type=int, default=64,
                   help='Base channel width (default: 64)')
    p.add_argument('--num_workers', type=int, default=TCFG.NUM_WORKERS,
                   help=f'Number of data loading workers (default: {TCFG.NUM_WORKERS})')
    p.add_argument('--no_mixup', action='store_true',
                   help='Disable Mixup augmentation')
    p.add_argument('--no_amp', action='store_true',
                   help='Disable mixed precision')
    p.add_argument('--cache', action='store_true',
                   help='Cache all spectrograms in RAM')
    p.add_argument('--resume', type=str, default=None,
                   help='Path to checkpoint to resume from')

    p.add_argument('--seed', type=int, default=TCFG.SEED,
                   help=f'Random seed (default: {TCFG.SEED})')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Override TrainConfig from CLI
    TCFG.EPOCHS     = args.epochs
    TCFG.BATCH_SIZE = args.batch_size
    TCFG.LR         = args.lr
    TCFG.USE_MIXUP  = not args.no_mixup
    TCFG.AMP        = not args.no_amp
    TCFG.CACHE      = args.cache
    TCFG.SEED       = args.seed
    TCFG.NUM_WORKERS = args.num_workers

    run_cross_validation(args)