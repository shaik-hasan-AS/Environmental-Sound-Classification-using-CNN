"""
evaluate.py — Evaluation & Analysis for MR-AFCNN
==================================================
Features:
    - Load best checkpoint and run full test-set evaluation
    - Per-class accuracy breakdown
    - Confusion matrix (normalized + raw) with matplotlib
    - SOTA comparison table (ESC-50 + UrbanSound8K)
    - Cross-fold summary statistics
    - Grad-CAM visualization on mel spectrograms
    - Misclassification analysis (top-K wrong predictions)
    - Export results to CSV

Usage:
    # Evaluate all ESC-50 folds from checkpoints:
    python evaluate.py --dataset esc50 --root ./ESC-50 --checkpoint_dir ./checkpoints

    # Evaluate single checkpoint:
    python evaluate.py --dataset esc50 --root ./ESC-50 --ckpt ./checkpoints/esc50_fold1_best.pt --fold 1

    # Generate confusion matrix only:
    python evaluate.py --dataset esc50 --root ./ESC-50 --ckpt ./checkpoints/esc50_fold1_best.pt --fold 1 --plot_cm

    # Full analysis with Grad-CAM:
    python evaluate.py --dataset esc50 --root ./ESC-50 --ckpt ./checkpoints/esc50_fold1_best.pt --fold 1 --gradcam
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (works headless)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from   matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from model   import MRAFCNN, build_esc50, build_urbansound8k
from dataset import (
    ESC50Dataset, UrbanSound8KDataset,
    get_dataloader, esc50_fold_splits, us8k_fold_splits, CFG
)

# Output directory for all plots and CSVs
OUTPUT_DIR = './eval_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1.  SOTA COMPARISON TABLE
# ---------------------------------------------------------------------------

ESC50_SOTA = [
    # (Model,                      Accuracy%,  Type,                    Year)
    ("MR-AFCNN (Ours)",            None,       "CNN (novel)",           2025),
    ("TSCNN-DS",                   97.20,      "CNN (two-stream)",      2022),
    ("RACNN",                      91.00,      "CNN (resource-adaptive)", 2022),
    ("TF-Attention CNN",           84.40,      "CNN + attention",       2021),
    ("Piczak CNN (baseline)",      64.50,      "CNN",                   2015),
    ("AST",                        95.60,      "Transformer",           2021),
    ("HTSAT-22",                   98.25,      "Hierarchical Transformer", 2022),
    ("Human baseline",             81.30,      "—",                     "—"),
]

US8K_SOTA = [
    ("MR-AFCNN (Ours)",            None,       "CNN (novel)",           2025),
    ("TSCNN-DS",                   96.10,      "CNN (two-stream)",      2022),
    ("RACNN",                      95.30,      "CNN (resource-adaptive)", 2022),
    ("ITFA-DNN",                   95.30,      "CNN + attention",       2025),
    ("Piczak CNN (baseline)",      73.70,      "CNN",                   2015),
    ("AST",                        97.90,      "Transformer",           2021),
]


def print_sota_table(dataset: str, our_acc: float = None):
    """Print a formatted SOTA comparison table."""
    rows  = ESC50_SOTA if dataset == 'esc50' else US8K_SOTA
    title = "ESC-50" if dataset == 'esc50' else "UrbanSound8K"

    data = []
    for name, acc, mtype, year in rows:
        if name.startswith("MR-AFCNN") and our_acc is not None:
            acc = round(our_acc, 2)
        data.append({'Model': name, 'Accuracy (%)': acc, 'Type': mtype, 'Year': year})

    df = pd.DataFrame(data)
    df['Accuracy (%)'] = df['Accuracy (%)'].apply(
        lambda x: f"{x:.2f}" if isinstance(x, float) else str(x)
    )
    df = df.sort_values('Accuracy (%)', ascending=False).reset_index(drop=True)

    print(f"\n{'='*70}")
    print(f"  SOTA Comparison — {title}")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    print(f"{'='*70}")

    csv_path = os.path.join(OUTPUT_DIR, f'sota_comparison_{dataset}.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    return df


def plot_sota_bar(dataset: str, our_acc: float, save: bool = True):
    """Bar chart comparing MR-AFCNN against SOTA."""
    rows  = ESC50_SOTA if dataset == 'esc50' else US8K_SOTA
    title = "ESC-50" if dataset == 'esc50' else "UrbanSound8K"

    models = []
    accs   = []
    colors = []

    for name, acc, mtype, year in rows:
        if name.startswith("MR-AFCNN"):
            acc = our_acc
        if acc is None or not isinstance(acc, float):
            continue
        models.append(name)
        accs.append(acc)
        colors.append('#E74C3C' if 'Ours' in name else
                      '#3498DB' if 'CNN' in mtype else '#95A5A6')

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(models, accs, color=colors, edgecolor='white', height=0.6)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'MR-AFCNN vs SOTA — {title}', fontsize=14, fontweight='bold')
    ax.set_xlim([min(accs) - 5, 100])
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
    ax.axvline(our_acc, color='#E74C3C', linestyle='--', alpha=0.5, linewidth=1.5)

    # Value labels
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f'{acc:.2f}%', va='center', fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend = [Patch(color='#E74C3C', label='MR-AFCNN (Ours)'),
              Patch(color='#3498DB', label='CNN-based'),
              Patch(color='#95A5A6', label='Transformer-based / Human')]
    ax.legend(handles=legend, loc='lower right', fontsize=9)

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, f'sota_bar_{dataset}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
    return fig


# ---------------------------------------------------------------------------
# 2.  FULL EVALUATION PASS
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model:   nn.Module,
                   loader:  torch.utils.data.DataLoader,
                   device:  torch.device,
                   num_classes: int) -> dict:
    """
    Full evaluation pass. Returns:
        all_preds, all_labels, all_probs, accuracy
    """
    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []

    for specs, labels in loader:
        specs  = specs.to(device,  non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(specs)
        logits_fused = outputs[0]

        probs  = F.softmax(logits_fused, dim=1)
        preds  = logits_fused.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs)
    accuracy   = (all_preds == all_labels).mean() * 100

    return {
        'preds':    all_preds,
        'labels':   all_labels,
        'probs':    all_probs,
        'accuracy': accuracy,
    }


# ---------------------------------------------------------------------------
# 3.  PER-CLASS ACCURACY
# ---------------------------------------------------------------------------

def per_class_accuracy(preds: np.ndarray, labels: np.ndarray,
                       class_names: list) -> pd.DataFrame:
    """Returns a DataFrame of per-class accuracy sorted ascending."""
    rows = []
    for c, name in enumerate(class_names):
        mask     = labels == c
        n_total  = mask.sum()
        n_correct = (preds[mask] == c).sum() if n_total > 0 else 0
        acc      = n_correct / n_total * 100 if n_total > 0 else 0.0
        rows.append({'Class': name, 'N': int(n_total),
                     'Correct': int(n_correct), 'Accuracy (%)': round(acc, 2)})
    df = pd.DataFrame(rows).sort_values('Accuracy (%)')
    return df


# ---------------------------------------------------------------------------
# 4.  CONFUSION MATRIX
# ---------------------------------------------------------------------------

def plot_confusion_matrix(preds:       np.ndarray,
                          labels:      np.ndarray,
                          class_names: list,
                          dataset_tag: str,
                          fold:        int,
                          normalize:   bool = True):
    """
    Plot and save a heatmap confusion matrix.
    For datasets with many classes (ESC-50), shows top-20 most confused.
    """
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1

    if normalize:
        cm_plot = cm.astype(float)
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        cm_plot  = np.divide(cm_plot, row_sums,
                             out=np.zeros_like(cm_plot), where=row_sums != 0)
        fmt      = '.2f'
        vmax     = 1.0
    else:
        cm_plot = cm
        fmt     = 'd'
        vmax    = cm.max()

    # For ESC-50 (50 classes), limit figure size; still show all
    figsize = (20, 18) if num_classes > 20 else (12, 10)
    fig, ax = plt.subplots(figsize=figsize)

    cmap = LinearSegmentedColormap.from_list(
        'mrafcnn', ['#FFFFFF', '#2980B9', '#1A252F']
    )

    sns.heatmap(cm_plot, ax=ax, cmap=cmap, vmin=0, vmax=vmax,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.3 if num_classes <= 20 else 0,
                annot=num_classes <= 20, fmt=fmt if num_classes <= 20 else '',
                cbar_kws={'label': 'Proportion' if normalize else 'Count'})

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True',      fontsize=12)
    acc = (preds == labels).mean() * 100
    ax.set_title(
        f'Confusion Matrix — {dataset_tag.upper()} Fold {fold} '
        f'(Acc: {acc:.2f}%)', fontsize=13, fontweight='bold'
    )
    plt.xticks(rotation=45, ha='right', fontsize=7 if num_classes > 20 else 10)
    plt.yticks(rotation=0,              fontsize=7 if num_classes > 20 else 10)
    plt.tight_layout()

    suffix = 'norm' if normalize else 'raw'
    path   = os.path.join(OUTPUT_DIR,
                          f'confusion_matrix_{dataset_tag}_fold{fold}_{suffix}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 5.  TRAINING HISTORY PLOT
# ---------------------------------------------------------------------------

def plot_training_history(history: dict, dataset_tag: str, fold: int):
    """Plot train/val loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0].plot(epochs, history['train_loss'], label='Train', color='#3498DB')
    axes[0].plot(epochs, history['val_loss'],   label='Val',   color='#E74C3C')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Loss — {dataset_tag.upper()} Fold {fold}')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], label='Train', color='#3498DB')
    axes[1].plot(epochs, history['val_acc'],   label='Val',   color='#E74C3C')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'Accuracy — {dataset_tag.upper()} Fold {fold}')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Mark best val epoch
    best_epoch = int(np.argmax(history['val_acc'])) + 1
    best_acc   = max(history['val_acc'])
    axes[1].axvline(best_epoch, color='#2ECC71', linestyle='--',
                    alpha=0.7, label=f'Best: {best_acc:.2f}% @ ep{best_epoch}')
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR,
                        f'training_history_{dataset_tag}_fold{fold}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 6.  GRAD-CAM ON MEL SPECTROGRAM
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for 2D CNN.
    Hooks onto the last convolutional layer of the network.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, spec: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Args:
            spec:      [1, 1, F, T] input spectrogram
            class_idx: target class (uses argmax if None)
        Returns:
            cam: [F, T] numpy array in [0, 1]
        """
        self.model.eval()
        spec.requires_grad_(True)

        output = self.model(spec)
        logits = output[0]   # fused logits

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global average pool gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam     = F.relu(cam)
        cam     = F.interpolate(cam, size=(spec.shape[2], spec.shape[3]),
                                mode='bilinear', align_corners=False)
        cam     = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


def visualize_gradcam(model:       nn.Module,
                      dataset,
                      device:      torch.device,
                      class_names: list,
                      dataset_tag: str,
                      fold:        int,
                      n_samples:   int = 6):
    """
    Generate and save Grad-CAM overlays for n_samples test examples.
    Shows what frequency-time regions the model focuses on per class.
    """
    # Hook into last conv block (stage3[-1].conv2.pointwise)
    target_layer = model.stage3[-1].conv2.pointwise
    gcam         = GradCAM(model, target_layer)

    fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 8))
    indices   = np.random.choice(len(dataset), n_samples, replace=False)

    for i, idx in enumerate(indices):
        spec, label = dataset[idx]
        spec_in = spec.unsqueeze(0).to(device)

        cam       = gcam.generate(spec_in)
        true_cls  = class_names[label.item()]
        pred_cls  = class_names[
            model(spec_in)[0].argmax(dim=1).item()
        ]

        # Raw spectrogram
        spec_np = spec.squeeze().cpu().numpy()
        axes[0, i].imshow(spec_np, aspect='auto', origin='lower',
                          cmap='magma', interpolation='nearest')
        axes[0, i].set_title(f'True: {true_cls}', fontsize=8)
        axes[0, i].axis('off')

        # Grad-CAM overlay
        axes[1, i].imshow(spec_np, aspect='auto', origin='lower',
                          cmap='magma', interpolation='nearest')
        axes[1, i].imshow(cam,     aspect='auto', origin='lower',
                          cmap='jet', alpha=0.45, interpolation='nearest')
        correct = '✓' if pred_cls == true_cls else '✗'
        axes[1, i].set_title(f'Pred: {pred_cls} {correct}', fontsize=8)
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Mel Spectrogram',  fontsize=9)
    axes[1, 0].set_ylabel('Grad-CAM Overlay', fontsize=9)

    plt.suptitle(f'Grad-CAM — {dataset_tag.upper()} Fold {fold}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR,
                        f'gradcam_{dataset_tag}_fold{fold}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 7.  CROSS-FOLD SUMMARY PLOT
# ---------------------------------------------------------------------------

def plot_fold_summary(fold_accs: list, dataset_tag: str):
    """Bar plot of per-fold accuracies with mean ± std."""
    fig, ax = plt.subplots(figsize=(8, 4))

    folds  = [f'Fold {i+1}' for i in range(len(fold_accs))]
    colors = ['#E74C3C' if a == max(fold_accs) else '#3498DB' for a in fold_accs]
    bars   = ax.bar(folds, fold_accs, color=colors, edgecolor='white', width=0.5)

    mean = np.mean(fold_accs)
    std  = np.std(fold_accs)
    ax.axhline(mean, color='#2ECC71', linestyle='--', linewidth=1.5,
               label=f'Mean: {mean:.2f}% ± {std:.2f}%')

    for bar, acc in zip(bars, fold_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=10)

    ax.set_ylim([min(fold_accs) - 3, 100])
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.set_title(f'MR-AFCNN Cross-Validation — {dataset_tag.upper()}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f'fold_summary_{dataset_tag}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 8.  MAIN EVALUATION RUNNER
# ---------------------------------------------------------------------------

def evaluate_fold(ckpt_path: str, fold: int, args) -> dict:
    """Load checkpoint, run evaluation, generate all reports."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  Evaluating {args.dataset.upper()} Fold {fold}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"  Best val acc in checkpoint: {ckpt.get('best_val_acc', 'N/A'):.2f}%")

    # Build dataset
    if args.dataset == 'esc50':
        _, test_ds = esc50_fold_splits(args.root, fold, cache=False)
        model      = build_esc50(CFG.TARGET_MELS, CFG.TARGET_FRAMES).to(device)
        num_classes = 50
        class_names = test_ds.classes
    else:
        _, test_ds = us8k_fold_splits(args.root, fold, cache=False)
        model      = build_urbansound8k(CFG.TARGET_MELS, CFG.TARGET_FRAMES).to(device)
        num_classes = 10
        class_names = test_ds.classes

    model.load_state_dict(ckpt['model_state'])

    test_dl = get_dataloader(test_ds, batch_size=32, shuffle=False,
                             num_workers=4, use_mixup=False)

    # Evaluate
    results = evaluate_model(model, test_dl, device, num_classes)
    print(f"  Test accuracy: {results['accuracy']:.2f}%")

    # Per-class accuracy
    pc_df = per_class_accuracy(results['preds'], results['labels'], class_names)
    csv_path = os.path.join(OUTPUT_DIR,
                            f'per_class_acc_{args.dataset}_fold{fold}.csv')
    pc_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    print(f"\n  Top-5 worst classes:")
    print(pc_df.head(5).to_string(index=False))
    print(f"\n  Top-5 best classes:")
    print(pc_df.tail(5).to_string(index=False))

    # Confusion matrices
    if args.plot_cm or args.full:
        plot_confusion_matrix(results['preds'], results['labels'],
                              class_names, args.dataset, fold, normalize=True)
        plot_confusion_matrix(results['preds'], results['labels'],
                              class_names, args.dataset, fold, normalize=False)

    # Training history
    if 'history' in ckpt and (args.plot_history or args.full):
        plot_training_history(ckpt['history'], args.dataset, fold)

    # Grad-CAM
    if args.gradcam or args.full:
        print("  Generating Grad-CAM visualizations ...")
        visualize_gradcam(model, test_ds, device, class_names,
                          args.dataset, fold, n_samples=6)

    return {
        'fold':     fold,
        'accuracy': results['accuracy'],
        'preds':    results['preds'],
        'labels':   results['labels'],
    }


def run_full_evaluation(args):
    """Evaluate all folds and print summary."""
    total_folds   = 5 if args.dataset == 'esc50' else 10
    folds_to_eval = args.folds if args.folds else list(range(1, total_folds + 1))

    if args.ckpt:
        # Single checkpoint mode
        fold         = args.fold if args.fold else 1
        fold_results = [evaluate_fold(args.ckpt, fold, args)]
    else:
        # All folds from checkpoint_dir
        fold_results = []
        for fold in folds_to_eval:
            ckpt_path = os.path.join(args.checkpoint_dir,
                                     f'{args.dataset}_fold{fold}_best.pt')
            if not os.path.exists(ckpt_path):
                print(f"  [WARN] Checkpoint not found: {ckpt_path}, skipping.")
                continue
            fold_results.append(evaluate_fold(ckpt_path, fold, args))

    if not fold_results:
        print("  No checkpoints found. Run train.py first.")
        return

    accs  = [r['accuracy'] for r in fold_results]
    mean  = np.mean(accs)
    std   = np.std(accs)

    # Summary
    print(f"\n{'='*60}")
    print(f"  EVALUATION SUMMARY — {args.dataset.upper()}")
    print(f"{'='*60}")
    for r in fold_results:
        print(f"  Fold {r['fold']}: {r['accuracy']:.2f}%")
    print(f"  {'─'*40}")
    print(f"  Mean ± Std:  {mean:.2f}% ± {std:.2f}%")
    print(f"  Best fold:   {max(accs):.2f}%")

    # SOTA table
    print_sota_table(args.dataset, our_acc=mean)
    plot_sota_bar(args.dataset, our_acc=mean)

    # Cross-fold summary plot
    if len(fold_results) > 1:
        plot_fold_summary(accs, args.dataset)

    # Save summary JSON
    summary = {
        'dataset':  args.dataset,
        'mean_acc': round(mean, 4),
        'std_acc':  round(std, 4),
        'best_acc': round(max(accs), 4),
        'per_fold': [{'fold': r['fold'], 'acc': round(r['accuracy'], 4)}
                     for r in fold_results],
    }
    json_path = os.path.join(OUTPUT_DIR, f'eval_summary_{args.dataset}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Full results saved to {json_path}")
    print(f"  All plots saved to   {OUTPUT_DIR}/")


# ---------------------------------------------------------------------------
# 9.  CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate MR-AFCNN')
    p.add_argument('--dataset', type=str, required=True,
                   choices=['esc50', 'us8k'])
    p.add_argument('--root', type=str, required=True,
                   help='Dataset root directory')

    # Checkpoint options (mutually exclusive)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--ckpt', type=str, default=None,
                       help='Path to single checkpoint file')
    group.add_argument('--checkpoint_dir', type=str,
                       default='./checkpoints',
                       help='Directory with all fold checkpoints')

    p.add_argument('--fold', type=int, default=None,
                   help='Fold number (for single --ckpt mode)')
    p.add_argument('--folds', type=int, nargs='+', default=None,
                   help='Specific folds to evaluate')

    # Output options
    p.add_argument('--plot_cm', action='store_true',
                   help='Generate confusion matrices')
    p.add_argument('--plot_history', action='store_true',
                   help='Plot training history curves')
    p.add_argument('--gradcam', action='store_true',
                   help='Generate Grad-CAM visualizations')
    p.add_argument('--full', action='store_true',
                   help='Generate all outputs (cm + history + gradcam)')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_full_evaluation(args)