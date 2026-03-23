"""
dataset.py — Feature Extraction & Data Pipeline for MR-AFCNN
=============================================================
Handles:
    - Log-mel spectrogram extraction (librosa)
    - Augmentation: SpecAugment (time/freq masking), Mixup, time-shift, pitch-shift
    - ESC-50 dataset loader  (5-fold cross-validation, fold-aware split)
    - UrbanSound8K loader    (10-fold cross-validation, fold-aware split)
    - Normalization: per-dataset global mean/std computed from training folds only

Directory structure expected:
    ESC-50/
        audio/          *.wav files  (2000 total)
        meta/
            esc50.csv   (filename, fold, target, category, ...)

    UrbanSound8K/
        audio/
            fold1/ ... fold10/  *.wav files
        metadata/
            UrbanSound8K.csv    (slice_file_name, fold, classID, class, ...)

Usage:
    from dataset import ESC50Dataset, UrbanSound8KDataset, get_dataloader

    # ESC-50: hold out fold 5 for test
    train_ds = ESC50Dataset(root='./ESC-50', folds=[1,2,3,4], split='train')
    test_ds  = ESC50Dataset(root='./ESC-50', folds=[5],       split='test')

    train_dl = get_dataloader(train_ds, batch_size=32, shuffle=True)
    test_dl  = get_dataloader(test_ds,  batch_size=32, shuffle=False)
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import functools

try:
    import librosa
except ImportError:
    raise ImportError("pip install librosa soundfile")


# ---------------------------------------------------------------------------
# 1.  FEATURE EXTRACTION CONFIG
# ---------------------------------------------------------------------------

class AudioConfig:
    """Global audio / feature hyper-parameters."""
    # Audio
    SAMPLE_RATE:   int   = 22050
    DURATION:      float = 5.0          # seconds (both datasets padded/trimmed to this)
    N_SAMPLES:     int   = int(SAMPLE_RATE * DURATION)   # 110 250

    # Mel spectrogram
    N_FFT:         int   = 1024
    HOP_LENGTH:    int   = 512
    N_MELS:        int   = 128
    F_MIN:         float = 20.0
    F_MAX:         float = 8000.0
    N_FRAMES:      int   = int(np.ceil(N_SAMPLES / HOP_LENGTH))  # ≈ 216

    # Model input size (fixed via resize after mel)
    TARGET_FRAMES: int   = 128          # time axis fed to model
    TARGET_MELS:   int   = 128          # freq axis fed to model (= N_MELS)


CFG = AudioConfig()


# ---------------------------------------------------------------------------
# 2.  AUDIO LOADING & MEL EXTRACTION
# ---------------------------------------------------------------------------

def load_audio(path: str, sr: int = CFG.SAMPLE_RATE,
               duration: float = CFG.DURATION) -> np.ndarray:
    """
    Load mono audio, resample if needed, pad/trim to fixed duration.
    Returns: float32 array of shape [N_SAMPLES].
    """
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration + 0.1)
    n     = int(sr * duration)

    if len(y) < n:
        # Repeat-pad (better than zero-pad for short clips)
        repeats = int(np.ceil(n / len(y)))
        y       = np.tile(y, repeats)
    y = y[:n]
    return y.astype(np.float32)


def extract_log_mel(y: np.ndarray) -> np.ndarray:
    """
    Compute log-mel spectrogram.
    Returns: float32 array of shape [N_MELS, N_FRAMES] (before resize).
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=CFG.SAMPLE_RATE,
        n_fft=CFG.N_FFT, hop_length=CFG.HOP_LENGTH,
        n_mels=CFG.N_MELS, fmin=CFG.F_MIN, fmax=CFG.F_MAX
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def resize_spectrogram(spec: np.ndarray,
                       target_mels:   int = CFG.TARGET_MELS,
                       target_frames: int = CFG.TARGET_FRAMES) -> np.ndarray:
    """
    Resize [F, T] spectrogram to fixed [target_mels, target_frames]
    using bilinear interpolation (via torch). Ensures uniform model input.
    """
    t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)   # [1,1,F,T]
    t = F.interpolate(t, size=(target_mels, target_frames),
                      mode='bilinear', align_corners=False)
    return t.squeeze().numpy()


def audio_to_tensor(path: str) -> torch.Tensor:
    """
    Full pipeline: audio file → [1, TARGET_MELS, TARGET_FRAMES] tensor.
    """
    y   = load_audio(path)
    mel = extract_log_mel(y)
    mel = resize_spectrogram(mel)
    return torch.from_numpy(mel).unsqueeze(0)   # [1, F, T]


# ---------------------------------------------------------------------------
# 3.  AUGMENTATION
# ---------------------------------------------------------------------------

class SpecAugment:
    """
    SpecAugment: random frequency masking + random time masking.
    Applied in the frequency-time domain on the [1, F, T] tensor.

    Paper: Park et al. (2019) — SpecAugment: A Simple Data Augmentation
    Method for Automatic Speech Recognition.

    Args:
        num_freq_masks: number of frequency masks to apply
        freq_mask_param: max width of each frequency mask (F_0)
        num_time_masks: number of time masks to apply
        time_mask_param: max width of each time mask (T_0)
    """
    def __init__(self, num_freq_masks: int = 2, freq_mask_param: int = 20,
                 num_time_masks: int = 2, time_mask_param: int = 30):
        self.num_freq_masks  = num_freq_masks
        self.freq_mask_param = freq_mask_param
        self.num_time_masks  = num_time_masks
        self.time_mask_param = time_mask_param

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:  spec: [1, F, T]
        Returns: augmented [1, F, T]
        """
        spec = spec.clone()
        _, F, T = spec.shape
        mean_val = spec.mean()

        # Frequency masking
        for _ in range(self.num_freq_masks):
            f  = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(0, F - f))
            spec[:, f0:f0+f, :] = mean_val

        # Time masking
        for _ in range(self.num_time_masks):
            t  = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, max(0, T - t))
            spec[:, :, t0:t0+t] = mean_val

        return spec


class TimeShift:
    """Random circular time shift of the spectrogram."""
    def __init__(self, max_shift: float = 0.2):
        self.max_shift = max_shift   # fraction of total time frames

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        _, F, T = spec.shape
        shift = int(random.uniform(-self.max_shift, self.max_shift) * T)
        return torch.roll(spec, shift, dims=2)


class FreqShift:
    """Random circular frequency shift (simulates pitch shift in mel domain)."""
    def __init__(self, max_shift: int = 4):
        self.max_shift = max_shift   # bins

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        shift = random.randint(-self.max_shift, self.max_shift)
        return torch.roll(spec, shift, dims=1)


class AddGaussianNoise:
    """Add Gaussian noise (SNR-controlled)."""
    def __init__(self, snr_db_range: tuple = (20, 40)):
        self.snr_min, self.snr_max = snr_db_range

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        snr_db  = random.uniform(self.snr_min, self.snr_max)
        signal_power = spec.pow(2).mean()
        noise_power  = signal_power / (10 ** (snr_db / 10))
        noise        = torch.randn_like(spec) * noise_power.sqrt()
        return spec + noise


class Normalize:
    """Normalize spectrogram to zero mean, unit std using dataset statistics."""
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std  = max(std, 1e-6)

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        return (spec - self.mean) / self.std


class ComposeTransforms:
    """Compose a list of callable transforms."""
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def build_train_transforms(mean: float, std: float) -> ComposeTransforms:
    return ComposeTransforms([
        TimeShift(max_shift=0.15),
        FreqShift(max_shift=4),
        SpecAugment(num_freq_masks=2, freq_mask_param=18,
                    num_time_masks=2, time_mask_param=25),
        AddGaussianNoise(snr_db_range=(25, 45)),
        Normalize(mean, std),
    ])


def build_test_transforms(mean: float, std: float) -> ComposeTransforms:
    return ComposeTransforms([
        Normalize(mean, std),
    ])


# ---------------------------------------------------------------------------
# 4.  DATASET STATISTICS (computed from train folds)
# ---------------------------------------------------------------------------

def compute_dataset_stats(file_paths: list) -> tuple:
    """
    Compute global mean and std of log-mel spectrograms over a list of files.
    Used to normalize train and test sets consistently.

    Returns: (mean: float, std: float)
    """
    all_vals = []
    for path in file_paths:
        try:
            y   = load_audio(path)
            mel = extract_log_mel(y)
            mel = resize_spectrogram(mel)
            all_vals.append(mel.ravel())
        except Exception:
            continue
    all_vals = np.concatenate(all_vals)
    return float(all_vals.mean()), float(all_vals.std())


# ---------------------------------------------------------------------------
# 5.  ESC-50 DATASET
# ---------------------------------------------------------------------------

class ESC50Dataset(Dataset):
    """
    ESC-50: Environmental Sound Classification dataset.
    2000 audio clips, 50 classes, 5 folds, 40 clips per class.

    Args:
        root:       Path to ESC-50 root directory.
        folds:      List of fold numbers to include (1–5).
        split:      'train' (with augmentation) or 'test' (no augmentation).
        cache:      If True, pre-loads all spectrograms into RAM.
        norm_stats: Optional (mean, std) tuple; computed from data if None.
    """

    def __init__(self, root: str, folds: list,
                 split: str = 'train',
                 cache: bool = False,
                 norm_stats: tuple = None):
        super().__init__()
        self.root  = root
        self.folds = folds
        self.split = split
        self.cache = cache

        # Load metadata
        meta_path = os.path.join(root, 'meta', 'esc50.csv')
        df        = pd.read_csv(meta_path)
        df        = df[df['fold'].isin(folds)].reset_index(drop=True)

        self.file_paths = [
            os.path.join(root, 'audio', row['filename'])
            for _, row in df.iterrows()
        ]
        self.labels = df['target'].tolist()
        self.classes = sorted(df['category'].unique().tolist())

        # Normalization stats
        if norm_stats is not None:
            mean, std = norm_stats
        else:
            print(f"  [ESC-50] Computing stats for folds {folds} ...")
            mean, std = compute_dataset_stats(self.file_paths)
        self.mean, self.std = mean, std

        # Transforms
        if split == 'train':
            self.transform = build_train_transforms(mean, std)
        else:
            self.transform = build_test_transforms(mean, std)

        # Optional cache
        self._cache = {}
        if cache:
            print(f"  [ESC-50] Caching {len(self.file_paths)} spectrograms ...")
            for i, p in enumerate(self.file_paths):
                self._cache[i] = audio_to_tensor(p)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        if idx in self._cache:
            spec = self._cache[idx].clone()
        else:
            spec = audio_to_tensor(self.file_paths[idx])

        spec  = self.transform(spec)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return spec, label

    @property
    def num_classes(self) -> int:
        return 50

    def get_norm_stats(self) -> tuple:
        return (self.mean, self.std)


# ---------------------------------------------------------------------------
# 6.  URBANSOUND8K DATASET
# ---------------------------------------------------------------------------

class UrbanSound8KDataset(Dataset):
    """
    UrbanSound8K: 10 urban sound classes, ~8732 clips, 10 folds.

    Args:
        root:       Path to UrbanSound8K root directory.
        folds:      List of fold numbers to include (1–10).
        split:      'train' or 'test'.
        cache:      Pre-load spectrograms into RAM.
        norm_stats: Optional (mean, std); computed if None.
    """

    def __init__(self, root: str, folds: list,
                 split: str = 'train',
                 cache: bool = False,
                 norm_stats: tuple = None):
        super().__init__()
        self.root  = root
        self.folds = folds
        self.split = split
        self.cache = cache

        meta_path = os.path.join(root, 'metadata', 'UrbanSound8K.csv')
        df        = pd.read_csv(meta_path)
        df        = df[df['fold'].isin(folds)].reset_index(drop=True)

        self.file_paths = [
            os.path.join(root, 'audio', f'fold{row["fold"]}',
                         row['slice_file_name'])
            for _, row in df.iterrows()
        ]
        self.labels = df['classID'].tolist()
        self.classes = sorted(df['class'].unique().tolist())

        # Normalization stats
        if norm_stats is not None:
            mean, std = norm_stats
        else:
            print(f"  [US8K] Computing stats for folds {folds} ...")
            mean, std = compute_dataset_stats(self.file_paths)
        self.mean, self.std = mean, std

        if split == 'train':
            self.transform = build_train_transforms(mean, std)
        else:
            self.transform = build_test_transforms(mean, std)

        self._cache = {}
        if cache:
            print(f"  [US8K] Caching {len(self.file_paths)} spectrograms ...")
            for i, p in enumerate(self.file_paths):
                self._cache[i] = audio_to_tensor(p)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        if idx in self._cache:
            spec = self._cache[idx].clone()
        else:
            spec = audio_to_tensor(self.file_paths[idx])

        spec  = self.transform(spec)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return spec, label

    @property
    def num_classes(self) -> int:
        return 10

    def get_norm_stats(self) -> tuple:
        return (self.mean, self.std)


# ---------------------------------------------------------------------------
# 7.  MIXUP COLLATE FUNCTION
# ---------------------------------------------------------------------------

def mixup_collate(batch, alpha: float = 0.4, num_classes: int = 50):
    """
    Mixup data augmentation applied at the batch level.

    For each sample pair (i, j) with mixing coefficient λ ~ Beta(α, α):
        x_mix   = λ * x_i   + (1-λ) * x_j
        y_mix   = λ * onehot(y_i) + (1-λ) * onehot(y_j)   [soft labels]

    Returns: (specs, soft_labels) where soft_labels are float tensors.
    The training loop must use KL-divergence or soft cross-entropy instead
    of standard CrossEntropyLoss when mixup is active.
    """
    specs, labels = zip(*batch)
    specs  = torch.stack(specs)    # [B, 1, F, T]
    labels = torch.tensor(labels)  # [B]

    B = len(specs)
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0

    # Random permutation for mixing partners
    idx = torch.randperm(B)

    mixed_specs = lam * specs + (1 - lam) * specs[idx]

    # Soft labels
    y_one = F.one_hot(labels,      num_classes=num_classes).float()
    y_mix = F.one_hot(labels[idx], num_classes=num_classes).float()
    mixed_labels = lam * y_one + (1 - lam) * y_mix   # [B, C]

    return mixed_specs, mixed_labels


def soft_cross_entropy(logits: torch.Tensor,
                       soft_labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss compatible with soft (mixup) labels."""
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_labels * log_probs).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# 8.  DATALOADER FACTORY
# ---------------------------------------------------------------------------

def get_dataloader(dataset: Dataset,
                   batch_size: int = 32,
                   shuffle: bool = True,
                   num_workers: int = 4,
                   use_mixup: bool = False,
                   mixup_alpha: float = 0.4) -> DataLoader:
    """
    Build a DataLoader with optional Mixup collate function.

    Args:
        dataset:     ESC50Dataset or UrbanSound8KDataset
        batch_size:  mini-batch size
        shuffle:     shuffle samples (True for train)
        num_workers: parallel data loading workers
        use_mixup:   apply Mixup at batch level
        mixup_alpha: Beta distribution alpha for Mixup (0 = disabled)
    """
    collate_fn = None
    if use_mixup and shuffle:   # only mix during training
        nc = dataset.num_classes
        collate_fn = functools.partial(mixup_collate, alpha=mixup_alpha, num_classes=nc)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        drop_last=shuffle,      # drop last incomplete batch during training
    )


# ---------------------------------------------------------------------------
# 9.  FOLD ITERATOR HELPERS
# ---------------------------------------------------------------------------

def esc50_fold_splits(root: str, test_fold: int,
                      cache: bool = False) -> tuple:
    """
    Returns (train_dataset, test_dataset) for ESC-50 with the given test fold.
    Stats are computed from training folds only (no data leakage).

    Args:
        root:      ESC-50 root directory
        test_fold: fold number held out for testing (1–5)
        cache:     cache spectrograms in RAM
    """
    train_folds = [f for f in range(1, 6) if f != test_fold]

    # Compute stats from train folds only
    train_ds_temp = ESC50Dataset(root, train_folds, split='train', cache=False)
    mean, std     = train_ds_temp.mean, train_ds_temp.std

    train_ds = ESC50Dataset(root, train_folds, split='train',
                            cache=cache, norm_stats=(mean, std))
    test_ds  = ESC50Dataset(root, [test_fold], split='test',
                            cache=cache, norm_stats=(mean, std))
    return train_ds, test_ds


def us8k_fold_splits(root: str, test_fold: int,
                     cache: bool = False) -> tuple:
    """
    Returns (train_dataset, test_dataset) for UrbanSound8K.

    Args:
        root:      UrbanSound8K root directory
        test_fold: fold held out for testing (1–10)
        cache:     cache spectrograms in RAM
    """
    train_folds = [f for f in range(1, 11) if f != test_fold]

    train_ds_temp = UrbanSound8KDataset(root, train_folds, split='train', cache=False)
    mean, std     = train_ds_temp.mean, train_ds_temp.std

    train_ds = UrbanSound8KDataset(root, train_folds, split='train',
                                   cache=cache, norm_stats=(mean, std))
    test_ds  = UrbanSound8KDataset(root, [test_fold], split='test',
                                   cache=cache, norm_stats=(mean, std))
    return train_ds, test_ds


# ---------------------------------------------------------------------------
# 10. QUICK VALIDATION (no audio files needed)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("  dataset.py — Structure Validation")
    print("=" * 60)

    # Validate transforms with synthetic data
    spec = torch.randn(1, 128, 128)

    sa     = SpecAugment()
    ts     = TimeShift()
    fs     = FreqShift()
    noise  = AddGaussianNoise()
    norm   = Normalize(mean=-30.0, std=20.0)
    train_t = build_train_transforms(-30.0, 20.0)
    test_t  = build_test_transforms(-30.0, 20.0)

    print(f"  Input shape:              {list(spec.shape)}")
    print(f"  After SpecAugment:        {list(sa(spec).shape)}")
    print(f"  After TimeShift:          {list(ts(spec).shape)}")
    print(f"  After FreqShift:          {list(fs(spec).shape)}")
    print(f"  After GaussianNoise:      {list(noise(spec).shape)}")
    print(f"  After Normalize:          {list(norm(spec).shape)}")
    print(f"  After train transforms:   {list(train_t(spec).shape)}")
    print(f"  After test transforms:    {list(test_t(spec).shape)}")

    # Validate Mixup collate with dummy batch
    dummy_batch = [(torch.randn(1, 128, 128), i % 50) for i in range(8)]
    mixed_specs, mixed_labels = mixup_collate(dummy_batch, alpha=0.4, num_classes=50)
    print(f"\n  Mixup specs shape:        {list(mixed_specs.shape)}")
    print(f"  Mixup labels shape:       {list(mixed_labels.shape)}")
    print(f"  Mixup labels sum (≈1.0):  {mixed_labels[0].sum().item():.4f}")

    # Validate soft CE
    logits = torch.randn(8, 50)
    loss   = soft_cross_entropy(logits, mixed_labels)
    print(f"  Soft CE loss (random):    {loss.item():.4f}")

    print()
    print("  ✓ All transform / collate checks passed.")
    print("=" * 60)