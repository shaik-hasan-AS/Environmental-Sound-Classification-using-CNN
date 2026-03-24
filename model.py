import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDWConv(nn.Module):
    """
    Novelty: Captures audio events at varying time and frequency scales.
    Uses parallel depthwise convolutions of different kernel shapes.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw_3x3 = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, stride=stride, groups=in_ch, bias=False)
        self.dw_1x5 = nn.Conv2d(in_ch, in_ch, kernel_size=(1, 5), padding=(0, 2), stride=stride, groups=in_ch, bias=False)
        self.dw_5x1 = nn.Conv2d(in_ch, in_ch, kernel_size=(5, 1), padding=(2, 0), stride=stride, groups=in_ch, bias=False)
        
        self.pointwise = nn.Conv2d(in_ch * 3, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        x1 = self.dw_3x3(x)
        x2 = self.dw_1x5(x)
        x3 = self.dw_5x1(x)
        merged = torch.cat([x1, x2, x3], dim=1)
        return self.silu(self.bn(self.pointwise(merged)))


class FreqSpatialAttention(nn.Module):
    """
    Novelty: Pools over Time axis only, retaining per-frequency-bin attention.
    Specifically designed for mel-spectrogram inputs.
    """
    def __init__(self, in_ch, reduction=4):
        super().__init__()
        mid_ch = max(1, in_ch // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_ch, in_ch, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        time_pooled = x.mean(dim=3, keepdim=True)
        return x * self.mlp(time_pooled)


class MRAFCNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.silu = nn.SiLU(inplace=True)
        self.conv2 = MultiScaleDWConv(out_ch, out_ch, stride=stride)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        
        self.attn = FreqSpatialAttention(out_ch)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.silu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.attn(out)
        out += residual
        return self.silu(out)


class MRAFCNN(nn.Module):
    def __init__(self, num_classes=50, base_channels=64):
        super().__init__()
        bc = base_channels
        
        # 3-channel input: static mel + delta + delta-delta
        self.stem = nn.Sequential(
            nn.Conv2d(3, bc, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(bc),
            nn.SiLU(inplace=True)
        )
        
        self.stage1 = self._make_stage(bc,   bc*2, num_blocks=2, stride=2)
        self.stage2 = self._make_stage(bc*2, bc*4, num_blocks=2, stride=2)
        self.stage3 = self._make_stage(bc*4, bc*8, num_blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        out_ch = bc * 8
        
        # Head 1: Cross Entropy
        self.head_ce = nn.Linear(out_ch, num_classes)
        
        # Head 2: SupCon projection (128-d normalized)
        self.head_supcon = nn.Sequential(
            nn.Linear(out_ch, out_ch),
            nn.SiLU(inplace=True),
            nn.Linear(out_ch, 128)
        )
        
        # Head 3: Fused (stage2 + stage3 features)
        self.head_fused = nn.Linear(out_ch + bc*4, num_classes)

    def _make_stage(self, in_ch, out_ch, num_blocks, stride):
        blocks = [MRAFCNNBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            blocks.append(MRAFCNNBlock(out_ch, out_ch, 1))
        return nn.Sequential(*blocks)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        
        p2 = self.pool(s2).flatten(1)
        p3 = self.pool(s3).flatten(1)
        
        fused_features = torch.cat([p2, p3], dim=1)
        
        logits_ce = self.head_ce(p3)
        proj_supcon = self.head_supcon(p3)
        proj_supcon = F.normalize(proj_supcon, dim=1)
        logits_fused = self.head_fused(fused_features)
        
        return logits_fused, logits_ce, proj_supcon


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning loss."""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        labels = labels.contiguous().view(-1, 1)
        label_mask = torch.eq(labels, labels.T).float()
        label_mask = label_mask.masked_fill(mask, 0)
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        exp_logits = torch.exp(logits) * (~mask).float()
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        num_positives = label_mask.sum(1)
        pos_mask = (num_positives > 0)
        if not pos_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)
        log_prob_pos = (label_mask * log_prob).sum(1) / torch.clamp(num_positives, min=1.0)
        return -log_prob_pos[pos_mask].mean()


class MRAFCNNLoss(nn.Module):
    def __init__(self, num_classes=50, lambda_ce=0.4, lambda_ls=0.3, lambda_fused=0.3, smoothing=0.1):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.lambda_supcon = lambda_ls
        self.lambda_fused = lambda_fused
        self.ce_loss = nn.CrossEntropyLoss()
        self.supcon_loss = SupConLoss(temperature=0.1)
        self.fused_loss = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def forward(self, outputs, targets):
        logits_fused, logits_ce, proj_supcon = outputs
        loss_ce = self.ce_loss(logits_ce, targets)
        if targets.dim() == 2:
            hard_targets = targets.argmax(dim=1)
        else:
            hard_targets = targets
        loss_supcon = self.supcon_loss(proj_supcon, hard_targets)
        loss_fused = self.fused_loss(logits_fused, targets)
        return self.lambda_ce * loss_ce + self.lambda_supcon * loss_supcon + self.lambda_fused * loss_fused


def build_esc50(n_mels=128, n_frames=128):
    return MRAFCNN(num_classes=50, base_channels=64)

def build_urbansound8k(n_mels=128, n_frames=128):
    return MRAFCNN(num_classes=10, base_channels=64)
