import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, stride=stride, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))

class MRAFCNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.SiLU(inplace=True)
        # Expected by GradCAM in evaluate.py: target_layer = model.stage3[-1].conv2.pointwise
        self.conv2 = DepthwiseSeparableConv(out_ch, out_ch, stride=stride)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
            
        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch // 4 if out_ch // 4 > 0 else 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch // 4 if out_ch // 4 > 0 else 1, out_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = out * self.ca(out)
        out += residual
        return self.relu(out)

class MRAFCNN(nn.Module):
    def __init__(self, num_classes=50, base_channels=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True)
        )
        
        self.stage1 = self._make_stage(base_channels, base_channels*2, num_blocks=2, stride=2)
        self.stage2 = self._make_stage(base_channels*2, base_channels*4, num_blocks=2, stride=2)
        self.stage3 = self._make_stage(base_channels*4, base_channels*8, num_blocks=2, stride=2)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        out_ch = base_channels*8
        self.head_ce = nn.Linear(out_ch, num_classes)
        self.head_ls = nn.Linear(out_ch, num_classes)
        self.head_fused = nn.Linear(out_ch + base_channels*4, num_classes)

    def _make_stage(self, in_ch, out_ch, num_blocks, stride):
        blocks = []
        blocks.append(MRAFCNNBlock(in_ch, out_ch, stride))
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
        logits_ls = self.head_ls(p3)
        logits_fused = self.head_fused(fused_features)
        
        return logits_fused, logits_ce, logits_ls

class MRAFCNNLoss(nn.Module):
    def __init__(self, num_classes=50, lambda_ce=0.4, lambda_ls=0.3, lambda_fused=0.3, smoothing=0.1):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.lambda_ls = lambda_ls
        self.lambda_fused = lambda_fused
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.ls_loss = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.fused_loss = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def forward(self, outputs, targets):
        logits_fused, logits_ce, logits_ls = outputs
        
        loss_ce = self.ce_loss(logits_ce, targets)
        loss_ls = self.ls_loss(logits_ls, targets)
        loss_fused = self.fused_loss(logits_fused, targets)
        
        return self.lambda_ce * loss_ce + self.lambda_ls * loss_ls + self.lambda_fused * loss_fused

def build_esc50(n_mels=128, n_frames=128):
    return MRAFCNN(num_classes=50)

def build_urbansound8k(n_mels=128, n_frames=128):
    return MRAFCNN(num_classes=10)
