import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class DenseNetFeature(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        self.features = model.features

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        return x


class EfficientNetFeature(nn.Module):
    def __init__(self, weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1):
        super().__init__()
        model = models.efficientnet_b0(weights=weights)
        self.features = model.features

    def forward(self, x):
        x = self.features(x)
        return x


class SwinFeature(nn.Module):
    def __init__(self, weights=models.Swin_T_Weights.IMAGENET1K_V1):
        super().__init__()
        model = models.swin_t(weights=weights)
        self.features = model.features
        self.norm = model.norm

    def forward(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class MultiFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = DenseNetFeature()
        self.effi = EfficientNetFeature()
        self.swin = SwinFeature()

    def forward(self, x):
        f1 = self.dense(x)
        f2 = self.effi(x)
        f3 = self.swin(x)
        return f1, f2, f3


class SoftMoE(nn.Module):
    def __init__(self, common_dim=512):
        super().__init__()
        self.experts = MultiFeatureExtractor()

        self.ln_dense = nn.LayerNorm(common_dim)
        self.ln_effi = nn.LayerNorm(common_dim)
        self.ln_swin = nn.LayerNorm(common_dim)

        self.proj_dense = nn.Conv2d(1024, common_dim, kernel_size=1)
        self.proj_effi = nn.Conv2d(1280, common_dim, kernel_size=1)
        self.proj_swin = nn.Conv2d(768, common_dim, kernel_size=1)

        self.router = nn.Sequential(
            nn.Linear(common_dim, common_dim // 2),
            nn.ReLU(),
            nn.Linear(common_dim // 2, 3)
        )

    def apply_ln(self, x, ln):
        x = x.permute(0, 2, 3, 1)
        x = ln(x)
        x = x.permute(0, 3, 1, 2)
        return x

    def forward(self, x):
        f_dense, f_effi, f_swin = self.experts(x)

        p_dense = self.apply_ln(self.proj_dense(f_dense), self.ln_dense)
        p_effi = self.apply_ln(self.proj_effi(f_effi), self.ln_effi)
        p_swin = self.apply_ln(self.proj_swin(f_swin), self.ln_swin)

        gap_dense = F.adaptive_avg_pool2d(p_dense, (1, 1)).flatten(1)
        gap_effi = F.adaptive_avg_pool2d(p_effi, (1, 1)).flatten(1)
        gap_swin = F.adaptive_avg_pool2d(p_swin, (1, 1)).flatten(1)

        context_vector = gap_dense + gap_effi + gap_swin

        gate_logits = self.router(context_vector)
        gate_weights = F.softmax(gate_logits, dim=1)

        w_dense = gate_weights[:, 0].view(-1, 1, 1, 1)
        w_effi = gate_weights[:, 1].view(-1, 1, 1, 1)
        w_swin = gate_weights[:, 2].view(-1, 1, 1, 1)

        out = (w_dense * p_dense) + (w_effi * p_effi) + (w_swin * p_swin)

        return out


class MoE_Classifier(nn.Module):
    def __init__(self, num_classes, p=0.3):
        super().__init__()
        self.feature_extractor = SoftMoE(common_dim=512)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=p)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)

        x = self.pool(features)
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        logits = self.fc(x)

        return logits