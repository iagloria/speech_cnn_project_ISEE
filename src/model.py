"""
CNN 模型定义 — 语音命令分类器 + 可选的降噪自编码器
=====================================================

主模型: SpeechCNN
  - 3 个卷积块, 每个包含 2 层 Conv2D
  - BatchNorm + ReLU + MaxPool + Dropout
  - Global Average Pooling → Dense → Softmax
  - 参数量: ~400K (轻量, CPU 可训练)

可选模块: DenoisingCAE (卷积自编码器)
  - 对标 EMG 项目的 AE+CNN 方案
  - 用于频谱图降噪, 提升噪声鲁棒性

Author: Speech CNN Project
Date: 2026-06
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 卷积块组件
# ============================================================

class ConvBlock(nn.Module):
    """
    双卷积块: Conv2D → BN → ReLU → Conv2D → BN → ReLU → MaxPool → Dropout

    每个 Block 包含两层卷积, 增强特征提取能力,
    MaxPool 降低空间分辨率, Dropout 防过拟合。
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 pool_size: int = 2,
                 dropout_rate: float = 0.2):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size,
                               padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.pool(x)
        x = self.dropout(x)
        return x


# ============================================================
# 主模型: SpeechCNN
# ============================================================

class SpeechCNN(nn.Module):
    """
    语音命令分类 CNN

    架构:
      Input: (1, n_mels, n_frames)           ≈ (1, 128, 100)
        ↓
      ConvBlock(1 → 32):  (32, 128, 100) → (32, 64, 50)
      ConvBlock(32 → 64):   (64, 64, 50)  → (64, 32, 25)
      ConvBlock(64 → 128):  (128, 32, 25) → (128, 16, 12)
        ↓
      GlobalAveragePooling2D:  (128, 16, 12) → (128,)
        ↓
      Dense(128 → 256) → ReLU → Dropout(0.5)
        ↓
      Dense(256 → num_classes) → Softmax

    设计理念:
      - 逐步增加通道数 (32→64→128), 学习从低级到高级的特征
      - GAP 替代 Flatten, 大幅减少参数量, 且具有平移不变性
      - 两层 Dense 提供足够的分类能力

    Args:
        num_classes: 分类类别数
        input_shape: (n_mels, n_frames) 输入形状
        dropout_rates: 各 Block 的 Dropout 率
    """

    def __init__(self,
                 num_classes: int = 20,
                 input_shape: Tuple[int, int] = (128, 100),
                 dropout_rates: Tuple[float, ...] = (0.2, 0.3, 0.4)):
        super().__init__()

        self.num_classes = num_classes
        self.input_shape = input_shape

        # 三个卷积块
        self.block1 = ConvBlock(1, 32, kernel_size=3, dropout_rate=dropout_rates[0])
        self.block2 = ConvBlock(32, 64, kernel_size=3, dropout_rate=dropout_rates[1])
        self.block3 = ConvBlock(64, 128, kernel_size=3, dropout_rate=dropout_rates[2])

        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self._initialize_weights()
        self._print_model_info()

    def _initialize_weights(self):
        """Kaiming 初始化 — 适配 ReLU 激活函数"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _print_model_info(self):
        """打印模型参数统计"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[SpeechCNN] num_classes={self.num_classes}, "
              f"input_shape={self.input_shape}")
        print(f"[SpeechCNN] Total params: {total_params:,} "
              f"({total_params/1e6:.2f}M)")
        print(f"[SpeechCNN] Trainable params: {trainable_params:,} "
              f"({trainable_params/1e6:.2f}M)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, n_mels, n_frames)

        Returns:
            logits: (batch, num_classes)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)  # (batch, 128)
        x = self.classifier(x)
        return x

    def extract_features(self, x: torch.Tensor,
                          layer_name: str = 'gap') -> torch.Tensor:
        """
        提取中间层特征 (用于可视化, 如 t-SNE)

        Args:
            x: (batch, 1, n_mels, n_frames)
            layer_name: 'block1', 'block2', 'block3', 'gap', 'fc1'

        Returns:
            features: (batch, feature_dim)
        """
        x = self.block1(x)
        if layer_name == 'block1':
            return torch.flatten(self.gap(x), 1)

        x = self.block2(x)
        if layer_name == 'block2':
            return torch.flatten(self.gap(x), 1)

        x = self.block3(x)
        if layer_name == 'block3':
            return torch.flatten(self.gap(x), 1)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        if layer_name == 'gap':
            return x

        # 分类器第一层 (fc1)
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            if layer_name == 'fc1' and i == 0:
                return x

        return x


# ============================================================
# 可选模块: 降噪卷积自编码器 (对应 EMG 项目的 AE+CNN 方案)
# ============================================================

class DenoisingCAE(nn.Module):
    """
    降噪卷积自编码器 (Denoising Convolutional Autoencoder)

    对标 EMG 手写识别项目中的 AE_with_same_input_of_NN.py。
    用于在频谱图层面做降噪: 加噪频谱 → CAE → 干净频谱。

    训练完成后:
      encoder = CAE.encoder
      features = encoder(noisy_spec)  → CNN → 分类

    架构:
      Encoder: Conv2D(1→16→32) + MaxPool → 压缩特征
      Decoder: Upsample + Conv2D(32→16→1) → 重建频谱
    """

    def __init__(self, input_channels: int = 1, latent_channels: int = 32):
        super().__init__()

        # --- Encoder ---
        self.encoder = nn.Sequential(
            # Layer 1: (1, H, W) → (16, H, W)
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # → (16, H/2, W/2)

            # Layer 2: (16, H/2, W/2) → (32, H/2, W/2)
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # → (32, H/4, W/4)

            # Layer 3 (bottleneck): (32, H/4, W/4) → (32, H/4, W/4)
            nn.Conv2d(32, latent_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
        )

        # --- Decoder ---
        self.decoder = nn.Sequential(
            # Layer 1: (32, H/4, W/4) → (16, H/2, W/2)
            nn.Conv2d(latent_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # Layer 2: (16, H/2, W/2) → (8, H, W)
            nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # Layer 3: (8, H, W) → (1, H, W)
            nn.Conv2d(8, input_channels, kernel_size=3, padding=1),
            nn.Tanh(),  # 输出范围 [-1, 1], 与归一化后的频谱匹配
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, 1, H, W) 加噪频谱

        Returns:
            reconstructed: (batch, 1, H, W) 重建频谱
            encoded: (batch, C, H/4, W/4) 编码特征
        """
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed, encoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """仅编码 (提取特征)"""
        return self.encoder(x)


# ============================================================
# 轻量版模型 (用于快速实验/嵌入式部署)
# ============================================================

class SpeechCNNLight(nn.Module):
    """
    轻量版 SpeechCNN — 参数量更少, 速度更快

    适用场景:
      - CPU 训练
      - 快速原型验证
      - 资源受限环境

    参数量: ~80K
    """

    def __init__(self, num_classes: int = 20, input_shape: Tuple[int, int] = (128, 100)):
        super().__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(24, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(48, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(96, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ============================================================
# 模型工厂函数
# ============================================================

def create_model(model_type: str = 'standard',
                 num_classes: int = 20,
                 input_shape: Tuple[int, int] = (128, 100),
                 **kwargs) -> nn.Module:
    """
    模型工厂 — 根据类型创建不同的模型

    Args:
        model_type: 'standard' | 'light' | 'deep'
        num_classes: 分类数
        input_shape: 输入形状

    Returns:
        model 实例
    """
    if model_type == 'standard':
        return SpeechCNN(num_classes=num_classes, input_shape=input_shape, **kwargs)
    elif model_type == 'light':
        return SpeechCNNLight(num_classes=num_classes, input_shape=input_shape)
    elif model_type == 'deep':
        # 加深版: 4 个 Block
        model = SpeechCNN(num_classes=num_classes, input_shape=input_shape)
        model.block4 = ConvBlock(128, 256, kernel_size=3, dropout_rate=0.4)
        # 调整分类器输入维度
        model.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================================
# 测试入口
# ============================================================

if __name__ == '__main__':
    # 模型前向传播测试
    batch_size = 4
    n_mels, n_frames = 128, 100
    num_classes = 20

    dummy_input = torch.randn(batch_size, 1, n_mels, n_frames)

    for model_type in ['standard', 'light', 'deep']:
        print(f"\n{'='*50}")
        print(f"Testing {model_type} model...")
        model = create_model(model_type, num_classes=num_classes,
                              input_shape=(n_mels, n_frames))

        # 前向传播
        output = model(dummy_input)
        print(f"Input:  {dummy_input.shape}")
        print(f"Output: {output.shape}")

        # 特征提取
        if hasattr(model, 'extract_features'):
            features = model.extract_features(dummy_input, 'gap')
            print(f"Features (gap): {features.shape}")

    # 测试 CAE
    print(f"\n{'='*50}")
    print("Testing DenoisingCAE...")
    cae = DenoisingCAE(input_channels=1)
    noisy = dummy_input + 0.1 * torch.randn_like(dummy_input)
    recon, encoded = cae(noisy)
    print(f"Noisy input:  {noisy.shape}")
    print(f"Reconstructed: {recon.shape}")
    print(f"Encoded:       {encoded.shape}")
    loss = F.mse_loss(recon, dummy_input)
    print(f"MSE loss (vs clean): {loss.item():.4f}")

    print("\n[Test] All model tests passed!")
