"""
PyTorch Dataset 类 — 数据加载与增强流水线
============================================

为训练、验证、测试提供统一的数据加载接口。
训练集: 应用 SpecAugment + 波形增强
验证集/测试集: 不做增强, 保持原始数据

Author: Speech CNN Project
Date: 2026-06
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader

# 添加父目录到路径以导入 preprocess 模块
sys.path.insert(0, str(Path(__file__).parent))
from preprocess import (
    SpecAugment, Config, load_and_preprocess_audio,
    pad_or_truncate, GSC_20_WORDS, BILINGUAL_40_WORDS
)


class SpeechCommandDataset(Dataset):
    """
    语音命令数据集

    支持两种模式:
      1. 从预处理的 .npy 文件加载 (快速)
      2. 从原始 .wav 文件实时处理 (灵活, 支持在线波形增强)
    """

    def __init__(self,
                 data_dir: str = '../data/processed',
                 split: str = 'train',
                 words: List[str] = None,
                 use_raw_audio: bool = False,
                 augment: bool = True,
                 specaugment: Optional[SpecAugment] = None,
                 use_waveform_aug: bool = True):
        """
        Args:
            data_dir: 预处理数据目录 或 原始GSC数据目录
            split: 'train', 'val', 'test'
            words: 词汇列表 (None=使用默认20词)
            use_raw_audio: True=从wav实时处理, False=从npy加载
            augment: 是否做数据增强 (仅训练集)
            specaugment: SpecAugment 实例
            use_waveform_aug: 是否在波形级别做增强
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.words = words if words is not None else GSC_20_WORDS
        self.num_classes = len(self.words)
        self.word_to_idx = {w: i for i, w in enumerate(self.words)}
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}
        self.augment = augment and (split == 'train')
        self.specaugment = specaugment
        self.use_waveform_aug = use_waveform_aug and self.augment
        self.use_raw_audio = use_raw_audio

        if use_raw_audio:
            self._init_from_raw()
        else:
            self._init_from_npy()

    def _init_from_npy(self):
        """从预处理好的 .npy 文件加载数据"""
        self.X = np.load(self.data_dir / f'X_{self.split}.npy')
        self.y = np.load(self.data_dir / f'y_{self.split}.npy')

        # 可选加载原始文件路径 (用于调试)
        fp_file = self.data_dir / f'filepaths_{self.split}.npy'
        if fp_file.exists():
            self.filepaths = np.load(fp_file, allow_pickle=True)
        else:
            self.filepaths = np.array([''] * len(self))

        print(f"[Dataset] {self.split}: {len(self)} samples, "
              f"shape={self.X.shape}, "
              f"classes={len(np.unique(self.y))}")

    def _init_from_raw(self):
        """从原始 GSC 数据集目录扫描文件"""
        from preprocess import scan_gsc_files
        self.filepaths, self.labels = scan_gsc_files(str(self.data_dir), self.words)
        # 随机划分 (简化版, 正式使用建议用 npy 模式)
        np.random.seed(Config.SEED)
        indices = np.random.permutation(len(self.filepaths))
        if self.split == 'test':
            idx = indices[:int(0.15 * len(indices))]
        elif self.split == 'val':
            start = int(0.15 * len(indices))
            end = start + int(0.15 * len(indices))
            idx = indices[start:end]
        else:
            idx = indices[int(0.3 * len(indices)):]

        self.filepaths = [self.filepaths[i] for i in idx]
        self.labels = [self.labels[i] for i in idx]
        print(f"[Dataset] {self.split}: {len(self.filepaths)} raw audio files, "
              f"classes={len(set(self.labels))}")

    def __len__(self) -> int:
        if self.use_raw_audio:
            return len(self.filepaths)
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            spec: (1, n_mels, n_frames) — 单通道频谱图
            label: () — 类别标签
        """
        if self.use_raw_audio:
            spec = self._load_from_audio(idx)
        else:
            spec = self.X[idx].copy()
            label = self.y[idx]

        label = self.y[idx] if not self.use_raw_audio else self.labels[idx]

        # 数据增强 (仅训练集)
        if self.augment:
            if self.use_waveform_aug and self.use_raw_audio:
                pass  # 波形增强已在 _load_from_audio 中处理
            if self.specaugment is not None:
                spec = self.specaugment(spec)

        # 添加 channel 维度: (n_mels, n_frames) → (1, n_mels, n_frames)
        spec = torch.from_numpy(spec).float().unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        return spec, label

    def _load_from_audio(self, idx: int) -> np.ndarray:
        """从原始音频文件加载并处理为频谱图"""
        filepath = self.filepaths[idx]

        # 加载音频
        import librosa
        audio, sr = librosa.load(filepath, sr=Config.SAMPLE_RATE, mono=True)

        # 固定长度
        target_len = int(Config.SAMPLE_RATE * Config.AUDIO_DURATION)
        audio = pad_or_truncate(audio, target_len)

        # 可选波形增强
        if self.use_waveform_aug:
            from preprocess import audio_augment_waveform
            audio = audio_augment_waveform(audio, sr)

        # 计算梅尔频谱图
        from preprocess import compute_mel_spectrogram, normalize_spectrogram
        spec = compute_mel_spectrogram(audio, sr=sr, n_mels=Config.N_MELS)
        spec = normalize_spectrogram(spec, method='global')

        return spec

    def get_class_weights(self) -> torch.Tensor:
        """
        计算类别权重 (用于处理类别不平衡的加权 Loss)

        Returns:
            weights: shape (num_classes,)
        """
        if self.use_raw_audio:
            labels = np.array(self.labels)
        else:
            labels = self.y

        counts = np.bincount(labels, minlength=self.num_classes)
        counts = np.maximum(counts, 1)  # 避免除零
        weights = 1.0 / counts
        weights = weights / weights.sum() * self.num_classes  # 归一化
        return torch.tensor(weights, dtype=torch.float32)


class NoiseTestDataset(Dataset):
    """
    噪声鲁棒性测试数据集

    在给定 SNR 下对测试集添加噪声, 用于绘制 SNR-Accuracy 曲线。
    """

    def __init__(self,
                 base_dataset: SpeechCommandDataset,
                 snr_db: float,
                 noise_type: str = 'gaussian'):
        """
        Args:
            base_dataset: 基础测试集 (augment=False)
            snr_db: 目标 SNR (dB)
            noise_type: 噪声类型
        """
        self.base_dataset = base_dataset
        self.snr_db = snr_db
        self.noise_type = noise_type

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            noisy_spec: 加噪频谱图
            clean_spec: 原始频谱图
            label: 类别标签
        """
        from preprocess import add_noise_for_snr_test

        clean_spec, label = self.base_dataset[idx]
        clean_spec_np = clean_spec.squeeze(0).numpy()

        # 还原到波形域再加噪 (模拟真实场景)
        # 简化处理: 直接在频谱上近似加噪
        signal_power = np.mean(clean_spec_np ** 2)
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = signal_power / (snr_linear + 1e-10)
        noise = np.sqrt(noise_power) * np.random.randn(*clean_spec_np.shape)

        noisy_spec = clean_spec_np + noise.astype(np.float32)
        noisy_spec = torch.from_numpy(noisy_spec).unsqueeze(0)

        return noisy_spec, clean_spec, label


def create_dataloaders(data_dir: str = '../data/processed',
                       batch_size: int = 32,
                       num_workers: int = 2,
                       augment: bool = True,
                       specaugment_params: dict = None) -> Dict[str, DataLoader]:
    """
    创建训练/验证/测试 DataLoader

    Args:
        data_dir: 预处理数据目录
        batch_size: 批大小
        num_workers: 数据加载进程数（Windows 上强制为 0，避免 multiprocessing 问题）
        augment: 是否增强训练集
        specaugment_params: SpecAugment 参数字典

    Returns:
        dataloaders: {'train', 'val', 'test'} → DataLoader
    """
    import platform
    # Windows 上 multiprocessing spawn 方式容易导致 pickle 错误
    if platform.system() == 'Windows' and num_workers > 0:
        print(f"[INFO] Windows detected, setting num_workers=0 (was {num_workers})")
        num_workers = 0
    # SpecAugment 配置
    if specaugment_params is None:
        specaugment_params = dict(
            freq_mask_param=27,
            time_mask_param=25,
            n_freq_masks=2,
            n_time_masks=2,
            time_warp_param=0,
        )

    specaug = SpecAugment(**specaugment_params) if augment else None

    datasets = {}
    for split in ['train', 'val', 'test']:
        datasets[split] = SpeechCommandDataset(
            data_dir=data_dir,
            split=split,
            augment=(augment and split == 'train'),
            specaugment=specaug if split == 'train' else None,
            use_raw_audio=False,
        )

    dataloaders = {}
    for split, ds in datasets.items():
        shuffle = (split == 'train')
        dataloaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train'),  # 训练时丢掉最后不完整的 batch
        )

    return dataloaders


if __name__ == '__main__':
    # 快速测试 Dataset 是否正常工作
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/processed')
    args = parser.parse_args()

    # 测试 npy 模式
    ds = SpeechCommandDataset(data_dir=args.data_dir, split='train', augment=True)
    print(f"\n[Test] Dataset length: {len(ds)}")
    spec, label = ds[0]
    print(f"[Test] Sample 0: spec.shape={spec.shape}, label={label}, word='{ds.idx_to_word[label.item()]}'")

    # 测试 SpecAugment
    specaug = SpecAugment(freq_mask_param=27, time_mask_param=25)
    spec_np = spec.squeeze(0).numpy()
    spec_aug = specaug(spec_np)
    print(f"[Test] SpecAugment: original range [{spec_np.min():.2f}, {spec_np.max():.2f}], "
          f"augmented range [{spec_aug.min():.2f}, {spec_aug.max():.2f}]")

    print("\n[Test] All tests passed!")
