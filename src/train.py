"""
训练主脚本 — 端到端 CNN 语音命令分类训练流水线
===============================================

完整的训练流程:
  1. 加载预处理数据
  2. 创建 DataLoader + 数据增强
  3. 构建 CNN 模型
  4. 训练循环 (含验证、早停、学习率调度)
  5. 保存最佳模型
  6. 在测试集上评估

用法:
  python train.py                          # 默认配置训练
  python train.py --model light            # 轻量模型快速实验
  python train.py --epochs 150 --lr 0.001  # 自定义超参数
  python train.py --resume checkpoints/    # 从断点恢复训练

Author: Speech CNN Project
Date: 2026-06
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

# 导入本地模块
sys.path.insert(0, str(Path(__file__).parent))
from model import create_model, SpeechCNN, DenoisingCAE
from dataset import SpeechCommandDataset, create_dataloaders, SpecAugment
from preprocess import Config


# ============================================================
# 训练配置
# ============================================================

class TrainConfig:
    """训练超参数配置"""
    # 数据
    DATA_DIR: str = '../data/processed'
    NUM_CLASSES: int = 20

    # 模型
    MODEL_TYPE: str = 'standard'  # 'standard' | 'light' | 'deep'

    # 训练
    BATCH_SIZE: int = 32
    EPOCHS: int = 100
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    LABEL_SMOOTHING: float = 0.1

    # 优化器
    OPTIMIZER: str = 'adamw'  # 'adam' | 'adamw' | 'sgd'

    # 学习率调度
    LR_SCHEDULER: str = 'plateau'  # 'plateau' | 'cosine' | 'none'
    LR_PATIENCE: int = 8
    LR_FACTOR: float = 0.5
    LR_MIN: float = 1e-6

    # 早停
    EARLY_STOPPING_PATIENCE: int = 20

    # 数据增强
    AUGMENT: bool = True

    # 过采样: 强制中英 batch 均衡 (仅 40 类有效)
    OVER_SAMPLE: bool = False
    SPECAUGMENT_FREQ_MASK: int = 27
    SPECAUGMENT_TIME_MASK: int = 25
    SPECAUGMENT_N_FREQ: int = 2
    SPECAUGMENT_N_TIME: int = 2

    # 混合精度训练
    USE_AMP: bool = True  # 仅 GPU 可用

    # 日志
    LOG_INTERVAL: int = 50  # 每 N 个 batch 打印一次

    # 设备
    DEVICE: str = 'auto'  # 'auto' | 'cuda' | 'cpu'

    # 类别加权 (处理数据不均衡)
    USE_CLASS_WEIGHTS: bool = False
    CLASS_WEIGHT_POWER: float = 1.0  # 1.0=full inverse freq, 0.5=sqrt, 0=uniform

    # 输出
    OUTPUT_DIR: str = '../models'
    RUN_NAME: str = None  # None = 自动生成时间戳

    # 随机种子
    SEED: int = 42


def set_seed(seed: int):
    """固定随机种子以确保可复现性"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str = 'auto') -> torch.device:
    """自动选择最佳设备"""
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device_str)


# ============================================================
# 训练引擎
# ============================================================

class Trainer:
    """
    训练器 — 封装完整的训练/验证/测试逻辑

    功能:
      - 自动设备管理 (CUDA/MPS/CPU)
      - 混合精度训练 (AMP)
      - 学习率调度 (Plateau/Cosine)
      - 早停 (Early Stopping)
      - 模型断点保存/恢复
      - TensorBoard 兼容日志
    """

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 config: TrainConfig,
                 device: torch.device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device

        # 损失函数 (支持标签平滑 + 类别加权)
        class_weights = None
        if config.USE_CLASS_WEIGHTS:
            all_labels = train_loader.dataset.y
            counts = np.bincount(all_labels, minlength=config.NUM_CLASSES)
            counts = np.maximum(counts, 1)
            inv_freq = np.power(1.0 / counts, config.CLASS_WEIGHT_POWER)
            class_weights = torch.tensor(
                len(counts) * inv_freq / np.sum(inv_freq),
                dtype=torch.float32
            ).to(device)
            print(f"[INFO] Class weights: EN avg={class_weights[:20].mean():.1f} "
                  f"ZH avg={class_weights[20:].mean():.1f}")
            print(f"[INFO] Class weight power: {config.CLASS_WEIGHT_POWER:.2f}")
            if config.OVER_SAMPLE and config.NUM_CLASSES >= 40:
                print("[WARN] Both --over_sample and --use_class_weights are enabled.")
                print("[WARN] This can over-correct Chinese classes; consider --class_weight_power 0.5 or disabling one.")

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config.LABEL_SMOOTHING
        )

        # 优化器
        self.optimizer = self._create_optimizer()

        # 学习率调度器
        self.scheduler = self._create_scheduler()

        # 混合精度
        self.scaler = GradScaler(enabled=config.USE_AMP and device.type == 'cuda')
        self.use_amp = config.USE_AMP and device.type == 'cuda'

        # 训练状态
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.train_history = {'loss': [], 'acc': []}
        self.val_history = {'loss': [], 'acc': []}

        # 输出目录 (自动按 V1/V2 分类)
        self.run_name = config.RUN_NAME or datetime.now().strftime('%Y%m%d_%H%M%S')
        if config.NUM_CLASSES >= 40:
            version_subdir = 'v2_bilingual_40'
        else:
            version_subdir = 'v1_english_20'
        self.output_dir = Path(config.OUTPUT_DIR) / version_subdir / self.run_name
        if self.output_dir.exists():
            print(f"[WARN] Output directory already exists: {self.output_dir}")
            print("[WARN] New checkpoints/results may overwrite files from a previous run.")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置
        self._save_config()

        print(f"\n{'='*60}")
        print(f"Trainer initialized")
        print(f"  Device:      {self.device}")
        print(f"  Model:       {self._count_params():,} params")
        print(f"  Optimizer:   {type(self.optimizer).__name__}")
        print(f"  Scheduler:   {type(self.scheduler).__name__}")
        print(f"  AMP:         {self.use_amp}")
        print(f"  Output dir:  {self.output_dir}")
        print(f"{'='*60}\n")

    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        lr = self.config.LEARNING_RATE
        wd = self.config.WEIGHT_DECAY

        if self.config.OPTIMIZER == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif self.config.OPTIMIZER == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif self.config.OPTIMIZER == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9,
                             weight_decay=wd, nesterov=True)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.OPTIMIZER}")

    def _create_scheduler(self):
        """创建学习率调度器"""
        if self.config.LR_SCHEDULER == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config.LR_FACTOR,
                patience=self.config.LR_PATIENCE,
                min_lr=self.config.LR_MIN,
                verbose=True,
            )
        elif self.config.LR_SCHEDULER == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.EPOCHS,
                eta_min=self.config.LR_MIN,
            )
        else:
            return None

    def _count_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _save_config(self):
        """保存训练配置为 JSON"""
        config_dict = {k: v for k, v in vars(self.config).items()
                       if not k.startswith('_') and k.isupper()}
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

    def train_epoch(self) -> Tuple[float, float]:
        """训练一个 epoch, 返回 (平均 loss, 准确率)"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            # 混合精度前向
            with autocast(enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            # 混合精度反向
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # 统计
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 日志
            if batch_idx % self.config.LOG_INTERVAL == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'  Batch [{batch_idx:4d}/{len(self.train_loader)}] '
                      f'Loss: {loss.item():.4f} '
                      f'Acc: {100.*correct/total:.1f}% '
                      f'LR: {current_lr:.2e}')

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, desc: str = 'Eval') -> Tuple[float, float]:
        """评估模型, 返回 (平均 loss, 准确率)"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def train(self):
        """主训练循环"""
        print(f"{'='*60}")
        print(f"Starting training: {self.config.EPOCHS} epochs")
        print(f"{'='*60}")

        start_time = time.time()

        for epoch in range(self.current_epoch, self.config.EPOCHS):
            epoch_start = time.time()

            # 训练
            train_loss, train_acc = self.train_epoch()

            # 验证
            val_loss, val_acc = self.evaluate(self.val_loader, 'Val')

            # 记录历史
            self.train_history['loss'].append(train_loss)
            self.train_history['acc'].append(train_acc)
            self.val_history['loss'].append(val_loss)
            self.val_history['acc'].append(val_acc)

            # 学习率调度
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()

            # 耗时
            epoch_time = time.time() - epoch_start

            # 打印
            best_marker = ' ★' if val_acc > self.best_val_acc else ''
            print(f'\nEpoch [{epoch+1:3d}/{self.config.EPOCHS}] '
                  f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%{best_marker} | '
                  f'Time: {epoch_time:.1f}s | LR: {current_lr:.2e}\n')

            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                self._save_checkpoint('best_model.pth', epoch, val_acc, is_best=True)
            else:
                self.patience_counter += 1

            # 定期保存
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, val_acc)

            # 早停检查
            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f'[EarlyStopping] No improvement for '
                      f'{self.config.EARLY_STOPPING_PATIENCE} epochs. Stopping.')
                break

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"  Best val acc: {self.best_val_acc:.2f}% at epoch {self.best_epoch}")
        print(f"  Total time:   {total_time/60:.1f} min")
        print(f"{'='*60}\n")

        # 保存训练历史
        self._save_history()

        # 加载最佳模型并在测试集上评估
        self._load_best_model()
        return self.best_val_acc

    def test(self) -> float:
        """在测试集上最终评估"""
        test_loss, test_acc = self.evaluate(self.test_loader, 'Test')
        print(f"\n{'='*60}")
        print(f"TEST RESULTS")
        print(f"  Loss:     {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.2f}%")
        print(f"{'='*60}\n")
        return test_acc

    def _save_checkpoint(self, filename: str, epoch: int, val_acc: float, is_best: bool = False):
        """保存模型断点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': {k: v for k, v in vars(self.config).items()
                       if not k.startswith('_') and k.isupper()},
        }
        torch.save(checkpoint, self.output_dir / filename)

        if is_best:
            # 额外保存一份纯模型权重 (方便推理)
            torch.save(self.model.state_dict(), self.output_dir / 'model_weights.pth')
            print(f'  [Checkpoint] Best model saved (acc={val_acc:.2f}%)')

    def _load_best_model(self):
        """加载最佳模型"""
        best_path = self.output_dir / 'best_model.pth'
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f'[INFO] Loaded best model (epoch {checkpoint["best_epoch"]}, '
                  f'acc={checkpoint["best_val_acc"]:.2f}%)')
        else:
            print('[WARNING] No best model checkpoint found, using current model')

    def _save_history(self):
        """保存训练历史为 JSON"""
        history = {
            'train_loss': [float(x) for x in self.train_history['loss']],
            'train_acc': [float(x) for x in self.train_history['acc']],
            'val_loss': [float(x) for x in self.val_history['loss']],
            'val_acc': [float(x) for x in self.val_history['acc']],
            'best_val_acc': float(self.best_val_acc),
            'best_epoch': int(self.best_epoch),
        }
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

    def resume_from_checkpoint(self, checkpoint_path: str):
        """从断点恢复训练"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_epoch = checkpoint['best_epoch']
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
        print(f'[INFO] Resumed from epoch {self.current_epoch}, '
              f'best val acc={self.best_val_acc:.2f}%')


# ============================================================
# 主入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Speech CNN for Voice Command Recognition'
    )

    # 数据
    parser.add_argument('--data_dir', type=str, default=TrainConfig.DATA_DIR,
                        help='Preprocessed data directory')
    parser.add_argument('--num_classes', type=int, default=TrainConfig.NUM_CLASSES,
                        help='Number of classes')

    # 模型
    parser.add_argument('--model', type=str, default=TrainConfig.MODEL_TYPE,
                        choices=['standard', 'light', 'deep'],
                        help='Model architecture')

    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=TrainConfig.BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=TrainConfig.EPOCHS)
    parser.add_argument('--lr', type=float, default=TrainConfig.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=TrainConfig.WEIGHT_DECAY)
    parser.add_argument('--optimizer', type=str, default=TrainConfig.OPTIMIZER,
                        choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--lr_scheduler', type=str, default=TrainConfig.LR_SCHEDULER,
                        choices=['plateau', 'cosine', 'none'])

    # 增强
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Auto-balance class weights (for imbalanced data)')
    parser.add_argument('--class_weight_power', type=float, default=TrainConfig.CLASS_WEIGHT_POWER,
                        help='Class weight strength: 1.0=inverse freq, 0.5=sqrt, 0=uniform')
    parser.add_argument('--over_sample', action='store_true',
                        help='Force equal EN/ZH samples per batch (V4)')
    parser.add_argument('--label_smoothing', type=float, default=TrainConfig.LABEL_SMOOTHING)

    # 设备
    parser.add_argument('--device', type=str, default=TrainConfig.DEVICE,
                        choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision')

    # 其他
    parser.add_argument('--output_dir', type=str, default=TrainConfig.OUTPUT_DIR)
    parser.add_argument('--run_name', type=str, default=TrainConfig.RUN_NAME)
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--seed', type=int, default=TrainConfig.SEED)
    parser.add_argument('--num_workers', type=int, default=2)

    return parser.parse_args()


def main():
    args = parse_args()

    # 固定随机种子
    set_seed(args.seed)

    # 设备
    device = get_device(args.device)
    print(f"[INFO] Using device: {device}")

    # 配置
    config = TrainConfig()
    config.DATA_DIR = args.data_dir
    config.NUM_CLASSES = args.num_classes
    config.MODEL_TYPE = args.model
    config.BATCH_SIZE = args.batch_size
    config.EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    config.WEIGHT_DECAY = args.weight_decay
    config.OPTIMIZER = args.optimizer
    config.LR_SCHEDULER = args.lr_scheduler
    config.AUGMENT = not args.no_augment
    config.USE_CLASS_WEIGHTS = args.use_class_weights
    config.CLASS_WEIGHT_POWER = args.class_weight_power
    config.OVER_SAMPLE = args.over_sample
    config.LABEL_SMOOTHING = args.label_smoothing
    config.DEVICE = args.device
    config.USE_AMP = not args.no_amp
    config.OUTPUT_DIR = args.output_dir
    config.RUN_NAME = args.run_name
    config.SEED = args.seed

    # 创建 DataLoader
    specaug_params = dict(
        freq_mask_param=config.SPECAUGMENT_FREQ_MASK,
        time_mask_param=config.SPECAUGMENT_TIME_MASK,
        n_freq_masks=config.SPECAUGMENT_N_FREQ,
        n_time_masks=config.SPECAUGMENT_N_TIME,
    ) if config.AUGMENT else None

    dataloaders = create_dataloaders(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=args.num_workers,
        augment=config.AUGMENT,
        specaugment_params=specaug_params,
    )

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']

    # V4 过采样: 强制每个 batch 中英各半
    if config.OVER_SAMPLE and config.NUM_CLASSES >= 40:
        from torch.utils.data import WeightedRandomSampler
        train_labels = train_loader.dataset.y
        en_mask = train_labels < 20
        zh_mask = train_labels >= 20
        en_count = en_mask.sum()
        zh_count = zh_mask.sum()
        # 权重: 中文样本权重 = EN/ZH, 使中英文被采样概率均等
        sample_weights = np.ones(len(train_labels), dtype=np.float32)
        sample_weights[zh_mask] = en_count / max(zh_count, 1)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_labels),
            replacement=True,
        )
        import platform
        nw = 0 if platform.system() == 'Windows' else args.num_workers
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=config.BATCH_SIZE,
            sampler=sampler,
            num_workers=nw,
            pin_memory=True,
        )
        print(f"[INFO] Over-sampling enabled: EN={en_count}, ZH={zh_count}, "
              f"ZH weight={en_count/zh_count:.1f}x")

    # 获取输入形状
    sample_input, _ = train_loader.dataset[0]
    input_shape = (sample_input.shape[1], sample_input.shape[2])  # (n_mels, n_frames)
    print(f"[INFO] Input shape: {input_shape}")

    # 创建模型
    model = create_model(
        model_type=config.MODEL_TYPE,
        num_classes=config.NUM_CLASSES,
        input_shape=input_shape,
    )

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
    )

    # 断点恢复
    if args.resume:
        trainer.resume_from_checkpoint(args.resume)

    # 训练
    try:
        trainer.train()

        # 测试
        test_acc = trainer.test()

        # 写入最终结果
        with open(trainer.output_dir / 'final_results.txt', 'w') as f:
            f.write(f"Test Accuracy: {test_acc:.2f}%\n")
            f.write(f"Best Validation Accuracy: {trainer.best_val_acc:.2f}%\n")
            f.write(f"Best Epoch: {trainer.best_epoch}\n")
            f.write(f"Model: {config.MODEL_TYPE}\n")
            f.write(f"Params: {trainer._count_params():,}\n")

        print(f"\n[SUCCESS] Training finished! Results saved to {trainer.output_dir}")

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted. Best model saved.")
        trainer._save_checkpoint('interrupted.pth', trainer.current_epoch, trainer.best_val_acc)

    return trainer


if __name__ == '__main__':
    main()
