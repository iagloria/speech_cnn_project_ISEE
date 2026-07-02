"""
评估与可视化脚本 — 模型性能全面评测
=======================================

功能:
  1. 测试集评估 (Accuracy, F1, Precision, Recall)
  2. 混淆矩阵 (Confusion Matrix)
  3. ROC 曲线 (one-vs-rest)
  4. t-SNE 特征空间可视化
  5. 噪声鲁棒性测试 (SNR-Accuracy 曲线)
  6. CNN 特征图可视化
  7. 训练历史曲线绘制
  8. 错误分析 (展示分类错误的样本)

用法:
  python evaluate.py --model_path ../models/xxx/best_model.pth
  python evaluate.py --model_path ../models/xxx/best_model.pth --all

Author: Speech CNN Project
Date: 2026-06
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')  # 无头模式, 不弹出窗口
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, roc_curve,
    classification_report
)
from tqdm import tqdm

# 导入本地模块
sys.path.insert(0, str(Path(__file__).parent))
from model import create_model, SpeechCNN
from dataset import SpeechCommandDataset, create_dataloaders
from preprocess import GSC_20_WORDS, BILINGUAL_40_WORDS, add_noise_for_snr_test, load_and_preprocess_audio

# 全局设置
sns.set_style('whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})


SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_path_from_script(path_value: str) -> Path:
    """Resolve paths stored by training commands launched from src/."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


class Evaluator:
    """
    模型评估器 — 全面的性能评测和可视化
    """

    def __init__(self,
                 model: nn.Module,
                 test_loader: DataLoader,
                 device: torch.device,
                 output_dir: str = '../results',
                 word_list: List[str] = None,
                 train_history: dict = None,
                 metadata: dict = None):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 确定 num_classes：优先从 word_list，其次从模型
        if word_list is not None:
            self.word_list = word_list
        elif hasattr(model, 'num_classes'):
            if model.num_classes == 40:
                self.word_list = BILINGUAL_40_WORDS
            else:
                self.word_list = GSC_20_WORDS
        else:
            self.word_list = GSC_20_WORDS
        self.num_classes = len(self.word_list)
        self.train_history = train_history
        self.metadata = metadata or {}

        # 缓存预测结果
        self._predictions = None
        self._targets = None
        self._probabilities = None

    @torch.no_grad()
    def _get_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取所有测试样本的预测结果"""
        if self._predictions is not None:
            return self._predictions, self._targets, self._probabilities

        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []

        for inputs, targets in tqdm(self.test_loader, desc='Evaluating'):
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            probs = F.softmax(outputs, dim=1)

            all_preds.append(outputs.argmax(1).cpu().numpy())
            all_targets.append(targets.numpy())
            all_probs.append(probs.cpu().numpy())

        self._predictions = np.concatenate(all_preds)
        self._targets = np.concatenate(all_targets)
        self._probabilities = np.concatenate(all_probs)

        return self._predictions, self._targets, self._probabilities

    def compute_metrics(self) -> Dict:
        """计算全部分类指标"""
        preds, targets, probs = self._get_predictions()

        accuracy = accuracy_score(targets, preds)

        # 各类别指标
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, preds, labels=range(self.num_classes), average=None, zero_division=0
        )

        # 宏平均
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            targets, preds, average='macro', zero_division=0
        )

        # 加权平均
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            targets, preds, average='weighted', zero_division=0
        )

        metrics = {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'metadata': self.metadata,
            'per_class': {},
        }

        for i, word in enumerate(self.word_list):
            metrics['per_class'][word] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i]),
            }

        # 打印
        print(f"\n{'='*60}")
        print("CLASSIFICATION METRICS")
        print(f"{'='*60}")
        print(f"Accuracy:            {accuracy*100:.2f}%")
        print(f"Macro F1:            {macro_f1*100:.2f}%")
        print(f"Weighted F1:         {weighted_f1*100:.2f}%")
        print(f"\n{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
        print("-" * 52)
        for i, word in enumerate(self.word_list):
            print(f"{word:<12} {precision[i]*100:>9.1f}% {recall[i]*100:>9.1f}% "
                  f"{f1[i]*100:>9.1f}% {support[i]:>8}")
        print(f"{'='*60}\n")

        # 保存
        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def plot_confusion_matrix(self, normalize: bool = True, save: bool = True):
        """绘制混淆矩阵"""
        preds, targets, _ = self._get_predictions()
        cm = confusion_matrix(targets, preds, labels=range(self.num_classes))

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)

        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(cm, cmap='Blues', aspect='auto')

        # 标注
        if normalize:
            fmt = '.0%'
        else:
            fmt = 'd'
        threshold = cm.max() / 2
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if cm[i, j] > 0.01 or not normalize:
                    color = 'white' if cm[i, j] > threshold else 'black'
                    if normalize:
                        text = f'{cm[i,j]:.1%}'
                    else:
                        text = f'{cm[i,j]:.0f}'
                    ax.text(j, i, text, ha='center', va='center',
                            color=color, fontsize=8)

        ax.set_xticks(range(self.num_classes))
        ax.set_yticks(range(self.num_classes))
        ax.set_xticklabels(self.word_list, rotation=45, ha='right')
        ax.set_yticklabels(self.word_list)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))

        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()

        if save:
            suffix = '_normalized' if normalize else '_counts'
            plt.savefig(self.output_dir / f'confusion_matrix{suffix}.png')
            print(f"[INFO] Confusion matrix saved to {self.output_dir}/confusion_matrix{suffix}.png")
        plt.close()

    def plot_roc_curves(self, save: bool = True):
        """绘制各类别 ROC 曲线 (one-vs-rest)"""
        _, targets, probs = self._get_predictions()

        n_classes = min(self.num_classes, 10)  # 最多显示 10 类
        display_indices = list(range(n_classes))

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        for i, idx in enumerate(display_indices):
            y_true_bin = (targets == idx).astype(int)
            y_score = probs[:, idx]

            if len(np.unique(y_true_bin)) < 2:
                continue

            fpr, tpr, _ = roc_curve(y_true_bin, y_score)
            auc = roc_auc_score(y_true_bin, y_score)
            ax.plot(fpr, tpr, color=colors[i], lw=1.5,
                    label=f'{self.word_list[idx]} (AUC={auc:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves (One-vs-Rest)')
        ax.legend(loc='lower right', fontsize=8, ncol=2)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'roc_curves.png')
            print(f"[INFO] ROC curves saved to {self.output_dir}/roc_curves.png")
        plt.close()

    def plot_tsne_features(self, n_samples: int = 2000, save: bool = True):
        """t-SNE 特征空间可视化"""
        print("[INFO] Computing t-SNE features...")

        # 收集特征
        self.model.eval()
        all_features = []
        all_labels = []

        n_collected = 0
        for inputs, targets in self.test_loader:
            if n_collected >= n_samples:
                break
            inputs = inputs.to(self.device)
            features = self.model.extract_features(inputs, 'gap')
            all_features.append(features.detach().cpu().numpy())
            all_labels.append(targets.numpy())
            n_collected += len(inputs)

        features = np.concatenate(all_features)[:n_samples]
        labels = np.concatenate(all_labels)[:n_samples]

        # t-SNE 降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        features_2d = tsne.fit_transform(features)

        # 绘图
        fig, ax = plt.subplots(figsize=(12, 10))
        n_classes = min(self.num_classes, 20)
        colors = plt.cm.tab20(np.linspace(0, 1, n_classes))

        for i in range(n_classes):
            mask = labels == i
            if mask.sum() > 0:
                ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                           c=[colors[i]], label=self.word_list[i],
                           s=10, alpha=0.7, edgecolors='none')

        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title('t-SNE Visualization of CNN Feature Space')
        ax.legend(loc='lower left', fontsize=7, ncol=2, markerscale=3)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'tsne_features.png')
            print(f"[INFO] t-SNE plot saved to {self.output_dir}/tsne_features.png")
        plt.close()

    def plot_feature_maps(self, sample_input: torch.Tensor, save: bool = True):
        """可视化 CNN 各层特征图"""
        self.model.eval()
        x = sample_input.unsqueeze(0).to(self.device)  # (1, 1, H, W)

        # 逐层提取
        layers = {
            'block1_conv1': self.model.block1.conv1,
            'block1_conv2': self.model.block1.conv2,
            'block2_conv1': self.model.block2.conv1,
            'block2_conv2': self.model.block2.conv2,
            'block3_conv1': self.model.block3.conv1,
        }

        fig, axes = plt.subplots(len(layers), 8, figsize=(20, 3 * len(layers)))

        with torch.no_grad():
            for row_idx, (name, layer) in enumerate(layers.items()):
                x = layer(x)
                if 'conv1' in name and row_idx > 0:
                    # 每 block 的第一层后加激活和池化
                    pass
                features = x[0].cpu()  # (C, H, W)

                # 显示前 8 个通道
                n_show = min(8, features.shape[0])
                for col_idx in range(n_show):
                    ax = axes[row_idx, col_idx] if len(layers) > 1 else axes[col_idx]
                    im = ax.imshow(features[col_idx].numpy(), cmap='viridis', aspect='auto')
                    ax.axis('off')
                    if col_idx == 0:
                        ax.set_ylabel(name, fontsize=8, rotation=90, labelpad=15)
                    if row_idx == 0:
                        ax.set_title(f'Ch {col_idx+1}', fontsize=7)

        plt.suptitle('CNN Feature Maps Across Layers', fontsize=14, y=1.02)
        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'feature_maps.png')
            print(f"[INFO] Feature maps saved to {self.output_dir}/feature_maps.png")
        plt.close()

    def plot_training_history(self, save: bool = True):
        """绘制训练曲线"""
        if self.train_history is None:
            print("[WARNING] No training history provided, skipping.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        axes[0].plot(self.train_history['train_loss'], label='Train Loss', lw=1.5)
        axes[0].plot(self.train_history['val_loss'], label='Val Loss', lw=1.5)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(self.train_history['train_acc'], label='Train Acc', lw=1.5)
        axes[1].plot(self.train_history['val_acc'], label='Val Acc', lw=1.5)
        axes[1].axhline(y=self.train_history.get('best_val_acc', 0),
                        color='r', linestyle='--', alpha=0.5,
                        label=f"Best Val ({self.train_history.get('best_val_acc', 0):.1f}%)")
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'training_history.png')
            print(f"[INFO] Training curves saved to {self.output_dir}/training_history.png")
        plt.close()

    def noise_robustness_test(self, snr_levels: List[float] = None, save: bool = True):
        """
        噪声鲁棒性测试: 在测试集上叠加不同 SNR 的噪声, 绘制 SNR-Accuracy 曲线

        信号与系统关联: 这是 SNR (信噪比) 概念的直接应用。
        SNR(dB) = 10·log₁₀(P_signal / P_noise)
        """
        if snr_levels is None:
            snr_levels = [float('inf'), 20, 15, 10, 5, 0, -5, -10]

        print(f"\n{'='*60}")
        print("NOISE ROBUSTNESS TEST (SNR-Accuracy Curve)")
        print(f"{'='*60}")

        accuracies = []
        snr_labels = []

        from preprocess import add_noise_for_snr_test, load_and_preprocess_audio

        self.model.eval()

        for snr in snr_levels:
            correct = 0
            total = 0

            for inputs, targets in tqdm(self.test_loader, desc=f'  SNR={snr}dB'):
                inputs_np = inputs.squeeze(1).numpy()  # (B, H, W)

                # 对每个样本加噪
                noisy_inputs = np.zeros_like(inputs_np)
                for i in range(len(inputs_np)):
                    if snr == float('inf'):
                        noisy_inputs[i] = inputs_np[i]
                    else:
                        sig_power = np.mean(inputs_np[i] ** 2)
                        snr_linear = 10 ** (snr / 10)
                        noise_power = sig_power / (snr_linear + 1e-10)
                        noise = np.sqrt(noise_power) * np.random.randn(*inputs_np[i].shape)
                        noisy_inputs[i] = inputs_np[i] + noise.astype(np.float32)

                noisy_tensor = torch.from_numpy(noisy_inputs).unsqueeze(1).to(self.device)
                targets = targets.to(self.device)

                with torch.no_grad():
                    outputs = self.model(noisy_tensor)
                    _, preds = outputs.max(1)
                    correct += preds.eq(targets).sum().item()
                    total += targets.size(0)

            acc = 100.0 * correct / total
            accuracies.append(acc)

            label = 'Clean' if snr == float('inf') else f'{snr}dB'
            snr_labels.append(label)
            print(f"  SNR {label:>6}: {acc:.2f}%")

        # 绘图
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = range(len(snr_levels))
        bars = ax.bar(x_pos, accuracies, color=plt.cm.RdYlGn(
            np.array(accuracies) / 100), edgecolor='black', linewidth=0.5)

        # 数值标注
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(snr_labels)
        ax.set_xlabel('Signal-to-Noise Ratio (SNR)')
        ax.set_ylabel('Classification Accuracy (%)')
        ax.set_title('Noise Robustness: SNR vs Accuracy\n'
                     '(信号与系统 — SNR 对语音识别性能的影响)')
        ax.set_ylim(0, max(accuracies) * 1.15)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'noise_robustness.png')
            print(f"[INFO] Noise robustness plot saved to {self.output_dir}/noise_robustness.png")
        plt.close()

        # 保存数据
        np.savez(self.output_dir / 'noise_robustness.npz',
                 snr_levels=np.array([s if s != float('inf') else 999 for s in snr_levels]),
                 accuracies=np.array(accuracies),
                 snr_labels=snr_labels)

        print(f"{'='*60}\n")
        return accuracies, snr_levels

    def error_analysis(self, top_k: int = 20, save: bool = True):
        """错误分析: 展示分类置信度最高的错误样本"""
        preds, targets, probs = self._get_predictions()

        errors = []
        for i in range(len(preds)):
            if preds[i] != targets[i]:
                confidence = probs[i, preds[i]]
                errors.append({
                    'index': i,
                    'true_label': int(targets[i]),
                    'pred_label': int(preds[i]),
                    'true_word': self.word_list[targets[i]],
                    'pred_word': self.word_list[preds[i]],
                    'confidence': float(confidence),
                })

        # 按置信度降序 (最"自信"的错误)
        errors.sort(key=lambda x: x['confidence'], reverse=True)
        top_errors = errors[:top_k]

        print(f"\n{'='*60}")
        print(f"ERROR ANALYSIS — Top {top_k} High-Confidence Mistakes")
        print(f"{'='*60}")
        print(f"{'True':<12} {'Predicted':<12} {'Confidence':>10}")
        print("-" * 40)
        for e in top_errors:
            print(f"{e['true_word']:<12} {e['pred_word']:<12} {e['confidence']*100:>9.1f}%")
        print(f"{'='*60}\n")

        if save:
            with open(self.output_dir / 'error_analysis.txt', 'w') as f:
                f.write("Error Analysis — High-Confidence Mistakes\n")
                f.write("=" * 60 + "\n")
                for e in top_errors:
                    f.write(f"{e['true_word']:<12} → {e['pred_word']:<12} "
                            f"(confidence: {e['confidence']*100:.1f}%)\n")

    def run_all(self, snr_levels: List[float] = None):
        """运行所有评估项目"""
        print(f"\n{'#'*60}")
        print(f"# COMPREHENSIVE MODEL EVALUATION")
        print(f"{'#'*60}")

        # 1. 基础指标
        self.compute_metrics()

        # 2. 混淆矩阵
        self.plot_confusion_matrix(normalize=True)
        self.plot_confusion_matrix(normalize=False)

        # 3. ROC 曲线
        self.plot_roc_curves()

        # 4. 训练历史
        if self.train_history:
            self.plot_training_history()

        # 5. t-SNE 特征可视化
        self.plot_tsne_features()

        # 6. 特征图可视化
        sample_input, _ = self.test_loader.dataset[0]
        self.plot_feature_maps(sample_input)

        # 7. 噪声鲁棒性测试
        self.noise_robustness_test(snr_levels)

        # 8. 错误分析
        self.error_analysis()

        print(f"\n[SUCCESS] All evaluation results saved to {self.output_dir}/")


# ============================================================
# 命令行入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Speech CNN Model')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint (.pth) or "latest" for auto-detect')
    parser.add_argument('--list_models', action='store_true',
                        help='List all available trained models and exit')
    parser.add_argument('--model_type', type=str, default='standard',
                        choices=['standard', 'light', 'deep'])
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Test data directory (auto-detect from model if not set)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (auto: results/v1_english_20 or results/v2_bilingual_40)')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--input_shape', type=str, default='128,100')
    parser.add_argument('--all', action='store_true',
                        help='Run all evaluation tasks')
    parser.add_argument('--snr_levels', type=float, nargs='+',
                        default=[float('inf'), 20, 15, 10, 5, 0, -5],
                        help='SNR levels for noise robustness test')
    parser.add_argument('--task', type=str, default='all',
                        choices=['all', 'metrics', 'confusion', 'roc', 'tsne',
                                 'history', 'noise', 'error', 'feature_maps'])

    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(device_str)


def list_all_models():
    """列出所有已训练模型"""
    models_root = Path(__file__).parent.parent / 'models'
    if not models_root.exists():
        print("No models directory found.")
        return

    all_models = sorted(models_root.glob('*/*/best_model.pth'), reverse=True)
    if not all_models:
        print("No trained models found.")
        return

    print(f"\n{'='*70}")
    print(f"  AVAILABLE MODELS")
    print(f"{'='*70}")
    for i, p in enumerate(all_models):
        version = p.parent.parent.name
        run = p.parent.name
        # 读 final_results.txt
        fr = p.parent / 'final_results.txt'
        if fr.exists():
            info = {}
            for line in open(fr).readlines():
                if ':' in line:
                    k, v = line.split(':', 1)
                    info[k.strip()] = v.strip()
            acc = info.get('Test Accuracy', '?')
            val = info.get('Best Validation Accuracy', '?')
        else:
            acc, val = '?', '?'
        marker = ' <-- LATEST' if i == 0 else ''
        print(f"  [{i}] {version}/{run}")
        print(f"      Test Acc: {acc}  |  Val Acc: {val}{marker}")
    print(f"{'='*70}\n")
    print("Usage: python evaluate.py --model_path <path> --all")
    print("       python evaluate.py --model_path latest --all  (auto-use latest)\n")


def main():
    args = parse_args()

    if args.list_models:
        list_all_models()
        return

    device = get_device(args.device)
    print(f"[INFO] Using device: {device}")

    # 自动选择模型
    if args.model_path is None or args.model_path == 'latest':
        models_root = Path(__file__).parent.parent / 'models'
        all_models = sorted(models_root.glob('*/*/best_model.pth'), reverse=True)
        if not all_models:
            print("[ERROR] No models found. Train first.")
            sys.exit(1)
        checkpoint_path = all_models[0]
        print(f"[INFO] Auto-selected latest model: {checkpoint_path}")
    else:
        checkpoint_path = Path(args.model_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    # 解析输入形状
    h, w = map(int, args.input_shape.split(','))
    input_shape = (h, w)

    # 从 config.json 自动推断 num_classes 和训练数据目录
    config_path = checkpoint_path.parent / 'config.json'
    model_config = {}
    if config_path.exists():
        with open(config_path, encoding='utf-8') as f:
            model_config = json.load(f)
            actual_nc = model_config.get('NUM_CLASSES', args.num_classes)
    else:
        actual_nc = args.num_classes

    # 创建模型
    model = create_model(
        model_type=args.model_type,
        num_classes=actual_nc,
        input_shape=input_shape,
    )
    print(f"[INFO] Model created with num_classes={actual_nc}")

    if checkpoint_path.suffix == '.pth':
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            train_history = {
                'train_loss': checkpoint.get('train_history', {}).get('loss', []),
                'train_acc': checkpoint.get('train_history', {}).get('acc', []),
                'val_loss': checkpoint.get('val_history', {}).get('loss', []),
                'val_acc': checkpoint.get('val_history', {}).get('acc', []),
                'best_val_acc': checkpoint.get('best_val_acc', 0),
            }
        else:
            model.load_state_dict(checkpoint)
            train_history = None
        print(f"[INFO] Model loaded from {args.model_path}")
    else:
        # 纯权重文件
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        train_history = None
        print(f"[INFO] Weights loaded from {args.model_path}")

    # -- 数据集匹配：优先使用模型 config.json 中记录的 DATA_DIR --
    run_name = checkpoint_path.parent.name  # 从模型路径取 run_name
    if actual_nc >= 40:
        fallback_data_dir = '../data/processed_mixed'
        default_output_dir = f'../results/v2_bilingual_40/{run_name}'
    else:
        fallback_data_dir = '../data/processed'
        default_output_dir = f'../results/v1_english_20/{run_name}'

    config_data_dir = model_config.get('DATA_DIR') or fallback_data_dir
    auto_data_dir = str(resolve_path_from_script(config_data_dir))

    # 用户显式指定的优先级更高, 但会警告
    if args.data_dir:
        data_dir = str(resolve_path_from_script(args.data_dir))
        if data_dir != auto_data_dir:
            print(f"[WARN] Manual --data_dir={data_dir} overrides model config ({auto_data_dir})")
    else:
        data_dir = auto_data_dir

    output_dir = args.output_dir or default_output_dir
    print(f"[INFO] Data dir: {data_dir}  |  Output dir: {output_dir}")
    print(f"[INFO] Data source: {'manual --data_dir' if args.data_dir else 'model config DATA_DIR'}")

    # 创建 DataLoader
    test_dataset = SpeechCommandDataset(
        data_dir=data_dir,
        split='test',
        augment=False,
    )

    # 验证: 数据集标签数是否匹配模型类别数
    actual_labels = len(np.unique(test_dataset.y))
    if actual_labels != actual_nc:
        print(f"[ERROR] Dataset mismatch! Model expects {actual_nc} classes, "
              f"but dataset has {actual_labels} unique labels.")
        print(f"[FIX]  Use --data_dir {auto_data_dir}")
        sys.exit(1)
    print(f"[INFO] Dataset verified: {actual_labels} classes match model")
    if actual_nc >= 40:
        n_en = int(np.sum(test_dataset.y < 20))
        n_zh = int(np.sum(test_dataset.y >= 20))
        print(f"[INFO] Test split: {n_en} EN + {n_zh} ZH samples")

    import platform
    nw = 0 if platform.system() == 'Windows' else 2
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
    )

    # 创建评估器
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        train_history=train_history,
        metadata={
            'model_path': str(checkpoint_path.resolve()),
            'config_path': str(config_path.resolve()) if config_path.exists() else None,
            'config_data_dir': config_data_dir,
            'evaluated_data_dir': data_dir,
            'data_dir_source': 'manual --data_dir' if args.data_dir else 'model config DATA_DIR',
            'num_classes': int(actual_nc),
            'run_name': run_name,
            'use_class_weights': bool(model_config.get('USE_CLASS_WEIGHTS', False)),
            'over_sample': bool(model_config.get('OVER_SAMPLE', False)),
        },
    )

    # 执行评估
    task = args.task
    if task == 'all' or args.all:
        evaluator.run_all(args.snr_levels)
    elif task == 'metrics':
        evaluator.compute_metrics()
    elif task == 'confusion':
        evaluator.plot_confusion_matrix(normalize=True)
        evaluator.plot_confusion_matrix(normalize=False)
    elif task == 'roc':
        evaluator.plot_roc_curves()
    elif task == 'tsne':
        evaluator.plot_tsne_features()
    elif task == 'history':
        evaluator.plot_training_history()
    elif task == 'noise':
        evaluator.noise_robustness_test(args.snr_levels)
    elif task == 'error':
        evaluator.error_analysis()
    elif task == 'feature_maps':
        sample_input, _ = test_dataset[0]
        evaluator.plot_feature_maps(sample_input)


if __name__ == '__main__':
    main()
