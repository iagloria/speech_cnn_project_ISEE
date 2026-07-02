"""
实时推理脚本 — 用自己录制的语音测试模型
=========================================

功能:
  1. 单文件推理: 对 .wav 文件进行分类
  2. 文件夹批量测试: 遍历文件夹所有 .wav
  3. 交互式麦克风录制 (可选)
  4. Top-K 预测输出 + 置信度
  5. 频谱图可视化对比

用法:
  # 单文件测试
  python inference.py --model ../models/xxx/best_model.pth --audio my_voice.wav

  # 文件夹批量测试
  python inference.py --model ../models/xxx/best_model.pth --folder my_recordings/

  # 交互式录制 (需要安装 sounddevice)
  python inference.py --model ../models/xxx/best_model.pth --record

  # Top-3 预测 + 可视化
  python inference.py --model ../models/xxx/best_model.pth --audio test.wav --top_k 3 --visualize

Author: Speech CNN Project
Date: 2026-06
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 导入本地模块
sys.path.insert(0, str(Path(__file__).parent))
from model import create_model
from preprocess import (
    load_and_preprocess_audio, compute_mel_spectrogram, normalize_spectrogram,
    GSC_20_WORDS, BILINGUAL_40_WORDS, Config
)


class VoiceRecognizer:
    """
    语音命令识别器 — 加载训练好的模型进行推理
    """

    def __init__(self,
                 model_path: str,
                 model_type: str = 'standard',
                 num_classes: int = 20,
                 word_list: List[str] = None,
                 device: str = 'auto'):
        """
        Args:
            model_path: 模型权重路径 (.pth)
            model_type: 模型类型
            num_classes: 类别数
            word_list: 词汇列表
            device: 计算设备
        """
        self.device = self._get_device(device)
        if word_list is not None:
            self.word_list = word_list
        elif num_classes == 40:
            self.word_list = BILINGUAL_40_WORDS
        else:
            self.word_list = GSC_20_WORDS
        self.num_classes = len(self.word_list)

        # 加载模型
        print(f"[INFO] Loading model from {model_path}...")
        self.model = create_model(
            model_type=model_type,
            num_classes=self.num_classes,
        )

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        print(f"[INFO] Model loaded. Device: {self.device}")
        print(f"[INFO] Vocabulary ({self.num_classes} words): {self.word_list}")

    @staticmethod
    def _get_device(device_str: str) -> torch.device:
        if device_str == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            return torch.device('cpu')
        return torch.device(device_str)

    def predict(self,
                audio_path: str,
                top_k: int = 3) -> List[Tuple[str, float]]:
        """
        对单个音频文件进行预测

        Args:
            audio_path: .wav 文件路径
            top_k: 返回前 K 个预测

        Returns:
            [(word, confidence), ...] 按置信度降序排列
        """
        # 预处理: wav → 梅尔频谱图
        spec = load_and_preprocess_audio(
            audio_path,
            target_sr=Config.SAMPLE_RATE,
            target_duration=Config.AUDIO_DURATION,
            n_mels=Config.N_MELS,
        )

        # 转为 tensor: (1, 1, H, W)
        spec_tensor = torch.from_numpy(spec).float().unsqueeze(0).unsqueeze(0)
        spec_tensor = spec_tensor.to(self.device)

        # 推理
        with torch.no_grad():
            output = self.model(spec_tensor)
            probs = F.softmax(output, dim=1).squeeze(0)

        # Top-K
        topk_probs, topk_indices = torch.topk(probs, k=min(top_k, self.num_classes))

        results = []
        for idx, prob in zip(topk_indices.cpu().numpy(), topk_probs.cpu().numpy()):
            results.append((self.word_list[idx], float(prob)))

        return results

    def predict_batch(self,
                      folder_path: str,
                      top_k: int = 3) -> List[dict]:
        """
        批量测试文件夹中的所有 .wav 文件

        Args:
            folder_path: 包含 .wav 文件的文件夹
            top_k: 返回前 K 个预测

        Returns:
            [{filepath, predictions, label (if filename starts with word)}, ...]
        """
        folder = Path(folder_path)
        wav_files = sorted(folder.glob("*.wav")) + sorted(folder.glob("*.WAV"))

        if not wav_files:
            print(f"[WARNING] No .wav files found in {folder_path}")
            return []

        print(f"[INFO] Found {len(wav_files)} audio files")
        results = []

        for i, wav_path in enumerate(wav_files):
            try:
                predictions = self.predict(str(wav_path), top_k=top_k)
                result = {
                    'index': i,
                    'filepath': str(wav_path),
                    'filename': wav_path.name,
                    'top_prediction': predictions[0][0],
                    'top_confidence': predictions[0][1],
                    'predictions': predictions,
                }
                results.append(result)

                status = '✓' if predictions[0][1] > 0.6 else '?'
                print(f"  [{i+1:3d}/{len(wav_files)}] {status} {wav_path.name:<40s} "
                      f"→ {predictions[0][0]:<8s} ({predictions[0][1]*100:.1f}%)")
            except Exception as e:
                print(f"  [{i+1:3d}/{len(wav_files)}] ✗ {wav_path.name} — Error: {e}")
                results.append({
                    'index': i,
                    'filepath': str(wav_path),
                    'filename': wav_path.name,
                    'top_prediction': 'ERROR',
                    'top_confidence': 0.0,
                    'predictions': [],
                    'error': str(e),
                })

        return results

    def predict_with_visualization(self,
                                    audio_path: str,
                                    top_k: int = 5,
                                    save_path: str = None):
        """
        预测 + 生成可视化对比图 (波形 + 频谱 + Top-K 预测)

        Args:
            audio_path: .wav 文件路径
            top_k: 显示前 K 个预测
            save_path: 图片保存路径 (None=自动命名)
        """
        import librosa

        # 加载音频
        audio, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE, mono=True)
        audio = audio[:int(Config.SAMPLE_RATE * Config.AUDIO_DURATION)]

        # 计算梅尔频谱图
        mel_spec = compute_mel_spectrogram(audio, sr=sr, n_mels=Config.N_MELS)
        mel_spec_norm = normalize_spectrogram(mel_spec, method='global')

        # 推理
        predictions = self.predict(audio_path, top_k=top_k)

        # 绘图
        fig, axes = plt.subplots(2, 2, figsize=(14, 8),
                                  gridspec_kw={'width_ratios': [3, 1]})

        # 波形
        t = np.arange(len(audio)) / sr
        axes[0, 0].plot(t, audio, linewidth=0.5, color='steelblue')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_title('Waveform')
        axes[0, 0].set_xlim(0, len(audio)/sr)

        # 频谱图
        im = axes[1, 0].imshow(mel_spec_norm, aspect='auto', origin='lower',
                                cmap='magma')
        axes[1, 0].set_xlabel('Time Frames')
        axes[1, 0].set_ylabel('Mel Frequency Bands')
        axes[1, 0].set_title('Mel Spectrogram')
        plt.colorbar(im, ax=axes[1, 0])

        # Top-K 预测柱状图
        words = [p[0] for p in predictions[::-1]]
        confs = [p[1] * 100 for p in predictions[::-1]]
        colors = plt.cm.RdYlGn(np.array(confs) / 100)
        axes[0, 1].barh(words, confs, color=colors, edgecolor='black', linewidth=0.5)
        axes[0, 1].set_xlabel('Confidence (%)')
        axes[0, 1].set_title(f'Top-{top_k} Predictions')
        axes[0, 1].set_xlim(0, 105)
        for bar, conf in zip(axes[0, 1].patches, confs):
            axes[0, 1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                           f'{conf:.1f}%', va='center', fontsize=9)

        # 信息面板
        axes[1, 1].axis('off')
        info_text = f"FILE: {Path(audio_path).name}\n\n"
        info_text += f"TOP PREDICTION:\n"
        info_text += f"  {predictions[0][0]}\n"
        info_text += f"  {predictions[0][1]*100:.1f}%\n\n"
        info_text += f"ALL PREDICTIONS:\n"
        for word, conf in predictions:
            bar = '█' * int(conf * 30)
            info_text += f"  {word:<8s} {bar} {conf*100:.1f}%\n"
        axes[1, 1].text(0.05, 0.95, info_text, transform=axes[1, 1].transAxes,
                        fontfamily='monospace', fontsize=9,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(f'Voice Recognition Result', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path is None:
            save_path = str(Path(audio_path).with_suffix('.result.png'))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Visualization saved to {save_path}")

    def print_help_recording(self):
        """打印录音帮助信息"""
        print(f"""
{'='*60}
  HOW TO RECORD YOUR VOICE FOR TESTING
{'='*60}

  Option A: Use any voice recorder app on your phone/PC
    1. Record a single word (e.g., "yes", "stop", "zero")
    2. Save as .wav (16kHz mono recommended)
    3. Run: python inference.py --model <model> --audio your_file.wav

  Option B: Use Python (install sounddevice first)
    pip install sounddevice
    python inference.py --model <model> --record

  Option C: Record multiple words to a folder
    mkdir my_voice
    # Record files named like: yes.wav, no.wav, zero.wav, ...
    # Run batch test:
    python inference.py --model <model> --folder my_voice/

  Recording tips:
    - Speak clearly, one word per file
    - Keep silence before and after the word (~0.2s each)
    - Record in a quiet environment
    - Sample rate 16kHz is ideal (phone default is usually fine)
    - Duration: 0.5-1.5 seconds
{'='*60}
""")


def interactive_record(model_path: str, model_type: str = 'standard',
                        device: str = 'auto'):
    """交互式麦克风录制并识别"""
    try:
        import sounddevice as sd
    except ImportError:
        print("[ERROR] sounddevice not installed. Run: pip install sounddevice")
        print("[INFO] You can still use --audio or --folder instead of --record")
        return

    recognizer = VoiceRecognizer(model_path, model_type=model_type, device=device)
    sr = Config.SAMPLE_RATE
    duration = 1.5  # 录制 1.5 秒

    print(f"\n{'='*60}")
    print(f"  INTERACTIVE VOICE RECOGNITION")
    print(f"  Sample rate: {sr} Hz, Duration: {duration}s")
    print(f"  Vocabulary: {', '.join(recognizer.word_list)}")
    print(f"{'='*60}")
    print("\n  Press Enter to start recording, then say ONE word.")
    print("  Type 'q' to quit.\n")

    while True:
        cmd = input("  [Press Enter to record, 'q' to quit] ").strip()
        if cmd.lower() == 'q':
            break

        print(f"  🎤 Recording {duration}s... ", end='', flush=True)
        try:
            audio = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype='float32')
            sd.wait()
            audio = audio.flatten()
            print("Done!")

            # 预处理
            spec = load_and_preprocess_audio.__wrapped__ if hasattr(
                load_and_preprocess_audio, '__wrapped__') else None

            # 直接处理
            audio = audio[:int(sr * Config.AUDIO_DURATION)]
            if len(audio) < int(sr * Config.AUDIO_DURATION):
                audio = np.pad(audio, (0, int(sr * Config.AUDIO_DURATION) - len(audio)))

            mel_spec = compute_mel_spectrogram(audio, sr=sr, n_mels=Config.N_MELS)
            mel_spec = normalize_spectrogram(mel_spec, method='global')
            spec_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0)
            spec_tensor = spec_tensor.to(recognizer.device)

            # 推理
            with torch.no_grad():
                output = recognizer.model(spec_tensor)
                probs = F.softmax(output, dim=1).squeeze(0)

            topk_probs, topk_indices = torch.topk(probs, k=3)
            print(f"\n  Results:")
            for i, (idx, prob) in enumerate(zip(topk_indices, topk_probs)):
                word = recognizer.word_list[idx]
                bar = '█' * int(prob * 40)
                print(f"    {'►' if i == 0 else ' '} {word:<8s} {bar} {prob*100:.1f}%")

            if topk_probs[0] > 0.5:
                print(f"  ✓ Detected: {recognizer.word_list[topk_indices[0]]}\n")
            else:
                print(f"  ⚠ Low confidence — maybe try again?\n")

        except Exception as e:
            print(f"\n  [ERROR] Recording failed: {e}")

    print("  Goodbye!\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Voice Command Recognition — Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --model ../models/xxx/best_model.pth --audio hello.wav
  python inference.py --model ../models/xxx/best_model.pth --folder my_voice/
  python inference.py --model ../models/xxx/best_model.pth --audio test.wav --top_k 5 --visualize
  python inference.py --model ../models/xxx/best_model.pth --record
        """
    )

    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='standard',
                        choices=['standard', 'light', 'deep'])
    parser.add_argument('--audio', type=str, default=None,
                        help='Single .wav file to test')
    parser.add_argument('--folder', type=str, default=None,
                        help='Folder of .wav files to test')
    parser.add_argument('--record', action='store_true',
                        help='Interactive microphone recording')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Show top-K predictions')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization for single file')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--help_recording', action='store_true',
                        help='Print recording help')

    return parser.parse_args()


def main():
    args = parser.parse_args()

    # 创建识别器
    recognizer = VoiceRecognizer(
        model_path=args.model,
        model_type=args.model_type,
        num_classes=args.num_classes,
        device=args.device,
    )

    # 录音帮助
    if args.help_recording:
        recognizer.print_help_recording()
        return

    # 交互式录音
    if args.record:
        interactive_record(args.model, args.model_type, args.device)
        return

    # 单文件测试
    if args.audio:
        if not Path(args.audio).exists():
            print(f"[ERROR] File not found: {args.audio}")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"  SINGLE FILE INFERENCE")
        print(f"{'='*60}")
        print(f"  File: {args.audio}\n")

        start = time.time()
        predictions = recognizer.predict(args.audio, top_k=args.top_k)
        elapsed = (time.time() - start) * 1000

        for i, (word, conf) in enumerate(predictions):
            bar = '▓' * int(conf * 50) + '░' * (50 - int(conf * 50))
            marker = '▶' if i == 0 else ' '
            print(f"  {marker} {word:<10s} {bar} {conf*100:5.1f}%")

        print(f"\n  Inference time: {elapsed:.1f} ms")
        print(f"{'='*60}\n")

        if args.visualize:
            recognizer.predict_with_visualization(args.audio, top_k=args.top_k)

        return

    # 文件夹批量测试
    if args.folder:
        if not Path(args.folder).exists():
            print(f"[ERROR] Folder not found: {args.folder}")
            sys.exit(1)

        results = recognizer.predict_batch(args.folder, top_k=args.top_k)

        # 汇总
        if results:
            n = len(results)
            high_conf = sum(1 for r in results if r['top_confidence'] > 0.6)
            print(f"\n{'='*60}")
            print(f"  BATCH SUMMARY: {high_conf}/{n} high-confidence ({100*high_conf/n:.0f}%)")
            print(f"{'='*60}\n")
        return

    # 没指定操作
    print("[INFO] No action specified. Use --audio, --folder, --record, or --help_recording")
    recognizer.print_help_recording()


if __name__ == '__main__':
    main()
