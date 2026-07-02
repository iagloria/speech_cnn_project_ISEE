#!/usr/bin/env python3
"""
一键运行脚本 — 端到端语音命令识别全流程
==========================================

功能: 下载数据 → 预处理 → 训练 → 评估

用法:
  python run.py                     # 全流程 (下载+预处理+训练+评估)
  python run.py --skip_download     # 跳过下载 (已有数据)
  python run.py --skip_preprocess   # 跳过预处理 (已有 .npy)
  python run.py --train_only        # 仅训练
  python run.py --eval_only         # 仅评估 (需已有模型)

示例 (快速试用 10 类数字):
  python run.py --words zero one two three four five six seven eight nine --run_name digits10

示例 (中英双语 40 类):
  python run.py --bilingual --tts_dir data/raw/tts_chinese --run_name bilingual_v1

Author: Speech CNN Project
Date: 2026-06
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / 'src'
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw' / 'gsc'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'          # V1
PROCESSED_MIXED_DIR = DATA_DIR / 'processed_mixed'   # V2
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_V1_DIR = MODELS_DIR / 'v1_english_20'
MODELS_V2_DIR = MODELS_DIR / 'v2_bilingual_40'
RESULTS_DIR = PROJECT_ROOT / 'results'
RESULTS_V1_DIR = RESULTS_DIR / 'v1_english_20'
RESULTS_V2_DIR = RESULTS_DIR / 'v2_bilingual_40'


def run_cmd(cmd: str, desc: str = None):
    """运行命令并检查退出码"""
    if desc:
        print(f"\n{'='*60}")
        print(f">>> {desc}")
        print(f"{'='*60}")
    print(f"[CMD] {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=str(SRC_DIR))
    if result.returncode != 0:
        print(f"[ERROR] Command failed (exit code {result.returncode}): {cmd}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description='Speech CNN — End-to-End Voice Command Recognition'
    )
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip dataset download')
    parser.add_argument('--skip_preprocess', action='store_true',
                        help='Skip preprocessing')
    parser.add_argument('--train_only', action='store_true',
                        help='Only train (skip eval)')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only evaluate (need existing model)')

    # 可覆盖的参数
    parser.add_argument('--words', type=str, nargs='+', default=None,
                        help='Words to include (default: 20 core words)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model', type=str, default='standard',
                        choices=['standard', 'light', 'deep'])
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--bilingual', action='store_true',
                        help='Use bilingual 40-class mode (EN + ZH)')
    parser.add_argument('--tts_dir', type=str, default='../data/raw/tts_chinese',
                        help='TTS Chinese data directory (for bilingual mode)')

    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"# Speech CNN Project — End-to-End Pipeline")
    print(f"{'#'*60}")

    # --- Step 0: 环境检查 ---
    print(f"\n[Step 0] Environment check...")
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("[ERROR] PyTorch not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    try:
        import librosa
        print(f"  librosa: {librosa.__version__}")
    except ImportError:
        print("[ERROR] librosa not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    # 如果仅评估
    if args.eval_only:
        # 找最新的 best_model.pth (V2 优先)
        checkpoints = sorted(MODELS_DIR.glob('**/best_model.pth'))
        if not checkpoints:
            print(f"[ERROR] No model found in {MODELS_DIR}")
            sys.exit(1)
        latest = str(checkpoints[-1])
        print(f"[INFO] Using model: {latest}")
        # evaluate.py 会自动推断 data_dir 和 output_dir
        run_cmd(
            f'python evaluate.py --model_path "{latest}" --all',
            'Model Evaluation'
        )
        return

    # --- Step 1: 下载数据 ---
    if not args.skip_download:
        words_arg = ' '.join(args.words) if args.words else ''
        run_cmd(
            f'python preprocess.py --data_dir "{RAW_DATA_DIR}" '
            f'--output_dir "{PROCESSED_DATA_DIR}" '
            f'--download {words_arg}',
            'Step 1: Downloading Google Speech Commands v2 Dataset'
        )
    else:
        print("[Step 1] Skipping download.")

    # --- Step 2: 预处理 ---
    if not args.skip_preprocess:
        if args.bilingual:
            run_cmd(
                f'python preprocess.py --mode mixed '
                f'--data_dir "{RAW_DATA_DIR}" '
                f'--tts_dir "{args.tts_dir}" '
                f'--output_dir "{PROCESSED_MIXED_DIR}"',
                'Step 2: Preprocessing Mixed EN+ZH → Mel Spectrograms'
            )
        else:
            words_arg = ' '.join(args.words) if args.words else ''
            run_cmd(
                f'python preprocess.py --data_dir "{RAW_DATA_DIR}" '
                f'--output_dir "{PROCESSED_DATA_DIR}" '
                f'{words_arg}',
                'Step 2: Preprocessing Audio → Mel Spectrograms'
            )
    else:
        print("[Step 2] Skipping preprocessing.")

    # --- Step 3: 训练 ---
    run_name = args.run_name or 'speech_cnn'
    if args.bilingual:
        data_dir = str(PROCESSED_MIXED_DIR)
        output_dir = str(MODELS_V2_DIR)
        num_classes = 40
    else:
        data_dir = str(PROCESSED_DATA_DIR)
        output_dir = str(MODELS_V1_DIR)
        num_classes = 20 if not args.words else len(args.words)

    train_cmd = (
        f'python train.py '
        f'--data_dir "{data_dir}" '
        f'--epochs {args.epochs} '
        f'--batch_size {args.batch_size} '
        f'--lr {args.lr} '
        f'--model {args.model} '
        f'--output_dir "{output_dir}" '
        f'--num_classes {num_classes} '
        f'--run_name {run_name} '
    )
    run_cmd(train_cmd, 'Step 3: Training CNN Model')

    if args.train_only:
        print("\n[SUCCESS] Training complete! Run evaluate.py for testing.")
        return

    # --- Step 4: 评估 ---
    model_path = output_dir / run_name / 'best_model.pth'
    eval_cmd = (
        f'python evaluate.py '
        f'--model_path "{model_path}" '
        f'--data_dir "{data_dir}" '
        f'--all '
    )
    if args.words:
        eval_cmd += f'--num_classes {len(args.words)} '

    run_cmd(eval_cmd, 'Step 4: Comprehensive Evaluation')

    print(f"\n{'#'*60}")
    print(f"# ALL DONE!")
    print(f"# Model: {model_path}")
    print(f"# Results: results/{'v2_bilingual_40' if args.bilingual else 'v1_english_20'}/")
    print(f"{'#'*60}\n")


if __name__ == '__main__':
    main()
