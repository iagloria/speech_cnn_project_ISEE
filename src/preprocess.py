"""
音频预处理模块 — 完整的信号处理流水线
=============================================
将原始 .wav 语音文件转换为梅尔频谱图 (Mel Spectrogram),
作为 2D CNN 的输入"图像"。

信号与系统核心概念映射:
  预加重   → 一阶高通 FIR 滤波器 H(z)=1-αz⁻¹
  分帧加窗 → 短时傅里叶变换 (STFT) 的预处理
  STFT    → 离散傅里叶变换 (DFT) 的滑动窗口应用
  Mel滤波器组 → 非均匀三角滤波器组，模拟人耳听觉特性
  对数压缩 → dB 尺度转换，模仿人耳对数感知

Author: Speech CNN Project
Date: 2026-06
"""

import os
import sys
import shutil
import warnings
import urllib.request
import tarfile
import hashlib
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import librosa
import librosa.display
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# 全局配置参数
# ============================================================

class Config:
    """音频处理和特征提取的全局配置"""
    # 目标采样率 (Hz) — 满足 Nyquist 定理, 覆盖语音主要频段 0-4kHz
    SAMPLE_RATE: int = 16000

    # 音频固定长度 (秒) — 所有样本统一到此长度
    AUDIO_DURATION: float = 1.0

    # 预加重系数 — 补偿高频衰减, H(z) = 1 - α·z⁻¹
    PRE_EMPHASIS: float = 0.97

    # STFT 参数
    FRAME_LENGTH_MS: float = 25.0      # 帧长 25ms — 语音短时平稳假设
    FRAME_SHIFT_MS: float = 10.0       # 帧移 10ms — 帧间 60% 重叠
    N_FFT: int = 512                    # FFT 点数 (补零到 2 的幂)

    # Mel 滤波器组参数
    N_MELS: int = 128                   # Mel 滤波器数量 (频率分辨率)
    F_MIN: float = 20.0                 # 最低频率 (Hz)
    F_MAX: float = 8000.0               # 最高频率 (Hz, = Nyquist freq)

    # 数据增强开关
    USE_SPECAUGMENT: bool = True

    # 随机种子
    SEED: int = 42


# ============================================================
# 核心信号处理函数
# ============================================================

def pre_emphasis(signal_in: np.ndarray, alpha: float = Config.PRE_EMPHASIS) -> np.ndarray:
    """
    预加重 — 一阶高通 FIR 滤波器

    数学表达: y[n] = x[n] - α·x[n-1]
    系统函数: H(z) = 1 - α·z⁻¹

    物理意义: 语音信号的高频分量在唇辐射过程中会衰减约 6dB/oct,
              预加重补偿高频, 使频谱更加平坦, 有利于后续分析。

    Args:
        signal_in: 输入信号, shape (N,)
        alpha: 预加重系数, 通常 0.9~1.0

    Returns:
        预加重后的信号
    """
    if alpha <= 0:
        return signal_in
    return np.append(signal_in[0], signal_in[1:] - alpha * signal_in[:-1])


def frame_signal(signal_in: np.ndarray,
                 frame_length: int,
                 frame_shift: int,
                 window: str = 'hamming') -> np.ndarray:
    """
    分帧加窗 — STFT 预处理

    数学表达: x̃_m[n] = x[n + m·H] · w[n],  n = 0,...,N-1

    其中 N=帧长, H=帧移, w[n] 为汉明窗:
      w[n] = 0.54 - 0.46·cos(2πn/(N-1))

    物理意义: 语音信号在 10-30ms 内可视作平稳, 通过滑动窗口
              将其分割为短时帧, 对每帧单独做频谱分析。

    Args:
        signal_in: 输入信号
        frame_length: 帧长 (采样点数)
        frame_shift: 帧移 (采样点数)
        window: 窗函数类型 ('hamming', 'hann', 'rectangular')

    Returns:
        分帧后的矩阵, shape (n_frames, frame_length)
    """
    sig_len = len(signal_in)
    if sig_len < frame_length:
        # 尾部补零
        padded = np.zeros(frame_length)
        padded[:sig_len] = signal_in
        signal_in = padded
        sig_len = frame_length

    n_frames = 1 + (sig_len - frame_length) // frame_shift
    frames = np.zeros((n_frames, frame_length), dtype=np.float32)

    if window == 'hamming':
        win = np.hamming(frame_length)
    elif window == 'hann':
        win = np.hanning(frame_length)
    else:
        win = np.ones(frame_length)

    for i in range(n_frames):
        start = i * frame_shift
        frames[i] = signal_in[start:start + frame_length] * win

    return frames


def compute_mel_spectrogram(signal_in: np.ndarray,
                             sr: int = Config.SAMPLE_RATE,
                             n_mels: int = Config.N_MELS,
                             n_fft: int = Config.N_FFT,
                             frame_length: int = None,
                             frame_shift: int = None,
                             f_min: float = Config.F_MIN,
                             f_max: float = Config.F_MAX,
                             return_db: bool = True) -> np.ndarray:
    """
    计算梅尔频谱图 — 信号处理核心流水线

    完整流程:
      信号 s(t) → 预加重 → 分帧加窗 → FFT → |·|² → Mel滤波器组 → log₁₀ → 梅尔频谱图

    理论要点:
      1. STFT:  X_m[k] = Σₙ x̃_m[n]·exp(-j2πkn/K),  K=N_FFT
      2. 功率谱: P_m[k] = |X_m[k]|² / K
      3. Mel域映射:
           mel(f) = 2595·log₁₀(1 + f/700)
           f(mel) = 700·(10^(mel/2595) - 1)
      4. 三角滤波器 H_b[k] 在 Mel 域等间距, Hz 域非均匀
      5. Mel 能量:  S_m[b] = Σₖ P_m[k]·H_b[k]
      6. dB 转换:  S_dB = 10·log₁₀(S + ε)

    Args:
        signal_in: 输入语音信号
        sr: 采样率
        n_mels: Mel 滤波器数量
        n_fft: FFT 点数
        frame_length: 帧长 (采样点), None则自动计算
        frame_shift: 帧移 (采样点), None则自动计算
        f_min: 最低频率
        f_max: 最高频率
        return_db: 是否返回 dB 尺度

    Returns:
        mel_spectrogram: shape (n_mels, n_frames)
    """
    if frame_length is None:
        frame_length = int(Config.FRAME_LENGTH_MS / 1000 * sr)
    if frame_shift is None:
        frame_shift = int(Config.FRAME_SHIFT_MS / 1000 * sr)

    # Step 1: 预加重
    sig = pre_emphasis(signal_in, Config.PRE_EMPHASIS)

    # Step 2: 分帧加窗
    frames = frame_signal(sig, frame_length, frame_shift, 'hamming')
    n_frames = frames.shape[0]

    # Step 3: FFT (对每帧)
    spec = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
    for i in range(n_frames):
        fft_result = np.fft.rfft(frames[i], n=n_fft)
        spec[:, i] = np.abs(fft_result) ** 2 / n_fft  # 功率谱

    # Step 4: 构建 Mel 滤波器组
    mel_filterbank = _create_mel_filterbank(n_mels, n_fft, sr, f_min, f_max)

    # Step 5: 应用 Mel 滤波器组
    mel_spec = np.dot(mel_filterbank, spec)  # (n_mels, n_frames)

    # Step 6: dB 转换
    if return_db:
        mel_spec = 10.0 * np.log10(mel_spec + 1e-10)

    return mel_spec.astype(np.float32)


def _create_mel_filterbank(n_mels: int,
                           n_fft: int,
                           sr: int,
                           f_min: float,
                           f_max: float) -> np.ndarray:
    """
    创建 Mel 三角滤波器组

    数学表达:
      在 Mel 域等间距放置 B 个中心频率, 映射回 Hz 域后构建三角滤波器:

                      ┌ 0,                         k < f_{b-1}
                      │ (k - f_{b-1})/(f_b - f_{b-1}),  f_{b-1} ≤ k ≤ f_b
        H_b[k] =    │ (f_{b+1} - k)/(f_{b+1} - f_b),  f_b < k ≤ f_{b+1}
                      └ 0,                         k > f_{b+1}

    物理意义: 模拟人耳基底膜对频率的非线性感知,
              低频区域滤波器密集 (高分辨), 高频区域滤波器稀疏 (低分辨)。

    Args:
        n_mels: Mel 滤波器个数
        n_fft: FFT 点数
        sr: 采样率
        f_min: 最低频率
        f_max: 最高频率

    Returns:
        filterbank: shape (n_mels, n_fft//2 + 1)
    """
    # Hz → Mel 映射
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    # Mel 域等间距点
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # FFT bin 对应的频率
    freq_bins = np.linspace(0, sr / 2, n_fft // 2 + 1)

    # 构建三角滤波器
    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for b in range(n_mels):
        left = hz_points[b]
        center = hz_points[b + 1]
        right = hz_points[b + 2]

        # 左半边: 从 0 线性增长到 1
        left_slope = (freq_bins - left) / (center - left + 1e-10)
        # 右半边: 从 1 线性减小到 0
        right_slope = (right - freq_bins) / (right - center + 1e-10)

        filterbank[b] = np.maximum(0, np.minimum(left_slope, right_slope))

    return filterbank


def normalize_spectrogram(spec: np.ndarray, method: str = 'global') -> np.ndarray:
    """
    频谱图归一化

    Args:
        spec: 输入频谱图 (n_mels, n_frames)
        method: 'global' (全局 z-score), 'sample' (逐样本 min-max), 'none'

    Returns:
        归一化后的频谱图
    """
    if method == 'global':
        # Z-score: (x - μ) / σ
        mean = np.mean(spec)
        std = np.std(spec) + 1e-8
        return (spec - mean) / std
    elif method == 'sample':
        # Min-Max: 映射到 [0, 1]
        s_min = np.min(spec)
        s_max = np.max(spec)
        if s_max - s_min < 1e-8:
            return np.zeros_like(spec)
        return (spec - s_min) / (s_max - s_min)
    else:
        return spec


def pad_or_truncate(signal_in: np.ndarray,
                    target_length: int) -> np.ndarray:
    """
    将信号截断或补零到固定长度

    Args:
        signal_in: 输入信号
        target_length: 目标长度 (采样点数)

    Returns:
        长度 = target_length 的信号
    """
    if len(signal_in) > target_length:
        return signal_in[:target_length]
    elif len(signal_in) < target_length:
        padded = np.zeros(target_length, dtype=signal_in.dtype)
        padded[:len(signal_in)] = signal_in
        return padded
    return signal_in


def load_and_preprocess_audio(filepath: str,
                               target_sr: int = Config.SAMPLE_RATE,
                               target_duration: float = Config.AUDIO_DURATION,
                               n_mels: int = Config.N_MELS) -> np.ndarray:
    """
    端到端音频处理: wav 文件 → 归一化梅尔频谱图

    这是对外的主要接口, 包含完整流水线:
      load → resample → pad/trunc → pre_emph → frame → STFT → Mel → dB → normalize

    Args:
        filepath: .wav 文件路径
        target_sr: 目标采样率
        target_duration: 目标时长 (秒)
        n_mels: Mel 频带数

    Returns:
        mel_spec: shape (n_mels, n_frames), 归一化后的梅尔频谱图
    """
    # 使用 librosa 加载 (处理各种格式/采样率)
    audio, sr = librosa.load(filepath, sr=target_sr, mono=True)

    # 固定长度
    target_len = int(target_sr * target_duration)
    audio = pad_or_truncate(audio, target_len)

    # 计算梅尔频谱图
    mel_spec = compute_mel_spectrogram(audio, sr=target_sr, n_mels=n_mels)

    # 归一化
    mel_spec = normalize_spectrogram(mel_spec, method='global')

    return mel_spec


# ============================================================
# Google Speech Commands 数据集下载与处理
# ============================================================

# GSC v2 数据集信息
GSC_V2_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
GSC_V2_EXPECTED_SIZE = 2_428_189_560  # ~2.3 GB
GSC_V2_EXPECTED_MD5 = None  # 不强制校验, 可自行添加

# 20 个核心词汇 (从 35 个中精选)
GSC_20_WORDS = [
    'yes', 'no', 'up', 'down', 'left', 'right',
    'on', 'off', 'stop', 'go',
    'zero', 'one', 'two', 'three', 'four',
    'five', 'six', 'seven', 'eight', 'nine'
]

# 中文词汇映射 (TTS 合成数据)
ZH_DIGITS = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
ZH_COMMANDS = ['是', '不', '上', '下', '左', '右', '开', '关', '停', '走']
ZH_ALL = ZH_DIGITS + ZH_COMMANDS  # 20 个中文词

# 中英双语 40 词: 英文 20 + 中文 20
BILINGUAL_40_WORDS = (
    GSC_20_WORDS +
    [f'zh_{w}' for w in GSC_20_WORDS]  # 中文用 zh_ 前缀区分
)

# 中英映射表: 英文词 → 中文词
EN_TO_ZH = {
    'yes': '是', 'no': '不', 'up': '上', 'down': '下',
    'left': '左', 'right': '右', 'on': '开', 'off': '关',
    'stop': '停', 'go': '走',
    'zero': '零', 'one': '一', 'two': '二', 'three': '三',
    'four': '四', 'five': '五', 'six': '六', 'seven': '七',
    'eight': '八', 'nine': '九',
}

# 背景噪声类 (用于信噪比测试, 可选)
BACKGROUND_NOISE = '_background_noise_'


def download_gsc_dataset(data_dir: str, force: bool = False) -> str:
    """
    下载并解压 Google Speech Commands v2 数据集

    Args:
        data_dir: 数据存储目录
        force: 是否强制重新下载

    Returns:
        dataset_path: 解压后的数据集路径
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    tar_path = data_path / "speech_commands_v0.02.tar.gz"
    extracted_path = data_path  # 解压到 data/raw/gsc/ 目录

    # 检查是否已解压
    if not force and (extracted_path / "yes").exists():
        print(f"[INFO] GSC dataset already exists at {extracted_path}")
        return str(extracted_path)

    # 下载
    if not tar_path.exists() or force:
        print(f"[INFO] Downloading Google Speech Commands v2 from {GSC_V2_URL}")
        print(f"[INFO] File size: ~2.3 GB, this may take a while...")

        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                sys.stdout.write(f"\r  Download progress: {percent:.1f}%")
                sys.stdout.flush()

        try:
            urllib.request.urlretrieve(GSC_V2_URL, tar_path, reporthook=report_progress)
            print("\n[INFO] Download complete.")
        except Exception as e:
            print(f"\n[ERROR] Download failed: {e}")
            print("[INFO] You can manually download from:")
            print("  http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz")
            print(f"  and place it at: {tar_path}")
            raise

    # 解压
    if not (extracted_path / "yes").exists() or force:
        print(f"[INFO] Extracting dataset to {extracted_path}...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=extracted_path, filter='data')
        print("[INFO] Extraction complete.")

    return str(extracted_path)


def scan_gsc_files(data_dir: str,
                    words: List[str] = GSC_20_WORDS) -> Tuple[List[str], List[int]]:
    """
    扫描 GSC 数据集, 收集指定词汇的文件路径和标签

    GSC 目录结构:
      data_dir/
        ├── yes/
        │   ├── 0a7c2a8d_nohash_0.wav
        │   └── ...
        ├── no/
        ├── ...
        └── _background_noise_/

    Args:
        data_dir: GSC 数据集根目录
        words: 要包含的词汇列表

    Returns:
        filepaths: wav 文件路径列表
        labels: 对应的标签 (0 ~ len(words)-1)
    """
    filepaths = []
    labels = []
    word_to_idx = {w: i for i, w in enumerate(words)}

    for word in words:
        word_dir = Path(data_dir) / word
        if not word_dir.exists():
            print(f"[WARNING] Directory not found: {word_dir}, skipping '{word}'")
            continue

        wav_files = sorted(word_dir.glob("*.wav"))
        for f in wav_files:
            filepaths.append(str(f))
            labels.append(word_to_idx[word])

        print(f"  [{word}]: {len(wav_files)} samples")

    return filepaths, labels


def preprocess_gsc_dataset(data_dir: str,
                            words: List[str] = GSC_20_WORDS,
                            output_dir: str = None,
                            test_size: float = 0.15,
                            val_size: float = 0.15) -> Dict[str, np.ndarray]:
    """
    完整数据集预处理: 扫描 → 预处理 → 划分 → 保存

    这是训练前一次性运行的数据准备脚本。

    Args:
        data_dir: GSC 数据集根目录
        words: 词汇列表
        output_dir: 输出目录 (保存 .npy 文件)
        test_size: 测试集比例
        val_size: 验证集比例

    Returns:
        data_dict: {'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test', 'word_list'}
    """
    if output_dir is None:
        output_dir = str(Path(data_dir).parent.parent / "processed")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 检查是否已处理
    train_file = output_path / "X_train.npy"
    if train_file.exists():
        print(f"[INFO] Preprocessed data found at {output_path}, skipping processing.")
        data = {}
        for name in ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test',
                      'filepaths_train', 'filepaths_val', 'filepaths_test']:
            fpath = output_path / f"{name}.npy"
            if fpath.exists():
                data[name] = fpath  # 存路径，按需加载
        data['word_list'] = output_path / "word_list.npy"
        return data

    print("[INFO] Scanning dataset files...")
    filepaths, labels = scan_gsc_files(data_dir, words)

    total = len(filepaths)
    print(f"[INFO] Total samples found: {total} ({len(words)} classes)")

    # 划分 train / val / test
    np.random.seed(Config.SEED)
    indices = np.random.permutation(total)
    n_test = int(total * test_size)
    n_val = int(total * val_size)
    n_train = total - n_test - n_val

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    splits = {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx,
    }

    data = {}  # 轻量 dict，仅存形状信息和标签（非大数组）
    n_expected_frames = compute_expected_frames(Config.SAMPLE_RATE, Config.AUDIO_DURATION,
                                                  Config.FRAME_LENGTH_MS, Config.FRAME_SHIFT_MS)
    print(f"[INFO] Mel spectrogram shape: ({Config.N_MELS}, {n_expected_frames})")

    # 分批处理大小（避免一次性分配过大数组导致 MemoryError）
    BATCH_SIZE = 5000
    temp_dir = output_path / "_temp"
    temp_dir.mkdir(exist_ok=True)

    for split_name, split_idx in splits.items():
        n_total = len(split_idx)
        print(f"\n[INFO] Processing {split_name} set ({n_total} samples)...")

        n_batches = (n_total + BATCH_SIZE - 1) // BATCH_SIZE
        temp_spec_files = []
        temp_label_files = []
        all_fps = []
        n_valid_total = 0

        for batch_i in range(n_batches):
            start = batch_i * BATCH_SIZE
            end = min(start + BATCH_SIZE, n_total)
            batch_indices = split_idx[start:end]
            batch_size = end - start

            specs_batch = np.zeros((batch_size, Config.N_MELS, n_expected_frames), dtype=np.float32)
            y_batch = np.zeros(batch_size, dtype=np.int64)
            fps_batch = []

            for i, idx in enumerate(tqdm(batch_indices,
                                         desc=f"  {split_name} batch {batch_i+1}/{n_batches}")):
                try:
                    spec = load_and_preprocess_audio(
                        filepaths[idx],
                        target_sr=Config.SAMPLE_RATE,
                        target_duration=Config.AUDIO_DURATION,
                        n_mels=Config.N_MELS
                    )
                    if spec.shape[1] < n_expected_frames:
                        pad_width = n_expected_frames - spec.shape[1]
                        spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
                    elif spec.shape[1] > n_expected_frames:
                        spec = spec[:, :n_expected_frames]

                    specs_batch[i] = spec
                    y_batch[i] = labels[idx]
                    fps_batch.append(filepaths[idx])
                except Exception as e:
                    print(f"\n  [WARNING] Failed to process {filepaths[idx]}: {e}")
                    specs_batch[i] = np.zeros((Config.N_MELS, n_expected_frames), dtype=np.float32)
                    y_batch[i] = -1

            # 过滤当前 batch 的有效样本
            valid_mask = y_batch >= 0
            specs_valid = specs_batch[valid_mask]
            y_valid = y_batch[valid_mask]
            fps_valid = [fp for j, fp in enumerate(fps_batch) if valid_mask[j]]

            if len(specs_valid) > 0:
                sf = temp_dir / f"X_{split_name}_batch_{batch_i}.npy"
                lf = temp_dir / f"y_{split_name}_batch_{batch_i}.npy"
                np.save(sf, specs_valid)
                np.save(lf, y_valid)
                temp_spec_files.append(sf)
                temp_label_files.append(lf)
                n_valid_total += len(specs_valid)

            all_fps.extend(fps_valid)
            del specs_batch, y_batch, specs_valid, y_valid, fps_batch

        print(f"  Valid samples: {n_valid_total}")

        # 合并 batch 并保存为最终 .npy（使用 memmap 控制峰值内存）
        final_spec_path = output_path / f"X_{split_name}.npy"
        final_label_path = output_path / f"y_{split_name}.npy"
        final_fp_path = output_path / f"filepaths_{split_name}.npy"

        if len(temp_spec_files) == 1:
            # 只有一个 batch，直接重命名
            shutil.move(str(temp_spec_files[0]), str(final_spec_path))
            shutil.move(str(temp_label_files[0]), str(final_label_path))
            data[f'X_{split_name}'] = final_spec_path  # 存路径，不是数组
            data[f'y_{split_name}'] = final_label_path
        else:
            print(f"  Merging {len(temp_spec_files)} batches...")
            specs_mmap = np.memmap(temp_dir / f"X_{split_name}_mmap.dat",
                                   dtype=np.float32, mode='w+',
                                   shape=(n_valid_total, Config.N_MELS, n_expected_frames))
            y_all = np.zeros(n_valid_total, dtype=np.int64)
            offset = 0
            for sf, lf in zip(temp_spec_files, temp_label_files):
                batch_specs = np.load(sf)
                batch_labels = np.load(lf)
                n = batch_specs.shape[0]
                specs_mmap[offset:offset+n] = batch_specs
                y_all[offset:offset+n] = batch_labels
                offset += n
                del batch_specs, batch_labels
            # 从 memmap 保存为 .npy，然后释放
            np.save(final_spec_path, specs_mmap)
            np.save(final_label_path, y_all)
            del specs_mmap, y_all
            (temp_dir / f"X_{split_name}_mmap.dat").unlink(missing_ok=True)
            data[f'X_{split_name}'] = final_spec_path
            data[f'y_{split_name}'] = final_label_path

        # 保存 filepaths
        fps_arr = np.array(all_fps)
        np.save(final_fp_path, fps_arr)
        data[f'filepaths_{split_name}'] = final_fp_path
        del fps_arr, all_fps

        # 清理该 split 的临时 batch 文件
        for sf in temp_spec_files:
            sf.unlink(missing_ok=True)
        for lf in temp_label_files:
            lf.unlink(missing_ok=True)

    # 清理临时目录
    shutil.rmtree(temp_dir, ignore_errors=True)

    # 保存 word_list（小文件）
    np.save(output_path / "word_list.npy", np.array(words))
    data['word_list'] = output_path / "word_list.npy"

    # 保存统计信息
    _save_dataset_stats(output_path, data, words)

    print("[INFO] Preprocessing complete!")
    return data


def compute_expected_frames(sr: int, duration: float,
                             frame_len_ms: float, frame_shift_ms: float) -> int:
    """计算给定参数下梅尔频谱图的预期帧数"""
    total_samples = int(sr * duration)
    frame_length = int(frame_len_ms / 1000 * sr)
    frame_shift = int(frame_shift_ms / 1000 * sr)
    if total_samples < frame_length:
        return 1
    return 1 + (total_samples - frame_length) // frame_shift


def _save_dataset_stats(output_path: Path, data: dict, words: list):
    """保存数据集统计信息到文本文件"""
    stats_file = output_path / "dataset_stats.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Google Speech Commands v2 — Dataset Statistics\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Number of classes: {len(words)}\n")
        f.write(f"Classes: {words}\n\n")
        f.write(f"Mel bands: {Config.N_MELS}\n")
        f.write(f"Sample rate: {Config.SAMPLE_RATE} Hz\n")
        f.write(f"Audio duration: {Config.AUDIO_DURATION}s\n\n")

        for split in ['train', 'val', 'test']:
            key = f'X_{split}'
            if key in data:
                # data 中可能存的是 array 或 Path，统一处理
                x = data[key] if isinstance(data[key], np.ndarray) else np.load(data[key])
                y = data[f'y_{split}'] if isinstance(data[f'y_{split}'], np.ndarray) else np.load(data[f'y_{split}'])
                f.write(f"{split.upper()} set:\n")
                f.write(f"  Samples: {len(x)}\n")
                f.write(f"  Shape: {x.shape}\n")
                f.write(f"  Size: {x.nbytes / 1024 / 1024:.1f} MB\n")
                f.write(f"  Classes distribution:\n")
                for i, word in enumerate(words):
                    count = np.sum(y == i)
                    f.write(f"    [{i}] {word}: {count}\n")
                f.write("\n")
                # 如果是从文件加载的，释放引用
                if not isinstance(data[key], np.ndarray):
                    del x, y


# ============================================================
# SpecAugment — 频谱图数据增强 (Google Brain, 2019)
# ============================================================

class SpecAugment:
    """
    SpecAugment: 直接在频谱图上做数据增强

    参考文献: Park et al., "SpecAugment: A Simple Data Augmentation
             Method for Automatic Speech Recognition", Interspeech 2019.

    三种增强操作:
      1. Time Warping (时间扭曲): 模拟语速变化
      2. Frequency Masking (频率遮盖): 模拟部分频带丢失
      3. Time Masking (时间遮盖): 模拟短时噪声干扰

    这些增强迫使模型从部分信息中学习完整的时频表示,
    大幅提升泛化能力和噪声鲁棒性。
    """

    def __init__(self,
                 freq_mask_param: int = 27,
                 time_mask_param: int = 25,
                 n_freq_masks: int = 2,
                 n_time_masks: int = 2,
                 time_warp_param: int = 0):
        """
        Args:
            freq_mask_param: 频率遮盖的最大宽度 (Mel 频带数)
            time_mask_param: 时间遮盖的最大宽度 (帧数)
            n_freq_masks: 频率遮盖次数
            n_time_masks: 时间遮盖次数
            time_warp_param: 时间扭曲的最大偏移 (0=禁用)
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.time_warp_param = time_warp_param

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        """
        对单个频谱图应用 SpecAugment

        Args:
            spec: 输入频谱图, shape (n_mels, n_frames)

        Returns:
            增强后的频谱图
        """
        spec = spec.copy()

        # Time warping (可选, 计算量较大)
        if self.time_warp_param > 0:
            spec = self._time_warp(spec)

        # Frequency masking
        for _ in range(self.n_freq_masks):
            spec = self._freq_mask(spec)

        # Time masking
        for _ in range(self.n_time_masks):
            spec = self._time_mask(spec)

        return spec

    def _freq_mask(self, spec: np.ndarray) -> np.ndarray:
        """频率遮盖 — 遮盖连续的 Mel 频带"""
        n_mels = spec.shape[0]
        f = np.random.randint(0, self.freq_mask_param + 1)
        if f == 0:
            return spec
        f0 = np.random.randint(0, max(1, n_mels - f + 1))
        spec[f0:f0 + f, :] = 0.0
        return spec

    def _time_mask(self, spec: np.ndarray) -> np.ndarray:
        """时间遮盖 — 遮盖连续的时间帧"""
        n_frames = spec.shape[1]
        t = np.random.randint(0, self.time_mask_param + 1)
        if t == 0:
            return spec
        t0 = np.random.randint(0, max(1, n_frames - t + 1))
        spec[:, t0:t0 + t] = 0.0
        return spec

    def _time_warp(self, spec: np.ndarray) -> np.ndarray:
        """时间扭曲 — 沿时间轴局部拉伸/压缩"""
        n_frames = spec.shape[1]
        if n_frames < 4:
            return spec
        warp_size = min(self.time_warp_param, n_frames - 2)
        center = np.random.randint(warp_size // 2, n_frames - warp_size // 2)
        warped = np.zeros_like(spec)
        src_start = max(0, center - warp_size // 2)
        src_end = min(n_frames, center + warp_size // 2)
        dst_start = max(0, center - warp_size // 4)
        width = min(src_end - src_start, n_frames - dst_start)

        for ch in range(spec.shape[0]):
            # 线性插值实现时间扭曲
            src_x = np.linspace(0, n_frames - 1, n_frames)
            # 在扭曲区域: 对时间轴做局部非线性变换
            warp_x = src_x.copy()
            local_indices = np.arange(src_start, src_start + width)
            if len(local_indices) > 0 and dst_start < n_frames:
                dst_indices = np.arange(dst_start, min(dst_start + width, n_frames))
                min_len = min(len(local_indices), len(dst_indices))
                warp_x[dst_indices[:min_len]] = local_indices[:min_len]
            warped[ch] = np.interp(src_x, warp_x, spec[ch], left=0, right=0)

        return warped


# ============================================================
# 工具函数
# ============================================================

def audio_augment_waveform(audio: np.ndarray,
                            sr: int,
                            time_stretch_range: Tuple[float, float] = (0.8, 1.2),
                            pitch_shift_steps: int = 2,
                            noise_std: float = 0.005) -> np.ndarray:
    """
    波形级别的数据增强 (在原始音频上操作)

    Args:
        audio: 原始音频
        sr: 采样率
        time_stretch_range: 时间拉伸范围 (倍率)
        pitch_shift_steps: 音调变换的半音数范围
        noise_std: 叠加高斯噪声的标准差

    Returns:
        增强后的音频
    """
    augmented = audio.copy()

    # 时间拉伸 (改变速度, 保持音调)
    if np.random.random() < 0.5:
        rate = np.random.uniform(*time_stretch_range)
        augmented = librosa.effects.time_stretch(y=augmented, rate=rate)

    # 音调变换
    if np.random.random() < 0.3:
        steps = np.random.randint(-pitch_shift_steps, pitch_shift_steps + 1)
        if steps != 0:
            augmented = librosa.effects.pitch_shift(y=augmented, sr=sr, n_steps=steps)

    # 高斯噪声
    if np.random.random() < 0.3:
        noise_actual = np.random.uniform(0, noise_std)
        augmented = augmented + np.random.randn(len(augmented)) * noise_actual

    return augmented


def add_noise_for_snr_test(audio: np.ndarray,
                            snr_db: float,
                            noise_type: str = 'gaussian') -> np.ndarray:
    """
    向音频添加指定 SNR 的噪声 (用于鲁棒性测试)

    SNR 定义: SNR(dB) = 10·log₁₀(P_signal / P_noise)

    Args:
        audio: 纯净音频
        snr_db: 目标信噪比 (dB), 如 10, 5, 0, -5
        noise_type: 'gaussian' 或 'babble' (使用白噪声近似)

    Returns:
        加噪后的音频
    """
    signal_power = np.mean(audio ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / (snr_linear + 1e-10)

    if noise_type == 'gaussian':
        noise = np.sqrt(noise_power) * np.random.randn(len(audio))
    else:
        # 近似 babble noise: 用有色噪声近似
        white_noise = np.random.randn(len(audio))
        b = signal.firwin(101, [0.2, 0.6], pass_zero=False)
        noise = signal.lfilter(b, 1.0, white_noise)
        noise = np.sqrt(noise_power / (np.mean(noise ** 2) + 1e-10)) * noise

    return audio + noise


# ============================================================
# TTS 中文数据集扫描
# ============================================================

def scan_tts_files(tts_dir: str,
                    en_words: List[str] = GSC_20_WORDS) -> Tuple[List[str], List[int]]:
    """
    扫描 TTS 合成数据集目录

    TTS 目录结构 (与 GSC 一致):
      tts_dir/
        ├── zero/    (英文命名的类别目录)
        │   ├── ling_Xiaoxiao_r100_00.wav
        │   └── ...
        ├── one/
        ├── ...
        ├── yes/
        ├── ...
        └── stop/

    Args:
        tts_dir: TTS 数据根目录
        en_words: 英文词汇列表 (类别子目录名)

    Returns:
        filepaths: wav 文件路径列表
        labels: 标签 (使用 GSC_20_WORDS 的索引, 中文偏移 +20)
    """
    filepaths = []
    labels = []

    for word in en_words:
        word_dir = Path(tts_dir) / word
        if not word_dir.exists():
            print(f"[WARNING] TTS directory not found: {word_dir}, skipping '{word}'")
            continue

        wav_files = sorted(word_dir.glob("*.wav"))
        for f in wav_files:
            filepaths.append(str(f))
            # 中文词用 offset=20 的标签
            labels.append(len(GSC_20_WORDS) + GSC_20_WORDS.index(word))

        print(f"  [TTS-{word}]: {len(wav_files)} samples")

    return filepaths, labels


# ============================================================
# 混合数据集预处理 (英文 GSC + 中文 TTS)
# ============================================================

def preprocess_mixed_dataset(gsc_data_dir: str,
                              tts_data_dir: str,
                              words: List[str] = BILINGUAL_40_WORDS,
                              output_dir: str = None,
                              test_size: float = 0.15,
                              val_size: float = 0.15) -> Dict:
    """
    混合数据集预处理: GSC 英文 + TTS 中文 → 40 类联合 .npy

    处理流程:
      1. 扫描 GSC 20 词 → 标签 0-19
      2. 扫描 TTS 20 中文词 → 标签 20-39
      3. 对每个 wav 执行完整的信号处理流水线
      4. 合并并随机划分 train/val/test
      5. 保存为 .npy

    Args:
        gsc_data_dir: GSC 数据集根目录
        tts_data_dir: TTS 数据集根目录
        words: 40 类完整词汇列表
        output_dir: 输出目录
        test_size: 测试集比例
        val_size: 验证集比例

    Returns:
        data_dict
    """
    if output_dir is None:
        from pathlib import Path as _Path
        output_dir = str(_Path(gsc_data_dir).parent.parent / "processed_mixed")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 检查缓存
    train_file = output_path / "X_train.npy"
    if train_file.exists():
        print(f"[INFO] Preprocessed mixed data found at {output_path}, loading...")
        data = {}
        for name in ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']:
            data[name] = np.load(output_path / f"{name}.npy")
        data['word_list'] = np.load(output_path / "word_list.npy", allow_pickle=True)
        return data

    # 扫描 GSC 数据 (标签 0-19)
    print("\n[INFO] Scanning GSC English dataset...")
    gsc_files, gsc_labels = scan_gsc_files(gsc_data_dir, GSC_20_WORDS)

    # 扫描 TTS 数据 (标签 20-39)
    print("\n[INFO] Scanning TTS Chinese dataset...")
    tts_files, tts_labels = scan_tts_files(tts_data_dir, GSC_20_WORDS)

    # 合并
    all_files = gsc_files + tts_files
    all_labels = gsc_labels + tts_labels

    print(f"\n[INFO] Total: {len(gsc_files)} English + {len(tts_files)} Chinese = {len(all_files)} samples")
    print(f"[INFO] Classes: {len(words)} (20 English + 20 Chinese)")

    # 随机划分
    np.random.seed(Config.SEED)
    indices = np.random.permutation(len(all_files))
    n_test = int(len(indices) * test_size)
    n_val = int(len(indices) * val_size)
    n_train = len(indices) - n_test - n_val

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    splits = {'train': train_idx, 'val': val_idx, 'test': test_idx}

    n_expected_frames = compute_expected_frames(Config.SAMPLE_RATE, Config.AUDIO_DURATION,
                                                  Config.FRAME_LENGTH_MS, Config.FRAME_SHIFT_MS)
    print(f"[INFO] Mel spectrogram shape: ({Config.N_MELS}, {n_expected_frames})")

    data = {}

    for split_name, split_idx in splits.items():
        print(f"\n[INFO] Processing {split_name} set ({len(split_idx)} samples)...")
        specs = np.zeros((len(split_idx), Config.N_MELS, n_expected_frames), dtype=np.float32)
        y = np.zeros(len(split_idx), dtype=np.int64)

        valid_count = 0
        for i, idx in enumerate(tqdm(split_idx, desc=f"  {split_name}")):
            try:
                spec = load_and_preprocess_audio(
                    all_files[idx],
                    target_sr=Config.SAMPLE_RATE,
                    target_duration=Config.AUDIO_DURATION,
                    n_mels=Config.N_MELS
                )
                if spec.shape[1] < n_expected_frames:
                    pad_width = n_expected_frames - spec.shape[1]
                    spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
                elif spec.shape[1] > n_expected_frames:
                    spec = spec[:, :n_expected_frames]

                specs[valid_count] = spec
                y[valid_count] = all_labels[idx]
                valid_count += 1
            except Exception as e:
                pass  # 静默跳过失败文件

        specs = specs[:valid_count]
        y = y[:valid_count]

        print(f"  Valid samples: {valid_count}")

        data[f'X_{split_name}'] = specs
        data[f'y_{split_name}'] = y

    data['word_list'] = np.array(words)

    # 保存
    print(f"\n[INFO] Saving preprocessed mixed data to {output_path}...")
    for key, value in data.items():
        np.save(output_path / f"{key}.npy", value)

    # 保存统计
    _save_mixed_dataset_stats(output_path, data, words)

    print("[INFO] Mixed preprocessing complete!")
    return data


def _save_mixed_dataset_stats(output_path: Path, data: dict, words: list):
    """保存混合数据集统计信息"""
    stats_file = output_path / "dataset_stats.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Bilingual Speech Dataset (GSC + TTS Chinese) — Statistics\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Number of classes: {len(words)}\n")
        f.write(f"  English (0-19): {words[:20]}\n")
        f.write(f"  Chinese (20-39): {words[20:]}\n\n")

        for split in ['train', 'val', 'test']:
            key = f'X_{split}'
            if key in data:
                x = data[key]
                y = data[f'y_{split}']
                n_en = int(np.sum(y < 20))
                n_zh = int(np.sum(y >= 20))
                f.write(f"{split.upper()} set:\n")
                f.write(f"  Samples: {len(x)} ({n_en} EN + {n_zh} ZH)\n")
                f.write(f"  Shape: {x.shape}\n")
                f.write(f"  Size: {x.nbytes / 1024 / 1024:.1f} MB\n")
                for i, word in enumerate(words):
                    count = int(np.sum(y == i))
                    if count > 0:
                        lang = 'EN' if i < 20 else 'ZH'
                        f.write(f"    [{i:2d}] {lang} {word}: {count}\n")
                f.write("\n")


# ============================================================
# 主入口: 独立运行数据预处理
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audio preprocessing for Speech CNN")
    parser.add_argument('--data_dir', type=str,
                        default='../data/raw/gsc',
                        help='GSC dataset directory')
    parser.add_argument('--output_dir', type=str,
                        default='../data/processed',
                        help='Output directory for processed spectrograms')
    parser.add_argument('--words', type=str, nargs='+',
                        default=GSC_20_WORDS,
                        help='Words to include')
    parser.add_argument('--download', action='store_true',
                        help='Download GSC dataset first')
    parser.add_argument('--demo', type=str, default=None,
                        help='Demo: process a single wav file and show info')
    parser.add_argument('--tts_dir', type=str, default=None,
                        help='TTS Chinese dataset directory (for mixed mode)')
    parser.add_argument('--mode', type=str, default='gsc',
                        choices=['gsc', 'mixed'],
                        help='Preprocessing mode: gsc (English only) or mixed (EN+ZH)')

    args = parser.parse_args()

    if args.demo:
        # 演示模式: 处理单个文件并打印信息
        spec = load_and_preprocess_audio(args.demo)
        print(f"Input file: {args.demo}")
        print(f"Mel spectrogram shape: {spec.shape}")
        print(f"Value range: [{spec.min():.3f}, {spec.max():.3f}]")
        print(f"Mean: {spec.mean():.3f}, Std: {spec.std():.3f}")

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 6))

        audio, sr = librosa.load(args.demo, sr=Config.SAMPLE_RATE)
        t = np.arange(len(audio)) / sr
        axes[0].plot(t, audio, linewidth=0.5)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Raw Waveform')
        axes[0].set_xlim(0, len(audio) / sr)

        im = axes[1].imshow(spec, aspect='auto', origin='lower',
                            cmap='magma')
        axes[1].set_xlabel('Time Frames')
        axes[1].set_ylabel('Mel Frequency Bands')
        axes[1].set_title('Mel Spectrogram')
        plt.colorbar(im, ax=axes[1], label='dB (normalized)')
        plt.tight_layout()

        out_path = str(Path(args.demo).with_suffix('.png'))
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {out_path}")
        sys.exit(0)

    # 主模式: 批量预处理
    if args.mode == 'mixed':
        # 混合模式: GSC 英文 + TTS 中文
        if not args.tts_dir:
            print("[ERROR] Mixed mode requires --tts_dir pointing to TTS Chinese data")
            print("  Example: python preprocess.py --mode mixed --tts_dir ../data/raw/tts_chinese")
            sys.exit(1)

        data = preprocess_mixed_dataset(
            gsc_data_dir=args.data_dir,
            tts_data_dir=args.tts_dir,
            words=BILINGUAL_40_WORDS,
            output_dir=args.output_dir,
        )
    else:
        # GSC 英文模式
        if args.download:
            data_dir = download_gsc_dataset(args.data_dir)
        else:
            data_dir = args.data_dir

        data = preprocess_gsc_dataset(
            data_dir=data_dir,
            words=args.words,
            output_dir=args.output_dir
        )

    print("\n[INFO] Preprocessing finished successfully!")
    for split in ['train', 'val', 'test']:
        x_data = data[f'X_{split}']
        if isinstance(x_data, np.ndarray):
            shape = x_data.shape
            nbytes = x_data.nbytes
            print(f"  {split}: {shape} → {nbytes/1024/1024:.1f} MB")
        else:
            print(f"  {split}: {x_data} (loaded)")
