# 🎙️ 端到端 CNN 语音命令识别 — 技术文档

> **信号与系统课程设计 · 基于梅尔频谱图 + 2D CNN 的语音分类系统**
>
> 对标 EMG 手势识别项目 (CAE+CNN), 将信号处理理论 (傅里叶变换、STFT、Mel 滤波器组) 与现代深度学习 (CNN) 完美结合。

---

## 📌 项目版本

| 版本 | 内容 | 类别数 | 数据 | 状态 |
|:---|:---|:---|:---|:---|
| **V1 — 英文版** | 英文命令词 + 数字 | 20 | GSC v2 真人录音 (77,454 条) | ✅ 已完成 · 测试准确率 **96.49%** |
| **V2 — 中英双语版** | EN 20 + ZH 20 | 40 | GSC v2 + TTS 合成 (EN:ZH = 52:1) | ✅ 95.46%, EN 95.8%, ZH 76.1% |
| **V3 — 加权平衡版** | 同上 + --use_class_weights | 40 | 同上 | ✅ 94.29%, EN 94.5%, **ZH 82.9%** |
| **V4 — 数据扩充版** | 扩 TTS + 过采样 + 加权 | 40 | ZH 9,491 条 (8.2:1) | ✅ 95.74%, EN 96.1%, ZH 77.5% |

> 传统方法对比 (SVM/RF/KNN vs CNN) 已分别归入 V1 (20类) 和 V3 (40类) 的实验分析中。

> 📋 **报告数据**: 所有关键实验数据已整理至 [`PROJECT_LOG.md`](PROJECT_LOG.md)，可直接用于撰写课程报告

---

## 📋 目录

- [1. 项目概述](#1-项目概述)
- [2. 实际训练结果](#2-实际训练结果)
- [3. 理论基础与信号处理](#3-理论基础与信号处理)
- [4. 环境配置](#4-环境配置)
- [5. 数据集](#5-数据集)
- [6. 快速开始](#6-快速开始)
- [7. 详细使用指南](#7-详细使用指南)
- [8. 🆕 中英双语扩展](#8--中英双语扩展)
- [9. 用自己的语音测试模型](#9-用自己的语音测试模型)
- [10. 模型架构](#10-模型架构)
- [11. 超参数配置](#11-超参数配置)
- [12. 评估与分析](#12-评估与分析)
- [13. 项目文件结构](#13-项目文件结构)
- [14. 常见问题 (FAQ)](#14-常见问题-faq)
- [15. 后续优化方向](#15-后续优化方向)
- [16. 参考文献](#16-参考文献)

---

## 1. 项目概述

### 1.1 项目目标

使用端到端的 **2D 卷积神经网络 (CNN)** 对语音命令进行分类识别。

| 模式 | 类别数 | 语言 | 数据来源 |
|:---|:---|:---|:---|
| **V1 (已完成)** | 20 类 | English | Google Speech Commands v2 |
| **V2 (新增)** | 40 类 | English + 中文 | GSC v2 + Microsoft TTS 合成 |

将音频信号通过**短时傅里叶变换 (STFT) + 梅尔滤波器组**转换为梅尔频谱图, 然后将其作为"图像"输入 CNN 进行分类。

### 1.2 核心特点

| 特点 | 说明 |
|:---|:---|
| 🧠 **端到端学习** | 从原始频谱图直接学习分类特征, 无需手工特征工程 |
| 📐 **理论扎实** | 完整覆盖傅里叶变换、STFT、滤波器组等信号与系统核心概念 |
| 🚀 **轻量高效** | 模型参数量 ~400K, GPU 训练 <15 分钟, CPU 也可训练 |
| 📊 **全面评估** | 混淆矩阵、ROC、t-SNE、噪声鲁棒性、特征图可视化共 **8 项** |
| 🎤 **自录测试** | 独立推理脚本, 支持单文件/文件夹/麦克风实时识别 |
| 🌐 **中英双语** | TTS 合成中文数据, 混合训练 40 类联合模型 |
| 🔬 **可扩展** | 支持可选的 CAE 降噪自编码器、Deep/Light 模型变体 |

### 1.3 V1 — 20 类英文词汇

```
命令词 (10):  yes, no, up, down, left, right, on, off, stop, go
数字 (10):    zero, one, two, three, four, five, six, seven, eight, nine
```

### 1.4 🆕 V2 — 40 类中英双语词汇

```
英文 (20):  yes no up down left right on off stop go ...
中文 (20):  是 不 上 下 左 右 开 关 停 走 零 一 二 三 四 五 六 七 八 九
```

---

## 2. 实际训练结果

> **模型**: SpeechCNN Standard · **训练时间**: 2026-06-11 22:07 · **设备**: GPU

### 2.1 总体指标

| 指标 | 训练集 | 验证集 (最佳) | 测试集 |
|:---|:---|:---|:---|
| **Accuracy** | 79.58% | **96.52%** (epoch 96) | **96.49%** |
| **Loss** | 1.256 | 0.867 | — |
| **Macro Precision** | — | — | 96.52% |
| **Macro Recall** | — | — | 96.47% |
| **Macro F1** | — | — | **96.48%** |

### 2.2 各类别详细指标 (测试集)

| 类别 | Precision | Recall | F1 | Support |
|:---|:---|:---|:---|:---|
| yes | 97.93% | 97.09% | 97.51% | 584 |
| no | 96.05% | 95.89% | 95.97% | 584 |
| up | 87.75% | 95.72% | 91.56% | 561 |
| down | 97.18% | 95.34% | 96.25% | 579 |
| left | 95.57% | 96.28% | 95.93% | 538 |
| right | 97.89% | 96.23% | 97.05% | 530 |
| on | 95.81% | 96.78% | 96.29% | 590 |
| off | 94.37% | 95.51% | 94.94% | 579 |
| stop | 98.10% | 97.93% | 98.02% | 581 |
| go | 94.01% | 94.33% | 94.17% | 582 |
| zero | 97.61% | 97.75% | 97.68% | 668 |
| one | 98.21% | 95.81% | 97.00% | 573 |
| two | 95.41% | 96.43% | 95.92% | 561 |
| three | 98.76% | 95.39% | 97.05% | 586 |
| four | 98.04% | 95.49% | 96.75% | 576 |
| five | 97.36% | 97.52% | 97.44% | 604 |
| six | 98.45% | 97.77% | 98.11% | 584 |
| seven | 97.50% | 97.01% | 97.25% | 602 |
| eight | 97.76% | 97.59% | 97.68% | 582 |
| nine | 96.55% | 97.56% | 97.05% | 574 |

### 2.3 训练曲线

训练过程平滑收敛, 验证准确率从第 1 轮的 10.3% 稳步上升至 96.5%。
- Loss 持续下降: 训练 3.03 → 1.26, 验证 2.88 → 0.87
- 无过拟合迹象 (验证准确率始终 ≥ 训练准确率, 因为训练集用了 SpecAugment)
- 早停未触发 (第 96 轮仍创新高), 100 轮正常结束

### 2.4 可视化结果

所有评估图表已生成至 `results/figures/`:

| 文件 | 内容 |
|:---|:---|
| `confusion_matrix_normalized.png` | 归一化混淆矩阵 (20×20) |
| `confusion_matrix_counts.png` | 原始计数混淆矩阵 |
| `roc_curves.png` | 各类别 ROC 曲线 |
| `training_history.png` | 训练/验证 Loss 和 Accuracy 曲线 |
| `tsne_features.png` | CNN 特征空间的 t-SNE 二维投影 |
| `feature_maps.png` | 各卷积层激活特征图 |
| `noise_robustness.png` | SNR-Accuracy 噪声鲁棒性曲线 |
| `metrics.json` | 全部数值指标 (JSON 格式) |
| `error_analysis.txt` | 高置信度分类错误分析 |

---

## 3. 理论基础与信号处理

### 3.1 完整信号处理流水线

```
原始音频 x[n] (16kHz, 1s = 16000点)
    │
    ▼
┌──────────────────────────────┐
│ 1. 预加重 (Pre-emphasis)      │  H(z) = 1 - αz⁻¹, α=0.97
│    ─ 阶高通 FIR 滤波器         │  补偿高频分量在唇辐射中的衰减
│    y[n] = x[n] - 0.97·x[n-1] │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 2. 分帧 + 加窗                │  帧长 N=400 (25ms @ 16kHz)
│    x̃ₘ[n] = x[n+mH] · w[n]    │  帧移 H=160 (10ms @ 16kHz)
│    w[n]: 汉明窗               │  短时平稳假设 (10-30ms)
│    w[n]=0.54-0.46cos(2πn/N)  │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 3. 短时傅里叶变换 (STFT)      │  K=512 (FFT点数, 补零)
│    Xₘ[k] = Σₙ x̃ₘ[n]e⁻ʲ²πᵏⁿ/ᴷ │  对每帧做 DFT
│    Pₘ[k] = |Xₘ[k]|² / K      │  得到功率谱
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 4. Mel 滤波器组               │  B=128 个三角滤波器
│    mel(f)=2595·log₁₀(1+f/700)│  Hz → Mel 非线性映射
│    Sₘ[b] = Σₖ Pₘ[k]·H_b[k]   │  模拟人耳基底膜频率感知
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 5. 对数压缩                   │  模仿人耳对数响度感知
│    S_dB = 10·log₁₀(S + ε)    │  ε=10⁻¹⁰ (避免 log 0)
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 6. Z-score 归一化             │
│    Ŝ = (S - μ) / σ           │  全局标准化
└──────────────┬───────────────┘
               ▼
    梅尔频谱图 (128 × 98)
    → CNN 输入 "图像"
```

### 3.2 与信号与系统课程的关联

| 课程章节 | 核心概念 | 在项目中的应用 |
|:---|:---|:---|
| 傅里叶变换 | FT / DFT / FFT | STFT 频谱分析, 512 点 FFT |
| 采样定理 | Nyquist-Shannon | 16kHz 采样率 = 2×8kHz 带宽 |
| 滤波器设计 | FIR / IIR, 窗函数法 | Mel 三角滤波器组 (128 个 FIR 带通) |
| 时频分析 | 短时傅里叶变换 | 分帧加窗 → 频谱图 |
| LTI 系统 | 卷积、系统函数 | H(z)=1-0.97z⁻¹ (预加重) |
| 噪声 | SNR, 加性高斯噪声 | 鲁棒性测试: inf, 20, 15, 10, 5, 0, -5 dB |

### 3.3 SpecAugment 数据增强原理

```
原始频谱图                    增强后频谱图
┌──────────────────┐         ┌──────────────────┐
│ ████████████████ │         │ ████████████████ │
│ ████████████████ │  Freq   │ ██████░░░░██████ │ ← 频率遮盖 (模拟频带丢失)
│ ████████████████ │  Mask   │ ████████████████ │
│ ████████████████ │ ──────→ │ ██████░░░░██████ │
│ ████████████████ │  Time   │ ████░░░██░██████ │ ← 时间遮盖 (模拟短时噪声)
│ ████████████████ │  Mask   │ ████████████████ │
└──────────────────┘         └──────────────────┘
```

参考文献: Park et al., "SpecAugment", Interspeech 2019.

---

## 4. 环境配置

### 4.1 系统要求

| 项目 | 最低要求 | 推荐配置 |
|:---|:---|:---|
| 操作系统 | Windows 10+ / Linux / macOS | 任意 |
| Python | 3.9+ | 3.10+ |
| RAM | 8 GB | 16 GB |
| 磁盘 | 5 GB (含数据集) | 10 GB |
| GPU (可选) | — | NVIDIA GTX 1060+ (训练快 5-10×) |

### 4.2 安装步骤

```bash
# 1. 创建虚拟环境 (推荐)
python -m venv venv

# 2. 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. 安装依赖
cd speech_cnn_project
pip install -r requirements.txt

# 4. 验证安装
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import librosa; print('librosa:', librosa.__version__)"
```

### 4.3 GPU 支持 (推荐)

```bash
# CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 验证
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 5. 数据集

### 5.1 Google Speech Commands v2

| 属性 | 描述 |
|:---|:---|
| 总样本数 | 105,829 (本项目使用 77,454 条, 20 类) |
| 词汇数 | 35 个命令词 |
| 说话人数 | 2,618 人 |
| 采样率 | 16 kHz |
| 格式 | WAV (单声道, 16-bit PCM) |
| 时长 | 每个文件 1 秒 |
| 大小 | ~2.3 GB (压缩) |
| 许可证 | Creative Commons BY 4.0 |
| 下载地址 | `http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz` |

### 5.2 数据集划分

| 集合 | 比例 | 样本数 | 用途 |
|:---|:---|:---|:---|
| 训练集 | 70% | 54,218 | 模型训练 + SpecAugment 增强 |
| 验证集 | 15% | 11,618 | 超参数调优、早停监控 |
| 测试集 | 15% | 11,618 | 最终性能评估 |

### 5.3 预处理输出

```
data/processed/
├── X_train.npy           # 训练集频谱图 (54218, 128, 98)
├── y_train.npy           # 训练集标签
├── X_val.npy             # 验证集频谱图 (11618, 128, 98)
├── y_val.npy             # 验证集标签
├── X_test.npy            # 测试集频谱图 (11618, 128, 98)
├── y_test.npy            # 测试集标签
├── word_list.npy         # 词汇列表 (20,)
├── filepaths_*.npy       # 每个样本的 GSC 原始文件路径
└── dataset_stats.txt     # 数据集统计信息 (各类别分布)
```

---

## 6. 快速开始

### 6.1 查看所有可用模型

```bash
cd src
python evaluate.py --list_models
# 自动列出所有已训练模型及其准确率
```

### 6.2 V1 — 英文 20 类 (可直接用)

```bash
cd src

# 评估 (数据集自动匹配, 读 config.json)
python evaluate.py --model_path ../models/v1_english_20/20260611_220716/best_model.pth --all
# 或 python evaluate.py --model_path latest --all

# 推理
python inference.py --model ../models/v1_english_20/20260611_220716/best_model.pth --audio test.wav --visualize
```

### 6.3 V2 — 中英双语 40 类 (带类别加权)

```bash
cd src

# Step 1: 生成 TTS 中文数据 (已完成, ~1,500 条)
# pip install edge-tts && python generate_tts_data.py

# Step 2: 混合预处理 (已完成)
# python preprocess.py --mode mixed --data_dir ../data/raw/gsc --tts_dir ../data/raw/tts_chinese --output_dir ../data/processed_mixed

# Step 3: 训练 40 类模型 (--use_class_weights 解决中英文不均衡)
python train.py --data_dir ../data/processed_mixed --num_classes 40 --epochs 100 --run_name bilingual_v2_balanced --use_class_weights --batch_size 8

# Step 4: 评估 (注意: 必须指定 --data_dir ../data/processed_mixed)
python evaluate.py --model_path ../models/v2_bilingual_40/bilingual_v1/best_model.pth --data_dir ../data/processed_mixed --num_classes 40 --all

# Step 5: 推理
python inference.py --model ../models/v2_bilingual_40/bilingual_v1/best_model.pth --num_classes 40 --audio test.wav --visualize
```

### 6.4 🆕 CNN vs 传统方法对比

```bash
cd src

# V1 英文 20 类对比
python compare_traditional.py --data_dir ../data/processed --num_classes 20

# V2 中英 40 类对比 (含中英文分别统计)
python compare_traditional.py --data_dir ../data/processed_mixed --num_classes 40
```

输出在 `results/comparison/`：
- `method_comparison.png` — 各方法准确率/F1 柱状图
- `en_zh_comparison.png` — 中英文分别对比（仅 40 类）
- `comparison_results.json` — 原始数据

#### 40 类中英双语对比结果

| 方法 | 总体 Acc | Macro F1 | 英文 | 中文 | 训练时间 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **CNN (V3)** | **94.29%** | 82.17% | **94.5%** | 82.9% | 15 min |
| SVM (RBF) | 48.88% | 46.08% | 49.1% | 37.8% | 2 s |
| Random Forest | 69.63% | **83.57%** | 69.1% | **98.2%** | 31 s |
| KNN (k=5) | 49.97% | 40.55% | 49.3% | 83.8% | <1 s |

> **关键发现**: RF 在中文上 98.2% 但英文仅 69.1%——TTS 合成语音的模式单一，RF 容易过拟合；CNN 是唯一中英均衡的方法（94.5% EN + 82.9% ZH），体现了端到端特征学习的泛化优势。

### 6.5 🎤 交互式语音测试 (泛化能力验证)

```bash
cd src

# 自动加载最新模型, 测试全部词汇
python interactive_test.py

# 仅测试英文
python interactive_test.py --test_en_only

# 仅测试中文
python interactive_test.py --test_zh_only

# 自定义测试词汇
python interactive_test.py --words yes no stop go zero one two

# 指定模型
python interactive_test.py --model ../models/v1_english_20/20260611_220716/best_model.pth
```

交互流程：
```
屏幕显示要说的词 → 按 Enter 录音 → 实时显示 Top-3 预测 → 按 Enter 下一个
随时按 q 退出, 按 r 重录当前词
```

测试结束自动生成汇总报告：
- `recordings/<时间戳>/test_report.json` — 详细数据
- `recordings/<时间戳>/test_report.txt` — 可读报告
- `recordings/<时间戳>/*.wav` — 所有录音文件

### 6.6 一键运行

```bash
# V1: 英文 20 类
python run.py

# V2: 中英双语 40 类
python run.py --bilingual --tts_dir data/raw/tts_chinese --run_name bilingual_v1
```

---

## 7. 详细使用指南

### 7.1 `preprocess.py` — 数据预处理

```bash
python preprocess.py [OPTIONS]

选项:
  --data_dir PATH       GSC 数据集根目录
  --output_dir PATH     预处理输出目录
  --words w1 w2 ...     要包含的词汇
  --download            先下载数据集
  --demo FILE           演示模式: 单文件 → 频谱图可视化

示例:
  # 处理全部 20 词 (首次运行, 含下载)
  python preprocess.py --data_dir ../data/raw/gsc --download

  # 仅处理数字 0-9
  python preprocess.py --words zero one two three four five six seven eight nine

  # 可视化单个 wav 的频谱图
  python preprocess.py --demo my_voice.wav
```

### 7.2 `train.py` — 模型训练

```bash
python train.py [OPTIONS]

关键选项:
  --model {standard,light,deep}   模型架构
  --epochs N                      训练轮数
  --batch_size N                  批大小
  --lr FLOAT                      学习率
  --optimizer {adam,adamw,sgd}    优化器
  --lr_scheduler {plateau,cosine} 学习率调度
  --label_smoothing FLOAT         标签平滑 (0=禁用)
  --no_augment                    禁用数据增强
  --device {auto,cuda,cpu}        计算设备
  --resume PATH                   从断点恢复
  --run_name NAME                 运行名称

示例:
  # 标准训练 (默认配置, 96%+ 准确率)
  python train.py

  # 追求最高分
  python train.py --model deep --epochs 150 --lr 5e-4 --batch_size 16 --run_name deep_v2

  # 快速实验
  python train.py --model light --epochs 40 --no_augment

  # 从断点继续
  python train.py --resume ../models/run_xxx/checkpoint_epoch_50.pth
```

### 7.3 `evaluate.py` — 模型评估

```bash
python evaluate.py --model_path PATH [OPTIONS]

选项:
  --model_path PATH    模型路径 (必需)
  --all                运行 8 项完整评估
  --task TASK          单项: metrics | confusion | roc | tsne |
                       history | noise | error | feature_maps
  --snr_levels DB ...  噪声测试 SNR 级别

示例:
  # 全量评估
  python evaluate.py --model_path ../models/v1_english_20/20260611_220716/best_model.pth --all

  # 仅混淆矩阵
  python evaluate.py --model_path ../models/v1_english_20/20260611_220716/best_model.pth --task confusion

  # 自定义 SNR
  python evaluate.py --model_path ../models/v1_english_20/20260611_220716/best_model.pth --task noise --snr_levels 30 20 10 0 -10
```

---

## 8. 🆕 中英双语扩展

### 8.1 方案概述

在现有 20 类英文模型基础上, 使用 **Microsoft Edge TTS** (免费) 合成中文语音数据,
扩展为 **40 类中英双语** 语音识别系统。

```
┌─────────────────────────────────────────────────┐
│              数据来源对比                         │
├──────────────────────┬──────────────────────────┤
│  英文 (20 类)         │  中文 (20 类)             │
│  GSC v2 真人录音      │  Edge TTS 合成            │
│  77,454 条            │  1,497 条 (303 条失败)     │
│  2,618 人             │  6 种声音 × 3 种语速      │
│  16kHz 自然语音       │  16kHz 高保真合成         │
└──────────────────────┴──────────────────────────┘
```

### 8.2 中文词汇映射

| 类别 | 英文 | 中文 | 拼音 |
|:---|:---|:---|:---|
| 数字 | zero ~ nine | 零 一 二 三 四 五 六 七 八 九 | líng yī èr sān sì wǔ liù qī bā jiǔ |
| 命令 | yes | 是 | shì |
| | no | 不 | bù |
| | up / down | 上 / 下 | shàng / xià |
| | left / right | 左 / 右 | zuǒ / yòu |
| | on / off | 开 / 关 | kāi / guān |
| | stop / go | 停 / 走 | tíng / zǒu |

### 8.3 TTS 声音配置

| 类型 | 声音名称 | 描述 |
|:---|:---|:---|
| 女声1 | `zh-CN-XiaoxiaoNeural` | 青年女声 |
| 女声2 | `zh-CN-XiaoyiNeural` | 青年女声 |
| 女声3 | `zh-CN-XiaochenNeural` | 中年女声 |
| 男声1 | `zh-CN-YunxiNeural` | 青年男声 |
| 男声2 | `zh-CN-YunyangNeural` | 中年男声 |
| 男声3 | `zh-CN-YunjianNeural` | 老年男声 |

语速变化: 0.85x / 1.0x / 1.15x, 每种配置重复 5 遍
生成计划: 20 词 × 6 声音 × 3 语速 × 5 遍 = 1,800 条
实际成功: **1,497 条** (83.2%, 303 条因网络波动失败)
每类均衡: 73-75 条 / 类别

### 8.4 一键运行中英双语流程

```bash
cd speech_cnn_project

# Step 1: 生成中文 TTS 数据 (~5-10 分钟)
cd src
pip install edge-tts  # 首次需要安装
python generate_tts_data.py

# Step 2: 混合预处理 (GSC 英文 + TTS 中文 → 40 类)
python preprocess.py --mode mixed --data_dir ../data/raw/gsc --tts_dir ../data/raw/tts_chinese --output_dir ../data/processed_mixed

# Step 3: 训练 40 类模型
python train.py --data_dir ../data/processed_mixed --num_classes 40 --epochs 100 --run_name bilingual_v1

# Step 4: 评估 (必须加 --data_dir ../data/processed_mixed)
python evaluate.py --model_path ../models/v2_bilingual_40/bilingual_v1/best_model.pth --data_dir ../data/processed_mixed --num_classes 40 --all

# Step 5: 推理测试 (中英文均可)
python inference.py --model ../models/v2_bilingual_40/bilingual_v1/best_model.pth --audio test.wav --num_classes 40
```

### 8.5 也可以用 run.py 一键运行

```bash
python run.py --bilingual --tts_dir data/raw/tts_chinese --run_name bilingual_v1
```

### 8.6 仅生成部分数据

```bash
# 仅中文数字
python generate_tts_data.py --digits_only

# 仅中文命令词
python generate_tts_data.py --commands_only

# 快速测试 (少量数据)
python generate_tts_data.py --num_repeats 2 --voices zh-CN-XiaoxiaoNeural zh-CN-YunxiNeural
```

---

## 9. 用自己的语音测试模型

> 完整推理脚本: `src/inference.py` — 支持单文件 / 批量 / 麦克风三种模式

### 9.1 单文件测试

```bash
cd src

# 基本推理 (Top-3 预测)
python inference.py --model ../models/v1_english_20/20260611_220716/best_model.pth --audio my_voice.wav --top_k 3

# 推理 + 可视化 (波形 + 频谱 + Top-K 柱状图)
python inference.py --model ../models/v1_english_20/20260611_220716/best_model.pth --audio my_voice.wav --top_k 5 --visualize
```

输出示例:
```
============================================================
  SINGLE FILE INFERENCE
============================================================
  File: my_yes.wav

  ▶ yes       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░  97.3%
    no        ▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   1.2%
    go        ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.8%

  Inference time: 2.3 ms
============================================================
```

### 8.2 批量测试文件夹

```bash
# 将多个录音放入一个文件夹
mkdir my_voice
# 放入 yes.wav, no.wav, stop.wav, zero.wav ...

python inference.py --model ../models/v1_english_20/20260611_220716/best_model.pth --folder my_voice/
```

### 8.3 交互式麦克风录制

```bash
pip install sounddevice  # 先装这个
python inference.py --model ../models/v1_english_20/20260611_220716/best_model.pth --record
```

按 Enter 开始录制 1.5 秒, 说出一个词, 模型实时预测。

### 8.4 录音建议

| 要点 | 说明 |
|:---|:---|
| 采样率 | 16kHz 最佳, 手机默认即可 |
| 时长 | 0.5-1.5 秒, 前后留 0.2s 静音 |
| 环境 | 安静室内, 避免回声 |
| 音量 | 正常说话音量 |
| 格式 | .wav, 单声道 |
| 命名 | 建议用单词命名, 如 `yes.wav`, `stop.wav` |

---

## 10. 模型架构

### 9.1 主模型: SpeechCNN (Standard)

```
Input: (1, 128, 98) 梅尔频谱图
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Block 1:
  Conv2D(1 → 32, 3×3, same) + BN + ReLU
  Conv2D(32 → 32, 3×3, same) + BN + ReLU
  MaxPool2D(2×2) + Dropout(0.2)
  → Output: (32, 64, 49)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Block 2:
  Conv2D(32 → 64, 3×3, same) + BN + ReLU
  Conv2D(64 → 64, 3×3, same) + BN + ReLU
  MaxPool2D(2×2) + Dropout(0.3)
  → Output: (64, 32, 24)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Block 3:
  Conv2D(64 → 128, 3×3, same) + BN + ReLU
  Conv2D(128 → 128, 3×3, same) + BN + ReLU
  GlobalAveragePooling2D + Dropout(0.4)
  → Output: (128,)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Classifier:
  Linear(128 → 256) + ReLU + Dropout(0.5)
  Linear(256 → 20)
  → Output: (20,) — 各类别 logits
```

### 9.2 模型变体

| 变体 | 参数量 | Block 数 | 通道数 | 适用场景 |
|:---|:---|:---|:---|:---|
| **Standard** ✅ | ~400K | 3 | 32→64→128 | 默认, 96.5% 准确率 |
| **Light** | ~80K | 3 | 24→48→96 | CPU 训练, 快速验证 |
| **Deep** | ~900K | 4 | 32→64→128→256 | 追求 97%+ |

### 9.3 可选: 降噪自编码器 (DenoisingCAE)

对标 EMG 手势识别项目的 AE+CNN 方案，用于频谱图级别的降噪预处理:

```
加噪频谱 → Encoder → 压缩特征 → Decoder → 去噪频谱
                          ↓
                    提取特征 → CNN → 分类

Encoder: Conv2D(1→16)→MP(2)→Conv2D(16→32)→MP(2)→Conv2D(32→32)
Decoder: Conv2D(32→16)→UP(2)→Conv2D(16→8)→UP(2)→Conv2D(8→1)
```

---

## 11. 超参数配置

### 10.1 当前最佳配置 (96.52% 验证准确率)

| 超参数 | 值 | 说明 |
|:---|:---|:---|
| `MODEL_TYPE` | standard | SpeechCNN Standard |
| `LEARNING_RATE` | 1e-3 | AdamW 初始学习率 |
| `WEIGHT_DECAY` | 1e-4 | L2 正则化 |
| `BATCH_SIZE` | 32 | 训练批大小 |
| `EPOCHS` | 100 | 训练轮数 |
| `LABEL_SMOOTHING` | 0.1 | 标签平滑正则化 |
| `OPTIMIZER` | adamw | Adam with decoupled weight decay |
| `LR_SCHEDULER` | plateau | ReduceLROnPlateau (factor=0.5, patience=8) |
| `EARLY_STOPPING` | 20 | 仅在第 100 轮正常运行结束 |

### 10.2 SpecAugment 参数

| 参数 | 值 | 说明 |
|:---|:---|:---|
| `FREQ_MASK_PARAM` | 27 | 频率遮盖最大宽度 (Mel 频带) |
| `TIME_MASK_PARAM` | 25 | 时间遮盖最大宽度 (帧数) |
| `N_FREQ_MASKS` | 2 | 频率遮盖次数 |
| `N_TIME_MASKS` | 2 | 时间遮盖次数 |

### 10.3 调参建议

```bash
# 追求最高准确率 (预期 +0.5~1.5%):
python train.py --model deep --epochs 150 --lr 5e-4 --batch_size 16 --label_smoothing 0.15

# 快速实验:
python train.py --model light --epochs 40 --batch_size 64 --no_augment

# 处理过拟合 (如果 val_acc >> train_acc 缩小):
python train.py --weight_decay 5e-4 --label_smoothing 0.15

# 处理欠拟合:
python train.py --model deep --epochs 200 --lr 2e-3
```

---

## 12. 评估与分析

### 11.1 评估项目清单

| 编号 | 评估项目 | 输出文件 | 说明 |
|:---|:---|:---|:---|
| 1 | 分类指标 | `metrics.json` | Accuracy, F1, Precision, Recall (含各类别) |
| 2 | 混淆矩阵 | `confusion_matrix_*.png` | 归一化 + 原始计数版本 (20×20) |
| 3 | ROC 曲线 | `roc_curves.png` | one-vs-rest, 含 AUC 值 |
| 4 | 训练曲线 | `training_history.png` | Loss & Accuracy 随 epoch 变化 |
| 5 | t-SNE 可视化 | `tsne_features.png` | CNN 特征空间的 2D 投影 |
| 6 | 特征图 | `feature_maps.png` | 各卷积层的激活可视化 |
| 7 | 🔊 噪声鲁棒性 | `noise_robustness.png` | **SNR-Accuracy 曲线** (信号与系统核心) |
| 8 | 错误分析 | `error_analysis.txt` | 高置信度分类错误样本 Top-20 |

### 11.2 如何解读结果

**混淆矩阵** — `confusion_matrix_normalized.png`:
- 对角线越亮越好 (正确分类率)
- 关注非对角线亮点: 这些是易混淆词对
- 本项目最易混淆: `up` (87.75% precision), `go` (94.01%)

**SNR-Accuracy 曲线** — `noise_robustness.png`:
- Clean ~20dB: 准确率应基本不变 (>95%)
- 20dB → 10dB: 略有下降正常
- 10dB → 0dB: 显著下降 (噪声功率接近信号功率)
- < 0dB: 噪声功率 > 信号功率, 趋于随机水平 (~5%)

**t-SNE 特征图** — `tsne_features.png`:
- 同类样本应聚集形成簇
- 不同类之间应有清晰边界
- 数字类 (zero-nine) 通常聚得比命令词类更紧

---

## 13. 项目文件结构

```
speech_cnn_project/
│
├── run.py                      # 🔥 一键运行: 下载→预处理→训练→评估
├── requirements.txt            # Python 依赖清单
├── README.md                   # 📖 本文档 (技术文档 + 实验结果 + 操作指南)
│
├── src/                        # 📦 源代码 (共 3,700+ 行 Python)
│   ├── __init__.py
│   ├── preprocess.py           # 音频→梅尔频谱 (STFT/Mel滤波器组/SpecAugment)
│   ├── dataset.py              # PyTorch Dataset + DataLoader + 噪声测试集
│   ├── model.py                # CNN 模型 (Standard/Light/Deep) + CAE
│   ├── train.py                # 训练引擎 (AMP/早停/LR调度/断点恢复)
│   ├── evaluate.py             # 8 项评估指标 + 全面可视化
│   ├── inference.py            # 自录语音推理 (单文件/批量/麦克风)
│   ├── gui_test.py             # 🆕 GUI 交互测试 (模型选择+录制+实时推理)
│   ├── interactive_test.py     # 终端交互式测试 (引导录制+汇总报告)
│   ├── compare_spectrograms.py # 🆕 频谱对比诊断 (真人 vs TTS)
│   ├── generate_tts_data.py    # TTS 中文语音合成 (Edge TTS)
│   └── compare_traditional.py  # CNN vs SVM/RF/KNN 对比验证
│
├── data/                       # 📊 数据
│   ├── raw/gsc/                # GSC v2 原始 WAV (77,454 文件, 20 类)
│   ├── raw/tts_chinese/       # 🆕 TTS 中文合成 WAV (20 类 × 因子)
│   ├── processed/              # V1 预处理 .npy (20 类英文)
│   └── processed_mixed/        # 🆕 V2 预处理 .npy (40 类中英)
│
├── models/                     # 💾 已训练模型
│   ├── v1_english_20/          # V1: 英文 20 类
│   │   └── 20260611_220716/    # 最佳运行 (96.52% val acc)
│   │       ├── best_model.pth
│   │       ├── model_weights.pth
│   │       ├── config.json
│   │       └── training_history.json
│   └── v2_bilingual_40/        # V2: 中英 40 类
│       └── bilingual_v1/       # 最佳运行 (95.47% val acc)
│           ├── best_model.pth
│           └── ...
│
├── results/                    # 📈 评估结果 (按版本/run 分目录)
│   ├── v1_english_20/
│   │   └── 20260611_220716/    # V1: 10 files
│   └── v2_bilingual_40/
│       ├── bilingual_v1/       # V2 (待跑)
│       └── bilingual_v3_weighted/  # V3: 10 files ✅
│
├── notebooks/                  # 📓 Jupyter Notebook (可选)
└── report/                     # 📝 课程报告目录 (待撰写)
```

---

## 14. 常见问题 (FAQ)

### Q1: 下载数据集太慢怎么办?

```bash
# 用浏览器或下载工具手动下载:
# http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
# 放到 data/ 目录下, 然后手动解压到 data/raw/gsc/
# Windows: 用 7-Zip 解压 .tar.gz
tar -xzf speech_commands_v0.02.tar.gz -C data/raw/gsc/
```

### Q2: V2 中文准确率为 0%？

> ⚠️ 根因：英文:中文 ≈ 50:1，极度不均衡。

**解决方案**：`--use_class_weights` 自动给中文加 ~50x 权重。

```bash
python train.py --data_dir ../data/processed_mixed --num_classes 40 --epochs 100 --run_name bilingual_v2_balanced --use_class_weights --batch_size 8
```

### Q3: 训练时显存/内存不够 (OOM)?

```bash
# 方案 A: 减小 batch_size (最直接)
python train.py --batch_size 8

# 方案 B: 轻量模型 + 小 batch (最省显存)
python train.py --model light --batch_size 16

# 方案 C: CPU 训练 (不占显存, 但慢)
python train.py --device cpu --batch_size 32

# 方案 D: 减少词汇数
python train.py --words yes no stop go --num_classes 4
```

> V2 40 类数据量 ~3.8GB, 比 V1 20 类 (~1.8GB) 大两倍。显存 <4GB 建议用方案 B 或 C。

```bash
python train.py --batch_size 8                      # 减小 batch
python train.py --model light --batch_size 16        # 轻量模型
python train.py --words yes no stop go --num_classes 4  # 减少类别
```

### Q3: 用自己数据训练?

将 .wav 文件按类别放入文件夹:
```
data/my_data/
├── cat/cat_001.wav, cat_002.wav, ...
└── dog/dog_001.wav, dog_002.wav, ...
```
然后通过 `--words cat dog` 指定类别名。

### Q4: 训练准确率上不去?

```
1. 检查 dataset_stats.txt 确认数据加载正确
2. 尝试: python train.py --model deep --epochs 150 --lr 5e-4
3. 尝试: python train.py --label_smoothing 0.15 --weight_decay 5e-4
4. 检查 SpecAugment 是否启用 (默认开启)
```

### Q5: CPU 训练时间?

| 模型 | 估计时间 |
|:---|:---|
| Light | 1-2 小时 |
| Standard | 4-6 小时 |
| Deep | >10 小时 (不推荐) |

### Q6: 如何部署?

```python
import torch
from src.preprocess import load_and_preprocess_audio
from src.model import create_model

model = create_model('standard', num_classes=20)
model.load_state_dict(torch.load('models/v1_english_20/20260611_220716/model_weights.pth'))
model.eval()

def predict(audio_path):
    spec = load_and_preprocess_audio(audio_path)
    spec_tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(spec_tensor)
        return logits.argmax(1).item()
```

### Q8: 如何运行中英双语 40 类模式?

```bash
# 1. 安装 TTS 依赖
pip install edge-tts

# 2. 生成中文语音
cd src
python generate_tts_data.py

# 3. 混合预处理
python preprocess.py --mode mixed --data_dir ../data/raw/gsc --tts_dir ../data/raw/tts_chinese --output_dir ../data/processed_mixed

# 4. 训练
python train.py --data_dir ../data/processed_mixed --num_classes 40 --run_name bilingual_v1

# 5. 评估 (注意: 必须指定正确的数据目录)
python evaluate.py --model_path ../models/v2_bilingual_40/bilingual_v1/best_model.pth --data_dir ../data/processed_mixed --num_classes 40 --all

# 6. 推理
python inference.py --model ../models/v2_bilingual_40/bilingual_v1/best_model.pth --audio test.wav --num_classes 40
```

### Q9: 与 EMG 手势识别项目的技术对应?

| EMG 项目文件 | 本项目文件 | 功能 |
|:---|:---|:---|
| `data_process_3_noise_new_1.m` | `preprocess.py` | 数据预处理 |
| `Feature_split.m` | `dataset.py` | 数据集划分 |
| `AE_with_same_input_of_NN.py` | `model.py` (DenoisingCAE) | 自编码器 |
| `CNN_same_input_with_NN.m` | `model.py` (SpeechCNN) | CNN 分类器 |
| `accuracy_10_percent.m` | `evaluate.py` (noise_robustness_test) | 噪声鲁棒性测试 |

---

## 15. 后续优化方向

当前 **96.49% 测试准确率** 已经是优秀水平，以下是可选的进一步提升方向:

| 优先级 | 方向 | 预期收益 | 工作量 | 与课程关联 |
|:---|:---|:---|:---|:---|
| ⭐⭐⭐ | **噪声鲁棒性深入分析** — 多种噪声类型 (白噪声/babble/car) + 更多 SNR 梯度 | 无 (结果更丰富) | 低 | SNR 理论 |
| ⭐⭐⭐ | **消融实验** — 对比 Mel 频带数 (64/128/256)、窗函数 (Hamming/Hann/Blackman) | 无 (分析价值) | 低 | 滤波器组设计 |
| ⭐⭐ | **Deep 模型** — 4-Block 架构重训 | +0.5~1.5% | 中 (重训~30min) | CNN 架构设计 |
| ⭐⭐ | **背景噪声训练** — 利用 GSC 的 `_background_noise_` 目录做混合训练 | +1~2% 鲁棒性 | 中 | 噪声建模 |
| ⭐ | **CAE 降噪** — 训练自编码器做前端降噪 (对标 EMG 项目) | 取决于噪声环境 | 高 | AE+CNN 联合 |
| ⭐ | **中英文对比** — 自录中文数字，测试英文模型的跨语言泛化 | 无 (探索价值) | 高 | 语谱图差异分析 |

---

## 16. 参考文献

1. **SpecAugment**: Park et al., "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition", *Interspeech 2019*.
2. **GSC Dataset**: Warden P., "Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition", *arXiv:1804.03209*, 2018.
3. **Mel Spectrogram**: Davis & Mermelstein, "Comparison of Parametric Representations for Monosyllabic Word Recognition", *IEEE Trans. ASSP*, 1980.
4. **CNN for Audio**: Hershey et al., "CNN Architectures for Large-Scale Audio Classification", *ICASSP 2017*.
5. **标签平滑**: Szegedy et al., "Rethinking the Inception Architecture for Computer Vision", *CVPR 2016*.
6. **AdamW**: Loshchilov & Hutter, "Decoupled Weight Decay Regularization", *ICLR 2019*.
7. **Batch Normalization**: Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network Training", *ICML 2015*.
8. **Kaiming 初始化**: He et al., "Delving Deep into Rectifiers", *ICCV 2015*.

---

## 📧 项目信息

| | |
|:---|:---|
| **项目名称** | Speech CNN — End-to-End Voice Command Recognition |
| **框架** | PyTorch 2.0+ |
| **最佳模型** | `models/v1_english_20/20260611_220716/best_model.pth` |
| **测试准确率** | **96.49%** (20 类, 11,618 样本) |
| **模型大小** | ~5 MB (权重文件) |
| **代码规模** | 7 个模块, 3,200+ 行 Python |
| **许可证** | MIT |

---

> **下一步**: 运行 `python src/inference.py --model models/v1_english_20/20260611_220716/best_model.pth --audio 你的录音.wav --visualize` 测试自己的语音, 然后撰写课程设计报告 (`report/report.md`)。
