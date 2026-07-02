"""
语音识别 GUI 测试工具 v2
=========================
改进:
  1. 启动时选择模型 (V1英文 / V4中英 / 自定义)
  2. 自动匹配测试词汇
  3. 按钮逻辑: 录制→查看结果→手动点下一个
  4. 中英文分别统计
"""

import sys, time, json, argparse, threading
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import scipy.signal as signal
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from model import create_model
from preprocess import (
    load_and_preprocess_audio, compute_mel_spectrogram,
    normalize_spectrogram, GSC_20_WORDS, BILINGUAL_40_WORDS, Config
)

import tkinter as tk
from tkinter import ttk, messagebox
import sounddevice as sd

# 中文显示映射: 内部标签 zh_xxx → 中文字符
ZH_DISPLAY = {
    'zh_zero':'零','zh_one':'一','zh_two':'二','zh_three':'三','zh_four':'四',
    'zh_five':'五','zh_six':'六','zh_seven':'七','zh_eight':'八','zh_nine':'九',
    'zh_yes':'是','zh_no':'不','zh_up':'上','zh_down':'下',
    'zh_left':'左','zh_right':'右','zh_on':'开','zh_off':'关',
    'zh_stop':'停','zh_go':'走',
}

def word_display(word):
    """zh_yes → '是  yes'  (中文在前)"""
    if word.startswith('zh_'):
        return f"{ZH_DISPLAY.get(word, '?')}   {word[3:]}"
    return word


class VoiceTestGUI:
    """语音识别 GUI 测试工具"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("语音识别泛化能力测试")
        self.root.geometry("700x600")
        self.root.resizable(False, False)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.SR = Config.SAMPLE_RATE
        self.DURATION = 1.5
        self.is_recording = False

        # 数据
        self.model = None
        self.word_list = []
        self.test_words = []
        self.current_idx = 0
        self.results = []

        # 统计
        self.correct = self.total = 0
        self.en_correct = self.en_total = 0
        self.zh_correct = self.zh_total = 0

        # 保存
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.rec_dir = Path(__file__).parent.parent / 'recordings' / self.timestamp

        # 构建界面 —— 先显示模型选择
        self._build_select_screen()

    # ================================================================
    #  模型选择界面
    # ================================================================
    def _build_select_screen(self):
        """启动界面: 选择模型"""
        for w in self.root.winfo_children():
            w.destroy()

        ttk.Label(self.root, text="语音识别泛化能力测试",
                  font=('Arial', 20, 'bold')).pack(pady=25)
        ttk.Label(self.root, text="请选择要测试的模型",
                  font=('Arial', 12)).pack(pady=5)

        # 扫描可用模型
        models_root = Path(__file__).parent.parent / 'models'
        v1_models = list(models_root.glob('v1_english_20/*/best_model.pth'))
        v2_models = list(models_root.glob('v2_bilingual_40/*/best_model.pth'))

        frame = ttk.Frame(self.root, padding=20)
        frame.pack()

        row = 0
        ttk.Label(frame, text="V1 — 英文 20 类", font=('Arial', 13, 'bold')).grid(
            row=row, column=0, sticky='w', pady=(15, 5))
        row += 1
        for mp in v1_models:
            run = mp.parent.name
            fr_file = mp.parent / 'final_results.txt'
            acc = ''
            if fr_file.exists():
                for l in open(fr_file).readlines():
                    if 'Test Accuracy' in l:
                        acc = f"  ({l.split(':')[1].strip()})"
            btn = ttk.Button(frame, text=f"  {run}{acc}",
                              command=lambda p=str(mp): self._on_model_selected(p))
            btn.grid(row=row, column=0, sticky='w', padx=20, pady=2)
            row += 1

        ttk.Label(frame, text="V2/V3/V4 — 中英 40 类", font=('Arial', 13, 'bold')).grid(
            row=row, column=0, sticky='w', pady=(15, 5))
        row += 1
        for mp in v2_models:
            run = mp.parent.name
            fr_file = mp.parent / 'final_results.txt'
            acc = ''
            if fr_file.exists():
                for l in open(fr_file).readlines():
                    if 'Test Accuracy' in l:
                        acc = f"  ({l.split(':')[1].strip()})"
            btn = ttk.Button(frame, text=f"  {run}{acc}",
                              command=lambda p=str(mp): self._on_model_selected(p))
            btn.grid(row=row, column=0, sticky='w', padx=20, pady=2)
            row += 1

        ttk.Label(self.root, text="或输入自定义模型路径:", font=('Arial', 9)).pack(pady=(15, 5))
        path_frame = ttk.Frame(self.root)
        path_frame.pack()
        self.path_entry = ttk.Entry(path_frame, width=55, font=('Arial', 9))
        self.path_entry.pack(side='left', padx=5)
        ttk.Button(path_frame, text="加载", command=self._load_custom).pack(side='left')

    def _on_model_selected(self, model_path):
        """用户选了一个模型"""
        self._load_model_and_go(model_path)

    def _load_custom(self):
        path = self.path_entry.get().strip()
        if path and Path(path).exists():
            self._load_model_and_go(path)
        else:
            messagebox.showerror("错误", "模型路径不存在")

    def _load_model_and_go(self, model_path):
        """加载模型并进入测试词汇选择"""
        cp = Path(model_path)
        config_path = cp.parent / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                nc = json.load(f).get('NUM_CLASSES', 20)
        else:
            nc = 40 if 'bilingual' in str(cp) else 20

        self.model = create_model('standard', num_classes=nc)
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        self.model.to(self.device)
        self.model.eval()

        self.word_list = list(BILINGUAL_40_WORDS if nc >= 40 else GSC_20_WORDS[:nc])
        self.model_version = 'bilingual' if nc >= 40 else 'english'
        self.num_classes = nc

        self.rec_dir.mkdir(parents=True, exist_ok=True)

        # 诊断: 验证中文类别是否在词表中
        zh_count = sum(1 for w in self.word_list if w.startswith('zh_'))
        print(f"[INFO] Model: {cp.parent.name}  |  Classes: {nc}  "
              f"|  EN: {nc-zh_count}  |  ZH: {zh_count}")
        if zh_count > 0:
            print(f"[INFO] Chinese labels OK: {[word_display(w) for w in self.word_list[-3:]]}")

        self._build_word_select_screen()

    # ================================================================
    #  词汇选择界面
    # ================================================================
    def _build_word_select_screen(self):
        for w in self.root.winfo_children():
            w.destroy()

        en_words = [w for w in self.word_list if not w.startswith('zh_')]
        zh_words = [w for w in self.word_list if w.startswith('zh_')]
        is_bilingual = len(zh_words) > 0

        ttk.Label(self.root, text="选择测试词汇",
                  font=('Arial', 18, 'bold')).pack(pady=20)

        info = f"已加载: {self.num_classes} 类模型"
        if is_bilingual:
            info += (f"\n英文标签 (0-19): {', '.join(self.word_list[:5])} ... "
                     f"{self.word_list[19]}")
            info += (f"\n中文标签 (20-39): {word_display(self.word_list[20])}, "
                     f"{word_display(self.word_list[21])}, ... "
                     f"{word_display(self.word_list[39])}")
            info += ("\n\n⚠ 提示: 中文训练数据来自 TTS 合成语音, 真人中文识别率可能偏低。"
                     "\n   如果 Top-5 始终无中文结果, 说明模型对真人中文泛化不足。")
        ttk.Label(self.root, text=info, font=('Arial', 9),
                  foreground='#546E7A', justify='left').pack(pady=8)

        btn_frame = ttk.Frame(self.root, padding=20)
        btn_frame.pack()

        ttk.Button(btn_frame, text="全部词汇 (随机顺序)",
                    command=lambda: self._start_test(list(self.word_list))).pack(
            pady=5, fill='x')

        ttk.Button(btn_frame, text="仅英文词汇",
                    command=lambda: self._start_test(en_words)).pack(
            pady=5, fill='x')

        if is_bilingual:
            ttk.Button(btn_frame, text="仅中文词汇",
                        command=lambda: self._start_test(zh_words)).pack(
                pady=5, fill='x')

        ttk.Separator(btn_frame, orient='horizontal').pack(fill='x', pady=10)

        ttk.Label(btn_frame, text="自定义测试词 (空格分隔):",
                  font=('Arial', 9)).pack()
        custom_frame = ttk.Frame(btn_frame)
        custom_frame.pack(pady=5)
        self.custom_entry = ttk.Entry(custom_frame, width=40, font=('Arial', 10))
        self.custom_entry.pack(side='left', padx=5)
        ttk.Button(custom_frame, text="开始", command=self._start_custom).pack(side='left')

        ttk.Label(btn_frame, text="可用词汇: " + ", ".join(self.word_list),
                  font=('Arial', 8), wraplength=600).pack(pady=10)

    def _start_custom(self):
        text = self.custom_entry.get().strip()
        if not text:
            return
        words = [w.strip() for w in text.split()]
        invalid = [w for w in words if w not in self.word_list]
        if invalid:
            messagebox.showerror("错误", f"未知词汇: {invalid}")
            return
        self._start_test(words)

    def _start_test(self, words):
        self.test_words = list(words)
        np.random.shuffle(self.test_words)
        self.current_idx = 0
        self.results = []
        self.correct = self.total = 0
        self.en_correct = self.en_total = 0
        self.zh_correct = self.zh_total = 0
        self._build_test_screen()
        self._show_word()

    # ================================================================
    #  测试主界面
    # ================================================================
    def _build_test_screen(self):
        for w in self.root.winfo_children():
            w.destroy()

        self.root.configure(bg='#FAFAFA')

        # 顶部标题栏
        top = tk.Frame(self.root, bg='#1565C0', height=50)
        top.pack(fill='x')
        top.pack_propagate(False)
        tk.Label(top, text="语音识别泛化能力测试", font=('Microsoft YaHei UI', 14, 'bold'),
                 fg='white', bg='#1565C0').pack(pady=12)

        # 当前词卡片
        card = tk.Frame(self.root, bg='white', relief='solid', bd=1)
        card.pack(fill='x', padx=25, pady=(15, 5))

        tk.Label(card, text="请说出", font=('Microsoft YaHei UI', 11),
                 fg='#78909C', bg='white').pack(pady=(12, 0))
        self.word_label = tk.Label(card, text="", font=('Microsoft YaHei UI', 42, 'bold'),
                                    bg='white')
        self.word_label.pack(pady=(0, 2))
        self.lang_label = tk.Label(card, text="", font=('Microsoft YaHei UI', 10, 'bold'),
                                    bg='white')
        self.lang_label.pack(pady=(0, 5))

        # 进度条
        prog_frame = tk.Frame(card, bg='white')
        prog_frame.pack(fill='x', padx=30, pady=(5, 12))
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(prog_frame, variable=self.progress_var,
                                         maximum=100, length=500,
                                         style='Green.Horizontal.TProgressbar')
        self.progress.pack(side='left', fill='x', expand=True)
        self.progress_text = tk.Label(prog_frame, text="0/0", font=('Microsoft YaHei UI', 9),
                                       fg='#90A4AE', bg='white', width=8)
        self.progress_text.pack(side='right', padx=(8, 0))

        # 状态
        self.status_label = tk.Label(self.root, text="点击【录制】开始",
                                      font=('Microsoft YaHei UI', 11),
                                      fg='#78909C', bg='#FAFAFA')
        self.status_label.pack(pady=(8, 2))

        # 按钮栏
        btn_frame = tk.Frame(self.root, bg='#FAFAFA')
        btn_frame.pack(pady=3)

        self.record_btn = tk.Button(btn_frame, text="🎤  录制",
                                     font=('Microsoft YaHei UI', 13, 'bold'),
                                     command=self._do_record, bg='#1565C0', fg='white',
                                     activebackground='#0D47A1', activeforeground='white',
                                     relief='flat', padx=28, pady=6, cursor='hand2')
        self.record_btn.pack(side='left', padx=4)

        self.next_btn = tk.Button(btn_frame, text="▶  下一个",
                                   font=('Microsoft YaHei UI', 11),
                                   command=self._go_next, state='disabled',
                                   bg='#43A047', fg='white',
                                   activebackground='#2E7D32', activeforeground='white',
                                   relief='flat', padx=18, pady=6, cursor='hand2')
        self.next_btn.pack(side='left', padx=4)

        self.skip_btn = tk.Button(btn_frame, text="跳过",
                                   font=('Microsoft YaHei UI', 10),
                                   command=self._skip, bg='#ECEFF1', fg='#546E7A',
                                   relief='flat', padx=14, pady=6, cursor='hand2')
        self.skip_btn.pack(side='left', padx=4)

        self.retry_btn = tk.Button(btn_frame, text="↩ 重录",
                                    font=('Microsoft YaHei UI', 10),
                                    command=self._retry, bg='#ECEFF1', fg='#546E7A',
                                    relief='flat', padx=14, pady=6, cursor='hand2')
        self.retry_btn.pack(side='left', padx=4)

        # 结果区
        pred_frame = tk.LabelFrame(self.root, text=" 识别结果 ",
                                    font=('Microsoft YaHei UI', 10, 'bold'),
                                    fg='#37474F', bg='#FAFAFA',
                                    relief='solid', bd=1, padx=8, pady=6)
        pred_frame.pack(fill='x', padx=25, pady=(10, 3))

        self.pred_text = tk.Text(pred_frame, height=5, width=65,
                                  font=('Cascadia Code', 10),
                                  bg='#263238', fg='#ECEFF1',
                                  insertbackground='white',
                                  relief='flat', borderwidth=0,
                                  state='disabled', padx=10, pady=8)
        self.pred_text.pack(fill='x')

        # 统计栏 — 单行
        stats_frame = tk.Frame(self.root, bg='#ECEFF1')
        stats_frame.pack(fill='x', padx=25, pady=(8, 10))
        self.stats_text = tk.Label(stats_frame, text="— 等待测试 —",
                                    font=('Microsoft YaHei UI', 11, 'bold'),
                                    fg='#37474F', bg='#ECEFF1',
                                    anchor='center', padx=12, pady=6)
        self.stats_text.pack(fill='x')

        # 样式
        style = ttk.Style()
        style.configure('Green.Horizontal.TProgressbar',
                        troughcolor='#E0E0E0', background='#43A047',
                        thickness=8)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ================================================================
    #  测试逻辑
    # ================================================================
    def _show_word(self):
        if self.current_idx >= len(self.test_words):
            self._show_summary()
            return

        word = self.test_words[self.current_idx]
        is_zh = word.startswith('zh_')

        self.word_label.config(text=word_display(word),
                                foreground='#D32F2F' if is_zh else '#1565C0')
        self.lang_label.config(text='🇨🇳 中文  (请说中文)' if is_zh else '🇬🇧 English  (say in English)')
        self.progress_var.set(self.current_idx / len(self.test_words) * 100)
        self.progress_text.config(text=f"{self.current_idx+1} / {len(self.test_words)}")

        self.status_label.config(text="点击【录制】开始", foreground='#555')
        self.record_btn.config(state='normal')
        self.next_btn.config(state='disabled')
        self.skip_btn.config(state='normal')
        self.retry_btn.config(state='normal' if self.current_idx > 0 else 'disabled')

        self.pred_text.config(state='normal')
        self.pred_text.delete(1.0, tk.END)
        self.pred_text.insert(tk.END, "(此处将显示识别结果)")
        self.pred_text.config(state='disabled')

    def _set_buttons_busy(self):
        for btn in [self.record_btn, self.next_btn, self.skip_btn, self.retry_btn]:
            btn.config(state='disabled')

    def _do_record(self):
        if self.is_recording:
            return
        self.is_recording = True
        self._set_buttons_busy()
        self.status_label.config(text="🔴 录音中 (1.5秒)...", foreground='#C62828')
        self.pred_text.config(state='normal')
        self.pred_text.delete(1.0, tk.END)
        self.pred_text.insert(tk.END, "...")
        self.pred_text.config(state='disabled')
        self.root.update()
        threading.Thread(target=self._record_thread, daemon=True).start()

    def _preprocess_audio(self, audio):
        """与 TTS 训练数据一致的音频预处理 (增强版)"""
        # 1. 去直流分量
        audio = audio - np.mean(audio)

        # 2. 去首尾静音 (更激进, TTS 数据几乎无静音)
        energy = np.abs(audio)
        threshold = 0.03 * energy.max()
        active = np.where(energy > threshold)[0]
        if len(active) > 10:
            # 首尾各留 50ms 的轻微缓冲
            pad_samples = int(0.05 * self.SR)
            start = max(0, active[0] - pad_samples)
            end = min(len(audio), active[-1] + pad_samples)
            audio = audio[start:end]

        # 3. 预加重 (与训练数据一致: H(z)=1-0.97z⁻¹)
        audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

        # 4. 音量归一化 (RMS 归一化, 与 TTS 一致)
        rms = np.sqrt(np.mean(audio**2))
        target_rms = 0.12
        if rms > 1e-6:
            audio = audio * (target_rms / rms)

        # 5. 峰值限幅 (避免削波)
        peak = np.abs(audio).max()
        if peak > 0.95:
            audio = audio * (0.95 / peak)

        # 6. 居中裁剪/补零到 1 秒
        target = int(self.SR * Config.AUDIO_DURATION)
        if len(audio) > target:
            start = (len(audio) - target) // 2
            audio = audio[start:start + target]
        else:
            padded = np.zeros(target, dtype=np.float32)
            off = (target - len(audio)) // 2
            padded[off:off+len(audio)] = audio
            audio = padded

        return audio

    def _predict_with_tta(self, audio):
        """测试时增强 (TTA): 原音频 + 变体 → 投票"""
        variants = [audio]  # 原始

        # 轻微音调偏移 (模拟不同音高)
        try:
            import librosa
            for steps in [-1, 1]:  # ±1 半音
                shifted = librosa.effects.pitch_shift(
                    audio, sr=self.SR, n_steps=steps)
                variants.append(shifted)
        except Exception:
            pass

        # 轻微时间拉伸 (模拟不同语速)
        try:
            import librosa
            for rate in [0.9, 1.1]:
                stretched = librosa.effects.time_stretch(audio, rate=rate)
                if len(stretched) > len(audio):
                    stretched = stretched[:len(audio)]
                else:
                    stretched = np.pad(stretched, (0, len(audio)-len(stretched)))
                variants.append(stretched)
        except Exception:
            pass

        # 对每个变体做推理, 取平均 logits
        all_logits = []
        for var in variants:
            mel = compute_mel_spectrogram(var, sr=self.SR, n_mels=Config.N_MELS)
            mel = normalize_spectrogram(mel, method='global')
            tensor = torch.from_numpy(mel).float().unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
            all_logits.append(logits)

        # 平均 logits → softmax → top-k
        avg_logits = torch.stack(all_logits).mean(dim=0)
        probs = F.softmax(avg_logits, dim=1).squeeze(0)
        top5 = torch.topk(probs, k=min(5, len(self.word_list)))
        return [(self.word_list[i], float(probs[i])) for i in top5.indices]

    def _record_thread(self):
        # 录音
        try:
            audio = sd.rec(int(self.SR * self.DURATION),
                          samplerate=self.SR, channels=1, dtype='float32')
            sd.wait()
            audio = audio.flatten()
        except Exception as e:
            self.root.after(0, lambda: self._on_error(f"录音失败: {e}"))
            return

        # 预处理 (与训练集一致: 去静音 + 音量归一化 + 定长)
        audio = self._preprocess_audio(audio)

        # 推理 (TTA: 多版本投票)
        word = self.test_words[self.current_idx]
        is_zh = word.startswith('zh_')
        try:
            preds = self._predict_with_tta(audio)
        except Exception as e:
            self.root.after(0, lambda: self._on_error(f"推理失败: {e}"))
            return

        # 保存录音
        rec_path = self.rec_dir / f"{word}_{self.current_idx:03d}.wav"
        try:
            import soundfile as sf
            sf.write(str(rec_path), audio, self.SR)
        except Exception:
            pass

        # 统计
        top_word, top_conf = preds[0]
        is_correct = (top_word == word)
        self.total += 1
        if is_correct:
            self.correct += 1
        if is_zh:
            self.zh_total += 1
            if is_correct: self.zh_correct += 1
        else:
            self.en_total += 1
            if is_correct: self.en_correct += 1

        self.results.append({
            'expected': word, 'predicted': top_word, 'confidence': top_conf,
            'correct': is_correct,
            'lang': 'ZH' if is_zh else 'EN',
            'top5': [{'word': w, 'conf': c} for w, c in preds],
        })

        self.root.after(0, lambda: self._on_result(word, preds, is_correct))

    def _beautify_result(self, expected, preds):
        """生成带颜色的识别结果文本"""
        lines = []
        max_bar = 28
        for i, (w, conf) in enumerate(preds):
            bar_len = int(conf * max_bar)
            if w == expected:
                # 正确: 绿色
                bar = '█' * bar_len + '░' * (max_bar - bar_len)
                tag = f"  ✅  [{i+1}] {word_display(w):<24s} │{bar}│ {conf*100:5.1f}%  ⇐ 正确!"
            elif conf > 0.3:
                # 高置信错误: 橙色
                bar = '▓' * bar_len + '░' * (max_bar - bar_len)
                tag = f"     [{i+1}] {word_display(w):<24s} │{bar}│ {conf*100:5.1f}%"
            else:
                # 低置信: 灰色
                bar = '▒' * bar_len + '░' * (max_bar - bar_len)
                tag = f"     [{i+1}] {word_display(w):<24s} │{bar}│ {conf*100:5.1f}%"
            lines.append(tag)
        return '\n'.join(lines)

    def _on_result(self, expected, preds, is_correct):
        self.is_recording = False

        self.pred_text.config(state='normal')
        self.pred_text.delete(1.0, tk.END)
        self.pred_text.insert(tk.END, self._beautify_result(expected, preds))
        self.pred_text.config(state='disabled')

        # 状态
        if is_correct:
            self.status_label.config(text="✅ 识别正确！点击【下一个】继续",
                                      foreground='#2E7D32')
        else:
            self.status_label.config(
                text=f"❌ 错误  期望: {expected}  实际: {preds[0][0]}  |  点击【下一个】继续",
                foreground='#C62828')

        # 统计 — 单行显示
        self._update_stats_text()

        # 按钮: 只启用【下一个】和【重录】
        self.record_btn.config(state='disabled')
        self.next_btn.config(state='normal')
        self.skip_btn.config(state='disabled')
        self.retry_btn.config(state='normal')

    def _update_stats_text(self):
        if self.total == 0:
            self.stats_text.config(text="— 等待测试 —")
            return
        acc = self.correct / self.total * 100
        parts = [f"总计  {self.correct}/{self.total}  ({acc:.1f}%)"]
        if self.en_total > 0:
            en_a = self.en_correct / max(1, self.en_total) * 100
            parts.append(f"EN  {self.en_correct}/{self.en_total}  ({en_a:.1f}%)")
        if self.zh_total > 0:
            zh_a = self.zh_correct / max(1, self.zh_total) * 100
            parts.append(f"ZH  {self.zh_correct}/{self.zh_total}  ({zh_a:.1f}%)")
        self.stats_text.config(text="    │    ".join(parts))

    def _on_error(self, msg):
        self.is_recording = False
        self.status_label.config(text=msg, foreground='#C62828')
        self.record_btn.config(state='normal')
        self.next_btn.config(state='disabled')
        self.skip_btn.config(state='normal')
        self.retry_btn.config(state='normal')

    def _go_next(self):
        self.current_idx += 1
        self._show_word()

    def _skip(self):
        word = self.test_words[self.current_idx]
        self.results.append({
            'expected': word, 'predicted': '(skipped)',
            'confidence': 0, 'correct': False,
            'lang': 'ZH' if word.startswith('zh_') else 'EN'})
        self.current_idx += 1
        self._show_word()

    def _retry(self):
        """重录当前词 (不是回退)"""
        # 撤销本次已记录的结果
        if self.results and self.results[-1].get('predicted') != '(skipped)':
            r = self.results.pop()
            self.total -= 1
            if r['correct']: self.correct -= 1
            if r['lang'] == 'ZH':
                self.zh_total -= 1
                if r['correct']: self.zh_correct -= 1
            else:
                self.en_total -= 1
                if r['correct']: self.en_correct -= 1
        elif self.results:
            self.results.pop()  # 移除 skip 记录
        # 不改变 current_idx, 重新显示当前词
        self._show_word()

    # ================================================================
    #  汇总
    # ================================================================
    def _show_summary(self):
        for w in self.root.winfo_children():
            w.destroy()

        ttk.Label(self.root, text="测试完成！",
                  font=('Arial', 22, 'bold')).pack(pady=15)

        t = max(1, self.total)
        en_a = self.en_correct / max(1, self.en_total) * 100
        zh_a = self.zh_correct / max(1, self.zh_total) * 100

        text = tk.Text(self.root, height=22, width=72, font=('Consolas', 10))
        text.insert(tk.END, f"总体: {self.correct/t*100:.1f}% ({self.correct}/{self.total})\n")
        if self.en_total > 0:
            text.insert(tk.END, f"英文: {en_a:.1f}% ({self.en_correct}/{self.en_total})\n")
        if self.zh_total > 0:
            text.insert(tk.END, f"中文: {zh_a:.1f}% ({self.zh_correct}/{self.zh_total})\n")
        text.insert(tk.END, "\n")

        per_word = defaultdict(lambda: {'c': 0, 't': 0})
        for r in self.results:
            if r.get('predicted') == '(skipped)': continue
            per_word[r['expected']]['t'] += 1
            if r['correct']: per_word[r['expected']]['c'] += 1
        for w in sorted(per_word.keys()):
            pw = per_word[w]
            a = pw['c'] / max(1, pw['t']) * 100
            bar = '#' * int(a / 5) + '-' * (20 - int(a / 5))
            text.insert(tk.END, f"  {w:<20s} {bar} {a:5.1f}% ({pw['c']}/{pw['t']})\n")
        text.config(state='disabled')
        text.pack(pady=10)

        ttk.Label(self.root, text=f"录音: recordings/{self.timestamp}/",
                  font=('Arial', 9)).pack()
        ttk.Button(self.root, text="关闭", command=self.root.destroy).pack(pady=10)

        # 保存
        report = {
            'timestamp': self.timestamp,
            'total': self.total, 'correct': self.correct,
            'overall_acc': self.correct / t * 100,
            'en_acc': en_a, 'zh_acc': zh_a,
            'per_word': {w: {'correct': pw['c'], 'total': pw['t'],
                              'acc': pw['c']/max(1,pw['t'])*100}
                         for w, pw in per_word.items()},
            'details': self.results,
        }
        with open(self.rec_dir / 'test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def _on_close(self):
        self.root.destroy()


def main():
    app = VoiceTestGUI()
    app.root.mainloop()


if __name__ == '__main__':
    main()
