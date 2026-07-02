"""
频谱对比诊断工具 — 真人录音 vs TTS 训练数据差异分析
=====================================================
录音后与 TTS 同词频谱图并排对比, 直观判断分布是否匹配。

用法:
  python compare_spectrograms.py --word 是

作者: Speech CNN Project
"""

import sys, argparse
from pathlib import Path
import numpy as np
import librosa, sounddevice as sd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import compute_mel_spectrogram, normalize_spectrogram, Config

ZH_DISPLAY = {'zero':'零','one':'一','two':'二','three':'三','four':'四',
    'five':'五','six':'六','seven':'七','eight':'八','nine':'九',
    'yes':'是','no':'不','up':'上','down':'下','left':'左','right':'右',
    'on':'开','off':'关','stop':'停','go':'走'}

def record_audio(duration=2.0, sr=16000):
    print(f"  Recording {duration}s ... ", end='', flush=True)
    audio = sd.rec(int(sr*duration), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()
    # 去静音
    energy = np.abs(audio); active = np.where(energy > 0.02*energy.max())[0]
    if len(active)>10: audio = audio[active[0]:active[-1]+1]
    # 定长
    target = int(sr*Config.AUDIO_DURATION)
    if len(audio)>target: audio=audio[:target]
    else: audio=np.pad(audio,(0,target-len(audio)))
    print("Done!")
    return audio

def generate_tts(word_en, word_zh, sr=16000):
    """用 edge-tts 生成对比音频"""
    import edge_tts, asyncio, tempfile
    mp3 = tempfile.mktemp(suffix='.mp3')
    async def _do():
        comm = edge_tts.Communicate(word_zh, 'zh-CN-XiaoxiaoNeural', rate='+0%')
        await comm.save(mp3)
    asyncio.run(_do())
    audio, _ = librosa.load(mp3, sr=sr, mono=True)
    Path(mp3).unlink()
    # 定长
    target = int(sr*Config.AUDIO_DURATION)
    if len(audio)>target: audio=audio[:target]
    else: audio=np.pad(audio,(0,target-len(audio)))
    return audio

def main():
    parser = argparse.ArgumentParser(description='Compare real vs TTS spectrograms')
    parser.add_argument('--word', type=str, required=True,
                        help='English word to test (e.g. yes, zero)')
    args = parser.parse_args()
    word_en = args.word.lower()
    word_en = {'shi':'yes','bu':'no','shang':'up','xia':'down','zuo':'left',
               'you':'right','kai':'on','guan':'off','ting':'stop','zou':'go',
               'ling':'zero','yi':'one','er':'two','san':'three','si':'four',
               'wu':'five','liu':'six','qi':'seven','ba':'eight','jiu':'nine'}.get(word_en, word_en)

    if word_en not in ZH_DISPLAY:
        print(f"Unknown word: {word_en}. Try: yes, no, zero, stop, ...")
        sys.exit(1)

    word_zh = ZH_DISPLAY[word_en]

    print(f"\nWord: {word_en} / {word_zh}")
    print()

    # 1. 录音
    print("[Step 1] Record your voice (say the word)")
    input("  Press Enter when ready...")
    real_audio = record_audio()

    # 2. 生成 TTS
    print("[Step 2] Generating TTS audio...")
    tts_audio = generate_tts(word_en, word_zh)

    # 3. 计算频谱
    real_spec = compute_mel_spectrogram(real_audio)
    tts_spec = compute_mel_spectrogram(tts_audio)

    # 4. 对比绘图
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # 波形
    t = np.arange(len(real_audio))/16000
    axes[0,0].plot(t, real_audio, linewidth=0.4, color='#1565C0')
    axes[0,0].set_title('Your Recording (waveform)')
    axes[0,0].set_xlabel('Time (s)'); axes[0,0].set_ylabel('Amplitude')
    t2 = np.arange(len(tts_audio))/16000
    axes[1,0].plot(t2, tts_audio, linewidth=0.4, color='#E64A19')
    axes[1,0].set_title('TTS Audio (waveform)')
    axes[1,0].set_xlabel('Time (s)'); axes[1,0].set_ylabel('Amplitude')

    # 频谱
    im1 = axes[0,1].imshow(real_spec, aspect='auto', origin='lower', cmap='magma')
    axes[0,1].set_title('Your Recording (mel spectrogram)')
    axes[0,1].set_xlabel('Time frames'); axes[0,1].set_ylabel('Mel bands')
    plt.colorbar(im1, ax=axes[0,1])
    im2 = axes[1,1].imshow(tts_spec, aspect='auto', origin='lower', cmap='magma')
    axes[1,1].set_title('TTS (mel spectrogram)')
    axes[1,1].set_xlabel('Time frames'); axes[1,1].set_ylabel('Mel bands')
    plt.colorbar(im2, ax=axes[1,1])

    # 差异
    real_norm = normalize_spectrogram(real_spec, 'global')
    tts_norm = normalize_spectrogram(tts_spec, 'global')
    diff = real_norm - tts_norm
    im3 = axes[0,2].imshow(diff, aspect='auto', origin='lower', cmap='RdBu_r',
                            vmin=-3, vmax=3)
    axes[0,2].set_title('Difference (Real - TTS)')
    axes[0,2].set_xlabel('Time frames'); axes[0,2].set_ylabel('Mel bands')
    plt.colorbar(im3, ax=axes[0,2])

    # 统计
    mse = np.mean((real_norm - tts_norm)**2)
    corr = np.corrcoef(real_norm.flatten(), tts_norm.flatten())[0,1]
    axes[1,2].axis('off')
    axes[1,2].text(0.1, 0.9,
        f"Similarity Analysis\n\n"
        f"MSE: {mse:.4f}\n"
        f"Correlation: {corr:.3f}\n\n"
        f"{'GOOD match — model should work' if corr>0.6 else 'POOR match — explains poor accuracy' if corr>0.3 else 'VERY DIFFERENT — distribution mismatch!'}",
        fontsize=12, fontfamily='monospace', verticalalignment='top')

    plt.suptitle(f'Spectrogram Comparison: "{word_en}" / "{word_zh}"',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    out = f'../results/comparison/spectrogram_{word_en}.png'
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[DONE] Saved to: {out}")
    print(f"  MSE={mse:.4f}  |  Correlation={corr:.3f}")
    if corr < 0.5:
        print(f"\n  ⚠  Conclusion: Your voice and TTS data are SIGNIFICANTLY different.")
        print(f"     This explains why Chinese recognition fails.")
        print(f"  → Solution: Record your own Chinese training data (10-20 samples/class)")
        print(f"     and fine-tune the model with --resume")

if __name__ == '__main__':
    main()
