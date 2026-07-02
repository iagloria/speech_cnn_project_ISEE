"""
交互式语音测试工具 — 录制并测试模型泛化能力
=============================================
功能:
  1. 引导式录制: 显示要说的词 → 按空格录音 → 实时推理
  2. 支持 V1(20类英文) / V4(40类中英) / 自定义模型
  3. 每词测试后即时显示 Top-3 预测
  4. 测试结束生成汇总报告 (JSON + 文本)
  5. 自动保存所有录音到 recordings/ 目录

用法:
  python interactive_test.py                          # 自动选最新模型
  python interactive_test.py --model latest           # 同上
  python interactive_test.py --model ../models/v2_bilingual_40/bilingual_v4/best_model.pth
  python interactive_test.py --test_all               # 测试全部 40 类
  python interactive_test.py --test_en_only           # 仅测试英文 20 类
  python interactive_test.py --test_zh_only           # 仅测试中文 20 类
  python interactive_test.py --words yes no stop go   # 自定义测试词汇
"""

import sys, time, json, argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from model import create_model
from preprocess import (
    load_and_preprocess_audio, compute_mel_spectrogram,
    normalize_spectrogram, GSC_20_WORDS, BILINGUAL_40_WORDS, Config
)

# ANSI colors
C_RESET = '\033[0m'; C_RED = '\033[91m'; C_GREEN = '\033[92m'
C_YELLOW = '\033[93m'; C_BLUE = '\033[94m'; C_CYAN = '\033[96m'
C_BOLD = '\033[1m'


def load_model(model_path, device='auto'):
    """加载模型，自动检测类别数"""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    cp = Path(model_path)
    if not cp.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # 读 config 获取 num_classes
    config_path = cp.parent / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            nc = json.load(f).get('NUM_CLASSES', 20)
    else:
        nc = 40 if 'bilingual' in str(cp) else 20

    model = create_model('standard', num_classes=nc)
    ckpt = torch.load(str(cp), map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.to(device)
    model.eval()

    word_list = BILINGUAL_40_WORDS if nc >= 40 else GSC_20_WORDS[:nc]
    return model, device, nc, word_list


def record_audio(duration=1.5, sr=16000):
    """录制音频，返回 numpy 数组 (sr,)"""
    try:
        import sounddevice as sd
    except ImportError:
        print(f"\n{C_RED}[ERROR] sounddevice not installed.{C_RESET}")
        print("  pip install sounddevice")
        return None

    print(f"  {C_YELLOW}Recording {duration}s...{C_RESET} ", end='', flush=True)
    audio = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()

    # 裁剪静音
    rms = np.sqrt(np.mean(audio**2))
    if rms < 0.001:
        print(f"{C_RED}Too quiet!{C_RESET}")
        return None

    target = int(sr * Config.AUDIO_DURATION)
    if len(audio) > target:
        audio = audio[:target]
    else:
        audio = np.pad(audio, (0, target - len(audio)))

    print(f"{C_GREEN}Done!{C_RESET}")
    return audio


def predict(model, device, audio, word_list):
    """推理：音频 → 频谱 → 模型 → Top-K 预测"""
    mel = compute_mel_spectrogram(audio, sr=Config.SAMPLE_RATE, n_mels=Config.N_MELS)
    mel = normalize_spectrogram(mel, method='global')
    tensor = torch.from_numpy(mel).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).squeeze(0)

    topk = torch.topk(probs, k=min(5, len(word_list)))
    return [(word_list[i], float(probs[i])) for i in topk.indices]


def print_header(title):
    print(f"\n{C_BOLD}{C_CYAN}{'='*60}{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}  {title}{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}{'='*60}{C_RESET}\n")


def print_predictions(predictions, true_label=None):
    """打印 Top-K 预测结果"""
    for i, (word, conf) in enumerate(predictions):
        bar_len = int(conf * 40)
        bar = '#' * bar_len + '-' * (40 - bar_len)
        marker = ' <-- CORRECT' if true_label and word == true_label else ''
        color = C_GREEN if (true_label and word == true_label) else (
            C_YELLOW if conf > 0.3 else C_RED if conf < 0.1 else C_RESET)
        correct_mark = ' <<' if true_label and word == true_label else ''
        print(f"  [{i+1}] {color}{word:<18s} {bar} {conf*100:5.1f}%{correct_mark}{C_RESET}")


def interactive_test(args):
    """主交互流程"""
    # 加载模型
    if args.model == 'latest' or args.model is None:
        models_root = Path(__file__).parent.parent / 'models'
        all_models = sorted(models_root.glob('*/*/best_model.pth'), reverse=True)
        if not all_models:
            print("No models found. Train first."); sys.exit(1)
        model_path = str(all_models[0])
    else:
        model_path = args.model

    print_header("LOADING MODEL")
    model, device, num_classes, word_list = load_model(model_path)
    print(f"  Model: {Path(model_path).parent.parent.name}/{Path(model_path).parent.name}")
    print(f"  Classes: {num_classes}  |  Device: {device}\n")
    print(f"  Vocabulary: {', '.join(word_list[:10])} ... ({num_classes} words)")

    # 确定测试词汇
    if args.words:
        test_words = args.words
        # 验证是否在词汇表中
        invalid = [w for w in test_words if w not in word_list]
        if invalid:
            print(f"\n{C_RED}[ERROR] Unknown words: {invalid}{C_RESET}")
            sys.exit(1)
    elif args.test_en_only:
        test_words = [w for w in word_list if not w.startswith('zh_')]
    elif args.test_zh_only:
        test_words = [w for w in word_list if w.startswith('zh_')]
    elif args.test_all:
        test_words = list(word_list)
    else:
        # 默认: 用户自选
        test_words = list(word_list)

    # 随机打乱顺序 (避免顺序偏差)
    if not args.keep_order:
        np.random.shuffle(test_words)

    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    rec_dir = Path(__file__).parent.parent / 'recordings' / timestamp
    rec_dir.mkdir(parents=True, exist_ok=True)

    print_header("INTERACTIVE VOICE TEST")
    print(f"  Words to test: {len(test_words)}")
    print(f"  Recordings saved to: {rec_dir}")
    print(f"\n  {C_BOLD}Instructions:{C_RESET}")
    print(f"    1. You will see a word on screen")
    print(f"    2. Press {C_YELLOW}Enter{C_RESET} to start recording")
    print(f"    3. Say the word clearly (keep silence before/after)")
    print(f"    4. See the prediction result")
    print(f"    5. Press {C_YELLOW}Enter{C_RESET} to continue to next word")
    print(f"    6. Type {C_RED}'q'{C_RESET} to quit at any time")
    print(f"    7. Type {C_YELLOW}'r'{C_RESET} to re-record current word")

    # 当前可用麦克风
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devs = [d for d in devices if d['max_input_channels'] > 0]
        if input_devs:
            print(f"\n  {C_GREEN}Microphone detected:{C_RESET} {input_devs[0]['name']}")
        else:
            print(f"\n  {C_RED}No microphone detected! Check your audio settings.{C_RESET}")
            sys.exit(1)
    except Exception:
        pass

    # 开始测试
    results = []
    correct = 0
    total = 0

    # 按语言分开统计
    en_correct = en_total = zh_correct = zh_total = 0

    i = 0
    while i < len(test_words):
        word = test_words[i]
        is_zh = word.startswith('zh_')
        display_word = word.replace('zh_', '') + (' (中文)' if is_zh else ' (EN)')

        # 显示进度
        progress = f"[{i+1}/{len(test_words)}]"
        if total > 0:
            acc_str = f"Acc: {correct/total*100:.1f}% ({correct}/{total})"
        else:
            acc_str = ""

        print(f"\n{C_BOLD}{'─'*60}{C_RESET}")
        print(f"  {C_BOLD}{progress} Say: {C_YELLOW}{display_word}{C_RESET}       {acc_str}")

        cmd = input(f"  {C_CYAN}[Enter=Record | r=Retry | q=Quit]{C_RESET} ").strip().lower()

        if cmd == 'q':
            print(f"\n  {C_YELLOW}Test interrupted. Saving results...{C_RESET}")
            break
        if cmd == 'r' and i > 0:
            i -= 1  # 回退一步重录
            continue

        # 录制
        audio = record_audio()
        if audio is None:
            print(f"  {C_RED}Recording failed, try again.{C_RESET}")
            continue

        # 保存录音
        rec_path = rec_dir / f"{word}_{i:03d}.wav"
        try:
            import soundfile as sf
            sf.write(str(rec_path), audio, Config.SAMPLE_RATE)
        except Exception:
            pass

        # 推理
        predictions = predict(model, device, audio, word_list)
        top_word, top_conf = predictions[0]

        total += 1
        is_correct = (top_word == word)
        if is_correct:
            correct += 1
        if is_zh:
            zh_total += 1
            if is_correct: zh_correct += 1
        else:
            en_total += 1
            if is_correct: en_correct += 1

        # 显示结果
        print()
        print_predictions(predictions, true_label=word)

        status = f"{C_GREEN}CORRECT!{C_RESET}" if is_correct else f"{C_RED}WRONG (should be: {word}){C_RESET}"
        print(f"\n  {status}")

        # 保存结果
        results.append({
            'index': i,
            'expected': word,
            'predicted': top_word,
            'confidence': top_conf,
            'correct': is_correct,
            'language': 'ZH' if is_zh else 'EN',
            'top5': [{'word': w, 'confidence': c} for w, c in predictions],
        })

        i += 1

    # 汇总报告
    print_header("TEST SUMMARY")

    if total == 0:
        print("  No tests completed.")
        return

    en_acc = en_correct / en_total * 100 if en_total > 0 else 0
    zh_acc = zh_correct / zh_total * 100 if zh_total > 0 else 0
    overall = correct / total * 100

    print(f"  {C_BOLD}Overall:{C_RESET}        {overall:.1f}% ({correct}/{total})")
    if en_total > 0:
        print(f"  {C_BOLD}English:{C_RESET}        {en_acc:.1f}% ({en_correct}/{en_total})")
    if zh_total > 0:
        print(f"  {C_BOLD}Chinese:{C_RESET}        {zh_acc:.1f}% ({zh_correct}/{zh_total})")

    # 按类别汇总
    print(f"\n  {C_BOLD}Per-word breakdown:{C_RESET}")
    per_word = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in results:
        w = r['expected']
        per_word[w]['total'] += 1
        if r['correct']:
            per_word[w]['correct'] += 1
    for w in sorted(per_word.keys()):
        pw = per_word[w]
        acc = pw['correct'] / pw['total'] * 100
        color = C_GREEN if acc >= 80 else C_YELLOW if acc >= 50 else C_RED
        bar = '#' * int(acc / 5) + '-' * (20 - int(acc / 5))
        print(f"    {w:<18s} {color}{bar} {acc:5.1f}% ({pw['correct']}/{pw['total']}){C_RESET}")

    # 保存报告
    report = {
        'timestamp': timestamp,
        'model': str(model_path),
        'total': total,
        'correct': correct,
        'overall_acc': overall,
        'en_acc': en_acc,
        'zh_acc': zh_acc,
        'en_total': en_total,
        'zh_total': zh_total,
        'per_word': {w: {'correct': pw['correct'], 'total': pw['total'],
                          'acc': pw['correct']/pw['total']*100}
                     for w, pw in per_word.items()},
        'details': results,
    }

    report_path = rec_dir / 'test_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 文本报告
    txt_path = rec_dir / 'test_report.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  VOICE RECOGNITION TEST REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Time: {timestamp}\n\n")
        f.write(f"Overall: {overall:.1f}% ({correct}/{total})\n")
        f.write(f"English: {en_acc:.1f}% ({en_correct}/{en_total})\n")
        f.write(f"Chinese: {zh_acc:.1f}% ({zh_correct}/{zh_total})\n\n")
        f.write("Per-word:\n")
        for w in sorted(per_word.keys()):
            pw = per_word[w]
            f.write(f"  {w:<20s} {pw['correct']/pw['total']*100:5.1f}% "
                    f"({pw['correct']}/{pw['total']})\n")
        f.write("\nDetails:\n")
        for r in results:
            mark = 'OK' if r['correct'] else 'XX'
            f.write(f"  [{mark}] {r['expected']:<20s} -> {r['predicted']:<20s} "
                    f"({r['confidence']*100:.1f}%)\n")

    print(f"\n  {C_GREEN}Report saved:{C_RESET}")
    print(f"    {report_path}")
    print(f"    {txt_path}")
    print(f"  {C_GREEN}Recordings:{C_RESET} {rec_dir}/")
    print(f"\n{C_BOLD}{C_CYAN}{'='*60}{C_RESET}")


def parse_args():
    p = argparse.ArgumentParser(
        description='Interactive Voice Recognition Test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python interactive_test.py                           # auto latest model, all classes
  python interactive_test.py --test_en_only            # English only
  python interactive_test.py --test_zh_only            # Chinese only
  python interactive_test.py --words yes no stop go   # custom words
  python interactive_test.py --model ../models/v1_english_20/20260611_220716/best_model.pth
        """
    )
    p.add_argument('--model', default='latest', help='Model path or "latest"')
    p.add_argument('--words', nargs='+', default=None, help='Custom test words')
    p.add_argument('--test_all', action='store_true', help='Test ALL classes')
    p.add_argument('--test_en_only', action='store_true')
    p.add_argument('--test_zh_only', action='store_true')
    p.add_argument('--keep_order', action='store_true', help='Keep word order')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    try:
        interactive_test(args)
    except KeyboardInterrupt:
        print(f"\n\n{C_YELLOW}Test interrupted.{C_RESET}\n")
