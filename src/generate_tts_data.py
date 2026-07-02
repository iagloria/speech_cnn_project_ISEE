"""
TTS 中文语音合成 — 恢复初版简洁架构
=====================================
使用 edge-tts 合成中文语音数据。
每条独立调用 asyncio.run(), 简单稳定。

用法:
  python generate_tts_data.py                    # 默认 1,800 条
  python generate_tts_data.py --num_repeats 50   # ~18,000 条
"""

import os, sys, argparse, time, json, asyncio
from pathlib import Path

import numpy as np

# ============================================================
DIGITS_ZH = ['零','一','二','三','四','五','六','七','八','九']
DIGITS_EN = ['zero','one','two','three','four','five','six','seven','eight','nine']
COMMANDS_ZH = ['是','不','上','下','左','右','开','关','停','走']
COMMANDS_EN = ['yes','no','up','down','left','right','on','off','stop','go']
ALL_ZH = DIGITS_ZH + COMMANDS_ZH
ALL_EN = DIGITS_EN + COMMANDS_EN

VOICES = [
    'zh-CN-XiaoxiaoNeural','zh-CN-XiaoyiNeural','zh-CN-XiaochenNeural',
    'zh-CN-YunxiNeural','zh-CN-YunyangNeural','zh-CN-YunjianNeural',
]
RATES = [0.85, 1.0, 1.15]
SAMPLE_RATE = 16000
DURATION_SEC = 1.0
DEFAULT_REPEATS = 5

ZH_PINYIN = {
    '零':'ling','一':'yi','二':'er','三':'san','四':'si',
    '五':'wu','六':'liu','七':'qi','八':'ba','九':'jiu',
    '是':'shi','不':'bu','上':'shang','下':'xia','左':'zuo',
    '右':'you','开':'kai','关':'guan','停':'ting','走':'zou',
}


def convert_mp3_to_wav(mp3_path, wav_path):
    import librosa, soundfile as sf
    audio, _ = librosa.load(mp3_path, sr=SAMPLE_RATE, mono=True)
    target = int(SAMPLE_RATE * DURATION_SEC)
    if len(audio) > target:
        audio = audio[(len(audio)-target)//2:(len(audio)-target)//2+target]
    elif len(audio) < target:
        pad = np.zeros(target, dtype=np.float32)
        off = (target - len(audio)) // 2
        pad[off:off+len(audio)] = audio
        audio = pad
    sf.write(wav_path, audio, SAMPLE_RATE)


def synthesize_one(text, voice, rate, out_wav):
    """合成单条, 返回 bool"""
    import edge_tts
    rate_str = f"{int((rate - 1.0) * 100):+d}%"
    mp3 = out_wav.replace('.wav', '.mp3')

    async def _do():
        comm = edge_tts.Communicate(text, voice, rate=rate_str)
        await comm.save(mp3)

    try:
        asyncio.run(_do())
        convert_mp3_to_wav(mp3, out_wav)
        Path(mp3).unlink(missing_ok=True)
        return True
    except Exception:
        Path(mp3).unlink(missing_ok=True)
        return False


def main():
    parser = argparse.ArgumentParser(description='TTS Chinese speech generation')
    parser.add_argument('--output_dir', default='../data/raw/tts_chinese')
    parser.add_argument('--digits_only', action='store_true')
    parser.add_argument('--commands_only', action='store_true')
    parser.add_argument('--num_repeats', type=int, default=DEFAULT_REPEATS)
    args = parser.parse_args()

    try:
        import edge_tts
    except ImportError:
        print("[FATAL] pip install edge-tts"); sys.exit(1)

    if args.digits_only:
        zh, en = DIGITS_ZH, DIGITS_EN
    elif args.commands_only:
        zh, en = COMMANDS_ZH, COMMANDS_EN
    else:
        zh, en = ALL_ZH, ALL_EN

    out = Path(args.output_dir)
    tasks = []
    for z, e in zip(zh, en):
        for v in VOICES:
            for r in RATES:
                for rep in range(args.num_repeats):
                    tasks.append((z, e, v, r, rep))

    stats = {'total': len(tasks), 'success': 0, 'skipped': 0, 'failed': 0,
             'per_word': {}}

    print(f"\n{'='*60}")
    print(f"  TTS GENERATION")
    print(f"{'='*60}")
    print(f"  Words: {len(zh)} | Voices: {len(VOICES)} | Rates: {RATES}")
    print(f"  Repeats: {args.num_repeats} | Total: {len(tasks)}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}\n")

    t0 = time.time()
    for i, (z, e, voice, rate, rep) in enumerate(tasks):
        d = out / e; d.mkdir(parents=True, exist_ok=True)
        py = ZH_PINYIN.get(z, z)
        vs = voice.replace('zh-CN-','').replace('Neural','')
        fn = f"{py}_{vs}_r{int(rate*100):03d}_{rep:02d}.wav"
        fp = d / fn

        if fp.exists() and fp.stat().st_size > 100:
            stats['skipped'] += 1
            stats['per_word'][e] = stats['per_word'].get(e, 0) + 1
        else:
            ok = synthesize_one(z, voice, rate, str(fp))
            if ok:
                stats['success'] += 1
            else:
                stats['failed'] += 1
            stats['per_word'][e] = stats['per_word'].get(e, 0) + 1

        if i % 50 == 0 or i == len(tasks) - 1:
            done = stats['success'] + stats['skipped']
            elapsed = time.time() - t0
            spd = done / elapsed if elapsed > 0 else 0
            eta = (len(tasks) - done) / spd if spd > 0 else 0
            fail = f" fail{stats['failed']}" if stats['failed'] else ""
            bar = '#' * int(done / len(tasks) * 30)
            print(f"  [{done:5d}/{len(tasks)}] {bar} {done*100/len(tasks):.0f}% "
                  f"| {spd:.1f}/s | ok{stats['success']} skip{stats['skipped']}{fail} "
                  f"| ETA {eta/60:.0f}min", flush=True)

        if i % 15 == 14:
            time.sleep(0.3)

    total_t = time.time() - t0
    done = stats['success'] + stats['skipped']
    print(f"\n  DONE! {total_t/60:.1f}min")
    print(f"  {done}/{stats['total']} ({done*100/max(1,stats['total']):.0f}%) "
          f"| ok{stats['success']} skip{stats['skipped']} fail{stats['failed']}")

    sp = out / 'tts_generation_stats.json'
    with open(sp, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n  Per class:")
    for w in en:
        n = stats['per_word'].get(w, 0)
        print(f"    {w:<10s}: {n:4d}  {'#'*(n//10)}")


if __name__ == '__main__':
    main()
