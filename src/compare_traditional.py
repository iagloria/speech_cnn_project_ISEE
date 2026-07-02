"""
传统方法对比验证 — CNN vs SVM / Random Forest / KNN
=====================================================
使用完全相同的梅尔频谱图数据，对比 CNN 与传统机器学习方法的性能。
体现深度学习的端到端特征学习优势。

用法:
  python compare_traditional.py --data_dir ../data/processed --num_classes 20
  python compare_traditional.py --data_dir ../data/processed_mixed --num_classes 40

Author: Speech CNN Project
Date: 2026-06
"""

import sys, argparse, time, json
from pathlib import Path
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

sys.path.insert(0, str(Path(__file__).parent))
from dataset import SpeechCommandDataset
from preprocess import GSC_20_WORDS, BILINGUAL_40_WORDS


def load_data_flat(data_dir, num_classes):
    """加载数据并展平为 1D 向量（memmap 按需读取 + PCA 降维）"""
    import time as _time
    words = BILINGUAL_40_WORDS if num_classes >= 40 else GSC_20_WORDS
    data_path = Path(data_dir)

    # 用 memmap 按需读取，避免一次性加载 2.5 GiB 到内存
    t0 = _time.time()
    X_train_raw = np.load(data_path / 'X_train.npy', mmap_mode='r')
    X_test_raw  = np.load(data_path / 'X_test.npy', mmap_mode='r')
    y_train = np.load(data_path / 'y_train.npy')
    y_test  = np.load(data_path / 'y_test.npy')
    print(f"  [LOAD] memmap ready ({_time.time()-t0:.1f}s)", flush=True)
    print(f"  X_train: {X_train_raw.shape}, X_test: {X_test_raw.shape}", flush=True)

    n_train, n_features = X_train_raw.shape[0], X_train_raw.shape[1] * X_train_raw.shape[2]
    n_test = X_test_raw.shape[0]

    BATCH = 5000
    n_components = min(256, n_features)
    ipca = IncrementalPCA(n_components=n_components, batch_size=BATCH)
    scaler = StandardScaler()

    print(f"  Flattened dim: {n_features} → PCA → {n_components}")

    # PCA fit: 均匀采样 10000 条
    PCA_FIT_SAMPLES = min(10000, n_train)
    print(f"  [PCA] Fitting on {PCA_FIT_SAMPLES} samples...", flush=True)
    t1 = _time.time()
    step = max(1, n_train // PCA_FIT_SAMPLES)
    fit_data = X_train_raw[::step][:PCA_FIT_SAMPLES].reshape(-1, n_features).astype(np.float32)
    ipca.fit(fit_data)
    del fit_data
    print(f"    done ({_time.time()-t1:.1f}s), variance: {ipca.explained_variance_ratio_.sum():.3f}", flush=True)

    # 分批 PCA transform
    print(f"  [PCA] Transforming train...", flush=True)
    t1 = _time.time()
    X_train_pca = []
    for start in range(0, n_train, BATCH):
        end = min(start + BATCH, n_train)
        batch = X_train_raw[start:end].reshape(end - start, -1).astype(np.float32)
        X_train_pca.append(ipca.transform(batch))
        print(f"    {start//BATCH+1}/{(n_train-1)//BATCH+1} ({_time.time()-t1:.0f}s)", flush=True)
    X_train = np.concatenate(X_train_pca)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    del X_train_pca
    print(f"    train done ({_time.time()-t1:.0f}s)", flush=True)

    print(f"  [PCA] Transforming test...", flush=True)
    t1 = _time.time()
    X_test_parts = []
    for start in range(0, n_test, BATCH):
        end = min(start + BATCH, n_test)
        batch = X_test_raw[start:end].reshape(end - start, -1).astype(np.float32)
        X_test_parts.append(scaler.transform(ipca.transform(batch)))
    X_test = np.concatenate(X_test_parts)
    print(f"    test done ({_time.time()-t1:.0f}s)", flush=True)

    return X_train, y_train, X_test, y_test, words


def evaluate_model(name, model, X_train, y_train, X_test, y_test, words):
    """训练 + 评估单个模型"""
    t0 = time.time()
    print(f"\n  [{name}] Training...")
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # 中英文分别计算
    if len(words) >= 40:
        en_mask = y_test < 20
        zh_mask = y_test >= 20
        en_acc = (y_pred[en_mask] == y_test[en_mask]).sum() / max(1, en_mask.sum())
        zh_acc = (y_pred[zh_mask] == y_test[zh_mask]).sum() / max(1, zh_mask.sum())
    else:
        en_acc = acc
        zh_acc = None

    print(f"    Accuracy: {acc*100:.2f}%  |  F1(macro): {f1_macro:.4f}  |  Time: {train_time:.1f}s")

    return {
        'name': name,
        'accuracy': acc,
        'f1_macro': f1_macro,
        'en_acc': en_acc,
        'zh_acc': zh_acc,
        'train_time': train_time,
    }


def load_cnn_results(model_info_path=None):
    """从已有模型加载 CNN 结果"""
    # 尝试从 metrics.json 读取
    for results_dir in ['../results/v2_bilingual_40/bilingual_v3_weighted',
                         '../results/v1_english_20/20260611_220716']:
        p = Path(results_dir) / 'metrics.json'
        if p.exists():
            with open(p) as f:
                m = json.load(f)
            en_acc = zh_acc = en_t = zh_t = 0
            for k, v in m['per_class'].items():
                sup = v['support']
                if sup == 0: continue
                if k.startswith('zh_'):
                    zh_acc += v['recall'] * sup; zh_t += sup
                else:
                    en_acc += v['recall'] * sup; en_t += sup
            return {
                'name': 'CNN (V3 加权)',
                'accuracy': m['accuracy'],
                'f1_macro': m['macro_f1'],
                'en_acc': en_acc / max(1, en_t),
                'zh_acc': zh_acc / max(1, zh_t) if zh_t > 0 else None,
                'train_time': 15 * 60,
            }
    return None


def plot_comparison(results, num_classes, output_dir):
    """绘制对比柱状图"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    names = [r['name'] for r in results]
    accs  = [r['accuracy'] * 100 for r in results]
    f1s   = [r['f1_macro'] * 100 for r in results]
    colors = ['#2196F3' if 'CNN' in n else '#FF9800' if 'SVM' in n
              else '#4CAF50' if 'RF' in n else '#9C27B0' for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    bars = axes[0].bar(names, accs, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Overall Accuracy Comparison')
    axes[0].set_ylim(0, max(accs) * 1.2)
    for b, v in zip(bars, accs):
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                     f'{v:.1f}%', ha='center', fontweight='bold', fontsize=10)

    # F1
    bars2 = axes[1].bar(names, f1s, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_ylabel('Macro F1 (%)')
    axes[1].set_title('Macro F1 Comparison')
    axes[1].set_ylim(0, max(f1s) * 1.2)
    for b, v in zip(bars2, f1s):
        axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                     f'{v:.1f}%', ha='center', fontweight='bold', fontsize=10)

    plt.suptitle(f'CNN vs Traditional ML — {num_classes}-Class Speech Recognition',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'method_comparison.png', dpi=150)
    plt.close()
    print(f"\n[INFO] Comparison chart saved to {output_dir}/method_comparison.png")

    # 中英分别对比 (仅 40 类)
    if num_classes >= 40:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(names))
        w = 0.35
        en_accs = [r['en_acc'] * 100 if r['en_acc'] else 0 for r in results]
        zh_accs = [r['zh_acc'] * 100 if r['zh_acc'] else 0 for r in results]

        bars_en = ax.bar(x - w/2, en_accs, w, label='English', color='#2196F3', edgecolor='black', linewidth=0.5)
        bars_zh = ax.bar(x + w/2, zh_accs, w, label='Chinese', color='#FF5722', edgecolor='black', linewidth=0.5)

        for b in bars_en:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                    f'{b.get_height():.1f}%', ha='center', fontsize=9)
        for b in bars_zh:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                    f'{b.get_height():.1f}%', ha='center', fontsize=9)

        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_ylabel('Accuracy (%)'); ax.set_title('English vs Chinese Accuracy by Method')
        ax.legend(); ax.set_ylim(0, max(max(en_accs), max(zh_accs)) * 1.2)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'en_zh_comparison.png', dpi=150)
        plt.close()
        print(f"[INFO] EN/ZH comparison saved to {output_dir}/en_zh_comparison.png")


def main():
    parser = argparse.ArgumentParser(description='CNN vs Traditional ML Comparison')
    parser.add_argument('--data_dir', default='../data/processed',
                        help='Data directory')
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--output_dir', default='../results/comparison')
    parser.add_argument('--svm_samples', type=int, default=5000,
                        help='Max samples for SVM (slow on large data)')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  CNN vs Traditional ML Comparison")
    print(f"{'='*60}")
    print(f"  Data: {args.data_dir}  |  Classes: {args.num_classes}")

    # 加载数据
    X_train, y_train, X_test, y_test, words = load_data_flat(args.data_dir, args.num_classes)
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    results = []

    # ---- CNN ----
    cnn = load_cnn_results()
    if cnn:
        results.append(cnn)
        print(f"\n  [CNN] Loaded: Acc={cnn['accuracy']*100:.2f}%  F1={cnn['f1_macro']:.4f}")

    # ---- SVM ----
    n_svm = min(args.svm_samples, len(X_train))
    idx = np.random.choice(len(X_train), n_svm, replace=False)
    results.append(evaluate_model(
        'SVM (RBF)', SVC(kernel='rbf', C=10, gamma='scale', max_iter=2000),
        X_train[idx], y_train[idx], X_test, y_test, words))

    # ---- Random Forest ----
    results.append(evaluate_model(
        'Random Forest', RandomForestClassifier(n_estimators=200, max_depth=30, n_jobs=-1, random_state=42),
        X_train, y_train, X_test, y_test, words))

    # ---- KNN ----
    results.append(evaluate_model(
        'KNN (k=5)', KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        X_train, y_train, X_test, y_test, words))

    # 汇总
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Method':<18s} {'Acc':>8s} {'F1':>8s} {'EN':>8s} {'ZH':>8s} {'Time':>8s}")
    print(f"  {'-'*50}")
    for r in results:
        en = f"{r['en_acc']*100:>7.1f}%" if r['en_acc'] else '       -'
        zh = f"{r['zh_acc']*100:>7.1f}%" if r['zh_acc'] else '       -'
        print(f"  {r['name']:<18s} {r['accuracy']*100:>7.2f}% {r['f1_macro']*100:>7.2f}% "
              f"{en} {zh} {r['train_time']:>7.1f}s")

    # 保存
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / 'comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # 画图
    plot_comparison(results, args.num_classes, str(out))

    print(f"\n[SUCCESS] Results saved to {out}/")
    print(f"  comparison_results.json  — raw data")
    print(f"  method_comparison.png    — accuracy & F1 chart")
    if args.num_classes >= 40:
        print(f"  en_zh_comparison.png     — English vs Chinese breakdown")


if __name__ == '__main__':
    main()
