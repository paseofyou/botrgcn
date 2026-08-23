"""
诊断时序嵌入质量：分布、方差、bot/human 区分度
"""
import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from collections import Counter

CUR_DIR = os.path.dirname(__file__)
NPZ_PATH = os.path.join(CUR_DIR, "feature_model_outputs", "twibot20_transformer_vectors.npz")
DATA_DIR = os.path.join(CUR_DIR, "Data", "Twibot-20")

def load_labels():
    """加载 train/dev/test 标签（support 无标签跳过）"""
    uid_label = {}
    for split in ['train', 'dev', 'test']:
        fpath = os.path.join(DATA_DIR, f"{split}.json")
        if not os.path.exists(fpath):
            continue
        with open(fpath, 'r', encoding='utf-8') as f:
            users = json.load(f)
        for u in users:
            uid = str(u['ID'])
            lbl = u.get('label')
            if lbl is not None:
                try:
                    uid_label[uid] = int(lbl)
                except:
                    pass
    return uid_label

def main():
    print("=" * 60)
    print("时序嵌入诊断报告")
    print("=" * 60)

    # 1) 加载嵌入
    data = np.load(NPZ_PATH, allow_pickle=True)
    vectors = data['vectors']
    user_ids = [str(uid) for uid in data['user_ids']]
    print(f"\n嵌入形状: {vectors.shape}")
    print(f"用户数量: {len(user_ids)}")

    # 2) 全局分布
    print(f"\n--- 全局分布 ---")
    print(f"均值: {vectors.mean():.6f}")
    print(f"标准差: {vectors.std():.6f}")
    print(f"最小值: {vectors.min():.6f}")
    print(f"最大值: {vectors.max():.6f}")
    print(f"全零行数: {(np.abs(vectors).sum(axis=1) == 0).sum()}")

    # 每维度统计
    dim_means = vectors.mean(axis=0)
    dim_stds = vectors.std(axis=0)
    print(f"\n每维度均值范围: [{dim_means.min():.4f}, {dim_means.max():.4f}]")
    print(f"每维度标准差范围: [{dim_stds.min():.4f}, {dim_stds.max():.4f}]")
    near_zero_dims = (dim_stds < 0.01).sum()
    print(f"近零方差维度 (std<0.01): {near_zero_dims}/{vectors.shape[1]}")

    # 3) L2 范数分布
    norms = np.linalg.norm(vectors, axis=1)
    print(f"\nL2 范数 — 均值: {norms.mean():.4f}, 标准差: {norms.std():.4f}, "
          f"最小: {norms.min():.4f}, 最大: {norms.max():.4f}")

    # 4) bot/human 区分度
    uid_label = load_labels()
    print(f"\n有标签用户数: {len(uid_label)}")
    label_dist = Counter(uid_label.values())
    print(f"标签分布: {dict(label_dist)}")

    # 匹配嵌入
    X_labeled = []
    y_labeled = []
    uid_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    for uid, lbl in uid_label.items():
        idx = uid_to_idx.get(uid)
        if idx is not None:
            X_labeled.append(vectors[idx])
            y_labeled.append(lbl)
    X_labeled = np.array(X_labeled)
    y_labeled = np.array(y_labeled)
    print(f"匹配到的有标签嵌入: {len(y_labeled)}")

    if len(y_labeled) > 0:
        bot_mask = y_labeled == 1
        human_mask = y_labeled == 0
        print(f"\n--- Bot vs Human 嵌入比较 ---")
        print(f"Bot 嵌入均值 L2: {np.linalg.norm(X_labeled[bot_mask].mean(axis=0)):.4f}")
        print(f"Human 嵌入均值 L2: {np.linalg.norm(X_labeled[human_mask].mean(axis=0)):.4f}")

        # 每维度 t-test 替代: 简单对比均值差异
        bot_means = X_labeled[bot_mask].mean(axis=0)
        human_means = X_labeled[human_mask].mean(axis=0)
        diff = np.abs(bot_means - human_means)
        pooled_std = X_labeled.std(axis=0) + 1e-8
        effect_size = diff / pooled_std  # Cohen's d 近似
        print(f"\n维度级别 Cohen's d (|bot_mean - human_mean| / std):")
        print(f"  最大效应: {effect_size.max():.4f} (dim {effect_size.argmax()})")
        print(f"  平均效应: {effect_size.mean():.4f}")
        print(f"  效应>0.2的维度数: {(effect_size > 0.2).sum()}/{vectors.shape[1]}")
        print(f"  效应>0.5的维度数: {(effect_size > 0.5).sum()}/{vectors.shape[1]}")

        # 5) 简单线性可分性测试
        print(f"\n--- 线性可分性测试 (Logistic Regression) ---")
        from sklearn.model_selection import cross_val_score
        lr = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
        scores = cross_val_score(lr, X_labeled, y_labeled, cv=5, scoring='accuracy')
        print(f"5折交叉验证准确率: {scores.mean():.4f} ± {scores.std():.4f}")

        f1_scores = cross_val_score(lr, X_labeled, y_labeled, cv=5, scoring='f1')
        print(f"5折交叉验证 F1: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")

        # 与随机基线对比
        majority_acc = max(bot_mask.sum(), human_mask.sum()) / len(y_labeled)
        print(f"多数类基线准确率: {majority_acc:.4f}")

    print("\n" + "=" * 60)
    print("诊断完成")

if __name__ == "__main__":
    main()
