"""
TwiBot-22 预处理脚本 — 生成与 train_twibot22.py 兼容的 .pt 张量
================================================================
用法:
    python preprocess_twibot22.py --data-dir /path/to/data --save-dir /path/to/save
    python preprocess_twibot22.py --stage 3   # 只运行 RoBERTa 编码

阶段:
  1. 加载 user.json → 提取数值/分类特征 + 描述文本
  2. 流式加载 tweet_*.json → 每用户推文文本
  3. RoBERTa 编码 → des_tensor, tweets_tensor
  4. 构建图 (edge.csv) → edge_index, edge_type
  5. 构建标签和划分 → label, train/val/test_idx
  6. 整合保存最终 .pt 张量
"""

import os
import sys
import json
import csv
import math
import logging
import argparse
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

# ---------------------- 默认路径 ----------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "Data", "TwiBot-22")
DEFAULT_SAVE_DIR = os.path.join(SCRIPT_DIR, "saved_data", "twibot22_data")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RoBERTa 配置
ROBERTA_MODEL_NAME = "roberta-base"
ROBERTA_MAX_LEN = 512
ROBERTA_BATCH_SIZE = 128

NUM_PROP_DIM = 5
CAT_PROP_DIM = 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_paths(args):
    """从命令行参数构建所有路径"""
    data_dir = args.data_dir
    save_dir = args.save_dir
    cache_dir = os.path.join(save_dir, "cache")
    return data_dir, save_dir, cache_dir


# =====================================================
# 阶段 1
# =====================================================
def stage1_load_users(data_dir, cache_dir):
    logging.info("=== 阶段 1: 加载 user.json ===")
    user_path = os.path.join(data_dir, "user.json")
    logging.info(f"加载 {user_path} ({os.path.getsize(user_path) / 1e9:.2f} GB)")

    with open(user_path, "r", encoding="utf-8") as f:
        users = json.load(f)

    n_users = len(users)
    logging.info(f"共 {n_users} 个用户")

    user_ids = []
    num_props = np.zeros((n_users, NUM_PROP_DIM), dtype=np.float32)
    cat_props = np.zeros((n_users, CAT_PROP_DIM), dtype=np.float32)
    descriptions = []

    for i, u in enumerate(tqdm(users, desc="提取用户特征")):
        uid = str(u.get("id", ""))
        user_ids.append(uid)

        pm = u.get("public_metrics") or {}
        followers = pm.get("followers_count", 0) or 0
        following = pm.get("following_count", 0) or 0
        tweet_count = pm.get("tweet_count", 0) or 0
        listed_count = pm.get("listed_count", 0) or 0
        ff_ratio = min(followers / (following + 1e-6), 1000.0)

        num_props[i] = [followers, following, tweet_count, listed_count, ff_ratio]

        verified = float(u.get("verified", False) or False)
        protected = float(u.get("protected", False) or False)
        has_url = float(bool(u.get("url", "")))
        cat_props[i] = [verified, protected, has_url]

        desc = u.get("description", "") or ""
        descriptions.append(desc[:1000])

    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "user_ids.npy"), np.array(user_ids, dtype=object))
    np.save(os.path.join(cache_dir, "num_properties.npy"), num_props)
    np.save(os.path.join(cache_dir, "cat_properties.npy"), cat_props)

    uid_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    with open(os.path.join(cache_dir, "uid_to_idx.json"), "w", encoding="utf-8") as f:
        json.dump(uid_to_idx, f)

    with open(os.path.join(cache_dir, "descriptions.jsonl"), "w", encoding="utf-8") as f:
        for desc in descriptions:
            f.write(json.dumps(desc, ensure_ascii=False) + "\n")

    logging.info(f"阶段 1 完成: {n_users} 用户")


# =====================================================
# 阶段 2
# =====================================================
def stage2_collect_tweets(data_dir, cache_dir):
    logging.info("=== 阶段 2: 收集推文文本 ===")

    with open(os.path.join(cache_dir, "uid_to_idx.json"), "r", encoding="utf-8") as f:
        uid_to_idx = json.load(f)

    n_users = len(uid_to_idx)
    user_tweets = [""] * n_users
    tweet_counts = np.zeros(n_users, dtype=np.int32)

    tweet_files = sorted([f for f in os.listdir(data_dir) if f.startswith("tweet_") and f.endswith(".json")])
    logging.info(f"找到 {len(tweet_files)} 个推文文件")

    for tf_name in tweet_files:
        tf_path = os.path.join(data_dir, tf_name)
        logging.info(f"处理 {tf_name} ({os.path.getsize(tf_path) / 1e9:.2f} GB)")

        with open(tf_path, "r", encoding="utf-8") as f:
            tweets = json.load(f)

        for tw in tqdm(tweets, desc=f"{tf_name}", mininterval=5):
            author_id = tw.get("author_id")
            if author_id is None:
                continue
            uid = f"u{author_id}"
            idx = uid_to_idx.get(uid)
            if idx is None:
                continue

            text = tw.get("text", "") or ""
            if text and len(user_tweets[idx]) < 2000:
                if user_tweets[idx]:
                    user_tweets[idx] += " [SEP] "
                user_tweets[idx] += text[:280]
                tweet_counts[idx] += 1

        del tweets
        import gc; gc.collect()

    with open(os.path.join(cache_dir, "user_tweets.jsonl"), "w", encoding="utf-8") as f:
        for text in user_tweets:
            f.write(json.dumps(text, ensure_ascii=False) + "\n")

    has_tweets = (tweet_counts > 0).sum()
    logging.info(f"阶段 2 完成: 有推文用户 = {has_tweets}, 平均推文数 = {tweet_counts[tweet_counts > 0].mean():.1f}")


# =====================================================
# 阶段 3
# =====================================================
def stage3_encode_text(cache_dir):
    logging.info("=== 阶段 3: RoBERTa 文本编码 ===")

    from transformers import RobertaTokenizer, RobertaModel

    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
    model = RobertaModel.from_pretrained(ROBERTA_MODEL_NAME).to(DEVICE)
    model.eval()
    hidden_size = model.config.hidden_size

    def encode_texts_batched(texts, desc="编码"):
        n = len(texts)
        result = torch.zeros((n, hidden_size), dtype=torch.float32)
        for start in tqdm(range(0, n, ROBERTA_BATCH_SIZE), desc=desc):
            end = min(start + ROBERTA_BATCH_SIZE, n)
            batch = [t if t.strip() else "empty" for t in texts[start:end]]
            tokens = tokenizer(batch, padding=True, truncation=True,
                               max_length=ROBERTA_MAX_LEN, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                output = model(**tokens)
                result[start:end] = output.last_hidden_state[:, 0, :].cpu()
        return result

    # 描述
    logging.info("3.1 编码描述...")
    descriptions = []
    with open(os.path.join(cache_dir, "descriptions.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            descriptions.append(json.loads(line.strip()))

    des_tensor = encode_texts_batched(descriptions, desc="编码描述")
    torch.save(des_tensor, os.path.join(cache_dir, "des_tensor.pt"))
    logging.info(f"  des_tensor: {des_tensor.shape}")
    del descriptions, des_tensor

    # 推文
    logging.info("3.2 编码推文...")
    user_tweets = []
    with open(os.path.join(cache_dir, "user_tweets.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            user_tweets.append(json.loads(line.strip()))

    tweets_tensor = encode_texts_batched(user_tweets, desc="编码推文")
    torch.save(tweets_tensor, os.path.join(cache_dir, "tweets_tensor.pt"))
    logging.info(f"  tweets_tensor: {tweets_tensor.shape}")
    del user_tweets, tweets_tensor

    logging.info("阶段 3 完成")


# =====================================================
# 阶段 4
# =====================================================
def stage4_build_graph(data_dir, cache_dir):
    logging.info("=== 阶段 4: 构建图 ===")

    with open(os.path.join(cache_dir, "uid_to_idx.json"), "r", encoding="utf-8") as f:
        uid_to_idx = json.load(f)

    edge_path = os.path.join(data_dir, "edge.csv")
    logging.info(f"处理 edge.csv ({os.path.getsize(edge_path) / 1e9:.2f} GB)")

    RELATION_MAP = {"following": 0, "followers": 1}
    src_list, dst_list, type_list = [], [], []
    skipped = 0

    with open(edge_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="处理边", mininterval=5):
            rel = row["relation"]
            if rel not in RELATION_MAP:
                skipped += 1
                continue
            src_idx = uid_to_idx.get(row["source_id"])
            dst_idx = uid_to_idx.get(row["target_id"])
            if src_idx is None or dst_idx is None:
                skipped += 1
                continue
            src_list.append(src_idx)
            dst_list.append(dst_idx)
            type_list.append(RELATION_MAP[rel])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_type = torch.tensor(type_list, dtype=torch.long)

    torch.save(edge_index, os.path.join(cache_dir, "edge_index.pt"))
    torch.save(edge_type, os.path.join(cache_dir, "edge_type.pt"))

    logging.info(f"阶段 4 完成: {edge_index.shape[1]} 条边, 跳过 {skipped}")


# =====================================================
# 阶段 5
# =====================================================
def stage5_build_labels_and_splits(data_dir, cache_dir):
    logging.info("=== 阶段 5: 构建标签和划分 ===")

    with open(os.path.join(cache_dir, "uid_to_idx.json"), "r", encoding="utf-8") as f:
        uid_to_idx = json.load(f)
    n_users = len(uid_to_idx)

    labels = torch.full((n_users,), -1, dtype=torch.long)
    label_map = {"human": 0, "bot": 1}
    with open(os.path.join(data_dir, "label.csv"), "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx = uid_to_idx.get(row["id"])
            if idx is not None:
                labels[idx] = label_map.get(row["label"], -1)

    train_indices, val_indices, test_indices = [], [], []
    split_target = {"train": train_indices, "val": val_indices, "test": test_indices}
    with open(os.path.join(data_dir, "split.csv"), "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx = uid_to_idx.get(row["id"])
            if idx is not None:
                target = split_target.get(row["split"])
                if target is not None:
                    target.append(idx)

    torch.save(labels, os.path.join(cache_dir, "label.pt"))
    torch.save(torch.tensor(train_indices, dtype=torch.long), os.path.join(cache_dir, "train_idx.pt"))
    torch.save(torch.tensor(val_indices, dtype=torch.long), os.path.join(cache_dir, "val_idx.pt"))
    torch.save(torch.tensor(test_indices, dtype=torch.long), os.path.join(cache_dir, "test_idx.pt"))

    logging.info(f"阶段 5 完成: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")


# =====================================================
# 阶段 6
# =====================================================
def stage6_save_final(save_dir, cache_dir):
    logging.info("=== 阶段 6: 保存最终张量 ===")

    out_dir = os.path.join(save_dir, "processed_data")
    os.makedirs(out_dir, exist_ok=True)

    # 数值特征: log1p + MinMax
    num_props = np.load(os.path.join(cache_dir, "num_properties.npy"))
    num_log = np.log1p(num_props).astype(np.float32)
    mins, maxs = num_log.min(axis=0), num_log.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-8] = 1.0
    num_norm = (num_log - mins) / ranges
    torch.save(torch.tensor(num_norm, dtype=torch.float32), os.path.join(out_dir, "num_properties_tensor.pt"))

    # 分类特征
    cat_props = np.load(os.path.join(cache_dir, "cat_properties.npy"))
    torch.save(torch.tensor(cat_props, dtype=torch.float32), os.path.join(out_dir, "cat_properties_tensor.pt"))

    # 文本 + 图 + 标签 + 划分
    copy_items = ["des_tensor.pt", "tweets_tensor.pt", "edge_index.pt", "edge_type.pt",
                  "label.pt", "train_idx.pt", "val_idx.pt", "test_idx.pt"]
    for name in copy_items:
        src = os.path.join(cache_dir, name)
        if os.path.exists(src):
            t = torch.load(src, map_location="cpu", weights_only=True)
            torch.save(t, os.path.join(out_dir, name))
            logging.info(f"  {name}: {t.shape}")

    # user_ids
    user_ids = np.load(os.path.join(cache_dir, "user_ids.npy"), allow_pickle=True)
    np.save(os.path.join(out_dir, "user_ids.npy"), user_ids)

    logging.info(f"阶段 6 完成: {out_dir}")

    # 汇总
    print("\n=== TwiBot-22 预处理完成 ===")
    for fname in sorted(os.listdir(out_dir)):
        fpath = os.path.join(out_dir, fname)
        if fname.endswith('.pt'):
            t = torch.load(fpath, map_location='cpu', weights_only=True)
            print(f"  {fname:35s} | shape: {str(t.shape):25s} | dtype: {t.dtype}")
        elif fname.endswith('.npy'):
            a = np.load(fpath, allow_pickle=True)
            print(f"  {fname:35s} | shape: {str(a.shape):25s} | dtype: {a.dtype}")


# =====================================================
# 主入口
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="TwiBot-22 预处理")
    parser.add_argument("--stage", type=int, default=0, help="只运行指定阶段 (1-6), 0=全部")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="TwiBot-22 原始数据目录")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR, help="输出目录")
    args = parser.parse_args()

    data_dir = args.data_dir
    save_dir = args.save_dir
    cache_dir = os.path.join(save_dir, "cache")

    logging.info(f"数据目录: {data_dir}")
    logging.info(f"保存目录: {save_dir}")

    stages = {
        1: ("加载用户", lambda: stage1_load_users(data_dir, cache_dir)),
        2: ("收集推文", lambda: stage2_collect_tweets(data_dir, cache_dir)),
        3: ("RoBERTa 编码", lambda: stage3_encode_text(cache_dir)),
        4: ("构建图", lambda: stage4_build_graph(data_dir, cache_dir)),
        5: ("标签和划分", lambda: stage5_build_labels_and_splits(data_dir, cache_dir)),
        6: ("保存最终张量", lambda: stage6_save_final(save_dir, cache_dir)),
    }

    if args.stage > 0:
        name, func = stages[args.stage]
        logging.info(f"单独运行阶段 {args.stage}: {name}")
        func()
    else:
        for num, (name, func) in stages.items():
            logging.info(f"\n{'='*50}\n阶段 {num}: {name}\n{'='*50}")
            func()

    logging.info("完成！")


if __name__ == "__main__":
    main()
