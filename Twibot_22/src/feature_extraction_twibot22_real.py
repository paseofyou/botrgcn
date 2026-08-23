#!/usr/bin/env python3
"""
TwiBot-22 真实时间序列特征提取脚本
====================================
基于推文级时间戳构造 Real-TS 和 Hybrid-TS 时间序列。

与 feature_extraction_twibot22.py (Pseudo-TS) 的区别:
  - Pseudo-TS: 从 14 维静态特征合成伪时间序列
  - Real-TS:   从真实推文时间戳聚合行为统计, 构造真实时间序列
  - Hybrid-TS: λ * Real-TS + (1-λ) * Pseudo-TS, λ 由时间桶覆盖率决定

输出维度与 Pseudo-TS 一致 (14 维), 因此 Transformer 架构无需修改。

用法:
    python feature_extraction_twibot22_real.py --mode real
    python feature_extraction_twibot22_real.py --mode hybrid
    python feature_extraction_twibot22_real.py --mode real --data-dir /path/to/data --work-dir /path/to/work
"""

import os
import sys
import json
import csv
import math
import logging
import datetime
import argparse
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score

# ----------------------
# 配置
# ----------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CRAWL_DATE_STR = '2022-06-01'
CRAWL_DATE = datetime.datetime.strptime(CRAWL_DATE_STR, '%Y-%m-%d')

SEQ_LEN = 32
D_MODEL = 64
N_HEAD = 8
NUM_ENCODER_LAYERS = 2
DROPOUT = 0.1
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
NUM_EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Real-TS 每个时间桶提取的特征数 (与 Pseudo-TS 14 维对齐)
N_BUCKET_FEATURES = 14

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
torch.manual_seed(42)
np.random.seed(42)


# =====================================================
# 工具函数
# =====================================================
def _parse_iso_date(date_str):
    """解析 TwiBot-22 的 ISO 格式时间戳"""
    if not date_str:
        return None
    try:
        clean = str(date_str).strip()
        if '+' in clean:
            clean = clean[:clean.index('+')]
        elif clean.endswith('Z'):
            clean = clean[:-1]
        return datetime.datetime.strptime(clean.strip(), '%Y-%m-%d %H:%M:%S')
    except Exception:
        try:
            return datetime.datetime.fromisoformat(str(date_str).replace('Z', '+00:00')).replace(tzinfo=None)
        except Exception:
            return None


def _tweet_features(tw):
    """从单条推文提取轻量级特征元组"""
    text = tw.get("text", "") or ""
    text_len = len(text)

    has_url = 1.0 if re.search(r'https?://\S+', text) else 0.0
    has_hashtag = 1.0 if '#' in text else 0.0
    has_mention = 1.0 if '@' in text else 0.0
    is_retweet = 1.0 if text.startswith("RT @") else 0.0

    pm = tw.get("public_metrics") or {}
    like_count = pm.get("like_count", 0) or 0
    retweet_count = pm.get("retweet_count", 0) or 0
    reply_count = pm.get("reply_count", 0) or 0
    quote_count = pm.get("quote_count", 0) or 0

    return (text_len, has_url, has_hashtag, has_mention, is_retweet,
            like_count, retweet_count, reply_count, quote_count)


# =====================================================
# 核心: 真实时间序列构造
# =====================================================
class UserBucketAggregator:
    """
    内存高效的用户时间桶聚合器。
    为每个用户维护 T 个桶的运行统计量, 支持流式追加推文。
    """

    def __init__(self, n_users, seq_len, user_bucket_edges):
        """
        Args:
            n_users: 用户总数
            seq_len: 时间桶数 T
            user_bucket_edges: dict {uid_str: np.array of T+1 timestamps (float)}
                               每个用户的桶边界 (seconds since epoch)
        """
        self.seq_len = seq_len
        self.bucket_edges = user_bucket_edges

        # 每个桶的聚合统计: (n_users, T, n_agg_fields)
        # 字段: [count, sum_text_len, sum_text_len_sq,
        #         sum_url, sum_hashtag, sum_mention, sum_retweet,
        #         sum_like, sum_rt, sum_reply, sum_quote,
        #         prev_timestamp, sum_interval, sum_interval_sq, interval_count]
        self.N_AGG = 15
        self.agg = {}  # uid_str → np.array (T, N_AGG), 懒初始化节省内存

    def _get_bucket_idx(self, uid_str, ts_epoch):
        """二分查找推文所属的时间桶"""
        edges = self.bucket_edges.get(uid_str)
        if edges is None:
            return -1
        idx = np.searchsorted(edges, ts_epoch, side='right') - 1
        return max(0, min(idx, self.seq_len - 1))

    def add_tweet(self, uid_str, tweet_ts_epoch, feat_tuple):
        """
        追加一条推文到对应用户的时间桶。
        feat_tuple: (text_len, has_url, has_hashtag, has_mention, is_retweet,
                     like_count, retweet_count, reply_count, quote_count)
        """
        bucket_idx = self._get_bucket_idx(uid_str, tweet_ts_epoch)
        if bucket_idx < 0:
            return

        if uid_str not in self.agg:
            self.agg[uid_str] = np.zeros((self.seq_len, self.N_AGG), dtype=np.float64)

        row = self.agg[uid_str][bucket_idx]
        (text_len, has_url, has_hashtag, has_mention, is_retweet,
         like_count, retweet_count, reply_count, quote_count) = feat_tuple

        row[0] += 1  # count
        row[1] += text_len  # sum_text_len
        row[2] += text_len ** 2  # sum_text_len_sq
        row[3] += has_url
        row[4] += has_hashtag
        row[5] += has_mention
        row[6] += is_retweet
        row[7] += like_count
        row[8] += retweet_count
        row[9] += reply_count
        row[10] += quote_count

        # 时间间隔 (同一桶内相邻推文)
        prev_ts = row[11]
        if prev_ts > 0 and tweet_ts_epoch > prev_ts:
            interval = tweet_ts_epoch - prev_ts
            row[12] += interval  # sum_interval
            row[13] += interval ** 2  # sum_interval_sq
            row[14] += 1  # interval_count
        row[11] = tweet_ts_epoch  # prev_timestamp

    def build_matrix(self, uid_str):
        """
        将聚合统计转化为 (T, 14) 的特征矩阵。

        14 维特征:
          0. tweet_count (log1p)
          1. avg_text_length
          2. text_length_std
          3. url_ratio
          4. hashtag_ratio
          5. mention_ratio
          6. retweet_ratio
          7. avg_like_count (log1p)
          8. avg_retweet_count (log1p)
          9. avg_reply_count (log1p)
         10. avg_quote_count (log1p)
         11. inter_tweet_interval_mean (log1p, seconds)
         12. inter_tweet_interval_std (log1p, seconds)
         13. bucket_position (0~1)
        """
        matrix = np.zeros((self.seq_len, N_BUCKET_FEATURES), dtype=np.float32)

        if uid_str not in self.agg:
            return matrix, 0  # 无推文

        agg = self.agg[uid_str]
        non_empty_count = 0

        for t in range(self.seq_len):
            row = agg[t]
            count = row[0]

            if count < 1:
                matrix[t, 13] = t / max(1, self.seq_len - 1)
                continue

            non_empty_count += 1
            n = count

            # 0: tweet_count (log1p)
            matrix[t, 0] = np.log1p(count)

            # 1: avg_text_length
            avg_len = row[1] / n
            matrix[t, 1] = avg_len

            # 2: text_length_std
            if n > 1:
                var = max(0, row[2] / n - avg_len ** 2)
                matrix[t, 2] = np.sqrt(var)

            # 3-6: ratios
            matrix[t, 3] = row[3] / n  # url_ratio
            matrix[t, 4] = row[4] / n  # hashtag_ratio
            matrix[t, 5] = row[5] / n  # mention_ratio
            matrix[t, 6] = row[6] / n  # retweet_ratio

            # 7-10: avg public_metrics (log1p)
            matrix[t, 7] = np.log1p(row[7] / n)   # avg_like
            matrix[t, 8] = np.log1p(row[8] / n)   # avg_retweet
            matrix[t, 9] = np.log1p(row[9] / n)   # avg_reply
            matrix[t, 10] = np.log1p(row[10] / n)  # avg_quote

            # 11-12: inter-tweet intervals
            ic = row[14]
            if ic > 0:
                mean_interval = row[12] / ic
                matrix[t, 11] = np.log1p(mean_interval)
                if ic > 1:
                    var_interval = max(0, row[13] / ic - mean_interval ** 2)
                    matrix[t, 12] = np.log1p(np.sqrt(var_interval))

            # 13: bucket_position
            matrix[t, 13] = t / max(1, self.seq_len - 1)

        return matrix, non_empty_count


# =====================================================
# Pseudo-TS 构建 (从 feature_extraction_twibot22.py 复用)
# =====================================================
_TS_KEYS = [
    'account_age_days',
    'log_tweets_per_day', 'log_followers_per_day', 'log_likes_per_day',
    'interaction_rate', 'topic_diversity', 'follow_balance',
    'url_ratio', 'avg_tweet_length', 'std_tweet_length',
    'has_default_profile_image', 'has_url_in_description',
    'followers_interaction_index',
    'activity_index'
]
_LINEAR_STRICT_KEYS = {'account_age_days'}
_GROWTH_KEYS = {
    'log_tweets_per_day', 'log_followers_per_day', 'log_likes_per_day',
    'activity_index'
}


def calculate_static_features_twibot22(user_data):
    """从 TwiBot-22 用户数据提取 14 维静态特征 (与 pseudo 版相同)"""
    pm = user_data.get('public_metrics') or {}
    followers_count = pm.get('followers_count', 0) or 0
    following_count = pm.get('following_count', 0) or 0
    tweet_count = pm.get('tweet_count', 0) or 0
    listed_count = pm.get('listed_count', 0) or 0

    created_at = _parse_iso_date(user_data.get('created_at'))
    age_days = max(1, (CRAWL_DATE - created_at).days) if created_at else 1

    log_tweets_per_day = np.log1p(tweet_count) / age_days
    log_followers_per_day = np.log1p(followers_count) / age_days
    log_likes_per_day = np.log1p(listed_count) / age_days

    follow_balance = np.clip(following_count / (followers_count + 1e-6), 0, 100)

    description = user_data.get('description', '') or ''
    avg_tweet_length = len(description) / 100.0
    std_tweet_length = len(user_data.get('username', '') or '') / 15.0
    url_ratio = np.clip(followers_count / (tweet_count + 1e-6), 0, 1000)
    interaction_rate = np.clip(tweet_count / (following_count + 1e-6), 0, 1000)
    topic_diversity = np.clip(listed_count / (following_count + 1e-6), 0, 100)

    has_default_profile_image = float(bool(user_data.get('url', '')))
    has_url_in_description = float(bool(re.search(r'https?://\S+|www\.\S+', description)))

    followers_interaction_index = listed_count / (followers_count + 1e-6)
    activity_index = np.clip(tweet_count / (followers_count + 1e-6), 0, 1000)

    return {
        "user_id": str(user_data.get("id", "")),
        "account_age_days": float(age_days),
        "log_tweets_per_day": float(log_tweets_per_day),
        "log_followers_per_day": float(log_followers_per_day),
        "log_likes_per_day": float(log_likes_per_day),
        "follow_balance": float(follow_balance),
        "interaction_rate": float(interaction_rate),
        "topic_diversity": float(topic_diversity),
        "avg_tweet_length": float(avg_tweet_length),
        "std_tweet_length": float(std_tweet_length),
        "url_ratio": float(url_ratio),
        "has_default_profile_image": float(has_default_profile_image),
        "has_url_in_description": float(has_url_in_description),
        "followers_interaction_index": float(followers_interaction_index),
        "activity_index": float(activity_index),
    }


def build_pseudo_timeseries_from_features(feat_dict, seq_len=SEQ_LEN):
    """伪时间序列生成器 (与 Pseudo-TS 版完全一致)"""
    keys = _TS_KEYS
    t = np.linspace(0, 1, seq_len)

    user_id_str = str(feat_dict.get('user_id', '0'))
    user_seed = int(hash(user_id_str) % 100000)
    rng = np.random.RandomState(user_seed)

    vals = np.array([float(feat_dict.get(k, 0.0)) for k in keys], dtype=float)

    seq_matrix = []
    for i, key in enumerate(keys):
        val = vals[i]
        base_curve = np.zeros(seq_len)

        if key in _LINEAR_STRICT_KEYS:
            base_curve = val * t
        elif key in _GROWTH_KEYS:
            if rng.random() < 0.5:
                k_steep = 5.0 + rng.uniform(0, 5)
                t0 = rng.uniform(0.3, 0.7)
                base_curve = val / (1.0 + np.exp(-k_steep * (t - t0)))
            else:
                base_curve = val * np.log1p(9 * t) / np.log1p(9)
        else:
            base_curve = np.full(seq_len, val)

        freq = rng.uniform(1, 5)
        phase = rng.uniform(0, 2 * np.pi)
        amp = val * 0.1
        fluctuation = amp * np.sin(2 * np.pi * freq * t + phase)

        burst_signal = np.zeros(seq_len)
        if key not in _LINEAR_STRICT_KEYS:
            num_bursts = rng.randint(0, 3)
            for _ in range(num_bursts):
                bc = rng.uniform(0.1, 0.9)
                bw = rng.uniform(0.02, 0.1)
                ba = val * rng.uniform(0.2, 0.5)
                burst_signal += ba * np.exp(-((t - bc) ** 2) / (2 * bw ** 2))

        noise = rng.normal(0, 0.02 * (val + 1e-6), size=seq_len)
        final_curve = base_curve + fluctuation + burst_signal + noise
        seq_matrix.append(final_curve)

    seq_matrix = np.stack(seq_matrix, axis=1)
    seq_matrix = np.clip(seq_matrix, a_min=0.0, a_max=1.5)
    return seq_matrix.astype(np.float32)


# =====================================================
# Transformer 模型组件 (与 Pseudo-TS 版完全一致)
# =====================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeSeriesEncoder(nn.Module):
    def __init__(self, in_dim, d_model, nhead, num_layers, dropout):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)

    def forward(self, src, src_key_padding_mask):
        x = self.input_proj(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        mask = src_key_padding_mask
        if mask is None:
            return out.mean(dim=1)
        mask_expanded = mask.unsqueeze(-1).expand_as(out)
        out = out.masked_fill(mask_expanded, 0.0)
        non_pad_counts = (~mask).sum(dim=1).unsqueeze(1).clamp(min=1).to(out.dtype)
        return out.sum(dim=1) / non_pad_counts


class BotClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.d_model, 1)

    def forward(self, src, mask):
        emb = self.encoder(src, src_key_padding_mask=mask)
        return self.classifier(emb)


# --- 数据集 ---
class NpzSplitDataset(Dataset):
    def __init__(self, npz_path, keep_unlabeled=False):
        data = np.load(npz_path, allow_pickle=True)
        self.matrices = data['matrices']
        self.labels = data['labels']
        self.ids = data['ids']
        if not keep_unlabeled:
            mask = (self.labels == 0) | (self.labels == 1)
            self.matrices = self.matrices[mask]
            self.labels = self.labels[mask]
            self.ids = self.ids[mask]
        self.labels = self.labels.astype(np.int8)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.matrices[idx], int(self.labels[idx]), str(self.ids[idx])


def collate_fixed(batch):
    matrices, labels, ids = zip(*batch)
    matrices = torch.tensor(np.stack(matrices, axis=0), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    mask = torch.zeros((matrices.size(0), matrices.size(1)), dtype=torch.bool)
    return matrices, mask, labels, list(ids)


# --- 训练 / 评估 ---
def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    for matrices, mask, labels, _ in tqdm(loader, desc="训练"):
        matrices, mask, labels = matrices.to(device), mask.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(matrices, mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for matrices, mask, labels, _ in tqdm(loader, desc="评估"):
            matrices, mask, labels = matrices.to(device), mask.to(device), labels.to(device)
            logits = model(matrices, mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy().reshape(-1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().reshape(-1).tolist())
    if not all_labels:
        return float('nan'), 0.0, 0.0
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return total_loss / max(1, len(loader)), acc, f1


# =====================================================
# 主管道: 构建 Real-TS / Hybrid-TS 矩阵
# =====================================================
def build_real_ts_matrices(data_dir, tmp_dir, seq_len=SEQ_LEN, mode="real", drop_rate=0.0):
    """
    从推文时间戳构建真实时间序列矩阵。

    两遍扫描策略:
      Pass 1: 加载 user.json, 计算每用户的时间桶边界
      Pass 2: 流式扫描 tweet_*.json, 逐条聚合到对应桶

    mode="real":   输出纯 Real-TS 矩阵
    mode="hybrid": 输出 λ*Real + (1-λ)*Pseudo 混合矩阵
    drop_rate: 随机丢弃推文时间戳的比例 (0.0=不丢弃, 0.75=丢弃 75%)
    """
    if drop_rate > 0:
        logging.info(f"\u26a0\ufe0f 时间戳丢弃率: {drop_rate:.0%} (保留率 ρ={(1-drop_rate):.0%})")
        drop_rng = np.random.RandomState(42)  # 固定种子保证可复现
    else:
        drop_rng = None
    user_json_path = os.path.join(data_dir, "user.json")
    split_csv_path = os.path.join(data_dir, "split.csv")
    label_csv_path = os.path.join(data_dir, "label.csv")

    # --- 加载标签 ---
    logging.info("加载标签...")
    label_map = {}
    with open(label_csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label_map[row["id"]] = 1 if row["label"] == "bot" else 0

    # --- 加载划分 ---
    logging.info("加载划分...")
    split_map = {}
    with open(split_csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split_map[row["id"]] = row["split"]

    # --- Pass 1: 加载用户, 计算桶边界 ---
    logging.info(f"Pass 1: 加载 user.json, 计算时间桶边界...")
    with open(user_json_path, "r", encoding="utf-8") as f:
        all_users = json.load(f)
    logging.info(f"共 {len(all_users)} 个用户")

    crawl_epoch = CRAWL_DATE.timestamp()

    # uid → {user_data, split, created_epoch, bucket_edges}
    user_info = {}
    uid_to_author_id = {}  # author_id (纯数字) → uid (带 u 前缀)

    for u in tqdm(all_users, desc="计算桶边界"):
        uid = str(u.get("id", ""))
        created_at = _parse_iso_date(u.get("created_at"))

        if created_at:
            created_epoch = created_at.timestamp()
        else:
            created_epoch = crawl_epoch - 365 * 24 * 3600  # 默认 1 年前

        # 时间桶边界: 从创建到采集日均匀分 T+1 个点
        edges = np.linspace(created_epoch, crawl_epoch, seq_len + 1)

        user_info[uid] = {
            "user_data": u,
            "split": split_map.get(uid),
            "created_epoch": created_epoch,
            "bucket_edges": edges,
        }

        # author_id 映射 (tweet 中是纯数字, user.json 中带 u 前缀)
        raw_id = uid
        if raw_id.startswith("u"):
            uid_to_author_id[raw_id[1:]] = uid
        else:
            uid_to_author_id[raw_id] = uid

    logging.info(f"用户信息加载完成, 共 {len(user_info)} 用户")

    # 初始化聚合器
    bucket_edges_dict = {uid: info["bucket_edges"] for uid, info in user_info.items()}
    aggregator = UserBucketAggregator(len(user_info), seq_len, bucket_edges_dict)

    # --- Pass 2: 流式扫描推文 ---
    tweet_files = sorted([f for f in os.listdir(data_dir) if f.startswith("tweet_") and f.endswith(".json")])
    logging.info(f"Pass 2: 扫描 {len(tweet_files)} 个推文文件...")

    total_tweets = 0
    matched_tweets = 0

    for tf_name in tweet_files:
        tf_path = os.path.join(data_dir, tf_name)
        file_size_gb = os.path.getsize(tf_path) / 1e9
        logging.info(f"处理 {tf_name} ({file_size_gb:.2f} GB)")

        with open(tf_path, "r", encoding="utf-8") as f:
            tweets = json.load(f)

        for tw in tqdm(tweets, desc=tf_name, mininterval=5):
            total_tweets += 1
            author_id = tw.get("author_id")
            if author_id is None:
                continue

            # 解析推文时间戳
            tw_created = _parse_iso_date(tw.get("created_at"))
            if tw_created is None:
                continue

            # 映射 author_id → uid
            uid = uid_to_author_id.get(str(author_id))
            if uid is None:
                continue

            # 时间戳随机丢弃 (用于时间缺失敏感性实验)
            if drop_rng is not None and drop_rng.random() < drop_rate:
                continue

            tw_epoch = tw_created.timestamp()
            feat_tuple = _tweet_features(tw)
            aggregator.add_tweet(uid, tw_epoch, feat_tuple)
            matched_tweets += 1

        del tweets
        import gc; gc.collect()

    logging.info(f"推文扫描完成: 总推文={total_tweets}, 匹配={matched_tweets}, "
                 f"有推文用户={len(aggregator.agg)}")

    # --- Pseudo-TS scaler (for hybrid mode) ---
    pseudo_scaler = None
    pseudo_feature_cols = None
    if mode == "hybrid":
        logging.info("Hybrid 模式: 拟合 Pseudo-TS scaler...")
        static_list = []
        for uid, info in user_info.items():
            if info["split"] == "train":
                sf = calculate_static_features_twibot22(info["user_data"])
                static_list.append(sf)
        static_df = pd.DataFrame(static_list)
        pseudo_feature_cols = [c for c in static_df.columns if c != 'user_id']
        pseudo_scaler = MinMaxScaler()
        pseudo_scaler.fit(static_df[pseudo_feature_cols].values)
        del static_list, static_df
        logging.info("Pseudo scaler 拟合完成")

    # --- 构建并保存矩阵 ---
    os.makedirs(tmp_dir, exist_ok=True)

    split_groups = {"train": [], "val": [], "test": [], "unlabeled": []}
    for uid, info in user_info.items():
        s = info["split"]
        if s and s in split_groups:
            split_groups[s].append(uid)
        else:
            split_groups["unlabeled"].append(uid)

    split_name_map = {"train": "train", "val": "dev", "test": "test", "unlabeled": "support"}

    # Real-TS scaler: 从训练集拟合
    logging.info("从训练集拟合 Real-TS scaler...")
    train_matrices_raw = []
    for uid in tqdm(split_groups["train"], desc="训练集 Real-TS"):
        mat, _ = aggregator.build_matrix(uid)
        train_matrices_raw.append(mat)

    if train_matrices_raw:
        train_stack = np.stack(train_matrices_raw, axis=0)  # (N, T, 14)
        # 对每个特征维度拟合 scaler (展平为 (N*T, 14))
        flat = train_stack.reshape(-1, N_BUCKET_FEATURES)
        real_scaler = MinMaxScaler()
        real_scaler.fit(flat)
        del flat, train_stack, train_matrices_raw
    else:
        real_scaler = None
        logging.warning("训练集无 Real-TS 数据!")

    # 遍历每个 split, 构建最终矩阵
    for split_key, out_name in split_name_map.items():
        uids = split_groups[split_key]
        if not uids:
            logging.info(f"跳过空 split: {split_key}")
            continue

        logging.info(f"构建 {split_key} → {out_name} ({len(uids)} 用户, mode={mode})")
        matrices, labels, ids, modes_used = [], [], [], []

        for uid in tqdm(uids, desc=f"处理 {out_name}"):
            info = user_info[uid]

            # Real-TS matrix
            real_mat, non_empty = aggregator.build_matrix(uid)

            # 归一化
            if real_scaler is not None:
                flat = real_mat.reshape(-1, N_BUCKET_FEATURES)
                real_mat = real_scaler.transform(flat).reshape(seq_len, N_BUCKET_FEATURES).astype(np.float32)

            if mode == "real":
                final_mat = real_mat
                mode_used = "real"
            elif mode == "hybrid":
                # Pseudo-TS matrix
                sf = calculate_static_features_twibot22(info["user_data"])
                feat_vector = np.array([sf[col] for col in pseudo_feature_cols], dtype=float).reshape(1, -1)
                feat_scaled = pseudo_scaler.transform(feat_vector)[0]
                feat_dict = {k: float(v) for k, v in zip(pseudo_feature_cols, feat_scaled)}
                feat_dict['user_id'] = uid
                pseudo_mat = build_pseudo_timeseries_from_features(feat_dict, seq_len=seq_len)

                # λ = 非空桶比例
                lam = non_empty / seq_len
                final_mat = (lam * real_mat + (1 - lam) * pseudo_mat).astype(np.float32)
                mode_used = f"hybrid(λ={lam:.2f})"
            else:
                final_mat = real_mat
                mode_used = "real"

            matrices.append(final_mat)
            labels.append(label_map.get(uid, -1))
            ids.append(uid)
            modes_used.append(mode_used)

        npz_path = os.path.join(tmp_dir, f"{out_name}_matrices.npz")
        mats_arr = np.stack(matrices, axis=0).astype(np.float32) if matrices else np.zeros((0, seq_len, N_BUCKET_FEATURES), dtype=np.float32)
        np.savez_compressed(npz_path,
                            matrices=mats_arr,
                            labels=np.array(labels, dtype=np.int8),
                            ids=np.array(ids, dtype=object),
                            modes=np.array(modes_used, dtype=object))

        # 统计
        if mode == "hybrid":
            lam_values = []
            for m in modes_used:
                if "λ=" in m:
                    lam_values.append(float(m.split("λ=")[1].rstrip(")")))
            if lam_values:
                logging.info(f"  λ 统计: mean={np.mean(lam_values):.3f}, "
                             f"median={np.median(lam_values):.3f}, "
                             f"min={np.min(lam_values):.3f}, max={np.max(lam_values):.3f}")

        logging.info(f"  已保存 {npz_path}: {len(matrices)} 样本")

    # 保存 scaler
    if real_scaler is not None:
        scaler_path = os.path.join(tmp_dir, "real_scaler_params.npz")
        np.savez_compressed(scaler_path, min=real_scaler.data_min_, max=real_scaler.data_max_)
        logging.info(f"Real scaler 已保存: {scaler_path}")

    logging.info("所有 splits 处理完成。")

    # 释放内存
    del user_info, aggregator
    import gc; gc.collect()


# =====================================================
# 主管道: Transformer 训练 + 嵌入提取
# =====================================================
def main(temporal_mode, data_dir, work_dir, drop_rate=0.0):
    mode_suffix = temporal_mode  # "real" or "hybrid"
    if drop_rate > 0:
        drop_tag = f"_drop{drop_rate:.2f}".replace(".", "")
        tmp_dir = os.path.join(work_dir, f"tmp_twibot22_{mode_suffix}{drop_tag}")
    else:
        drop_tag = ""
        tmp_dir = os.path.join(work_dir, f"tmp_twibot22_{mode_suffix}")

    model_save_dir = os.path.join(work_dir, "feature_model_outputs")
    model_save_path = os.path.join(model_save_dir, f"best_timeseries_model_twibot22_{mode_suffix}{drop_tag}.pt")
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    logging.info(f"设备: {DEVICE} | 时序模式: {temporal_mode} | drop_rate: {drop_rate:.2f}")
    logging.info(f"数据目录: {data_dir}")
    logging.info(f"工作目录: {work_dir}")
    logging.info(f"临时目录: {tmp_dir}")

    # 0) 构建矩阵
    expected = [os.path.join(tmp_dir, f"{s}_matrices.npz") for s in ['train', 'dev', 'test']]
    if not all(os.path.exists(p) for p in expected):
        logging.info(f"构建 {temporal_mode} 时序矩阵 (drop_rate={drop_rate:.2f})...")
        build_real_ts_matrices(data_dir, tmp_dir, seq_len=SEQ_LEN, mode=temporal_mode, drop_rate=drop_rate)
    else:
        logging.info(f"找到已有 .npz 文件, 跳过构建。如需重建请删除 {tmp_dir}/ 下的 .npz 文件。")

    # 1) 加载数据集
    train_npz = os.path.join(tmp_dir, "train_matrices.npz")
    dev_npz = os.path.join(tmp_dir, "dev_matrices.npz")

    train_dataset = NpzSplitDataset(train_npz, keep_unlabeled=False)
    dev_dataset = NpzSplitDataset(dev_npz, keep_unlabeled=False) if os.path.exists(dev_npz) else None

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fixed)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fixed) if dev_dataset else None

    logging.info(f"训练样本: {len(train_dataset)}; 开发样本: {len(dev_dataset) if dev_dataset else 0}")

    # 2) 模型
    tmp = np.load(train_npz, allow_pickle=True)
    in_dim = tmp['matrices'].shape[2]
    logging.info(f"输入维度: {in_dim}")

    # 计算类别权重 (sqrt策略, 与 train_twibot22.py 一致)
    train_labels = train_dataset.labels
    n_neg = int((train_labels == 0).sum())
    n_pos = int((train_labels == 1).sum())
    if n_pos > 0:
        raw_weight = n_neg / n_pos
        sqrt_weight = float(np.sqrt(raw_weight))
    else:
        sqrt_weight = 1.0
    logging.info(f"类别分布: neg={n_neg}, pos={n_pos}, raw_ratio={n_neg/(n_pos+1e-9):.2f}, sqrt_pos_weight={sqrt_weight:.4f}")

    encoder = TimeSeriesEncoder(in_dim=in_dim, d_model=D_MODEL, nhead=N_HEAD,
                                num_layers=NUM_ENCODER_LAYERS, dropout=DROPOUT)
    model = BotClassifier(encoder).to(DEVICE)
    pos_weight = torch.tensor([sqrt_weight], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3) 训练 (F1 选模 + early stopping)
    PATIENCE = 15
    best_f1 = -1.0
    patience_counter = 0
    for epoch in range(NUM_EPOCHS):
        logging.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        logging.info(f"  train loss: {train_loss:.4f}")

        if dev_loader:
            val_loss, val_acc, val_f1 = evaluate(model, dev_loader, criterion, DEVICE)
            logging.info(f"  val loss: {val_loss:.4f} | val acc: {val_acc:.4f} | val F1: {val_f1:.4f}")
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), model_save_path)
                logging.info(f"  ★ 新最佳模型 (F1={val_f1:.4f}, acc={val_acc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    logging.info(f"  Early stopping: {PATIENCE} epochs 无 F1 提升")
                    break
        else:
            torch.save(model.state_dict(), model_save_path)

    # 4) 推理: 提取所有 split 的嵌入
    logging.info("加载最佳模型进行推理...")
    model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
    model.eval()

    all_embeddings = {}
    for split in ['train', 'dev', 'test', 'support']:
        npz_path = os.path.join(tmp_dir, f"{split}_matrices.npz")
        if not os.path.exists(npz_path):
            continue
        logging.info(f"提取嵌入: {split}")
        ds = NpzSplitDataset(npz_path, keep_unlabeled=True)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fixed)
        with torch.no_grad():
            for matrices, mask, _, ids_batch in tqdm(loader, desc=f"嵌入 {split}"):
                matrices, mask = matrices.to(DEVICE), mask.to(DEVICE)
                emb = model.encoder(matrices, src_key_padding_mask=mask).cpu().numpy()
                for i, uid in enumerate(ids_batch):
                    all_embeddings[uid] = emb[i]

    # 5) 保存嵌入
    logging.info(f"总嵌入数: {len(all_embeddings)}")
    user_ids_ordered = list(all_embeddings.keys())
    vectors = np.stack([all_embeddings[uid] for uid in user_ids_ordered], axis=0).astype(np.float32)
    out_path = os.path.join(model_save_dir, f"twibot22_transformer_vectors_{mode_suffix}{drop_tag}.npz")
    np.savez_compressed(out_path, vectors=vectors, user_ids=np.array(user_ids_ordered, dtype=object))
    logging.info(f"嵌入已保存: {out_path} (形状={vectors.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TwiBot-22 真实时间序列特征提取")
    parser.add_argument("--mode", default="real", choices=["real", "hybrid"],
                        help="时序构造模式: real=纯真实时间序列, hybrid=Real+Pseudo混合")
    parser.add_argument("--data-dir", default=os.path.join(SCRIPT_DIR, "Data", "TwiBot-22"),
                        help="TwiBot-22 原始数据目录 (含 user.json, tweet_*.json)")
    parser.add_argument("--work-dir", default=SCRIPT_DIR,
                        help="工作目录 (保存临时文件和输出)")
    parser.add_argument("--drop-rate", type=float, default=0.0,
                        help="随机丢弃推文时间戳的比例 (0.0=不丢弃, 0.75=保留 25%%). "
                             "用于时间信息缺失敏感性实验 (\u00a75.7)")
    args = parser.parse_args()
    main(temporal_mode=args.mode, data_dir=args.data_dir, work_dir=args.work_dir,
         drop_rate=args.drop_rate)
