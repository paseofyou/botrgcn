#!/usr/bin/env python3
"""
TwiBot-22 时序特征提取脚本
==========================
适配 TwiBot-22 数据格式的特征提取管道。
复用 feature_extraction.py 中的核心组件（Transformer、时序构建器等），
仅重写数据加载和静态特征提取部分。

用法:
    python feature_extraction_twibot22.py --mode pseudo
    python feature_extraction_twibot22.py --mode pseudo --data-dir /path/to/data --work-dir /path/to/work
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
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score

# ----------------------
# 配置 (可通过命令行覆盖)
# ----------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# TwiBot-22 的采集日期 (约 2022 年中)
CRAWL_DATE_STR = '2022-06-01'
CRAWL_DATE = datetime.datetime.strptime(CRAWL_DATE_STR, '%Y-%m-%d')

# 模型 / 训练配置
SEQ_LEN = 32
D_MODEL = 64
N_HEAD = 8
NUM_ENCODER_LAYERS = 2
DROPOUT = 0.1
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
NUM_EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
torch.manual_seed(42)
np.random.seed(42)


# =====================================================
# TwiBot-22 专用: 静态特征提取
# =====================================================
def _parse_iso_date(date_str):
    """解析 TwiBot-22 的 ISO 格式时间戳，如 '2020-01-16 02:02:55+00:00'"""
    if not date_str:
        return None
    try:
        # 去掉时区部分简化解析
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


def calculate_static_features_twibot22(user_data):
    """
    从 TwiBot-22 用户数据提取 14 维静态特征。
    字段映射与 TwiBot-20 版保持相同的字典 key，以复用时序构建函数。
    """
    pm = user_data.get('public_metrics') or {}
    followers_count = pm.get('followers_count', 0) or 0
    following_count = pm.get('following_count', 0) or 0
    tweet_count = pm.get('tweet_count', 0) or 0
    listed_count = pm.get('listed_count', 0) or 0

    created_at = _parse_iso_date(user_data.get('created_at'))
    age_days = max(1, (CRAWL_DATE - created_at).days) if created_at else 1

    # 对数增长率
    log_tweets_per_day = np.log1p(tweet_count) / age_days
    log_followers_per_day = np.log1p(followers_count) / age_days
    # TwiBot-22 无 favourites_count → 用 log_listed_per_day 替代
    log_likes_per_day = np.log1p(listed_count) / age_days

    follow_balance = following_count / (followers_count + 1e-6)
    follow_balance = np.clip(follow_balance, 0, 100)

    # TwiBot-22 用户数据中不含嵌入的推文文本，以下用用户级统计替代
    description = user_data.get('description', '') or ''
    avg_tweet_length = len(description) / 100.0  # 用描述长度替代
    std_tweet_length = len(user_data.get('username', '') or '') / 15.0  # 用用户名长度替代
    url_ratio = followers_count / (tweet_count + 1e-6)  # followers/tweets 比
    url_ratio = np.clip(url_ratio, 0, 1000)
    interaction_rate = tweet_count / (following_count + 1e-6)  # 发推/关注比
    interaction_rate = np.clip(interaction_rate, 0, 1000)
    topic_diversity = listed_count / (following_count + 1e-6)  # listed/following 比
    topic_diversity = np.clip(topic_diversity, 0, 100)

    # 二值特征
    has_default_profile_image = float(bool(user_data.get('url', '')))  # 用 has_url 替代
    has_url_in_description = float(bool(re.search(r'https?://\S+|www\.\S+', description)))

    followers_interaction_index = listed_count / (followers_count + 1e-6)
    activity_index = tweet_count / (followers_count + 1e-6)
    activity_index = np.clip(activity_index, 0, 1000)

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


# =====================================================
# 从 feature_extraction.py 复用的核心组件
# =====================================================

# --- 时序构建常量 ---
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
_STABLE_KEYS = {
    'interaction_rate', 'topic_diversity', 'follow_balance',
    'url_ratio', 'avg_tweet_length', 'std_tweet_length',
    'has_default_profile_image', 'has_url_in_description',
    'followers_interaction_index'
}


def build_pseudo_timeseries_from_features(feat_dict, seq_len=SEQ_LEN):
    """伪时间序列生成器 (与 TwiBot-20 版完全一致)"""
    keys = _TS_KEYS
    F = len(keys)
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


def build_anchor_timeseries_from_features(feat_dict, account_age_days_raw, seq_len=SEQ_LEN):
    """生命周期锚定版伪时间序列 (与 TwiBot-20 版完全一致)"""
    keys = _TS_KEYS
    tau = np.linspace(0, 1, seq_len)
    maturity = np.clip(account_age_days_raw / 3650.0, 0.01, 1.0)

    user_id_str = str(feat_dict.get('user_id', '0'))
    user_seed = int(hash(user_id_str) % 100000)
    rng = np.random.RandomState(user_seed)
    vals = np.array([float(feat_dict.get(k, 0.0)) for k in keys], dtype=float)

    seq_matrix = []
    for i, key in enumerate(keys):
        val = vals[i]
        if key in _LINEAR_STRICT_KEYS:
            base_curve = val * tau
        elif key in _GROWTH_KEYS:
            if rng.random() < 0.5:
                k_steep = 3.0 + (1.0 - maturity) * 8.0
                t0 = 0.2 + maturity * 0.4
                base_curve = val / (1.0 + np.exp(-k_steep * (tau - t0)))
            else:
                base_curve = val * np.log1p(9 * tau) / np.log1p(9)
        else:
            base_curve = np.full(seq_len, val)

        freq = rng.uniform(1, 5)
        phase = rng.uniform(0, 2 * np.pi)
        amp = val * 0.1
        fluctuation = amp * np.sin(2 * np.pi * freq * tau + phase)

        burst_signal = np.zeros(seq_len)
        if key not in _LINEAR_STRICT_KEYS:
            num_bursts = rng.randint(0, 3)
            for _ in range(num_bursts):
                bc = rng.uniform(0.1, 0.9)
                bw = rng.uniform(0.02, 0.1)
                ba = val * rng.uniform(0.15, 0.4)
                burst_signal += ba * np.exp(-((tau - bc) ** 2) / (2 * bw ** 2))

        noise = rng.normal(0, 0.02 * (val + 1e-6), size=seq_len)
        final_curve = base_curve + fluctuation + burst_signal + noise
        seq_matrix.append(final_curve)

    seq_matrix = np.stack(seq_matrix, axis=1)
    seq_matrix = np.clip(seq_matrix, a_min=0.0, a_max=1.5)
    return seq_matrix.astype(np.float32)


def build_temporal_sequence(user_data, feat_dict, mode="pseudo", seq_len=SEQ_LEN,
                            account_age_days_raw=None, **kwargs):
    """统一时间序列构造入口"""
    if mode == "anchor":
        if account_age_days_raw is None:
            matrix = build_pseudo_timeseries_from_features(feat_dict, seq_len=seq_len)
            mode = "pseudo"
        else:
            matrix = build_anchor_timeseries_from_features(feat_dict, account_age_days_raw, seq_len=seq_len)
    else:
        matrix = build_pseudo_timeseries_from_features(feat_dict, seq_len=seq_len)
        mode = "pseudo"
    return {"matrix": matrix, "mode_used": mode}


# --- Transformer 模型组件 ---
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
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for matrices, mask, labels, _ in tqdm(loader, desc="训练"):
        matrices, mask, labels = matrices.to(device), mask.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(matrices, mask)
        loss = criterion(logits, labels)
        loss.backward()
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
# TwiBot-22 专用: 构建并保存所有 split 的矩阵
# =====================================================
def build_and_save_all_splits_twibot22(data_dir, tmp_dir, seq_len=SEQ_LEN, temporal_mode="pseudo"):
    """
    从 TwiBot-22 user.json + split.csv + label.csv 构建时序矩阵。
    """
    user_json_path = os.path.join(data_dir, "user.json")
    split_csv_path = os.path.join(data_dir, "split.csv")
    label_csv_path = os.path.join(data_dir, "label.csv")

    # 加载标签
    logging.info("加载标签...")
    label_map_csv = {}
    with open(label_csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label_map_csv[row["id"]] = 1 if row["label"] == "bot" else 0

    # 加载划分
    logging.info("加载划分...")
    split_map = {}  # uid → split_name
    with open(split_csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split_map[row["id"]] = row["split"]  # train/val/test

    # 加载所有用户
    logging.info(f"加载 user.json ({os.path.getsize(user_json_path) / 1e9:.2f} GB)...")
    with open(user_json_path, "r", encoding="utf-8") as f:
        all_users = json.load(f)
    logging.info(f"共 {len(all_users)} 个用户")

    # 按 split 分组用户
    split_users = {"train": [], "val": [], "test": [], "unlabeled": []}
    for u in all_users:
        uid = str(u.get("id", ""))
        s = split_map.get(uid, None)
        if s and s in split_users:
            split_users[s].append(u)
        else:
            split_users["unlabeled"].append(u)

    for s, users in split_users.items():
        logging.info(f"  {s}: {len(users)} 用户")

    del all_users  # 释放内存

    # 第一阶段: 从训练集拟合 scaler
    logging.info("第一阶段: 从训练集拟合 scaler...")
    static_list = []
    for u in tqdm(split_users["train"], desc="训练集静态特征"):
        sf = calculate_static_features_twibot22(u)
        static_list.append(sf)
    static_df = pd.DataFrame(static_list)
    feature_cols = [c for c in static_df.columns if c != 'user_id']
    scaler = MinMaxScaler()
    scaler.fit(static_df[feature_cols].values)
    logging.info("Scaler 已拟合。")

    # 第二阶段: 遍历每个 split 生成矩阵
    os.makedirs(tmp_dir, exist_ok=True)

    # TwiBot-22 的 split 名称映射 (val → dev 以与 TwiBot-20 兼容)
    split_name_map = {"train": "train", "val": "dev", "test": "test", "unlabeled": "support"}

    for split_key, out_name in split_name_map.items():
        users = split_users[split_key]
        if not users:
            logging.info(f"跳过空的 split: {split_key}")
            continue

        logging.info(f"处理 {split_key} → {out_name} ({len(users)} 用户)")
        matrices, labels, ids, modes_used = [], [], [], []

        for u in tqdm(users, desc=f"处理 {out_name}"):
            sf = calculate_static_features_twibot22(u)
            uid = sf['user_id']
            raw_age = sf['account_age_days']

            feat_vector = np.array([sf[col] for col in feature_cols], dtype=float).reshape(1, -1)
            feat_scaled = scaler.transform(feat_vector)[0]
            feat_dict = {k: float(v) for k, v in zip(feature_cols, feat_scaled)}

            ts_result = build_temporal_sequence(
                user_data=u, feat_dict=feat_dict, mode=temporal_mode,
                seq_len=seq_len, account_age_days_raw=raw_age
            )
            matrices.append(ts_result["matrix"])
            modes_used.append(ts_result["mode_used"])
            labels.append(label_map_csv.get(uid, -1))
            ids.append(uid)

        npz_path = os.path.join(tmp_dir, f"{out_name}_matrices.npz")
        mats_arr = np.stack(matrices, axis=0).astype(np.float32) if matrices else np.zeros((0, seq_len, 14), dtype=np.float32)
        np.savez_compressed(npz_path,
                            matrices=mats_arr,
                            labels=np.array(labels, dtype=np.int8),
                            ids=np.array(ids, dtype=object),
                            modes=np.array(modes_used, dtype=object))
        mode_dist = Counter(modes_used)
        logging.info(f"  已保存 {npz_path}: {len(matrices)} 样本, 模式分布: {dict(mode_dist)}")

    # 保存 scaler
    scaler_path = os.path.join(tmp_dir, "scaler_params.npz")
    np.savez_compressed(scaler_path, min=scaler.data_min_, max=scaler.data_max_,
                        feature_cols=np.array(feature_cols, dtype=object))
    logging.info("所有 splits 处理完成。")


# =====================================================
# 主管道
# =====================================================
def main(temporal_mode, data_dir, work_dir, seq_len=None, num_layers=None):
    seq_len = seq_len or SEQ_LEN
    num_layers = num_layers or NUM_ENCODER_LAYERS

    # 参数化目录: 不同 mode / T / L 组合保存到不同子目录, 避免互相覆盖
    if seq_len == SEQ_LEN and num_layers == NUM_ENCODER_LAYERS:
        suffix = f"_{temporal_mode}"
    else:
        suffix = f"_{temporal_mode}_T{seq_len}_L{num_layers}"
    tmp_dir = os.path.join(work_dir, f"tmp_twibot22{suffix}")

    model_save_dir = os.path.join(work_dir, "feature_model_outputs")
    model_save_path = os.path.join(model_save_dir, f"best_timeseries_model_twibot22{suffix}.pt")
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    logging.info(f"设备: {DEVICE} | 时序模式: {temporal_mode} | T={seq_len} | L={num_layers}")
    logging.info(f"数据目录: {data_dir}")
    logging.info(f"工作目录: {work_dir}")
    logging.info(f"临时目录: {tmp_dir}")

    # 0) 构建矩阵
    expected = [os.path.join(tmp_dir, f"{s}_matrices.npz") for s in ['train', 'dev', 'test']]
    if not all(os.path.exists(p) for p in expected):
        logging.info(f"正在构建时序矩阵 (T={seq_len})...")
        build_and_save_all_splits_twibot22(data_dir, tmp_dir, seq_len=seq_len, temporal_mode=temporal_mode)
    else:
        logging.info("找到已有 .npz 文件，跳过构建。如需重建请删除对应 tmp 目录下的 .npz 文件。")

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

    # 类别权重 (sqrt策略, 与 feature_extraction_twibot22_real.py 完全一致,
    # 否则 pseudo / real 的时序编码器训练目标不同, 两者不可比)
    train_labels = train_dataset.labels
    n_neg = int((train_labels == 0).sum())
    n_pos = int((train_labels == 1).sum())
    sqrt_weight = float(np.sqrt(n_neg / n_pos)) if n_pos > 0 else 1.0
    logging.info(f"类别分布: neg={n_neg}, pos={n_pos}, "
                 f"raw_ratio={n_neg / (n_pos + 1e-9):.2f}, sqrt_pos_weight={sqrt_weight:.4f}")

    encoder = TimeSeriesEncoder(in_dim=in_dim, d_model=D_MODEL, nhead=N_HEAD,
                                num_layers=num_layers, dropout=DROPOUT)
    model = BotClassifier(encoder).to(DEVICE)
    pos_weight = torch.tensor([sqrt_weight], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3) 训练
    best_acc = -1.0
    for epoch in range(NUM_EPOCHS):
        logging.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        logging.info(f"  train loss: {train_loss:.4f}")

        if dev_loader:
            val_loss, val_acc, val_f1 = evaluate(model, dev_loader, criterion, DEVICE)
            logging.info(f"  val loss: {val_loss:.4f} | val acc: {val_acc:.4f} | val F1: {val_f1:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_save_path)
                logging.info(f"  ★ 新最佳模型 (acc={val_acc:.4f})")
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
    out_name = f"twibot22_transformer_vectors{suffix}.npz"
    out_path = os.path.join(model_save_dir, out_name)
    np.savez_compressed(out_path, vectors=vectors, user_ids=np.array(user_ids_ordered, dtype=object))
    logging.info(f"嵌入已保存: {out_path} (形状={vectors.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TwiBot-22 时序特征提取")
    parser.add_argument("--mode", default="pseudo", choices=["pseudo", "anchor"],
                        help="时序构造模式")
    parser.add_argument("--data-dir", default=os.path.join(SCRIPT_DIR, "Data", "TwiBot-22"),
                        help="TwiBot-22 原始数据目录")
    parser.add_argument("--work-dir", default=SCRIPT_DIR,
                        help="工作目录 (保存临时文件和输出)")
    parser.add_argument("--seq-len", type=int, default=None,
                        help=f"时间序列长度 (默认: {SEQ_LEN})")
    parser.add_argument("--num-layers", type=int, default=None,
                        help=f"Transformer编码层数 (默认: {NUM_ENCODER_LAYERS})")
    args = parser.parse_args()
    main(temporal_mode=args.mode, data_dir=args.data_dir, work_dir=args.work_dir,
         seq_len=args.seq_len, num_layers=args.num_layers)
