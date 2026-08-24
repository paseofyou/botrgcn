#!/usr/bin/env python3
# feature_extraction_v6.py
"""
BotRGCN 特征提取模块 v6 — 单文件训练 + 推理脚本
- 修复了 v5 中的所有致命 bug（mask shape、label 过滤、保存/加载、ID 对齐等）
- 提供稳健的伪时间序列生成（固定长度）
- 流式保存每个 split 的矩阵到 disk (.npz)，训练/推理从磁盘读取以节省内存
"""

import os
import json
import math
import logging
import datetime
from collections import Counter
import re

import matplotlib.pyplot as plt
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
# 代码现在位于 Twibot_20/src/，而数据/产物位于 Twibot_20/ 下
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "Data", "Twibot-20")
FEATURE_OUTPUT_DIR = os.path.join(BASE_DIR, "feature_model_outputs")
TWITTER_DATE_FORMAT = '%a %b %d %H:%M:%S +0000 %Y'
CRAWL_DATE_STR = '2020-09-01'
CRAWL_DATE = datetime.datetime.strptime(CRAWL_DATE_STR, '%Y-%m-%d')

# 模型 / 训练配置
SEQ_LEN = 32                # 固定时序长度（可调）
D_MODEL = 64
N_HEAD = 8
NUM_ENCODER_LAYERS = 2
DROPOUT = 0.1
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
NUM_EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = os.path.join(FEATURE_OUTPUT_DIR, "best_timeseries_model.pt")

# 临时文件 (每个分割)
TMP_DIR = os.path.join(BASE_DIR, "tmp", "tmp_v6")
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(FEATURE_OUTPUT_DIR, exist_ok=True)

# 日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
torch.manual_seed(42)
np.random.seed(42)

# ----------------------
# 工具函数 & 特征提取
# ----------------------
def _safe_str_to_int(s, default=0):
    if s is None:
        return default
    try:
        return int(float(str(s).strip()))
    except (ValueError, TypeError):
        return default

def _parse_twitter_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.datetime.strptime(str(date_str).strip(), TWITTER_DATE_FORMAT)
    except Exception:
        return None

def calculate_static_features(user_data):
    """
    提取一系列静态特征（原始数值，未归一化）
    返回 dict 包含 user_id 与若干特征
    """
    profile = user_data.get('profile', {}) or {}
    tweets = user_data.get('tweet', []) or []
    created_at = _parse_twitter_date(profile.get('created_at'))
    age_days = max(1, (CRAWL_DATE - created_at).days) if created_at else 1

    statuses_count = _safe_str_to_int(profile.get('statuses_count'))
    followers_count = _safe_str_to_int(profile.get('followers_count'))
    favourites_count = _safe_str_to_int(profile.get('favourites_count'))
    friends_count = _safe_str_to_int(profile.get('friends_count'))

    # 计算对数增长率 (更稳健的特征)
    log_tweets_per_day = np.log1p(statuses_count) / age_days
    log_followers_per_day = np.log1p(followers_count) / age_days
    log_likes_per_day = np.log1p(favourites_count) / age_days

    follow_balance = friends_count / followers_count if followers_count > 0 else 0.0
    follow_balance = np.clip(follow_balance, 0, 100)

    avg_tweet_length = 0.0
    std_tweet_length = 0.0
    url_ratio = 0.0
    interaction_rate = 0.0
    topic_diversity = 0.0

    if tweets:
        tweet_texts = [str(t) for t in tweets if t]
        tweet_lengths = [len(t) for t in tweet_texts]
        if tweet_lengths:
            avg_tweet_length = float(np.mean(tweet_lengths))
            std_tweet_length = float(np.std(tweet_lengths)) if len(tweet_lengths) > 1 else 0.0

        url_count = sum(1 for t in tweet_texts if re.search(r'https?://t\.co/|https?://\S+|www\.\S+', t))
        url_ratio = url_count / len(tweet_texts)

        retweets = 0
        mentions = 0
        mention_pattern = re.compile(r'@\w+')
        for t in tweet_texts:
            ct = t.strip()
            if ct.startswith("RT @"):
                retweets += 1
            elif mention_pattern.search(ct):
                mentions += 1
        interaction_rate = (retweets + mentions) / len(tweet_texts)

        hashtag_pattern = re.compile(r'#\w+')
        hashtags = [tag.lower() for t in tweet_texts for tag in hashtag_pattern.findall(t)]
        if hashtags:
            counts = Counter(hashtags)
            probs = np.array(list(counts.values())) / len(hashtags)
            # 香农熵
            topic_diversity = float(-(probs * np.log2(probs + 1e-12)).sum())

    # --- 新增的四个特征 ---
    # 修复 `default_profile_image` 可能为带空格的字符串的问题
    raw_dpi = profile.get('default_profile_image', False)
    if isinstance(raw_dpi, str):
        processed_dpi = raw_dpi.strip().lower() == 'true'
    else:
        processed_dpi = bool(raw_dpi)
    has_default_profile_image = float(processed_dpi)

    description_text = profile.get('description', '') or ""
    has_url_in_description = float(bool(re.search(r'https?://\S+|www\.\S+', description_text)))

    # 避免除以零
    followers_interaction_index = _safe_str_to_int(profile.get('listed_count')) / (followers_count + 1e-6)
    activity_index = favourites_count / (statuses_count + 1e-6)

    return {
        "user_id": user_data.get("ID", ""),
        "account_age_days": float(age_days),
        "log_tweets_per_day": float(log_tweets_per_day), # 替换
        "log_followers_per_day": float(log_followers_per_day), # 替换
        "log_likes_per_day": float(log_likes_per_day), # 替换
        "follow_balance": float(follow_balance),
        "interaction_rate": float(interaction_rate),
        "topic_diversity": float(topic_diversity),
        "avg_tweet_length": float(avg_tweet_length),
        "std_tweet_length": float(std_tweet_length),
        "url_ratio": float(url_ratio),
        # 移除原有的 log_statuses_count, log_followers_count, log_friends_count, log_favourites_count
        # 新增的四个特征
        "has_default_profile_image": has_default_profile_image,
        "has_url_in_description": has_url_in_description,
        "followers_interaction_index": followers_interaction_index,
        "activity_index": activity_index,
    }

# ----------------------
# 鲁棒的伪时间序列生成器 (增强版)
# ----------------------
def build_pseudo_timeseries_from_features(feat_dict, seq_len=SEQ_LEN):
    """
    使用缩放后的特征来生成一个固定长度 seq_len 的伪时间序列矩阵 (seq_len, F)。
    增强版改进：
      1. 引入基于 user_id 的确定性随机性 (deterministic randomness)，让不同用户的曲线更有区分度。
      2. 增加“突发”(burst) 模式和随机波动，模拟真实社交媒体行为。
      3. 【核心改进】根据特征类型分配更真实的演变趋势，而不是完全随机。
    """
    # 扩展特征列表，现在包含所有新旧静态特征
    keys = ['account_age_days', # 账号年龄，作为基础特征保留
            'log_tweets_per_day', 'log_followers_per_day', 'log_likes_per_day',
            'interaction_rate', 'topic_diversity', 'follow_balance',
            'url_ratio', 'avg_tweet_length', 'std_tweet_length', # 启用
            'has_default_profile_image', 'has_url_in_description',
            'followers_interaction_index', # 启用
            'activity_index']
    
    # --- 根据特征的现实规律进行分类 ---
    # 确定性线性增长
    LINEAR_STRICT_KEYS = {'account_age_days'}
    # S形或对数增长
    GROWTH_KEYS = {
        'log_tweets_per_day', 'log_followers_per_day', 'log_likes_per_day',
        'activity_index'
    }
    # 在基线附近波动的稳定/比率型特征
    STABLE_KEYS = {
        'interaction_rate', 'topic_diversity', 'follow_balance',
        'url_ratio', 'avg_tweet_length', 'std_tweet_length', # 添加
        'has_default_profile_image', 'has_url_in_description',
        'followers_interaction_index' # 添加
    }
    F = len(keys)
    t = np.linspace(0, 1, seq_len)  # 归一化时间轴 0..1

    # 获取 user_id 用于生成确定性种子
    user_id_str = str(feat_dict.get('user_id', '0'))
    user_seed = int(hash(user_id_str) % 100000)
    rng = np.random.RandomState(user_seed)

    vals = np.array([float(feat_dict.get(k, 0.0)) for k in keys], dtype=float)

    seq_matrix = []
    
    for i, key in enumerate(keys):
        val = vals[i]
        base_curve = np.zeros(seq_len)

        # 1. 基础趋势 (Base Trend) - 基于规则分配
        trend_type = ''
        if key in LINEAR_STRICT_KEYS:
            trend_type = 'linear_strict'
        elif key in GROWTH_KEYS:
            trend_type = rng.choice(['logistic', 'logarithmic'])
        elif key in STABLE_KEYS:
            trend_type = 'stable'
        
        if trend_type == 'linear_strict':
            # 账号年龄严格线性增长
            base_curve = val * t
        elif trend_type == 'logistic':
            # S形曲线 (模拟增长爆发)
            k = 5.0 + rng.uniform(0, 5) # 陡峭程度
            t0 = rng.uniform(0.3, 0.7)  # 爆发时间点
            base_curve = val / (1.0 + np.exp(-k * (t - t0)))
        elif trend_type == 'logarithmic':
            # 对数增长 (快到慢)
            base_curve = val * np.log1p(9 * t) / np.log1p(9)
        else: # 'stable' 或其他未分类的
            # 稳定趋势，围绕基线值波动
            base_curve = np.full(seq_len, val)

        # 2. 波动 (Fluctuations) - 模拟日常起伏
        freq = rng.uniform(1, 5)
        phase = rng.uniform(0, 2 * np.pi)
        amp = val * 0.1 # 波动幅度为值的 10%
        fluctuation = amp * np.sin(2 * np.pi * freq * t + phase)

        # 3. 突发 (Bursts) - 模拟热点事件或病毒式传播
        burst_signal = np.zeros(seq_len)
        if trend_type != 'linear_strict': # 账号年龄不应有突发
            num_bursts = rng.randint(0, 3)
            for _ in range(num_bursts):
                burst_center = rng.uniform(0.1, 0.9)
                burst_width = rng.uniform(0.02, 0.1)
                burst_amp = val * rng.uniform(0.2, 0.5)
                burst_signal += burst_amp * np.exp(-((t - burst_center)**2) / (2 * burst_width**2))

        # 组合
        final_curve = base_curve + fluctuation + burst_signal
        
        # 4. 噪声 (Noise)
        noise = rng.normal(0, 0.02 * (val + 1e-6), size=seq_len)
        final_curve += noise

        seq_matrix.append(final_curve)

    seq_matrix = np.stack(seq_matrix, axis=1)  # (seq_len, F)
    
    # 裁剪到合理范围，允许稍微超过 1.0 (因为有突发)
    seq_matrix = np.clip(seq_matrix, a_min=0.0, a_max=1.5)
    
    return seq_matrix.astype(np.float32)

# ----------------------
# 时间可得性感知：辅助判断 & 锚定时间序列
# ----------------------
def has_account_created_at(user_data):
    """判断用户是否有可解析的 profile.created_at 字段"""
    profile = user_data.get('profile', {}) or {}
    created_at_str = profile.get('created_at')
    if not created_at_str:
        return False
    return _parse_twitter_date(created_at_str) is not None


def has_reliable_tweet_timestamps(user_data, min_tweets=5):
    """判断用户推文是否包含 created_at 时间戳（适用于 TwiBot-22 dict 格式推文）"""
    tweets = user_data.get('tweet', []) or []
    if not tweets or isinstance(tweets[0], str):
        return False
    count = 0
    for tw in tweets:
        if isinstance(tw, dict) and tw.get('created_at'):
            count += 1
            if count >= min_tweets:
                return True
    return False


# 特征分类常量（供 pseudo 和 anchor 共用）
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


def build_anchor_timeseries_from_features(feat_dict, account_age_days_raw, seq_len=SEQ_LEN):
    """
    生命周期锚定版伪时间序列。
    与纯 pseudo 版的区别：
      1. 增长型特征的陡峭度和拐点由真实 account_age_days 控制
      2. 账龄短 → 增长曲线更陡；账龄长 → 更平缓
    参数:
        feat_dict: MinMax 缩放后的静态特征字典
        account_age_days_raw: 原始账号年龄（天，未归一化）
        seq_len: 序列长度
    返回: (seq_len, F) 的 float32 矩阵
    """
    keys = _TS_KEYS
    F = len(keys)
    tau = np.linspace(0, 1, seq_len)

    # 成熟度因子：以 10 年为满刻度
    maturity = np.clip(account_age_days_raw / 3650.0, 0.01, 1.0)

    user_id_str = str(feat_dict.get('user_id', '0'))
    user_seed = int(hash(user_id_str) % 100000)
    rng = np.random.RandomState(user_seed)

    vals = np.array([float(feat_dict.get(k, 0.0)) for k in keys], dtype=float)

    seq_matrix = []
    for i, key in enumerate(keys):
        val = vals[i]

        # --- 基础趋势 ---
        if key in _LINEAR_STRICT_KEYS:
            base_curve = val * tau
        elif key in _GROWTH_KEYS:
            if rng.random() < 0.5:
                # Logistic：年轻账号更陡，成熟账号拐点更靠前
                k_steep = 3.0 + (1.0 - maturity) * 8.0
                t0 = 0.2 + maturity * 0.4
                base_curve = val / (1.0 + np.exp(-k_steep * (tau - t0)))
            else:
                # 对数增长
                base_curve = val * np.log1p(9 * tau) / np.log1p(9)
        else:
            base_curve = np.full(seq_len, val)

        # --- 波动 ---
        freq = rng.uniform(1, 5)
        phase = rng.uniform(0, 2 * np.pi)
        amp = val * 0.1
        fluctuation = amp * np.sin(2 * np.pi * freq * tau + phase)

        # --- 突发 ---
        burst_signal = np.zeros(seq_len)
        if key not in _LINEAR_STRICT_KEYS:
            num_bursts = rng.randint(0, 3)
            for _ in range(num_bursts):
                bc = rng.uniform(0.1, 0.9)
                bw = rng.uniform(0.02, 0.1)
                ba = val * rng.uniform(0.15, 0.4)
                burst_signal += ba * np.exp(-((tau - bc) ** 2) / (2 * bw ** 2))

        # --- 噪声 ---
        noise = rng.normal(0, 0.02 * (val + 1e-6), size=seq_len)

        final_curve = base_curve + fluctuation + burst_signal + noise
        seq_matrix.append(final_curve)

    seq_matrix = np.stack(seq_matrix, axis=1)
    seq_matrix = np.clip(seq_matrix, a_min=0.0, a_max=1.5)
    return seq_matrix.astype(np.float32)


def build_temporal_sequence(user_data, feat_dict, mode="auto", seq_len=SEQ_LEN,
                            account_age_days_raw=None, time_range=None):
    """
    统一时间序列构造入口。
    参数:
        user_data: 原始用户 JSON 字典
        feat_dict: MinMax 缩放后的静态特征字典
        mode: "pseudo" | "anchor" | "auto"  (第一批；"real"/"hybrid" 留待第二批)
        seq_len: 序列长度
        account_age_days_raw: 原始账号年龄（天），用于 anchor 模式
        time_range: 全局时间范围（第二批 real 模式使用，暂不启用）
    返回:
        dict: {
            "matrix": np.ndarray (seq_len, F),
            "mode_used": str,
        }
    """
    if mode == "auto":
        if has_account_created_at(user_data) and account_age_days_raw is not None:
            mode = "anchor"
        else:
            mode = "pseudo"

    if mode == "anchor":
        if account_age_days_raw is None:
            matrix = build_pseudo_timeseries_from_features(feat_dict, seq_len=seq_len)
            mode = "pseudo"
        else:
            matrix = build_anchor_timeseries_from_features(feat_dict, account_age_days_raw, seq_len=seq_len)
    else:
        # pseudo（以及任何 fallback）
        matrix = build_pseudo_timeseries_from_features(feat_dict, seq_len=seq_len)
        mode = "pseudo"

    return {"matrix": matrix, "mode_used": mode}


# ----------------------
# 将每个分割的矩阵保存到磁盘 (流式, 内存友好)
# ----------------------
def build_and_save_all_splits(seq_len=SEQ_LEN, temporal_mode="auto", tmp_dir=None):
    """
    两阶段方法:
      1) 遍历 train.json 计算静态特征的 MinMax 缩放器
      2) 遍历所有分割，使用缩放后的特征生成每个用户的矩阵，并保存每个分割的 .npz 文件
    参数:
        seq_len: 固定序列长度
        temporal_mode: "pseudo" | "anchor" | "auto"
        tmp_dir: 保存目录 (默认使用全局 TMP_DIR)
    """
    if tmp_dir is None:
        tmp_dir = TMP_DIR
    os.makedirs(tmp_dir, exist_ok=True)
    splits = ['train', 'dev', 'test', 'support']
    static_list = []
    id_label_list = {s: [] for s in splits}

    # 1) 从训练集收集原始静态特征 (用于缩放器)
    train_path = os.path.join(DATA_DIR, 'train.json')
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.json 未在 {DATA_DIR} 中找到")

    logging.info("第一阶段: 正在从训练集收集静态特征以拟合scaler...")
    with open(train_path, 'r', encoding='utf-8') as f:
        users = json.load(f)
    for user in tqdm(users, desc="收集训练集静态特征"):
        sf = calculate_static_features(user)
        static_list.append(sf)
    static_df_train = pd.DataFrame(static_list)
    feature_cols = [c for c in static_df_train.columns if c != 'user_id']
    scaler = MinMaxScaler()
    scaler.fit(static_df_train[feature_cols].values)  # 仅在训练集上拟合
    logging.info("Scaler已在训练集上拟合完成。")

    # 2) 遍历每个分割，创建缩放后的特征和矩阵，保存 .npz 文件
    for split in splits:
        logging.info(f"正在处理split: {split}")
        fpath = os.path.join(DATA_DIR, f"{split}.json")
        if not os.path.exists(fpath):
            logging.warning(f"文件未找到，已跳过: {fpath}")
            continue

        matrices = []
        labels = []
        ids = []
        modes_used = []
        static_rows_for_split = []  # 可选: 用于保存每个分割的静态 DataFrame

        with open(fpath, 'r', encoding='utf-8') as f:
            users = json.load(f)

        for user in tqdm(users, desc=f"处理 {split}"):
            sf = calculate_static_features(user)
            uid = sf['user_id']
            raw_age = sf['account_age_days']  # 原始账号年龄（未归一化），用于 anchor 模式
            # 使用训练集缩放器缩放特征 (保留用户差异)
            feat_vector = np.array([sf[col] for col in feature_cols], dtype=float).reshape(1, -1)
            feat_scaled = scaler.transform(feat_vector)[0]  # 形状 (F,)
            # 映射回字典，键与构建函数期望的相同
            feat_dict = {k: float(v) for k, v in zip(feature_cols, feat_scaled)}
            # 通过统一入口构建时间序列矩阵
            ts_result = build_temporal_sequence(
                user_data=user,
                feat_dict=feat_dict,
                mode=temporal_mode,
                seq_len=seq_len,
                account_age_days_raw=raw_age,
            )
            matrices.append(ts_result["matrix"])
            modes_used.append(ts_result["mode_used"])
            labels.append(_safe_str_to_int(user.get('label'), default=-1))
            ids.append(uid)
            static_rows_for_split.append(sf)

        # 转换并保存
        npz_path = os.path.join(tmp_dir, f"{split}_matrices.npz")
        logging.info(f"正在保存 {split} 矩阵到 {npz_path} (数量={len(matrices)})")
        # 堆叠成数组形状 (N, seq_len, F)
        mats_arr = np.stack(matrices, axis=0).astype(np.float32) if matrices else np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32)
        labels_arr = np.array(labels, dtype=np.int8)
        ids_arr = np.array(ids, dtype=object)
        modes_arr = np.array(modes_used, dtype=object)
        np.savez_compressed(npz_path, matrices=mats_arr, labels=labels_arr, ids=ids_arr, modes=modes_arr)
        # 打印时序模式分布
        mode_dist = Counter(modes_used)
        logging.info(f"  时序模式分布: {dict(mode_dist)}")
        # 同时保存静态特征以供下游使用
        static_df = pd.DataFrame(static_rows_for_split)
        static_df.to_csv(os.path.join(tmp_dir, f"{split}_static.csv"), index=False)
    # 保存缩放器以备重用
    scaler_path = os.path.join(tmp_dir, "scaler_params.npz")
    np.savez_compressed(scaler_path, min=scaler.data_min_, max=scaler.data_max_, feature_cols=np.array(feature_cols, dtype=object))
    logging.info("所有splits已处理并保存到临时目录。")

# ----------------------
# 从 .npz 读取数据集
# ----------------------
class NpzSplitDataset(Dataset):
    def __init__(self, npz_path, keep_unlabeled=False):
        data = np.load(npz_path, allow_pickle=True)
        self.matrices = data['matrices']  # (N, seq_len, F)
        self.labels = data['labels']      # (N,)
        self.ids = data['ids']            # (N,)
        # 可选地过滤未标记的数据
        if not keep_unlabeled:
            mask = (self.labels == 0) | (self.labels == 1)
            self.matrices = self.matrices[mask]
            self.labels = self.labels[mask]
            self.ids = self.ids[mask]
        # 确保数据类型
        self.labels = self.labels.astype(np.int8)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.matrices[idx], int(self.labels[idx]), str(self.ids[idx])

def collate_fixed(batch):
    """
    batch: 列表，包含 (矩阵 (seq_len, F), 标签, ID)
    返回 矩阵张量 (B, seq_len, F), 掩码 (B, seq_len) 布尔值 (False 表示真实数据), 标签 (B,1), ID 列表
    """
    matrices, labels, ids = zip(*batch)
    matrices = torch.tensor(np.stack(matrices, axis=0), dtype=torch.float32)  # 已经固定了 seq_len
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    # 掩码: 全为 False，因为固定长度序列没有填充 (如果以后生成可变长度，请调整)
    mask = torch.zeros((matrices.size(0), matrices.size(1)), dtype=torch.bool)
    return matrices, mask, labels, list(ids)

# ----------------------
# 模型: TimeSeriesEncoder + 分类器
# ----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, S, E)
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
        # src: (B, S, F_in)
        x = self.input_proj(src) * math.sqrt(self.d_model)  # (B, S, d_model)
        x = self.pos_encoder(x)
        # src_key_padding_mask: (B, S) 布尔值，True 表示要 *忽略* 的位置
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        # 掩码平均池化 (处理可能全填充的行)
        mask = src_key_padding_mask  # 布尔值
        if mask is None:
            # 没有填充
            emb = out.mean(dim=1)
            return emb
        mask_expanded = mask.unsqueeze(-1).expand_as(out)  # (B,S,d)
        out = out.masked_fill(mask_expanded, 0.0)
        non_pad_counts = (~mask).sum(dim=1).unsqueeze(1).clamp(min=1).to(out.dtype)  # 避免除以0
        summed = out.sum(dim=1)
        emb = summed / non_pad_counts
        return emb  # (B, d_model)

class BotClassifier(nn.Module):
    def __init__(self, encoder: TimeSeriesEncoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.d_model, 1)

    def forward(self, src, mask):
        emb = self.encoder(src, src_key_padding_mask=mask)
        logits = self.classifier(emb)
        return logits

# ----------------------
# 训练 / 评估
# ----------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for matrices, mask, labels, _ in tqdm(loader, desc="训练"):
        matrices = matrices.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
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
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for matrices, mask, labels, _ in tqdm(loader, desc="评估"):
            matrices = matrices.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            logits = model(matrices, mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().cpu().numpy().reshape(-1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().reshape(-1).tolist())
    if len(all_labels) == 0:
        return float('nan'), 0.0, 0.0
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return total_loss / max(1, len(loader)), acc, f1

# ----------------------
# Flat-Static 模式: 直接使用缩放后的静态特征作为向量
# ----------------------
def generate_flat_static_vectors():
    """
    Flat-Static 消融变体：跳过 Transformer 编码，
    直接将每个用户的 14 维缩放静态特征保存为向量文件。
    """
    logging.info("Flat-Static 模式：直接使用缩放后的静态特征作为时序向量")
    splits = ['train', 'dev', 'test', 'support']

    # 检查静态 CSV 和 scaler 是否存在
    csv_paths = {s: os.path.join(TMP_DIR, f"{s}_static.csv") for s in splits}
    scaler_path = os.path.join(TMP_DIR, "scaler_params.npz")

    if not all(os.path.exists(p) for p in csv_paths.values()) or not os.path.exists(scaler_path):
        logging.info("静态 CSV 或 scaler 不存在，先运行构建步骤（pseudo 模式）...")
        build_and_save_all_splits(seq_len=SEQ_LEN, temporal_mode="pseudo")

    # 加载 scaler 参数
    scaler_data = np.load(scaler_path, allow_pickle=True)
    feature_cols = list(scaler_data['feature_cols'])
    scaler = MinMaxScaler()
    scaler.data_min_ = scaler_data['min']
    scaler.data_max_ = scaler_data['max']
    scaler.data_range_ = scaler.data_max_ - scaler.data_min_
    scaler.scale_ = 1.0 / (scaler.data_range_ + 1e-10)
    scaler.min_ = -scaler.data_min_ * scaler.scale_
    scaler.n_features_in_ = len(feature_cols)
    scaler.feature_range = (0, 1)
    logging.info(f"已加载 scaler，特征维度: {len(feature_cols)}")

    # 逐 split 加载静态 CSV，缩放后收集向量
    all_vectors = {}
    for split in splits:
        csv_path = csv_paths[split]
        if not os.path.exists(csv_path):
            logging.warning(f"跳过缺失的 CSV: {csv_path}")
            continue
        df = pd.read_csv(csv_path, dtype={'user_id': str})
        logging.info(f"加载 {split}: {len(df)} 条记录")
        for _, row in df.iterrows():
            uid = str(row['user_id'])
            feat_vector = np.array([row[col] for col in feature_cols], dtype=float).reshape(1, -1)
            feat_scaled = scaler.transform(feat_vector)[0]  # (F,)
            all_vectors[uid] = feat_scaled.astype(np.float32)

    # 保存
    logging.info(f"Total vectors: {len(all_vectors)}")
    user_ids_ordered = list(all_vectors.keys())
    vectors = np.stack([all_vectors[uid] for uid in user_ids_ordered], axis=0).astype(np.float32)
    out_path = os.path.join(FEATURE_OUTPUT_DIR, "twibot20_transformer_vectors_flat_static.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, vectors=vectors, user_ids=np.array(user_ids_ordered, dtype=object))
    logging.info(f"Flat-Static vectors saved to: {out_path} (形状={vectors.shape})")

# ----------------------
# 完整管道: 构建 -> 训练 -> 推理
# ----------------------
def main(temporal_mode="auto", seq_len=None, num_layers=None, d_model=None, n_head=None, dropout=None, suffix_override=None):
    global SEQ_LEN, NUM_ENCODER_LAYERS, D_MODEL, N_HEAD, DROPOUT
    
    # 覆盖全局变量
    if seq_len is not None: SEQ_LEN = seq_len
    if num_layers is not None: NUM_ENCODER_LAYERS = num_layers
    if d_model is not None: D_MODEL = d_model
    if n_head is not None: N_HEAD = n_head
    if dropout is not None: DROPOUT = dropout

    # 参数化目录: 不同 T/L 组合保存到不同子目录
    if suffix_override is not None:
        suffix = suffix_override
        tmp_dir = os.path.join(TMP_DIR, f"{suffix.strip('_')}")
        os.makedirs(tmp_dir, exist_ok=True)
    elif SEQ_LEN == 32 and NUM_ENCODER_LAYERS == 2:
        tmp_dir = TMP_DIR
        suffix = ""
    else:
        tmp_dir = os.path.join(TMP_DIR, f"T{SEQ_LEN}_L{NUM_ENCODER_LAYERS}")
        suffix = f"_T{SEQ_LEN}_L{NUM_ENCODER_LAYERS}"
        os.makedirs(tmp_dir, exist_ok=True)

    logging.info(f"设备: {DEVICE}  |  时序模式: {temporal_mode} | T={SEQ_LEN} | L={NUM_ENCODER_LAYERS} | D={D_MODEL} | H={N_HEAD}")
    logging.info(f"临时目录: {tmp_dir}")

    # Flat-Static 模式: 跳过 Transformer，直接保存静态特征向量
    if temporal_mode == "flat_static":
        generate_flat_static_vectors()
        return

    # 0) 构建矩阵并保存到 tmp_dir (如果不存在)
    expected_files = [os.path.join(tmp_dir, f"{s}_matrices.npz") for s in ['train','dev','test','support']]
    if not all(os.path.exists(p) for p in expected_files):
        logging.info(f"未找到临时 .npz 文件 — 正在从原始 JSON 文件构建 (T={SEQ_LEN})...")
        build_and_save_all_splits(seq_len=SEQ_LEN, temporal_mode=temporal_mode, tmp_dir=tmp_dir)
    else:
        logging.info("找到现有临时 .npz 文件，跳过构建步骤。")
        logging.info("如需切换时序模式，请先删除对应 tmp 目录下的 .npz 文件再重新运行。")

    # 1) 加载训练/开发数据集 (过滤未标记的数据)
    train_npz = os.path.join(tmp_dir, "train_matrices.npz")
    dev_npz = os.path.join(tmp_dir, "dev_matrices.npz")

    if not os.path.exists(train_npz):
        raise FileNotFoundError("train_matrices.npz 未找到。构建步骤是否成功？")

    train_dataset = NpzSplitDataset(train_npz, keep_unlabeled=False)
    dev_dataset = NpzSplitDataset(dev_npz, keep_unlabeled=False) if os.path.exists(dev_npz) else None

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fixed)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fixed) if dev_dataset else None

    logging.info(f"训练样本: {len(train_dataset)}; 开发样本: {len(dev_dataset) if dev_dataset else 0}")

    # 2) 模型初始化
    # 从训练集 .npz 文件确定输入维度 (矩阵形状 (N, seq_len, F))
    tmp = np.load(train_npz, allow_pickle=True)
    in_dim = tmp['matrices'].shape[2]
    logging.info(f"输入特征维度: {in_dim}")

    # 根据 temporal_mode 动态生成模型权重文件名
    mode_str = temporal_mode if temporal_mode in ["anchor", "pseudo"] else "auto"
    model_save_path = os.path.join(FEATURE_OUTPUT_DIR, f"best_timeseries_model_{mode_str}{suffix}.pt")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    encoder = TimeSeriesEncoder(in_dim=in_dim, d_model=D_MODEL, nhead=N_HEAD, num_layers=NUM_ENCODER_LAYERS, dropout=DROPOUT)
    model = BotClassifier(encoder).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3) 训练循环并保存最佳模型
    best_acc = -1.0
    train_loss_history = []
    val_loss_history = []

    for epoch in range(NUM_EPOCHS):
        logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        train_loss_history.append(train_loss)
        logging.info(f"train loss: {train_loss:.4f}")
        
        if dev_loader:
            val_loss, val_acc, val_f1 = evaluate(model, dev_loader, criterion, DEVICE)
            val_loss_history.append(val_loss)
            logging.info(f"val loss: {val_loss:.4f} | val acc: {val_acc:.4f} | val F1: {val_f1:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_save_path)
                logging.info(f"已保存新的最佳模型 (acc={val_acc:.4f})\n")
        else:
            # 如果没有开发集，仍然保存最后一个 epoch
            torch.save(model.state_dict(), model_save_path)
            logging.info("已保存模型状态 (无开发集)。")
        logging.info("") # 在每个 epoch 结束时添加空行

    # 训练后绘制并保存损失曲线
    # if train_loss_history and val_loss_history:
    #     logging.info("正在绘制并保存训练/验证损失曲线...")
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(train_loss_history, label='Training Loss')
    #     plt.plot(val_loss_history, label='Validation Loss')
    #     plt.title('Feature Extraction Model Training and Validation Loss')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     plt.grid(True)
    #     plot_save_path = os.path.join(FEATURE_OUTPUT_DIR, "loss_plot_feature_extraction.png")
    #     os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    #     plt.savefig(plot_save_path)
    #     logging.info(f"损失曲线图已保存至: {plot_save_path}")

    # 4) 推理: 加载最佳模型并提取所有分割的嵌入
    logging.info("正在加载最佳模型进行推理...")
    if not os.path.exists(model_save_path):
        raise FileNotFoundError("模型检查点未找到！训练可能失败。")
    # 重新加载模型权重
    model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
    model.eval()

    all_embeddings = {}
    for split in ['train','dev','test','support']:
        npz_path = os.path.join(tmp_dir, f"{split}_matrices.npz")
        if not os.path.exists(npz_path):
            logging.info(f"跳过缺失的分割文件: {npz_path}")
            continue
        logging.info(f"Extracting embeddings for split: {split}")
        data = np.load(npz_path, allow_pickle=True)
        mats = data['matrices']  # (N, S, F)
        ids = data['ids'].tolist()
        N = mats.shape[0]
        # 构建数据集/加载器，允许未标记的数据
        ds = NpzSplitDataset(npz_path, keep_unlabeled=True)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fixed)
        
        with torch.no_grad():
            for matrices, mask, _, ids_batch in tqdm(loader, desc=f"嵌入 {split}"):
                matrices = matrices.to(DEVICE)
                mask = mask.to(DEVICE)
                emb = model.encoder(matrices, src_key_padding_mask=mask)  # (B, d_model)
                emb = emb.cpu().numpy()
                for i, uid in enumerate(ids_batch):
                    all_embeddings[uid] = emb[i]

    # 5) 以稳定顺序保存嵌入数组 (ID 列表)
    logging.info(f"Total embeddings extracted: {len(all_embeddings)}")
    user_ids_ordered = list(all_embeddings.keys())
    vectors = np.stack([all_embeddings[uid] for uid in user_ids_ordered], axis=0).astype(np.float32)
    
    # 根据 temporal_mode 动态生成文件名
    mode_str = temporal_mode if temporal_mode in ["anchor", "pseudo"] else "auto"
    if suffix:
        out_name = f"twibot20_transformer_vectors_{mode_str}{suffix}.npz"
    else:
        out_name = f"twibot20_transformer_vectors_{mode_str}.npz"
        
    out_path = os.path.join(FEATURE_OUTPUT_DIR, out_name)
    np.savez_compressed(out_path, vectors=vectors, user_ids=np.array(user_ids_ordered, dtype=object))
    logging.info(f"Saved embeddings to: {out_path} (形状={vectors.shape})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="时间可得性感知特征提取")
    parser.add_argument("--mode", default="auto", choices=["pseudo", "anchor", "auto", "flat_static"],
                        help="时序构造模式: pseudo=纯伪时间序列, anchor=生命周期锚定, auto=自动选择, flat_static=直接使用静态特征")
    parser.add_argument("--seq-len", type=int, default=None,
                        help=f"时间序列长度 (默认: {SEQ_LEN})")
    parser.add_argument("--num-layers", type=int, default=None,
                        help=f"Transformer编码层数 (默认: {NUM_ENCODER_LAYERS})")
    parser.add_argument("--d-model", type=int, default=None,
                        help=f"Transformer隐藏层维度 (默认: {D_MODEL})")
    parser.add_argument("--n-head", type=int, default=None,
                        help=f"Transformer注意力头数 (默认: {N_HEAD})")
    parser.add_argument("--dropout", type=float, default=None,
                        help=f"Transformer Dropout率 (默认: {DROPOUT})")
    parser.add_argument("--lr", type=float, default=None,
                        help=f"Transformer 学习率 (默认: {LEARNING_RATE})")
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"Transformer Batch Size (默认: {BATCH_SIZE})")
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"Transformer 训练轮数 (默认: {NUM_EPOCHS})")
    parser.add_argument("--suffix", type=str, default=None,
                        help="强制指定保存文件的后缀 (用于 run_exp.py 联动)")
    args = parser.parse_args()
    
    # 覆盖全局变量
    if args.lr is not None: LEARNING_RATE = args.lr
    if args.batch_size is not None: BATCH_SIZE = args.batch_size
    if args.epochs is not None: NUM_EPOCHS = args.epochs
    
    main(temporal_mode=args.mode, seq_len=args.seq_len, num_layers=args.num_layers, 
         d_model=args.d_model, n_head=args.n_head, dropout=args.dropout, suffix_override=args.suffix)