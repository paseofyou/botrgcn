import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import traceback # 确保在所有导入语句之前
import csv
import datetime
import subprocess
import torch
from torch import nn
import random
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score
import json
import argparse
import swanlab

from model import BotRGCN, BotRGCN_Pure, BotRGCN_v2, BotRGCN_v3
# from model import BotRGCN_E2E, info_nce_loss
from utils import accuracy, init_weights

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 仓库根目录 (Twibot_20/src → 上两级)，实验结果表放这里以便随代码一起版本控制
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_RESULTS_CSV = os.path.join(REPO_ROOT, "experiments", "results.csv")


def set_seed(seed=42):
    """固定所有随机种子以确保可复现性"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已固定: {seed}")


def get_git_state():
    """
    返回当前代码的 git 版本标识, 形如 'a1b2c3d' 或 'a1b2c3d-dirty'。
    带 -dirty 后缀说明运行时工作区存在未提交改动, 该结果不可复现, 不应写入论文。
    训练产物(swanlog/, checkpoints/, experiments/results.csv) 不计入 dirty。
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        if not status:
            return sha
        # 忽略训练产物, 只看真正的代码改动
        ignore_paths = {"experiments/results.csv", "swanlog", "checkpoints", "Twibot_20/swanlog",
                        "Twibot_20/checkpoints", "Twibot_22/swanlog", "Twibot_22/checkpoints"}
        changed = [line for line in status.splitlines() if line.split()[-1] not in ignore_paths]
        return sha + ("-dirty" if changed else "")
    except Exception:
        return "unknown"


def append_result_row(csv_path, row):
    """把一次运行的最终测试指标追加到 results.csv (不存在则建表头)。"""
    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(f"结果已记录: {csv_path}")
    except Exception as e:
        print(f"警告: 写入结果表失败 — {e}")


parser = argparse.ArgumentParser(description="TwiBot-20 训练")
parser.add_argument("--model", default="v2", choices=["pure", "v1", "v2", "v3"],
                    help="模型版本 (pure=纯净BotRGCN基线, 不使用任何时序特征)")
parser.add_argument("--seed", type=int, default=42, help="随机种子")
parser.add_argument("--variant", type=str, default="",
                    help="实验变体标签, 写入 results.csv 用于聚合 (如 full/wo_temporal/flat_static)")
parser.add_argument("--temporal-npz", type=str, default=None,
                    help="直接指定时序特征 npz 路径 (优先于 --ts-mode)")
parser.add_argument("--results-csv", type=str, default=DEFAULT_RESULTS_CSV,
                    help="实验结果汇总表路径")
parser.add_argument("--no-temporal", action="store_true",
                    help="置零时序特征 (Only-RGCN 消融实验用)")
parser.add_argument("--no-graph", action="store_true", 
                    help="跳过RGCN层，仅使用MLP (Only-Time 消融实验用)")
parser.add_argument("--concat-fusion", action="store_true", 
                    help="使用简单拼接融合代替门控融合")
parser.add_argument("--ts-mode", type=str, default="anchor", 
                    help="Which temporal feature file to load (e.g., anchor_T32_D64...)")
parser.add_argument("--save-suffix", type=str, default="",
                    help="模型/图片保存后缀, 避免覆盖 (e.g. '_no_temporal')")

# --- 联动调参专用参数 (由 run_exp.py 传入) ---
parser.add_argument("--feat-seq-len", type=int, default=32, help="仅用于SwanLab记录")
parser.add_argument("--feat-d-model", type=int, default=64, help="用于SwanLab记录和模型初始化")
parser.add_argument("--feat-n-head", type=int, default=8, help="仅用于SwanLab记录")
parser.add_argument("--feat-num-layers", type=int, default=2, help="仅用于SwanLab记录")
parser.add_argument("--feat-dropout", type=float, default=0.1, help="仅用于SwanLab记录")
parser.add_argument("--feat-lr", type=float, default=1e-4, help="仅用于SwanLab记录")
parser.add_argument("--feat-batch-size", type=int, default=128, help="仅用于SwanLab记录")
parser.add_argument("--feat-epochs", type=int, default=100, help="仅用于SwanLab记录")
parser.add_argument("--emb-size", type=int, default=128, help="图网络隐藏层维度")
parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
parser.add_argument("--dropout", type=float, default=0.3, help="图网络Dropout")
parser.add_argument("--weight-decay", type=float, default=0.001, help="图网络权重衰减 (L2正则)")
parser.add_argument("--epochs", type=int, default=150, help="图网络训练轮数")

# --- E2E 端到端训练参数 ---
# parser.add_argument("--e2e", action="store_true",
#                     help="启用端到端训练 (Transformer+GNN联合优化)")
# parser.add_argument("--seq-len", type=int, default=16,
#                     help="E2E: 时间序列长度")
# parser.add_argument("--ts-nhead", type=int, default=2,
#                     help="E2E: Transformer注意力头数")
# parser.add_argument("--ts-layers", type=int, default=2,
#                     help="E2E: Transformer编码层数")
# parser.add_argument("--ts-lr-scale", type=float, default=0.1,
#                     help="E2E: Transformer学习率缩放因子")
# parser.add_argument("--align-beta", type=float, default=0.0,
#                     help="对齐损失权重 (0=禁用)")
# parser.add_argument("--align-temp", type=float, default=0.5,
#                     help="InfoNCE温度参数")
# parser.add_argument("--ts-warmup-epochs", type=int, default=10,
#                     help="E2E: Transformer 冻结预热轮数 (前 K 轮只训练 GNN, 默认 10)")
# parser.add_argument("--ts-pretrained", type=str, default=None,
#                     help="E2E: 预训练 Transformer 权重路径 (BotClassifier .pt), 用于热启动")
args = parser.parse_args()

# ------------------- 参数配置 -------------------
set_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_size = args.emb_size
dropout = args.dropout
lr = args.lr
weight_decay = args.weight_decay
epochs = args.epochs
model_version = args.model  # 'pure' = 官方基线, 'v1'/'v2'/'v3' = 本文模型
# 代码位于 Twibot_20/src/, 而数据与产物目录位于 Twibot_20/ 下, 因此以 src 的上级为基准
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# ------------------- 代码版本标识 -------------------
git_state = get_git_state()
run_name = f"BotRGCN_{model_version}{args.save_suffix}_s{args.seed}"
print(f"代码版本: {git_state}")
if git_state.endswith("-dirty"):
    print("!!! 警告: 工作区存在未提交改动, 本次结果无法复现, 不要写入论文 !!!")

# ------------------- 初始化 SwanLab -------------------
config_dict = vars(args)
config_dict.update({
    'dataset': 'TwiBot-20',
    'model_version': model_version,
    'weight_decay': weight_decay,
    'epochs': epochs,
    'git_commit': git_state
})

swanlab.init(
    project="TwiBot20-BotDetection",
    name=run_name,
    config=config_dict,
    logdir=os.path.join(BASE_DIR, "swanlog")
)

# ------------------- 加载数据 -------------------
print("=== 加载预处理数据 ===")


def load_tensor(path):
    print(f"加载 {path}")
    return torch.load(path, map_location=device, weights_only=True)


processed_dir = os.path.join(BASE_DIR, 'saved_data', 'processed_data')
if not os.path.isdir(processed_dir):
    raise SystemExit(f"预处理数据目录不存在: {processed_dir}")

des_tensor = load_tensor(os.path.join(processed_dir, 'des_tensor.pt'))
tweets_tensor = load_tensor(os.path.join(processed_dir, 'tweets_tensor.pt'))
num_prop = load_tensor(os.path.join(processed_dir, 'num_properties_tensor.pt'))
cat_prop = load_tensor(os.path.join(processed_dir, 'cat_properties_tensor.pt'))
edge_index = load_tensor(os.path.join(processed_dir, 'edge_index.pt'))
edge_type = load_tensor(os.path.join(processed_dir, 'edge_type.pt'))
labels = load_tensor(os.path.join(processed_dir, 'label.pt'))
labels = labels.long().squeeze()

# 加载时序特征
import pandas as pd

JSON_DATA_DIR = os.path.join(BASE_DIR, 'Data', 'Twibot-20')
STATIC_CSV_DIR = os.path.join(BASE_DIR, 'tmp', 'tmp_v6')
num_nodes = des_tensor.shape[0]
print(f"节点数量: {num_nodes}")

# ---- 构建 user_id → graph_index 映射, 同时得到各 split 的大小 ----
# 图节点顺序为 train → dev → test → support (与特征提取时的拼接顺序一致)。
# 优先用官方 JSON; 若缺失, 退回到 feature_extraction 落盘的 *_static.csv
# (两者用户顺序完全一致, 已核验)。
SPLIT_ORDER = ['train', 'dev', 'test', 'support']
ordered_user_ids = []
split_sizes = {}
if all(os.path.exists(os.path.join(JSON_DATA_DIR, f'{s}.json')) for s in SPLIT_ORDER[:3]):
    print(f"划分来源: 官方 JSON ({JSON_DATA_DIR})")
    import ijson
    for split in SPLIT_ORDER:
        split_path = os.path.join(JSON_DATA_DIR, f'{split}.json')
        n = 0
        if os.path.exists(split_path):
            with open(split_path, 'r', encoding='utf-8') as f:
                for user in ijson.items(f, 'item'):
                    ordered_user_ids.append(str(user['ID']))
                    n += 1
        split_sizes[split] = n
elif all(os.path.exists(os.path.join(STATIC_CSV_DIR, f'{s}_static.csv')) for s in SPLIT_ORDER):
    print(f"划分来源: static CSV ({STATIC_CSV_DIR}) — 官方 JSON 缺失, 回退")
    for split in SPLIT_ORDER:
        col = pd.read_csv(os.path.join(STATIC_CSV_DIR, f'{split}_static.csv'),
                          usecols=['user_id'])['user_id'].astype(str).tolist()
        ordered_user_ids.extend(col)
        split_sizes[split] = len(col)
else:
    raise SystemExit(
        f"无法确定数据划分: 既找不到官方 JSON ({JSON_DATA_DIR}), "
        f"也找不到 static CSV ({STATIC_CSV_DIR})")

if len(ordered_user_ids) != num_nodes:
    raise SystemExit(
        f"用户顺序表长度 {len(ordered_user_ids)} 与图节点数 {num_nodes} 不一致, "
        f"时序特征无法正确对齐, 请检查预处理产物。")

id_to_idx_map = {uid: i for i, uid in enumerate(ordered_user_ids)}
print(f"已构建 {len(ordered_user_ids)} 个用户的有序ID列表。划分大小: {split_sizes}")

# if args.e2e:
#     # === E2E 模式: 加载原始时间序列矩阵 ===
#     print("=== E2E模式: 加载原始时间序列矩阵 ===")
#     tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp_v6')
#     target_T = args.seq_len
#     raw_ts_tensor = torch.zeros((num_nodes, target_T, 14), dtype=torch.float32, device=device)
#     total_mapped = 0
#     for split_name in ['train', 'dev', 'test', 'support']:
#         split_npz = os.path.join(tmp_dir, f"{split_name}_matrices.npz")
#         if not os.path.exists(split_npz):
#             continue
#         data = np.load(split_npz, allow_pickle=True)
#         matrices = data['matrices']  # (N_split, T_orig, F)
#         ids = data['ids']
#         T_orig = matrices.shape[1]
#         if T_orig >= target_T:
#             matrices = matrices[:, :target_T, :]
#         else:
#             pad = np.zeros((matrices.shape[0], target_T - T_orig, matrices.shape[2]), dtype=np.float32)
#             matrices = np.concatenate([matrices, pad], axis=1)
#         graph_indices = []
#         npz_indices = []
#         for i, uid in enumerate(ids):
#             idx = id_to_idx_map.get(str(uid))
#             if idx is not None:
#                 graph_indices.append(idx)
#                 npz_indices.append(i)
#         if graph_indices:
#             raw_ts_tensor[graph_indices] = torch.from_numpy(matrices[npz_indices]).to(device)
#             total_mapped += len(graph_indices)
#         print(f"  {split_name}: 映射 {len(graph_indices)} 个用户")
#     print(f"  总映射: {total_mapped}, 原始时序形状: {raw_ts_tensor.shape}")
#     temporal_tensor = None
# else:
# === 原始模式: 加载预计算嵌入 ===
print(f"=== 加载 Transformer 生成的新时序特征 ({args.ts_mode} 模式) ===")
raw_ts_tensor = None
if args.temporal_npz:
    npz_path = args.temporal_npz
else:
    npz_path = os.path.join(BASE_DIR, 'feature_model_outputs', f'twibot20_transformer_vectors_{args.ts_mode}.npz')
try:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"未在指定路径找到时序特征文件: {npz_path}")

    npz_data = np.load(npz_path, allow_pickle=True)
    npz_vectors = npz_data['vectors']
    npz_user_ids = [str(uid) for uid in npz_data['user_ids']]
    print(f"已从 {npz_path} 加载 {len(npz_user_ids)} 个用户的向量。")

    embedding_dim = npz_vectors.shape[1]
    aligned_temporal_tensor = torch.zeros((len(ordered_user_ids), embedding_dim), device=device)
    unmapped_count = 0
    for i, user_id in enumerate(npz_user_ids):
        target_idx = id_to_idx_map.get(str(user_id))
        if target_idx is not None:
            aligned_temporal_tensor[target_idx] = torch.from_numpy(npz_vectors[i]).to(device)
        else:
            unmapped_count += 1
    if unmapped_count > 0:
        print(f"警告: {unmapped_count} 个来自.npz的向量无法在目标张量中找到映射。")
    temporal_tensor = aligned_temporal_tensor
    print(f"成功对齐时序特征，最终形状: {temporal_tensor.shape}")

    if args.no_temporal:
        print("*** --no-temporal: 时序特征已置零 (Only-RGCN 消融模式) ***")
        temporal_tensor = torch.zeros_like(temporal_tensor)

except Exception as e:
    print(f"错误: 加载或对齐新的时序特征失败。")
    traceback.print_exc()
    print("将使用全0代替时序特征！")
    temporal_tensor = torch.zeros((num_nodes, 64), device=device)

# ------------------- 使用官方固定划分 -------------------
# 图节点按 train → dev → test → support 顺序排列, 因此官方划分就是前三段连续区间。
# split_sizes 来自上面构建 ordered_user_ids 时的同一数据源, 保证两者绝不会错位。
# 注意: saved_data/processed_data 下的 train_idx.pt / val_idx.pt / test_idx.pt 是
# 早期的随机划分 (8278/1182/2366, 索引不连续), 与官方划分不同, 这里一律不使用。
n_train, n_dev, n_test = split_sizes['train'], split_sizes['dev'], split_sizes['test']

# 必须保持在 CPU 上，不要 .to(device)
train_idx = torch.arange(0, n_train)
val_idx = torch.arange(n_train, n_train + n_dev)
test_idx = torch.arange(n_train + n_dev, n_train + n_dev + n_test)

n_labeled = labels.shape[0]
if n_train + n_dev + n_test != n_labeled:
    raise SystemExit(
        f"划分总数 {n_train + n_dev + n_test} 与标签数 {n_labeled} 不一致, "
        f"划分与标签存在错位, 拒绝继续训练。")

print(f"使用 TwiBot-20 官方划分 — Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

# ------------------- 模型/损失/优化器 -------------------
# if args.e2e:
#     print("=== 初始化 E2E 端到端模型 ===")
#     model = BotRGCN_E2E(
#         num_prop_size=num_prop.shape[1],
#         cat_prop_size=cat_prop.shape[1],
#         ts_input_dim=raw_ts_tensor.shape[2],  # 14
#         d_model=embedding_size,
#         nhead=args.ts_nhead,
#         num_transformer_layers=args.ts_layers,
#         embedding_dimension=embedding_size,
#         dropout=dropout,
#         chunk_size=50000,
#         use_checkpoint=False  # TwiBot-20 小数据集无需 checkpoint
#     ).to(device)
#     model.apply(init_weights)
#     model._init_ts_weights()
#     # 加载预训练 Transformer 权重 (热启动)
#     if args.ts_pretrained:
#         print(f"  加载预训练 Transformer: {args.ts_pretrained}")
#         pretrained_sd = torch.load(args.ts_pretrained, map_location=device, weights_only=True)
#         mapping = {
#             'encoder.input_proj.': 'ts_input_proj.',
#             'encoder.pos_encoder.': 'ts_pos_enc.',
#             'encoder.transformer.': 'ts_transformer.',
#         }
#         mapped_sd = {}
#         skipped = 0
#         model_keys = set(model.state_dict().keys())
#         for pt_key, pt_val in pretrained_sd.items():
#             matched = False
#             for src_prefix, dst_prefix in mapping.items():
#                 if pt_key.startswith(src_prefix):
#                     e2e_key = dst_prefix + pt_key[len(src_prefix):]
#                     if e2e_key in model_keys:
#                         mapped_sd[e2e_key] = pt_val
#                     else:
#                         skipped += 1
#                     matched = True
#                     break
#             if not matched:
#                 skipped += 1
#         model.load_state_dict(mapped_sd, strict=False)
#         print(f"  预训练权重: 加载 {len(mapped_sd)} 个, 跳过 {skipped} 个 (classifier 等)")
#     n_params = sum(p.numel() for p in model.parameters())
#     n_ts = sum(p.numel() for n, p in model.named_parameters() if 'ts_' in n)
#     print(f"  总参数量: {n_params:,}, 其中 Transformer: {n_ts:,}")
# else:
print(f"=== 使用模型: {model_version} ===")
# time_size 必须取实际加载到的时序嵌入维度, 不能用 --feat-d-model 猜
# (flat_static 等消融的嵌入维度与 Transformer d_model 不同)
time_size = temporal_tensor.shape[1]
if model_version == 'pure':
    model = BotRGCN_Pure(num_prop_size=num_prop.shape[1], cat_prop_size=cat_prop.shape[1], embedding_dimension=embedding_size, dropout=dropout).to(device)
elif model_version == 'v3':
    model = BotRGCN_v3(num_prop_size=num_prop.shape[1], cat_prop_size=cat_prop.shape[1], embedding_dimension=embedding_size, dropout=dropout, time_size=time_size).to(device)
elif model_version == 'v2':
    model = BotRGCN_v2(num_prop_size=num_prop.shape[1], cat_prop_size=cat_prop.shape[1], embedding_dimension=embedding_size, dropout=dropout, time_size=time_size).to(device)
else:
    model = BotRGCN(num_prop_size=num_prop.shape[1], cat_prop_size=cat_prop.shape[1], embedding_dimension=embedding_size, dropout=dropout, time_size=time_size).to(device)
model.apply(init_weights)
n_params = sum(p.numel() for p in model.parameters())
print(f"  总参数量: {n_params:,}")

loss_fn = nn.CrossEntropyLoss()

# if args.e2e:
#     ts_params = [p for n, p in model.named_parameters() if 'ts_' in n]
#     other_params = [p for n, p in model.named_parameters() if 'ts_' not in n]
#     optimizer = torch.optim.AdamW([
#         {'params': ts_params, 'lr': lr * args.ts_lr_scale},
#         {'params': other_params, 'lr': lr}
#     ], weight_decay=weight_decay)
#     print(f"  Transformer lr: {lr * args.ts_lr_scale:.2e}, GNN lr: {lr:.2e}")
#     print(f"  Transformer 预热: 前 {args.ts_warmup_epochs} 轮冻结 Transformer, 仅训练 GNN")
#     if args.align_beta > 0:
#         print(f"  对齐损失: InfoNCE, beta={args.align_beta}, temp={args.align_temp}")

#     # 预热阶段: 冻结 Transformer 参数
#     ts_param_names = [n for n, _ in model.named_parameters() if 'ts_' in n]
#     if args.ts_warmup_epochs > 0:
#         for n, p in model.named_parameters():
#             if 'ts_' in n:
#                 p.requires_grad = False
#         optimizer.param_groups[0]['lr'] = 0.0
#         print(f"  [预热] 已冻结 {len(ts_param_names)} 个 Transformer 参数")
# else:
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# 学习率调度器
# effective_epochs = epochs - args.ts_warmup_epochs if args.e2e and args.ts_warmup_epochs > 0 else epochs
effective_epochs = epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, effective_epochs), eta_min=1e-6)

# 选择正确的时序输入
# ts_input = raw_ts_tensor if args.e2e else temporal_tensor
# use_align = args.e2e and args.align_beta > 0
ts_input = temporal_tensor
use_align = False

# 融合方式在整个 run 内固定，训练与测试必须一致
if model_version == 'pure':
    FUSION_MODE = 'n/a'  # 纯净基线没有时序通路, 不存在融合
elif args.no_graph:
    FUSION_MODE = 'no_graph'
elif args.no_temporal:
    FUSION_MODE = 'none'
elif args.concat_fusion:
    FUSION_MODE = 'concat'
else:
    FUSION_MODE = 'gated'
print(f"=== 融合模式: {FUSION_MODE} ===")


# ------------------- 训练函数 -------------------
def train(epoch):
    model.train()
    fusion_mode = FUSION_MODE

    output = model(des_tensor, tweets_tensor, num_prop, cat_prop,
                   ts_input, edge_index, edge_type, fusion_type=fusion_mode)

    loss_cls = loss_fn(output[train_idx], labels[train_idx])
    loss_total = loss_cls

    # if use_align:
    #     loss_align = info_nce_loss(
    #         g_emb[train_idx], t_emb[train_idx],
    #         temperature=args.align_temp)
    #     loss_total = loss_cls + args.align_beta * loss_align

    acc_train = accuracy(output[train_idx], labels[train_idx])
    loss_val = loss_fn(output[val_idx], labels[val_idx])
    acc_val = accuracy(output[val_idx], labels[val_idx])

    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()

    # align_str = f' | L_align: {loss_align.item():.4f}' if use_align else ''
    # print(f'Epoch: {epoch + 1:04d} | loss_train: {loss_total.item():.4f}{align_str} | '
    #       f'acc_train: {acc_train.item():.4f} | loss_val: {loss_val.item():.4f} | acc_val: {acc_val.item():.4f}')
    print(f'Epoch: {epoch + 1:04d} | loss_train: {loss_total.item():.4f} | '
          f'acc_train: {acc_train.item():.4f} | loss_val: {loss_val.item():.4f} | acc_val: {acc_val.item():.4f}')

    cur_lr = optimizer.param_groups[0]['lr']
    swanlab.log({
        "Train/Loss": loss_total.item(),
        "Train/Accuracy": acc_train.item(),
        "Val/Loss": loss_val.item(),
        "Val/Accuracy": acc_val.item(),
       
    }, step=epoch + 1)

    return loss_total.item(), loss_val.item(), acc_val.item()


# ------------------- 测试函数 -------------------
@torch.no_grad()
def test(epochs_run, best_val_acc):
    model.eval()
    with torch.no_grad():
        fusion_mode = FUSION_MODE

        output = model(des_tensor, tweets_tensor, num_prop, cat_prop,
                       ts_input, edge_index, edge_type, fusion_type=fusion_mode)
        loss_test = loss_fn(output[test_idx], labels[test_idx])
        acc_test = accuracy(output[test_idx], labels[test_idx])

        preds = output.max(1)[1].cpu().numpy()
        true = labels.cpu().numpy()
        test_true = true[test_idx.cpu().numpy()]
        test_preds = preds[test_idx.cpu().numpy()]
        f1 = f1_score(test_true, test_preds)
        mcc = matthews_corrcoef(test_true, test_preds)
        prec = precision_score(test_true, test_preds)
        rec = recall_score(test_true, test_preds)

        print(f"Test set results: test_loss= {loss_test.item():.4f} | "
              f"test_accuracy= {acc_test.item():.4f} | precision= {prec:.4f} | "
              f"recall= {rec:.4f} | f1_score= {f1:.4f} | mcc= {mcc:.4f}")

        swanlab.log({
            "Test/Loss": loss_test.item(),
            "Test/Accuracy": acc_test.item(),
            "Test/Precision": prec,
            "Test/Recall": rec,
            "Test/F1": f1,
            "Test/MCC": mcc
        })

        # 追加到实验结果汇总表 (与 TwiBot-22 共用同一张表, 表头必须完全一致)
        append_result_row(args.results_csv, {
            "dataset": "TwiBot-20",
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "commit": git_state,
            "run_name": run_name,
            "model": model_version,
            "variant": args.variant or FUSION_MODE,
            "fusion": FUSION_MODE,
            "seed": args.seed,
            "no_temporal": int(args.no_temporal),
            "temporal_source": os.path.basename(npz_path) if npz_path else "",
            "align_beta": "",
            "seq_len": args.feat_seq_len,
            "ts_layers": args.feat_num_layers,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "embedding_size": args.emb_size,
            "scheduler": "cosine",
            "class_weight": "none",
            "epochs_run": epochs_run,
            "best_val_acc": round(float(best_val_acc), 4),
            "test_acc": round(acc_test.item(), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "mcc": round(float(mcc), 4),
        })


# ------------------- 训练循环 -------------------
# early_stopping = EarlyStopping(patience=1000, verbose=True, path='best_model.pth') # 已禁用

train_losses = []
val_losses = []
best_val_acc = -1.0  # 初始化最佳验证准确率
epochs_run = 0
suffix = args.save_suffix if args.save_suffix else ""
checkpoint_dir = os.path.join(BASE_DIR, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)
best_model_path = os.path.join(checkpoint_dir, f'best_model{suffix}_s{args.seed}.pth')

for epoch in range(epochs):
    # E2E 预热结束: 解冻 Transformer
    # if args.e2e and epoch == args.ts_warmup_epochs and args.ts_warmup_epochs > 0:
    #     print(f"\n{'='*60}")
    #     print(f">>> 预热结束 (epoch {epoch}): 解冻 Transformer, 开始联合训练")
    #     print(f"{'='*60}")
    #     for n, p in model.named_parameters():
    #         if 'ts_' in n:
    #             p.requires_grad = True
    #     optimizer.param_groups[0]['lr'] = lr * args.ts_lr_scale
    #     best_val_acc = -1.0  # 重置，给联合训练新机会
    #     print(f"  Transformer lr 恢复为 {lr * args.ts_lr_scale:.2e}")
    #     print(f"  best_val_acc 已重置\n")

    epochs_run = epoch + 1
    train_loss, val_loss, acc_val = train(epoch) # train函数需要返回acc_val
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    # warmup 期间不 step scheduler
    # if not (args.e2e and args.ts_warmup_epochs > 0 and epoch < args.ts_warmup_epochs):
    #     scheduler.step()
    scheduler.step()

    # 基于验证准确率检查并保存最佳模型
    if acc_val > best_val_acc:
        best_val_acc = acc_val
        torch.save(model.state_dict(), best_model_path)
        print(f"Validation accuracy improved. Saving model to {best_model_path} (acc: {acc_val:.4f})")


# Plot loss
# plt.figure(figsize=(10, 6))
# plt.plot(train_losses, label='Training Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# suffix = args.save_suffix if args.save_suffix else ""
# plot_path = os.path.join(os.path.dirname(__file__), f'loss_plot{suffix}.png')
# plt.savefig(plot_path)
# print(f"Loss plot saved to {plot_path}")

# Load the best model
model.load_state_dict(torch.load(best_model_path, weights_only=True))

test(epochs_run, best_val_acc)

swanlab.finish()