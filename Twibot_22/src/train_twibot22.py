#!/usr/bin/env python3
"""
TwiBot-22 训练脚本 — 适配 TwiBot-22 预处理数据
================================================
用法:
    python train_twibot22.py
    python train_twibot22.py --work-dir /root/autodl-tmp/twibot22 --model v2
    python train_twibot22.py --work-dir /root/autodl-tmp/twibot22 --model v2 --epochs 200 --patience 30
"""

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys
import csv
import math
import random
import argparse
import copy
import datetime
import subprocess
import traceback

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup
import numpy as np
import swanlab

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha # 可以是 tensor

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score

from model import BotRGCN, BotRGCN_Pure, BotRGCN_v2, BotRGCN_v3, BotRGCN_E2E, info_nce_loss
from utils import accuracy, init_weights

# ------------------- 参数配置 -------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 仓库根目录 (Twibot_22/src → 上两级)，实验结果表放这里以便随代码一起版本控制
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
    # torch.use_deterministic_algorithms(True)
    print(f"随机种子已固定: {seed}")


def get_git_state():
    """
    返回当前代码的 git 版本标识, 形如 'a1b2c3d' 或 'a1b2c3d-dirty'。
    带 -dirty 后缀说明运行时工作区存在未提交改动, 该结果不可复现, 不应写入论文。
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        return sha + ("-dirty" if dirty else "")
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


def main():
    parser = argparse.ArgumentParser(description="TwiBot-22 GNN 训练")
    parser.add_argument("--work-dir", default=SCRIPT_DIR, help="工作目录")
    parser.add_argument("--model", default="v2", choices=["pure", "v1", "v2", "v3"],
                        help="模型版本 (pure=纯净BotRGCN基线, 不使用任何时序特征)")
    parser.add_argument("--gnn-layers", type=int, default=3, help="v3: GNN层数")
    parser.add_argument("--gat-heads", type=int, default=4, help="v3: GAT注意力头数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30, help="早停耐心值(基于 val F1)")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪 (<=0 表示禁用)")
    parser.add_argument("--no-scheduler", action="store_true",
                        help="禁用学习率调度器 (使用恒定lr, 复现官方 BotRGCN 时需要)")
    parser.add_argument("--no-class-weight", action="store_true", help="禁用类别权重")
    parser.add_argument("--class-weight", type=str, default=None,
                        help="手动指定类别权重,格式: 'w0,w1' 如 '1.0,3.0'")
    parser.add_argument("--weight-mode", default="sqrt", choices=["sqrt", "inv", "manual"],
                        help="权重计算方式: sqrt(平方根), inv(反比例), manual(手动)")
    parser.add_argument("--use-focal-loss", action="store_true", help="使用 Focal Loss 替代 CrossEntropy")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal Loss 的 gamma 参数")
    parser.add_argument("--no-graph", action="store_true",
                        help="消融: 跳过 GNN 层, 仅静态+时序过 MLP (Time-only)")
    parser.add_argument("--concat-fusion", action="store_true",
                        help="消融: 用简单拼接融合替代门控融合")
    parser.add_argument("--variant", type=str, default="",
                        help="实验变体标签, 写入 results.csv 用于聚合 (如 full/wo_temporal/flat_static)")
    parser.add_argument("--no-temporal", action="store_true",
                        help="置零时序特征 (Only-RGCN 消融实验用)")
    parser.add_argument("--temporal-npz", type=str, default=None,
                        help="指定时序特征 npz 文件路径 (默认: twibot22_transformer_vectors.npz)")
    parser.add_argument("--save-suffix", type=str, default="",
                        help="模型保存后缀, 避免覆盖 (e.g. '_no_temporal')")
    # --- E2E 端到端训练参数 ---
    parser.add_argument("--e2e", action="store_true",
                        help="启用端到端训练 (Transformer+GNN联合优化)")
    parser.add_argument("--seq-len", type=int, default=16,
                        help="E2E模式下的时间序列长度 (截断或填充至此长度)")
    parser.add_argument("--ts-nhead", type=int, default=2,
                        help="E2E: Transformer注意力头数")
    parser.add_argument("--ts-layers", type=int, default=2,
                        help="E2E: Transformer编码层数")
    parser.add_argument("--ts-lr-scale", type=float, default=0.1,
                        help="E2E: Transformer学习率缩放因子 (相对于主lr)")
    parser.add_argument("--chunk-size", type=int, default=50000,
                        help="E2E: Transformer分块处理大小")
    parser.add_argument("--align-beta", type=float, default=0.0,
                        help="跨分支对齐损失权重 (0=禁用, 建议0.1~0.5)")
    parser.add_argument("--align-temp", type=float, default=0.5,
                        help="InfoNCE温度参数")
    parser.add_argument("--e2e-matrix-dir", type=str, default=None,
                        help="E2E: 原始时序矩阵目录 (默认: $work-dir/tmp_twibot22)")
    parser.add_argument("--ts-warmup-epochs", type=int, default=10,
                        help="E2E: Transformer 冻结预热轮数 (前 K 轮只训练 GNN, 默认 10)")
    parser.add_argument("--ts-pretrained", type=str, default=None,
                        help="E2E: 预训练 Transformer 权重路径 (BotClassifier .pt), 用于热启动")
    parser.add_argument("--results-csv", type=str, default=DEFAULT_RESULTS_CSV,
                        help=f"实验结果汇总表路径 (默认: {DEFAULT_RESULTS_CSV})")
    args = parser.parse_args()

    if args.model == "pure" and args.e2e:
        parser.error("--model pure 与 --e2e 互斥: 纯净基线不含任何时序通路")

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print(f"模型版本: {args.model}")
    print(f"工作目录: {args.work_dir}")

    # ------------------- 代码版本标识 -------------------
    git_state = get_git_state()
    run_name = f"BotRGCN_{args.model}{args.save_suffix}_s{args.seed}"

    # 融合方式在整个 run 内固定，训练与测试必须一致
    if args.model == 'pure' and not args.e2e:
        fusion_mode = 'n/a'  # 纯净基线没有时序通路, 不存在融合
    elif args.no_graph:
        fusion_mode = 'no_graph'
    elif args.no_temporal:
        fusion_mode = 'none'
    elif args.concat_fusion:
        fusion_mode = 'concat'
    else:
        fusion_mode = 'gated'
    print(f"融合模式: {fusion_mode}")
    # E2E 模型的 forward 不接受 fusion_type
    fwd_kwargs = {} if args.e2e else {"fusion_type": fusion_mode}
    print(f"代码版本: {git_state}")
    if git_state.endswith("-dirty"):
        print("!!! 警告: 工作区存在未提交改动, 本次结果无法复现, 不要写入论文 !!!")

    # ------------------- 初始化 SwanLab -------------------
    # 将 argparse 的参数转换为字典，方便记录
    config_dict = vars(args)
    # 添加一些额外的配置信息
    config_dict['dataset'] = 'TwiBot-22'
    config_dict['class_weight_strategy'] = 'sqrt' if not args.no_class_weight else 'none'
    config_dict['git_commit'] = git_state

    # 初始化 SwanLab 实验
    # project: 项目名称，比如 "TwiBot22-BotDetection"
    # name: 本次实验的名称，可以用 save_suffix 区分
    # config: 记录超参数
    swanlab.init(
        project="TwiBot22-BotDetection",
        name=run_name,
        config=config_dict,
        logdir=os.path.join(args.work_dir, "swanlog")
    )

    # ------------------- 路径 -------------------
    processed_dir = os.path.join(args.work_dir, "saved_data", "twibot22_data", "processed_data")
    if args.temporal_npz:
        temporal_npz_path = args.temporal_npz
    else:
        temporal_npz_path = os.path.join(args.work_dir, "feature_model_outputs", "twibot22_transformer_vectors.npz")
    user_ids_path = os.path.join(processed_dir, "user_ids.npy")

    # ------------------- 加载预处理数据 -------------------
    print("=== 加载预处理数据 ===")

    def load_tensor(name):
        path = os.path.join(processed_dir, name)
        print(f"  加载 {name}")
        return torch.load(path, map_location=device, weights_only=True)

    des_tensor = load_tensor("des_tensor.pt")
    tweets_tensor = load_tensor("tweets_tensor.pt")
    num_prop = load_tensor("num_properties_tensor.pt")
    cat_prop = load_tensor("cat_properties_tensor.pt")
    edge_index = load_tensor("edge_index.pt")
    edge_type = load_tensor("edge_type.pt")
    labels = load_tensor("label.pt").long().squeeze()
    train_idx = load_tensor("train_idx.pt")
    val_idx = load_tensor("val_idx.pt")
    test_idx = load_tensor("test_idx.pt")

    num_nodes = des_tensor.shape[0]
    print(f"节点数: {num_nodes}")
    print(f"边数: {edge_index.shape[1]}")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # ------------------- 加载时序特征 -------------------
    # 构建 user_id → graph_index 映射 (pure 模式不需要)
    if args.model == "pure":
        id_to_idx = None
    else:
        ordered_user_ids = list(np.load(user_ids_path, allow_pickle=True))
        id_to_idx = {str(uid): i for i, uid in enumerate(ordered_user_ids)}

    if args.model == "pure":
        # === 纯净基线: 完全不涉及时序, 连 npz 都不读 ===
        print("=== 纯净 BotRGCN 模式: 不加载任何时序特征 ===")
        raw_ts_tensor = None
        temporal_tensor = None
    elif args.e2e:
        # === E2E 模式: 加载原始时间序列矩阵 ===
        print("=== E2E模式: 加载原始时间序列矩阵 ===")
        tmp_dir = args.e2e_matrix_dir or os.path.join(args.work_dir, "tmp_twibot22")
        print(f"  矩阵目录: {tmp_dir}")
        target_T = args.seq_len
        raw_ts_tensor = torch.zeros((num_nodes, target_T, 14), dtype=torch.float32, device=device)
        total_mapped = 0
        for split_name in ['train', 'dev', 'test', 'support']:
            split_npz = os.path.join(tmp_dir, f"{split_name}_matrices.npz")
            if not os.path.exists(split_npz):
                continue
            data = np.load(split_npz, allow_pickle=True)
            matrices = data['matrices']  # (N_split, T_orig, F)
            ids = data['ids']
            T_orig = matrices.shape[1]
            # 截断或填充到 target_T
            if T_orig >= target_T:
                matrices = matrices[:, :target_T, :]
            else:
                pad = np.zeros((matrices.shape[0], target_T - T_orig, matrices.shape[2]), dtype=np.float32)
                matrices = np.concatenate([matrices, pad], axis=1)
            # 对齐到图索引
            graph_indices = []
            npz_indices = []
            for i, uid in enumerate(ids):
                idx = id_to_idx.get(str(uid))
                if idx is not None:
                    graph_indices.append(idx)
                    npz_indices.append(i)
            if graph_indices:
                raw_ts_tensor[graph_indices] = torch.from_numpy(
                    matrices[npz_indices]).to(device)
                total_mapped += len(graph_indices)
            print(f"  {split_name}: 映射 {len(graph_indices)} 个用户")
        print(f"  总映射: {total_mapped}, 原始时序形状: {raw_ts_tensor.shape}")
        temporal_tensor = None  # E2E 模式不使用预计算嵌入
    else:
        # === 原始模式: 加载预计算的 Transformer 嵌入 ===
        print("=== 加载预计算时序特征 ===")
        raw_ts_tensor = None  # 非 E2E 模式不使用原始矩阵
        try:
            if not os.path.exists(temporal_npz_path):
                raise FileNotFoundError(f"时序特征文件未找到: {temporal_npz_path}")

            npz_data = np.load(temporal_npz_path, allow_pickle=True)
            npz_vectors = npz_data['vectors']
            npz_user_ids = [str(uid) for uid in npz_data['user_ids']]
            print(f"  npz 向量: {npz_vectors.shape}, 用户数: {len(npz_user_ids)}")

            embedding_dim = npz_vectors.shape[1]
            temporal_tensor = torch.zeros((num_nodes, embedding_dim), device=device)

            mapped, unmapped = 0, 0
            for i, uid in enumerate(npz_user_ids):
                idx = id_to_idx.get(uid)
                if idx is not None:
                    temporal_tensor[idx] = torch.from_numpy(npz_vectors[i]).to(device)
                    mapped += 1
                else:
                    unmapped += 1

            print(f"  已映射: {mapped}, 未映射: {unmapped}")
            print(f"  时序张量: {temporal_tensor.shape}")

        except Exception as e:
            print(f"错误: 加载时序特征失败 — {e}")
            traceback.print_exc()
            print("使用全 0 时序特征")
            temporal_tensor = torch.zeros((num_nodes, 64), device=device)

        if args.no_temporal:
            print("*** --no-temporal: 时序特征已置零 (Only-RGCN 消融模式) ***")
            temporal_tensor = torch.zeros_like(temporal_tensor)

    # ------------------- 模型 -------------------
    if args.e2e:
        print(f"=== 初始化 E2E 端到端模型 ===")
        model = BotRGCN_E2E(
            num_prop_size=num_prop.shape[1],
            cat_prop_size=cat_prop.shape[1],
            ts_input_dim=raw_ts_tensor.shape[2],  # 14
            d_model=args.embedding_size,
            nhead=args.ts_nhead,
            num_transformer_layers=args.ts_layers,
            embedding_dimension=args.embedding_size,
            dropout=args.dropout,
            chunk_size=args.chunk_size,
            use_checkpoint=True
        ).to(device)
        model.apply(init_weights)
        model._init_ts_weights()  # 重新初始化 Transformer 权重
        # 加载预训练 Transformer 权重 (热启动)
        if args.ts_pretrained:
            print(f"  加载预训练 Transformer: {args.ts_pretrained}")
            pretrained_sd = torch.load(args.ts_pretrained, map_location=device, weights_only=True)
            # 映射: BotClassifier.encoder.* → BotRGCN_E2E.ts_*
            mapping = {
                'encoder.input_proj.': 'ts_input_proj.',
                'encoder.pos_encoder.': 'ts_pos_enc.',
                'encoder.transformer.': 'ts_transformer.',
            }
            mapped_sd = {}
            skipped = 0
            model_keys = set(model.state_dict().keys())
            for pt_key, pt_val in pretrained_sd.items():
                matched = False
                for src_prefix, dst_prefix in mapping.items():
                    if pt_key.startswith(src_prefix):
                        e2e_key = dst_prefix + pt_key[len(src_prefix):]
                        if e2e_key in model_keys:
                            mapped_sd[e2e_key] = pt_val
                        else:
                            skipped += 1
                        matched = True
                        break
                if not matched:
                    skipped += 1
            model.load_state_dict(mapped_sd, strict=False)
            print(f"  预训练权重: 加载 {len(mapped_sd)} 个, 跳过 {skipped} 个 (classifier 等)")
        n_params = sum(p.numel() for p in model.parameters())
        n_ts_params = sum(p.numel() for n, p in model.named_parameters() if 'ts_' in n)
        print(f"  总参数量: {n_params:,}, 其中 Transformer: {n_ts_params:,}")
    else:
        print(f"=== 初始化模型: {args.model} ===")
        if args.model == "pure":
            model = BotRGCN_Pure(
                num_prop_size=num_prop.shape[1],
                cat_prop_size=cat_prop.shape[1],
                embedding_dimension=args.embedding_size,
                dropout=args.dropout
            ).to(device)
        elif args.model == "v3":
            model = BotRGCN_v3(
                num_prop_size=num_prop.shape[1],
                cat_prop_size=cat_prop.shape[1],
                time_size=temporal_tensor.shape[1],
                embedding_dimension=args.embedding_size,
                gnn_layers=args.gnn_layers,
                gat_heads=args.gat_heads
            ).to(device)
        elif args.model == "v2":
            model = BotRGCN_v2(
                num_prop_size=num_prop.shape[1],
                cat_prop_size=cat_prop.shape[1],
                time_size=temporal_tensor.shape[1],
                embedding_dimension=args.embedding_size
            ).to(device)
        else:
            model = BotRGCN(
                num_prop_size=num_prop.shape[1],
                cat_prop_size=cat_prop.shape[1],
                time_size=temporal_tensor.shape[1],
                embedding_dimension=args.embedding_size
            ).to(device)
        model.apply(init_weights)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  总参数量: {n_params:,}")

    # ------------------- 类别权重 -------------------
    train_labels = labels[train_idx]
    valid_mask = train_labels >= 0
    valid_labels = train_labels[valid_mask]
    n_class0 = (valid_labels == 0).sum().float()
    n_class1 = (valid_labels == 1).sum().float()
    total = n_class0 + n_class1
    print(f"类别分布: class0={int(n_class0)}, class1={int(n_class1)}, 比例={n_class0/total:.4f}/{n_class1/total:.4f}")

    if args.no_class_weight:
        class_weights = None
        print("未使用类别权重")
    elif args.class_weight:
        # 手动权重
        w0, w1 = [float(x) for x in args.class_weight.split(',')]
        class_weights = torch.tensor([w0, w1], device=device)
        print(f"手动类别权重: [{w0:.4f}, {w1:.4f}]")
    else:
        # 自动计算
        inv_w0 = total / (2.0 * n_class0)
        inv_w1 = total / (2.0 * n_class1)
        if args.weight_mode == "sqrt":
            w0 = math.sqrt(inv_w0)
            w1 = math.sqrt(inv_w1)
            print(f"sqrt 类别权重: [{w0:.4f}, {w1:.4f}] (原始反比例: [{inv_w0:.4f}, {inv_w1:.4f}])")
        else:  # inv
            w0, w1 = inv_w0, inv_w1
            print(f"反比例类别权重: [{w0:.4f}, {w1:.4f}]")
        class_weights = torch.tensor([w0, w1], device=device)

    # ------------------- 损失函数与优化器 -------------------
    if args.use_focal_loss:
        print(f"使用 Focal Loss (gamma={args.focal_gamma})")
        loss_fn = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    if args.e2e:
        # 分组学习率: Transformer 用较小 lr，GNN 用正常 lr
        ts_params = [p for n, p in model.named_parameters() if 'ts_' in n]
        other_params = [p for n, p in model.named_parameters() if 'ts_' not in n]
        optimizer = torch.optim.AdamW([
            {'params': ts_params, 'lr': args.lr * args.ts_lr_scale},
            {'params': other_params, 'lr': args.lr}
        ], weight_decay=args.weight_decay)
        print(f"  Transformer lr: {args.lr * args.ts_lr_scale:.2e}, GNN lr: {args.lr:.2e}")
        print(f"  Transformer 预热: 前 {args.ts_warmup_epochs} 轮冻结 Transformer, 仅训练 GNN")
        if args.align_beta > 0:
            print(f"  对齐损失: InfoNCE, beta={args.align_beta}, temp={args.align_temp}")

        # 预热阶段: 冻结 Transformer 参数
        ts_param_names = [n for n, _ in model.named_parameters() if 'ts_' in n]
        if args.ts_warmup_epochs > 0:
            for n, p in model.named_parameters():
                if 'ts_' in n:
                    p.requires_grad = False
            # Transformer param group lr 设为 0
            optimizer.param_groups[0]['lr'] = 0.0
            print(f"  [预热] 已冻结 {len(ts_param_names)} 个 Transformer 参数")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 计算总步数和 warmup 步数
    if args.no_scheduler:
        scheduler = None
        print("已禁用学习率调度器: 使用恒定 lr")
    else:
        total_steps = args.epochs
        warmup_steps = int(total_steps * 0.1) # 10% 的时间用于 warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

    # ------------------- 预计算有效索引 -------------------
    train_mask = labels[train_idx] >= 0
    valid_train_idx = train_idx[train_mask]
    val_mask = labels[val_idx] >= 0
    valid_val_idx = val_idx[val_mask]

    # ------------------- 动态权重计算函数 -------------------
    def get_dynamic_weights(epoch, base_weights, total_epochs=100, decay_start=30, decay_end=60):
        """
        动态计算类别权重：
        - epoch < decay_start: 使用 base_weights (如 sqrt 权重)
        - decay_start <= epoch <= decay_end: 线性衰减到 [1.0, 1.0]
        - epoch > decay_end: 使用 [1.0, 1.0] (无权重)
        若 base_weights 为 None (--no-class-weight), 则始终返回 None。
        """
        if base_weights is None:
            return None
        if epoch < decay_start:
            return base_weights
        elif epoch > decay_end:
            return torch.tensor([1.0, 1.0], device=device)
        else:
            # 线性插值
            alpha = (epoch - decay_start) / (decay_end - decay_start)
            target_weights = torch.tensor([1.0, 1.0], device=device)
            return (1 - alpha) * base_weights + alpha * target_weights

    # ------------------- 训练 -------------------
    # 选择正确的时序输入
    # pure 模式下 temporal_tensor 为 None, 模型内部会忽略该形参
    ts_input = raw_ts_tensor if args.e2e else temporal_tensor
    use_align = args.e2e and args.align_beta > 0

    def do_train(epoch):
        model.train()
        
        # 强制清空缓存，缓解 scatter_add_ 的显存峰值
        torch.cuda.empty_cache()

        # 动态更新 Loss 函数的权重
        current_weights = get_dynamic_weights(epoch, class_weights, args.epochs)
        if args.use_focal_loss:
            loss_fn = FocalLoss(alpha=current_weights, gamma=args.focal_gamma)
        else:
            loss_fn = nn.CrossEntropyLoss(weight=current_weights)

        if use_align:
            output, g_emb, t_emb = model(
                des_tensor, tweets_tensor, num_prop, cat_prop,
                ts_input, edge_index, edge_type, return_embeddings=True)
        else:
            output = model(des_tensor, tweets_tensor, num_prop, cat_prop,
                           ts_input, edge_index, edge_type, **fwd_kwargs)

        loss_cls = loss_fn(output[valid_train_idx], labels[valid_train_idx])
        loss_train = loss_cls

        if use_align:
            loss_align = info_nce_loss(
                g_emb[valid_train_idx], t_emb[valid_train_idx],
                temperature=args.align_temp)
            loss_train = loss_cls + args.align_beta * loss_align

        acc_train = accuracy(output[valid_train_idx], labels[valid_train_idx])

        optimizer.zero_grad()
        loss_train.backward()
        
        # 再次清空缓存
        torch.cuda.empty_cache()
        
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        # 注意: scheduler.step() 统一在主训练循环中调用, 此处不要重复调用,
        # 否则每个 epoch 会前进两步, 余弦周期提前走完。

        # 验证
        model.eval()
        with torch.no_grad():
            output_val = model(des_tensor, tweets_tensor, num_prop, cat_prop,
                               ts_input, edge_index, edge_type, **fwd_kwargs)
            loss_val = loss_fn(output_val[valid_val_idx], labels[valid_val_idx])
            
            preds_val = output_val[valid_val_idx].argmax(dim=1)
            acc_val = (preds_val == labels[valid_val_idx]).float().mean()
            
            preds_np = preds_val.cpu().numpy()
            labels_np = labels[valid_val_idx].cpu().numpy()
            f1_val = f1_score(labels_np, preds_np, average='binary')
            prec_val = precision_score(labels_np, preds_np, average='binary', zero_division=0)
            rec_val = recall_score(labels_np, preds_np, average='binary', zero_division=0)

        cur_lr = optimizer.param_groups[0]['lr']
        align_str = f' | L_align: {loss_align.item():.4f}' if use_align else ''
        print(f'Epoch {epoch + 1:04d} | loss_train: {loss_train.item():.4f}{align_str} | '
              f'acc_train: {acc_train.item():.4f} | '
              f'loss_val: {loss_val.item():.4f} | acc_val: {acc_val.item():.4f} | '
              f'F1: {f1_val:.4f} | P: {prec_val:.4f} | R: {rec_val:.4f} | lr: {cur_lr:.2e}')
              
        # 将指标记录到 SwanLab
        swanlab.log({
            "Train/Loss": loss_train.item(),
            "Train/Accuracy": acc_train.item(),
            "Val/Loss": loss_val.item(),
            "Val/Accuracy": acc_val.item(),
            "Learning_Rate": cur_lr
        }, step=epoch + 1)
        
        return loss_train.item(), loss_val.item(), acc_val.item(), f1_val

    def do_test(epochs_run, best_val_acc):
        model.eval()
        with torch.no_grad():
            output = model(des_tensor, tweets_tensor, num_prop, cat_prop,
                           ts_input, edge_index, edge_type, **fwd_kwargs)

            test_mask = labels[test_idx] >= 0
            valid_test_idx = test_idx[test_mask]

            # 用无权重 loss 评估
            loss_test = nn.CrossEntropyLoss()(output[valid_test_idx], labels[valid_test_idx])
            acc_test = accuracy(output[valid_test_idx], labels[valid_test_idx])

            preds = output[valid_test_idx].max(1)[1].cpu().numpy()
            true = labels[valid_test_idx].cpu().numpy()
            f1 = f1_score(true, preds)
            mcc = matthews_corrcoef(true, preds)
            prec = precision_score(true, preds, zero_division=0)
            rec = recall_score(true, preds, zero_division=0)

            print(f"\n=== 测试集结果 ===")
            print(f"  test_loss     = {loss_test.item():.4f}")
            print(f"  test_accuracy = {acc_test.item():.4f}")
            print(f"  precision     = {prec:.4f}")
            print(f"  recall        = {rec:.4f}")
            print(f"  f1_score      = {f1:.4f}")
            print(f"  mcc           = {mcc:.4f}")

            # 将最终测试结果记录到 SwanLab
            swanlab.log({
                "Test/Loss": loss_test.item(),
                "Test/Accuracy": acc_test.item(),
                "Test/F1": f1,
                "Test/Precision": prec,
                "Test/Recall": rec,
                "Test/MCC": mcc
            })

            # 追加到实验结果汇总表 (受版本控制, 每行绑定一个 commit)
            append_result_row(args.results_csv, {
                "dataset": "TwiBot-22",
                "time": datetime.datetime.now().isoformat(timespec="seconds"),
                "commit": git_state,
                "run_name": run_name,
                "model": "e2e" if args.e2e else args.model,
                "variant": args.variant or fusion_mode,
                "fusion": fusion_mode,
                "seed": args.seed,
                "no_temporal": int(args.no_temporal),
                "temporal_source": os.path.basename(temporal_npz_path) if not args.e2e else "e2e_raw",
                "align_beta": args.align_beta,
                "seq_len": args.seq_len if args.e2e else "",
                "ts_layers": args.ts_layers if args.e2e else "",
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "dropout": args.dropout,
                "embedding_size": args.embedding_size,
                "scheduler": "none" if args.no_scheduler else "cosine_warmup",
                "class_weight": "none" if args.no_class_weight else args.weight_mode,
                "epochs_run": epochs_run,
                "best_val_acc": round(float(best_val_acc), 4),
                "test_acc": round(acc_test.item(), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1": round(float(f1), 4),
                "mcc": round(float(mcc), 4),
            })

    # 训练循环
    train_losses, val_losses, val_f1s = [], [], []
    best_val_acc = -1.0
    patience_counter = 0
    epochs_run = 0
    suffix = args.save_suffix if args.save_suffix else ""
    best_model_path = os.path.join(args.work_dir, f"best_model_twibot22{suffix}.pth")

    for epoch in range(args.epochs):
        # E2E 预热结束: 解冻 Transformer
        if args.e2e and epoch == args.ts_warmup_epochs and args.ts_warmup_epochs > 0:
            print(f"\n{'='*60}")
            print(f">>> 预热结束 (epoch {epoch}): 解冻 Transformer, 开始联合训练")
            print(f"{'='*60}")
            for n, p in model.named_parameters():
                if 'ts_' in n:
                    p.requires_grad = True
            optimizer.param_groups[0]['lr'] = args.lr * args.ts_lr_scale
            # 重置 patience，给联合训练一个全新的机会
            best_val_acc = -1.0
            patience_counter = 0
            print(f"  Transformer lr 恢复为 {args.lr * args.ts_lr_scale:.2e}")
            print(f"  patience 计数器已重置\n")

        epochs_run = epoch + 1
        tl, vl, va, vf1 = do_train(epoch)
        train_losses.append(tl)
        val_losses.append(vl)
        val_f1s.append(vf1)
        # warmup 期间不 step scheduler，让联合训练阶段拥有完整余弦周期
        if scheduler is not None and not (
                args.e2e and args.ts_warmup_epochs > 0 and epoch < args.ts_warmup_epochs):
            scheduler.step()

        if va > best_val_acc:
            best_val_acc = va
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ★ 保存最佳模型 (val_acc={va:.4f}, val_F1={vf1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n早停触发: {args.patience} 轮未提升 (best val_acc={best_val_acc:.4f})")
                break

    # 加载最佳模型并测试
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    do_test(epochs_run, best_val_acc)
    
    # 结束 SwanLab 实验
    swanlab.finish()


if __name__ == "__main__":
    main()
