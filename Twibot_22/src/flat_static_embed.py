#!/usr/bin/env python3
"""
Flat-Static 嵌入生成器
======================
将 14 维静态特征 (时间步均值) 零填充到 64 维，
作为 "无 Transformer 编码" 的消融基线。

用法:
    python flat_static_embed.py /root/autodl-tmp/twibot22
    python flat_static_embed.py /root/autodl-tmp/twibot22 --dataset twibot22
    python flat_static_embed.py /path/to/my_model --dataset twibot20
"""
import os
import sys
import argparse
import numpy as np


def generate_flat_embed(work_dir, dataset="twibot22", embed_dim=64):
    if dataset == "twibot22":
        tmp_dir = os.path.join(work_dir, "tmp_twibot22")
        out_name = "twibot22_transformer_vectors_flat.npz"
    else:
        tmp_dir = os.path.join(work_dir, "tmp_v6")
        out_name = "twibot20_transformer_vectors_flat.npz"

    out_dir = os.path.join(work_dir, "feature_model_outputs")
    os.makedirs(out_dir, exist_ok=True)

    all_ids, all_vecs = [], []
    for split in ['train', 'dev', 'test', 'support']:
        npz_path = os.path.join(tmp_dir, f"{split}_matrices.npz")
        if not os.path.exists(npz_path):
            print(f"  跳过: {npz_path}")
            continue
        data = np.load(npz_path, allow_pickle=True)
        matrices = data['matrices']  # (N, T, F)
        ids = data['ids']
        # 取时间步均值 → (N, F)
        flat = matrices.mean(axis=1)
        F = flat.shape[1]
        # 零填充到 embed_dim
        padded = np.zeros((flat.shape[0], embed_dim), dtype=np.float32)
        padded[:, :F] = flat
        all_ids.extend(ids)
        all_vecs.append(padded)
        print(f"  {split}: {len(ids)} 用户, F={F}")

    if not all_vecs:
        print("错误: 未找到任何矩阵文件!")
        sys.exit(1)

    vectors = np.concatenate(all_vecs, axis=0)
    out_path = os.path.join(out_dir, out_name)
    np.savez_compressed(out_path, vectors=vectors,
                        user_ids=np.array(all_ids, dtype=object))
    print(f"Flat-Static 嵌入已保存: {out_path}, shape={vectors.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flat-Static 嵌入生成器")
    parser.add_argument("work_dir", help="工作目录")
    parser.add_argument("--dataset", default="twibot22",
                        choices=["twibot22", "twibot20"],
                        help="数据集 (决定 tmp 目录和输出文件名)")
    parser.add_argument("--embed-dim", type=int, default=64,
                        help="嵌入维度 (默认 64)")
    args = parser.parse_args()
    generate_flat_embed(args.work_dir, args.dataset, args.embed_dim)
