import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="TwiBot-20 智能联动调参脚本")
    
    # --- Feature Extraction 参数 ---
    parser.add_argument("--mode", default="anchor", choices=["pseudo", "anchor", "flat_static"], help="时序构造模式")
    parser.add_argument("--seq-len", type=int, default=32, help="时间序列长度")
    parser.add_argument("--d-model", type=int, default=64, help="Transformer隐藏层维度")
    parser.add_argument("--n-head", type=int, default=8, help="Transformer注意力头数")
    parser.add_argument("--num-layers", type=int, default=2, help="Transformer编码层数")
    parser.add_argument("--feat-dropout", type=float, default=0.1, help="Transformer Dropout率")
    
    # --- Train 参数 ---
    parser.add_argument("--emb-size", type=int, default=128, help="图网络隐藏层维度")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--dropout", type=float, default=0.3, help="图网络Dropout")
    parser.add_argument("--epochs", type=int, default=120, help="训练轮次")
    
    # --- 控制参数 ---
    parser.add_argument("--force-extract", action="store_true", help="强制重新提取特征，即使文件已存在")
    parser.add_argument("--no-graph", action="store_true", help="消融实验: Time-only")
    parser.add_argument("--no-temporal", action="store_true", help="消融实验: Graph-only")
    parser.add_argument("--concat-fusion", action="store_true", help="消融实验: 简单拼接融合")
    
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"🚀 开始运行 Pipeline 实验组合:")
    print(f"Feature: Mode={args.mode}, SeqLen={args.seq_len}, DModel={args.d_model}, Heads={args.n_head}, Layers={args.num_layers}")
    print(f"Train:   EmbSize={args.emb_size}, LR={args.lr}, Dropout={args.dropout}")
    print(f"{'='*50}\n")

    # 1. 生成唯一后缀和特征文件路径
    feat_suffix = f"_T{args.seq_len}_D{args.d_model}_H{args.n_head}_L{args.num_layers}"
    feat_file_name = f"twibot20_transformer_vectors_{args.mode}{feat_suffix}.npz"
    feat_file_path = os.path.join("my_model", "feature_model_outputs", feat_file_name)

    # 2. 智能缓存判断：是否需要运行特征提取
    if args.mode == "flat_static":
        print(">>> 模式为 flat_static，跳过 Transformer 特征提取。")
        # flat_static 模式下，特征提取脚本会生成 twibot20_transformer_vectors_flat_static.npz
        # 我们需要确保这个文件存在，如果不存在则运行一次
        flat_static_path = os.path.join("my_model", "feature_model_outputs", "twibot20_transformer_vectors_flat_static.npz")
        if not os.path.exists(flat_static_path) or args.force_extract:
            print(">>> 正在生成 flat_static 特征...")
            subprocess.run(["python", "my_model/feature_extraction.py", "--mode", "flat_static"], check=True)
        feat_suffix = "" # flat_static 不需要后缀
    else:
        if os.path.exists(feat_file_path) and not args.force_extract:
            print(f">>> 命中缓存！特征文件已存在: {feat_file_name}")
            print(">>> 跳过特征提取阶段，直接开始训练。")
        else:
            print(f">>> 未命中缓存或强制提取。正在运行特征提取...")
            feat_cmd = [
                "python", "my_model/feature_extraction.py",
                "--mode", args.mode,
                "--seq-len", str(args.seq_len),
                "--d-model", str(args.d_model),
                "--n-head", str(args.n_head),
                "--num-layers", str(args.num_layers),
                "--dropout", str(args.feat_dropout),
                "--suffix", feat_suffix
            ]
            subprocess.run(feat_cmd, check=True)

    # 3. 运行模型训练
    print("\n>>> 阶段 2: 训练图神经网络...")
    train_cmd = [
        "python", "my_model/train.py",
        "--ts-mode", f"{args.mode}{feat_suffix}", # 告诉 train.py 读取哪个文件
        "--feat-seq-len", str(args.seq_len),      # 传给 train.py 用于 SwanLab 记录
        "--feat-d-model", str(args.d_model),
        "--feat-n-head", str(args.n_head),
        "--feat-num-layers", str(args.num_layers),
        "--feat-dropout", str(args.feat_dropout),
        "--emb-size", str(args.emb_size),
        "--lr", str(args.lr),
        "--dropout", str(args.dropout),
        "--save-suffix", f"_exp{feat_suffix}_E{args.emb_size}" # 给保存的模型加个后缀
    ]
    
    # 添加消融实验参数
    if args.no_graph: train_cmd.append("--no-graph")
    if args.no_temporal: train_cmd.append("--no-temporal")
    if args.concat_fusion: train_cmd.append("--concat-fusion")

    subprocess.run(train_cmd, check=True)
    print(f"\n✅ 实验组合完成！\n")

if __name__ == "__main__":
    main()
