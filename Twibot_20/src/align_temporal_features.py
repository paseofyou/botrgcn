import torch
import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm

# 路径配置 (根据你的文件结构调整)
CUR_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(CUR_DIR, "Data", "Twibot-20")
SAVED_DATA_DIR = os.path.join(CUR_DIR, "saved_data", "twibot20_data.pt", "Twibot20_processed_data", "processed_data")
TIME_SERIES_NPZ = os.path.join(CUR_DIR, "feature_model_outputs", "twibot20_transformer_vectors.npz")
OUTPUT_PATH = os.path.join(SAVED_DATA_DIR, "temporal_tensor.pt")

def align_temporal_features():
    print("1. 加载原始 ID 顺序以匹配现有图结构...")
    # 必须严格遵守 Dataset.py 的拼接顺序: train -> dev -> test -> support
    # 注意：Dataset.py 中 dev 在 train 之后，test 之前 (看 Dataset.py 第 24 行)
    # self.df_data = pd.concat([df_train, df_dev, df_test, df_support], ignore_index=True)
    
    try:
        df_train = pd.read_json(os.path.join(DATA_DIR, 'train.json'))
        df_dev = pd.read_json(os.path.join(DATA_DIR, 'dev.json'))
        df_test = pd.read_json(os.path.join(DATA_DIR, 'test.json'))
        df_support = pd.read_json(os.path.join(DATA_DIR, 'support.json'))
    except ValueError:
         print("错误：找不到 JSON 数据文件，请检查 DATA_DIR 路径")
         return

    # 提取 ID 列
    df_train = df_train[['ID']]
    df_dev = df_dev[['ID']]
    df_test = df_test[['ID']]
    df_support = df_support[['ID']]
    
    # 拼接 (顺序至关重要！)
    df_all = pd.concat([df_train, df_dev, df_test, df_support], ignore_index=True)
    ordered_ids = df_all['ID'].astype(str).tolist()
    
    num_nodes = len(ordered_ids)
    print(f"   总节点数: {num_nodes}")

    print("2. 加载时序特征 (NPZ)...")
    if not os.path.exists(TIME_SERIES_NPZ):
        print(f"错误：找不到时序特征文件 {TIME_SERIES_NPZ}")
        print("请先运行 feature_extraction.py 生成该文件。")
        return

    data = np.load(TIME_SERIES_NPZ, allow_pickle=True)
    vectors = data['vectors']      # Shape: (N_extracted, Hidden_Dim)
    user_ids = data['user_ids']    # Shape: (N_extracted,)
    
    # 创建快速查找字典
    id_to_vector = {}
    print("   构建查找字典...")
    for uid, vec in zip(user_ids, vectors):
        id_to_vector[str(uid)] = vec
        
    feature_dim = vectors.shape[1]
    print(f"   时序特征维度: {feature_dim}")

    print("3. 对齐特征并创建 Tensor...")
    # 初始化全 0 张量
    temporal_tensor = torch.zeros((num_nodes, feature_dim), dtype=torch.float32)
    
    missing_count = 0
    for i, uid in enumerate(tqdm(ordered_ids, desc="Aligning")):
        uid_str = str(uid)
        if uid_str in id_to_vector:
            temporal_tensor[i] = torch.tensor(id_to_vector[uid_str], dtype=torch.float32)
        else:
            # 如果找不到该用户的时序特征（可能爬取失败或数据缺失），保持为 0 或使用均值
            missing_count += 1
            
    print(f"   对齐完成。缺失（全0）节点数: {missing_count}")
    
    print(f"4. 保存结果到 {OUTPUT_PATH}")
    torch.save(temporal_tensor, OUTPUT_PATH)
    print("完成！现在你可以在 train.py 中加载这个 .pt 文件了。")

if __name__ == "__main__":
    align_temporal_features()
