import math
import torch
from torch import nn
from torch_geometric.nn import RGCNConv, FastRGCNConv, GCNConv, GATConv
import torch.nn.functional as F
import torch.utils.checkpoint as cp


class BotRGCN(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=6, cat_prop_size=11, time_size=64,
                 embedding_dimension=64,
                 dropout=0.3):
        super(BotRGCN, self).__init__()
        self.dropout = dropout
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, int(embedding_dimension / 4)),
            nn.LeakyReLU()
        )
        # self.linear_relu_time = nn.Sequential(
        #     nn.Linear(time_size, int(embedding_dimension / 4)),
        #     nn.LeakyReLU()
        # )

        # Early Fusion: 4 features * (emb/4) = 1.0 * emb
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )

        self.input_norm = nn.LayerNorm(embedding_dimension)
        self.time_norm = nn.LayerNorm(time_size)

        self.rgcn = RGCNConv(embedding_dimension, embedding_dimension, num_relations=2)

        # Late Fusion: Graph Emb (emb) + Time Emb (time_size)
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension + time_size, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

    def forward(self, des, tweet, num_prop, cat_prop, time_feature, edge_index, edge_type):
        # Normalize temporal features first
        time_feature = self.time_norm(time_feature)

        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        # tm = self.linear_relu_time(time_feature)

        # Early Fusion
        # x = torch.cat((d, t, n, c, tm), dim=1)
        x = torch.cat((d, t, n, c), dim=1)

        x = self.linear_relu_input(x)
        x = self.input_norm(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)

        # Late Fusion
        x = torch.cat((x, time_feature), dim=1)

        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x


class BotRGCN_v2(nn.Module):
    """
    改进版 BotRGCN：
    1. 可学习的时序投影层（冻结嵌入 → 任务空间）
    2. 两个独立 RGCN 层（非权重共享）
    3. 门控融合（自适应抑制噪声时序特征）
    """
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=6, cat_prop_size=11,
                 time_size=64, embedding_dimension=64, dropout=0.3):
        super(BotRGCN_v2, self).__init__()
        self.dropout = dropout
        emb = embedding_dimension

        # --- 静态特征编码器 ---
        self.linear_relu_des = nn.Sequential(nn.Linear(des_size, emb // 4), nn.LeakyReLU())
        self.linear_relu_tweet = nn.Sequential(nn.Linear(tweet_size, emb // 4), nn.LeakyReLU())
        self.linear_relu_num_prop = nn.Sequential(nn.Linear(num_prop_size, emb // 4), nn.LeakyReLU())
        self.linear_relu_cat_prop = nn.Sequential(nn.Linear(cat_prop_size, emb // 4), nn.LeakyReLU())

        # --- 时序投影层：冻结嵌入 → 可学习任务空间 ---
        self.time_proj = nn.Sequential(
            nn.Linear(time_size, emb),
            nn.LayerNorm(emb),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb, emb),
            nn.LeakyReLU()
        )

        # --- Early Fusion ---
        self.linear_relu_input = nn.Sequential(nn.Linear(emb, emb), nn.LeakyReLU())
        self.input_norm = nn.LayerNorm(emb)

        # --- 1个独立的 RGCN 层 ---
        self.rgcn = RGCNConv(emb, emb, num_relations=2)
        # self.rgcn2 = RGCNConv(emb, emb, num_relations=2)

        # --- 门控融合 ---
        self.gate_linear = nn.Linear(emb * 2, emb)
        
        # --- MLP 融合 (用于 w/o Graph 和 Concat 消融) ---
        self.mlp_linear = nn.Linear(emb * 2, emb)

        # --- 输出层 ---
        self.output1 = nn.Sequential(nn.Linear(emb, emb), nn.LeakyReLU())
        self.output2 = nn.Linear(emb, 2)

    def forward(self, des, tweet, num_prop, cat_prop, time_feature, edge_index, edge_type, fusion_type='gated'):
        # 1) 时序投影
        t_proj = self.time_proj(time_feature)  # (N, emb)

        # 2) 静态特征编码
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)

        # 3) Early Fusion（静态特征）
        x_static = torch.cat((d, t, n, c), dim=1)  # (N, emb)
        x_static = self.linear_relu_input(x_static)
        x_static = self.input_norm(x_static)

        # 4) 图传播：两个独立 RGCN 层 (如果不是 no_graph 模式)
        if fusion_type != 'no_graph':
            x = self.rgcn(x_static, edge_index, edge_type)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.rgcn(x, edge_index, edge_type)  # (N, emb)
        else:
            x = x_static # 如果没有图，直接使用静态特征

        # 5) 融合策略
        if fusion_type == 'no_graph':
            # Time-only (w/o Graph): 不使用图网络，直接拼接静态特征和时序特征，过 MLP
            x_fused = torch.cat((x_static, t_proj), dim=1)
            x_fused = F.leaky_relu(self.mlp_linear(x_fused))
        elif fusion_type == 'none':
            # Graph-only: 完全不使用时序特征和门控
            x_fused = x
        elif fusion_type == 'concat':
            # 简单拼接融合: 图特征和时序特征拼接，过 MLP 降维
            x_fused = torch.cat((x, t_proj), dim=1)
            x_fused = F.leaky_relu(self.mlp_linear(x_fused))
        else:
            # 默认：门控融合 (Gated Fusion)
            gate = torch.sigmoid(self.gate_linear(torch.cat((x, t_proj), dim=1)))  # (N, emb)
            x_fused = gate * x + (1 - gate) * t_proj

        # 6) 分类
        x_fused = self.output1(x_fused)
        x_fused = self.output2(x_fused)

        return x_fused


class BotRGCN_v3(nn.Module):
    """
    v3 升级版：
    1. GATConv + 边类型嵌入替代 RGCNConv（邻居注意力机制）
    2. 多层 GNN + 残差连接 + LayerNorm（更深的图信息传播）
    3. 可学习时序投影 + 门控融合（同 v2）
    """
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=6, cat_prop_size=11,
                 time_size=64, embedding_dimension=64, dropout=0.3,
                 num_relations=2, gnn_layers=3, gat_heads=4):
        super(BotRGCN_v3, self).__init__()
        self.dropout = dropout
        emb = embedding_dimension
        self.num_gnn_layers = gnn_layers

        # --- 静态特征编码器 (同 v2) ---
        self.linear_relu_des = nn.Sequential(nn.Linear(des_size, emb // 4), nn.LeakyReLU())
        self.linear_relu_tweet = nn.Sequential(nn.Linear(tweet_size, emb // 4), nn.LeakyReLU())
        self.linear_relu_num_prop = nn.Sequential(nn.Linear(num_prop_size, emb // 4), nn.LeakyReLU())
        self.linear_relu_cat_prop = nn.Sequential(nn.Linear(cat_prop_size, emb // 4), nn.LeakyReLU())

        # --- 时序投影层 (同 v2) ---
        self.time_proj = nn.Sequential(
            nn.Linear(time_size, emb), nn.LayerNorm(emb), nn.LeakyReLU(),
            nn.Dropout(dropout), nn.Linear(emb, emb), nn.LeakyReLU())

        self.linear_relu_input = nn.Sequential(nn.Linear(emb, emb), nn.LeakyReLU())
        self.input_norm = nn.LayerNorm(emb)

        # --- 边类型嵌入 ---
        edge_emb_dim = 16
        self.edge_embedding = nn.Embedding(num_relations, edge_emb_dim)

        # --- 多层 GATConv + 残差 + LayerNorm ---
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        for i in range(gnn_layers):
            self.gat_layers.append(
                GATConv(emb, emb // gat_heads, heads=gat_heads,
                        edge_dim=edge_emb_dim, concat=True, dropout=dropout))
            self.gat_norms.append(nn.LayerNorm(emb))

        # --- 门控融合 (同 v2) ---
        self.gate_linear = nn.Linear(emb * 2, emb)

        # --- 输出层 ---
        self.output1 = nn.Sequential(nn.Linear(emb, emb), nn.LeakyReLU())
        self.output2 = nn.Linear(emb, 2)

    def forward(self, des, tweet, num_prop, cat_prop, time_feature, edge_index, edge_type):
        t_proj = self.time_proj(time_feature)

        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)

        x = torch.cat((d, t, n, c), dim=1)
        x = self.linear_relu_input(x)
        x = self.input_norm(x)

        # 边类型 → 边特征向量
        edge_attr = self.edge_embedding(edge_type)

        # 多层 GAT + 残差连接
        for i in range(self.num_gnn_layers):
            x_res = x
            x = self.gat_layers[i](x, edge_index, edge_attr=edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res  # 残差
            x = self.gat_norms[i](x)

        # 门控融合
        gate = torch.sigmoid(self.gate_linear(torch.cat((x, t_proj), dim=1)))
        x_fused = gate * x + (1 - gate) * t_proj

        x_fused = self.output1(x_fused)
        x_fused = self.output2(x_fused)
        return x_fused




# =====================================================
# 端到端模型：内置 Transformer + RGCN + 门控融合
# =====================================================

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:, :x.size(1), :]
#         return self.dropout(x)


# class BotRGCN_E2E(nn.Module):
#     """
#     端到端统一模型：内置 Transformer 时间编码器 + RGCN 图编码器 + 门控融合。
#     """
#     def __init__(self, des_size=768, tweet_size=768, num_prop_size=6, cat_prop_size=11,
#                  ts_input_dim=14, d_model=64, nhead=2, num_transformer_layers=2,
#                  embedding_dimension=64, dropout=0.3, ts_dropout=0.1,
#                  chunk_size=50000, use_checkpoint=True):
#         super().__init__()
#         self.dropout = dropout
#         self.chunk_size = chunk_size
#         self.use_checkpoint = use_checkpoint
#         self.d_model = d_model
#         emb = embedding_dimension

#         self.ts_input_proj = nn.Linear(ts_input_dim, d_model)
#         self.ts_pos_enc = PositionalEncoding(d_model, ts_dropout)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
#             dropout=ts_dropout, activation='gelu', batch_first=True)
#         self.ts_transformer = nn.TransformerEncoder(
#             encoder_layer, num_layers=num_transformer_layers)
#         self.ts_out_proj = nn.Sequential(
#             nn.LayerNorm(d_model), nn.Linear(d_model, emb), nn.LeakyReLU())

#         self.linear_relu_des = nn.Sequential(nn.Linear(des_size, emb // 4), nn.LeakyReLU())
#         self.linear_relu_tweet = nn.Sequential(nn.Linear(tweet_size, emb // 4), nn.LeakyReLU())
#         self.linear_relu_num_prop = nn.Sequential(nn.Linear(num_prop_size, emb // 4), nn.LeakyReLU())
#         self.linear_relu_cat_prop = nn.Sequential(nn.Linear(cat_prop_size, emb // 4), nn.LeakyReLU())
#         self.linear_relu_input = nn.Sequential(nn.Linear(emb, emb), nn.LeakyReLU())
#         self.input_norm = nn.LayerNorm(emb)

#         self.rgcn1 = RGCNConv(emb, emb, num_relations=2)
#         self.rgcn2 = RGCNConv(emb, emb, num_relations=2)
#         self.gate_linear = nn.Linear(emb * 2, emb)
#         self.output1 = nn.Sequential(nn.Linear(emb, emb), nn.LeakyReLU())
#         self.output2 = nn.Linear(emb, 2)
#         self._init_ts_weights()

#     def _init_ts_weights(self):
#         nn.init.xavier_uniform_(self.ts_input_proj.weight)
#         if self.ts_input_proj.bias is not None:
#             nn.init.zeros_(self.ts_input_proj.bias)
#         # 门控偏置初始化: sigmoid(+2)≈0.88 → 初始时 ~88% 依赖图嵌入
#         nn.init.constant_(self.gate_linear.bias, 2.0)

#     def _encode_ts_chunk(self, raw_ts):
#         x = self.ts_input_proj(raw_ts) * math.sqrt(self.d_model)
#         x = self.ts_pos_enc(x)
#         x = self.ts_transformer(x)
#         x = x.mean(dim=1)
#         x = self.ts_out_proj(x)
#         return x

#     def encode_temporal(self, raw_ts):
#         N = raw_ts.shape[0]
#         if N <= self.chunk_size:
#             if self.use_checkpoint and self.training:
#                 return cp.checkpoint(self._encode_ts_chunk, raw_ts, use_reentrant=False)
#             return self._encode_ts_chunk(raw_ts)
#         chunks = raw_ts.split(self.chunk_size, dim=0)
#         embeddings = []
#         for chunk in chunks:
#             if self.use_checkpoint and self.training:
#                 emb = cp.checkpoint(self._encode_ts_chunk, chunk, use_reentrant=False)
#             else:
#                 emb = self._encode_ts_chunk(chunk)
#             embeddings.append(emb)
#         return torch.cat(embeddings, dim=0)

#     def forward(self, des, tweet, num_prop, cat_prop, raw_ts, edge_index, edge_type,
#                 return_embeddings=False):
#         t_emb = self.encode_temporal(raw_ts)
#         d = self.linear_relu_des(des)
#         t = self.linear_relu_tweet(tweet)
#         n = self.linear_relu_num_prop(num_prop)
#         c = self.linear_relu_cat_prop(cat_prop)
#         x = torch.cat((d, t, n, c), dim=1)
#         x = self.linear_relu_input(x)
#         x = self.input_norm(x)
#         x = self.rgcn1(x, edge_index, edge_type)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.rgcn2(x, edge_index, edge_type)
#         graph_emb = x
#         time_emb = t_emb
#         gate = torch.sigmoid(self.gate_linear(torch.cat((x, t_emb), dim=1)))
#         x_fused = gate * x + (1 - gate) * t_emb
#         out = self.output1(x_fused)
#         logits = self.output2(out)
#         if return_embeddings:
#             return logits, graph_emb, time_emb
#         return logits


# def info_nce_loss(z1, z2, temperature=0.5, max_samples=4096):
#     N = z1.size(0)
#     if N > max_samples:
#         indices = torch.randperm(N, device=z1.device)[:max_samples]
#         z1, z2 = z1[indices], z2[indices]
#     z1 = F.normalize(z1, dim=1)
#     z2 = F.normalize(z2, dim=1)
#     sim = torch.mm(z1, z2.t()) / temperature
#     labels = torch.arange(z1.size(0), device=z1.device)
#     return F.cross_entropy(sim, labels)
