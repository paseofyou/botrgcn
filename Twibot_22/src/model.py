import math
import torch
from torch import nn
from torch_geometric.nn import RGCNConv, GATConv
import torch.nn.functional as F
import torch.utils.checkpoint as cp

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = math.sqrt(dim)
        self.gamma = nn.Parameter(torch.zeros(1)) # 可学习的残差权重

    def forward(self, x_graph, x_temporal):
        # x_graph: [N, dim] -> Query
        # x_temporal: [N, dim] -> Key, Value
        
        # 为了使用 attention，我们需要增加一个 sequence 维度
        # 这里我们把 temporal 特征看作长度为 1 的序列
        Q = self.query(x_graph).unsqueeze(1)     # [N, 1, dim]
        K = self.key(x_temporal).unsqueeze(1)    # [N, 1, dim]
        V = self.value(x_temporal).unsqueeze(1)  # [N, 1, dim]

        # Attention scores: [N, 1, 1]
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Attended temporal features: [N, 1, dim] -> [N, dim]
        attended_temporal = torch.bmm(attn_probs, V).squeeze(1)

        # 残差连接 + 门控融合
        # 使用可学习的 gamma 控制时序特征的注入量
        out = x_graph + self.gamma * attended_temporal
        return out

class BotRGCN(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3, time_size=64,
                 embedding_dimension=64, dropout=0.3):
        super(BotRGCN, self).__init__()
        self.dropout = dropout
        self.linear_relu_des = nn.Sequential(nn.Linear(des_size, int(embedding_dimension / 4)), nn.GELU())
        self.linear_relu_tweet = nn.Sequential(nn.Linear(tweet_size, int(embedding_dimension / 4)), nn.GELU())
        self.linear_relu_num_prop = nn.Sequential(nn.Linear(num_prop_size, int(embedding_dimension / 4)), nn.GELU())
        self.linear_relu_cat_prop = nn.Sequential(nn.Linear(cat_prop_size, int(embedding_dimension / 4)), nn.GELU())

        self.linear_relu_input = nn.Sequential(nn.Linear(embedding_dimension, embedding_dimension), nn.GELU())
        self.input_norm = nn.LayerNorm(embedding_dimension)
        self.time_norm = nn.LayerNorm(time_size)

        self.rgcn = RGCNConv(embedding_dimension, embedding_dimension, num_relations=2)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension + time_size, embedding_dimension), nn.GELU())
        self.linear_output2 = nn.Linear(embedding_dimension, 2)

    def forward(self, des, tweet, num_prop, cat_prop, time_feature, edge_index, edge_type):
        time_feature = self.time_norm(time_feature)
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)

        x = torch.cat((d, t, n, c), dim=1)
        x = self.linear_relu_input(x)
        x = self.input_norm(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)

        x = torch.cat((x, time_feature), dim=1)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return x


class BotRGCN_v2(nn.Module):
    """
    改进版 BotRGCN：
    1. 可学习的时序投影层
    2. 两个独立 RGCN 层
    3. 门控融合
    """
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3,
                 time_size=64, embedding_dimension=64, dropout=0.3):
        super(BotRGCN_v2, self).__init__()
        self.dropout = dropout
        emb = embedding_dimension

        self.linear_relu_des = nn.Sequential(nn.Linear(des_size, emb // 4), nn.GELU())
        self.linear_relu_tweet = nn.Sequential(nn.Linear(tweet_size, emb // 4), nn.GELU())
        self.linear_relu_num_prop = nn.Sequential(nn.Linear(num_prop_size, emb // 4), nn.GELU())
        self.linear_relu_cat_prop = nn.Sequential(nn.Linear(cat_prop_size, emb // 4), nn.GELU())

        self.time_proj = nn.Sequential(
            nn.Linear(time_size, emb), nn.LayerNorm(emb), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(emb, emb), nn.GELU())

        self.linear_relu_input = nn.Sequential(nn.Linear(emb, emb), nn.GELU())
        self.input_norm = nn.LayerNorm(emb)

        self.rgcn1 = RGCNConv(emb, emb, num_relations=2)
        self.rgcn2 = RGCNConv(emb, emb, num_relations=2)

        self.gate_linear = nn.Linear(emb * 2, emb)

        self.output1 = nn.Sequential(nn.Linear(emb, emb), nn.GELU())
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

        x = self.rgcn1(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn2(x, edge_index, edge_type)

        gate = torch.sigmoid(self.gate_linear(torch.cat((x, t_proj), dim=1)))
        x_fused = gate * x + (1 - gate) * t_proj

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
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3,
                 time_size=64, embedding_dimension=64, dropout=0.3,
                 num_relations=2, gnn_layers=3, gat_heads=4):
        super(BotRGCN_v3, self).__init__()
        self.dropout = dropout
        emb = embedding_dimension
        self.num_gnn_layers = gnn_layers

        # --- 静态特征编码器 (同 v2) ---
        self.linear_relu_des = nn.Sequential(nn.Linear(des_size, emb // 4), nn.GELU())
        self.linear_relu_tweet = nn.Sequential(nn.Linear(tweet_size, emb // 4), nn.GELU())
        self.linear_relu_num_prop = nn.Sequential(nn.Linear(num_prop_size, emb // 4), nn.GELU())
        self.linear_relu_cat_prop = nn.Sequential(nn.Linear(cat_prop_size, emb // 4), nn.GELU())

        # --- 时序投影层 (同 v2) ---
        self.time_proj = nn.Sequential(
            nn.Linear(time_size, emb), nn.LayerNorm(emb), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(emb, emb), nn.GELU())

        self.linear_relu_input = nn.Sequential(nn.Linear(emb, emb), nn.GELU())
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
        self.output1 = nn.Sequential(nn.Linear(emb, emb), nn.GELU())
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
            x = F.gelu(x)
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
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
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class BotRGCN_E2E(nn.Module):
    """
    端到端统一模型：内置 Transformer 时间编码器 + RGCN 图编码器 + 门控融合。
    Transformer 和 GNN 联合训练，分类损失直接优化时间表示。
    使用分块处理 + 梯度检查点以支持百万级节点。
    """
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3,
                 ts_input_dim=14, d_model=64, nhead=2, num_transformer_layers=2,
                 embedding_dimension=64, dropout=0.3, ts_dropout=0.1,
                 chunk_size=50000, use_checkpoint=True):
        super().__init__()
        self.dropout = dropout
        self.chunk_size = chunk_size
        self.use_checkpoint = use_checkpoint
        self.d_model = d_model
        emb = embedding_dimension

        # --- Transformer 时间编码器 ---
        self.ts_input_proj = nn.Linear(ts_input_dim, d_model)
        self.ts_pos_enc = PositionalEncoding(d_model, ts_dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=ts_dropout, activation='gelu', batch_first=True)
        self.ts_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers)

        # --- 时序嵌入投影（对齐到 emb 维度）---
        self.ts_out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, emb),
            nn.GELU()
        )

        # --- 静态特征编码器 (同 v2) ---
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, emb // 4), nn.GELU())
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, emb // 4), nn.GELU())
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, emb // 4), nn.GELU())
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, emb // 4), nn.GELU())

        self.linear_relu_input = nn.Sequential(
            nn.Linear(emb, emb), nn.GELU())
        self.input_norm = nn.LayerNorm(emb)

        # --- 两个独立 RGCN 层 ---
        self.rgcn1 = RGCNConv(emb, emb, num_relations=2)
        self.rgcn2 = RGCNConv(emb, emb, num_relations=2)

        # --- 交叉注意力融合层 ---
        self.cross_attention = CrossAttentionFusion(emb)

        # --- 输出层 ---
        self.output1 = nn.Sequential(nn.Linear(emb, emb), nn.GELU())
        self.output2 = nn.Linear(emb, 2)

        self._init_ts_weights()

    def _init_ts_weights(self):
        nn.init.xavier_uniform_(self.ts_input_proj.weight)
        if self.ts_input_proj.bias is not None:
            nn.init.zeros_(self.ts_input_proj.bias)
        # 门控偏置初始化: sigmoid(+2)≈0.88 → 初始时 ~88% 依赖图嵌入
        # 这避免了随机 Transformer 输出在训练初期污染 GNN
        # nn.init.constant_(self.gate_linear.bias, 2.0)

    def _encode_ts_chunk(self, raw_ts):
        """编码单个块的时间序列: (B, T, F) → (B, emb)"""
        x = self.ts_input_proj(raw_ts) * math.sqrt(self.d_model)  # (B, T, d_model)
        x = self.ts_pos_enc(x)
        x = self.ts_transformer(x)
        x = x.mean(dim=1)  # mean pooling → (B, d_model)
        x = self.ts_out_proj(x)  # → (B, emb)
        return x

    def encode_temporal(self, raw_ts):
        """分块编码所有时间序列，支持梯度检查点"""
        N = raw_ts.shape[0]
        if N <= self.chunk_size:
            if self.use_checkpoint and self.training:
                return cp.checkpoint(self._encode_ts_chunk, raw_ts,
                                     use_reentrant=False)
            return self._encode_ts_chunk(raw_ts)

        chunks = raw_ts.split(self.chunk_size, dim=0)
        embeddings = []
        for chunk in chunks:
            if self.use_checkpoint and self.training:
                emb = cp.checkpoint(self._encode_ts_chunk, chunk,
                                    use_reentrant=False)
            else:
                emb = self._encode_ts_chunk(chunk)
            embeddings.append(emb)
        return torch.cat(embeddings, dim=0)

    def forward(self, des, tweet, num_prop, cat_prop, raw_ts, edge_index, edge_type,
                return_embeddings=False):
        """
        Args:
            raw_ts: (N, T, F) 原始伪时间序列矩阵
            return_embeddings: 若为 True，额外返回 graph_emb 和 time_emb 用于对比学习
        """
        # 1) Temporal encoding (端到端，有梯度)
        t_emb = self.encode_temporal(raw_ts)  # (N, emb)

        # 2) 静态特征编码
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)

        # 3) Early fusion → RGCN
        x = torch.cat((d, t, n, c), dim=1)
        x = self.linear_relu_input(x)
        x = self.input_norm(x)
        x = self.rgcn1(x, edge_index, edge_type)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn2(x, edge_index, edge_type)  # (N, emb)
        x = F.gelu(x)

        # 保存分支嵌入（用于对比学习）
        graph_emb = x
        time_emb = t_emb

        # 4) 交叉注意力融合
        x_fused = self.cross_attention(x, t_emb)

        # 5) 分类
        out = self.output1(x_fused)
        logits = self.output2(out)

        if return_embeddings:
            return logits, graph_emb, time_emb
        return logits


def info_nce_loss(z1, z2, temperature=0.5, max_samples=4096):
    """
    跨分支对齐损失 (InfoNCE)。
    z1: 图分支嵌入 (N, D)
    z2: 时间分支嵌入 (N, D)
    同节点对为正样本，不同节点对为负样本。
    max_samples: 子采样大小，避免超大相似度矩阵。
    """
    N = z1.size(0)
    if N > max_samples:
        indices = torch.randperm(N, device=z1.device)[:max_samples]
        z1, z2 = z1[indices], z2[indices]

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sim = torch.mm(z1, z2.t()) / temperature  # (M, M)
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(sim, labels)
