
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class PositionalEncoding(nn.Module):
    """
    位置编码层，用于为节点特征添加位置信息
    """
    def __init__(self, d_model: int, max_len: int = 100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class GraphTransformerLayer(nn.Module):
    """
    基于自注意力机制的真 图变换器层 (Graph Transformer)
    使用 Multi-Head Self-Attention 进行节点间的全局信息交互
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super(GraphTransformerLayer, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, num_nodes, hidden_dim)
        adj: (batch, num_nodes, num_nodes) 邻接矩阵，用于 mask 非邻接节点
        """
        batch_size, num_nodes, _ = x.size()
        
        residual = x
        
        q = self.q_linear(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
        adj_mask = (adj == 0).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        scores = scores.masked_fill(adj_mask, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        h = torch.matmul(attention, v)
        h = h.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.hidden_dim)
        
        h = self.out_linear(h)
        h = self.dropout(h)
        
        h = self.layer_norm(h + residual)
        
        return h


class GTDamagePredictor(nn.Module):
    """
    Graph Transformer 损伤预测模型
    
    1. 使用 1D 卷积提取节点时程特征
    2. 使用多层图变换器 (Multi-Head Self-Attention) 进行节点间全局信息交互
    3. 基于边连接关系预测单元损伤
    """
    def __init__(
        self,
        num_nodes: int,
        num_elements: int,
        seq_len: int,
        node_in_dim: int = 2,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(GTDamagePredictor, self).__init__()
        self.num_nodes = num_nodes
        self.num_elements = num_elements
        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        
        self.node_encoder = nn.Sequential(
            nn.Conv1d(node_in_dim, hidden_dim // 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=num_nodes)
        
        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.damage_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        x_nodes = x.view(batch_size, -1, self.num_nodes, self.node_in_dim).transpose(1, 2)
        
        x_flat = x_nodes.reshape(batch_size * self.num_nodes, -1, self.node_in_dim).transpose(1, 2)
        
        h_node = self.node_encoder(x_flat).squeeze(-1)
        
        h_node = h_node.view(batch_size, self.num_nodes, -1)
        
        h_node = self.pos_encoder(h_node)
        
        for layer in self.transformer_layers:
            h_node = layer(h_node, adj)
        
        node1_idx = edge_index[:, 0]
        node2_idx = edge_index[:, 1]
        
        h_node1 = h_node[:, node1_idx, :]
        h_node2 = h_node[:, node2_idx, :]
        
        h_edge = torch.cat([h_node1, h_node2], dim=-1)
        
        damage = self.damage_fc(h_edge).squeeze(-1)
        
        return damage
