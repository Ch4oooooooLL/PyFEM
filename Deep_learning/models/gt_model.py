
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class GraphAttentionLayer(nn.Module):
    """
    原生 PyTorch 实现的简易图注意力层 (GAT)
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2, alpha: float = 0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        h: Node features (batch, num_nodes, in_features)
        adj: Adjacency matrix (batch, num_nodes, num_nodes)
        """
        Wh = torch.matmul(h, self.W) # (batch, num_nodes, out_features)
        
        # 构建注意力系数
        batch_size, num_nodes, _ = Wh.size()
        
        # 广播机制构建所有特征对组合
        # Wh1: (batch, num_nodes, 1, out_features)
        # Wh2: (batch, 1, num_nodes, out_features)
        Wh1 = Wh.view(batch_size, num_nodes, 1, self.out_features)
        Wh2 = Wh.view(batch_size, 1, num_nodes, self.out_features)
        
        # a_input: (batch, num_nodes, num_nodes, 2 * out_features)
        Wh1 = Wh1.expand(-1, -1, num_nodes, -1)
        Wh2 = Wh2.expand(-1, num_nodes, -1, -1)
        a_input = torch.cat([Wh1, Wh2], dim=-1)
        
        # e: (batch, num_nodes, num_nodes)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # h_prime: (batch, num_nodes, out_features)
        h_prime = torch.matmul(attention, Wh)
        
        return h_prime


class GTDamagePredictor(nn.Module):
    """
    Graph Transformer 损伤预测模型
    
    1. 使用 1D 卷积提取节点时程特征
    2. 使用图注意力层进行节点间信息交换
    3. 基于边连接关系预测单元损伤
    """
    def __init__(
        self,
        num_nodes: int,
        num_elements: int,
        seq_len: int,
        node_in_dim: int = 2, # (ux, uy)
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super(GTDamagePredictor, self).__init__()
        self.num_nodes = num_nodes
        self.num_elements = num_elements
        
        # 1. 节点时程特征提取器 (Temporal Encoder)
        self.node_encoder = nn.Sequential(
            nn.Conv1d(node_in_dim, hidden_dim // 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # [batch * num_nodes, hidden_dim, 1]
        )
        
        # 2. 图变换器层 (Spatial Interaction)
        self.gat1 = GraphAttentionLayer(hidden_dim, hidden_dim, dropout=dropout)
        self.gat2 = GraphAttentionLayer(hidden_dim, hidden_dim, dropout=dropout)
        
        # 3. 损伤预测器 (Element-wise MLP)
        # 每个单元的损伤通过其连接的两个节点的特征拼接预测
        self.damage_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, num_nodes * node_in_dim) 如 (batch, 201, 14)
        adj: (num_nodes, num_nodes) 邻接矩阵
        edge_index: (num_elements, 2) 单元连接关系 (固定)
        """
        batch_size = x.size(0)
        
        # 1. 整理输入至 (batch, num_nodes, timesteps, 2)
        # 先将 14 拆分为 (7, 2)
        x_nodes = x.view(batch_size, -1, self.num_nodes, 2).transpose(1, 2) 
        # (batch, num_nodes, timesteps, 2)
        
        # 2. 时间特征提取 (Temporal Encoder)
        # 将 batch 和 node 维度合并处理
        x_flat = x_nodes.reshape(batch_size * self.num_nodes, -1, 2).transpose(1, 2)
        # (batch * num_nodes, 2, timesteps)
        
        h_node = self.node_encoder(x_flat).squeeze(-1)
        # (batch * num_nodes, hidden_dim)
        
        h_node = h_node.view(batch_size, self.num_nodes, -1)
        # (batch, num_nodes, hidden_dim)
        
        # 3. 图注意力层 (Spatial Interaction)
        # 如果 adj 是单一矩阵，扩展至 batch
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
            
        h_node = self.gat1(h_node, adj)
        h_node = F.relu(h_node)
        h_node = self.gat2(h_node, adj)
        
        # 4. 提取边特征进行损伤预测
        node1_idx = edge_index[:, 0]
        node2_idx = edge_index[:, 1]
        
        h_node1 = h_node[:, node1_idx, :]
        h_node2 = h_node[:, node2_idx, :]
        
        h_edge = torch.cat([h_node1, h_node2], dim=-1)
        # (batch, num_elements, hidden_dim * 2)
        
        damage = self.damage_fc(h_edge).squeeze(-1)
        # (batch, num_elements)
        
        return damage
