
import torch
import torch.nn as nn
from typing import Optional


class LSTMDamagePredictor(nn.Module):
    """
    方案二: 基于LSTM的损伤识别模型
    
    使用位移响应时程预测结构损伤程度。
    
    Architecture:
        Input (T, features) -> BiLSTM -> LSTM -> Dense -> Output
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 11,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Args:
            input_dim: 输入特征维度 (传感器数量 × 2)
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            output_dim: 输出维度 (单元数量)
            dropout: Dropout概率
            bidirectional: 是否使用双向LSTM
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, timesteps, features)
        
        Returns:
            damage: (batch, num_elements) 损伤程度 [0, 1]
        """
        lstm_out, _ = self.lstm(x)
        
        attn_weights = self.attention(lstm_out)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        damage = self.fc(context)
        
        return damage


class CNNDamagePredictor(nn.Module):
    """
    基于CNN的损伤识别模型 (备选方案)
    
    使用1D卷积处理时程数据。
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 11,
        hidden_channels: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * 4, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.fc(x)
        return x
