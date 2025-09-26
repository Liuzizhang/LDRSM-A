from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

class TimeAttentionGRU(nn.Module):  # 修改类名
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(  # GRU
            input_dim,
            hidden_dim // 2,
            num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        out, _ = self.gru(x)  # 修改为GRU前向传播
        attn_weights = self.attention(out)  # [B, T, 1]
        context = torch.sum(attn_weights * out, dim=1)  # [B, D]
        return context

class FPNBlock(nn.Module):
    def __init__(self, in_channel, out_channel, scale_factor):
        super().__init__()
        self.conv = nn.Sequential(
            # 深度可分离卷积 = 深度卷积 + 逐点卷积
            nn.Conv1d(in_channel, in_channel, 3, padding=1, groups=in_channel),  # 深度卷积
            nn.Conv1d(in_channel, out_channel, 1),  # 逐点卷积
            nn.BatchNorm1d(out_channel),
            nn.GELU()
        )
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=True)
        
    # forward保持不变
    def forward(self, x, lateral):
        x = self.conv(x)
        x = self.upsample(x)
        return x + lateral

class FPN_ESGRU_K_B(nn.Module):
    def __init__(self, _, in_channel, out_channel, base_dim=32):
        super().__init__()
        # 编码器结构
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channel, base_dim, 15, stride=2, padding=7),
            nn.BatchNorm1d(base_dim),
            nn.GELU(),
            
            nn.Conv1d(base_dim, base_dim*2, 7, stride=2, padding=3),
            nn.BatchNorm1d(base_dim*2),
            nn.GELU(),
            
            nn.Conv1d(base_dim*2, base_dim*4, 3, stride=2, padding=1),
            nn.BatchNorm1d(base_dim*4),
            nn.GELU()
        )
        
        # FPN结构
        self.fpn3 = FPNBlock(base_dim*4, base_dim*2, scale_factor=2)
        self.fpn2 = FPNBlock(base_dim*2, base_dim, scale_factor=2)
        self.fpn1 = FPNBlock(base_dim, base_dim//2, scale_factor=2)
        
        # 时间注意力GRU
        self.temporal = TimeAttentionGRU(  # 使用修改后的类名
            input_dim=base_dim//2,
            hidden_dim=base_dim*2,
            num_layers=2
        )
        
        # 修正分类器输入维度
        self.classifier = nn.Sequential(
            nn.Linear(base_dim*2, base_dim*4),  # 输入维度应为base_dim*2 (双向FRU输出维度)
            nn.GELU(),
            nn.Linear(base_dim*4, out_channel)
        )

    def forward(self, x: Tensor) -> Tensor:
        # 编码器
        c1 = self.encoder[0:3](x)       # [B, 32, L/2]
        c2 = self.encoder[3:6](c1)      # [B, 64, L/4]
        c3 = self.encoder[6:9](c2)      # [B, 128, L/8]
        
        # FPN特征融合
        p3 = self.fpn3(c3, c2)          # [B, 64, L/4]
        p2 = self.fpn2(p3, c1)          # [B, 32, L/2]
        p1 = self.fpn1(p2, x)           # [B, 16, L]
        
        # 时间维度处理
        p1 = p1.transpose(1, 2)         # [B, L, 16]
        temporal_feat = self.temporal(p1)
        
        # 分类
        return self.classifier(temporal_feat)