from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

class TimeAttentionGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,  # 修改为直接使用hidden_dim
            num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),  # 修正输入维度为双向后的实际维度
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        # 移除多余的维度合并操作
        attn_weights = self.attention(out)
        context = torch.sum(attn_weights * out, dim=1)
        return context

class FPNBlock(nn.Module):
    def __init__(self, in_channel, out_channel, scale_factor):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 3, padding=1),
            nn.BatchNorm1d(out_channel),
            nn.GELU()
        )
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=True)

    def forward(self, x, lateral):
        x = self.conv(x)
        x = self.upsample(x)
        return x + lateral


class FPN_ESGRU_K_L_B_NoDSC(nn.Module):
    def __init__(self, _, in_channel, out_channel, base_dim=32):
        super().__init__()
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
        # 修改temporal层输入维度匹配双向输出
        self.temporal = TimeAttentionGRU(
            input_dim=base_dim//2,
            hidden_dim=base_dim  # 保持base_dim，实际内部会分成两个base_dim//2
        )
        
        # 调整分类器输入维度为base_dim*2
        self.classifier = nn.Sequential(
            nn.Linear(base_dim*2, base_dim),  # 输入维度调整为双向GRU实际输出维度
            nn.GELU(),
            nn.Linear(base_dim, out_channel)
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

