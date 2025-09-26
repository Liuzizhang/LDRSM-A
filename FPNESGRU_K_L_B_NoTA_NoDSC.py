from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

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


class FPN_ESGRU_K_L_B_NoTA_NoDSC(nn.Module):
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
        self.fpn3 = FPNBlock(base_dim*4, base_dim*2, 2)
        self.fpn2 = FPNBlock(base_dim*2, base_dim, 2)
        self.fpn1 = FPNBlock(base_dim, base_dim//2, 2)
        
        # 替换时间注意力为双向GRU
        self.gru = nn.GRU(
            input_size=base_dim//2,
            hidden_size=base_dim//2,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(base_dim, out_channel)  # 双向GRU输出维度为base_dim
        )

    def forward(self, x: Tensor) -> Tensor:
        c1 = self.encoder[0:3](x)
        c2 = self.encoder[3:6](c1)
        c3 = self.encoder[6:9](c2)
        
        p3 = self.fpn3(c3, c2)
        p2 = self.fpn2(p3, c1)
        p1 = self.fpn1(p2, x)
        
        # GRU处理
        out, _ = self.gru(p1.transpose(1, 2))
        return self.classifier(out[:, -1, :])

