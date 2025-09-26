from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

class TimeAttentionGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        # 使用LightConvGRU替代原始GRU
        self.light_gru = LightConvGRU(input_dim, hidden_dim)
        
        # 保持原有时间注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # 输入维度转换 [B, T, D] -> [B, D, T]
        x = x.permute(0, 2, 1)
        
        # LightConvGRU前向传播
        gru_output = self.light_gru(x)  # 输出形状 [B, hidden_dim, T]
        
        # 调整维度顺序 [B, hidden_dim, T] -> [B, T, hidden_dim]
        gru_output = gru_output.permute(0, 2, 1)
        
        # 注意力权重计算
        attn_weights = self.attention(gru_output)  # [B, T, 1]
        
        # 上下文向量计算
        context = torch.sum(attn_weights * gru_output, dim=1)  # [B, hidden_dim]
        return context

class LightConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # 深度可分离卷积门控单元
        self.conv_z = nn.Sequential(
            nn.Conv1d(input_dim+hidden_dim, hidden_dim, 5, padding=2, groups=4),
            nn.Sigmoid()
        )
        self.conv_r = nn.Sequential(
            nn.Conv1d(input_dim+hidden_dim, hidden_dim, 5, padding=2, groups=4),
            nn.Sigmoid()
        )
        self.conv_h = nn.Sequential(
            nn.Conv1d(input_dim+hidden_dim, hidden_dim, 5, padding=2, groups=4),
            nn.Tanh()
        )
        self.hidden_dim = hidden_dim

    def forward(self, x):
        batch_size, channels, seq_len = x.shape
        h_list = []
        ht = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        
        for t in range(seq_len):
            xt = x[:, :, t]
            combined = torch.cat([xt, ht], dim=1).unsqueeze(-1)
            
            z = self.conv_z(combined).squeeze(-1)
            r = self.conv_r(combined).squeeze(-1)
            
            combined_update = torch.cat([xt, r * ht], dim=1).unsqueeze(-1)
            h_hat = self.conv_h(combined_update).squeeze(-1)
            
            ht = (1 - z) * ht + z * h_hat
            h_list.append(ht.unsqueeze(-1))
        
        return torch.cat(h_list, dim=-1)

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

class FPN_ESGRU_K_L_U(nn.Module):
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
            hidden_dim=base_dim  # 原为base_dim*2
        )
        
        # 分类器对应调整
        self.classifier = nn.Sequential(
            nn.Linear(base_dim, base_dim//2),  # 输入维度调整
            nn.GELU(),
            nn.Linear(base_dim//2, out_channel)
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