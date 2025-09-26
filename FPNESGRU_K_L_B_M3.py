import torch
import torch.nn as nn
from .FPNESGRU_K_L_B_M2 import FPN_ESGRU_K_L_B_M2

# 确保类名与参数完全一致
class FPN_ESGRU_K_L_B_M3(nn.Module):
    def __init__(self, args, in_channel, out_channel):
        super().__init__()
        self.base_model = FPN_ESGRU_K_L_B_M2(
            _=None,  # 添加占位符参数
            in_channel=in_channel,
            out_channel=out_channel
        )
        # 修改分类器结构
        self.base_model.classifier = nn.Sequential(
            nn.Linear(128, 128),  # 修改输入维度从256->128
            nn.ReLU(),
            nn.Linear(128, out_channel)
        )

    def forward(self, x):
        return self.base_model(x)