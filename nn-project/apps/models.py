import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, device=None, dtype="float32"):
        self.conv2d = nn.Conv(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            device=device, 
            dtype=dtype,
        )
        self.bn = nn.BatchNorm2d(
            dim=out_channels,
            device=device, 
            dtype=dtype,
        )
        self.relu = nn.ReLU()

    def forward(self, x: ndl.Tensor):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResNet9(nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        # TODO
        ### BEGIN YOUR SOLUTION ###
        # 第一层卷积
        self.conv1 = ConvBatchNorm(3, 16, 7, 4, device=device, dtype=dtype)
        
        # 第二层卷积
        self.conv2 = ConvBatchNorm(16, 32, 3, 2, device=device, dtype=dtype)
        
        # 残差块1
        self.res1 = nn.Sequential(
            ConvBatchNorm(32, 32, 3, 1, device=device, dtype=dtype),
            ConvBatchNorm(32, 32, 3, 1, device=device, dtype=dtype)
        )
        
        # 第三层卷积
        self.conv3 = ConvBatchNorm(32, 64, 3, 2, device=device, dtype=dtype)
        
        # 第四层卷积
        self.conv4 = ConvBatchNorm(64, 128, 3, 2, device=device, dtype=dtype)
        
        # 残差块2
        self.res2 = nn.Sequential(
            ConvBatchNorm(128, 128, 3, 1, device=device, dtype=dtype),
            ConvBatchNorm(128, 128, 3, 1, device=device, dtype=dtype)
        )
        
        # 全局平均池化和分类器
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128, 10, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x):
        # TODO
        ### BEGIN YOUR SOLUTION
        # 第一层卷积
        x = self.conv1(x)
        
        # 第二层卷积
        x = self.conv2(x)
        
        # 残差块1
        residual = x
        x = self.res1(x)
        x = x + residual  # 残差连接
        
        # 第三层卷积
        x = self.conv3(x)
        
        # 第四层卷积
        x = self.conv4(x)
        
        # 残差块2
        residual = x
        x = self.res2(x)
        x = x + residual  # 残差连接
        
        # 全局平均池化
        x = x.mean(axes=(2, 3))  # 对H和W维度取平均
        
        # 展平和分类
        x = self.flatten(x)
        x = self.linear(x)
        
        return x
        ### END YOUR SOLUTION

# 在 apps/models.py 或创建新文件
class ResidualMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, num_classes=10, num_layers=4, device=None, dtype="float32"):
        super().__init__()
        
        # 输入投影层
        self.input_proj = nn.Linear(input_dim, hidden_dim, device=device, dtype=dtype)
        
        # 残差MLP块
        self.res_blocks = nn.Sequential(*[
            ResMLPBlock(hidden_dim, device=device, dtype=dtype) 
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output = nn.Linear(hidden_dim, num_classes, device=device, dtype=dtype)
    
    def forward(self, x):
        # 展平图像: (batch, 28, 28) -> (batch, 784)
        x = x.reshape((x.shape[0], -1))
        
        # 输入投影
        x = self.input_proj(x)
        
        # 残差块
        x = self.res_blocks(x)
        
        # 分类
        return self.output(x)

class ResMLPBlock(nn.Module):
    def __init__(self, hidden_dim, device=None, dtype="float32"):
        super().__init__()
        self.ln = nn.LayerNorm1d(hidden_dim, device=device, dtype=dtype)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim, device=device, dtype=dtype)
        )
    
    def forward(self, x):
        return x + self.mlp(self.ln(x))  # 残差连接