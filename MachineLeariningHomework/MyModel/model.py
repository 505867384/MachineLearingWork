# improved_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNAttentionModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        # 使用args来定义CNN层的通道数和核大小
        cnn_channels_1 = args.cnn_channels_1
        cnn_channels_2 = args.cnn_channels_2
        kernel_size = args.kernel_size

        # 1. CNN层作为特征提取器
        self.conv1 = nn.Conv1d(
            in_channels=args.input_size, 
            out_channels=cnn_channels_1, 
            kernel_size=kernel_size, 
            padding='same' # 使用 same padding 保持序列长度不变
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(
            in_channels=cnn_channels_1, 
            out_channels=cnn_channels_2, 
            kernel_size=kernel_size, 
            padding='same'
        )
        self.relu2 = nn.ReLU()
        
        # 2. 注意力权重计算层
        self.attention_layer = nn.Linear(cnn_channels_2, 1)
        
        # 3. 最终的预测层
        self.fc = nn.Linear(cnn_channels_2, args.output_size)

    def forward(self, x):
        # x 初始形状: (batch_size, seq_len, input_size)
        x = x.transpose(1, 2) # -> (batch, input_size, seq_len)
        
        # 通过CNN层
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x)) # x 形状: (batch, cnn_channels_2, seq_len)
        
        x = x.transpose(1, 2) # -> (batch, seq_len, cnn_channels_2)
        
        # 计算注意力
        attention_scores = self.attention_layer(x)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # 加权求和
        context_vector = torch.bmm(x.transpose(1, 2), attention_weights).squeeze(2)
        
        # 最终预测
        out = self.fc(context_vector)
        return out