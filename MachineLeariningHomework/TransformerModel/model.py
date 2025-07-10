# model.py

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """为输入序列添加位置信息"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2) # 形状变为 (1, max_len, d_model) 以适配batch_first
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
        """
        # x的第1维是序列长度，pe的第1维也是
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """
    一个用于时间序列预测的Transformer模型。
    它使用了一个Encoder-Only的结构。
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Transformer模型的核心维度，相当于LSTM的hidden_size
        self.d_model = args.hidden_size
        
        # 1. 输入投影层
        # 原始特征维度可能与模型的d_model不同，需要一个线性层进行投影
        self.input_projection = nn.Linear(args.input_size, self.d_model)
        
        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(self.d_model, args.dropout)
        
        # 3. Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=args.nhead, 
            dim_feedforward=args.dim_feedforward, 
            dropout=args.dropout,
            batch_first=True  # 非常重要！确保输入形状是 (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers, 
            num_layers=args.num_encoder_layers
        )
        
        # 4. 输出层
        # 将编码器输出的表征映射到期望的输出序列长度
        self.output_layer = nn.Linear(self.d_model, args.output_size)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        参数:
            src: 源序列, 形状为 (batch_size, input_seq_len, input_size)
        """
        # 1. 输入投影: (batch, seq, feature) -> (batch, seq, d_model)
        src = self.input_projection(src)
        
        # 2. 添加位置编码
        src = self.pos_encoder(src)
        
        # 3. 通过Transformer编码器
        # 输出形状同样为 (batch, seq, d_model)
        output = self.transformer_encoder(src)
        
        # 4. 解码/池化
        # 我们取序列的第一个时间步的输出作为整个序列的聚合表示
        # 这类似于BERT中的[CLS] token的用法
        output = output[:, 0, :] # 形状变为 (batch, d_model)
        
        # 5. 通过输出层得到最终预测
        # 形状变为 (batch, output_size)
        output = self.output_layer(output)
        
        return output