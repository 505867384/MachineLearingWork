import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    一个用于时间序列预测的LSTM模型。

    需要参数:
        input_size (int): 输入特征的数量 (例如，您数据中的列数, 10)。
        hidden_size (int): LSTM隐藏层中的单元数 (这是一个可以调整的超参数)。64
        num_layers (int): LSTM网络的层数。1
        output_size (int): 模型需要预测的未来时间点的数量 (应等于您的 output_seqlen)。90
        dropout_prob (float): 在全连接层之前应用的Dropout比率，用于防止过拟合。
    """
    
    def __init__(self,args):
        super().__init__()

        self.input_size = args.input_size 
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.output_size = args.output_size
        self.dropout = args.dropout

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.dropout_layer = nn.Dropout(self.dropout)

        self.fc = nn.Linear(in_features=self.hidden_size,out_features=self.output_size)

    def forward(self,x):
        lstm_out , (h_n,c_n) = self.lstm(x)
        pred = self.dropout_layer(lstm_out[:,-1,:])
        out = self.fc(pred)
        return out



