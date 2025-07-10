import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import os 
import pandas as pd
import numpy as np
import argparse
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from DataSet import TimeSeriesDataset
from MyModel.model import CNNAttentionModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="日志等级:%(levelname)s || 时间:%(asctime)s || 消息:%(message)s",
    handlers=[
        logging.FileHandler('./logs/transformer训练日志.log', mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

def arg_parser():
    parser = argparse.ArgumentParser(description='LSTM训练参数')
    # ---准备数据用到的参数：
    parser.add_argument('--traindata_path',type=str,default='./data/train.csv',help='训练集')
    parser.add_argument('--testdata_path',type=str,default='./data/test.csv',help='测试集')
    
    # ---保存数据用到的参数：
    parser.add_argument('--save_plot_path', type=str, default='./results/plot1', help='图表保存路径')



    # ---dataset用到的参数：
    parser.add_argument('--input_seqlen',type=int,default=90,help='输入的序列长度')
    parser.add_argument('--output_seqlen',type=int,default=90,help='预测的序列长度')


    # ---模型用到的参数：
    parser.add_argument('--input_size',type=int,default=10,help='输入的特征维度')
    parser.add_argument('--cnn_channels_1', type=int, default=32, help='第一个CNN层的输出通道数')
    parser.add_argument('--cnn_channels_2', type=int, default=64, help='第二个CNN层的输出通道数')
    parser.add_argument('--kernel_size', type=int, default=3, help='CNN的卷积核大小')
    parser.add_argument('--output_size', type=int, default=90, help='需要预测未来多少天')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout比率 (注:当前模型未使用,可保留)')

    # ---训练用到的参数：
    parser.add_argument('--epochs', type=int, default=150, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批处理大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--num_runs', type=int, default=5, help='进行多少次独立实验')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='使用设备')


    arg = parser.parse_args()
    return arg



def evaluate_model(model, test_loader, scaler, args):
    """
    封装的评估函数，包括预测、反归一化和计算指标

    参数:
        model: 待评估的模型
        test_loader: 测试数据的DataLoader
        scaler: 用于反归一化的StandardScaler对象
        args: 命令行参数
    返回:
        mse (float): 均方误差
        mae (float): 平均绝对误差
        preds_unscaled (np.array): 反归一化后的预测值
        targets_unscaled (np.array): 反归一化后的真实值
    """
    logging.info("开始评估模型...")
    model.eval()
    all_preds_scaled = []
    all_targets_scaled = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            all_preds_scaled.append(outputs.cpu().numpy())
            all_targets_scaled.append(targets.cpu().numpy())

    all_preds_scaled = np.concatenate(all_preds_scaled, axis=0)
    all_targets_scaled = np.concatenate(all_targets_scaled, axis=0)

    # 反归一化
    target_mean = scaler.mean_[0]
    target_scale = scaler.scale_[0]
    preds_unscaled = all_preds_scaled * target_scale + target_mean
    targets_unscaled = all_targets_scaled * target_scale + target_mean

    # 计算指标
    mse = mean_squared_error(targets_unscaled, preds_unscaled)
    mae = mean_absolute_error(targets_unscaled, preds_unscaled)
    
    return mse, mae, preds_unscaled, targets_unscaled



def plot_results(targets, predictions, save_path, args):
    """
    封装的绘图函数

    参数:
        targets (np.array): 真实值
        predictions (np.array): 预测值
        save_path (str): 图片保存路径
        args: 命令行参数，用于获取 output_seqlen
    """
    logging.info("正在生成预测与真实值对比图...")
    plt.figure(figsize=(15, 7))
    
    # 动态获取序列长度
    seq_len = args.output_seqlen
    
    # 评论和标题都使用动态变量
    # 我们只绘制测试集中第一个样本的 {seq_len} 天预测情况，以保持图表清晰
    plt.plot(targets[0, :], 'b-', label='Ground Truth (真实值)')
    plt.plot(predictions[0, :], 'r--', label='Prediction (预测值)')
    plt.title(f'Prediction vs Ground Truth for One Sample ({seq_len} Days)') # 使用f-string动态生成标题
    plt.xlabel('Time Step (Day)')
    plt.ylabel('Global Active Power')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    logging.info(f"图表已保存至: {save_path}")

def train_model(model, train_loader, criterion, optimizer, args):
    """
    封装的训练函数

    参数:
        model: 待训练的模型
        train_loader: 训练数据的DataLoader
        criterion: 损失函数
        optimizer: 优化器
        args: 命令行参数
    返回:
        model: 训练完成的模型
    """
    logging.info("开始训练模型...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # logging.info(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss/len(train_loader):.6f}")
    
    return model


def main():
    args = arg_parser()

    # ---储存评价标准:
    all_mse_scores = []
    all_mae_scores = []

    # ---用于最后绘图的数据：
    final_run_preds = None
    final_run_targets = None

    for run in range(args.num_runs):
        logging.info(f"\n{'='*20} 开始第 {run + 1}/{args.num_runs} 次实验 {'='*20}")

        # --- 1. 准备数据 ---
        df_train = pd.read_csv(args.traindata_path)
        df_test = pd.read_csv(args.testdata_path)
        
        train_dataset = TimeSeriesDataset(args, df_train, is_training=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        test_dataset = TimeSeriesDataset(args, df_test, feature_scaler=train_dataset.feature_scaler, is_training=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # --- 2. 初始化模型和优化器 ---
        model = CNNAttentionModel(args).to(args.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        # --- 3. 调用训练函数 ---
        model = train_model(model, train_loader, criterion, optimizer, args)
        
        # --- 4. 调用评估函数 ---
        mse, mae, preds, targets = evaluate_model(model, test_loader, train_dataset.feature_scaler, args)
        
        # --- 5. 记录结果 ---
        all_mse_scores.append(mse)
        all_mae_scores.append(mae)
        logging.info(f"第 {run + 1} 次实验结果 - MSE: {mse:.4f}, MAE: {mae:.4f}")

        # 保存最后一次实验的数据用于绘图
        if run == args.num_runs - 1:
            final_run_preds = preds
            final_run_targets = targets

    # --- 6. 报告最终统计结果 ---
    avg_mse = np.mean(all_mse_scores)
    std_mse = np.std(all_mse_scores)
    avg_mae = np.mean(all_mae_scores)
    std_mae = np.std(all_mae_scores)

    logging.info(f"\n{'='*20} 所有实验完成 {'='*20}")
    logging.info(f"{args.num_runs} 次实验的平均结果:")
    logging.info(f"均方误差 (MSE): {avg_mse:.4f} ± {std_mse:.4f}")
    logging.info(f"平均绝对误差 (MAE): {avg_mae:.4f} ± {std_mae:.4f}")

    # --- 7. 调用绘图函数 ---
    if final_run_preds is not None and final_run_targets is not None:
        plot_results(final_run_targets, final_run_preds, args.save_plot_path, args)

if __name__ == '__main__':
    main()