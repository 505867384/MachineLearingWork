import torch
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import StandardScaler    #进行数据缩放
import pandas as pd
import numpy as np
import logging
import os 
import argparse


class TimeSeriesDataset(Dataset):
    def __init__(self,args,data,feature_scaler=None,is_training = True):
        super().__init__()
        self.data = data
        self.input_seqlen = args.input_seqlen
        self.output_seqlen = args.output_seqlen



        #读取数据

        self.feature = data

        self.feature = self.feature.astype('float32')



        if is_training:
            self.feature_scaler = StandardScaler()
            self.feature_scaled = self.feature_scaler.fit_transform(self.feature)

        else:
            if feature_scaler is None:
                logging.error("在非训练模式下, 必须提供一个 feature_scaler。")
                exit()
            self.feature_scaler = feature_scaler
            self.feature_scaled = self.feature_scaler.transform(self.feature)
            
        # print("Feature scaled:", self.feature_scaled)
        # print("Feature scaled's shape:", self.feature_scaled.shape)
        # print("第一行:",self.feature_scaled[0])
        # print("第一行第三列:",self.feature_scaled[0][2])
        # print("Feature scaler:", self.feature_scaler.mean_, self.feature_scaler.var_)

        self.x,self.y =self._create_sequence(self.feature_scaled,self.input_seqlen,self.output_seqlen)


    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index],self.y[index]

    def _create_sequence(self,feature,input_seqlen,output_seqlen):
        """
        实现滑动窗口
        
        参数:
            feature:包含标签和特征,作为前90天的输入
            target:只包含标签
            input_seqlen:输入序列长度
            output_seqlen:输出序列长度
        返回值:
            x:list y:list
            x是包含长度为90
            
        """
        x , y = [] , [] 
        for i in range(len(feature)-self.input_seqlen-self.output_seqlen+1):
            input_seq = feature[i:i+input_seqlen]
            output_seq = feature[i+input_seqlen:i+input_seqlen+output_seqlen,0]
            x.append(input_seq)
            y.append(output_seq)

        return np.array(x),np.array(y)
    



# ---测试一下看看对不对---
def test():
    file_path='./data/train.csv'
    data = pd.read_csv(file_path)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_seqlen',type= int ,default=90)
    parser.add_argument('--output_seqlen',type=int ,default=90)
    args = parser.parse_args()

    dataset=TimeSeriesDataset(args,data)
    

if __name__ == "__main__":

    test()




        


