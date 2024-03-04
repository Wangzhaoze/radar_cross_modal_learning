import numpy as np
from func import *
import os
import glob
import torch.nn as nn
import torch

data_path = './dataset'

pcl2traj_label_path = glob.glob(os.path.join(data_path, 'traj_eval/*.npy'))
pcl2traj_pred_path = glob.glob(os.path.join(data_path, 'ogm2traj_pred/*.npy'))

n = min(len(pcl2traj_label_path), len(pcl2traj_pred_path))
made = 0
mfde = 0
label_arr = np.zeros((1000, 300))
pred_arr = np.zeros((1000, 300))

res1_25 = 0
res1_50 = 0
res1_75 = 0


softmax = nn.Softmax(dim=1)


import numpy as np

def weighted_average_x(pred_matrix):
    """
    对每行进行softmax归一化,并使用横坐标的加权平均值进行估计。

    Parameters:
    - pred_matrix: 输入的预测矩阵，每行包含预测的概率分布。

    Returns:
    - estimated_x: 每行的加权平均横坐标估计。
    """
    # 对每行进行softmax操作
    softmax_probs = np.exp(pred_matrix) / np.sum(np.exp(pred_matrix), axis=1, keepdims=True)

    # 计算加权平均横坐标估计
    estimated_x = np.sum(softmax_probs * np.arange(pred_matrix.shape[1]), axis=1)

    return estimated_x

for i in range(n):

    try: 

        label = np.load(pcl2traj_label_path[i])
        label = label[:, 300:700]
        label = torch.from_numpy(label)
        label = label.to(device='cuda', dtype=torch.float32)
        label = softmax(label)
        label = label.to(device='cpu', dtype=torch.float32)

        pred = np.load(pcl2traj_label_path[i].replace('traj_eval', 'ogm2traj_pred').replace('.npy', '_ogm2traj.npy'))

        pred_traj = np.zeros((1000, 2))
        label_traj = np.zeros((1000, 2))

        pred_traj[:, 0] = np.arange(0, 1000)
        pred_traj[:, 1] = weighted_average_x(pred)

        label_traj[:, 0] = np.arange(0, 1000)
        label_traj[:, 1] = np.argmax(label, axis=1)



        delta = pred_traj[:, 1] - label_traj[:, 1] 
        res1_25 += delta[250]**2 / n
        res1_50 += delta[500]**2 / n
        res1_75 += delta[750]**2 / n

    except: continue


res1 = (np.sqrt(res1_25) + np.sqrt(res1_50) + np.sqrt(res1_75)) / 3
res2 = np.sqrt(res1_25)
print(res1)
print(res2)
