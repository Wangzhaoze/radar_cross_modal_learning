import glob
import numpy as np
import torch
import cv2
import sys
import os
sys.path.append(os.getcwd())
import glob
from models.sat2seg_model import sat2seg_UNet



def sat2seg(sat_image):

    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载网络，图片单通道，分类为1。
    net = sat2seg_UNet(n_channels=1, n_classes=1)

    # 将网络拷贝到deivce中
    net.to(device=device)

    # 加载模型参数
    net.load_state_dict(torch.load('./pth/best_model_sat2seg_k3k7.pth', map_location=device))

    # 测试模式
    net.eval()

    origin_shape = sat_image.shape

    # 转为灰度图
    img = cv2.cvtColor(sat_image, cv2.COLOR_RGB2GRAY)
    #img = cv2.resize(img, (1000, 1000))

    # 转为batch为1，通道为1，大小为512*512的数组
    img = img.reshape(1, 1, img.shape[0], img.shape[1])

    # 转为tensor
    img_tensor = torch.from_numpy(img)

    # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)

    # 预测
    pred = net(img_tensor)

    # 提取结果
    pred = np.array(pred.data.cpu()[0])[0]

    # 处理结果
    pred[pred >= 0.5] = 255
    pred[pred < 0.5] = 0
    
    return pred

if __name__ == '__main__':


    val_images_path = './dataset/sat/'
    save_path = './dataset/sat2seg_pred/'

    img_paths = os.listdir(val_images_path)
    for img_path in img_paths:

        sat = cv2.imread(val_images_path+img_path)
        pred = sat2seg(sat)
        cv2.imwrite(save_path+img_path, pred)