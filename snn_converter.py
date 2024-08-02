from model import bisenetv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import os
from torchvision.transforms import ToPILImage
import spikingjelly.activation_based.ann2snn as ann2snn
from tools import lanenet_postprocess
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
from torch.utils.data import DataLoader
from tools import dataset
import torch.nn.functional as F
from thop import profile,clever_format

class CombinedModel(nn.Module):
    def __init__(self, n_classes, embedding_dims):
        super(CombinedModel, self).__init__()
        self.bisenet = bisenetv2.BiSeNetV2(n_classes=n_classes)
        self.backend = bisenetv2.LaneNetBackEnd(embedding_dims=embedding_dims, num_classes=n_classes)

    def forward(self, x):
        binary_seg_logits, instance_seg_logits = self.bisenet(x)
        binary_seg_prediction, instance_seg_prediction = self.backend(binary_seg_logits, instance_seg_logits)
        return binary_seg_logits,instance_seg_logits,binary_seg_prediction,instance_seg_prediction
    

def minmax_scale(input_arr):
    """
    :param input_arr:
    :return
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def load_checkpoint(model,path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"已经成功加载模型，该模型训练的轮数为{epoch},训练的最终损失为{loss},开始进行SNN的转化。")
    return model, epoch, loss

def val(snn_model, test_image,T,image_vis,device = "cpu",save_flag = False):
    save_dir = './result/'
    snn_model.eval().to(device)
    postprocessor = lanenet_postprocess.LaneNetPostProcessor()
    flops, params = profile(snn_model, inputs=(test_image, ))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"flops = {flops} params = {params}")
    with torch.no_grad():
        for m in snn_model.modules():
            if hasattr(m, 'reset'):
                m.reset()
        for t in range(T):
            if t == 0:
                binary_seg_logits, _, _, instance_seg_prediction = snn_model(test_image)
            else:
                binary_seg_logits_temp, _, _, instance_seg_prediction_temp = snn_model(test_image)
                binary_seg_logits += binary_seg_logits_temp; instance_seg_prediction += instance_seg_prediction_temp
        binary_seg_logits = binary_seg_logits/T; instance_seg_prediction= instance_seg_prediction/T
        binary_seg_score = F.softmax(binary_seg_logits, dim=1)
        binary_seg_prediction = torch.argmax(binary_seg_score, dim=1)

    if save_flag:
        binary_seg_prediction = np.array(binary_seg_prediction)
        instance_seg_prediction = np.transpose(np.array(instance_seg_prediction),(0,2,3,1))
        print(f"image_vis = {image_vis.shape},{type(image_vis)}")

        print(f"binary_seg_prediction = {binary_seg_prediction[0].shape} instance_seg_prediction = {instance_seg_prediction.shape}")
        postprocess_result = postprocessor.postprocess(
            binary_seg_result=binary_seg_prediction[0],
            instance_seg_result=instance_seg_prediction[0],
            source_image=image_vis,
            with_lane_fit=True,
            data_source='tusimple'
        )

        mask_image = postprocess_result['mask_image']
        lane_params = postprocess_result['fit_params']
        if lane_params != None:
            print('Model have fitted {:d} lanes'.format(len(lane_params)))
            for i in range(len(lane_params)):
                print('Fitted 2-order lane {:d} curve param: {}'.format(i + 1, lane_params[i]))
        for i in range(4):
            instance_seg_prediction[0][:, :, i] = minmax_scale(instance_seg_prediction[0][:, :, i])
        embedding_image = np.array(instance_seg_prediction[0], np.uint8)
        # Save images
        print(f"image_vis = {image_vis.shape},embedding_image =  {type(embedding_image)}")
        cv2.imwrite(os.path.join(save_dir, f'{T}_binary_image.png'), binary_seg_prediction[0] * 255)
        cv2.imwrite(os.path.join(save_dir, f'{T}_src_image.png'), image_vis[:, :, (2, 1, 0)])
        cv2.imwrite(os.path.join(save_dir, f'{T}_instance_image.png'), embedding_image[:, :, (2, 1, 0)])
        cv2.imwrite(os.path.join(save_dir, f'{T}_mask_image.png'), mask_image[:, :, (2, 1, 0)])

        x_array = []
        y_array = []
        z_array = []

        height,weight = binary_seg_prediction[0].shape
        for i in range(height):
            for j in range(weight):
                # print(binary_seg_prediction[0][i,j])
                if binary_seg_prediction[0][i,j]:
                    x_array.append(embedding_image[i,j,0])
                    y_array.append(embedding_image[i,j,1])
                    z_array.append(embedding_image[i,j,2])

        cord = np.array([x_array,y_array,z_array]).transpose()
        print(cord.shape)
        flattened_image = cord.reshape(-1, 3)
        # 使用 Counter 统计四元组的出现次数
        counts = Counter([tuple(row) for row in flattened_image])
        x_array = []
        y_array = []
        z_array = []
        # 打印每个四元组及其出现次数
        for quad, count in counts.items():
            if(count > 5):
                # print(f"Quadruplet {quad} appears {count} times")
                x_array.append(quad[0])
                y_array.append(quad[1])
                z_array.append(quad[2])
        
        # 创建一个新的三维图形窗口
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 绘制三维散点图
        ax.scatter(x_array, y_array, z_array)

        # 保存图形到文件
        plt.savefig('3d_scatter_plot.png')

        # 关闭图形窗口
        plt.close(fig)

    return binary_seg_prediction,instance_seg_prediction

def eval():
    save_dir = './result/'
    with_lane_fit = True
    T = [1]
    os.makedirs(save_dir, exist_ok=True) 
    model = CombinedModel(n_classes=2, embedding_dims=4)
    # 加载模型权重到CPU
    model, epoch, resume_loss = load_checkpoint(model, './weight/last_model_checkpoint.pth')
    print(model)
    #加载数据集，用于计算SNN归一化参数
    dataset_dir = './mytraining_data_example'
    train_data = dataset.My_Data(txt_path=os.path.join(dataset_dir, "train.txt"), mode='train')
    #train_data = torch.utils.data.Subset(train_data, range(1))
    train_loader = DataLoader(dataset=train_data, batch_size=4, shuffle=True,num_workers=4,pin_memory=True)

    model_converter = ann2snn.Converter(mode='99.9%', dataloader=train_loader)
    snn_model = model_converter(model)
    print(snn_model)

    snn_model.eval()
    transform = transforms.Compose([
        # transforms.Resize([512, 256], interpolation=2),  # 将图像调整为目标大小，并使用线性插值
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print("数据集中有%d张图片",len(mydataloader))
    image_path = './mytraining_data_example/image/frame_2.png'
    image = Image.open(image_path)
    image_vis = np.array(image)

    image = transform(image).unsqueeze(0)
    for t in T:
        img = image_vis.copy()
        val(snn_model, image,t,img,device = "cpu",save_flag = True)

if __name__ == '__main__':
    eval()
