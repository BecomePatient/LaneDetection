#!/home/luofan/anaconda3/envs/lanedetection/bin/python
import sys  
print(sys.executable)
from model import bisenetv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import os
from torchvision.transforms import ToPILImage
from tools import lanenet_postprocess
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
import math
from datetime import datetime 
from spikingjelly.activation_based import neuron, functional, surrogate, layer,learning
from torch.utils.data import DataLoader
from tools import dataset,loss
import time
from thop import profile,clever_format

class CombinedModel(nn.Module):
    def __init__(self, n_classes, embedding_dims):
        super(CombinedModel, self).__init__()
        self.bisenet = bisenetv2.SNN_BiSeNetV2(n_classes=n_classes)
        self.backend = bisenetv2.LaneNetBackEnd(embedding_dims=embedding_dims, num_classes=n_classes)

    def forward(self, x):
        binary_seg_logits, instance_seg_logits = self.bisenet(x)
        binary_seg_prediction, instance_seg_prediction = self.backend(binary_seg_logits, instance_seg_logits)
        return binary_seg_logits,instance_seg_logits,binary_seg_prediction,instance_seg_prediction

def minmax_scale(input_arr):
    """
    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def save_image(postprocess_result,binary_seg_prediction,instance_seg_prediction,image_vis):
        with_lane_fit = True
        save_dir = '/home/suepr20/luofan/my_lanedetection/Real_data/data/sample_image'
        mask_image = postprocess_result['mask_image']
        if with_lane_fit:
            lane_params = postprocess_result['fit_params']
            print('Model have fitted {:d} lanes'.format(len(lane_params)))
            for i in range(len(lane_params)):
                print('Fitted 2-order lane {:d} curve param: {}'.format(i + 1, lane_params[i]))
        print(instance_seg_prediction.shape)
        for i in range(4):
            instance_seg_prediction[0][:, :, i] = minmax_scale(instance_seg_prediction[0][:, :, i])
        embedding_image = np.array(instance_seg_prediction[0], np.uint8)
        # Save images
        print(f"image_vis = {image_vis.shape},embedding_image =  {type(embedding_image)}")
        cv2.imwrite(os.path.join(save_dir, 'mask_image.png'), mask_image[:, :, (2, 1, 0)])
        cv2.imwrite(os.path.join(save_dir, 'src_image.png'), image_vis[:, :, (2, 1, 0)])
        cv2.imwrite(os.path.join(save_dir, 'instance_image.png'), embedding_image[:, :, (2, 1, 0)])
        cv2.imwrite(os.path.join(save_dir, 'binary_image.png'), binary_seg_prediction[0]* 255)

        # x_array = []
        # y_array = []
        # z_array = []

        # height,weight = binary_seg_prediction[0].shape
        # for i in range(height):
        #     for j in range(weight):
        #         # print(binary_seg_prediction[0][i,j])
        #         if binary_seg_prediction[0][i,j]:
        #             x_array.append(embedding_image[i,j,0])
        #             y_array.append(embedding_image[i,j,1])
        #             z_array.append(embedding_image[i,j,2])

        # cord = np.array([x_array,y_array,z_array]).transpose()
        # print(cord.shape)
        # flattened_image = cord.reshape(-1, 3)
        # # 使用 Counter 统计四元组的出现次数
        # counts = Counter([tuple(row) for row in flattened_image])
        # x_array = []
        # y_array = []
        # z_array = []
        # # 打印每个四元组及其出现次数
        # for quad, count in counts.items():
        #     if(count > 5):
        #         print(f"Quadruplet {quad} appears {count} times")
        #         x_array.append(quad[0])
        #         y_array.append(quad[1])
        #         z_array.append(quad[2])
        
        # # 创建一个新的三维图形窗口
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # # 绘制三维散点图
        # ax.scatter(x_array, y_array, z_array)

        # # 保存图形到文件
        # plt.savefig('3d_scatter_plot.png')

        # # 关闭图形窗口
        # plt.close(fig)

def load_checkpoint(model,path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"已经成功加载模型，该模型训练的轮数为{epoch},训练的最终损失为{loss}")
    return model, epoch, loss

class ImageProcessor:  
    def __init__(self,weight_path = '/home/luofan/catkin_ws/src/my_lanedetection/weight/best_model_checkpoint_351.pth', save_dir='./result/'):  
        self.save_dir = save_dir 
        os.makedirs(self.save_dir, exist_ok=True)  
        self.model = CombinedModel(n_classes=2, embedding_dims=4)  
        self.model, self.epoch, self.resume_loss = load_checkpoint(self.model, weight_path)  
        self.postprocessor = lanenet_postprocess.LaneNetPostProcessor()  
        self.model.eval()  
        self.header_x = np.array([256]*50)
        self.header_y = np.array(list(range(207,257)))
        self.transform = transforms.Compose([  
            # transforms.Resize([512, 256], interpolation=2),  # 根据需要决定是否取消注释  
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ])  
  
    def eval(self, cv_image):  
        # image = Image.open(image_path)  
        # image_vis = np.array(image)  
  
        # image = self.transform(image).unsqueeze(0)  
        image_vis = cv_image[:, :, ::-1]  # OpenCV图像是BGR，转换为RGB以匹配PIL   
        image_pil = Image.fromarray(image_vis) 
        # 可能需要将图像转换为你的模型需要的尺寸和格式
        image = self.transform(image_pil).unsqueeze(0)
        #print(image.shape)
        # 获取当前时间  
        now = datetime.now()  
        # 格式化时间戳为字符串，例如 'YYYY-MM-DD_HH-MM-SS'  
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S") 

        with torch.no_grad():  
            binary_seg_logits, instance_seg_logits, binary_seg_prediction, instance_seg_prediction = self.model(image)  
            binary_seg_prediction = np.array(binary_seg_prediction)
            instance_seg_prediction = np.transpose(np.array(instance_seg_prediction),(0,2,3,1))
            #print(f"binary_seg_prediction = {binary_seg_prediction.shape} instance_seg_prediction = {instance_seg_prediction.shape} image_vis = {image_vis.shape}")
            postprocess_result = self.postprocessor.postprocess(  
                binary_seg_result=binary_seg_prediction[0],  
                instance_seg_result=instance_seg_prediction[0],  
                source_image=image_vis,  
                with_lane_fit=True,  
                data_source='tusimple'  
            )  
            lane_params = postprocess_result['fit_params']  
        nums = len(lane_params)
        theta_deg = 0
        #save_image(postprocess_result,binary_seg_prediction,instance_seg_prediction,image_vis)
        if nums == 2:    #只有检测两条车道线才对他进行处理
            [height,weight] = image_vis.shape[:2]
            #print(f"height = {height} weight = {weight}")
            central_y = np.linspace(height-130, height - 50)
            left_x = lane_params[0][0] * central_y ** 2 + lane_params[0][1] * central_y + lane_params[0][2]
            right_x = lane_params[1][0] * central_y ** 2 + lane_params[1][1] * central_y + lane_params[1][2]
            central_x = (left_x + right_x)//2
            #二次曲线拟合
            central_y = np.concatenate((central_y, self.header_y))
            central_x = np.concatenate((central_x, self.header_x))
            fit_param = np.polyfit(central_y, central_x, 1)
            plot_x = fit_param[0] * central_y + fit_param[1]
            plot_x = plot_x.astype(np.int32)  # 确保 x 坐标是整数类型
            
            # 中线是由一系列点组成的，所以需要将其转换为 (n, 1, 2) 的形状，其中 n 是点的数量  
            points = np.array([[[x, y] for x, y in zip(plot_x, central_y)]], dtype=np.int32) 
            #print(points)
            # 绘制中线，这里我们使用蓝色 (B, G, R)，线宽为 2  
            lane_color = (255, 0, 0)  # BGR 格式，蓝色  
            thickness = 2  
            line_img = image_vis.copy()
            line_img = cv2.polylines(line_img, points, isClosed=False, color=lane_color, thickness=thickness)  
            #cv2.imwrite(os.path.join(self.save_dir, f'line_img_{timestamp}.png'), line_img[:, :, (2, 1, 0)])
            line_k = fit_param[0]
            # 计算与x轴的夹角（弧度）  
            theta_rad = math.atan(line_k)  
            # 转换为度数
            #print(line_k)
            theta_deg = math.degrees(theta_rad)
            #print(theta_deg)
        return  nums,theta_deg   


def eval():
    save_dir = '/home/hnu/files/luofan/my_lanedetection/Real_data/data/sample_image'
    os.makedirs(save_dir, exist_ok=True) 
    model = CombinedModel(n_classes=2, embedding_dims=4)
    total_classes = [0.0000, 0.1176, 0.4902]
    tolerance = 1e-3
    # 加载模型权重到CPU
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    model, epoch, resume_loss = load_checkpoint(model, './weight/snn_sew_best_model_checkpoint.pth')
    model.to(device)
    print(device)
    ##测试帧率
    dataset_dir = '/home/hnu/files/luofan/my_lanedetection/mytraining_data_example'
    test_data = dataset.My_Data(txt_path=os.path.join(dataset_dir, "test.txt"), mode='test')
    test_data = torch.utils.data.Subset(test_data, range(4))
    test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=False,drop_last=True,num_workers=4,pin_memory=False)
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    num_classes = 2
    criterion_disc = loss.DiscriminativeLoss(delta_var=0.5,
                            delta_dist=3.0,
                            norm=2,
                            usegpu=True)
    mean_iou = loss.MeanIoU(num_classes)
    with torch.no_grad():
        for images, binary_label, instance_label, inverse_weights in test_loader:
            inverse_weights = torch.mean(inverse_weights, dim=0).to(device).squeeze(0)
            images = images.to(device)
            print(images.shape)
            binary_label = binary_label.to(device)
            instance_label = instance_label.to(device)

            binary_seg_logits,instance_seg_logits,binary_seg_prediction,instance_seg_prediction = model(images)
            # flops, params = profile(model, inputs=(images, ))
            # flops, params = clever_format([flops, params], "%.3f")
            # print(f"flops = {flops} params = {params}")
            binary_loss = loss.compute_binary_loss( binary_label,binary_seg_logits,weights = inverse_weights)

            n_clusters = []
            for i in range(len(images)):
                target_values, target_indices = torch.unique(instance_label[i], return_inverse=True)
                mapping = {v: i for i, v in enumerate(total_classes)}
                mapped_indices = [mapping[next((x for x in total_classes if abs(x - value.item()) < tolerance), None)] for value in target_values]
                n_clusters.append(mapped_indices)

            functional.reset_net(model)
            instance_loss = criterion_disc(instance_seg_prediction, instance_label,n_clusters)
            total_loss += instance_loss

            iou = mean_iou.compute_iou(binary_seg_prediction, binary_label)
            total_iou += iou
    avg_loss = total_loss / len(test_loader)
    avg_iou = total_iou / len(test_loader)
    print(f"Validation Loss: {avg_loss}, Validation IoU: {avg_iou}")
    
    
    # t2 = time.perf_counter()
    # print(f"fps = {len(test_data)/(t2-t1)}") 
    # print(len(test_loader))



    # postprocessor = lanenet_postprocess.LaneNetPostProcessor()

    # model.eval()
    # transform = transforms.Compose([
    #     # transforms.Resize([512, 256], interpolation=2),  # 将图像调整为目标大小，并使用线性插值
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    # image_path = '/home/suepr20/luofan/my_lanedetection/Real_data/data/sample_image/9.png'
    # image = Image.open(image_path)
    # image_vis = np.array(image)

    # image = transform(image).unsqueeze(0)
    # print(image.shape)

    # with torch.no_grad():
    #     binary_seg_logits, instance_seg_logits, binary_seg_prediction, instance_seg_prediction = model(image)
    #     binary_seg_prediction = np.array(binary_seg_prediction)
    #     instance_seg_prediction = np.transpose(np.array(instance_seg_prediction),(0,2,3,1))
    #     #print(f"image_vis = {image_vis.shape},{type(image_vis)}")
    #     print(f"binary_seg_prediction = {binary_seg_prediction.shape} instance_seg_prediction = {instance_seg_prediction.shape} image_vis = {image_vis.shape}")
    #     postprocess_result = postprocessor.postprocess(
    #         binary_seg_result=binary_seg_prediction[0],
    #         instance_seg_result=instance_seg_prediction[0],
    #         source_image=image_vis,
    #         with_lane_fit=True,
    #         data_source='tusimple'
    #     )
    #     #对检测得车道线进行处理，获取中点，中点拟合二次取曲线
    #     lane_params = postprocess_result['fit_params']

    #     if len(lane_params) == 2:    #只有检测两条车道线才对他进行处理
    #         [height,weight] = image_vis.shape[:2]
    #         #print(f"height = {height} weight = {weight}")
    #         central_y = np.linspace(height//3, height - 10)
    #         left_x = lane_params[0][0] * central_y ** 2 + lane_params[0][1] * central_y + lane_params[0][2]
    #         right_x = lane_params[1][0] * central_y ** 2 + lane_params[1][1] * central_y + lane_params[1][2]
    #         central_x = (left_x + right_x)//2
    #         fit_param = np.polyfit(central_y, central_x, 2)
    #         plot_x = fit_param[0] * central_y ** 2 + fit_param[1] * central_y + fit_param[2]
    #         plot_x = plot_x.astype(np.int32)  # 确保 x 坐标是整数类型  
    #         # 中线是由一系列点组成的，所以需要将其转换为 (n, 1, 2) 的形状，其中 n 是点的数量  
    #         points = np.array([[[x, y] for x, y in zip(plot_x, central_y)]], dtype=np.int32) 
    #         #print(points)
    #         # 绘制中线，这里我们使用蓝色 (B, G, R)，线宽为 2  
    #         lane_color = (255, 0, 0)  # BGR 格式，蓝色  
    #         thickness = 2  
    #         line_img = image_vis.copy()
    #         line_img = cv2.polylines(line_img, points, isClosed=False, color=lane_color, thickness=thickness)  
    #         #cv2.imwrite(os.path.join(save_dir, 'line_img.png'), line_img[:, :, (2, 1, 0)])
    #         line_k = 2*fit_param[0]*(height - 10) + fit_param[1]
    #         # 计算与x轴的夹角（弧度）  
    #         theta_rad = math.atan(line_k)  
    #         # 转换为度数
    #         print(line_k)
    #         theta_deg = math.degrees(theta_rad)
    #         print(theta_deg)

    #     save_image(postprocess_result,binary_seg_prediction,instance_seg_prediction,image_vis)


if __name__ == '__main__':
    eval()
