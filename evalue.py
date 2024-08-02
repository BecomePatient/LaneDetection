#!/home/luofan/anaconda3/envs/lanedetection/bin/python
import sys  
print(sys.executable)
from model import bisenetv2
import torch
import argparse
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
import torch.nn.functional as F
from pidnet_model import pidnet

class CombinedModel(nn.Module):
    def __init__(self, n_classes, embedding_dims):
        super(CombinedModel, self).__init__()
        self.bisenet = bisenetv2.BiSeNetV2(n_classes=n_classes)
        self.backend = bisenetv2.LaneNetBackEnd(embedding_dims=embedding_dims, num_classes=n_classes)

    def forward(self, x):
        binary_seg_logits, instance_seg_logits = self.bisenet(x)
        binary_seg_prediction, instance_seg_prediction = self.backend(binary_seg_logits, instance_seg_logits)
        return binary_seg_logits,instance_seg_logits,binary_seg_prediction,instance_seg_prediction

class Pidnet_Model(nn.Module):
    def __init__(self, n_classes, embedding_dims):
        super(Pidnet_Model, self).__init__()
        self.pidnet = pidnet.get_pred_model(name='pidnet_s', embedding_dims=embedding_dims, num_classes=n_classes,augment = True)
        #self.PostprocessNet = pidnet.PostprocessNet(max = 4, scale=10.0)

    def forward(self, x):
        binary_seg_logits, instance_seg_logits,_ = self.pidnet(x)
        print(f"binary_seg_logits = {binary_seg_logits.shape} instance_seg_logits = {instance_seg_logits.shape}")
        #实例输出
        instance_seg_prediction = instance_seg_logits

        #input shape
        bs,embeding,h,w = instance_seg_prediction.shape

        binary_seg_logits = F.softmax(binary_seg_logits, dim=1)

        # get lanes
        binary_mask = binary_seg_logits[:,1].unsqueeze(1)

        # expand
        print(f"binary_mask = {binary_mask.shape}")
        binary_mask = binary_mask.expand(bs,embeding,h,w)  
        print(f"binary_mask = {binary_mask.shape}")
        # fuse
        features = binary_mask * instance_seg_logits
        print(f"features = {features.shape}")
        # get correct pixels
        pixelcls = self.PostprocessNet(features)
        print(f"pixelcls = {pixelcls.shape}")

        binary_seg_prediction = torch.argmax(binary_seg_logits, dim=1)
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


def calculate_similarity_percentage(unique_coords, model_pts, threshold=20):
    """
    计算两个等长点序列的相似性百分比，因为模型输出范围大，因此先根据标注设定模型输出范围，利用模型
    输出中的点判断正确与否计算精确度。
    :param lane_cords: 标注,字典
    :param model_pts: 模型输出
    :param threshold: 判断两点是否相似的距离阈值
    :return: 相似性百分比
    """
    sum_count = 0
    true_count = 0
    means = []
    for i in range(len(model_pts)):
        x = [coord[0] for coord in model_pts[i]]
        mean = sum(x) / len(x)
        means.append(mean)
    sorted_means_and_pts = sorted(zip(means, model_pts), key=lambda pair: pair[0])
    means, model_pts = zip(*sorted_means_and_pts)
    means = list(means);model_pts =list(model_pts)

    acc = 0 
    if(len(unique_coords) == len(model_pts)):
        for i in range(len(model_pts)):
            min_idx = min(list(unique_coords[i].keys()))
            max_idx = max(list(unique_coords[i].keys()))
            for j in range(len(model_pts[i])):
                if(model_pts[i][j][1] >= min_idx and model_pts[i][j][1] <= max_idx):
                    sum_count += 1
                    diff = abs(unique_coords[i][model_pts[i][j][1]] - model_pts[i][j][0])
                    # print(f"diff = {diff}")
                    if(diff <= threshold):
                        # print(f"diff = {diff}")
                        true_count+= 1
        acc = true_count/sum_count
    return acc

def save_image(postprocess_result,binary_seg_prediction,instance_seg_prediction,image_vis,name,save_dir):
        with_lane_fit = True
        save_dir = save_dir
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
        cv2.imwrite(os.path.join(save_dir, f'mask_image_{name}.png'), mask_image[:, :, (2, 1, 0)])
        print("hhh")
        cv2.imwrite(os.path.join(save_dir, f'src_image_{name}.png'), image_vis[:, :, (2, 1, 0)])
        print("hhh")
        cv2.imwrite(os.path.join(save_dir, f'instance_image{name}.png'), embedding_image[:, :, (2, 1, 0)])
        cv2.imwrite(os.path.join(save_dir, f'binary_image{name}.png'), binary_seg_prediction[0]* 255)

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

def eval(args):
    save_dir = f'./{args.model}_output'
    device = args.device
    os.makedirs(save_dir, exist_ok=True)
    if(args.model == "bisenet"):
        model = CombinedModel(n_classes=2,embedding_dims = 4).to(device)
    elif(args.model == "pidnet"):
        model = Pidnet_Model(n_classes=2,embedding_dims = 4).to(device)
    else:
        raise ValueError(f"Unknown model name: {args.model}. Please choose 'bisenet' or 'pidnet'.")
    total_classes = torch.tensor([0.0000, 0.1176, 0.4902]).to(device)
    tolerance = 1e-3
    # 加载模型权重到CPU
    model, epoch, resume_loss = load_checkpoint(model, args.weights)
    model.to(device)
    print(device)
    ##测试帧率
    dataset_dir = args.dataset_dir
    test_data = dataset.My_Data(txt_path=os.path.join(dataset_dir, "test.txt"), mode='test')
    # test_data = torch.utils.data.Subset(test_data, range(1))
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False,drop_last=True,num_workers=4,pin_memory=False)
    model.eval()

    num_classes = 2
    postprocessor = lanenet_postprocess.LaneNetPostProcessor()
    acc = 0
    with torch.no_grad():
        for images, binary_label, instance_label, inverse_weights,path,binary_seg_path in test_loader:
            name = dataset.extract_frame_number(path[0])
            inverse_weights = torch.mean(inverse_weights, dim=0).to(device).squeeze(0)
            image_vis = np.array(Image.open(path[0]))

            images = images.to(device)
            binary_label = binary_label.to(device)
            instance_label = instance_label.to(device)

            binary_seg_logits,instance_seg_logits,binary_seg_prediction,instance_seg_prediction = model(images)

            binary_seg_prediction  = np.array(binary_seg_prediction.cpu())
            instance_seg_prediction = np.transpose(np.array(instance_seg_prediction.cpu()),(0,2,3,1))

            postprocess_result = postprocessor.postprocess(
                binary_seg_result=binary_seg_prediction[0],
                instance_seg_result=instance_seg_prediction[0],
                source_image=image_vis,
                with_lane_fit=True,
                data_source='tusimple'
            )

            print(instance_label.shape)

            target_flat = instance_label.flatten().unsqueeze(1)  # (N, 1)
            total_classes_expanded = total_classes.unsqueeze(0)  # (1, C)
            diff = torch.abs(target_flat - total_classes_expanded)
            min_diff, target_indices = torch.min(diff, dim=1)
            target_indices = target_indices.view(instance_label.shape)

            ## take different coords into lane_cords
            lane_cords = [[] for _ in range(len(total_classes)-1)]
            for idx, class_val in enumerate(range(1, len(total_classes))):
                coords = (target_indices[0] == class_val).nonzero(as_tuple=False)
                lane_cords[idx] = coords.tolist()

            ## draw blue label on image_vis
            # for i in range(len(total_classes) - 1):
            #     nonzero_y = [coord[0] for coord in lane_cords[i]]
            #     nonzero_x = [coord[1] for coord in lane_cords[i]]
            #     for i in range(len(nonzero_y)):
            #         cv2.circle(image_vis, (int(nonzero_x[i]),
            #                     int(nonzero_y[i])), 1, np.array([0, 0, 255]).tolist(), -1)

            ## get label point for single row
            unique_coords = []
            for i in range(len(total_classes) - 1):
                unique_coord = {}
                for y, x in lane_cords[i]:
                    if y not in unique_coord:
                        unique_coord[y] = x
                unique_coords.append(unique_coord)

            model_pts = postprocess_result['pts']
            print(f"模型检测{len(model_pts)}条车道")

            if(len(model_pts) > 0):
                Acc = calculate_similarity_percentage(unique_coords, model_pts, threshold=25)
            if(len(model_pts) != 2):
                Acc = 0
                save_image(postprocess_result,binary_seg_prediction,instance_seg_prediction,image_vis,name,save_dir)
            acc += Acc

    acc = acc/len(test_data)
    print(f"模型准确率为：{acc}")
    # avg_loss = total_loss / len(test_loader)
    # avg_iou = total_iou / len(test_loader)
    # print(f"Validation Loss: {avg_loss}, Validation IoU: {avg_iou}")
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

    # image_path = '/home/hnu/cfiles/luofan/my_lanedetection/mytraining_data_example/image/frame_99.png'
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
    #     postprocess_label = postprocessor.postprocess(
    #         binary_seg_result=binary_seg_prediction[0],
    #         instance_seg_result=instance_seg_prediction[0],
    #         source_image=image_vis,
    #         with_lane_fit=True,
    #         data_source='tusimple'
    #     )
    # save_image(postprocess_result,binary_seg_prediction,instance_seg_prediction,image_vis)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--model', type=str, required=True, help="The name of the model to be used.")
    parser.add_argument('--dataset_dir', type=str, default='./mytraining_data_example', help="The path to the dataset. Default is 'path/to/default/dataset'.")
    parser.add_argument('--weights', type=str, default='./weight/opt_best_model_pidnet_checkpoint.pth',help="The path to the model weights.")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda:0','cuda:1'], default='cpu', help="The device to be used (cpu or cuda).")
    args = parser.parse_args()
    eval(args)
