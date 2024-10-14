#!/home/luofan/anaconda3/envs/lanedetection/bin/python
import sys  
print(sys.executable)
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
from ptflops import get_model_complexity_info
import torch.nn.functional as F
from models.model import CombinedModel,Pidnet_Model,STDC_Model
import tqdm

def minmax_scale(input_arr):
    """
    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr

def EndeValue(pixelcls,src_image,device,max = 8):
    bs, n_class, h, w = pixelcls.shape
    th = 0.35
    pixelcls = F.softmax(pixelcls, dim=1)
    # mmmax = torch.max(pixelcls)
    # mmmin = torch.min(pixelcls)
    # print(f"max= {mmmax} min = {mmmin}")
    pixelcls[:,0,:,:] += th
    pixelcls[:,1:,:,:] -= th
    pixelcls = torch.argmax(pixelcls, dim=1)

    # print(f"pixelcls = {pixelcls.shape}")
    # pixelcls_numpy = pixelcls.numpy()
    # # 将 NumPy 数组转换为字符串
    # np.savetxt('pixelcls.txt', pixelcls_numpy.reshape(-1, pixelcls_numpy.shape[-1]), fmt='%d', delimiter='\t')

    categories = torch.arange(max)  # [0, 1, 2, 3]
    total_classes_expanded = categories.view(1, -1, 1, 1).to(device)  # (1, C, 1, 1)
    pixelcls_expanded = pixelcls.unsqueeze(1)  # (bs, 1, h, w)

    diff = torch.abs(pixelcls_expanded - total_classes_expanded)

    min_diff, mapped_indices = torch.min(diff, dim=1)  # (bs, h, w)

    category_indices = {i: [] for i in range(max)}

    for category in categories:
        indices = torch.nonzero(mapped_indices == category, as_tuple=True)
        b_indices, h_indices, w_indices = indices
        for b in range(bs):
            mask = b_indices == b
            hw_indices = list(zip(h_indices[mask].tolist(), w_indices[mask].tolist()))
            category_indices[category.item()].append(hw_indices)

    # 拟合参数存储
    lane_fits = []

    # 输出每个类别的 (h, w) 索引
    lanes = 0
    for category, indices in category_indices.items():
        for b in range(bs):
            count = len(indices[b])
            # print(f"category = {category} Batch {b}: {count}")
            # 如果车道线长度大于100，进行曲线拟合
            if category != 0 and count > 10:
                lanes += 1
                hw_indices = np.array(indices[b])
                # 提取h和w坐标
                h_coords = hw_indices[:, 0]
                w_coords = hw_indices[:, 1]
                max_h = np.max(h_coords)*8
                min_h = np.min(h_coords)*8
                for x, y in zip(w_coords, h_coords):
                    cv2.circle(src_image, (8*x, 8*y), radius=5, color=(lanes*30, lanes*30, lanes*30), thickness=-1)
                # 使用二次多项式拟合
                fit_params = np.polyfit(h_coords, w_coords, 3)  # 二次拟合
                lane_fits.append((fit_params,max_h,min_h))

        # 获取图像的高度和宽度
    height, width, _ = src_image.shape
    # print(f"height = {height} width = {width}")

    model_pts = []  # 用于存储车道线坐标

    # 取消注释绘制车道线
    # for fit_params,max_h,min_h in lane_fits:
    #     # fit_params[1] = fit_params[1] * 8
    #     ##根据尺度变换参数
    #     fit_params[0] = fit_params[0]/8/8
    #     fit_params[1] = fit_params[1]/8
    #     fit_params[3] = fit_params[3]*8
    #     fit_line = np.poly1d(fit_params)

    #     # 计算在图像宽度范围内的车道线坐标
    #     h_coords = np.arange(height)
    #     w_coords = fit_line(h_coords) 

    #     # 确保坐标在图像范围内
    #     valid_indices = (w_coords >= 0) & (w_coords < width) &  (h_coords <= max_h) & (h_coords >=  min_h)
    #     w_coords = w_coords[valid_indices].astype(int)
    #     h_coords = h_coords[valid_indices].astype(int)

    #     # 将有效的 (w, h) 坐标对添加到 model_pts
    #     model_pts.append(list(zip(w_coords, h_coords)))
    #     # 在图像上绘制车道线
    #     for x, y in zip(w_coords, h_coords):
    #         cv2.circle(src_image, (x, y), radius=5, color=(255, 0, 255), thickness=-1)

    return model_pts,src_image

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

    Acc = [] ## 所有苗带的准确率
    crop_acc = 0
    print(f"unique_coords = {len(unique_coords)}")
    # for i in range(len(unique_coords)):
    #     print(f"第{i}条苗带的标注点数{len(list(unique_coords[i].keys()))}")
    correct_crop = 0 # 识别正确的苗带数
    for i in range(len(model_pts)):
        acc = 0  ## 单条苗带的准确率，与所有的标注进行计算，找到最大的认为就是最匹配的
        for k in range(len(unique_coords)):
            sum_count = 0
            true_count = 0
            if(len(list(unique_coords[k].keys())) == 0):
                continue
            if(len(list(unique_coords[k].keys())) != 0):
                min_idx = min(list(unique_coords[k].keys()))
                max_idx = max(list(unique_coords[k].keys()))
                for j in range(len(model_pts[i])):
                    if(model_pts[i][j][1] >= min_idx and model_pts[i][j][1] <= max_idx):
                        sum_count += 1
                        try:
                            diff = abs(unique_coords[k][model_pts[i][j][1]] - model_pts[i][j][0])
                        except KeyError as e:
                            print(f"在 unique_coords[{k}] 中未找到键 {model_pts[i][j][1]}: {e}")
                            print(f"unique_coords[{k}] 中可用的键有: {unique_coords[k].keys()}")
                            continue
                        # print(f"diff = {diff}")
                        if(diff <= threshold):
                            # print(f"diff = {diff}")
                            true_count+= 1
            if(sum_count != 0):
                acc = max(acc,true_count/sum_count)
        if(acc > 0.5):  ## 大于0.8认为苗带正确分类
            correct_crop += 1
        Acc.append(acc)
    print(f"每条苗带的准确率如下：{Acc}")
    mean_acc = sum(Acc)/len(Acc)
    print(f"苗带的平均准确率如下：{mean_acc}")
    all_crop = 0 # 总的苗带数
    for k in range(len(unique_coords)):
        if(len(list(unique_coords[k].keys())) != 0):
            all_crop += 1
    crop_acc = correct_crop/all_crop
    print(f"总的苗带数为{all_crop},正确识别的苗带数为{correct_crop},苗带正确率为{crop_acc}")        
    return mean_acc,crop_acc

def save_image(model_pts,binary_seg_prediction,instance_seg_prediction,middle_map,image_vis,name,save_dir):
        for points in model_pts:
            for (x,y) in points:
                cv2.circle(image_vis, (x, y), radius=5, color=(255, 0, 255), thickness=-1)        
        for i in range(4):
            instance_seg_prediction[0][:, :, i] = minmax_scale(instance_seg_prediction[0][:, :, i])
        for i in range(4):
            middle_map[0][:, :, i] = minmax_scale(middle_map[0][:, :, i])
        embedding_image = np.array(instance_seg_prediction[0], np.uint8)
        fuse_image = np.array(middle_map[0], np.uint8)
        cv2.imwrite(os.path.join(save_dir, f'src_image_{name}.png'), image_vis[:, :, (2, 1, 0)])
        cv2.imwrite(os.path.join(save_dir, f'instance_image{name}.png'), embedding_image[:, :, (2, 1, 0)])
        cv2.imwrite(os.path.join(save_dir, f'fuse_image{name}.png'), fuse_image[:, :, (2, 1, 0)])
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
    print(f"model = {model}")
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
    width,height = args.image_size
    max_class = 8
    total_classes = torch.tensor([0.1176 * x for x in range(8)]).to(device)
    os.makedirs(save_dir, exist_ok=True)
    if(args.model == "bisenet"):
        model = CombinedModel(n_classes=2,embedding_dims =4,height=height,width=width,max_class = max_class,mode = 'eval').to(device)
    elif(args.model == "pidnet"):
        model = Pidnet_Model(n_classes=2,embedding_dims = 4,height=height,width=width,max_class = max_class,mode = 'eval').to(device)
    elif(args.model == "stdcnet"):
        model = STDC_Model(n_classes=2,embedding_dims = 4,height=height,width=width,max_class = max_class,mode = 'eval').to(device)
    else:
        raise ValueError(f"Unknown model name: {args.model}. Please choose 'bisenet' or 'pidnet'.")
    # 加载模型权重到CPU
    model, epoch, resume_loss = load_checkpoint(model, args.weights)
    # 计算模型参数与浮点操作数
    model.to(device)
    print(device)
    ##测试帧率
    dataset_dir = args.dataset_dir
    test_data = dataset.My_Data(txt_path=os.path.join(dataset_dir, "test.txt"), mode='test')
    # test_data = torch.utils.data.Subset(test_data, range(1))
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False,drop_last=True,num_workers=4,pin_memory=False)
    model.eval()
    # postprocessor = lanenet_postprocess.LaneNetPostProcessor()

    acc = 0
    crop_acc = 0
    fps = 0
    
    all_time = 0
    pbar = tqdm.tqdm(test_loader)
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(pbar):
            images =  batch_data['image']
            binary_label =  batch_data['binary_label_onehot'] 
            instance_label =  batch_data['instance_seg']
            label_seg =  batch_data['label_seg'] 
            inverse_weights =  batch_data['inverse_weights'] 
            path = batch_data['img_path']
            binary_seg_path = batch_data['binary_seg_path']
            print(f"path = {path[0]}")
            name = dataset.extract_frame_number(path[0])
            inverse_weights = torch.mean(inverse_weights, dim=0).to(device).squeeze(0)
            image_vis = np.array(Image.open(path[0]).resize((width,height)))

            images = images.to(device)
            print(f"images = {images.shape}")
            binary_label = binary_label.to(device)
            instance_label = instance_label.to(device)
            label_seg = label_seg.to(device)

            start_time = time.time()

            result = model(images)
            binary_seg_prediction = result['binary_pred']
            instance_seg_prediction = result['seg_pred']
            pixelcls = result['pixelcls']
            middle_map = result['middle_map']
            con0 = result['con0']

            # mean_con0 = torch.mean(con0, dim=1, keepdim=True).squeeze()
            # binary_seg_prediction = binary_seg_prediction + 1 - mean_con0
            # binary_seg_prediction[binary_seg_prediction <= 0.1] = 0
            # binary_seg_prediction[binary_seg_prediction > 0.1] = 1
            # binary_seg_prediction = binary_seg_prediction.int()

            model_pts,image_vis = EndeValue(pixelcls,image_vis,device = device)

            # for i, elem in enumerate(model_pts):
            #     print(f"元素 {i}: {np.shape(elem)}")

            end_time = time.time()
            elapsed_time = end_time - start_time
            if(batch_idx > 0):
                print(f"infer: {elapsed_time} seconds")
                all_time += elapsed_time
            binary_seg_prediction  = np.array(binary_seg_prediction.cpu())
            instance_seg_prediction = np.transpose(np.array(instance_seg_prediction.cpu()),(0,2,3,1))
            middle_map = np.transpose(np.array(middle_map.cpu()),(0,2,3,1))

            target_flat = label_seg.flatten().unsqueeze(1)  # (N, 1)
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
            for i in range(len(total_classes) - 1):
                nonzero_y = [coord[0] for coord in lane_cords[i]]
                nonzero_x = [coord[1] for coord in lane_cords[i]]
                for i in range(len(nonzero_y)):
                    cv2.circle(image_vis, (int(nonzero_x[i]),
                                int(nonzero_y[i])), 1, np.array([0, 0, 255]).tolist(), -1)
            ## get label point for single row
            unique_coords = []
            for i in range(len(total_classes) - 1):
                unique_coord = {}
                for y, x in lane_cords[i]:
                    if y not in unique_coord:
                        unique_coord[y] = x
                unique_coords.append(unique_coord)
            Acc = 0
            Crop_acc = 0
            ## 取消注释可以计算准确度
            # if(len(model_pts) > 0):
            #     Acc,Crop_acc = calculate_similarity_percentage(unique_coords, model_pts, threshold=12)
                # if(Acc < 0.45):
                #     postprocess_result = postprocessor.postprocess(
                #             binary_seg_result=binary_seg_prediction[0],
                #             instance_seg_result=instance_seg_prediction[0],
                #             source_image=image_vis,
                #             with_lane_fit=True,
                #             data_source='tusimple'
                #         )
                #     save_image(postprocess_result,binary_seg_prediction,instance_seg_prediction,image_vis,name,save_dir)               
            acc += Acc
            crop_acc += Crop_acc
    print(f"test_data = {len(test_data)}")
    acc = acc/len(test_data)
    crop_acc = crop_acc/len(test_data)
    all_time = all_time/(len(test_data) - 1)  ## 计算帧率跳过第一张
    print(f"苗带像素点准确率为：{acc}")
    print(f"苗带数目准确率为：{crop_acc}")
    print(f"测试集在{device}推理单张图片的平均时间：{all_time}")
    print(f"测试集在{device}推理图片的帧率：{1/all_time}")
    # macs, params = get_model_complexity_info(model, (3,height,width), as_strings=True, backend='pytorch',
    #                                        print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--model', type=str, required=True, help="The name of the model to be used.")
    parser.add_argument('--dataset_dir', type=str, default='./CRDLD', help="The path to the dataset. Default is 'path/to/default/dataset'.")
    parser.add_argument('--weights', type=str, default='./weight_256/CRDLD_opt_end_best_model_stdcnet_checkpoint.pth',help="The path to the model weights.")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda:0','cuda:1'], default='cuda:0', help="The device to be used (cpu or cuda).")
    parser.add_argument('--image_size', type=int, nargs=2, metavar=('width', 'height'), default=(256, 256), help="The size of the image (width height). Default is (640, 480).")
    args = parser.parse_args()
    eval(args)
