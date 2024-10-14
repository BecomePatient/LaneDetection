from tools import dataset,loss
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
import tqdm
import random
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import pandas as pd
import time
from models.model import CombinedModel,Pidnet_Model,STDC_Model

def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False

def unfreeze_module(module):
    for param in module.parameters():
        param.requires_grad = True

def save_checkpoint(model, optimizer, epoch, loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def train(args):
    weight_dir = './weight/'
    os.makedirs(weight_dir, exist_ok=True)  
    epochs = 100
    num_classes = 2
    best_loss = float('inf')
    #实例分割具体类别 tolerance 代表浮点数接近程度
    max_class = 8
    total_classes = [0.1176 * x for x in range(8)] 
    tolerance = 1e-3
    device = args.device
    if(device != "cpu"):
        args.cuda = True
    width,height = args.image_size
    mean_iou = loss.MeanIoU(num_classes,device)
    if(args.model == "bisenet"):
        model = CombinedModel(n_classes=2,embedding_dims =4,height=height,width=width,max_class = max_class,mode = 'train').to(device)
    elif(args.model == "pidnet"):
        model = Pidnet_Model(n_classes=2,embedding_dims = 4,height=height,width=width,max_class = max_class,mode = 'train').to(device)
    elif(args.model == "stdcnet"):
        model = STDC_Model(n_classes=2,embedding_dims = 4,height=height,width=width,max_class = max_class,mode = 'train').to(device)
    else:
        raise ValueError(f"Unknown model name: {args.model}. Please choose 'bisenet' or 'pidnet'.")
    print(model)
    dataset_dir = args.dataset_dir
    print(dataset_dir)
    train_data = dataset.My_Data(txt_path=os.path.join(dataset_dir, "train.txt"), mode='train',model = args.model)
    # train_data = torch.utils.data.Subset(train_data, range(8))
    
    test_data = dataset.My_Data(txt_path=os.path.join(dataset_dir, "test.txt"), mode='test',model = args.model)
    
    train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True,num_workers=8,pin_memory=True,drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size=8, shuffle=False,num_workers=8,pin_memory=True,drop_last=True)

    criterion_disc = loss.DiscriminativeLoss(delta_var=0.5,
                                delta_dist=3.0,
                                norm=2,
                                usegpu=device,
                                total_classes=total_classes,
    )
    criterion_con = loss.SegmentationLosses(cuda=args.cuda).build_loss(mode="focal")
    pixelcls_loss = loss.EndLossCalculator(device = device,total_classes=total_classes)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    epoch = 0
    #恢复训练
    if args.resume:
        model, optimizer, epoch, resume_loss = load_checkpoint(model, optimizer, f'./weight/CRDLD_opt_end_last_model_{args.model}_checkpoint.pth')
        print(f"已经重新恢复训练，从第{epoch}轮重新开始训练，损失值为{resume_loss}")

    binary_loss_arr = []
    instance_loss_arr = []

    stage = args.stage
    # 冻结后处理部分，只训练获取分割与分类特征图部分
    if(stage == 1):
        for epoch in range(epoch,epochs):
            pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}')
            unfreeze_module(model.firstNet)
            freeze_module(model.PostprocessNet)
            model.train()
            for batch_idx, batch_data in enumerate(pbar):
                if(args.model == "bisenet"):
                    images =  batch_data['image']
                    binary_label =  batch_data['binary_label_onehot'] 
                    instance_label =  batch_data['instance_seg']
                    border_label =  batch_data['border_onehot'] 
                    label_seg =  batch_data['label_seg'] 
                    inverse_weights =  batch_data['inverse_weights'] 
                    connect_label = batch_data['connect_label']   
                    border_label = border_label.to(device)
                elif(args.model == "pidnet" or args.model == "stdcnet"):
                    images =  batch_data['image']
                    binary_label =  batch_data['binary_label_onehot'] 
                    instance_label =  batch_data['instance_seg']
                    border_label =  batch_data['border_onehot'] 
                    label_seg =  batch_data['label_seg'] 
                    inverse_weights =  batch_data['inverse_weights'] 
                    connect_label = batch_data['connect_label']   
                    border_label = border_label.to(device)

                inverse_weights = torch.mean(inverse_weights, dim=0).to(device).squeeze(0)
                # inverse_weights = torch.flip(inverse_weights, dims=[0])

                images = images.to(device)
                binary_label = binary_label.to(device)
                connect_label = connect_label.to(device)
                # instance_label = instance_label.to(device)
                label_seg = label_seg.to(device)
                instance_label = label_seg

                if(args.model == "bisenet"):
                    result = model(images)
                    binary_seg_logits = result['binary_logits']
                    binary_seg_prediction = result['binary_pred']
                    instance_seg_prediction = result['seg_pred']
                    pixelcls = result['pixelcls']
                    con0 = result['con0']
                elif(args.model == "pidnet"):
                    result = model(images)
                    binary_seg_logits = result['binary_logits']
                    binary_seg_prediction = result['binary_pred']
                    instance_seg_prediction = result['seg_pred']
                    x_extra_d = result['x_extra_d']
                    pixelcls = result['pixelcls']
                    con0 = result['con0']
                    border_loss = loss.compute_border_loss(binary_label,x_extra_d,weights = inverse_weights)
                elif(args.model == "stdcnet"):
                    result = model(images)
                    binary_seg_logits = result['binary_logits']
                    binary_seg_prediction = result['binary_pred']
                    instance_seg_prediction = result['seg_pred']
                    x_extra_d = result['x_extra_d']
                    pixelcls = result['pixelcls']
                    con0 = result['con0']
                    detail_loss = loss.compute_border_loss(binary_label,x_extra_d,weights = inverse_weights)

                # cls_loss = pixelcls_loss.compute_loss_V2(label_seg,pixelcls,inverse_weights)
                binary_loss = loss.compute_binary_loss(binary_label,binary_seg_logits,weights = inverse_weights)
                # print(f"connect_label = {connect_label.shape}")
                connect_loss = criterion_con(con0, connect_label,inverse_weights)
                n_clusters = []

                for i in range(len(images)):
                    target_values, target_indices = torch.unique(instance_label[i], return_inverse=True)
                    # print(f"target_values = {target_values}")
                    mapping = {v: i for i, v in enumerate(total_classes)}
                    mapped_indices = [mapping[next((x for x in total_classes if abs(x - value.item()) < tolerance), None)] for value in target_values]
                    n_clusters.append(mapped_indices)
                # print(n_clusters)

                instance_loss = 0.25*criterion_disc(instance_seg_prediction, instance_label,n_clusters)

                if(args.model == "bisenet"):
                    total_loss = binary_loss + instance_loss + connect_loss
                elif(args.model == "pidnet"):
                    total_loss = binary_loss + instance_loss + border_loss + connect_loss
                elif(args.model == "stdcnet"):
                    total_loss = binary_loss + instance_loss + detail_loss + connect_loss
                binary_loss_arr.append(binary_loss.detach().cpu().numpy())
                instance_loss_arr.append(instance_loss.detach().cpu().numpy())

                # iou = mean_iou.compute_iou(binary_seg_prediction,binary_label)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                scheduler.step()

                # 更新进度条的后缀信息
                pbar.set_postfix({
                    'T': f'{total_loss.item():.2f}',
                    'B': f'{binary_loss.item():.2f}',
                    'I': f'{instance_loss.item():.2f}',
                    'C': f'{connect_loss.item():.2f}',
                })

            #测试阶段
            model.eval()
            total_loss = 0.0
            total_iou = 0.0
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(test_loader):

                    images =  batch_data['image']
                    binary_label =  batch_data['binary_label_onehot'] 
                    instance_label =  batch_data['instance_seg']
                    label_seg =  batch_data['label_seg'] 
                    inverse_weights =  batch_data['inverse_weights'] 

                    start_time = time.time()
                    inverse_weights = torch.mean(inverse_weights, dim=0).to(device).squeeze(0)
                    images = images.to(device)
                    binary_label = binary_label.to(device)
                    # instance_label = instance_label.to(device)
                    label_seg = label_seg.to(device)
                    instance_label = label_seg

                    if(args.model == "bisenet" or args.model == "pidnet" or "stdcnet"):
                        result = model(images)
                        binary_seg_logits = result['binary_logits']
                        binary_seg_prediction = result['binary_pred']
                        instance_seg_prediction = result['seg_pred']
                        pixelcls = result['pixelcls']
                        con0 = result['con0']

                        mean_con0 = torch.mean(con0, dim=1, keepdim=True).squeeze()
                        binary_seg_prediction = binary_seg_prediction + mean_con0
                        binary_seg_prediction[binary_seg_prediction <= 0.9] = 0
                        binary_seg_prediction[binary_seg_prediction > 0.9] = 1
                        binary_seg_prediction = binary_seg_prediction.int()
                        
                        # 获取唯一值及其出现次数
                        unique, counts = torch.unique(binary_seg_prediction, return_counts=True)
                        # 打印结果
                        for value, count in zip(unique, counts):
                            print(f"值: {value.item()} 出现次数: {count.item()}")
                        print(f"mean_con0 = {mean_con0.shape}")
                    
                    binary_loss = loss.compute_binary_loss(binary_label,binary_seg_logits,weights = inverse_weights)
                    cls_loss = pixelcls_loss.compute_loss_V2(label_seg,pixelcls,inverse_weights)
                    n_clusters = []

                    for i in range(len(images)):
                        target_values, target_indices = torch.unique(instance_label[i], return_inverse=True)
                        mapping = {v: i for i, v in enumerate(total_classes)}
                        mapped_indices = [mapping[next((x for x in total_classes if abs(x - value.item()) < tolerance), None)] for value in target_values]
                        n_clusters.append(mapped_indices)

                    instance_loss = criterion_disc(instance_seg_prediction, instance_label, n_clusters)

                    total_loss += binary_loss + instance_loss + cls_loss

                    end_time = time.time()
                    print(f"interval time = {end_time - start_time}")

                    iou = mean_iou.compute_iou(binary_seg_prediction, binary_label)

                    end_time = time.time()
                    print(f"interval time = {end_time - start_time}")

                    total_iou += iou

            avg_loss = total_loss / len(test_loader)
            avg_iou = total_iou / len(test_loader)
            print(f"Validation Loss: {avg_loss}, Validation IoU: {avg_iou}")
            df = pd.DataFrame({'binary_loss':binary_loss_arr,'instance_loss':instance_loss_arr})
            df.to_excel('loss_values.xlsx',index = False)
            
            # the best
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(model, optimizer, epoch, best_loss, f'./weight/CRDLD_opt_end_best_model_{args.model}_{stage}_checkpoint.pth')
                # best_weights = model.state_dict()

            # the last
            if epoch == epochs - 1:
                save_checkpoint(model, optimizer, epoch, avg_loss, f'./weight/CRDLD_opt_end_last_model_{args.model}_{stage}_checkpoint.pth')

    if(stage == 2):
        ## 训练分类头
        print("正在训练模型后半段,加载权重......")
        model, optimizer, epoch, resume_loss = load_checkpoint(model, optimizer, f'./weight/CRDLD_opt_end_last_model_{args.model}_1_checkpoint.pth')
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        epoch = 0
        # 遍历训练数据

        for epoch in range(epoch,epochs):
            model.firstNet.eval()
            model.PostprocessNet.train()
            pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}')
            # 冻结后处理部分，只训练获取分割与分类特征图部分
            freeze_module(model.firstNet)
            unfreeze_module(model.PostprocessNet)
            for batch_idx, batch_data in enumerate(pbar):
                if(args.model == "bisenet"):
                    images =  batch_data['image']
                    binary_label =  batch_data['binary_label_onehot'] 
                    instance_label =  batch_data['instance_seg']
                    border_label =  batch_data['border_onehot'] 
                    label_seg =  batch_data['label_seg'] 
                    inverse_weights =  batch_data['inverse_weights'] 
                    connect_label = batch_data['connect_label']   
                    border_label = border_label.to(device)
                elif(args.model == "pidnet" or args.model == "stdcnet"):
                    images =  batch_data['image']
                    binary_label =  batch_data['binary_label_onehot'] 
                    instance_label =  batch_data['instance_seg']
                    border_label =  batch_data['border_onehot'] 
                    label_seg =  batch_data['label_seg'] 
                    inverse_weights =  batch_data['inverse_weights'] 
                    connect_label = batch_data['connect_label']   
                    border_label = border_label.to(device)

                inverse_weights = torch.mean(inverse_weights, dim=0).to(device).squeeze(0)
                # inverse_weights = torch.flip(inverse_weights, dims=[0])

                images = images.to(device)
                binary_label = binary_label.to(device)
                connect_label = connect_label.to(device)
                # instance_label = instance_label.to(device)
                label_seg = label_seg.to(device)
                instance_label = label_seg

                # if():
                #     binary_seg_logits,instance_seg_logits,binary_seg_prediction,instance_seg_prediction,pixelcls = model(images)
                if(args.model == "bisenet" or args.model == "pidnet" or args.model == "stdcnet"):
                    result = model(images)
                    binary_seg_logits = result['binary_logits']
                    binary_seg_prediction = result['binary_pred']
                    instance_seg_prediction = result['seg_pred']
                    pixelcls = result['pixelcls']
                    con0 = result['con0']

                cls_loss = pixelcls_loss.compute_loss_V2(label_seg,pixelcls,inverse_weights)

                if(args.model == "bisenet" or args.model == "pidnet" or args.model == "stdcnet"):
                    total_loss = cls_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                scheduler.step()

                # 更新进度条的后缀信息
                pbar.set_postfix({
                    'T': f'{total_loss.item():.4f}',
                })

            #测试阶段
            model.eval()
            total_loss = 0.0
            total_iou = 0.0
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(test_loader):

                    images =  batch_data['image']
                    binary_label =  batch_data['binary_label_onehot'] 
                    instance_label =  batch_data['instance_seg']
                    label_seg =  batch_data['label_seg'] 
                    inverse_weights =  batch_data['inverse_weights'] 

                    start_time = time.time()
                    inverse_weights = torch.mean(inverse_weights, dim=0).to(device).squeeze(0)
                    images = images.to(device)
                    binary_label = binary_label.to(device)
                    # instance_label = instance_label.to(device)
                    label_seg = label_seg.to(device)
                    instance_label = label_seg

                    if(args.model == "bisenet" or args.model == "pidnet" or args.model == "stdcnet"):
                        result = model(images)
                        binary_seg_logits = result['binary_logits']
                        binary_seg_prediction = result['binary_pred']
                        instance_seg_prediction = result['seg_pred']
                        pixelcls = result['pixelcls']
                        con0 = result['con0']

                        mean_con0 = torch.mean(con0, dim=1, keepdim=True).squeeze()
                        binary_seg_prediction = binary_seg_prediction + mean_con0
                        binary_seg_prediction[binary_seg_prediction <= 0.9] = 0
                        binary_seg_prediction[binary_seg_prediction > 0.9] = 1
                        binary_seg_prediction = binary_seg_prediction.int()
                        
                        # 获取唯一值及其出现次数
                        unique, counts = torch.unique(binary_seg_prediction, return_counts=True)
                        # 打印结果
                        for value, count in zip(unique, counts):
                            print(f"值: {value.item()} 出现次数: {count.item()}")
                        print(f"mean_con0 = {mean_con0.shape}")
                    
                    binary_loss = loss.compute_binary_loss(binary_label,binary_seg_logits,weights = inverse_weights)
                    cls_loss = pixelcls_loss.compute_loss_V2(label_seg,pixelcls,inverse_weights)
                    n_clusters = []

                    for i in range(len(images)):
                        target_values, target_indices = torch.unique(instance_label[i], return_inverse=True)
                        mapping = {v: i for i, v in enumerate(total_classes)}
                        mapped_indices = [mapping[next((x for x in total_classes if abs(x - value.item()) < tolerance), None)] for value in target_values]
                        n_clusters.append(mapped_indices)

                    instance_loss = criterion_disc(instance_seg_prediction, instance_label, n_clusters)

                    total_loss += binary_loss + instance_loss + cls_loss

                    end_time = time.time()
                    print(f"interval time = {end_time - start_time}")
                    iou = mean_iou.compute_iou(binary_seg_prediction, binary_label)
                    end_time = time.time()
                    print(f"interval time = {end_time - start_time}")
                    total_iou += iou

            avg_loss = total_loss / len(test_loader)
            avg_iou = total_iou / len(test_loader)
            print(f"Validation Loss: {avg_loss}, Validation IoU: {avg_iou}")
            df = pd.DataFrame({'binary_loss':binary_loss_arr,'instance_loss':instance_loss_arr})
            df.to_excel('loss_values.xlsx',index = False)
            
            # the best
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(model, optimizer, epoch, best_loss, f'./weight_256/CRDLD_opt_end_best_model_{args.model}_checkpoint.pth')
                # best_weights = model.state_dict()

            # the last
            if epoch == epochs - 1:
                save_checkpoint(model, optimizer, epoch, avg_loss, f'./weight_256/CRDLD_opt_end_last_model_{args.model}_checkpoint.pth')
                # last_weights = model.state_dict()
                # torch.save(last_weights, './weight/last_model_weights.pth')


        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--model', type=str, required=True, help="The name of the model to be used.")
    parser.add_argument('--dataset_dir', type=str, default='./CRDLD', help="The path to the dataset. Default is 'path/to/default/dataset'.")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda:0','cuda:1'], default='cuda:0', help="The device to be used (cpu or cuda).")
    parser.add_argument('--cuda',  default=False,help="Flag to use CUDA if available. Default is False.")
    parser.add_argument('--image_size', type=int, nargs=2, metavar=('width', 'height'), default=(256, 256), help="The size of the image (width height). Default is (640, 480).")
    parser.add_argument('--stage', type=int, choices=[1, 2], required=True,  help="The stage value, must be 1 or 2.")
    args = parser.parse_args()
    train(args)
