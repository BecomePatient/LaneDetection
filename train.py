from tools import dataset,loss
from model import bisenetv2
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
from pidnet_model import pidnet
import time
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
    def forward(self, x):
        binary_seg_logits, instance_seg_logits,x_extra_d = self.pidnet(x)

        #ʵ�����
        instance_seg_prediction = instance_seg_logits
        #��ֵ�����
        #input shape
        binary_score = F.softmax(binary_seg_logits, dim=1)

        # get lanes
        # binary_mask = binary_score[:,1].unsqueeze(1)

        # expand
        # print(f"binary_mask = {binary_mask.shape}")
        # binary_mask = binary_mask.expand(bs,embeding,h,w)  
        # print(f"binary_mask = {binary_mask.shape}")
        # fuse
        # features = binary_mask * instance_seg_logits
        # print(f"features = {features.shape}")
        # # get correct pixels
        # pixelcls = self.PostprocessNet(features)
        # print(f"pixelcls = {pixelcls.shape}")

        binary_seg_prediction = torch.argmax(binary_score, dim=1)
        return binary_seg_logits,instance_seg_logits,binary_seg_prediction,instance_seg_prediction,x_extra_d

def train(args):
    weight_dir = './weight/'
    os.makedirs(weight_dir, exist_ok=True)  
    epochs = 50
    num_classes = 2
    best_loss = float('inf')
    #ʵ���ָ������� tolerance ���������ӽ���??
    total_classes = [0.0000, 0.1176, 0.4902]
    tolerance = 1e-3
    mean_iou = loss.MeanIoU(num_classes)
    device = args.device
    if(args.model == "bisenet"):
        model = CombinedModel(n_classes=2,embedding_dims = 4).to(device)
    elif(args.model == "pidnet"):
        model = Pidnet_Model(n_classes=2,embedding_dims = 4).to(device)
    else:
        raise ValueError(f"Unknown model name: {args.model}. Please choose 'bisenet' or 'pidnet'.")
    print(model)
    dataset_dir = args.dataset_dir
    print(dataset_dir)
    train_data = dataset.My_Data(txt_path=os.path.join(dataset_dir, "train.txt"), mode='train',model = args.model)
    # train_data = torch.utils.data.Subset(train_data, range(8))
    
    test_data = dataset.My_Data(txt_path=os.path.join(dataset_dir, "test.txt"), mode='test',model = args.model)
    # test_data = torch.utils.data.Subset(test_data, range(8))
    
    train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True,num_workers=8,pin_memory=True)
    test_loader = DataLoader(dataset=test_data, batch_size=8, shuffle=False,num_workers=8,pin_memory=True)

    criterion_disc = loss.DiscriminativeLoss(delta_var=0.5,
                                delta_dist=3.0,
                                norm=2,
                                usegpu=device
    )
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    epoch = 0
    #�ָ�ѵ��
    if args.resume:
        model, optimizer, epoch, resume_loss = load_checkpoint(model, optimizer, './weight/best_model_checkpoint.pth')
        print(f"�Ѿ����»ָ�ѵ�����ӵ�{epoch}�����¿�ʼѵ������ʧֵΪ{resume_loss}")
    binary_loss_arr = []
    instance_loss_arr = []
    # ����ѵ������
    for epoch in range(epoch,epochs):
        pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}')
        #ѵ���׶�
        model.train()
        for batch_idx, batch_data in enumerate(pbar):
            if(args.model == "bisenet"):
                images, binary_label, instance_label, inverse_weights = batch_data
            elif(args.model == "pidnet"):
                images, binary_label, instance_label, border_label, inverse_weights = batch_data
            inverse_weights = torch.mean(inverse_weights, dim=0).to(device).squeeze(0)

            images = images.to(device)
            binary_label = binary_label.to(device)
            instance_label = instance_label.to(device)
            border_label = border_label.to(device)

            if(args.model == "bisenet"):
                binary_seg_logits,instance_seg_logits,binary_seg_prediction,instance_seg_prediction = model(images)
            elif(args.model == "pidnet"):
                binary_seg_logits,instance_seg_logits,binary_seg_prediction,instance_seg_prediction,x_extra_d = model(images)
                border_loss = loss.compute_border_loss(binary_label,x_extra_d,weights = inverse_weights)
            binary_loss = loss.compute_binary_loss(binary_label,binary_seg_logits,weights = inverse_weights)
            n_clusters = []

            for i in range(len(images)):
                target_values, target_indices = torch.unique(instance_label[i], return_inverse=True)
                mapping = {v: i for i, v in enumerate(total_classes)}
                mapped_indices = [mapping[next((x for x in total_classes if abs(x - value.item()) < tolerance), None)] for value in target_values]
                n_clusters.append(mapped_indices)
            print(n_clusters)

            instance_loss = criterion_disc(instance_seg_prediction, instance_label,n_clusters)

            if(args.model == "bisenet"):
                total_loss = binary_loss + instance_loss 
            elif(args.model == "pidnet"):
                total_loss = binary_loss + instance_loss + border_loss
            binary_loss_arr.append(binary_loss.detach().cpu().numpy())
            instance_loss_arr.append(instance_loss.detach().cpu().numpy())

            # iou = mean_iou.compute_iou(binary_seg_prediction,binary_label)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({'Total Loss': total_loss.item(), 'border_loss':border_loss.item(),'Instance Loss': instance_loss.item()})

        #���Խ׶�
        model.eval()
        total_loss = 0.0
        total_iou = 0.0
        with torch.no_grad():
            for images, binary_label, instance_label, inverse_weights,_,_ in test_loader:
                inverse_weights = torch.mean(inverse_weights, dim=0).to(device).squeeze(0)
                images = images.to(device)
                binary_label = binary_label.to(device)
                instance_label = instance_label.to(device)

                if(args.model == "bisenet"):
                    binary_seg_logits,instance_seg_logits,binary_seg_prediction,instance_seg_prediction = model(images)
                elif(args.model == "pidnet"):
                    binary_seg_logits,instance_seg_logits,binary_seg_prediction,instance_seg_prediction,_ = model(images)
                binary_loss = loss.compute_binary_loss(binary_label,binary_seg_logits,weights = inverse_weights)

                n_clusters = []

                for i in range(len(images)):
                    target_values, target_indices = torch.unique(instance_label[i], return_inverse=True)
                    mapping = {v: i for i, v in enumerate(total_classes)}
                    mapped_indices = [mapping[next((x for x in total_classes if abs(x - value.item()) < tolerance), None)] for value in target_values]
                    n_clusters.append(mapped_indices)

                instance_loss = criterion_disc(instance_seg_prediction, instance_label, n_clusters)

                total_loss += binary_loss + instance_loss

                iou = mean_iou.compute_iou(binary_seg_prediction, binary_label)
                total_iou += iou
        avg_loss = total_loss / len(test_loader)
        avg_iou = total_iou / len(test_loader)
        print(f"Validation Loss: {avg_loss}, Validation IoU: {avg_iou}")
        df = pd.DataFrame({'binary_loss':binary_loss_arr,'instance_loss':instance_loss_arr})
        df.to_excel('loss_values.xlsx',index = False)
        
        # the best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, best_loss, f'./weight/opt_best_model_{args.model}_checkpoint.pth')
            # best_weights = model.state_dict()

        # the last
        if epoch == epochs - 1:
            save_checkpoint(model, optimizer, epoch, avg_loss, f'./weight/opt_last_model_{args.model}checkpoint.pth')
            # last_weights = model.state_dict()
            # torch.save(last_weights, './weight/last_model_weights.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--model', type=str, required=True, help="The name of the model to be used.")
    parser.add_argument('--dataset_dir', type=str, default='/home/suepr20/luofan/my_lanedetection/mytraining_data_example', help="The path to the dataset. Default is 'path/to/default/dataset'.")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda:0','cuda:1'], default='cpu', help="The device to be used (cpu or cuda).")
    args = parser.parse_args()
    train(args)
