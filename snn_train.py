from tools import dataset,loss
import torch.nn.functional as F
from model import bisenetv2
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
from spikingjelly.activation_based import neuron, functional, surrogate, layer,learning
import argparse
import pandas as pd
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
        self.bisenet = bisenetv2.SNN_BiSeNetV2(n_classes=n_classes)
        self.backend = bisenetv2.LaneNetBackEnd(embedding_dims=embedding_dims, num_classes=n_classes)

    def forward(self, x):
        binary_seg_logits, instance_seg_logits = self.bisenet(x)
        binary_seg_prediction, instance_seg_prediction = self.backend(binary_seg_logits, instance_seg_logits)
        return binary_seg_logits,instance_seg_logits,binary_seg_prediction,instance_seg_prediction

def train(args):
    weight_dir = './weight/'
    os.makedirs(weight_dir, exist_ok=True)  
    epochs = 50
    num_classes = 2
    best_loss = float('inf')
    #实例分割具体类别 tolerance 代表浮点数接近程度
    total_classes = [0.0000, 0.1176, 0.4902]
    tolerance = 1e-3

    mean_iou = loss.MeanIoU(num_classes)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = CombinedModel(n_classes=2,embedding_dims = 4).to(device)
    # print(model)
    # model = bisenetv2.BiSeNetV2(n_classes=2).to(device)
    # BackEnd = bisenetv2.LaneNetBackEnd(embedding_dims = 4,num_classes=2).to(device)

    dataset_dir = './mytraining_data_example'

    train_data = dataset.My_Data(txt_path=os.path.join(dataset_dir, "train.txt"), mode='train')
    test_data = dataset.My_Data(txt_path=os.path.join(dataset_dir, "test.txt"), mode='test')
    train_loader = DataLoader(dataset=train_data, batch_size=4,drop_last=True, shuffle=True,num_workers=4,pin_memory=True)
    test_loader = DataLoader(dataset=test_data, batch_size=4,drop_last=True, shuffle=False,num_workers=4,pin_memory=True)

    criterion_disc = loss.DiscriminativeLoss(delta_var=0.5,
                                delta_dist=3.0,
                                norm=2,
                                usegpu=True)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    epoch = 0
    #恢复训练
    if args.resume:
        model, optimizer, epoch, resume_loss = load_checkpoint(model, optimizer, './weight/snn_sew_best_model_checkpoint.pth')
        print(f"已经重新恢复训练，从第{epoch}轮重新开始训练，损失值为{resume_loss}")

    # 遍历训练数据集
    binary_loss_arr = []
    instance_loss_arr = []
    for epoch in range(epoch,epochs):
        pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}')
        #训练阶段
        model.train()
        for batch_idx, batch_data in enumerate(pbar):
            images, binary_label, instance_label, inverse_weights = batch_data
            inverse_weights = torch.mean(inverse_weights, dim=0).to(device).squeeze(0)
            images = images.to(device)
            binary_label = binary_label.to(device)
            instance_label = instance_label.to(device)

            #print(f"batch_idx = {batch_idx}, image = {images.shape},binary_label = {binary_label.shape},instance_label = {instance_label.shape},")
            binary_seg_logits,instance_seg_logits,binary_seg_prediction,instance_seg_prediction = model(images)
            binary_loss = loss.compute_binary_loss(binary_label,binary_seg_logits,weights = inverse_weights)
            n_clusters = []
            for i in range(len(images)):
                target_values, target_indices = torch.unique(instance_label[i], return_inverse=True)
                mapping = {v: i for i, v in enumerate(total_classes)}
                mapped_indices = [mapping[next((x for x in total_classes if abs(x - value.item()) < tolerance), None)] for value in target_values]
                n_clusters.append(mapped_indices)
            # print(n_clusters)
            instance_loss = criterion_disc(instance_seg_prediction, instance_label,n_clusters)
            total_loss = binary_loss + instance_loss
            binary_loss_arr.append(binary_loss.cpu().detach().numpy())
            instance_loss_arr.append(instance_loss.cpu().detach().numpy())
            # iou = mean_iou.compute_iou(binary_seg_prediction,binary_label)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            functional.reset_net(model)
            # print("已经完成梯度更新")
            scheduler.step()
            pbar.set_postfix({'Total Loss': total_loss.item(), 'Binary Loss': binary_loss.item(), 'Instance Loss': instance_loss.item()})

        #测试阶段
        model.eval()
        total_loss = 0.0
        total_iou = 0.0
        with torch.no_grad():
            for images, binary_label, instance_label, inverse_weights in test_loader:
                inverse_weights = torch.mean(inverse_weights, dim=0).to(device).squeeze(0)
                images = images.to(device)
                binary_label = binary_label.to(device)
                instance_label = instance_label.to(device)

                binary_seg_logits,instance_seg_logits,binary_seg_prediction,instance_seg_prediction = model(images)
                binary_loss = loss.compute_binary_loss( binary_label,binary_seg_logits,weights = inverse_weights)

                n_clusters = []
                for i in range(len(images)):
                    target_values, target_indices = torch.unique(instance_label[i], return_inverse=True)
                    mapping = {v: i for i, v in enumerate(total_classes)}
                    mapped_indices = [mapping[next((x for x in total_classes if abs(x - value.item()) < tolerance), None)] for value in target_values]
                    n_clusters.append(mapped_indices)

                instance_loss = criterion_disc(instance_seg_prediction, instance_label,n_clusters)
                total_loss += binary_loss + instance_loss
                functional.reset_net(model)
                iou = mean_iou.compute_iou(binary_seg_prediction, binary_label)
                total_iou += iou
        avg_loss = total_loss / len(test_loader)
        avg_iou = total_iou / len(test_loader)
        print(f"Validation Loss: {avg_loss}, Validation IoU: {avg_iou}")
        df = pd.DataFrame({'binary_loss':binary_loss_arr,'instance_loss':instance_loss_arr})
        df.to_excel('snn_loss_values.xlsx',index = False)

        # 保存最佳权重
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, best_loss, './weight/snn_best_model_checkpoint.pth')
            # best_weights = model.state_dict()

        # 保存最后一轮权重
        if epoch == epochs - 1:
            save_checkpoint(model, optimizer, epoch, avg_loss, './weight/snn_last_model_checkpoint.pth')
            # last_weights = model.state_dict()
            # torch.save(last_weights, './weight/last_model_weights.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume training')
    args = parser.parse_args()
    train(args)

