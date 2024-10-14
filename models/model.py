
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
from pidnet_model import pidnet
from bisenet_model import bisenetv2
from stdc_model import model_stages
import time

class CombinedModel(nn.Module):
    def __init__(self, n_classes, embedding_dims,height,width,max_class,mode):
        super(CombinedModel, self).__init__()
        self.mode = mode
        self.firstNet = bisenetv2.firstNet(n_classes,embedding_dims,aux_mode=mode)
        self.PostprocessNet = pidnet.PostprocessNet_V3(height=height,width=width,d_model=embedding_dims,max = max_class,device = "cuda:0",mode = mode)

    def forward(self, x):
        result = self.firstNet(x)
        binary_seg_logits = result['binary_logits']
        instance_seg_logits = result['seg_logits']
        con0 = result['con0']

        binary_seg_prediction = torch.argmax(binary_seg_logits, dim=1)
        mean_con0 = torch.mean(con0, dim=1, keepdim=True).squeeze()
        binary_seg_prediction = binary_seg_prediction + 1 - mean_con0
        binary_seg_prediction[binary_seg_prediction <= 0.5] = 0
        binary_seg_prediction[binary_seg_prediction > 0.5] = 1
        binary_seg_prediction = binary_seg_prediction.int()

        instance_seg_prediction = instance_seg_logits
        #实例输出
        # fuse
        features = binary_seg_prediction.unsqueeze(dim = 1) * instance_seg_logits
        
        # get correct pixels
        pixelcls,middle_map = self.PostprocessNet(features)


        if self.mode == 'train':
            return {
                'binary_logits': binary_seg_logits,
                'binary_pred': binary_seg_prediction,
                'seg_logits': instance_seg_logits,
                'seg_pred': instance_seg_prediction,
                'pixelcls': pixelcls,
                'con0': con0,
            }
        
        else:
            return {
                'binary_logits': binary_seg_logits,
                'binary_pred':binary_seg_prediction,
                'seg_logits': instance_seg_logits,
                'seg_pred': instance_seg_prediction,
                'pixelcls': pixelcls,
                'middle_map':middle_map,
                'con0': con0,
            }
        # if self.mode == "train":
        #     return binary_seg_logits,instance_seg_logits,binary_seg_prediction,instance_seg_prediction,pixelcls
        # else:
        #     return binary_seg_logits,instance_seg_logits,binary_seg_prediction,instance_seg_prediction,pixelcls,middle_map

class Pidnet_Model(nn.Module):
    def __init__(self, n_classes, embedding_dims,height,width,max_class,mode):
        super(Pidnet_Model, self).__init__()
        self.mode = mode
        self.firstNet = pidnet.get_pred_model(name='pidnet_s', embedding_dims=embedding_dims, num_classes=n_classes,augment = True)
        self.PostprocessNet = pidnet.PostprocessNet_V3(height=height,width=width,d_model=embedding_dims,max = max_class,device = "cuda:0",mode = mode)
    def forward(self, x):
        result = self.firstNet(x)

        binary_seg_logits = result['binary_logits']
        instance_seg_logits = result['seg_logits']
        x_extra_d = result['x_extra_d']
        con0 = result['con0']

        binary_seg_prediction = torch.argmax(binary_seg_logits, dim=1)
        mean_con0 = torch.mean(con0, dim=1, keepdim=True).squeeze()
        binary_seg_prediction = binary_seg_prediction + 1 - mean_con0
        binary_seg_prediction[binary_seg_prediction <= 0.5] = 0
        binary_seg_prediction[binary_seg_prediction > 0.5] = 1
        binary_seg_prediction = binary_seg_prediction.int()

        instance_seg_prediction = instance_seg_logits
        #实例输出
        # fuse
        features = binary_seg_prediction.unsqueeze(dim = 1) * instance_seg_logits
        
        # get correct pixels
        pixelcls,middle_map = self.PostprocessNet(features)


        if self.mode == 'train':
            return {
                'binary_logits': binary_seg_logits,
                'binary_pred':binary_seg_prediction,
                'seg_logits': instance_seg_logits,
                'seg_pred': instance_seg_prediction,
                'x_extra_d': x_extra_d,
                'pixelcls': pixelcls,
                'con0': con0,
            }
        
        else:
            return {
                'binary_logits': binary_seg_logits,
                'binary_pred':binary_seg_prediction,
                'seg_logits': instance_seg_logits,
                'seg_pred': instance_seg_prediction,
                'x_extra_d': x_extra_d,
                'pixelcls': pixelcls,
                'middle_map':middle_map,
                'con0': con0,
            }

class STDC_Model(nn.Module):
    def __init__(self, n_classes, embedding_dims,height,width,max_class,mode):
        super(STDC_Model, self).__init__()
        self.mode = mode
        self.firstNet = model_stages.BiSeNet('STDCNet813', n_classes=n_classes, embedding_dims=embedding_dims,mode = self.mode)
        self.PostprocessNet = pidnet.PostprocessNet_V3(height=height,width=width,d_model=embedding_dims,max = max_class,device = "cuda:0",mode = mode)
    def forward(self, x):
        result = self.firstNet(x)
        binary_seg_logits = result['binary_logits']
        instance_seg_logits = result['seg_logits']
        con0 = result['con0']
        if(self.mode == "train"):
            x_extra_d = result['x_extra_d']


        binary_seg_prediction = torch.argmax(binary_seg_logits, dim=1)
        mean_con0 = torch.mean(con0, dim=1, keepdim=True).squeeze()
        binary_seg_prediction = binary_seg_prediction + 1 - mean_con0
        binary_seg_prediction[binary_seg_prediction <= 0.5] = 0
        binary_seg_prediction[binary_seg_prediction > 0.5] = 1
        binary_seg_prediction = binary_seg_prediction.int()

        instance_seg_prediction = instance_seg_logits
        #实例输出
        # fuse
        features = binary_seg_prediction.unsqueeze(dim = 1) * instance_seg_logits
        
        # get correct pixels
        pixelcls,middle_map = self.PostprocessNet(features)


        if self.mode == 'train':
            return {
                'binary_logits': binary_seg_logits,
                'binary_pred':binary_seg_prediction,
                'seg_logits': instance_seg_logits,
                'seg_pred': instance_seg_prediction,
                'x_extra_d': x_extra_d,
                'pixelcls': pixelcls,
                'con0': con0,
            }
        
        else:
            return {
                'binary_logits': binary_seg_logits,
                'binary_pred':binary_seg_prediction,
                'seg_logits': instance_seg_logits,
                'seg_pred': instance_seg_prediction,
                'pixelcls': pixelcls,
                'middle_map':middle_map,
                'con0': con0,
            }