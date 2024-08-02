import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image,ImageOps
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
import random

import re

def extract_frame_number(filepath):
    match = re.search(r'frame_\d+', filepath)
    if match:
        return match.group()
    return None

def random_crop_three_images(img1, img2, img3,img4, crop_size):
    """
    瀵逛笁寮犲浘鐗囪繘琛岀浉鍚岀殑闅忔満瑁佸壀

    Args:
    - img1 (PIL.Image): 绗竴寮犲浘鐗�
    - img2 (PIL.Image): 绗簩寮犲浘鐗�
    - img3 (PIL.Image): 绗笁寮犲浘鐗�
    - crop_size (tuple): 瑁佸壀灏哄锛屾牸寮忎负 (height, width)

    Returns:
    - tuple: 瑁佸壀鍚庣殑涓夊紶鍥剧墖
    """
    assert img1.size == img2.size == img3.size == img4.size, "Images must have the same size"

    w, h = img1.size
    th, tw = crop_size
    if w == tw and h == th:
        return img1, img2, img3,img4

    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)

    img1_cropped = img1.crop((x1, y1, x1 + tw, y1 + th))
    img2_cropped = img2.crop((x1, y1, x1 + tw, y1 + th))
    img3_cropped = img3.crop((x1, y1, x1 + tw, y1 + th))
    img4_cropped = img4.crop((x1, y1, x1 + tw, y1 + th))

    return img1_cropped, img2_cropped, img3_cropped, img4_cropped


class My_Data(Dataset):
    def __init__(self, txt_path, mode='train', model = "bisenet"):
        self.mode = mode
        self.crop_size = [640,384]
        if mode == 'train':
            self.transform1 = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
            ])
        else:
            self.transform1 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
            ])
        self.transform2 = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.data_list = []
        self.num_classes = 2
        self.model = model

        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_path, binary_seg_path, instance_seg_path,border_seg_path = line.strip().split()
                self.data_list.append((img_path, binary_seg_path, instance_seg_path,border_seg_path))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, binary_seg_path, instance_seg_path,border_seg_path = self.data_list[idx]
        # print(img_path)
        image = Image.open(img_path)
        binary_seg = Image.open(binary_seg_path)
        instance_seg = Image.open(instance_seg_path)
        border_seg = Image.open(border_seg_path)

        # random crop for train
        # if(self.mode == "train"):
        #     image, binary_seg, instance_seg, border_seg = random_crop_three_images(image, binary_seg, instance_seg,border_seg, self.crop_size)

        image = self.transform1(image)
        binary_seg = self.transform2(binary_seg)
        instance_seg = self.transform2(instance_seg)
        border_seg = self.transform2(border_seg)
        
        instance_seg = instance_seg.permute(1,2,0)

        instance_seg = instance_seg.squeeze(-1)


        # print(f"binary_seg = {binary_seg.shape}")
        # print(f"instance_seg = {instance_seg.shape}")
        # print(f"image = {image.shape}")

        # to_pil = ToPILImage()

        # # 淇濆瓨 binary_seg
        # binary_seg_pil = to_pil(binary_seg.squeeze(0))
        # binary_seg_pil.save('binary_seg.png')

        # # 淇濆瓨 instance_seg
        # instance_seg_pil = to_pil(instance_seg)
        # instance_seg_pil.save('instance_seg.png')

        # # 灏哛GB鏍煎紡杞崲涓築GR鏍煎紡
        # image1 = (image * 0.225 + 0.456) * 255
        # image1 = image1.numpy().astype(np.uint8)
        # image1 = np.transpose(image1, (1, 2, 0))  # 灏嗛€氶亾缁村害鏀惧埌鏈€鍚�
        # image_bgr = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('image.png', image_bgr)

        unique_labels, counts = torch.unique(binary_seg, return_counts=True)
        inverse_weights = 1.0 / torch.log(counts.float() / torch.sum(counts).float() + 1.02)
        inverse_weights = torch.cat((inverse_weights.unsqueeze(0), torch.zeros(1, self.num_classes - len(counts))), dim=1)

        #binary hot
        binary_values, binary_indices = torch.unique(binary_seg, return_inverse=True)
        binary_indices = binary_indices.view(binary_seg.shape)
        binary_label_onehot = F.one_hot(binary_indices, num_classes=self.num_classes)
        binary_label_onehot = binary_label_onehot.permute(0,3,1,2).squeeze()
        #border hot
        border_values, border_indices = torch.unique(border_seg, return_inverse=True)
        border_indices = border_indices.view(border_seg.shape)
        border_onehot = F.one_hot(binary_indices, num_classes=self.num_classes)
        border_onehot = border_onehot.permute(0,3,1,2).squeeze()

        if self.mode == 'train' and self.model == "bisenet":
            return image, binary_label_onehot, instance_seg, inverse_weights
        elif self.mode == 'train' and self.model == "pidnet":
            return image, binary_label_onehot, instance_seg, border_onehot,inverse_weights
        else:
            return image, binary_label_onehot, instance_seg, inverse_weights,img_path,binary_seg_path

# 绀轰緥鐢ㄦ硶
dataset_dir = '/home/suepr20/luofan/my_lanedetection/mytraining_data_example'

# train_data = My_Data(txt_path=os.path.join(dataset_dir, "train.txt"), mode='train')
# test_data = My_Data(txt_path=os.path.join(dataset_dir, "test.txt"), mode='test')

# print(f"train_data = {len(train_data)}")
# print(f"test_data = {len(test_data)}")
# image, binary_seg, instance_seg = train_data[0]
# print(f"train_data = {image.shape},train_data = {binary_seg.shape},train_data = {instance_seg.shape}")
