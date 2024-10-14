# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from .model_utils import BasicBlock, Bottleneck, segmenthead, DAPPM, PAPPM, PagFM, Bag, Light_Bag,binary_segmenthead,instance_segmenthead
import logging
import numpy as np

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False

class PIDNet(nn.Module):
    def __init__(self, m=2, n=3, embedding_dims=4, num_classes=2, planes=64, ppm_planes=96, head_planes=128, augment=False):
        super(PIDNet, self).__init__()
        self.augment = augment
        
        # I Branch
        self.conv1 =  nn.Sequential(
                          nn.Conv2d(3,planes,kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(planes,planes,kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                      )

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 =  self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)
        
        # P Branch
        self.compression3 = nn.Sequential(
                                          nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
                                          BatchNorm2d(planes * 2, momentum=bn_mom),
                                          )

        self.compression4 = nn.Sequential(
                                          nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
                                          BatchNorm2d(planes * 2, momentum=bn_mom),
                                          )
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)

        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)
        
        # D Branch
        if m == 2:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
            self.diff3 = nn.Sequential(
                                        nn.Conv2d(planes * 4, planes, kernel_size=3, padding=1, bias=False),
                                        BatchNorm2d(planes, momentum=bn_mom),
                                        )
            self.diff4 = nn.Sequential(
                                     nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                                     BatchNorm2d(planes * 2, momentum=bn_mom),
                                     )
            self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Light_Bag(planes * 4, planes * 4)
        else:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.layer4_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.diff3 = nn.Sequential(
                                        nn.Conv2d(planes * 4, planes * 2, kernel_size=3, padding=1, bias=False),
                                        BatchNorm2d(planes * 2, momentum=bn_mom),
                                        )
            self.diff4 = nn.Sequential(
                                     nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                                     BatchNorm2d(planes * 2, momentum=bn_mom),
                                     )
            self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4, planes * 4)
            
        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)
        
        # Prediction Head
        if self.augment:
            self.seghead_p = segmenthead(planes * 2, head_planes, num_classes)
            self.seghead_d = segmenthead(planes * 2, planes, num_classes)           
        self.binary_final_layer = binary_segmenthead(planes * 4, head_planes, num_classes,scale_factor = 8)
        self.seg_final_layer = instance_segmenthead(planes * 4, head_planes, embedding_dims,scale_factor = 8)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)
    
    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layer = block(inplanes, planes, stride, downsample, no_relu=True)
        
        return layer

    def forward(self, x):

        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))
        x_ = self.layer3_(x)
        x_d = self.layer3_d(x)
        
        x = self.relu(self.layer3(x))
        x_ = self.pag3(x_, self.compression3(x))
        x_d = x_d + F.interpolate(
                        self.diff3(x),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=algc)
        if self.augment:
            temp_p = x_
        
        x = self.relu(self.layer4(x))
        x_ = self.layer4_(self.relu(x_))
        x_d = self.layer4_d(self.relu(x_d))
        
        x_ = self.pag4(x_, self.compression4(x))
        x_d = x_d + F.interpolate(
                        self.diff4(x),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=algc)
        if self.augment:
            temp_d = x_d
            
        x_ = self.layer5_(self.relu(x_))
        x_d = self.layer5_d(self.relu(x_d))
        x = F.interpolate(
                        self.spp(self.layer5(x)),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=algc)

        binary_logits,con0 = self.binary_final_layer(self.dfm(x_, x, x_d))
        # print(f"aux_binary = {aux_binary.shape}")
        seg_logits = self.seg_final_layer(self.dfm(x_, x, x_d))

        if self.augment: 
            # x_extra_p = self.seghead_p(temp_p)
            x_extra_d = self.seghead_d(temp_d)
            return {
                'binary_logits': binary_logits,
                'seg_logits': seg_logits,
                'x_extra_d': x_extra_d,
                'con0': con0,
            }
        else:
            return {
                'binary_logits': binary_logits,
                'seg_logits': seg_logits,
                'con0': con0,
            }

def get_seg_model(cfg, imgnet_pretrained):
    
    if 's' in cfg.MODEL.NAME:
        model = PIDNet(m=2, n=3, num_classes=cfg.DATASET.NUM_CLASSES, planes=32, ppm_planes=96, head_planes=128, augment=True)
    elif 'm' in cfg.MODEL.NAME:
        model = PIDNet(m=2, n=3, num_classes=cfg.DATASET.NUM_CLASSES, planes=64, ppm_planes=96, head_planes=128, augment=True)
    else:
        model = PIDNet(m=3, n=4, num_classes=cfg.DATASET.NUM_CLASSES, planes=64, ppm_planes=112, head_planes=256, augment=True)
    
    if imgnet_pretrained:
        pretrained_state = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')['state_dict'] 
        model_dict = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(pretrained_state)
        msg = 'Loaded {} parameters!'.format(len(pretrained_state))
        logging.info('Attention!!!')
        logging.info(msg)
        logging.info('Over!!!')
        model.load_state_dict(model_dict, strict = False)
    else:
        pretrained_dict = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
        msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
        logging.info('Attention!!!')
        logging.info(msg)
        logging.info('Over!!!')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict = False)
    
    return model

def get_pred_model(name, embedding_dims, num_classes,augment):
    
    if 's' in name:
        model = PIDNet(m=2, n=3, embedding_dims=embedding_dims, num_classes=num_classes, planes=32, ppm_planes=96, head_planes=128, augment=augment)
    elif 'm' in name:
        model = PIDNet(m=2, n=3, embedding_dims=embedding_dims, num_classes=num_classes, planes=64, ppm_planes=96, head_planes=128, augment=augment)
    else:
        model = PIDNet(m=3, n=4, embedding_dims=embedding_dims, num_classes=num_classes, planes=64, ppm_planes=112, head_planes=256, augment=augment)
    
    return model

class SmoothStepFunction(nn.Module):
    def __init__(self, max, scale=10.0):
        """
        centers: the max value
        scale: Controls the sharpness of the transitions.
        """
        super(SmoothStepFunction, self).__init__()
        self.scale = scale
        self.centers = []
        self.a = []
        for i in range(max):
            self.centers.append(i+0.5)
            self.a.append(4*(i+1))

    def forward(self, x):
        output = torch.zeros_like(x)
        for i, center in enumerate(self.centers):
            sigmoid_component = self.a[i]*torch.sigmoid(self.scale*(x - center))*(1 - torch.sigmoid(self.scale*(x - center)))
            output += sigmoid_component
        return output

def create_horizontal_position_encoding(height, width, d_model):
    """
    创建仅包含水平位置编码的矩阵，并扩展到指定的维度
    :param height: 图像高度
    :param width: 图像宽度
    :param d_model: 编码的维度
    :return: 扩展到指定维度的水平位置编码矩阵
    """
    # 创建水平位置编码，从0到1递增
    x_position = torch.linspace(0, 1, width).unsqueeze(0)  # (1, width)
    
    # 将水平位置编码扩展到图像的高度
    position_enc = x_position.expand(height, width)  # (height, width)
    
    # 扩展到指定的编码维度
    position_enc = position_enc.unsqueeze(0)  # (1, height, width)
    position_enc = position_enc.expand(d_model, height, width)  # (d_model, height, width)
    
    return position_enc

class PostprocessNet(nn.Module):
    def __init__(self,max,scale):
        super(PostprocessNet, self).__init__()
        # 输入 [1, 4, 960, 544]
        self.max = max
        self.scale = scale
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=2, padding=1) # [1, 16, 480, 272]
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1) # [1, 32, 240, 136]
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1) # [1, 16, 240, 136]
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1)  # [1, 1, 120, 68]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        activation = SmoothStepFunction(max = self.max, scale = self.scale)
        x = activation(self.conv4(x))
        x = x.squeeze(1)  # 去掉通道维度 [1, 1, 120, 68] -> [1, 120, 68]
        return x

class PostprocessNet_V2(nn.Module):
    def __init__(self,max,scale):
        super(PostprocessNet_V2, self).__init__()
        # 输入 [1, 4, 960, 544]
        self.max = max
        self.scale = scale
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=2, padding=1) # [1, 16, 480, 272]
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1) # [1, 32, 240, 136]
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1) # [1, 16, 240, 136]
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=self.max, kernel_size=3, stride=2, padding=1)  # [1, 4, 120, 68]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x

class PostprocessNet_V3(nn.Module):
    def __init__(self,height,width,d_model,max,device,mode):
        super(PostprocessNet_V3, self).__init__()
        # 输入 [1, 4, 960, 544]
        self.device = device
        self.mode = mode
        self.height, self.width, self.d_model,self.max = height,width,d_model,max
        ## 位置参数可以被训练
        self.position_enc = nn.Parameter(create_horizontal_position_encoding(self.height, self.width, self.d_model).to(device), requires_grad=True)
        # 卷积层1 + 批归一化层 + ReLU激活函数 + 池化层
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=48, kernel_size=3, stride=2, padding=1)  # [1, 32, h, w]
        self.bn1 = nn.BatchNorm2d(48)  # 批归一化层
        self.relu1 = nn.ReLU()  # ReLU激活函数
        
        # 卷积层2 + 批归一化层 + ReLU激活函数 + 池化层
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=3, stride=2, padding=1)  # [1, max_class, h/8, w/8]
        self.bn2 = nn.BatchNorm2d(32)  # 批归一化层
        self.relu2 = nn.ReLU()  # ReLU激活函数

        # # 卷积层3 + 批归一化层 + ReLU激活函数
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1)  # [1, 16, 120, 68]
        self.bn3 = nn.BatchNorm2d(16)  # 批归一化层
        self.relu3 = nn.ReLU()  # ReLU激活函数

        # # 卷积层4，输出的通道数改为4
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=self.max, kernel_size=3, stride=1, padding=1)  # [1, 4, 120, 68]
        self.bn4 = nn.BatchNorm2d(self.max)  # 批归一化层
        self.relu4 = nn.ReLU()  # ReLU激活函数

    def forward(self, x):
        #x = torch.cat((x, self.position_enc[0].unsqueeze(0)), dim=1)
        # if(self.mode == "eval"):
        #     self.eval_position_enc = F.interpolate(self.position_enc.unsqueeze(0), size=(480, 480), mode='bilinear', align_corners=False).to(self.device)
        #     x = x + self.eval_position_enc
        # else:
        x = x + self.position_enc
        x = self.conv1(x)
        y = x
        x = self.bn1(x)    # 批归一化层
        x = self.relu1(x)  # 激活函数
        
        x = self.conv2(x)
        x = self.bn2(x)    # 批归一化层
        x = self.relu2(x)  # 激活函数
        
        x = self.conv3(x)
        x = self.bn3(x)  # 批归一化层
        x = self.relu3(x)  # 激活函数

        x = self.conv4(x)
        x = self.bn4(x)  # 批归一化层
        x = self.relu4(x)  # 激活函数        
        return x,y



if __name__ == '__main__':
    
    # 示例用法
    activation = SmoothStepFunction(4)
    input_tensor = torch.tensor([-2.0,-1.0, 0.3, 0.7, 1.2, 1.8, 2.5, 3.0, 3.7,4.5,7], requires_grad=True)
    output_tensor = activation(input_tensor)

    print("Input Tensor:", input_tensor)
    print("Output Tensor:", output_tensor)

    # 如果需要，计算梯度
    output_tensor.sum().backward()
    print("Gradients:", input_tensor.grad)
    # Comment batchnorms here and in model_utils before testing speed since the batchnorm could be integrated into conv operation
    # (do not comment all, just the batchnorm following its corresponding conv layer)
    # device = torch.device('cuda')
    # model = get_pred_model(name='pidnet_s', num_classes=19)
    # model.eval()
    # model.to(device)
    # iterations = None
    
    # input = torch.randn(1, 3, 544, 960).cuda()
    # with torch.no_grad():
    #     for _ in range(10):
    #         output = model(input)
    #         print(output[0].shape)
    #         print(output[1].shape)
    
    #     if iterations is None:
    #         elapsed_time = 0
    #         iterations = 100
    #         while elapsed_time < 1:
    #             torch.cuda.synchronize()
    #             torch.cuda.synchronize()
    #             t_start = time.time()
    #             for _ in range(iterations):
    #                 model(input)
    #             torch.cuda.synchronize()
    #             torch.cuda.synchronize()
    #             elapsed_time = time.time() - t_start
    #             iterations *= 2
    #         FPS = iterations / elapsed_time
    #         iterations = int(FPS * 6)
    
    #     print('=========Speed Testing=========')
    #     torch.cuda.synchronize()
    #     torch.cuda.synchronize()
    #     t_start = time.time()
    #     for _ in range(iterations):
    #         model(input)
    #     torch.cuda.synchronize()
    #     torch.cuda.synchronize()
    #     elapsed_time = time.time() - t_start
    #     latency = elapsed_time / iterations * 1000
    # torch.cuda.empty_cache()
    # FPS = 1000 / latency
    # print(FPS)


    
    
    


