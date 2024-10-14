
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo
from spikingjelly.activation_based import neuron, functional, surrogate, layer,learning
from thop import profile,clever_format
BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False

backbone_url = 'https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/backbone_v2.pth'

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat

class SNN_ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(SNN_ConvBNReLU, self).__init__()
        self.conv = layer.Conv2d(in_chan, out_chan, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=groups, bias=bias, dilation=dilation)
        self.bn = layer.BatchNorm2d(out_chan,affine=False)
        self.sn = neuron.IFNode(surrogate_function=surrogate.ATan())

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.sn(feat)
        return feat


class UpSample(nn.Module):

    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)



class DetailBranch(nn.Module):

    def __init__(self):
        super(DetailBranch, self).__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(3, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2),
            ConvBNReLU(128, 128, 3, stride=1),
            ConvBNReLU(128, 128, 3, stride=1),
        )

    def forward(self, x):
        feat = self.S1(x)
        feat = self.S2(feat)
        feat = self.S3(feat)
        return feat


class SNN_DetailBranch(nn.Module):
    def __init__(self):
        super(SNN_DetailBranch, self).__init__()
        self.S1 = nn.Sequential(
            SNN_ConvBNReLU(3, 64, 3, stride=2),
            SNN_ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S2 = nn.Sequential(
            SNN_ConvBNReLU(64, 64, 3, stride=2),
            SNN_ConvBNReLU(64, 64, 3, stride=1),
            SNN_ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S3 = nn.Sequential(
            SNN_ConvBNReLU(64, 128, 3, stride=2),
            SNN_ConvBNReLU(128, 128, 3, stride=1),
            SNN_ConvBNReLU(128, 128, 3, stride=1),
        )

    def forward(self, x):
        feat = self.S1(x)
        feat = self.S2(feat)
        feat = self.S3(feat)
        return feat

class StemBlock(nn.Module):
    def __init__(self):
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(3, 16, 3, stride=2)
        self.left = nn.Sequential(
            ConvBNReLU(16, 8, 1, stride=1, padding=0),
            ConvBNReLU(8, 16, 3, stride=2),
        )
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(32, 16, 3, stride=1)
    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat

class SNN_StemBlock(nn.Module):
    def __init__(self):
        super(SNN_StemBlock, self).__init__()
        self.conv = SNN_ConvBNReLU(3, 16, 3, stride=2)
        self.left = nn.Sequential(
            SNN_ConvBNReLU(16, 8, 1, stride=1, padding=0),
            SNN_ConvBNReLU(8, 16, 3, stride=2),
        )
        #注意这里的最大值池化模块
        self.right = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.right = nn.MaxPool2d(
        #     kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = SNN_ConvBNReLU(32, 16, 3, stride=1)
    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat

class CEBlock(nn.Module):

    def __init__(self):
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(128)
        self.conv_gap = ConvBNReLU(128, 128, 1, stride=1, padding=0)
        #TODO: in paper here is naive conv2d, no bn-relu
        self.conv_last = ConvBNReLU(128, 128, 3, stride=1)

    def forward(self, x):
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        return feat

class SNN_CEBlock(nn.Module):

    def __init__(self):
        super(SNN_CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(128)
        self.conv_gap = SNN_ConvBNReLU(128, 128, 1, stride=1, padding=0)
        #TODO: in paper here is naive conv2d, no bn-relu
        self.conv_last = SNN_ConvBNReLU(128, 128, 3, stride=1)

    def forward(self, x):
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        return feat

class GELayerS1(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS1, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True), # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        feat = self.relu(feat)
        return feat


class SNN_GELayerS1(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(SNN_GELayerS1, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = SNN_ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv = nn.Sequential(
            layer.Conv2d(in_chan, mid_chan, kernel_size=3, stride=1,
                        padding=1, groups=in_chan, bias=False),
            layer.BatchNorm2d(mid_chan,affine=False),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )
        self.conv2 = nn.Sequential(
            layer.Conv2d(mid_chan, out_chan, kernel_size=1, stride=1,padding=0, bias=False),
            layer.BatchNorm2d(out_chan,affine=False),
            # nn.Conv2d(
            #     mid_chan, out_chan, kernel_size=1, stride=1,
            #     padding=0, bias=False),
            # nn.BatchNorm2d(out_chan),
        )
        ##这里没有使用到，需要注意
        # self.conv2[1].last_bn = True
        self.sn = neuron.IFNode(surrogate_function=surrogate.ATan())

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        feat = self.sn(feat)
        return feat

#改进后的SEW_Resnet
class SEW_SNN_GELayerS1(nn.Module):
    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(SEW_SNN_GELayerS1, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = SNN_ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv = nn.Sequential(
            layer.Conv2d(in_chan, mid_chan, kernel_size=3, stride=1,
                        padding=1, groups=in_chan, bias=False),
            layer.BatchNorm2d(mid_chan,affine=False),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )
        self.conv2 = nn.Sequential(
            layer.Conv2d(mid_chan, out_chan, kernel_size=1, stride=1,padding=0, bias=False),
            layer.BatchNorm2d(out_chan,affine=False),
            # nn.Conv2d(
            #     mid_chan, out_chan, kernel_size=1, stride=1,
            #     padding=0, bias=False),
            # nn.BatchNorm2d(out_chan),
        )
        ##这里没有使用到，需要注意
        # self.conv2[1].last_bn = True
        self.sn = neuron.IFNode(surrogate_function=surrogate.ATan())

    def forward(self, x):
        feat = self.sn(x)
        feat = self.conv1(feat)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        return feat

class GELayerS2(nn.Module):
    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=2,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True), # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chan, in_chan, kernel_size=3, stride=2,
                    padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(in_chan),
                nn.Conv2d(
                    in_chan, out_chan, kernel_size=1, stride=1,
                    padding=0, bias=False),
                nn.BatchNorm2d(out_chan),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat

class SEW_SNN_GELayerS2(nn.Module):
    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(SEW_SNN_GELayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = SNN_ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv1 = nn.Sequential(
            layer.Conv2d(in_chan, mid_chan, kernel_size=3, stride=2,padding=1,groups=in_chan,bias=False),
            layer.BatchNorm2d(mid_chan,affine=False),
        )
        self.dwconv2 = nn.Sequential(
            layer.Conv2d(mid_chan, mid_chan, kernel_size=3, stride=1,padding=1, groups=mid_chan, bias=False),
            layer.BatchNorm2d(mid_chan,affine=False),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )
        self.conv2 = nn.Sequential(
            layer.Conv2d(mid_chan, out_chan, kernel_size=1, stride=1,padding=0,bias=False),
            layer.BatchNorm2d(out_chan,affine=False),
        )
        #self.conv2[1].last_bn = True
        ####这里需要更改
        self.shortcut = nn.Sequential(
                layer.Conv2d(in_chan, in_chan, kernel_size=1, stride=1,padding=0,bias=False),
                layer.BatchNorm2d(in_chan,affine=False),
        )
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.sn1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.sn2 = neuron.IFNode(surrogate_function=surrogate.ATan())

    def forward(self, x):
        feat = self.sn1(x)
        feat = self.conv1(feat)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut1 = self.maxpool(x)
        # print(shortcut1.shape)
        shortcut2 = self.sn2(shortcut1)
        shortcut2 = self.shortcut(shortcut2)
        # print(shortcut2.shape)
        shortcut =  torch.cat((shortcut1, shortcut2), dim=1)
        # print(shortcut.shape)
        # shortcut = self.shortcut(x)
        feat = feat + shortcut

        return feat

class SNN_GELayerS2(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(SNN_GELayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = SNN_ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv1 = nn.Sequential(
            layer.Conv2d(in_chan, mid_chan, kernel_size=3, stride=2,padding=1,groups=in_chan,bias=False),
            layer.BatchNorm2d(mid_chan,affine=False),
        )
        self.dwconv2 = nn.Sequential(
            layer.Conv2d(mid_chan, mid_chan, kernel_size=3, stride=1,padding=1, groups=mid_chan, bias=False),
            layer.BatchNorm2d(mid_chan,affine=False),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )
        self.conv2 = nn.Sequential(
            layer.Conv2d(mid_chan, out_chan, kernel_size=1, stride=1,padding=0,bias=False),
            layer.BatchNorm2d(out_chan,affine=False),
        )
        #self.conv2[1].last_bn = True
        self.shortcut = nn.Sequential(
                layer.Conv2d(in_chan, in_chan, kernel_size=3, stride=2,padding=1,groups=in_chan,bias=False),
                layer.BatchNorm2d(in_chan,affine=False),
                layer.Conv2d(in_chan, out_chan, kernel_size=1, stride=1,padding=0,bias=False),
                layer.BatchNorm2d(out_chan,affine=False),
        )
        self.sn = neuron.IFNode(surrogate_function=surrogate.ATan())

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.sn(feat)
        return feat


class SegmentBranch(nn.Module):

    def __init__(self):
        super(SegmentBranch, self).__init__()
        self.S1S2 = StemBlock()
        self.S3 = nn.Sequential(
            GELayerS2(16, 32),
            GELayerS1(32, 32),
        )
        self.S4 = nn.Sequential(
            GELayerS2(32, 64),
            GELayerS1(64, 64),
        )
        self.S5_4 = nn.Sequential(
            GELayerS2(64, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
        )
        self.S5_5 = CEBlock()

    def forward(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5_4 = self.S5_4(feat4)
        feat5_5 = self.S5_5(feat5_4)
        return feat5_5


class SNN_SegmentBranch(nn.Module):

    def __init__(self):
        super(SNN_SegmentBranch, self).__init__()
        self.S1S2 = SNN_StemBlock()
        self.S3 = nn.Sequential(
            SNN_GELayerS2(16, 32),
            SNN_GELayerS1(32, 32),
        )
        self.S4 = nn.Sequential(
            SNN_GELayerS2(32, 64),
            SNN_GELayerS1(64, 64),
        )
        self.S5_4 = nn.Sequential(
            SNN_GELayerS2(64, 128),
            SNN_GELayerS1(128, 128),
            SNN_GELayerS1(128, 128),
            SNN_GELayerS1(128, 128),
        )
        self.S5_5 = SNN_CEBlock()

    def forward(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5_4 = self.S5_4(feat4)
        feat5_5 = self.S5_5(feat5_4)
        return feat5_5

# class SNN_SegmentBranch(nn.Module):

#     def __init__(self):
#         super(SNN_SegmentBranch, self).__init__()
#         self.S1S2 = SNN_StemBlock()
#         self.S3 = nn.Sequential(
#             SEW_SNN_GELayerS2(16, 32),
#             SEW_SNN_GELayerS1(32, 32),
#         )
#         self.S4 = nn.Sequential(
#             SEW_SNN_GELayerS2(32, 64),
#             SEW_SNN_GELayerS1(64, 64),
#         )
#         self.S5_4 = nn.Sequential(
#             SEW_SNN_GELayerS2(64, 128),
#             SEW_SNN_GELayerS1(128, 128),
#             SEW_SNN_GELayerS1(128, 128),
#             SEW_SNN_GELayerS1(128, 128),
#         )
#         self.S5_5 = SNN_CEBlock()

#     def forward(self, x):
#         feat2 = self.S1S2(x)
#         feat3 = self.S3(feat2)
#         feat4 = self.S4(feat3)
#         feat5_4 = self.S5_4(feat4)
#         feat5_5 = self.S5_5(feat5_4)
#         return feat5_5


class BGALayer(nn.Module):
    def __init__(self):
        super(BGALayer, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=2,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.up1 = nn.Upsample(scale_factor=4)
        self.up2 = nn.Upsample(scale_factor=4)
        ##TODO: does this really has no relu?
        self.conv = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), # not shown in paper
        )

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = self.up1(right1)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = self.up2(right)
        out = self.conv(left + right)
        return out


class SNN_BGALayer(nn.Module):
    def __init__(self):
        super(SNN_BGALayer, self).__init__()
        self.left1 = nn.Sequential(
            layer.Conv2d(128, 128, kernel_size=3, stride=1,padding=1,groups=128,bias=False),
            layer.BatchNorm2d(128,affine=False),
            layer.Conv2d(128, 128, kernel_size=1, stride=1,padding=0,bias=False),
        )
        self.left2 = nn.Sequential(
            layer.Conv2d(128, 128, kernel_size=3, stride=2,padding=1,bias=False),
            layer.BatchNorm2d(128,affine=False),
            layer.AdaptiveAvgPool2d((1, 1))
            #nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),
        )
        self.right1 = nn.Sequential(
            layer.Conv2d(128, 128, kernel_size=3, stride=1,padding=1,bias=False),
            layer.BatchNorm2d(128,affine=False),
        )
        self.right2 = nn.Sequential(
            layer.Conv2d(128, 128, kernel_size=3, stride=1,padding=1,groups=128,bias=False),
            layer.BatchNorm2d(128,affine=False),
            layer.Conv2d(128, 128, kernel_size=1, stride=1,padding=0,bias=False),
        )
        self.up1 = nn.Upsample(scale_factor=4)
        self.up2 = nn.Upsample(scale_factor=4)
        ##TODO: does this really has no relu?
        self.conv = nn.Sequential(
            layer.Conv2d(128, 128, kernel_size=3, stride=1,padding=1, bias=False),
            layer.BatchNorm2d(128,affine=False),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )
    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = self.up1(right1)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = self.up2(right)
        out = self.conv(left + right)
        return out

class SegmentHead(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.up_factor = up_factor

        out_chan = n_classes
        mid_chan2 = up_factor * up_factor if aux else mid_chan
        up_factor = up_factor // 2 if aux else up_factor
        self.conv_out = nn.Sequential(
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBNReLU(mid_chan, mid_chan2, 3, stride=1)
                ) if aux else nn.Identity(),
            nn.Conv2d(mid_chan2, out_chan, 1, 1, 0, bias=True),
            nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat

class Binary_SegmentHead(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
        super(Binary_SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.up_factor = up_factor

        out_chan = n_classes
        mid_chan2 = up_factor * up_factor if aux else mid_chan
        up_factor = up_factor // 2 if aux else up_factor
        self.conv_out = nn.Sequential(
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBNReLU(mid_chan, mid_chan2, 3, stride=1)
                ) if aux else nn.Identity(),
            nn.Conv2d(mid_chan2, out_chan, 1, 1, 0, bias=True),
            nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat

class SELayer(nn.Module):
    def __init__(self, channel, reduction=3):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.out =  nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = self.out(x * y.expand_as(x))
        return out

class binary_segmenthead(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None,num_neighbor = 9):
        super(binary_segmenthead, self).__init__()

        self.seg_branch = nn.Sequential(BatchNorm2d(inplanes, momentum=bn_mom),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False),
                                BatchNorm2d(interplanes, momentum=bn_mom),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True),
                                nn.Sigmoid(),
                                )
        
        self.scale_factor = scale_factor
        # connect branch
        self.connect_branch = nn.Sequential(nn.Conv2d(inplanes, interplanes, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(interplanes, num_neighbor, 3, padding=1, dilation=1),
                                    )
        self.se = SELayer(num_neighbor)
                                    

    def forward(self, x):
        binary_seg = self.seg_branch(x)
        if self.scale_factor is not None:
            height = binary_seg.shape[-2] * self.scale_factor
            width = binary_seg.shape[-1] * self.scale_factor
            binary_seg = F.interpolate(binary_seg,
                        size=[height, width],
                        mode='bilinear', align_corners=algc)
        # print(f"out = {out.shape}")
        connet_branch = self.connect_branch(x)
        connet0 = self.se(connet_branch)
        if self.scale_factor is not None:
            height = connet0.shape[-2] * self.scale_factor
            width = connet0.shape[-1] * self.scale_factor
            connet0 = F.interpolate(connet0,
                        size=[height, width],
                        mode='bilinear', align_corners=algc)
        return binary_seg,connet0

# class Binary_SegmentHead(nn.Module):

#     def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
#         super(Binary_SegmentHead, self).__init__()
#         self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
#         self.drop = nn.Dropout(0.1)
#         self.up_factor = up_factor

#         out_chan = n_classes
#         mid_chan2 = up_factor * up_factor if aux else mid_chan
#         up_factor = up_factor // 2 if aux else up_factor
#         self.conv_out = nn.Sequential(
#             nn.Sequential(
#                 nn.Upsample(scale_factor=2),
#                 ConvBNReLU(mid_chan, mid_chan2, 3, stride=1)
#                 ) if aux else nn.Identity(),
#             nn.Conv2d(mid_chan2, out_chan, 1, 1, 0, bias=True),
#             nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False)
#         )

#     def forward(self, x):
#         feat = self.conv(x)
#         feat = self.drop(feat)
#         feat = self.conv_out(feat)
#         return feat

class Instance_SegmentHead(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
        super(Instance_SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.up_factor = up_factor

        out_chan = n_classes
        mid_chan2 = up_factor * up_factor if aux else mid_chan
        up_factor = up_factor // 2 if aux else up_factor
        self.conv_out = nn.Sequential(
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBNReLU(mid_chan, mid_chan2, 3, stride=1)
                ) if aux else nn.Identity(),
            nn.Conv2d(mid_chan2, out_chan, 1, 1, 0, bias=True),
            nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat

class BiSeNetV2(nn.Module):
    def __init__(self, n_classes, aux_mode='train'):
        super(BiSeNetV2, self).__init__()
        self.aux_mode = aux_mode
        self.detail = DetailBranch()
        self.segment = SegmentBranch()
        self.bga = BGALayer()

        ## TODO: what is the number of mid chan ?
        self.binary_head = binary_segmenthead(128, 1024, n_classes, scale_factor = 8)
        self.instance_head = Instance_SegmentHead(128, 1024, n_classes = 128, up_factor=8, aux=False)
        # if self.aux_mode == 'train':
        #     self.aux2 = Binary_SegmentHead(16, 128, n_classes, up_factor=4)
        #     self.aux3 = Binary_SegmentHead(32, 128, n_classes, up_factor=8)
        #     self.aux4 = Binary_SegmentHead(64, 128, n_classes, up_factor=16)
        #     self.aux5_4 = Binary_SegmentHead(128, 128, n_classes, up_factor=32)

        # self.init_weights()

    def forward(self, x):
        # size = x.size()[2:]
        feat_d = self.detail(x)
        feat_s = self.segment(x)
        feat_head = self.bga(feat_d, feat_s)
        # print(f"hahhahha = {feat_head.shape}")

        binary_logits,con0 = self.binary_head(feat_head)
        instance_logits = self.instance_head(feat_head)
        if self.aux_mode == 'train':
            return binary_logits, instance_logits,con0
        elif self.aux_mode == 'eval':
            return binary_logits, instance_logits,con0

    def init_weights(self):
        for name, module in self.named_modules():
            # print(name)
            # print(module)
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        self.load_pretrain()

    def load_pretrain(self):
        state = modelzoo.load_url(backbone_url)
        for name, child in self.named_children():
            if name in state.keys():
                child.load_state_dict(state[name], strict=True)

    def get_params(self):
        def add_param_to_list(mod, wd_params, nowd_params):
            for param in mod.parameters():
                if param.dim() == 1:
                    nowd_params.append(param)
                elif param.dim() == 4:
                    wd_params.append(param)
                else:
                    print(name)

        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if 'head' in name or 'aux' in name:
                add_param_to_list(child, lr_mul_wd_params, lr_mul_nowd_params)
            else:
                add_param_to_list(child, wd_params, nowd_params)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

class SNN_BiSeNetV2(nn.Module):
    def __init__(self, n_classes, aux_mode='train'):
        super(SNN_BiSeNetV2, self).__init__()
        self.aux_mode = aux_mode
        self.detail = SNN_DetailBranch()
        self.segment = SNN_SegmentBranch()
        self.bga = SNN_BGALayer()
        self.T = 4

        ## TODO: what is the number of mid chan ?
        self.binary_head = Binary_SegmentHead(128, 1024, n_classes, up_factor=8, aux=False)
        self.instance_head = Instance_SegmentHead(128, 1024, n_classes = 128, up_factor=8, aux=False)
        # if self.aux_mode == 'train':
        #     self.aux2 = Binary_SegmentHead(16, 128, n_classes, up_factor=4)
        #     self.aux3 = Binary_SegmentHead(32, 128, n_classes, up_factor=8)
        #     self.aux4 = Binary_SegmentHead(64, 128, n_classes, up_factor=16)
        #     self.aux5_4 = Binary_SegmentHead(128, 128, n_classes, up_factor=32)

        # self.init_weights()

    def forward(self, x):
        # size = x.size()[2:]
        feat_heads = []  
        for _ in range(self.T):    
            feat_d = self.detail(x)  # 假设返回的是一个特征张量  
            feat_s = self.segment(x)  # 假设返回的是一个特征张量  
            feat_head = self.bga(feat_d, feat_s)  # 假设bga函数返回一个特征张量，其第一个维度是时间步
            feat_heads.append(feat_head)  
        stacked_feat_heads = torch.stack(feat_heads, dim=0)
        feat_head = stacked_feat_heads.mean(dim=0)
        # 将每个时间步的feat_head添加到列表中  
        feat_heads.append(feat_head) 
        print()

        binary_logits = self.binary_head(feat_head)
        instance_logits = self.instance_head(feat_head)
        if self.aux_mode == 'train':
            # logits_aux2 = self.aux2(feat2)
            # logits_aux3 = self.aux3(feat3)
            # logits_aux4 = self.aux4(feat4)
            # logits_aux5_4 = self.aux5_4(feat5_4)
            return binary_logits, instance_logits
        elif self.aux_mode == 'eval':
            return binary_logits, instance_logits
        # elif self.aux_mode == 'pred':
        #     pred = logits.argmax(dim=1)
        #   return pred
        else:
            raise NotImplementedError

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # self.load_pretrain()

    def load_pretrain(self):
        state = modelzoo.load_url(backbone_url)
        for name, child in self.named_children():
            if name in state.keys():
                child.load_state_dict(state[name], strict=True)

    def get_params(self):
        def add_param_to_list(mod, wd_params, nowd_params):
            for param in mod.parameters():
                if param.dim() == 1:
                    nowd_params.append(param)
                elif param.dim() == 4:
                    wd_params.append(param)
                else:
                    print(name)

        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if 'head' in name or 'aux' in name:
                add_param_to_list(child, lr_mul_wd_params, lr_mul_nowd_params)
            else:
                add_param_to_list(child, wd_params, nowd_params)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

class LaneNetBackEnd(nn.Module):
    def __init__(self, embedding_dims, num_classes):
        super(LaneNetBackEnd, self).__init__()
        self.embedding_dims = embedding_dims
        self.num_classes = num_classes

        # Define layers for binary segmentation
        self.seg_conv_bn = nn.BatchNorm2d(128)
        self.seg_conv_relu = nn.ReLU()

        # Define layers for instance segmentation
        self.seg_conv = nn.Conv2d(128, self.embedding_dims, kernel_size=1)  # Fill in the missing parameters

    def forward(self, binary_seg_logits, instance_seg_logits):
        # Binary segmentation
        binary_seg_score = F.softmax(binary_seg_logits, dim=1)
        binary_seg_prediction = torch.argmax(binary_seg_score, dim=1)

        # Instance segmentation
        pix_bn = self.seg_conv_bn(instance_seg_logits)
        pix_relu = self.seg_conv_relu(pix_bn)
        instance_seg_prediction = self.seg_conv(pix_relu)

        return binary_seg_prediction, instance_seg_prediction

class firstNet(nn.Module):
    def __init__(self, n_classes,embedding_dims,aux_mode='train'):
        super(firstNet, self).__init__()
        self.aux_mode = aux_mode
        self.bisenet = BiSeNetV2(n_classes=n_classes)
        self.backend = LaneNetBackEnd(embedding_dims=embedding_dims, num_classes=n_classes)
    def forward(self, x):
        H, W = x.size()[2:]
        binary_seg_logits, middle_instance_seg_logits,con0 = self.bisenet(x)
        binary_seg_prediction, instance_seg_logits = self.backend(binary_seg_logits, middle_instance_seg_logits)

        binary_out = F.interpolate(binary_seg_logits, (H, W), mode='bilinear', align_corners=True)
        instance_out = F.interpolate(instance_seg_logits, (H, W), mode='bilinear', align_corners=True)
        connet0 = F.interpolate(con0, (H, W), mode='bilinear', align_corners=True)

        if self.aux_mode == "train":
            return {
                'binary_logits': binary_out,
                'seg_logits': instance_out,
                'con0': connet0,
            }   
        else:
            return {
                'binary_logits': binary_out,
                'seg_logits': instance_out,
                'con0': connet0,
            }  

def my_hook(Module, input, output):
    outshapes.append(output.shape)
    modules.append(Module)

def calc_paras_flops(modules,outshapes):
    total_para_nums = 0
    total_mul_flops = 0
    total_add_flops = 0
    for i,m in enumerate(modules):
        Cin = m.in_channels
        Cout = m.out_channels
        k = m.kernel_size
        #p = m.padding
        #s = m.stride
        #d = m.dilation
        g = m.groups
        Hout = outshapes[i][2]
        Wout = outshapes[i][3]
        if m.bias is None:
            para_nums = k[0] * k[1] * Cin / g * Cout
            mul_flops = (k[0] * k[1] * Cin/g) * Cout * Hout * Wout
            add_flops = (k[0] * k[1] * Cin/g - 1)* Cout * Hout * Wout
        else:
            para_nums = (k[0] * k[1] * Cin / g +1) * Cout
            mul_flops = (k[0] * k[1] * Cin/g) * Cout * Hout * Wout
            add_flops = (k[0] * k[1] * Cin/g - 1)* Cout * Hout * Wout
        para_nums = int(para_nums)
        mul_flops = int(mul_flops)
        add_flops = int(add_flops)        
        print(names[i], 'para:', para_nums, 'mul_flops:',mul_flops,"add_flops",add_flops)
        total_para_nums += para_nums
        total_mul_flops += mul_flops
        total_add_flops += add_flops
    print('total conv parameters:',total_para_nums, 'total mul FLOPs:',total_mul_flops,"total_add_flops",total_add_flops)

if __name__ == "__main__":
    model = BiSeNetV2(n_classes=2)
    names,modules,outshapes = [],[],[]
    for name,m in model.named_modules():
        if isinstance(m,nn.Conv2d):
            m.register_forward_hook(my_hook)
            names.append(name)
    input = torch.rand(2,3,960,544)#需要先提供一个输入张量
    y = model(input)
    calc_paras_flops(modules,outshapes)

    # x = torch.randn(16, 3, 256, 512)
    # model = BiSeNetV2(n_classes=2)
    # snn_model = SNN_BiSeNetV2(n_classes=2)
    # outs = model(x)
    # for out in outs:
    #     print(out.size())
    # outs = snn_model(x)
    # for out in outs:
    #     print(out.size()) 
    # tensor = torch.randint(0, 2, (1,10,10)).float() 
    # print(tensor)
    # maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
    # tensor = maxpool(tensor)
    # print(tensor)

    # x = torch.randn(1, 3, 960, 512)
    # model = SNN_BiSeNetV2(n_classes=2)
    # flops, params = profile(model, inputs=(x, ))
    # flops, params = clever_format([flops, params], "%.3f")
    # print(f"flops = {flops} params = {params}")

    # layer = SNN_ConvBNReLU(3, 64, 3, stride=2)
    # x = torch.randn(1, 3, 960, 512)
    # flops, params = profile(layer, inputs=(x, ))
    # flops, params = clever_format([flops, params], "%.3f")
    # print(f"flops = {flops} params = {params}")

     