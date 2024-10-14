import torch
import torch.nn as nn
from copy import deepcopy
from spikingjelly.activation_based import functional, neuron, layer
import torch.nn.functional as F

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torchvision._internally_replaced_utils import load_state_dict_from_url

__all__ = [
    'SpikingVGG',
    'spiking_vgg11','spiking_vgg11_bn',
    'spiking_vgg13','spiking_vgg13_bn',
    'spiking_vgg16','spiking_vgg16_bn',
    'spiking_vgg19','spiking_vgg19_bn',
]

model_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-8a719046.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-19584684.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}

# modified by https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py

class SpikingVGG(nn.Module):

    def __init__(self, cfg, batch_norm=False, norm_layer=None, num_classes=10, init_weights=True,
                 spiking_neuron: callable = neuron.IFNode, **kwargs):
        super(SpikingVGG, self).__init__()
        self.features = self.make_layers(cfg=cfg, batch_norm=batch_norm,
                                         norm_layer=norm_layer, neuron=neuron.IFNode, **kwargs)
        self.avgpool = layer.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            layer.Linear(512 * 7 * 7, 4096),
            neuron.IFNode(),
            layer.Dropout(),
            layer.Linear(4096, 4096),
            neuron.IFNode(),
            layer.Dropout(),
            layer.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def make_layers(cfg, batch_norm=False, norm_layer=None, neuron: callable = neuron.IFNode, **kwargs):
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [layer.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = layer.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, norm_layer(v), neuron(**deepcopy(kwargs))]
                else:
                    layers += [conv2d, neuron(**deepcopy(kwargs))]
                in_channels = v
        return nn.Sequential(*layers)


def sequential_forward(sequential, x_seq):
    assert isinstance(sequential, nn.Sequential)
    out = x_seq
    for i in range(len(sequential)):
        m = sequential[i]
        if isinstance(m, neuron.BaseNode):
            out = m(out)
        else:
            out = functional.seq_to_ann_forward(out, m)
    return out




cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class SpikingVGGMM(nn.Module):
    def __init__(self, vgg_name, num_classes,timesteps):
        super(SpikingVGGMM, self).__init__()
        self.num_classes = num_classes
        self.vgg_name = vgg_name
        self.timestep = timesteps
        self.conv_layers = self._make_conv_layers()

        self.fc1 = layer.Linear(512, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.spike1 = neuron.IFNode()

        self.fc2 = layer.Linear(512, num_classes)
        self.spike2 = neuron.IFNode()

    def forward(self, x):
        x = torch.mean(x, dim=0)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.unsqueeze(x, dim=0)
        x = self.spike1(x)
        x = torch.mean(x, dim=0)
        x = self.fc2(x)
        x = torch.unsqueeze(x, dim=0)
        x = self.spike2(x)
        return x

    def _make_conv_layers(self):
        layers = []
        in_channels = 3
        for cfg in self.vgg_cfg[self.vgg_name]:
            if cfg == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, cfg, kernel_size=3, padding=1),
                           nn.BatchNorm2d(cfg),
                           nn.ReLU(inplace=True)]
                in_channels = cfg
        return nn.Sequential(*layers)

    vgg_cfg = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    }

    

class SpikingVGGM(nn.Module):
    def __init__(self,  num_classes=100, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, spiking_neuron: callable = None,T= 4, **kwargs):
        super(SpikingVGGM,self).__init__()
        
        self.conv1 = layer.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = layer.BatchNorm2d(128)
        self.pool1 = layer.MaxPool2d(kernel_size=2, stride=2)
        self.spike1 = neuron.IFNode()
        
        self.conv2 = layer.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = layer.BatchNorm2d(256)
        self.pool2 = layer.MaxPool2d(kernel_size=2, stride=2)
        self.spike2 = neuron.IFNode()
        
        self.conv3 = layer.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn3 = layer.BatchNorm2d(512)
        self.pool3 = layer.MaxPool2d(kernel_size=2, stride=2)
        self.spike3 = neuron.IFNode()
        
        self.conv4 = layer.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = layer.BatchNorm2d(512)
        self.pool4 = layer.MaxPool2d(kernel_size=2, stride=2)
        self.spike4 = neuron.IFNode()
        
        self.flatten = layer.Flatten()
        self.fc1 = layer.Linear(512*4, 1024)
        self.spike5 = neuron.IFNode()
        self.fc2 = layer.Linear(1024, 1024)
        self.spike6 =neuron.IFNode()
        self.fc3 = layer.Linear(1024, 10)
        self.spike7 = neuron.IFNode()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.spike1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.spike2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.spike3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        x = self.spike4(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.spike5(x)
        x = self.fc2(x)
        x = self.spike6(x)
        x = self.fc3(x)
        x = self.spike7(x)
        
        return x
    # def forward(self, x):
    #     return self._forward_impl(x)

def _spiking_vgg(arch, cfg, batch_norm, pretrained, progress, norm_layer: callable = None, spiking_neuron: callable = None, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    if batch_norm:
        norm_layer = norm_layer
    else:
        norm_layer = None
    model = SpikingVGG(cfg=cfgs[cfg], batch_norm=batch_norm, norm_layer=norm_layer, spiking_neuron=spiking_neuron, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def spiking_vgg11(pretrained=False, progress=True, spiking_neuron: callable = None, **kwargs):
    """
        :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
        :type pretrained: bool
        :param progress: If True, displays a progress bar of the download to stderr
        :type progress: bool
        :param spiking_neuron: a spiking neuron layer
        :type spiking_neuron: callable
        :param kwargs: kwargs for `spiking_neuron`
        :type kwargs: dict
        :return: Spiking VGG-11
        :rtype: torch.nn.Module

        A spiking version of VGG-11 model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _spiking_vgg('vgg11', 'A', False, pretrained, progress, None, spiking_neuron, **kwargs)




def spiking_vgg11_bn(pretrained=False, progress=True, norm_layer: callable = None, spiking_neuron: callable = None, **kwargs):
    """
        :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
        :type pretrained: bool
        :param progress: If True, displays a progress bar of the download to stderr
        :type progress: bool
        :param norm_layer: a batch norm layer
        :type norm_layer: callable
        :param spiking_neuron: a spiking neuron layer
        :type spiking_neuron: callable
        :param kwargs: kwargs for `spiking_neuron`
        :type kwargs: dict
        :return: Spiking VGG-11 with norm layer
        :rtype: torch.nn.Module

        A spiking version of VGG-11-BN model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _spiking_vgg('vgg11_bn', 'A', True, pretrained, progress, norm_layer, spiking_neuron, **kwargs)



def spiking_vgg13(pretrained=False, progress=True, spiking_neuron: callable = None, **kwargs):
    """
        :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
        :type pretrained: bool
        :param progress: If True, displays a progress bar of the download to stderr
        :type progress: bool
        :param spiking_neuron: a spiking neuron layer
        :type spiking_neuron: callable
        :param kwargs: kwargs for `spiking_neuron`
        :type kwargs: dict
        :return: Spiking VGG-13
        :rtype: torch.nn.Module

        A spiking version of VGG-13 model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _spiking_vgg('vgg13', 'B', False, pretrained, progress, None, spiking_neuron, **kwargs)




def spiking_vgg13_bn(pretrained=False, progress=True, norm_layer: callable = None, spiking_neuron: callable = None, **kwargs):
    """
        :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
        :type pretrained: bool
        :param progress: If True, displays a progress bar of the download to stderr
        :type progress: bool
        :param norm_layer: a batch norm layer
        :type norm_layer: callable
        :param spiking_neuron: a spiking neuron layer
        :type spiking_neuron: callable
        :param kwargs: kwargs for `spiking_neuron`
        :type kwargs: dict
        :return: Spiking VGG-11 with norm layer
        :rtype: torch.nn.Module

        A spiking version of VGG-13-BN model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _spiking_vgg('vgg13_bn', 'B', True, pretrained, progress, norm_layer, spiking_neuron, **kwargs)




def spiking_vgg16(pretrained=False, progress=True, spiking_neuron: callable = None, **kwargs):
    """
        :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
        :type pretrained: bool
        :param progress: If True, displays a progress bar of the download to stderr
        :type progress: bool
        :param spiking_neuron: a spiking neuron layer
        :type spiking_neuron: callable
        :param kwargs: kwargs for `spiking_neuron`
        :type kwargs: dict
        :return: Spiking VGG-16
        :rtype: torch.nn.Module

        A spiking version of VGG-16 model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _spiking_vgg('vgg16', 'D', False, pretrained, progress, None, spiking_neuron, **kwargs)



def spiking_vgg16_bn(pretrained=False, progress=True, norm_layer: callable = None, spiking_neuron: callable = None, **kwargs):
    """
        :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
        :type pretrained: bool
        :param progress: If True, displays a progress bar of the download to stderr
        :type progress: bool
        :param norm_layer: a batch norm layer
        :type norm_layer: callable
        :param spiking_neuron: a spiking neuron layer
        :type spiking_neuron: callable
        :param kwargs: kwargs for `spiking_neuron`
        :type kwargs: dict
        :return: Spiking VGG-16 with norm layer
        :rtype: torch.nn.Module

        A spiking version of VGG-16-BN model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _spiking_vgg('vgg16_bn', 'D', True, pretrained, progress, norm_layer, spiking_neuron, **kwargs)



def spiking_vgg19(pretrained=False, progress=True, spiking_neuron: callable = None, **kwargs):
    """
        :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
        :type pretrained: bool
        :param progress: If True, displays a progress bar of the download to stderr
        :type progress: bool
        :param spiking_neuron: a spiking neuron layer
        :type spiking_neuron: callable
        :param kwargs: kwargs for `spiking_neuron`
        :type kwargs: dict
        :return: Spiking VGG-19
        :rtype: torch.nn.Module

        A spiking version of VGG-19 model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _spiking_vgg('vgg19', 'E', False, pretrained, progress, None, spiking_neuron, **kwargs)



def spiking_vgg19_bn(pretrained=False, progress=True, norm_layer: callable = None, spiking_neuron: callable = None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param norm_layer: a batch norm layer
    :type norm_layer: callable
    :param spiking_neuron: a spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking VGG-19 with norm layer
    :rtype: torch.nn.Module

    A spiking version of VGG-19-BN model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _spiking_vgg('vgg19_bn', 'E', True, pretrained, progress, norm_layer, spiking_neuron, **kwargs)


