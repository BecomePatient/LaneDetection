import torch
from spikingjelly.activation_based import neuron, functional, surrogate, layer,learning

x = torch.rand(1,3, 8, 8)
# print(x)
conv1 = layer.Conv2d(3, 3, kernel_size=3, stride=1,padding=1,bias=False)
conv1.weight.data = 0.5*torch.rand_like(conv1.weight)
# print(conv1.weight)
# bn1 = layer.BatchNorm2d(3,affine=False)
sn1 = neuron.CompressIFNode(surrogate_function=surrogate.ATan())
sn1 = sn1.eval()
out,out_v = sn1(conv1(x))
print(out)
print(out_v)