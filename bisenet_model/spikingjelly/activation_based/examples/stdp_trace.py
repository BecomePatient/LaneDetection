import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, learning
from matplotlib import pyplot as plt
import torchvision
from spikingjelly.activation_based.model.tv_ref_classify.sampler import RASampler
from spikingjelly.activation_based.model.tv_ref_classify import presets
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
import time

def f_weight(x):
    return torch.clamp(x, -1, 1.)

torch.manual_seed(0)
# plt.style.use(['science'])

def load_CIFAR10(self, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = args.val_resize_size, args.val_crop_size, args.train_crop_size
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    dataset = torchvision.datasets.CIFAR10(
        root=args.data_path,
        train=True,
        download=True,
        transform=presets.ClassificationPresetTrain(
            crop_size=train_crop_size,
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
            interpolation=interpolation,
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob,
        ),
    )

    print("Took", time.time() - st)

    print("Loading validation data")

    dataset_test = torchvision.datasets.CIFAR10(
        root=args.data_path,
        train=False,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
    )

    print("Creating data loaders")
    loader_g = torch.Generator()
    loader_g.manual_seed(args.seed)

    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps, seed=args.seed)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, seed=args.seed)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset, generator=loader_g)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler

if __name__ == '__main__':

    def f_pre(x, w_min, alpha=0.):
        return (x - w_min) ** alpha

    def f_post(x, w_max, alpha=0.):
        return (w_max - x) ** alpha

    w_min, w_max = -1., 1.
    tau_pre, tau_post = 2., 2.
    N_in, N_out = 4, 3
    T = 128
    batch_size = 2
    lr = 0.01
    net = nn.Sequential(
        layer.Linear(N_in, N_out, bias=False),
        neuron.IFNode()
    )
    nn.init.constant_(net[0].weight.data, 0.4)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.)

    in_spike = (torch.rand([T, batch_size, N_in]) > 0.7).float()
    learner = learning.STDPLearner(step_mode='s', synapse=net[0], sn=net[1], 
                                   tau_pre=tau_pre, tau_post=tau_post,
                                   f_pre=f_weight, f_post=f_weight)

    out_spike = []
    trace_pre = []
    trace_post = []
    weight = []
    with torch.no_grad():
        for t in range(T):
            optimizer.zero_grad()
            out_spike.append(net(in_spike[t]))
            learner.step(on_grad=True)
            optimizer.step()
            net[0].weight.data.clamp_(w_min, w_max)
            weight.append(net[0].weight.data.clone())
            trace_pre.append(learner.trace_pre)
            trace_post.append(learner.trace_post)

    out_spike = torch.stack(out_spike)   # [T, batch_size, N_out]
    trace_pre = torch.stack(trace_pre)   # [T, batch_size, N_in]
    trace_post = torch.stack(trace_post) # [T, batch_size, N_out]
    weight = torch.stack(weight)         # [T, N_out, N_in]

    t = torch.arange(0, T).float()
    
    in_spike = in_spike[:, 0, 0]
    out_spike = out_spike[:, 0, 0]
    trace_pre = trace_pre[:, 0, 0]
    trace_post = trace_post[:, 0, 0]
    weight = weight[:, 0, 0]

    cmap = plt.get_cmap('tab10')
    plt.subplot(5, 1, 1)
    plt.eventplot((in_spike * t)[in_spike == 1], lineoffsets=0, colors=cmap(0))
    plt.xlim(-0.5, T + 0.5)
    plt.ylabel('$s[i]$', rotation=0, labelpad=10)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(5, 1, 2)
    plt.plot(t, trace_pre, c=cmap(1))
    plt.xlim(-0.5, T + 0.5)
    plt.ylabel('$tr_{pre}$', rotation=0)
    plt.yticks([trace_pre.min().item(), trace_pre.max().item()])
    plt.xticks([])

    plt.subplot(5, 1, 3)
    plt.eventplot((out_spike * t)[out_spike == 1], lineoffsets=0, colors=cmap(2))
    plt.xlim(-0.5, T + 0.5)
    plt.ylabel('$s[j]$', rotation=0, labelpad=10)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(5, 1, 4)
    plt.plot(t, trace_post, c=cmap(3))
    plt.ylabel('$tr_{post}$', rotation=0)
    plt.yticks([trace_post.min().item(), trace_post.max().item()])
    plt.xlim(-0.5, T + 0.5)
    plt.xticks([])

    plt.subplot(5, 1, 5)
    plt.plot(t, weight, c=cmap(4))
    plt.xlim(-0.5, T + 0.5)
    plt.ylabel('$w[i][j]$', rotation=0)
    plt.yticks([weight.min().item(), weight.max().item()])
    plt.xlabel('time-step')
    
    plt.gcf().subplots_adjust(left=0.18)
    
    plt.show()
    plt.savefig('./docs/source/_static/tutorials/activation_based/stdp/stdp_trace.png')
    plt.savefig('./docs/source/_static/tutorials/activation_based/stdp/stdp_trace.svg')
    plt.savefig('./docs/source/_static/tutorials/activation_based/stdp/stdp_trace.pdf')
