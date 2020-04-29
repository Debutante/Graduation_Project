from abc import ABCMeta, abstractmethod
from configparser import ConfigParser
import torch
import torch.nn as nn
import torch.nn.functional as F


# NOTE: no longer debug/maintain the codes below since 04/20/20
# for verification problem/ranking problem
class SiameseNet(nn.Module, metaclass=ABCMeta):
    """An abstract siamese net"""
    @abstractmethod
    def forward_once(self, x):
        pass


# Multi-scale siamese net
class MssNet(SiameseNet):
    """An implementation of siamese net
        params 0.2876M, FLOPs(for size(3, 3, 128, 64), just consider conv operation) 267,583,488

    Args:
        config (ConfigParser): The ConfigParser which reads setting files.
        fixed (bool): Whether to fix the torch seed or not.
    """
    DEFAULT_TORCH_SEED = 0
    DEFAULT_FIXED = True

    def __init__(self, config, fixed: bool = DEFAULT_FIXED):
        super(MssNet, self).__init__()
        if fixed:
            torch.manual_seed(config.getint('Default', 'torch_seed', fallback=MssNet.DEFAULT_TORCH_SEED))
            # fix the seed for triplet mining
        self.conv0 = nn.Conv2d(3, 32, 5, padding=2, dilation=1)
        self.conv1 = nn.ModuleList([nn.Conv2d(32, 32, 3, padding=i, dilation=i) for i in range(1, 4)])
        self.convx = nn.ModuleList([nn.ModuleList([nn.Conv2d(96, 32, 3, padding=i, dilation=i) for i in range(1, 4)])
                                    for _ in range(3)])
        self.subconv1 = nn.ModuleList([nn.Conv2d(3, 16, 3, stride=2, padding=1) for _ in range(3)])
        self.subconv2 = nn.ModuleList([nn.Conv2d(16, 16, 3, stride=2, padding=1) for _ in range(3)])
        self.pool = nn.MaxPool2d(2, 2)

    def forward_once(self, x):
        # TODO batch norm
        if isinstance(x, (list, tuple)):
            x = torch.stack(x)
        elif len(x.size()) == 3:
            x = x.unsqueeze(0)
        subh = x.size()[2] // 4
        if len(x.size()) == 4:
            subx = [x[:, :, subh * i: subh * (i + 2), :] for i in range(3)]  # batchNum, 3, 80, 64
        else:
            raise IndexError('Operations in MssNet like conv2d only support 4 dim input, '
                             'but got {} dim input.'.format(len(x.size())))

        # global branch, start from batchNum, 3, 160, 64
        x = self.pool(F.relu(self.conv0(x)))  # batchNum, 32, 80, 32
        xbranch = [F.relu(self.conv1[i](x)) for i in range(3)]  # batchNum, 32, 80, 32
        x = self.pool(torch.cat(tuple(xbranch), 1))  # batchNum, 96, 40, 16
        for i in range(3):
          xbranch = [F.relu(self.convx[i][j](x)) for j in range(3)]
          x = self.pool(torch.cat(tuple(xbranch), 1))

        # local branch, start from batchNum, 3, 80, 64
        subx = [self.pool(F.relu(conv1(item))) for conv1, item in zip(self.subconv1, subx)]  # batchNum, 16, 20, 16
        subx = [self.pool(F.relu(conv2(item))) for conv2, item in zip(self.subconv2, subx)]  # batchNum, 16, 5, 4
        subx = torch.cat(tuple(subx), 1)  # batchNum, 48, 5, 4
        return torch.cat((x.view(x.size()[0], -1), subx.view(subx.size()[0], -1)), 1)  # batchNum, 960 + 960 = 1920

    def forward(self, a, p, n):
        """
        Args:
            a (torch.Tensor or list/tuple of torch.Tensor): Anchor tensors in batch.
            p (torch.Tensor or list/tuple of torch.Tensor): Positive tensors in batch.
            n (torch.Tensor or list/tuple of torch.Tensor): Negative tensors in batch.

        Returns:
            tuple(anchor_output, positive_output, negative_output)

        """
        return self.forward_once(a), self.forward_once(p), self.forward_once(n)
