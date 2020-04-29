from abc import ABCMeta, abstractmethod
from configparser import ConfigParser, ExtendedInterpolation
import torch
import torch.nn as nn
from torch.nn import init
from torchvision.models.resnet import resnet50

conf_path = '../settings.txt'
conf = ConfigParser(interpolation=ExtendedInterpolation(), default_section='Default')
conf.read(conf_path)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class ClassBlock(nn.Module):
    """A block which defines a new embedding layer and a classification layer. Usually used as the final block.

    Structure: |--Linear--|--bn--|--relu--|--dropout--|--Linear--|, only the last Linear layer is compulsory.

    Args:
        input_dim (int): The output dim of the last module, also the input dim for this block.
        class_num (int): The output dim of the last Linear layer(classification layer) in this block.
        drop_rate (float): The dropout ratio of the dropout layer. The dropout layer is added only if droprate > 0.
        num_bottleneck (int): The output dim of the first Linear layer(embedding layer) in this block.
        relu (bool): Add a relu activation before the last Linear layer(classification layer) or not.
        bnorm (bool): Add a batch norm normalization before the last Linear layer(classification layer) or not.
        linear (bool): Add a linear layer(embedding layer) before the last Linear layer(classification layer) or not.

    """
    DEFAULT_RELU = False
    DEFAULT_BNORM = True
    DEFAULT_LINEAR = True

    def __init__(self, input_dim: int, class_num: int, drop_rate, num_bottleneck: int,
                 relu: bool = DEFAULT_RELU, bnorm: bool = DEFAULT_BNORM, linear: bool = DEFAULT_LINEAR):
        super(ClassBlock, self).__init__()
        new_block = []
        if linear:
            new_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            new_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            new_block += [nn.LeakyReLU(0.1)]
        if drop_rate > 0:
            new_block += [nn.Dropout(p=drop_rate)]
        new_block = nn.Sequential(*new_block)
        new_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.embedding = new_block
        self.classifier = classifier

    def set_to_test(self):
        self.classifier = nn.Sequential()

    def forward(self, x):
        x = self.embedding(x)
        x = self.classifier(x)
        return x


# for classification problem
class IdentificationModel(nn.Module, metaclass=ABCMeta):
    """An abstract identification model"""
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def set_to_test(self):
        """Clear the classification layer for testing."""
        pass


class ResNet50(IdentificationModel):
    """An adapted ResNet50 for identification tasks.

    Structure: |--ResNet50--|--Linear--|--bn--|--relu(not in this implementation)--|--dropout--|--Linear--|

    Args:
        config (ConfigParser): The ConfigParser which reads setting files.
        class_num (int): The output dim of the last Linear layer(classification layer) in this model.
        pretrained (bool): Load pretrained weights on ImageNet or not.
        droprate (float): The dropout ratio of the dropout layer. The dropout layer is added only if droprate > 0.
        fixed (bool): Whether to fix the torch seed or not.

    Attributes:
        model (nn.Module): The structure of resnet50 defined in torchvision/models/resnet.py.
        final_block (nn.Module): The structure of ClassBlock defined above.

    """
    DEFAULT_BOTTLENECK = 512
    DEFAULT_DROPRATE = 0.5
    DEFAULT_TORCH_SEED = 0
    DEFAULT_FIXED = True
    DEFAULT_PRETRAIN = True

    def __init__(self, config, class_num: int, pretrained: bool = DEFAULT_PRETRAIN, droprate=DEFAULT_DROPRATE,
                 fixed=DEFAULT_FIXED):
        super(ResNet50, self).__init__()
        if fixed:
            torch.manual_seed(config.getint('Default', 'torch_seed', fallback=ResNet50.DEFAULT_TORCH_SEED))
        bottleneck = config.getint('Default', 'embedding_dim', fallback=ResNet50.DEFAULT_BOTTLENECK)
        model = resnet50(pretrained=pretrained)
        self.model = model
        # TODO: change self.model to self.backbone
        self.final_block = ClassBlock(self.model.fc.in_features, class_num, droprate, bottleneck)

    def set_to_test(self):
        self.final_block.set_to_test()

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.final_block(x)
        return x


if __name__ == '__main__':
    N, C, H, W = 2, 3, 128, 64
    model = ResNet50(conf, 751, False)
    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)
