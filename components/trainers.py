from datasets.dataset_processors import TripletDataset
from models.model import ResNet50
from models.siamese import MssNet
from base import BaseExecutor
from utils.utilities import type_error_msg, value_error_msg, load_model, format_path, save_model, merge_last
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from configparser import ConfigParser
from time import time
from math import ceil
from os import makedirs
from os.path import join, dirname, exists
from enum import Enum, unique
from scipy.io import loadmat, savemat


class BaseTrainer(BaseExecutor):
    """A general trainer scheme.

    Args:
        name (str): A name defined in base.py.
        dataset (str): A dataset defined in base.py.
        split (str): A split protocol defined in base.py.
        model (str): A model defined in base.py.
        optimizer (str): An optimizer defined below.
        start_epoch (int): An model@start_epoch to train from.

    Attributes:
        name (BaseExecutor.Name): A name enum defined in base.py.
        dataset (BaseExecutor.Dataset): A dataset enum defined in base.py.
        split (BaseExecutor.Split): A split enum defined in base.py.
        model (BaseExecutor.Model): A model enum defined in base.py.
        optimizer (BaseTrainer.Optimizer): An optimizer enum defined below.
        start_epoch (int): An model@start_epoch to train from.

    """
    @unique
    class Optimizer(Enum):
        SGD = 'sgd'
        # TODO: 'adam', 'rmsprop'

    OPTIMIZER_LIST = [item.value for item in Optimizer]

    DEFAULT_BATCH_SIZE = 32
    DEFAULT_NUM_WORKER = 8
    DEFAULT_MAX_EPOCH = 50
    DEFAULT_START_EPOCH = None

    def __init__(self, name, dataset, split, model, optimizer, start_epoch):
        if not isinstance(name, BaseTrainer.Name):
            if isinstance(name, str):
                if not name.islower():
                    name = name.lower()
                if name not in BaseTrainer.NAME_LIST:
                    raise ValueError(value_error_msg('name', name, BaseTrainer.NAME_LIST))
                name = BaseTrainer.Name(name)
            else:
                raise TypeError(type_error_msg('name', name, [BaseTrainer.Name, str]))

        if not isinstance(dataset, BaseTrainer.Dataset):
            if isinstance(dataset, str):
                if not dataset.islower():
                    dataset = dataset.lower()
                if dataset not in BaseTrainer.DATASET_LIST:
                    raise ValueError(value_error_msg('dataset', dataset, BaseTrainer.DATASET_LIST))
                dataset = BaseTrainer.Dataset(dataset)
            else:
                raise TypeError(type_error_msg('dataset', dataset, [BaseTrainer.Dataset, str]))

        if not isinstance(split, BaseTrainer.Split):
            if isinstance(split, str):
                if not split.islower():
                    split = split.lower()
                if split not in BaseTrainer.SPLIT_LIST:
                    raise ValueError(value_error_msg('split', split, BaseTrainer.SPLIT_LIST))
                split = BaseTrainer.Split(split)
            else:
                raise TypeError(type_error_msg('split', split, [BaseTrainer.Split, str]))

        if not isinstance(model, BaseTrainer.Model):
            if isinstance(model, str):
                if not model.islower():
                    model = model.lower()
                if model not in BaseTrainer.MODEL_LIST:
                    raise ValueError(value_error_msg('model', model, BaseTrainer.MODEL_LIST))
                model = BaseTrainer.Model(model)
            else:
                raise TypeError(type_error_msg('model', model, [BaseTrainer.MODEL_LIST, str]))

        if not isinstance(optimizer, BaseTrainer.Optimizer):
            if isinstance(optimizer, str):
                if not optimizer.islower():
                    optimizer = optimizer.lower()
                if optimizer not in BaseTrainer.OPTIMIZER_LIST:
                    raise ValueError(value_error_msg('optimizer', optimizer, BaseTrainer.OPTIMIZER_LIST))
                optimizer = BaseTrainer.Optimizer(optimizer)
            else:
                raise TypeError(type_error_msg('optimizer', optimizer, [BaseTrainer.OPTIMIZER_LIST, str]))

        if start_epoch is not None:
            if not start_epoch > 0:
                raise ValueError(value_error_msg('start_epoch', start_epoch, 'start_epoch > 0',
                                                 BaseTrainer.DEFAULT_START_EPOCH))
        else:
            start_epoch = 0

        self.name = name
        self.dataset = dataset
        self.split = split
        self.model = model
        self.optimizer = optimizer
        self.start_epoch = start_epoch

    def run(self):
        raise NotImplementedError


class Trainer(BaseTrainer):
    """A trainer for normal datasets(not reorganised datasets).

    Args:
        config (ConfigParser): The ConfigParser which reads setting files.
        config_path (str): Path to setting files.
        name (str): A name defined in base.py.
        dataset (str): A dataset defined in base.py.
        split (str): A split protocol defined in base.py.
        model (str): A model defined in base.py.
        pretrain (bool): Initialize model with pretrained weights on ImageNet or not.
        optimizer (str): An optimizer defined above.
        lr (float): The learning rate in the optimizer.
        momentum (float): The momentum in 'sgd'.
        weight_decay (float): The weight decay in 'sgd'.
        start_epoch (int): An model@start_epoch to train from.

    Attributes:
        model_path (str): Path to save the state_dict of model.
        train_path (str): Path to save loss/error in every epoch.
        criterion (torch.nn): The class of loss function parameterized by margin.

    """
    DEFAULT_LR = 0.05
    DEFAULT_MOMENTUM = 0.9
    DEFAULT_WEIGHT_DECAY = 5e-4

    def __init__(self, config, config_path, name, dataset, split, model, pretrain: bool, optimizer,
                 lr: float = DEFAULT_LR, momentum: float = DEFAULT_MOMENTUM, weight_decay: float = DEFAULT_WEIGHT_DECAY,
                 start_epoch: int = BaseTrainer.DEFAULT_START_EPOCH):
        super(Trainer, self).__init__(name, dataset, split, model, optimizer, start_epoch)

        if not lr > 0:
            raise ValueError(value_error_msg('lr', lr, 'lr > 0', Trainer.DEFAULT_LR))

        if not momentum >= 0:
            raise ValueError(value_error_msg('momentum', momentum, 'momentum >= 0', Trainer.DEFAULT_MOMENTUM))

        if not weight_decay >= 0:
            raise ValueError(value_error_msg('weight_decay', weight_decay, 'weight_decay >= 0',
                                             Trainer.DEFAULT_MOMENTUM))

        self.config = config
        self.model_path = format_path(self.config[self.name.value]['model_format'], self.name.value,
                                      self.config['Default']['delimiter'])

        if self.split == Trainer.Split.TRAIN_VAL:
            self.phase = ['train', 'val']
        elif self.split == Trainer.Split.TRAIN_ONLY:
            self.phase = ['train']
        else:
            raise ValueError(value_error_msg('split', split, BaseTrainer.SPLIT_LIST))

        if self.name == Trainer.Name.MARKET1501:
            transform_train_list = [
                # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3),
                transforms.Resize((256, 128), interpolation=3),
                transforms.Pad(10),
                transforms.RandomCrop((256, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]

            transform_val_list = [
                transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]

            data_transforms = {
                'train': transforms.Compose(transform_train_list),
                'val': transforms.Compose(transform_val_list),
            }
        else:
            raise ValueError(value_error_msg('name', self.name, Trainer.NAME_LIST))

        # dataset declaration
        self.dataset = {}
        self.dataset_sizes = {}

        # dataset loading
        for phase in self.phase:
            folder_name = phase
            if self.split == Trainer.Split.TRAIN_ONLY:
                folder_name = 'total_' + folder_name
            self.dataset[phase] = ImageFolder(join(self.config[self.name.value]['dataset_dir'], folder_name),
                                              data_transforms[phase])
            self.dataset_sizes[phase] = len(self.dataset[phase])

        # record train_class num on setting files
        model_name = self.model.value
        train_class = len(self.dataset['train'].classes)
        config[self.name.value]['train_class'] = str(train_class)
        with open(config_path, 'w+') as file:
            config.write(file)

        # initialize model weights
        if self.model == Trainer.Model.RESNET50:
            self.model = ResNet50(self.config, train_class, pretrained=pretrain)
            if self.start_epoch > 0:
                load_model(self.model, self.config[self.name.value]['model_format'] % (model_name, self.start_epoch))
        # else:
        #     raise ValueError(value_error_msg('model', model, Trainer.MODEL_LIST))

        self.suffix = 'pretrain' if pretrain else 'no_pretrain'
        self.train_path = self.config[self.name.value]['train_path'] % self.suffix

        # use different settings for different params in model when using optimizers
        ignored_params = list(map(id, self.model.final_block.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, self.model.parameters())

        if self.optimizer == BaseTrainer.Optimizer.SGD:
            self.optimizer = optim.SGD([
                {'params': base_params, 'lr': 0.1 * lr},
                {'params': self.model.final_block.parameters(), 'lr': lr}
            ], weight_decay=weight_decay, momentum=momentum, nesterov=True)
        # else:
        #     raise ValueError(value_error_msg('optimizer', optimizer, Trainer.OPTIMIZER_LIST))
        self.criterion = nn.CrossEntropyLoss()

    def run(self, max_epoch: int = BaseTrainer.DEFAULT_MAX_EPOCH, batch_size: int = BaseTrainer.DEFAULT_BATCH_SIZE):
        """
        Reads: None.

        Processes: Trains the model.

        Writes: Max_epoch pth files of model's state dict.
                A mat file of losses/errors.

        Args:
            max_epoch (int):
            batch_size (int):

        """
        Trainer.run_info(self.__class__.__name__, self.suffix)

        if not isinstance(max_epoch, int):
            raise TypeError(type_error_msg('max_epoch', max_epoch, [int]))
        if not max_epoch > 0:
            raise ValueError(value_error_msg('max_epoch', max_epoch, 'max_epoch > 0', TripletTrainer.DEFAULT_MAX_EPOCH))

        if not isinstance(batch_size, int):
            raise TypeError(type_error_msg('batch_size', batch_size, [int]))
        if not batch_size > 0:
            raise ValueError(value_error_msg('batch_size', batch_size, 'batch_size > 0',
                                             TripletTrainer.DEFAULT_BATCH_SIZE))

        # save the initial state for comparing
        if self.start_epoch == 0:
            save_model(self.model, self.model_path % 0)

        # prepare for fetching image
        dataloader = {}
        dataset_size = {}
        max_iteration = {}
        loss_history = {}
        error_history = {}
        for phase in self.phase:
            dataloader[phase] = DataLoader(self.dataset[phase], batch_size=batch_size, shuffle=True,
                                           num_workers=Trainer.DEFAULT_NUM_WORKER)
            dataset_size[phase] = len(self.dataset[phase])
            max_iteration[phase] = ceil(len(self.dataset[phase]) / batch_size)
            loss_history[phase] = []
            error_history[phase] = []

        start = time()
        for epoch in range(self.start_epoch, self.start_epoch + max_epoch):
            print('\n===============Epoch %s===============' % (epoch + 1))
            print('{:<6} {:<13} {:<6} {:<13}'.format('Phase', 'Iteration', 'loss', 'accuracy'))

            for phase in self.phase:
                if phase == 'train':
                    self.model.train()
                elif phase == 'val':
                    self.model.eval()  # batch norm performs differently when model is set to eval rather than train.
                    print('-----------------------------------')

                running_loss = 0.0
                running_correct = 0.0

                for i, data in enumerate(dataloader[phase]):
                    inputs, labels = data
                    now_batch_size = inputs.shape[0]

                    self.optimizer.zero_grad()

                    if phase == 'train':
                        outputs = self.model(inputs)
                    elif phase == 'val':
                        with torch.no_grad():
                            outputs = self.model(inputs)
                    else:
                        raise ValueError()

                    # for error computation
                    with torch.no_grad():
                        predictions = torch.argmax(outputs, 1)

                    loss = self.criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                    # loss/error computation
                    with torch.no_grad():
                        iteration_loss = loss.item()
                        iteration_correct = float(torch.sum(torch.eq(predictions, labels))) / now_batch_size

                    running_loss += iteration_loss * now_batch_size
                    running_correct += iteration_correct * now_batch_size

                    print('{:<6} {:<4}/{:<8} {:<.2f}   {:<.2f}'.format(phase, i + 1, max_iteration[phase],
                                                                       iteration_loss, iteration_correct))

                # compute loss and error for every phases in an epoch
                epoch_loss = running_loss / dataset_size[phase]
                epoch_accuracy = running_correct / dataset_size[phase]

                loss_history[phase].append(epoch_loss)
                error_history[phase].append(1 - epoch_accuracy)

            # save model at the end of every epochs
            save_model(self.model, self.model_path % (epoch + 1))

            now_time_elapsed = time() - start
            print('\nEpoch ({}/{})'.format(epoch + 1, self.start_epoch + max_epoch))
            for phase in self.phase:
                print('{:<5} loss {:.4f}, error {:.4f}'.format(phase, loss_history[phase][-1], error_history[phase][-1]))

            print('time elapsed {:.0f}h {:.0f}m {:.0f}s'.format(now_time_elapsed // 3600,
                                                                (now_time_elapsed % 3600) / 60, now_time_elapsed % 60))

        train_dir = dirname(self.train_path)

        if self.split == Trainer.Split.TRAIN_VAL:
            dictionary = {'training_loss': loss_history['train'], 'training_error': error_history['train'],
                          'validation_loss': loss_history['val'], 'validation_error': error_history['val']}
        elif self.split == Trainer.Split.TRAIN_ONLY:
            dictionary = {'training_loss': loss_history['train'], 'training_error': error_history['train']}
        else:
            raise ValueError(value_error_msg('split', self.split, BaseTrainer.SPLIT_LIST))

        if not exists(train_dir):
            makedirs(train_dir)
        if self.start_epoch == 0 or not exists(self.train_path):
            savemat(self.train_path, dictionary)
        else:  # train.mat already exists, so we should preserve original data and merge them with new data
            last_train_history = loadmat(self.train_path)
            savemat(self.train_path, merge_last(last_train_history, dictionary))


# NOTE: no longer debug/maintain the codes below since 04/20/20
class TripletTrainer(BaseTrainer):
    """A specific trainer for triplet datasets
        where datasets.generate_triplets() is defined and datasets.__getitem returns [img1, img2, img3]

    Args:
        config (ConfigParser): The ConfigParser which reads setting files.
        name (str): A name defined in base.py.
        dataset (str): A dataset defined in base.py.
        model (str): A model defined in base.py.
        optimizer (str): An optimizer defined in base class Trainer.
        lr (float): The learning rate in the optimizer.
        momentum (float): The momentum in 'sgd'.
        margin (float): The margin in nn.TripletMarginLoss.

    Attributes:
        model_path (str): Path to save the state_dict of model.
        triplet_path (str): Path to save generated triplet indices.
        loss_path (str): Path to save training loss for every epoch.
        criterion (torch.nn): The class of loss function parameterized by margin.

    """
    DEFAULT_LR = 0.001
    DEFAULT_MOMENTUM = 0.9
    DEFAULT_MARGIN = 1.0
    DEFAULT_REORDER_FREQ = 10

    @unique
    class Mode(Enum):
        EASY = 'easy'
        MODERATE = 'moderate'
        HARD = 'hard'
    MODE_LIST = [item.value for item in Mode]
    DEFAULT_MODE = Mode.EASY

    def __init__(self, config, name, dataset, model, optimizer, lr: float = DEFAULT_LR,
                 momentum: float = DEFAULT_MOMENTUM, margin: float = DEFAULT_MARGIN,
                 start_epoch: int = BaseTrainer.DEFAULT_START_EPOCH):
        super(TripletTrainer, self).__init__(name, dataset, 'train_only', model, optimizer, start_epoch)
        # triplet dataset doesn't support val

        if not lr > 0:
            raise ValueError(value_error_msg('lr', lr, 'lr > 0', TripletTrainer.DEFAULT_LR))

        if not momentum >= 0:
            raise ValueError(value_error_msg('momentum', momentum, 'momentum >= 0', TripletTrainer.DEFAULT_MOMENTUM))

        if not margin > 0:
            raise ValueError(value_error_msg('margin', margin, 'margin > 0', TripletTrainer.DEFAULT_MARGIN))

        transform_list = []

        if self.name == TripletTrainer.Name.MARKET1501:
            transform_list = [
                # transforms.Resize((160, 64)),
                # transforms.Pad(10),
                # transforms.RandomCrop((160, 64)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]
        # else:
        #     ValueError(value_error_msg('name', name, TripletTrainer.NAME_LIST))
        self.config = config
        self.model_path = self.format_path(self.model.value, 'model_format')
        self.triplet_path = self.format_path(self.dataset.value, 'triplet_format')
        self.loss_path = self.config[self.name.value]['loss_path']

        if self.dataset == TripletTrainer.Dataset.TRIPLET:
            self.dataset = TripletDataset(self.config, self.name.value,
                                          join(self.config[self.name.value]['dataset_dir'], 'total_train'),
                                          transforms.Compose(transform_list))
        else:
            raise ValueError(value_error_msg('dataset', dataset, TripletTrainer.Dataset.TRIPLET))

        model_name = self.model.value
        if self.model == TripletTrainer.Model.MSSNET:
            self.model = MssNet(self.config)
            if self.start_epoch > 0:
                self.load_model(model_name)
        # else:
        #     raise ValueError(value_error_msg('model', model, TripletTrainer.MODEL_LIST))
        if self.optimizer == TripletTrainer.Optimizer.SGD:
            self.optimizer = optim.SGD(self.model.parameters(), lr, momentum)
        # else:
        #     raise ValueError(value_error_msg('optimizer', optimizer, TripletTrainer.OPTIMIZER_LIST))
        self.criterion = nn.TripletMarginLoss(margin=margin)
        # self.writer = SummaryWriter(join('runs', self.name.value))

    def format_path(self, name, option: str):
        """Fills the placeholders to get the path

        Args:
            name (BaseExecutor.Name): A name enum defined in base.py.
            option (str): An option name of section name.value defined in settings.txt.

        Returns:
            path (str): Path to name's option.

        """
        delimiter = self.config['Default']['delimiter']
        prefix, suffix = self.config[self.name.value][option].split(delimiter)
        return prefix % name + delimiter + suffix

    def generate_triplets(self, mode):
        """Implicitly returns self.datasets.index

        Args:
            mode (TripletTrainer.Mode): A name enum defined above.

        """
        if mode == TripletTrainer.Mode.EASY:
            self.dataset.generate_triplets('random', 'class', 10, 'sample', 3)
        elif mode == TripletTrainer.Mode.MODERATE:
            self.dataset.generate_triplets('offline', 'cam', 10, 'class', 3, self.model)
        elif mode == TripletTrainer.Mode.HARD:
            self.dataset.generate_triplets('offline', 'class', 10, 'index', 3, self.model, sort_by='both')
        # else:
        #     raise ValueError(value_error_msg('mode', mode, TripletTrainer.MODE_LIST))

    def run(self, max_epoch: int = BaseTrainer.DEFAULT_MAX_EPOCH, batch_size: int = BaseTrainer.DEFAULT_BATCH_SIZE,
            mode: str = DEFAULT_MODE, reorder_freq: int = DEFAULT_REORDER_FREQ):
        """
        Reads: None.

        Processes: Trains the model.

        Writes: Max_epoch pth files of model's state dict.
                Max_epoch // reorder_freq mat file(s) of triplet indices.
                A mat file of training loss.

        Args:
            max_epoch (int):
            batch_size (int):
            mode (str): A mode defined above.
            reorder_freq (int): Re-generate triplets every reorder_freq epoch(s).

        """

        TripletTrainer.run_info(self.__class__.__name__)

        if not max_epoch > 0:
            raise ValueError(value_error_msg('max_epoch', max_epoch, 'max_epoch > 0', TripletTrainer.DEFAULT_MAX_EPOCH))

        if not batch_size > 0:
            raise ValueError(value_error_msg('batch_size', batch_size, 'batch_size > 0',
                                             TripletTrainer.DEFAULT_BATCH_SIZE))

        if not isinstance(mode, TripletTrainer.Mode):
            if isinstance(mode, str):
                if not mode.islower():
                    mode = mode.lower()
                if mode not in TripletTrainer.MODE_LIST:
                    raise ValueError(value_error_msg('mode', mode, TripletTrainer.MODE_LIST))
                mode = TripletTrainer.Mode(mode)
            else:
                raise TypeError(type_error_msg('mode', mode, [TripletTrainer.MODE_LIST, str]))

        if not reorder_freq > 0:
            raise ValueError(value_error_msg('reorder_freq', reorder_freq, 'reorder_freq > 0',
                                             TripletTrainer.DEFAULT_REORDER_FREQ))

        # # self.writer.add_figure(mode.value, images_show(self.dataset.triplet_index[0], self.dataset[0],
        # #                                                mode.value, 0))
        self.model.train()  # for dropout & batch normalization
        if self.start_epoch == 0:
            self.save_model(0)
        loss_history = []
        start = time()
        for epoch in range(self.start_epoch, self.start_epoch + max_epoch):
            running_loss = 0.0
            if epoch % reorder_freq == 0:
                self.generate_triplets(mode)
                dataloader = DataLoader(self.dataset, batch_size=batch_size,
                                        num_workers=TripletTrainer.DEFAULT_NUM_WORKER)
                self.save_triplet(epoch)
                max_iteration = ceil(len(self.dataset) / batch_size)

            print('\n=====Epoch %s=====' % (epoch + 1))
            print('{:<13} {:<6}'.format('Iteration', 'loss'))

            for i, data in enumerate(dataloader):

                now_batch_size = data[0].size()[0]

                self.optimizer.zero_grad()

                outputs = self.model(*data)

                loss = self.criterion(*outputs)

                loss.backward()

                self.optimizer.step()

                running_loss += loss.item() * now_batch_size

                print('{:<4}/{:<8} {:.2f}'.format(i + 1, max_iteration, loss.item()))

            epoch_loss = running_loss / len(self.dataset)
            loss_history.append(epoch_loss)
            now_time_elapsed = time() - start
            self.save_model(epoch + 1)
            print('Epoch ({}/{})\nloss {:.4f}, time elapsed {}m {}s'.format(epoch + 1, self.start_epoch + max_epoch,
                                                                            epoch_loss,
                                                                            int(now_time_elapsed // 60),
                                                                            int(now_time_elapsed % 60)))
        loss_path = self.config[self.name.value]['loss_path']
        loss_dir = dirname(loss_path)
        if not exists(loss_dir):
            makedirs(loss_dir)
        if self.start_epoch == 0 or not exists(loss_path):
            savemat(loss_path, {'training_loss': loss_history})
        else:
            last_loss_history = loadmat(loss_path)['training_loss'].reshape(-1).tolist()
            savemat(loss_path, {'training_loss': last_loss_history + loss_history})

    def load_model(self, model_name):
        self.model.load_state_dict(torch.load(self.config[self.name.value]['model_format']
                                              % (model_name, self.start_epoch)))

    def save_model(self, index):
        torch.save(self.model.state_dict(), self.model_path % index)

    def save_triplet(self, index):
        savemat(self.triplet_path % index, {'index': self.dataset.triplet_index})
