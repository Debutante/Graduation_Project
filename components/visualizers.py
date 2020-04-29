from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from datasets.dataset_processors import ExtendedDataset
from models.model import ResNet50
from models.siamese import MssNet
from base import BaseExecutor
from utils.utilities import type_error_msg, value_error_msg, timer, load_model
from configparser import ConfigParser
from os.path import join
import matplotlib.pyplot as plt
from scipy.io import loadmat
from collections import Counter, OrderedDict, Iterable
from shutil import rmtree
from math import ceil
import torch
import numpy as np


class Visualizer(BaseExecutor):
    """A general visualizer.

    Args:
        config (ConfigParser): The ConfigParser which reads setting files.
        name (str): A name defined in base.py.
        model (str): A model defined in base.py.
        pretrain (bool): Use saved pretrained models or not.
        epoch (int): The epoch of the saved trained model for visualizing.
        split (str): A split protocol defined in base.py.
        scene (str): A scene defined in base.py.
        query_list (list): A list of query indices.
        length (int): The length of ranking list.

    Attributes:
        writer (SummaryWriter): The writer writes out events and summaries to the event file.
        train_path (str): Path to save loss/error in every epoch.
        test_path (str): Path to the dictionary of gallery and query info.
        evaluation_path (str): Path to the dictionary of evaluation indicators and rank lists.

    """
    DEFAULT_LIST_LENGTH = 10

    def __init__(self, config, name, model, pretrain, epoch, split, scene, query_list: list, length: int):
        if not isinstance(name, Visualizer.Name):
            if isinstance(name, str):
                if not name.islower():
                    name = name.lower()
                if name not in Visualizer.NAME_LIST:
                    raise ValueError(value_error_msg('name', name, Visualizer.NAME_LIST))
                name = Visualizer.Name(name)
            else:
                raise TypeError(type_error_msg('name', name, [Visualizer.Name, str]))

        if not isinstance(model, Visualizer.Model):
            if isinstance(model, str):
                if not model.islower():
                    model = model.lower()
                if model not in Visualizer.MODEL_LIST:
                    raise ValueError(value_error_msg('model', model, Visualizer.MODEL_LIST))
                model = Visualizer.Model(model)
            else:
                raise TypeError(type_error_msg('model', model, [Visualizer.Model, str]))

        if not isinstance(epoch, int):
            raise TypeError(type_error_msg('epoch', epoch, [int]))
        if not epoch >= 0:
            raise ValueError(value_error_msg('epoch', epoch, 'epoch >= 0'))

        if not isinstance(split, Visualizer.Split):
            if isinstance(split, str):
                if not split.islower():
                    split = split.lower()
                if split not in Visualizer.SPLIT_LIST:
                    raise ValueError(value_error_msg('split', split, Visualizer.SPLIT_LIST))
                split = Visualizer.Split(split)
            else:
                raise TypeError(type_error_msg('split', split, [Visualizer.Split, str]))

        if not isinstance(scene, Visualizer.Scene):
            if isinstance(scene, str):
                if not scene.islower():
                    scene = scene.lower()
                if scene not in Visualizer.SCENE_LIST:
                    raise ValueError(value_error_msg('scene', scene, Visualizer.SCENE_LIST))
                scene = Visualizer.Scene(scene)
            else:
                raise TypeError(type_error_msg('scene', scene, [Visualizer.Scene, str]))

        if not isinstance(query_list, Iterable):
            raise TypeError(type_error_msg('query_list', query_list, Iterable))

        if not isinstance(length, int):
            raise TypeError(type_error_msg('length', length, [int]))
        if not length > 0:
            raise ValueError(value_error_msg('length', length, 'length > 0',
                                             Visualizer.DEFAULT_LIST_LENGTH))

        self.config = config
        self.name = name
        self.model = model

        self.train_class = config.getint(self.name.value, 'train_class')

        # initialize model
        model_name = self.model.value
        if self.model == Visualizer.Model.RESNET50:
            self.model = ResNet50(self.config, self.train_class, False)
        elif self.model == Visualizer.Model.MSSNET:
            self.model = MssNet(self.config)
        else:
            raise ValueError(value_error_msg('model', model, [Visualizer.Model.RESNET50]))

        # load weights
        load_model(self.model, self.config[self.name.value]['model_format'] % (model_name, epoch))

        self.split = split
        self.scene = scene
        self.query_list = query_list
        self.length = length
        self.tensorboard_dir = self.config[self.name.value]['tensorboard_dir']

        # WARNING: the log files won' t be saved
        rmtree(self.tensorboard_dir, ignore_errors=True)

        self.writer = SummaryWriter(join(self.tensorboard_dir, self.name.value))

        # dataset loading
        self.dataset_type = ['gallery', 'query']
        if self.scene == Visualizer.Scene.MULTI_SHOT:
            self.dataset_type.append('multi_query')
        self.dataset = {}
        for item in self.dataset_type:
            self.dataset[item] = ExtendedDataset(self.name.value,
                                                 join(self.config[self.name.value]['dataset_dir'], item))
        self.suffix = 'pretrain' if pretrain else 'no_pretrain'
        self.train_path = self.config[self.name.value]['train_path'] % self.suffix
        self.test_path = self.config[self.name.value]['test_path'] % self.scene.value
        self.evaluation_path = self.config[self.name.value]['evaluation_path'] % self.scene.value

    def draw_projection(self):
        """
        Reads: A mat file of saved gallery and query info.

        Processes: Draws images' embedding to illustrate clustering.

        Writes: Log files.

        """
        class_dict = OrderedDict(Counter(self.dataset['gallery'].ids))
        # {class1: num1, class2: num2, ...}(dict) <-
        # "the dictionary made up of the class and the corresponding indices num"
        class_num = tuple(class_dict.values())
        # (num1, num2, ...)(tuple) <- "the corresponding indices num of each class"
        class_index_range = np.cumsum(np.asarray((0,) + class_num)).tolist()
        # [0, num1, num1+num2, ...](list) <- "the starting and ending indices of each class, like ticks"
        chosen_index = []
        for i in class_index_range[2: 32]:  # skip id -1(stands for junk image) and id 0(stands for distractor)
            chosen_index.extend(list(range(i, i + 3)))
        images = self.dataset['gallery'][chosen_index]
        labels = torch.as_tensor(self.dataset['gallery'].ids)[chosen_index]
        features = loadmat(self.test_path)['gallery_feature'][chosen_index]
        horizontal_pad = (max(images[0].size) - min(images[0].size)) // 2
        tranform_list = [
            transforms.Pad((horizontal_pad, 0, horizontal_pad, 0), (255, 255, 255)),
            # pad image in 2:1 to 1:1,
            # note that using white to pad on the left and the right will cause unwanted occlusion
            # when image density is higher,
            # one way to cope is convert image to RGBA mode and set A = 100 for padded pixels, (but it is troublesome)
            # transforms.CenterCrop(max(images[0].size)),
            # this way will create image edged with black(really unbearable for me)
            transforms.ToTensor()
        ]
        square_images = [transforms.Compose(tranform_list)(image) for image in images]
        self.writer.add_embedding(features, metadata=labels, label_img=torch.stack(square_images))
        self.writer.close()

    def draw_model(self):
        """
        Reads: None.

        Processes: Draws model's structure.

        Writes: Log files.

        """
        one_image = transforms.ToTensor()(self.dataset['query'][0]).unsqueeze(0)  # Any input image tensor.
        self.writer.add_graph(self.model, one_image)
        self.writer.close()

    @staticmethod
    def heatmap(ax, x):
        """
        Reads: None.

        Processes: Draws heatmap of one layer.

        Writes: Log files.

        """
        heatmap_numpy = x.squeeze().sum(dim=0).numpy()
        ax.set_xlabel('{}*{}'.format(heatmap_numpy.shape[0], heatmap_numpy.shape[1]), fontsize=8)
        return ax.imshow(heatmap_numpy, cmap='viridis')

    def draw_heatmap(self, query_list):
        """
        Reads: None.

        Processes: Draws heatmap of a specific model.

        Writes: Log files.

        """
        if isinstance(self.model, ResNet50):
            length = 6
            structure_list = ['original', 'conv1', 'maxpool', 'layer1', 'layer2', 'layer3']
            for j, query in enumerate(query_list):
                query_img = self.dataset['query'][query]
                query_label = self.dataset['query'].ids[query]
                fig, ax = plt.subplots(ncols=length, constrained_layout=False)
                ax[0].imshow(query_img)
                for i, one_ax in enumerate(ax):
                    one_ax.set_xticks([])
                    one_ax.set_yticks([])
                    if i == 0:
                        one_ax.set_title('query')
                    else:
                        one_ax.set_title(structure_list[i])
                x = transforms.ToTensor()(query_img).unsqueeze(0)
                x = self.model.model.conv1(x)  # 1
                Visualizer.heatmap(ax[1], x)
                x = self.model.model.bn1(x)
                x = self.model.model.relu(x)
                x = self.model.model.maxpool(x)  # 2
                Visualizer.heatmap(ax[2], x)
                x = self.model.model.layer1(x)  # 3
                Visualizer.heatmap(ax[3], x)
                x = self.model.model.layer2(x)  # 4
                Visualizer.heatmap(ax[4], x)
                x = self.model.model.layer3(x)  # 5
                heatmap = Visualizer.heatmap(ax[5], x)
                fig.colorbar(heatmap, pad=0.15)
                fig.suptitle(query_label)
                self.writer.add_figure('query_through_model', fig, j)
        else:
            raise ValueError(value_error_msg('model', self.model, ResNet50))
        self.writer.close()

    def draw_loss(self):
        """
        Reads: A mat file of losses/errors.

        Processes: Draws loss curve.

        Writes: Log files.

        """
        mat_dict = loadmat(self.train_path)
        training_loss = mat_dict['training_loss'].reshape(-1)
        if self.split == Visualizer.Split.TRAIN_ONLY:
            for epoch, loss in enumerate(training_loss):
                self.writer.add_scalar('training loss', loss, epoch)
        elif self.split == Visualizer.Split.TRAIN_VAL:
            validation_loss = mat_dict['validation_loss'].reshape(-1)
            for epoch, train_val in enumerate(zip(training_loss, validation_loss)):
                self.writer.add_scalars('loss', {
                    'train': train_val[0],
                    'val': train_val[1]
                }, epoch)
        self.writer.close()

    def draw_error(self):
        """
        Reads: A mat file of losses/errors.

        Processes: Draws error curve.

        Writes: Log files.

        """
        mat_dict = loadmat(self.train_path)
        training_error = mat_dict['training_error'].reshape(-1)
        if self.split == Visualizer.Split.TRAIN_ONLY:
            for epoch, error in enumerate(training_error):
                self.writer.add_scalar('training error', error, epoch)
        elif self.split == Visualizer.Split.TRAIN_VAL:
            validation_error = mat_dict['validation_error'].reshape(-1)
            for epoch, train_val in enumerate(zip(training_error, validation_error)):
                self.writer.add_scalars('error', {
                    'train': train_val[0],
                    'val': train_val[1]
                }, epoch)
        self.writer.close()

    def draw_CMC(self):
        """
        Reads: A mat file of evaluation indicators and rank lists.

        Processes: Draws Cumulative Match Curve.

        Writes: Log files.

        """
        CMC = loadmat(self.evaluation_path)['CMC'].reshape(-1)
        for index, cmc in enumerate(CMC):
            self.writer.add_scalar('CMC', cmc, index)
        self.writer.close()

    def draw_query_image(self, query_list):
        """
        Args:
            query_list (list): The list of query indices.

        Reads: A mat file of saved gallery and query info.

        Processes: Draws query images.

        Writes: Log files.

        """
        if self.scene == Visualizer.Scene.SINGLE_SHOT:
            for j, query in enumerate(query_list):
                query_img = self.dataset['query'][query]
                query_label = self.dataset['query'].ids[query]
                fig, ax = plt.subplots()
                ax.imshow(query_img)
                ax.set_xticks([])
                ax.set_yticks([])  # serve as thin black border
                fig.suptitle(query_label)
                self.writer.add_figure('query', fig, j)
        elif self.scene == Visualizer.Scene.MULTI_SHOT:
            multi_index = loadmat(self.test_path)['multi_index'].reshape(-1)[query_list]
            for i, indices in enumerate(multi_index):
                indices = indices.reshape(-1)
                length = indices.shape[0]
                sum_row = ceil(float(length) / Visualizer.DEFAULT_LIST_LENGTH)
                sum_column = min(Visualizer.DEFAULT_LIST_LENGTH, length)
                fig, ax = plt.subplots(sum_row, sum_column, constrained_layout=False, squeeze=False)
                for j in range(length):
                    row = j // Visualizer.DEFAULT_LIST_LENGTH
                    column = j % Visualizer.DEFAULT_LIST_LENGTH
                    one_ax = ax[row][column]
                    one_ax.set_title('%d' % (j + 1))
                    one_ax.set_xticks([])
                    one_ax.set_yticks([])
                    one_ax.imshow(self.dataset['multi_query'][indices[j]])
                column = length % sum_column
                if column != 0:
                    for j in range(column, sum_column):
                        ax[sum_row - 1][j].axis('off')  # clear not used axises on the figure
                fig.suptitle('%d@cam %d' % (self.dataset['query'].ids[query_list[i]],
                                            self.dataset['query'].cams[query_list[i]]))
                self.writer.add_figure('query list', fig, i)
        self.writer.close()

    def draw_rank_list(self, query_list, length: int = DEFAULT_LIST_LENGTH):
        """
        Args:
            query_list (list): The list of query indices.
            length (int): The length of rank list for every query.

        Reads: A mat file of evaluation indicators and rank lists.

        Processes: Draws rank list for each query.

        Writes: Log files.

        """
        evaluation_dict = loadmat(self.evaluation_path)
        index_array = evaluation_dict['index'].reshape(-1)
        ap_array = evaluation_dict['ap'].reshape(-1)
        for i, query in enumerate(query_list):
            query_img = self.dataset['query'][query]
            query_label = self.dataset['query'].ids[query]
            sum_row = ceil(float(length) / (Visualizer.DEFAULT_LIST_LENGTH + 1))
            sum_column = min(Visualizer.DEFAULT_LIST_LENGTH + 1, length + 1)
            fig, ax = plt.subplots(sum_row, sum_column, constrained_layout=False, squeeze=False)
            ax[0][0].set_xticks([])
            ax[0][0].set_yticks([])
            ax[0][0].imshow(query_img)
            ax[0][0].set_title('query')
            ax[0][0].set_xlabel(query_label, fontsize=8)
            fig.suptitle('query {}(ap={:.2f}%)'.format(query, ap_array[query] * 100))
            rank_index = index_array[query].reshape(-1)[:length]
            for j, index in enumerate(rank_index):
                gallery_img = self.dataset['gallery'][index]
                gallery_label = self.dataset['gallery'].ids[index]
                row = (j + 1) // (Visualizer.DEFAULT_LIST_LENGTH + 1)
                column = (j + 1) % (Visualizer.DEFAULT_LIST_LENGTH + 1)
                one_ax = ax[row][column]
                one_ax.set_xticks([])
                one_ax.set_yticks([])
                one_ax.imshow(gallery_img)
                if gallery_label == query_label:
                    color = 'green'
                    one_ax.set_title('%d' % (j + 1), color=color)
                    for item in one_ax.spines.values():
                        item.set_color(color)
                    one_ax.set_xlabel(gallery_label, fontsize=8, color=color)
                else:
                    color = 'red'
                    one_ax.set_title('%d' % (j + 1), color=color)
                    for item in one_ax.spines.values():
                        item.set_color(color)
                    one_ax.set_xlabel(gallery_label, fontsize=8, color=color)
            column = (length + 1) % sum_column
            if column != 0:
                for j in range(column, sum_column):
                    ax[sum_row - 1][j].axis('off')
            self.writer.add_figure('rank list', fig, i)
            # self.writer.add_figure('rank list({})'.format(self.scene.value), fig, i)
            # It is surprising that the tag string param in writer.add_figure can't be any brackets,
            # and the brackets are shown as '___'.
            self.writer.close()

    @timer
    def run(self):
        """
        Reads: A mat file of evaluation indicators and rank lists and a mat file of training loss.

        Processes: Draws pics using matplotlib.pyplot.

        Writes: Log files.

        """

        Visualizer.run_info(self.__class__.__name__, '{}, {}'.format(self.suffix, self.scene.value))
        with torch.no_grad():
            self.draw_projection()
            self.draw_model()
            self.draw_query_image(self.query_list)
            self.draw_heatmap(self.query_list)
            self.draw_rank_list(self.query_list, self.length)
            self.draw_loss()
            self.draw_error()
            self.draw_CMC()
        # os.system('tensorboard --logdir=%s' % self.tensorboard_dir)
