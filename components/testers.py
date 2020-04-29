from datasets.dataset_processors import ExtendedDataset
from models.model import IdentificationModel, ResNet50
from models.siamese import SiameseNet, MssNet
from base import BaseExecutor
from utils.utilities import type_error_msg, value_error_msg, timer, load_model
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from configparser import ConfigParser
from os import makedirs
from os.path import join, exists, dirname
from scipy.io import savemat
import numpy as np


class Tester(BaseExecutor):
    """A general tester for all.

    Args:
        config (ConfigParser): The ConfigParser which reads setting files.
        name (str): A name defined in base.py.
        dataset (str): A dataset defined in base.py.
        model (str): A model defined in base.py.
        epoch (int): The epoch of the saved trained model for testing.
        scene (str): A scene defined in base.py.

    Attributes:
        test_path (str): Path to save features/labels/cams.

    """
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_NUM_WORKER = 8

    def __init__(self, config, name, dataset, model, epoch: int, scene):
        if not isinstance(name, Tester.Name):
            if isinstance(name, str):
                if not name.islower():
                    name = name.lower()
                if name not in Tester.NAME_LIST:
                    raise ValueError(value_error_msg('name', name, Tester.NAME_LIST))
                name = Tester.Name(name)
            else:
                raise TypeError(type_error_msg('name', name, [Tester.Name, str]))

        if not isinstance(dataset, Tester.Dataset):
            if isinstance(dataset, str):
                if not dataset.islower():
                    dataset = dataset.lower()
                if dataset not in Tester.DATASET_LIST:
                    raise ValueError(value_error_msg('dataset', dataset, Tester.DATASET_LIST))
                dataset = Tester.Dataset(dataset)
            else:
                raise TypeError(type_error_msg('dataset', dataset, [Tester.Dataset, str]))

        if not isinstance(model, Tester.Model):
            if isinstance(model, str):
                if not model.islower():
                    model = model.lower()
                if model not in Tester.MODEL_LIST:
                    raise ValueError(value_error_msg('model', model, Tester.MODEL_LIST))
                model = Tester.Model(model)
            else:
                raise TypeError(type_error_msg('model', model, [Tester.MODEL_LIST, str]))

        if not isinstance(scene, Tester.Scene):
            if isinstance(scene, str):
                if not scene.islower():
                    scene = scene.lower()
                if scene not in Tester.SCENE_LIST:
                    raise ValueError(value_error_msg('scene', scene, Tester.SCENE_LIST))
                scene = Tester.Scene(scene)
            else:
                raise TypeError(type_error_msg('scene', scene, [Tester.SCENE_LIST, str]))

        if not isinstance(epoch, int):
            raise TypeError(type_error_msg('epoch', epoch, [int]))
        if not epoch >= 0:
            raise ValueError(value_error_msg('epoch', epoch, 'epoch >= 0'))

        self.name = name
        self.dataset = dataset
        self.model = model
        self.scene = scene
        self.config = config

        self.train_class = config.getint(self.name.value, 'train_class')

        # initialize model
        model_name = self.model.value
        if self.model == Tester.Model.MSSNET:
            self.model = MssNet(self.config)
        elif self.model == Tester.Model.RESNET50:
            self.model = ResNet50(self.config, self.train_class, False)
        # else:
        #     raise ValueError(value_error_msg('model', model, Tester.MODEL_LIST))

        transform_list = []

        if self.name == Tester.Name.MARKET1501:
            transform_list = [
                # transforms.Resize((160, 64)),
                # transforms.Pad(10),
                # transforms.RandomCrop((160, 64)),
                # transforms.RandomHorizontalFlip(),
                # transforms.ToTensor()

                transforms.Resize((256, 128), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]

        self.dataset_type = ['gallery', 'query']
        if self.scene == Tester.Scene.MULTI_SHOT:
            self.dataset_type.append('multi_query')

        # prepare datasets
        if self.dataset == Tester.Dataset.EXTENDED:
            self.dataset = {}
            for item in self.dataset_type:
                self.dataset[item] = ExtendedDataset(self.name.value,
                                                     join(self.config[self.name.value]['dataset_dir'], item),
                                                     transforms.Compose(transform_list))
        else:
            raise ValueError(value_error_msg('dataset', dataset, Tester.Dataset.EXTENDED))

        # load weights
        load_model(self.model, self.config[self.name.value]['model_format'] % (model_name, epoch))
        if isinstance(self.model, IdentificationModel):
            self.model.set_to_test()

        self.test_path = self.config[self.name.value]['test_path'] % self.scene.value

    @timer
    def run(self):
        """
        Reads: A pth file of model's state dict.

        Processes: Computes the features of gallery and query imgs.

        Writes: A mat file of saved gallery and query info.

        """

        Tester.run_info(self.__class__.__name__, self.scene.value)

        self.model.eval()  # for batch norm

        test_dict = {}
        dataloader = {}
        with torch.no_grad():
            if self.scene == Tester.Scene.SINGLE_SHOT:
                for item in self.dataset_type:
                    dataloader[item] = DataLoader(self.dataset[item],
                                                  num_workers=Tester.DEFAULT_NUM_WORKER,
                                                  batch_size=Tester.DEFAULT_BATCH_SIZE)
                    test_dict[item + '_feature'] = Tester.normalize(self.extract_feature(dataloader[item]))
                    test_dict[item + '_label'] = self.dataset[item].ids
                    test_dict[item + '_cam'] = self.dataset[item].cams
            elif self.scene == Tester.Scene.MULTI_SHOT:
                item = 'gallery'
                dataloader[item] = DataLoader(self.dataset[item], num_workers=Tester.DEFAULT_NUM_WORKER,
                                              batch_size=Tester.DEFAULT_BATCH_SIZE)
                test_dict[item + '_feature'] = Tester.normalize(self.extract_feature(dataloader[item]))
                test_dict[item + '_label'] = self.dataset[item].ids
                test_dict[item + '_cam'] = self.dataset[item].cams

                item = 'multi_query'  # no need to save multi_query features into dict
                dataloader[item] = DataLoader(self.dataset[item], num_workers=Tester.DEFAULT_NUM_WORKER,
                                              batch_size=Tester.DEFAULT_BATCH_SIZE)
                multi_query_feature = self.extract_feature(dataloader[item])  # no normalization
                multi_query_label = self.dataset[item].ids
                multi_query_cam = self.dataset[item].cams

                item = 'query'
                test_dict[item + '_label'] = self.dataset[item].ids
                test_dict[item + '_cam'] = self.dataset[item].cams
                test_dict[item + '_feature'] = Tester.normalize_numpy(Tester.mean_feature(multi_query_feature,
                                                                      np.asarray(multi_query_label),
                                                                      np.asarray(multi_query_cam),
                                                                      np.asarray(test_dict['query_label']),
                                                                      np.asarray(test_dict['query_cam']),
                                                                      test_dict))
        test_dir = dirname(self.test_path)
        if not exists(test_dir):
            makedirs(test_dir)
        # WARNING: save test.mat will trigger overwrite if test.mat is already exists
        savemat(self.test_path, test_dict)

    @staticmethod
    def mean_feature(mquery_feature, mquery_label, mquery_cam, query_label, query_cam, dictionary):
        """Averages multi query feature to get (mean) query feature.

        Args:
            mquery_feature (np.ndarray): The feature of multi query imgs, shape(#multi_query, embedding_dim).
            mquery_label (np.ndarray): The people labels of multi query imgs, an 1d int array, shape(#multi_query).
            mquery_cam (np.ndarray): The camera labels of multi query imgs, an 1d int array, shape(#multi_query).
            query_label (np.ndarray): The people labels of query imgs, an 1d int array, shape(#query).
            query_cam (np.ndarray): The camera labels of query imgs, an 1d int array, shape(#query).
            dictionary (dict): A mutable dictionary for adding
                                {'multi_index': [index_array1, index_array2, ...]} (Implicit returns).
        Returns:
            query_feature (ndarray): The mean feature of mquery_feature.

        """
        query_feature = []
        multi_index = []
        for i in range(len(query_label)):
            label_mask = mquery_label == query_label[i]
            cam_mask = mquery_cam == query_cam[i]
            index = np.flatnonzero(label_mask & cam_mask)
            multi_index.append(index)
            query_feature.append(np.mean(mquery_feature[index, :], axis=0))
        dictionary['multi_index'] = multi_index
        return np.asarray(query_feature)

    @staticmethod
    def flip_lr(img: torch.Tensor):
        """Flips image tensor horizontally.

        Args:
            img (torch.Tensor): The original image tensor.

        Returns:
            img_flip (torch.Tensor): The flipped image tensor.

        """
        inv_idx = torch.arange(img.size(3) - 1, -1, -1)  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    @staticmethod
    def normalize(x: torch.Tensor):
        """Normalizes the 2d torch tensor.

        Args:
            x (torch.Tensor): in 2d.

        Returns:
            normalized_x (torch.Tensor): in 2d.

        """
        xnorm = torch.norm(x, p=2, dim=1, keepdim=True)
        return x.div(xnorm.expand_as(x))

    @staticmethod
    def normalize_numpy(x: np.ndarray):  # 25% faster than normalize with torch above
        """Normalizes the 2d numpy array.

        Args:
            x (np.ndarray): in 2d.

        Returns:
            normalized_x (np.ndarray): in 2d.

        """
        xnorm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.repeat(xnorm, x.shape[1]).reshape(x.shape)

    def extract_feature(self, dataloader):
        """Extracts feature in batches.

        Args:
            dataloader (torch.utils.data.DataLoader): Initialized dataloader.

        Returns:
            feature (np.ndarray): shape(#gallery/query/multi_query, embedding_dim).

        """
        feature = []
        if isinstance(self.model, SiameseNet):
            for i, data in enumerate(dataloader):
                batch_feature = self.model.forward_once(data)
                feature.append(batch_feature)
        elif isinstance(self.model, IdentificationModel):
            for i, data in enumerate(dataloader):
                print(i * Tester.DEFAULT_BATCH_SIZE)
                batch_feature = self.model.forward(data)
                data = Tester.flip_lr(data)
                batch_feature += self.model.forward(data)
                feature.append(batch_feature)
        return torch.cat(feature, 0).numpy()
