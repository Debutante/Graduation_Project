from utils.utilities import timer, type_error_msg, value_error_msg
from base import BaseExecutor
import torch
from torchvision.datasets import ImageFolder
from configparser import ConfigParser, ExtendedInterpolation
from collections import Counter, OrderedDict
from itertools import product
from numbers import Number
from enum import Enum, unique
from math import ceil
import numpy as np
import random

# conf_path = '../settings.txt'
# conf = ConfigParser(interpolation=ExtendedInterpolation(), default_section='Default')
# conf.read(conf_path)


class ExtendedDataset(ImageFolder):
    """Extend the function of ImageFolder to support a variety of index's types.
        Returns images(not tuples of (image, label)) only.

    Args:
        name (str): A name defined in base.py.
        root (str): Path to root directory.
        transform (callable): A function/transform that takes in a sample and returns a transformed version.

    Attributes:
        cams (list): The camera label list.
        ids (list): People id list.

    """
    INDEX_LIST = [int, slice, tuple, list, np.ndarray, torch.Tensor]

    def __init__(self, name, root, transform=None):
        super(ExtendedDataset, self).__init__(root, transform)
        if not isinstance(name, BaseExecutor.Name):
            if isinstance(name, str):
                if not name.islower():
                    name = name.lower()
                if name not in BaseExecutor.NAME_LIST:
                    raise ValueError(value_error_msg('name', name, BaseExecutor.NAME_LIST))
                name = BaseExecutor.Name(name)
            else:
                raise TypeError(type_error_msg('name', name, [BaseExecutor.Name, str]))

        if name == BaseExecutor.Name.MARKET1501:
            # 0001_c1s1_001051_00.jpg, here 0001 is peopleId, and 1 in c1 is camId
            self.cams = [int(item[0].split('_c')[1].split('s')[0]) for item in self.imgs]
            # [cam1, cam1, cam2, ...](list)
            self.ids = [int(list(self.class_to_idx.keys())[item]) for item in self.targets]
            # [id1, id1, id2, ...](list)
        # else:
        #     raise TypeError(type_error_msg('name', name, [BaseExecutor.Name, str]))

    def __getitem__(self, index):
        # derive ImageFolder to support slice, list, tuple..., return PIL image(default)
        # or tensors(if transform: toTensor)
        try:
            # equals to if isinstance(index, Number):
            return ImageFolder.__getitem__(self, index)[0]
        except (ValueError, TypeError):
            if isinstance(index, slice):
                # tuple_list: [(img, tag), (img, tag)]......
                return [ImageFolder.__getitem__(self, i)[0] for i in
                        range(super(ExtendedDataset, self).__len__())[index]]
            elif isinstance(index, (tuple, list, np.ndarray, torch.Tensor)):
                return [ImageFolder.__getitem__(self, i)[0] for i in index]
            # only use [xxx], or tuple(xxx) as in default_collate:
            # batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'generator'>
            else:
                raise TypeError(type_error_msg('index', type(index), ExtendedDataset.INDEX_LIST))


# NOTE: no longer debug/maintain the codes below since 04/20/20
class ReorganisedDataset(ExtendedDataset):
    """A general dataset where images are reorganised, e.g. into pair/triplet forms.

    Args:
        root (str): Path to root directory.
        transform (callable): A function/transform that takes in a sample and returns a transformed version.

    """
    @unique
    class Mode(Enum):
        RANDOM = 'random'
        OFFLINE = 'offline'  # mining with the most recent network checkpoint
        # TODO 'online'

    MODE_LIST = [item.value for item in Mode]
    DEFAULT_MODE = Mode.RANDOM  # random

    @unique
    class PositiveRange(Enum):
        CLASS = 'class'  # sampling inside each class
        CAM = 'cam'  # sampling inside each cam

    POSITIVE_RANGE_LIST = [item.value for item in PositiveRange]
    DEFAULT_POSITIVE_RANGE = PositiveRange.CLASS  # class

    @unique
    class NegativeRange(Enum):
        SAMPLE = 'sample'  # sampling inside chosen samples
        CLASS = 'class'  # sampling inside each class
        INDEX = 'index'  # sampling inside the whole dataset

    NEGATIVE_RANGE_LIST = [item.value for item in NegativeRange]
    DEFAULT_NEGATIVE_RANGE = NegativeRange.SAMPLE  # sample

    def __init__(self, name, root, transform):
        super(ReorganisedDataset, self).__init__(name, root, transform)


class TripletDataset(ReorganisedDataset):
    """A specific dataset where self.generate_triplets() is defined and
        self.__getitem returns triplet [img1, img2, img3].

    Args:
        config (ConfigParser): The ConfigParser which reads setting files.
        root (str): Path to root directory.
        transform (callable): A function/transform that takes in a sample and returns a transformed version.
        fixed (bool): Whether to fix the torch/numpy seed or not.

    Attributes:
        class_num (tuple): The corresponding indices num of each class.
        class_ (ndarray): The array of all classes.
        class_count (int): The num of classes.
        class_index_range (list): The starting and ending indices of each class, like ticks.
        index_count (int): The num of indices.
        index_ (ndarray): The array of all indices.
        class_cam_index (list): The list made up of the dicts comprised of the corresponding cams of the class
                                                                        and the corresponding indices of the cam.
        sampled_class_cam (list): The chosen cams of each class, respectively.
        sampled_class_index (list): The chosen indices of the chosen cam, respectively.
        anchor_class (list): The list of the anchor classes where cam >= 2.
                            Note that according to the definition of ReID,
                            the anchor's cam should be different from the positive's cam,
                            so I choose classes where cam >= 2 to be anchor classes.
        anchor_class_count (int): The num of anchor classes.
        sampled_anchor_index (list): The anchor indices chosen from anchor classes.
        triplet_index (list): The list of triplets(tuple(anchor, positive, negative)).

    """
    DEFAULT_TORCH_SEED = 0
    DEFAULT_NUMPY_SEED = 0
    DEFAULT_FIXED = True

    DEFAULT_POSITIVE_PARAM = 10

    DEFAULT_MAX_NEGATIVE = 3

    DEFAULT_MODEL = None
    DEFAULT_BATCH_SIZE = 256

    @unique
    class Sort(Enum):
        NEITHER = 'neither'  # no sort
        POS = 'pos'  # sort by pos distance in ascending order
        NEG = 'neg'  # sort by neg distance in descending order
        BOTH = 'both'  # sort by the sum rank in POS and NEG
    SORT_LIST = [item.value for item in Sort]
    DEFAULT_SORT = Sort.NEITHER  # neither

    @timer
    def __init__(self, config, name, root, transform, fixed: bool = DEFAULT_FIXED):
        # approx 0.35 - 0.55 sec
        super(TripletDataset, self).__init__(name, root, transform)
        if fixed:
            torch.manual_seed(config.getint('Default', 'torch_seed', fallback=TripletDataset.DEFAULT_TORCH_SEED))
            np.random.seed(config.getint('Default', 'numpy_seed', fallback=TripletDataset.DEFAULT_NUMPY_SEED))
        class_dict = OrderedDict(Counter(self.targets))
        # {class1: num1, class2: num2, ...}(dict) <-
        # "the dictionary made up of the class and the corresponding indices num"
        self.class_num = tuple(class_dict.values())
        # (num1, num2, ...)(tuple) <- "the corresponding indices num of each class"
        self.class_ = np.asarray(tuple(class_dict.keys()))  # [class1 class2 ...](ndarray) <- "the array of all classes"
        self.class_count = len(self.class_)  # count(class)(int) <- "the class num"
        self.class_index_range = np.cumsum(np.asarray((0, ) + self.class_num)).tolist()
        # [0, num1, num1+num2, ...](list) <- "the starting and ending indices of each class, like ticks"
        self.index_count = len(self.targets)  # count(index)(int) <- "the index/img num"
        self.index_ = np.asarray(tuple(range(self.index_count)))
        # [index1 index2 ...](ndarray) <- "the array of all indices"
        self.class_cam_index = []
        # [
        #     {
        #         cam1: [index1, index2, ...](list),
        #         cam2: [index3, index4, ...]
        #         ...
        #     }(dict),
        #     {
        #         cam1: [index5, index6, ...],
        #         cam2: [index7, index8, ...]
        #         ...
        #     },
        #     ...
        # ](list) <- "the list made up of the dicts comprised of
        # the corresponding cams of the class and the corresponding indices of the cam"
        self.sampled_class_cam = []  # [cam1, cam2, ...](list) <- "the chosen cams of each class, respectively"
        self.sampled_class_index = []
        # [index1, index7, ...](list) <- "the chosen indices of the chosen cam, respectively"
        self.anchor_class = []
        # [class1, class2, ...](list) <- "the list of the anchor classes where cam >= 2"
        for i in self.class_:
            class_i_cam = self.cams[self.class_index_range[i]: self.class_index_range[i + 1]]
            # [cam1, cam1, cam2, ...](list) <- "the cam list of the corresponding class"
            class_i_cam_dict = dict(Counter(class_i_cam))
            # {cam1: num1, cam2: num2, ...}(dict) <-
            # "the dictionary made up of the cam class and the corresponding cams num"
            class_i_selected_cam = list(class_i_cam_dict.keys())[np.argmin(np.asarray(list(class_i_cam_dict.values())))]
            # camx(int) <- "the corresponding cam class of the minimum cam num, min(num)->camx"
            self.class_cam_index.append(
                {
                    key: (np.flatnonzero(np.asarray(class_i_cam) == key) + self.class_index_range[i]).tolist()
                    # note that np.flatnonzero() returns the relative indices of the class,
                    # so we add class_index_range[i] to obtain the absolute indices of the whole dataset
                    for key in class_i_cam_dict.keys()
                }
            )
            if len(class_i_cam_dict) >= 2:
                self.anchor_class.append(i.tolist())
            self.sampled_class_cam.append(class_i_selected_cam)
            self.sampled_class_index.append(random.choice(self.class_cam_index[i][class_i_selected_cam]))
            # randomly choose one index from the indices of the chosen cam
        # self.anchor_class = self.class_[np.flatnonzero(np.asarray(self.class_num) >= 2)].tolist()
        self.anchor_class_count = len(self.anchor_class)  # count(anchor_class)(int) <- "the anchor class num"
        self.sampled_anchor_index = np.asarray(self.sampled_class_index)[self.anchor_class].tolist()
        # [index1, index7, ...](list) <- "anchor index"
        self.triplet_index = []
        # [(index1, index5, index28)(tuple), (index1, index5, index92), ...](list) < - "triplet indices"

    @timer
    def generate_triplets(self, mode: str = ReorganisedDataset.DEFAULT_MODE,
                          positive_sample_range: str = ReorganisedDataset.DEFAULT_POSITIVE_RANGE,
                          max_positive_param: int = DEFAULT_POSITIVE_PARAM,
                          negative_sample_range: str = ReorganisedDataset.DEFAULT_NEGATIVE_RANGE,
                          max_negative: int = DEFAULT_MAX_NEGATIVE,
                          model=DEFAULT_MODEL,
                          batch_size: int = DEFAULT_BATCH_SIZE,
                          sort_by: str = DEFAULT_SORT):
        """Triplet_index generator.

        Ways to Generate triplets:
        OFF: randomly sample triplets.
        OFFLINE(every n steps): using the most recent network checkpoint
        and computing the argmin and argmax on a subset of the data.
        ONLINE: selecting the hard positive/negative exemplars from within a mini-batch.

        Reference: F. Schroff, D. Kalenichenko, and J. Philbin.
                    FaceNet: A Unified Embedding for Face Recognition and Clustering. In CVPR, 2015.

        Args:
            mode (str): A mode defined in base class ReorganisedDataset.
            positive_sample_range (str): A positive range defined in base class ReorganisedDataset.
            max_positive_param (int): An integer acts as denominator in
                                    #sampled positive indices = #anchor class // max_positive_param + 1.
            negative_sample_range (str): A negative range defined in base class ReorganisedDataset.
            max_negative (int): An integer acts as the max #sampled negative indices in
                                    #sampled negative indices = min(#negative class, max_negative)
            model (nn.Module or None):
            batch_size (int):
            sort_by (str): A sort defined above.

        """
        # self.used_by = 'user'
        if not isinstance(mode, TripletDataset.Mode):
            if isinstance(mode, str):
                if not mode.islower():
                    mode = mode.lower()
                if mode not in TripletDataset.MODE_LIST:
                    raise ValueError(value_error_msg('mode', mode, TripletDataset.MODE_LIST,
                                                     TripletDataset.DEFAULT_MODE))
                mode = TripletDataset.Mode(mode)
            else:
                raise TypeError(type_error_msg('mode', mode, [TripletDataset.Mode, str]))

        if not isinstance(positive_sample_range, TripletDataset.PositiveRange):
            if isinstance(positive_sample_range, str):
                if not positive_sample_range.islower():
                    positive_sample_range = positive_sample_range.lower()
                if positive_sample_range not in TripletDataset.POSITIVE_RANGE_LIST:
                    raise ValueError(value_error_msg('positive_sample_range', positive_sample_range,
                                                     TripletDataset.POSITIVE_RANGE_LIST,
                                                     TripletDataset.DEFAULT_POSITIVE_RANGE))
                positive_sample_range = TripletDataset.PositiveRange(positive_sample_range)
            else:
                raise TypeError(type_error_msg('positive_sample_range', positive_sample_range,
                                               [TripletDataset.PositiveRange, str]))

        if not isinstance(negative_sample_range, TripletDataset.NegativeRange):
            if isinstance(negative_sample_range, str):
                if not negative_sample_range.islower():
                    negative_sample_range = negative_sample_range.lower()
                if negative_sample_range not in TripletDataset.NEGATIVE_RANGE_LIST:
                    raise ValueError(value_error_msg('negative_sample_range', negative_sample_range,
                                                     TripletDataset.NEGATIVE_RANGE_LIST,
                                                     TripletDataset.DEFAULT_NEGATIVE_RANGE))
                negative_sample_range = TripletDataset.NegativeRange(negative_sample_range)
            else:
                raise TypeError(type_error_msg('negative_sample_range', negative_sample_range,
                                               [TripletDataset.NegativeRange, str]))

        if not max_positive_param >= 1:
            raise ValueError(value_error_msg('max_positive_param', max_positive_param,
                                             'max_positive_param >= 1', TripletDataset.DEFAULT_POSITIVE_PARAM))

        if not max_negative >= 1:
            raise ValueError(value_error_msg('max_negative', max_negative,
                                             'max_negative >= 1', TripletDataset.DEFAULT_MAX_NEGATIVE))

        if not batch_size >= 1:
            raise ValueError(value_error_msg('batch_size', batch_size,
                                             'batch_size >= 1', TripletDataset.DEFAULT_BATCH_SIZE))

        if not isinstance(sort_by, TripletDataset.Sort):
            if isinstance(sort_by, str):
                if not sort_by.islower():
                    sort_by = sort_by.lower()
                if sort_by not in TripletDataset.SORT_LIST:
                    raise ValueError(value_error_msg('sort_by', sort_by, TripletDataset.SORT_LIST,
                                                     TripletDataset.DEFAULT_SORT))
                sort_by = TripletDataset.Sort(sort_by)
            else:
                raise TypeError(type_error_msg('sort_by', sort_by, [TripletDataset.Sort, str]))

        if mode == TripletDataset.Mode.RANDOM:
            self.__random_selecting(positive_sample_range, max_positive_param, negative_sample_range, max_negative)
        elif mode == TripletDataset.Mode.OFFLINE:
            if not isinstance(model, torch.nn.Module):
                raise TypeError(type_error_msg('model', model, [torch.nn.Module]))
            self.__offline_mining(model, batch_size, positive_sample_range, max_positive_param,
                                  negative_sample_range, max_negative, sort_by)
        else:  # equals to elif mining mode == 'online'
            raise ValueError(value_error_msg('mode', mode, TripletDataset.MODE_LIST, TripletDataset.DEFAULT_MODE))

    @timer
    def __random_selecting(self, positive_sample_range, max_positive_param, negative_sample_range, max_negative):
        """Generates random triplets by randomly selecting positive and negative indices.
            Implicitly returns triplet_index.

        Args:
            positive_sample_range (str): A positive range defined in base class ReorganisedDataset.
            max_positive_param (int): An integer acts as denominator in
                                    #sampled positive indices = #anchor class // max_positive_param + 1.
            negative_sample_range (str): A negative range defined in base class ReorganisedDataset.
            max_negative (int): An integer acts as the max #sampled negative indices in
                                    #sampled negative indices = min(#negative class, max_negative)

        """
        random_positive_index = self.__random_positive(positive_sample_range, max_positive_param)
        # [[index1, index3, ...](list), [index5, index8, ...], ...](list) <- "positive indices"
        random_negative_index = self.__random_negative(negative_sample_range, max_negative)
        # [[index9, index11, ...](list), [index1, index4, ...], ...](list) <- "negative indices"
        self.triplet_index = [(self.sampled_anchor_index[i], ) + j for i in range(len(random_positive_index))
                              for j in list(product(random_positive_index[i], random_negative_index[i]))]
        # [(index1, index5, index28)(tuple), (index1, index5, index92), ...](list) <- "triplet indices"
        # first do a cartesian product on positive indices and negative indices,
        # then add corresponding anchor indices into tuples to get triplets

    @timer
    def __random_positive(self, sample_range, positive_param):
        random_positive_index = []
        # [[index1, index3, ...](list), [index5, index8, ...], ...](list) <- "positive indices"
        if sample_range == TripletDataset.PositiveRange.CLASS:
            for item in self.anchor_class:
                index_range = np.asarray(
                    tuple(range(self.class_index_range[item], self.class_index_range[item + 1])))
                # [index1 index2 ...](ndarray) <- "the index range of positive examples"
                cam_range = np.asarray(self.cams[self.class_index_range[item]: self.class_index_range[item + 1]])
                # [cam1 cam1 ...](ndarray) <- "the cam range of positive examples"
                sampled_index_range = index_range[cam_range != self.sampled_class_cam[item]].tolist()
                # [index3, index4, ...](list) <- "the index range of available positive examples"
                # available: the positive pic should be taken from a cam
                # which is not the same as the anchor img's cam(in cam_range != sampled_class_cam[i])
                random_positive_index.append(
                    random.sample(sampled_index_range, min(self.class_num[item] // positive_param + 1,
                                                           len(sampled_index_range))))
                # randomly choose at most len(anchor class num) // positive_param + 1 positive pics' indices from
                # available positive examples' indices. note that no repeat indices should be chosen
        elif sample_range == TripletDataset.PositiveRange.CAM:  # equals to elif sample_range == cam:
            for item in self.anchor_class:
                cams = np.asarray(tuple(self.class_cam_index[item].keys()))
                # [cam1 cam2 ...](ndarray) <- "the cam range of the anchor class"
                sampled_cams = cams[cams != self.sampled_class_cam[item]]
                # [cam2 ...](ndarray) <-
                # "the cam range of the anchor class except the cam of the sampled anchor index"
                random_indices = []
                # [index1, index3, ...](list) <- "the random indices of a certain anchor class"
                for cam in sampled_cams:
                    indices = self.class_cam_index[item][cam]  # [index1, index2, ...](list) <-
                    # "the indices given anchor class and cam"
                    random_indices.extend(random.sample(
                        indices, min(len(indices) // positive_param + 1, len(indices))))
                    # randomly choose at most len(indices given class and cam) // positive_param + 1
                    # positive pics' indices from the positive class.
                    # note that no repeat indices should be chosen
                random_positive_index.append(random_indices)
        # else:
        #     raise ValueError(value_error_msg(value_error_msg('positive_sample_range', sample_range,
        #                                                      TripletDataset.POSITIVE_RANGE_LIST)))
        return random_positive_index

    @timer
    def __random_negative(self, sample_range, max_negative):
        random_negative_index = []
        # [[index9, index11, ...](list), [index1, index4, ...], ...](list) <- "negative indices"
        if sample_range == TripletDataset.NegativeRange.SAMPLE:
            random_negative_index = [[self.sampled_class_index[j]
                                     for j in np.random.choice(self.class_[self.class_ != i],
                                     min(max_negative, len(self.class_[self.class_ != i])), replace=False)]
                                     for i in self.anchor_class]
        elif sample_range == TripletDataset.NegativeRange.CLASS:  # equals to elif sample_range == 'class':
            random_negative_index = [[random.randint(self.class_index_range[j], self.class_index_range[j + 1])
                                     for j in np.random.choice(self.class_[self.class_ != i],
                                     min(max_negative, len(self.class_[self.class_ != i])), replace=False)]
                                     for i in self.anchor_class]
        # randomly choose at most max_negative pics' indices from negative classes.
        # note that negative classes should be different from the anchor class(in class_ != i)
        # and negative classes could be different from each other(in replace=False) for diversity
        elif sample_range == TripletDataset.NegativeRange.INDEX:  # equals to elif sample_range == 'index':
            for item in self.anchor_class:
                indices_mask = np.ones(self.index_count, dtype=np.bool)
                # [True True ...](ndarray) <- "the all true array w.r.t all indices"
                positive_indices = list(range(self.class_index_range[item], self.class_index_range[item + 1]))
                # [index4, index5, ...](list) <- "the positive indices of class item"
                indices_mask[positive_indices] = False
                # [True False ...](ndarray) <- "the bool array w.r.t all negative indices"
                random_negative_index.append(np.random.choice(self.index_[indices_mask],
                                             min(max_negative, self.index_count - self.class_num[item]), replace=False))
        # else:
        #     raise ValueError(value_error_msg('negative_sample_range', sample_range, TripletDataset.NEGATIVE_RANGE_LIST))
        return random_negative_index

    @timer
    def __offline_mining(self, model, batch_size, positive_sample_range, max_positive_param,
                         negative_sample_range, max_negative, sort_by):
        """Generates triplets by offline mining. Implicitly returns triplet_index.

        Args:
            model (nn.Module or None):
            batch_size (int):
            positive_sample_range (str): A positive range defined in base class ReorganisedDataset.
            max_positive_param (int): An integer acts as denominator in
                                    #sampled positive indices = #anchor class // max_positive_param + 1.
            negative_sample_range (str): A negative range defined in base class ReorganisedDataset.
            max_negative (int): An integer acts as the max #sampled negative indices in
                                    #sampled negative indices = min(#negative class, max_negative)
            sort_by (str): A sort defined above.

        """
        with torch.no_grad():
            # approx 165 - 175 sec
            index_feature = torch.as_tensor([])
            # [[dim1, dim2, ... in index1], [dim1, dim2, ... in index2], ...](Tensor) <-
            # "the model output matrix of every imgs"
            for i in range(ceil(super(TripletDataset, self).__len__() / batch_size)):
                batch_feature = model.forward_once(super(TripletDataset, self).__getitem__(
                    slice(batch_size * i, min(batch_size * (i + 1), super(TripletDataset, self).__len__()))))
                index_feature = torch.cat((index_feature, batch_feature), 0)
            sampled_anchor_feature = index_feature[self.sampled_anchor_index]
            # [[dim1, dim2, ... in anchor1], [dim1, dim2, ... in anchor2], ...](Tensor) <-
            # "the model output matrix of sampled anchors"
            dist = TripletDataset.__feature_distance(sampled_anchor_feature, index_feature)  # approx 0.2 sec
            # [[dist11, dist12, ...], [dist21, dist22, ...], ...](Tensor) <-
            # "the distance between features of anchors and indices"
            sampled_positive_index, sampled_positive_value = self.__positive_sampling(positive_sample_range,
                                                                                      max_positive_param, dist)
            # [[index1, index3, ...](list), [index5, index8, ...], ...](list) <- "positive indices"
            # [[value1, value3, ...](list), [value5, value8, ...], ...](list) <-
            # "the distances between features of positive indices and anchor indices"
            sampled_negative_index, sampled_negative_value = self.__negative_sampling(negative_sample_range,
                                                                                      max_negative, dist)
            # [[index9, index11, ...](list), [index1, index4, ...], ...](list) <- "negative indices"
            # [[value9, value11, ...](list), [value1, value4, ...], ...](list) <-
            # "the distances between features of negative indices and anchor indices"
            self.triplet_index = [(self.sampled_anchor_index[i], ) + j for i in range(self.anchor_class_count)
                                  for j in list(product(sampled_positive_index[i], sampled_negative_index[i]))]
            # [(index1, index5, index28), (index1, index5, index92), ...](list) <- "triplet indices"
            # first do a cartesian product on positive indices and negative indices,
            # then add corresponding anchor indices into tuples to get triplets
            if sort_by in [TripletDataset.Sort.POS, TripletDataset.Sort.NEG, TripletDataset.Sort.BOTH]:
                # rearrange triplets from easy to hard:
                # the differences between features of anchor indices and positive indices should go from small to large,
                # the differences between features of anchor indices and negative indices should go from large to small.
                tuple_value = [j for i in range(self.anchor_class_count)
                               for j in list(product(sampled_positive_value[i], sampled_negative_value[i]))]
                # [(pos_value5, neg_value28), (pos_value5, neg_value92), ...](list) <- "tuple values"
                # a cartesian product on values corresponding to positive indices and negative indices
                if sort_by == TripletDataset.Sort.POS:
                    inverse_index = sorted(enumerate(tuple_value), key=lambda x: x[1][0])
                    # [(index3, (pos_value3, neg_value3)(tuple))(tuple), (index9, (pos_value9, neg_value9)), ...](list)
                    # <- "the tuple of indices and corresponding pos&neg value tuples,
                    # sort by pos_value in ascending order"
                elif sort_by == TripletDataset.Sort.NEG:
                    inverse_index = sorted(enumerate(tuple_value), key=lambda x: x[1][1], reverse=True)
                    # [(index7, (pos_value7, neg_value7)(tuple))(tuple), (index4, (pos_value4, neg_value4)), ...](list)
                    # <- "the tuple of indices and corresponding pos&neg value tuples,
                    # sort by neg_value in descending order"
                elif sort_by == TripletDataset.Sort.BOTH:  # elif sort_by == 'both':  approx 0.02 - 0.04 sec
                    pos_index = [-1] * len(tuple_value)
                    # [-1, -1, ...](list) <-
                    # "the rank in values corresponding to positive indices, sort by indices in ascending order"
                    neg_index = [-1] * len(tuple_value)
                    # [-1, -1, ...](list) <-
                    # "the rank in values corresponding to negative indices, sort by indices in ascending order"
                    pos_inverse_index = sorted(enumerate(tuple_value), key=lambda x: x[1][0])
                    # [(index3, (pos_value3, neg_value3)(tuple))(tuple), (index9, (pos_value9, neg_value9)), ...](list)
                    # <- "the tuple of indices and corresponding pos&neg value tuples,
                    # sort by pos_value in ascending order"
                    for i, item in enumerate(pos_inverse_index):
                        pos_index[item[0]] = i
                    neg_inverse_index = sorted(enumerate(tuple_value), key=lambda x: x[1][1], reverse=True)
                    # [(index7, (pos_value7, neg_value7)(tuple))(tuple), (index4, (pos_value4, neg_value4)), ...](list)
                    # <- "the tuple of indices and corresponding pos&neg value tuples,
                    # sort by neg_value in descending order"
                    for i, item in enumerate(neg_inverse_index):
                        neg_index[item[0]] = i
                    sum_index = [i + j for i, j in zip(pos_index, neg_index)]
                    # [rank_sum1, rank_sum2, ...](list) <-
                    # "the sum of pos rank and neg rank for all indices, sort by indices in ascending order"
                    inverse_index = sorted(enumerate(sum_index), key=lambda x: x[1])
                    # [(index15, rank_sum15)(tuple), (index6, rank_sum6), ...](list) <-
                    # "the tuple of indices and corresponding rank_sums, sort by rank_sum in ascending order"
                # else:
                #     raise ValueError(value_error_msg('sort_by', sort_by, TripletDataset.SORT_LIST))
                self.triplet_index = [self.triplet_index[item[0]] for item in inverse_index]
                # [(index1, index5, index28), (index1, index5, index92), ...](list) <- "triplet indices"
                # reorder indices according to inverse index

    @timer
    def __positive_sampling(self, sample_range, positive_param, distance_matrix):
        # start = time()
        sampled_positive_index = []
        # [[index1, index3, ...](list), [index5, index8, ...], ...](list) <- "positive indices"
        sampled_positive_value = []
        # [[value1, value3, ...](list), [value5, value8, ...], ...](list) <-
        # "the distances between features of positive indices and anchor indices"
        if sample_range == TripletDataset.PositiveRange.CLASS:  # approx 0.04 - 0.05 sec
            for i, item in enumerate(self.anchor_class):
                index_range = np.asarray(
                    tuple(range(self.class_index_range[item], self.class_index_range[item + 1])))
                # [index1 index2 ...](ndarray) <- "the index range of positive examples"
                cam_range = np.asarray(self.cams[self.class_index_range[item]: self.class_index_range[item + 1]])
                # [cam1 cam1 ...](ndarray) <- "the cam range of positive examples"
                sampled_index_range = index_range[cam_range != self.sampled_class_cam[item]].tolist()
                # [index3 index4 ...](ndarray) <- "the index range of available positive examples"
                # note that the available positive pic should be taken from a cam
                # which is not the same as the anchor img's cam(in cam_range != sampled_class_cam[i])
                dist = distance_matrix[i][sampled_index_range]
                # [dist1, dist2, ...](Tensor) <- "a 1D distance tensor, where tensor[i] represents
                # the distance of features between sampled_index[i] and the given anchor index"
                first_k_largest = torch.topk(dist, min(self.class_num[item] // positive_param + 1,
                                                       len(sampled_index_range)))
                # (values=[value6, value2, ...](Tensor), indices=[index6, index2, ...])(return_types.topk) <-
                # "the compound of the first k largest values and indices"
                sampled_positive_index.append([sampled_index_range[i] for i in first_k_largest.indices.tolist()])
                sampled_positive_value.append(first_k_largest.values.tolist())
        elif sample_range == TripletDataset.PositiveRange.CAM:  # equals to elif sample_range == cam:
            for i, item in enumerate(self.anchor_class):
                cams = np.asarray(tuple(self.class_cam_index[item].keys()))
                # [cam1 cam2 ...](ndarray) <- "the cam range of the anchor class"
                sampled_cams = cams[cams != self.sampled_class_cam[item]]
                # [cam2 ...](ndarray) <-
                # "the cam range of the anchor class except the cam of the sampled anchor index"
                sampled_indices = []
                # [index1, index3, ...](list) <- "the sampled indices of a certain anchor class"
                sampled_values = []
                # [value1, value3, ...](list) <- "the distances between positive indices and the sampled indices"
                for cam in sampled_cams:
                    indices = self.class_cam_index[item][cam]  # [index1, index2, ...](list) <-
                    # "the indices given anchor class and cam"
                    dist = distance_matrix[i][indices]
                    # [dist1, dist2, ...](Tensor) <- "a 1D distance tensor, where tensor[i] represents
                    # the distance of features between indices[i] and the given anchor index"
                    first_k_largest = torch.topk(dist, min(len(indices) // positive_param + 1, len(indices)))
                    # (values=[value6, value2, ...](Tensor), indices=[index6, index2, ...])(return_types.topk) <-
                    # "the compound of the first k largest values and indices"
                    sampled_indices.extend([indices[i] for i in first_k_largest.indices.tolist()])
                    sampled_values.extend(first_k_largest.values.tolist())
                    # note that the indices mined from different cams are put together in a single list
                sampled_positive_index.append(sampled_indices)
                sampled_positive_value.append(sampled_values)
        # print('positive', sample_range, time() - start)
        # else:
        #     raise ValueError(value_error_msg(value_error_msg('positive_sample_range', sample_range,
        #                                                      TripletDataset.POSITIVE_RANGE_LIST)))
        return sampled_positive_index, sampled_positive_value

    @timer
    def __negative_sampling(self, sample_range, max_negative, distance_matrix):
        # start = time()
        sampled_negative_index = []
        # [[index9, index11, ...](list), [index1, index4, ...], ...](list) <- "negative indices"
        sampled_negative_value = []
        # [[value1, value2, ...](list), [value3, value4, ...], ...](list) <-
        # "the distances between features of negative indices and anchor indices"
        if sample_range in [TripletDataset.NegativeRange.SAMPLE, TripletDataset.NegativeRange.CLASS]:
            dist = distance_matrix[:, self.sampled_class_index]
            # [[dist11, dist12, ...], [dist21, dist22, ...], ...](Tensor) <-
            # "the distance between features of anchors and classes"
            # a distance matrix, where dist[i][j] represents the distance between class i and j
            first_k_smallest = torch.topk(dist, min(max_negative + 1, self.class_count), largest=False)
            # (values=[[value11, value12, ...], [value21, value22, ...], ...](Tensor),
            # indices=[[index11, index12, ...], [index21, index22, ...], ...])(return_types.topk) <-
            # "the compound of the first 2: k + 1 smallest values and indices"
            mined_negative_class = first_k_smallest.indices[:, 1:]
            # [[class3, class8, ...], [class6, class9, ...], ...](Tensor) <-
            # "the tensor of selected negative classes for each anchor"
            # as the distance between the same indices is always zero(dist.diagonal is made up of 0),
            # the minimal distance of the anchor index is invalid, so need to look one item behind(in negative_num + 1)
            mined_negative_value = first_k_smallest.values[:, 1:]
            # [[value3, value8, ...], [value6, value9, ...], ...](Tensor) <-
            # "the tensor of distances between negative indices and anchor indices"
            if sample_range == TripletDataset.NegativeRange.SAMPLE:  # approx 0.02 - 0.03 sec
                sampled_negative_index = [[self.sampled_class_index[j] for j in i] for i in mined_negative_class]
                sampled_negative_value = mined_negative_value.tolist()
            elif sample_range == TripletDataset.NegativeRange.CLASS:  # equals to elif sample_range == 'class'
                for i, classes in enumerate(mined_negative_class):
                    sampled_indices = []
                    # [index1, index3, ...](list) <- "the sampled indices of a certain anchor class"
                    sampled_values = []
                    # [value1, value3, ...](list) <-
                    # "the distances between negative indices and the sampled indices"
                    for item in classes:
                        indices = list(range(self.class_index_range[item], self.class_index_range[item + 1]))
                        # [index1, index2, ...](list) <- "the indices of the mined negative class"
                        dist = distance_matrix[i][indices]
                        # [dist1, dist2, ...](Tensor) <- "a 1D distance tensor, where tensor[i] represents
                        # the distance of features between indices[i] and the given anchor index"
                        smallest = torch.min(dist, 0)
                        # (values=[value5](Tensor), indices=[index5])(return_types.topk) <-
                        # "the compound tuple of the smallest value and index"
                        sampled_indices.append(indices[smallest.indices.item()])
                        sampled_values.append(smallest.values.item())
                    sampled_negative_index.append(sampled_indices)
                    sampled_negative_value.append(sampled_values)
            # else:
            #     raise ValueError(value_error_msg('negative_sample_range', sample_range,
            #                                      TripletDataset.NEGATIVE_RANGE_LIST))
        elif sample_range == TripletDataset.NegativeRange.INDEX:
            # equals to elif sample_range == 'index'  # approx 0.2 - 0.4 sec
            for i, item in enumerate(self.anchor_class):
                indices_mask = torch.ones(self.index_count, dtype=torch.bool)
                # [True, True, ...](Tensor) <- "the all true tensor w.r.t all indices"
                exclude_indices = list(range(self.class_index_range[item], self.class_index_range[item + 1]))
                # [index4, index5, ...](list) <- "the positive indices of class item to be excluded"
                indices_mask[exclude_indices] = False
                # [True, False, ...](Tensor) <- "the bool tensor w.r.t all indices,
                # where positive indices are marked by False"
                indices = indices_mask.nonzero().squeeze(1).tolist()
                # [0, 2, ...](list) <- "the integer list w.r.t all indices, where positive indices are excluded"
                first_k_smallest = torch.topk(distance_matrix[i][indices_mask],
                                              min(max_negative, self.index_count - self.class_num[item]), largest=False)
                # (values=[value6, value2, ...](Tensor), indices=[index6, index2, ...])(return_types.topk) <-
                # "the compound of the first k smallest values and indices"
                sampled_negative_index.append([indices[i] for i in first_k_smallest.indices.tolist()])
                sampled_negative_value.append(first_k_smallest.values.tolist())
        # print('negative', sample_range, time() - start)
        # else:
        #     raise ValueError(value_error_msg('negative_sample_range', sample_range,
        #                                      TripletDataset.NEGATIVE_RANGE_LIST))
        return sampled_negative_index, sampled_negative_value

    @staticmethod
    def __feature_distance(x1, x2):
        return torch.cdist(x1, x2, p=2)

    def __len__(self):
        return len(self.triplet_index)

    def __getitem__(self, item):
        return super(TripletDataset, self).__getitem__(self.triplet_index[item])
