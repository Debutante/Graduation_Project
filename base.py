from abc import ABCMeta, abstractmethod
from enum import Enum, unique


class BaseExecutor(metaclass=ABCMeta):
    """The shared base class."""
    @unique
    class Name(Enum):
        MARKET1501 = 'market1501'

    NAME_LIST = [item.value for item in Name]

    @unique
    class Dataset(Enum):
        TRIPLET = 'triplet'
        EXTENDED = 'extended'
        IMAGE = 'image'

    DATASET_LIST = [item.value for item in Dataset]

    @unique
    class Split(Enum):
        TRAIN_VAL = 'train_val'
        TRAIN_ONLY = 'train_only'

    SPLIT_LIST = [item.value for item in Split]

    @unique
    class Model(Enum):
        MSSNET = 'mssnet'
        RESNET50 = 'resnet50'

    MODEL_LIST = [item.value for item in Model]

    @unique
    class Scene(Enum):  # also called setting protocol
        SINGLE_SHOT = 'single_shot'
        MULTI_SHOT = 'multi_shot'

    SCENE_LIST = [item.value for item in Scene]

    @staticmethod
    def run_info(class_name: str, additional_msg=None):
        """Prints info."""
        print('=' * 30)
        print('This is %s running.' % class_name)
        if additional_msg is not None:
            print('[%s]' % additional_msg)

    @abstractmethod
    def run(self):
        pass
