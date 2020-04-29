from base import BaseExecutor
from utils.utilities import value_error_msg, type_error_msg
from utils.folder_processors import Market1501Processor
from components.trainers import Trainer
from components.testers import Tester
from components.evaluators import Market1501Evaluator
from components.visualizers import Visualizer
from configparser import ConfigParser, ExtendedInterpolation

conf_path = 'settings.txt'
conf = ConfigParser(interpolation=ExtendedInterpolation(), default_section='Default')
conf.read(conf_path)


class Engine(BaseExecutor):
    """A general engine scheme.

    Args:
        name (str): A name defined in base.py.

    """
    def __init__(self, name):
        if not isinstance(name, Engine.Name):
            if isinstance(name, str):
                if not name.islower():
                    name = name.lower()
                if name not in Engine.NAME_LIST:
                    raise ValueError(value_error_msg('name', name, Engine.NAME_LIST))
                name = Engine.Name(name)
            else:
                raise TypeError(type_error_msg('name', name, [Engine.Name, str]))
        self.name = name
    
    def run(self):
        raise NotImplementedError


class Market1501Engine(Engine):
    """An engine for market1501 dataset.

    Args:
        config (ConfigParser): The ConfigParser which reads setting files.
        config_path (str): Path to setting files.
        model (str): A model defined in base.py.
        pretrain (bool): Initialize model with pretrained weights on ImageNet or not.
        epoch (int): The max_epoch to train, the model@epoch to test/evaluate/visualize.
        scene (str): A scene defined in base.py.

    """
    def __init__(self, config: ConfigParser, config_path: str, model: str, pretrain: bool, epoch: int, scene: str,
                 query: list, length: int):
        super(Market1501Engine, self).__init__('market1501')
        Market1501Processor(conf, conf_path).run()  # preprocessing
        self.trainer = Trainer(config, config_path, self.name.value, 'image', 'train_val', model, pretrain, 'sgd')
        self.tester = Tester(config, self.name.value, 'extended', model, epoch, scene)
        self.evaluator = Market1501Evaluator(config, scene, True)
        self.visualizer = Visualizer(config, self.name.value, model, pretrain, epoch, 'train_val', scene, query, length)
        self.epoch = epoch

    def run(self):
        self.trainer.run(max_epoch=self.epoch)
        self.tester.run()
        self.evaluator.run()
        self.visualizer.run()


Market1501Engine(conf, conf_path, 'resnet50', True, 8, 'single_shot', [163, 78], 30).run()
