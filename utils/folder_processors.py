from time import time
from shutil import copyfile
from configparser import ConfigParser, ExtendedInterpolation
from os import listdir, mkdir, makedirs
from os.path import exists, isdir, join
from base import BaseExecutor

# conf_path = '../settings.txt'
# conf = ConfigParser(interpolation=ExtendedInterpolation(), default_section='Default')
# conf.read(conf_path)


class Processor(BaseExecutor):
    """A general evaluator for all datasets.

    Args:
        config (ConfigParser): The ConfigParser which reads setting files.
        config_path (str): Path to setting files.

    """
    DEFAULT_DIR = 'processed'

    def __init__(self, config, config_path):
        self.config = config
        self.config_path = config_path

    def run(self):
        raise NotImplementedError


class Market1501Processor(Processor):
    """Transforms datasets' format for ImageFolder to recognise
        download dataset structure:         processed dataset structure
    ├── Market/                         ├── processed/
    │   ├── query/--------------------->│   ├── query/
    │   ├── bounding_box_test/--------->│   ├── gallery/
    │   │                      ├──----->│   ├── train/
    │   ├── bounding_box_train/├──----->│   ├── total_train/            /* train+val
    │   │                      ├──----->│   ├── val/
    │   ├── gt_bbox/------------------->│   ├── multi_query/
    │   ├── gt_query/                   │   │   ├── 0002/
    │   ├── readme.txt                  │   │   ├── 0007/
    │   │                               │   │   ...

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery), resolution 128*64.

    Args:
        config (ConfigParser): The ConfigParser which reads setting files.
        config_path (str): Path to setting files.

    """
    # DEFAULT_DIR = 'processed'

    def __init__(self, config, config_path):
        super(Market1501Processor, self).__init__(config, config_path)
        self.src_path = self.config['market1501']['download_dir']
        if not isdir(self.src_path):
            raise NotADirectoryError('The download_dir of market1501 in {} '
                                     'should be an existing directory.'.format(self.config_path))
        if self.config.has_option('market1501', 'dataset_dir'):
            self.dst_path = self.config['market1501']['dataset_dir']
            if not exists(self.dst_path):
                makedirs(self.dst_path)
            else:
                if not isdir(self.dst_path):
                    raise NotADirectoryError('If specified, the dataset_dir of market1501 in {} '
                                             'should be an existing directory; '
                                             'Delete dataset_dir of market1501 in {}'
                                             'to use default target directory '
                                             '"%(download_dir)/{}".'.
                                             format(self.config_path, self.config_path,
                                                    Market1501Processor.DEFAULT_DIR))
        else:  # default target directory
            self.dst_path = join(self.src_path, Market1501Processor.DEFAULT_DIR)
            if not exists(self.dst_path):
                mkdir(self.dst_path)
            self.config['market1501']['dataset_dir'] = '${download_dir}/' + Market1501Processor.DEFAULT_DIR
        self.single_mappings = {
            'query': 'query',
            'bounding_box_test': 'gallery',
            'bounding_box_train': 'total_train',
            'gt_bbox': 'multi_query'
        }
        self.multiple_mappings = {
            'bounding_box_train': ['train', 'val']
        }

    @staticmethod
    def print_info(name, num):
        print('ok@{:<12} [{:>6} new processed]'.format(name, num))

    def folder(self, src_name: str, dst_name: str):
        """Processes the folder named src_name to get the folder named dst_name.

        Args:
            src_name (str): the name input folder to read from.
            dst_name (str): the name output folder to write to(n pics from every class).

        """
        folder_src_path = join(self.src_path, src_name)
        folder_dst_path = join(self.dst_path, dst_name)
        if not exists(folder_dst_path):
            makedirs(folder_dst_path)
        processed_num = 0
        # 0001_c1s1_001051_00.jpg
        for file in listdir(folder_src_path):
            if file.endswith('.jpg') and file.split('_'):
                folder_sub_dst_path = join(folder_dst_path, file.split('_')[0])
                if not exists(folder_sub_dst_path):
                    mkdir(folder_sub_dst_path)
                elif exists(join(folder_sub_dst_path, file)):
                    continue
                copyfile(join(folder_src_path, file), join(folder_sub_dst_path, file))
                processed_num += 1
        Market1501Processor.print_info(dst_name, processed_num)

    def folder_split(self, src_name: str, dst_name1: str, dst_name2: str):
        """Processes the folder named src_name to get the folder named dst_name1 and dst_name2, respectively.

        Args:
            src_name (str): the name input folder to read from.
            dst_name1 (str): the name output folder to write to(n-1 pics from every class).
            dst_name2 (str): the name output folder to write to(1 pic from every class).

        """
        folder_src_path = join(self.src_path, src_name)
        folder_dst_path1 = join(self.dst_path, dst_name1)
        folder_dst_path2 = join(self.dst_path, dst_name2)
        if not exists(folder_dst_path1):
            makedirs(folder_dst_path1)
        if not exists(folder_dst_path2):
            makedirs(folder_dst_path2)
        processed_num1 = 0
        processed_num2 = 0
        for file in listdir(folder_src_path):
            if file.endswith('.jpg') and file.split('_'):
                folder_sub_dst_path1 = join(folder_dst_path1, file.split('_')[0])
                folder_sub_dst_path2 = join(folder_dst_path2, file.split('_')[0])
                # the first images of every person are split as val images
                if not exists(folder_sub_dst_path2):
                    mkdir(folder_sub_dst_path2)
                    copyfile(join(folder_src_path, file), join(folder_sub_dst_path2, file))
                    processed_num2 += 1
                    continue
                elif exists(join(folder_sub_dst_path1, file)):
                    continue
                if not exists(join(folder_sub_dst_path2, file)):
                    if not exists(folder_sub_dst_path1):
                        mkdir(folder_sub_dst_path1)
                    copyfile(join(folder_src_path, file), join(folder_sub_dst_path1, file))
                    processed_num1 += 1
        Market1501Processor.print_info(dst_name1, processed_num1)
        Market1501Processor.print_info(dst_name2, processed_num2)

    def run(self):
        start = time()
        Market1501Processor.run_info(self.__class__.__name__)
        print('Processing begins(from "{}" to "{}"):'.format(self.src_path, self.dst_path))
        for src, dst in self.single_mappings.items():
            self.folder(src, dst)
        for src, dst in self.multiple_mappings.items():
            self.folder_split(src, dst[0], dst[1])
        print('Processing finishes in {:d} minutes {:.3f} seconds.'.
              format(int((time() - start) // 60), (time() - start) % 60))
        with open(self.config_path, 'w+') as file:
            self.config.write(file)
