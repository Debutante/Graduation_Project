<h1 align="center"> Graduation Project(Mid-term Preview) </h1>

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/Debutante/Graduation_Project.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/Debutante/Graduation_Project/context:python)
[![Build Status](https://travis-ci.org/Debutante/Graduation_Project.svg?branch=master)](https://travis-ci.org/Debutante/Graduation_Project)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/Debutante/Graduation_Project.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/Debutante/Graduation_Project/alerts/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A standard, medium-sized baseline code for Person-reID (based on [pytorch](https://pytorch.org)).

- **Standard.** Arrived at Rank@1:86.7280% Rank@5:94.6853% Rank@10:96.3777% mAP:67.7160% @epoch16. 
You may learn from `console.log` if you want to know about the decline of losses and errors in detail.

ResNet(pretrained) @ epoch16: 

|Scenes |Rank@1 | Rank@5 | Rank@10| mAP| 
| ------ | -------- | ----- | ---- | ---- | 
| single_shot | 86.7280% | 94.6853% | 96.3777% | 67.7160% | 
| single_shot(re-ranking) | 89.3112% | 93.7945% | 95.2791% | 83.0887% | 
| multi_shot | 91.2411% | 96.4667% | 97.7138% | 76.8081% | 
| multi_shot(re-ranking) | 92.7553% | 96.0808% | 96.9715% | 87.4576% |

Quick facts:

> Running time(on CPU): 
> + train: 1 hours 8 minutes for one epoch
> + test: 28 minutes for 10000 pics
> + evaluate: 12 seconds if no re-rank, 3 minutes 20 seconds if re-rank
> + visualize: 6 seconds


- **Medium-sized.** The model, resnet-50 (from [Deep Residual Learning for Image Recognition](http://xxx.itp.ac.cn/abs/1512.03385)), has 27M parameters.
The baseline could be trained with only 2GB memory.

## Table of contents
* [Features](#features)
* [Some News](#some-news)
* [Model Structure](#model-structure)
* [Prerequisites](#prerequisites)
* [Getting Started](#getting-started)
    * [Installation](#installation)
    * [Dataset Preparation](#dataset--preparation)
    * [Quick Start](#quick-start)
    * [Train](#train)
    * [Test](#test)
    * [Evaluation](#evaluation)
    * [Visualization](#visualization)
* [Intro to some data files](#files)
* [Related Repos](#related-repos)

## Features
Now supported:
- ResNet50 with a self-defined classification block
- Single-shot/Multi-shot
- Train set split
- Re-ranking(supported by [Re-ranking Person Re-identification with k-reciprocal Encoding](http://xxx.itp.ac.cn/abs/1701.08398))
- Visualize Training Curves & Ranking Result(supported by [Tensorboard](https://tensorflow.google.cn/tensorboard))

## Some News
**25 May 2020** Finish the draft.

**20 May 2020** Start writing my thesis.

**15 May 2020** Collect related literature for reference. 

**29 April 2020** Try to improve the visualization of embeddings. The [t-SNE](https://dl.acm.org/doi/10.5555/2627435.2697068) produces a too dense embedding representation so I choose UMAP for better display effects.

**28 April 2020** Find that euclidean distance computation is 100 times slower than matrix multiplication. That really surprises me. So I replace all the euclidean distance with normalization and matrix computation, and it saves maybe 3 minutes running time on CPU.
 Nearly finishes the visualization module. It is hardly possible to interpret heatmaps and know what exactly the network learns. 

**27 April 2020** Do evaluation on test set and get nearly 90%@r1. ResNet really learns from (but maybe overfits) the Market-1501 dataset.

**19 April 2020** Run [Person re-ID baseline with triplet loss](https://github.com/layumi/Person-reID-triplet-loss) for 8 hours. 
It seems like the crazy code needs more than 10GB memory to run for option `batch_size=32` and it's impractical for my PC. 
I think maybe some unnecessary gradients are computed during online mining. I will check later if possible.

**13 April 2020** Visualize some rank list. Distractors have some notable impact on performance.

**12 April 2020** Get the awful result of Rank@1:5.0178% Rank@5:12.2328% Rank@10:17.7553% mAP:1.3463%. Wonder if the entire training is just a big joke.

**11 April 2020** Use 320 minutes to train MssNet for 50 epochs on CPU.

**23 March 2020** Make minor modification to [Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch) to train a pretrained Resnet-50 on CPU. 

The re-ranking in [Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf) enhance the mAP remarkably.

|Methods |Rank@1 | Rank@5 | Rank@10| mAP| 
| ------ | -------- | ----- | ---- | ---- | 
| ResNet(pretrained)@epoch0 | 8.5808% | 17.5772% | 23.7233% | 2.2954% | 
| ResNet(pretrained)@epoch1 | 67.6960% | 83.3432% | 88.7173% | 44.8274% | 
| ResNet after re-ranking |  71.0214% | 82.1556% | 85.8670% | 59.1543% | 
| ResNet@epoch0 | 3.4739% | 10.3029% | 16.0036% | 1.0970% | 
| ResNet@epoch1 | 5.7007% | 13.9252% | 20.6057% | 1.9820% | 

Quick facts:

> Running time(on CPU): 
> + train: 75 minutes for one epoch
> + test: 60 minutes
> + evaluate: less than 30 seconds
> + visualize: 3 minutes

> Use Cross Entropy Loss

> Use k-reciprokal for re-ranking



## Model Structure
The backbone of the model is [ResNet](http://xxx.itp.ac.cn/abs/1512.03385), which consists of four blocks of bottlenecks' structure. The design of bottleneck-like structure is very efficient as it reduces parameters by times.

You may learn more from `models/model.py`.

## Prerequisites
- CPU Memory >= 2G
- Python 3
- Numpy
- Pytorch
- torchvision
- Scipy
- Matplotlib

Note that the code is tested inside the environment listed in `requirements.txt`.

## Getting started
### Installation
- Install Pytorch using some package manager, e.g. pip/conda
```
pip install torch torchvision
```
or
```
conda install torch torchvision
```

### Dataset & Preparation
Download [Market1501 Dataset](http://www.liangzheng.com.cn/Project/project_reid.html) from [[Baidu link]](https://pan.baidu.com/s/1ntIi2Op).

Preparation: Change the download_dir in `settings.txt` to your own path.
```
# from settings.txt
[market1501]
download_dir = YOUR PATH HERE
```
Additionally, you can change other paths(e.g. `dataset_dir`, `train_path`, `test_path`, etc) in `settings.txt` to your desired paths.

### Quick Start
Make sure you've done the steps above.

Then run `main.py` and the code is supposed to work.

If any wrong happens, open `requirements.txt` to check if all required packages and their versions.

If you want to customize anything, please continue reading.

### Train
Train a model in `components/trainers.py`.

Read from: None

Write to: `models/resnet50_epochNum.pth`, `analysis/train.mat`
```
Trainer(self, config, config_path, name, dataset, split, model, pretrain: bool, optimizer, lr: float, momentum: float, weight_decay: float, start_epoch: int)
```
`config` a ConfigParser object which reads `settings.txt`

`config_path` the path to `settings.txt`

`name` a dataset's name in `name_list = ['market1501']`

`dataset` a dataset corresponding to an organizational form for data in `dataset_list = ['image', 'extended', 'triplet']`

`split` a training set split method in `split_list = ['train_only', 'train_val']`

`model` a model in `model_list = ['mssnet', 'resnet50']`

`pretrain` a pretrain option in `[True, False]`

`optimizer` an optimizer in `optimizer_list = ['sgd']`

(optional)`lr` the learning rate in `sgd`, default `0.05`

(optional)`momentum` the momentum in `sgd`, default `0.9`

(optional)`weight_decay` the weight decay in `sgd`, default `5e-4`

(optional)if you want to train from a saved model, set `start_epoch` to the epoch number and training will start from there.

### Test
Use a trained model to extract feature in `components/testers.py`.

Read from: `models/resnet50_epochNum.pth`

Write to: `analysis/test.mat`

```
Tester(self, config, name: str, dataset: str, model, epoch: int, scene: str)
```
`config` a ConfigParser object which reads `settings.txt`

`name` a dataset's name in `name_list = ['market1501']`

`dataset` a dataset corresponding to an organizational form for data in `dataset_list = ['image', 'extended', 'triplet']`

`model` a model in `model_list = ['mssnet', 'resnet50']`

`epoch` the epoch of the saved trained model for testing

`scene` the scene for testing & evaluating & visualizing in `scene_list = ['single_shot', 'multi_shot']`

### Evaluation
Output Rank@1, Rank@5, Rank@10 and mAP results in `components/evaluators.py`.

Use the triangle mAP calculation protocol here (consistent with the Market1501 original code [Baidu Link](http://pan.baidu.com/s/1hqMbd4K)).

Read from: `analysis/test.mat`

Write to: `analysis/evaluate.mat`

```
Market1501Evaluator(self, config, mode: str, re_rank: bool)
```
`config` a ConfigParser object which reads `settings.txt`

`mode` the mode for testing & evaluating in `mode_list = ['single_shot', 'multi_shot']`

`re_rank` whether to re-rank or not, note that it takes 3 minutes and at least 15GB memory(at peak) to re-rank

### Visualization
Display training loss curves, CMC curves and rank lists of selected queries in `components/visualizer.py`.

Use [Tensorboard](https://tensorflow.google.cn/tensorboard) for rendering.

Start service: `tensorboard --logdir=runs`

View the results:  http://localhost:6006/

Read from: `analysis/train.mat`, `analysis/test.mat`, `analysis/evaluate.mat`

Write to: `runs/market1501/SummaryWriterLogFiles`

```
Visualizer(self, config, name, model, pretrain, epoch, split, scene, query_list: list, length: int)
```
`config` a ConfigParser object which reads `settings.txt`

`name` a dataset's name in `name_list = ['market1501']`

`model` a model in `model_list = ['resnet50']`

`pretrain` a pretrain option in `[True, False]`

`epoch` the epoch of the saved trained model for visualizing

`split` a training set split method in `split_list = ['train_only', 'train_val']`

`scene` the scene for testing & evaluating & visualizing in `scene_list = ['single_shot', 'multi_shot']`

`query_list` a list of query indices

`length` the length of ranking list

## Files

**`models/resnet_epochNum.pth`**: model's static dictionary


**`analysis/train.mat`**: the dictionary of training loss

```
dictionary = {
    'training_loss': [[loss@epoch1 loss@epoch2 ...]](ndarray),
    'training_error': [[error@epoch1 error@epoch2 ...]](ndarray), 
    (if applicable)'validation_loss': [[loss@epoch1 loss@epoch2 ...]](ndarray),
    (if applicable)'validation_error': [[error@epoch1 error@epoch2 ...]](ndarray)
}
```

**`analysis/test.mat`**: the dictionary of gallery and query info

```
dictionary = {
    'gallery_feature': [[dim1 dim2 ... in index1] [dim1 dim2 ... in index2] ...](ndarray),
    'gallery_label': [[label1 label1 label2 ...]](ndarray),
    'gallery_cam': [[cam1 cam1 cam2 ...]](ndarray),
    'query_feature': [[dim1 dim2 ... in index1] [dim1 dim2 ... in index2] ...](ndarray),
    'query_label': [[label1 label1 label2 ...]](ndarray),
    'query_cam': [[cam1 cam1 cam2 ...]](ndarray), 
    (if applicable)'multi_index': [[[[index1 index2 index3 ...]] [[index4 index5 index6 ...]] ...]](ndarray)
}
```

**`analysis/evaluate.mat`**: the dictionary of evaluation indicators and rank lists

```
dictionary = {
    'index': [[[[index9 index11 index3 ...]] [[index1 index5 index9 ...]] ...]](ndarray),
    'ap': [[ap1 ap2 ap3 ...]](ndarray),
    'CMC': [[cmc1 cmc2 cmc3 ...]](ndarray)
}
```

## Related-Repos
1. This README is also in the manner of the README in [Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch) ![GitHub stars](https://img.shields.io/github/stars/layumi/Person_reID_baseline_pytorch.svg?style=flat&label=Star)
2. Use only 50 lines to train on many ReID datasets and get decent results in [Torchreid: Deep learning person re-identification in PyTorch](https://github.com/KaiyangZhou/deep-person-reid) ![GitHub stars](https://img.shields.io/github/stars/KaiyangZhou/deep-person-reid.svg?style=flat&label=Star)
3. Another advanced library for ReID in [Open source person re-identification library in python](https://github.com/Cysu/open-reid) ![GitHub stars](https://img.shields.io/github/stars/Cysu/open-reid.svg?style=flat&label=Star)