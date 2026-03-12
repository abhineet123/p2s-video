<!-- No Heading Fix -->

This is the official implementation of our extension of the [Pix2Seq language modeling framework](https://github.com/google-research/pix2seq) for autoregressive video object detection and panoptic segmentation in images and videos.

- [phd dissertation](docs/p2s_vid_phd_thesis.pdf)
- [video detection paper](https://ieeexplore.ieee.org/document/11115031) [[pdf]](docs/p2s_vid_det_paper.pdf)
- [semantic segmentation paper](https://arxiv.org/abs/2602.21627) [[pdf]](docs/p2s_vid_sem_seg_paper.pdf)

# Table of Contents

<!-- MarkdownTOC -->

- [Setup](#setup_)
    - [Code](#cod_e_)
    - [Install](#install_)
- [Run](#run_)
- [Models](#model_s_)
    - [ARIS dataset](#aris_datase_t_)
    - [IPSC dataset](#ipsc_datase_t_)

<!-- /MarkdownTOC -->

<a id="setup_"></a>
# Setup
<a id="cod_e_"></a>
## Code
This repo should be cloned to `~/pix2seq`:    
    - `git clone https://github.com/abhineet123/p2s-video ~/pix2seq`

In addition to this repo, code from two of our other repos is needed to run all the steps in the dataset generation - training – inference - evaluation pipeline:

- [river ice segmentation](https://github.com/abhineet123/river_ice_segmentation): This contains all the segmentation-specific parts of the data processing pipeline including creating and stitching patches, data augmentation and segmentation evaluation.
This should be cloned to `~/617`:
    - `git clone https://github.com/abhineet123/river_ice_segmentation ~/617`

- [ipsc prediction](https://github.com/abhineet123/ipsc_prediction): This contains general utility functions that are used in the other two repos, along with the object detection evaluation pipeline.
This should be cloned to `~/ipsc`:
    - `git clone https://github.com/abhineet123/ipsc_prediction ~/ipsc`

Each of these repos was created for a different project but they all use a lot of same data processing stuff so we have linked them together to avoid code duplication and fragmentation.

<a id="install_"></a>
## Install
We have used python 3.10 on Ubuntu 22.04 for most of our testing but it should also work on any python version > 3.8 and any non-ancient linux distro.    

Windows support is limited to CPU-only training and inference since pix2seq needs tensorflow 2.15 and Google stopped releasing GPU-versions of tensorflow pip package after 2.10.
GPU-support requires tensorflow 2.15 to be compiled from source.

virtualenv:
```
python3.10 -m pip install virtualenv virtualenvwrapper
mkvirtualenv -p python3.10  pix2seq
```
or conda:
```
conda create -n pix2seq python=3.10
conda activate pix2seq
```
packages:
```
python -m pip install -r requirements.txt
```

More detailed commands for other platforms along with bug-fixes and GPU related stuff (e.g. CUDA installation) are available in [cmd/p2s_setup.md](cmd/p2s_setup.md).
This file also contains lots of other setup-related commands we have used in the course of our experiments but which will likely be irrelevant to your use-case.

<a id="run_"></a>
# Run
We use markdown files to keep track of all the experiments we have done.
These files store the actual commands needed to run the scripts in various configurations and are organized hierarchically by task and dataset.
Each file has a table of contents for ease of navigation.
The `.md` files for each repo or module are in a subfolder named `cmd` within the folder containing that module.

This repo contains code for two parts of the pipeline:
- generating tfrecord files from a dataset
- running training and inference on these files

Therefore, there are two sets of corresponding `.md` files in the [`cmd`](cmd) folder, respectively suffixed with `tf` and `p2s`.    

Both sets of files follow this naming scheme: `<prefix>_<task>-<dataset>.md`

- `prefix` can be either  `tf` and `p2s`
- `task` can be one of `vid`, `seg` or `vid_seg` respectively for video object detection, static segmentation, and video segmentation.
    - `task` is empty for static object detection
- `dataset` can be `ipsc`, `isl`, `imgn`, `acamp`, `617`, `coco`, or `ctscp` respectively for
[IPSC](https://huggingface.co/abhineet123/ipsc_prediction),
[UA-DETRAC](https://www.kaggle.com/datasets/bratjay/ua-detrac-orig),
[Imagenet Vid](https://image-net.org/challenges/LSVRC/2017/),
[ACAD](https://huggingface.co/datasets/abhineet123/animal_detection),
[ARIS](https://ieee-dataport.org/open-access/alberta-river-ice-segmentation-dataset),
[COCO](https://cocodataset.org/#home), and
[Cityscapes](https://www.cityscapes-dataset.com/)
datasets
    - each dataset is only paired with a subset of the tasks that it supports

For example, [`p2s_vid-ipsc.md`](cmd/p2s_vid-ipsc.md) contains training and inference commands for performing video detection on IPSC dataset while  [`p2s_seg-ctscp.md`](cmd/p2s_seg-ctscp.md) contains commands for performing static segmentation on the Cityscapes dataset.
Similarly, [`tf_seg-617.md`](cmd/tf_seg-617.md) and [`tf_vid-isl.md`](cmd/tf_vid-isl.md) contain commands for generating tfrecords files for static segmentation on ARIS dataset and video object detection on UA-DETRAC dataset.

<a id="model_s_"></a>
# Models
The trained models reported in the papers are available in [this hugging face model repo](https://huggingface.co/abhineet123/p2s-video).
Each model archive should be extracted in the root of this repository while maintaining the folder structure inside the archive.
<a id="aris_datase_t_"></a>
## ARIS dataset
- [32 training images](https://huggingface.co/abhineet123/p2s-video/blob/main/ARIS/log_seg_resnet_1024_resize_1280-0_31-1024_1024-64_256-rot_15_345_16-flip-sub_8-lac-617-batch_4-seq3k.zip)

<a id="ipsc_datase_t_"></a>
## IPSC dataset
- early stage training
    - [static segmentation](https://huggingface.co/abhineet123/p2s-video/blob/main/IPSC/early_stage/log_seg_resnet_640_resize_640-16_53-640_640-640_640-rot_15_345_4-flip-sub_2-2d-lac-batch_8-seq3k.zip)
    - [video segmentation](https://huggingface.co/abhineet123/p2s-video/blob/main/IPSC/early_stage/log_video_seg_resnet_640_resize_2560-16_53-640_640-640_640-length-8-stride-1-sub_8-tac-mc-batch_4-seq3k.zip)
- late stage training
    - [static segmentation](https://huggingface.co/abhineet123/p2s-video/blob/main/IPSC/late_stage/log_seg_resnet_640_resize_2560-54_126-640_640-160_320-rot_15_345_1-sub_8-lac-batch_72-fbb.zip)
    - [video segmentation](https://huggingface.co/abhineet123/p2s-video/blob/main/IPSC/late_stage/log_video_seg_resnet_640_resize_2560-54_126-640_640-640_640-length-8-stride-1-sub_8-tac-mc-batch_8-voc15-seq3k-fbb.zip)
