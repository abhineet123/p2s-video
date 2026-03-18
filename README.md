<!-- No Heading Fix -->

This is the official implementation of our extension of the [Pix2Seq language modeling framework](https://github.com/google-research/pix2seq) for autoregressive video object detection and panoptic segmentation in images and videos.

- [semantic segmentation paper](https://arxiv.org/abs/2602.21627) [[pdf]](docs/p2s_vid_sem_seg_paper.pdf)
- [video detection paper](https://ieeexplore.ieee.org/document/11115031) [[pdf]](docs/p2s_vid_det_paper.pdf)
- [phd dissertation](docs/p2s_vid_phd_thesis.pdf)

# Table of Contents

<!-- MarkdownTOC -->

- [Setup](#setup_)
    - [Code](#cod_e_)
    - [Install](#install_)
- [Usage](#usage_)
    - [Commands](#command_s_)
        - [HTML caveat](#html_caveat_)
    - [Configs](#configs_)
- [Semantic Segmentation Pipeline](#semantic_segmentation_pipeline_)
    - [Setup dataset and softlinks](#setup_dataset_and_softlink_s_)
    - [Setup pretrained weights](#setup_pretrained_weight_s_)
    - [Generate sliding-window patches](#generate_sliding_window_patche_s_)
    - [Create tfrecord files](#create_tfrecord_file_s_)
    - [Train model](#train_mode_l_)
    - [Run inference](#run_inferenc_e_)
        - [Live-inference](#live_inferenc_e_)
    - [Evaluate segmentation performance](#evaluate_segmentation_performanc_e_)
        - [Generate summary plots](#generate_summary_plot_s_)
- [Video Object Detection Pipeline](#video_object_detection_pipeline_)
    - [Setup dataset and softlinks](#setup_dataset_and_softlink_s__1)
    - [Setup pretrained weights](#setup_pretrained_weight_s__1)
    - [Generate JSON files](#generate_json_file_s_)
        - [sampling](#samplin_g_)
        - [static detection](#static_detectio_n_)
    - [Create tfrecord files](#create_tfrecord_file_s__1)
        - [sampling](#samplin_g__1)
    - [Train model](#train_mode_l__1)
        - [Pseudo multi-machine mode](#pseudo_multi_machine_mode_)
    - [Run inference](#run_inferenc_e__1)
        - [Live-inference](#live_inferenc_e__1)
    - [Evaluate detection performance](#evaluate_detection_performanc_e_)
        - [Live-inference](#live_inferenc_e__2)
- [Models](#model_s_)
    - [Semantic Segmentation](#semantic_segmentation_)
        - [ARIS dataset](#aris_datase_t_)
        - [IPSC dataset](#ipsc_datase_t_)
    - [Object Detection](#object_detectio_n_)
        - [UA-DETRAC dataset](#ua_detrac_datase_t_)

<!-- /MarkdownTOC -->

<a id="setup_"></a>
# Setup
<a id="cod_e_"></a>
## Code
This repo should be cloned to `~/pix2seq`:    
    - `git clone https://github.com/abhineet123/p2s-video ~/pix2seq`

In addition to this repo, code from two of our other repos is needed to run all the steps in the dataset generation - training – inference - evaluation pipeline:

- [river ice segmentation](https://github.com/abhineet123/river_ice_segmentation): This contains segmentation-specific parts of the data processing pipeline including creating and stitching patches, data augmentation and segmentation evaluation.
This should be cloned to `~/617`:
    - `git clone https://github.com/abhineet123/river_ice_segmentation ~/617`

- [ipsc prediction](https://github.com/abhineet123/ipsc_prediction): This contains object detection-specifc parts of the data processing pipeline, along with some general utility functions that are used in the other two repos.
This should be cloned to `~/ipsc`:
    - `git clone https://github.com/abhineet123/ipsc_prediction ~/ipsc`

Each of these repos was created for a different project but they all use a lot of same data processing stuff so we have linked them together to avoid code duplication and fragmentation.

<a id="install_"></a>
## Install
We have used python 3.10 on Ubuntu 22.04 for most of our testing but it should also work on any python version > 3.8 and any non-ancient linux distro.    

Windows support is limited to CPU-only training and inference since pix2seq needs tensorflow 2.15 and Google stopped releasing GPU-versions of tensorflow pip package after 2.10.
GPU-support in Windows therefore requires tensorflow 2.15 to be compiled from source and has not been tested by us.

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

We recommend creating a separate virtual environment for running scripts from each of the two other repos using their own `requirements.txt` files.
A single environment where requirements from all of the repos have been installed might also work but we have not tested this yet.

<a id="usage_"></a>
# Usage
<a id="command_s_"></a>
## Commands
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

The other two repos follow a similar system for recording commands.    
There are two sets of [river ice segmentation commands](https://github.com/abhineet123/river_ice_segmentation/tree/master/cmd) that are needed for the semantic segmentation pipeline: 
- `sub_patch-<dataset>.md` (e.g. [`sub_patch-ipsc.md`](https://github.com/abhineet123/river_ice_segmentation/blob/master/cmd/sub_patch-ipsc.md)): commands for creating a segmentation dataset with optional sliding window patches, subsampling and augmentation.
- `stitch-<dataset>.md` (e.g. [`stitch-ipsc.md`](https://github.com/abhineet123/river_ice_segmentation/blob/master/cmd/stitch-ipsc.md)): commands for evaluating the segmentation outputs from trained models after optionally stitching them together and supersampling as needed.

The [ipsc data processing module](https://github.com/abhineet123/ipsc_prediction/tree/master/ipsc_data_processing/cmd)  likewise has two sets of commmand files needed for the object detection pipeline:
- [`xml_to_coco.md`](https://github.com/abhineet123/ipsc_prediction/blob/master/ipsc_data_processing/cmd/xml_to_coco.md) and [`xml_to_ytvis.md`](https://github.com/abhineet123/ipsc_prediction/blob/master/ipsc_data_processing/cmd/xml_to_ytvis.md): commands for converting annotations from either [Pascal VOC XML format](https://roboflow.com/formats/pascal-voc-xml) or our own CSV format into [COCO JSON format](https://roboflow.com/formats/coco-json) for static object detection or [youtube-vis 2019](https://youtube-vos.org/dataset/vis/) JSON format for video object detection, along with optional sampling to create dataset splits in a variety of ways.
- `eval_p2s_<task>-<dataset>.md` (e.g. [`eval_p2s_vid-ipsc.md`](https://github.com/abhineet123/ipsc_prediction/blob/master/ipsc_data_processing/cmd/eval_p2s_vid-ipsc.md)): commands for evaluating detection model outputs, along with optional visualization of fine-grained categories of detection errors as exemplified in the [supplementary material of the ipsc prediction paper](https://github.com/abhineet123/ipsc_prediction#visualizations).

<a id="html_caveat_"></a>
### HTML caveat
__Please make sure to copy commands directly from the .md file__ instead of the html version available on github because the latter automatically translates script-specific tokens into pre-formatted text.

For example, tokens like `_in_`, `_out_`, and `__var__` have special meanings for the [stitching script](https://github.com/abhineet123/river_ice_segmentation/blob/master/stitch.py), but the HTML version converts these into bold or italic text and removes the underscores that are necessary for the script to parse these tokens correctly.

The reason such tokens are used in the first place is because they show up with different formatting (e.g. color or italicization) from the surrounding parts of the command when the markdown files are opened in an editor  that supports syntax highlighting.
We find that this greatly helps with parsing the command quickly by dividing it into easy-to-see sections.

<a id="configs_"></a>
## Configs
Most of our scripts use [paramparse](https://pypi.org/project/paramparse/) `cfg` files to succinctly specify parameters for different configurations in a human-readable format.
These files are contained in a subfolder named `cfg` in the same folder that contains the `cmd` subfolder.

- The `.cfg` files largely follow the same naming scheme as the `.md` files. For example,
    - [`tf_seg-ipsc.cfg`](cfg/tf_seg-ipsc.cfg) contains configurations for generating semantic segmentation tfrecord files for IPSC dataset. Note that it imports [`tf_seg.cfg`](cfg/tf_seg.cfg) for parameters common to all semantic segmentation datasets.
    - [`tf_vid-detrac.cfg`](cfg/tf_vid-detrac.cfg) contains parameters for creating video detection tfrecord files for the UA-DETRAC dataset, while importing [`tf_vid.cfg`](cfg/tf_vid.cfg) for shared video detection parameters.
    
A partial exception to this `cfg` based parameter handling is the script for running pix2seq training and inference which uses similarly formatted [`json5 files`](https://github.com/abhineet123/pix2seq/tree/master/configs/j5) instead.
Pix2seq was already setup to use the [ml_collections](https://github.com/google/ml_collections) `ConfigDict` system which would be very tedious to adapt for paramparse, so we used this workaround.
However, even this script uses [`p2s.cfg`)](cfg/p2s.cfg) to specify some of the parameters for the semantic segmentation pipeline.

<a id="semantic_segmentation_pipeline_"></a>
# Semantic Segmentation Pipeline
Following sections demonstrate the complete step-by-step pipeline for creating semantic segmentation training and test sets from the IPSC dataset, generating tfrecord files from these, training a model, running inference on the trained model, and evaluating the inference outputs against the ground truth.

Assumptions:
- We want to train on the late-stage training dataset containing 73 images per ROI (from 54 to 126) and test it on the standard test dataset containing 16 images per ROI (from 0 to 15)
- We want to resize all images to _`I=2560`_, then extract sliding-window patches of size _`P=640`_ and finally subsample these by a factor of 8 to get masks of size _`S=80`_
- We want to use LAC-encoded RLE sequences with 1D starts
- We want to augment the training dataset with a single random rotation per image between 15 and 345 degrees and through overlapping patches generated with random strides between 160 and 320 pixels.
- We want to use the 640x640 input size version of the ResNet50 backbone and keep the backbone weights frozen during training
- We want to finetune a model with COCO-pretrained weights instead of training from scratch

These assumptions correspond to the primary late-stage training configuration reported in the [paper](docs/p2s_vid_sem_seg_paper.pdf).

<a id="setup_dataset_and_softlink_s_"></a>
## Setup dataset and softlinks
Download the [IPSC ROI images and labels](https://huggingface.co/abhineet123/ipsc_prediction/blob/main/ipsc_data/roi_images_and_labels.zip) archive and extract it in `/data` while  maintaining the directory structure inside the archive so that the ROI sequences end up in `/data/ipsc/well3/all_frames_roi/`.

Create a softlink to `/data` as `~/pix2seq/datasets`:
```
cd ~/pix2seq
ln -s /data datasets
```

Create `~/pix2seq/log` if needed and a softlink to it as `~/617/log/p2s`:
```
mkdir ~/pix2seq/log
cd ~/617
mkdir log && cd log
ln -s ~/pix2seq/log p2s
```

<a id="setup_pretrained_weight_s_"></a>
## Setup pretrained weights
Download [pretrained weights](https://huggingface.co/abhineet123/p2s-video/tree/main/pretrained) to `~/pix2seq/pretrained`

```
mkdir ~/pix2seq/pretrained && cd ~/pix2seq/pretrained
wget https://huggingface.co/abhineet123/p2s-video/blob/main/pretrained/resnet_640.zip
unzip resnet_640.zip
```
Most of the pretrained weights (except the [video swin transformer ones](https://github.com/SwinTransformer/Video-Swin-Transformer#kinetics-600)) should also be available in the [official pix2seq repo](https://github.com/google-research/pix2seq#coco-object-detection-fine-tuned-checkpoints).

<a id="generate_sliding_window_patche_s_"></a>
## Generate sliding-window patches
Run the following commands in `~/617` to generate subpatch datasets for training and testing:
```
python sub_patch_multi.py cfg=ipsc:p-640:r-2560:rot-15_345_1:strd-160_320:vid-1:frame-54_126

python sub_patch_multi.py cfg=ipsc:p-640:r-2560:vid-1:frame-0_15
```
These commands are available in
[`sub_patch-ipsc.md`](https://github.com/abhineet123/river_ice_segmentation/blob/master/cmd/sub_patch-ipsc.md) under [`p-640-aug-strd @ r-2560/54_126-->sub_patch-ipsc`](https://github.com/abhineet123/river_ice_segmentation/blob/master/cmd/sub_patch-ipsc.md#p_640_aug_strd___r_2560_54_12_6_)
and 
[`p-640 @ r-2560/0_15-->sub_patch-ipsc`](https://github.com/abhineet123/river_ice_segmentation/blob/master/cmd/sub_patch-ipsc.md#p_640___r_2560_0_1_5_) respectively

<a id="create_tfrecord_file_s_"></a>
## Create tfrecord files
Run the following commands in `~/pin2seq` to generate tfrecord files from the subpatch datasets:
```
python data/scripts/create_seg_tfrecord.py cfg=ipsc:54_126:p-640:r-2560:sub-8:rot-15_345_1:strd-160_320:lac

python data/scripts/create_seg_tfrecord.py cfg=ipsc:0_15:p-640:r-2560:sub-8:lac
```
These commands are available in
[`tf_seg-ipsc.md`](cmd/tf_seg-ipsc.md) under [`lac-sub-8 @ p-640-aug-strd/r-2560/54_126-->tf_seg-ipsc`](cmd/tf_seg-ipsc.md#lac-sub-8--------p-640-aug-strdr-256054_126--tf_seg-ipsc)
and 
[`lac @ p-640-sub-8/r-2560/0_15-->tf_seg-ipsc`](cmd/tf_seg-ipsc.md#lac--------p-640-sub-8r-25600_15--tf_seg-ipsc) respectively

<a id="train_mode_l_"></a>
## Train model
Run the following command in `~/pix2seq` to train a static semantic segmentation model with:
- 640x640 input-size version of the ResNet-50 backbone 
- pretrained weights loaded before training (instead of training from scratch)
- backbone weights kept frozen during training
- vocabulary of size _`V=8000`_ tokens
- maximum sequence length of _`L=512`_
- single-machine dual-gpu training over a pair of RTX 3090 GPUs using the corresponding maximum batch size of _`B=72`_

```
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-54_126:p-640:r-2560:rot-15_345_1:strd-160_320:sub-8,batch-72,dbg-0,dyn-1,dist-1,gz,pt-1,lac,fbb
```
This command is available in [`p2s_seg-ipsc.md`](cmd/p2s_seg-ipsc.md) under [`aug-strd-lac-fbb @ r-2560-p-640-sub-8/54_126-->p2s_seg-ipsc`](cmd/p2s_seg-ipsc.md#aug-strd-lac-fbb--------r-2560-p-640-sub-854_126--p2s_seg-ipsc)

This will save the checkpoints and tensorboard log files under `~/pix2seq/log/seg/resnet_640_resize_2560-54_126-640_640-160_320-rot_15_345_1-sub_8-lac-batch_72-fbb`.

<a id="run_inferenc_e_"></a>
## Run inference
Run the following command in `~/pix2seq` to perform inference on the latest trained checkpoint over the test set and using a single GPU with batch size 16:
```
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_2560-54_126-640_640-160_320-rot_15_345_1-sub_8-lac-batch_72-fbb,_eval_,batch-16,save-vis-0,dbg-0,dyn-1,seg-0_15:p-640:r-2560:sub-8,lac
```
This will save the predicted segmentation masks under `~/p2s/log/seg/resnet_640_resize_2560-54_126-640_640-160_320-rot_15_345_1-sub_8-lac-batch_72-fbb/ckpt-<step>-resize_2560-0_15-640_640-640_640-sub_8-lac/masks-batch_16`.

<a id="live_inferenc_e_"></a>
### Live-inference
A live-inference version of the above command is available in [`tf_seg-ipsc.md`](cmd/p2s_seg-ipsc.md) under [`on-0_15 @ aug-strd-lac-fbb/r-2560-p-640-sub-8/54_126-->p2s_seg-ipsc`](cmd/p2s_seg-ipsc.md#on-0_15--------aug-strd-lac-fbbr-2560-p-640-sub-854_126--p2s_seg-ipsc):
```
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_2560-54_126-640_640-160_320-rot_15_345_1-sub_8-lac-batch_72-fbb,_eval_,batch-16,save-vis-0,dbg-0,dyn-1,seg-0_15:p-640:r-2560:sub-8,lac,grs
```
The only difference is the `grs` at the end that specifies the network configuration to connect to the training server. 

Live-inference means that the inference is run repeatedly in parallel alongside training and preferably on a separate server using the latest available checkpoint
- the inference server periodically polls the training server for a new checkpoint (over ssh), copies the checkpoint (if one exists) to the inference server (using scp) and runs inference on it
- the default polling interval is 2 hours or the time it takes to run inference on a single checkpoint, whichever is longer
- training server configurations are specified in json5 file, e.g. [grs.json5](configs/j5/grs.json5) specifies the configuration for one our servers
    - the script supports reading network connection data from `~/.ssh/config` file

<a id="evaluate_segmentation_performanc_e_"></a>
## Evaluate segmentation performance
Run the following command in `~/617` to evaluate the segmentation performance of the trained model over the test set:
```
python stitch.py cfg=ipsc:r-2560:p-640:sub-8:lac:frame-0_15:batch-16:vis-0:_in_-resnet_640_resize_2560-54_126-640_640-160_320-rot_15_345_1-sub_8-lac-batch_72-fbb/ckpt-__var__:_out_-ipsc-54_126-r-2560-p-640-sub-8-aug-strd-lac-0_15-fbb
```
This command is available in [`stitch-ipsc.md`](https://github.com/abhineet123/river_ice_segmentation/blob/master/cmd/stitch-ipsc.md) under [`on-0_15       @ sub-8-aug-strd-lac-fbb/static/54_126-r-2560-p-640-->stitch-ipsc`](https://github.com/abhineet123/river_ice_segmentation/blob/master/cmd/stitch-ipsc.md#on-0_15--------sub-8-aug-strd-lac-fbbstatic54_126-r-2560-p-640--stitch-ipsc)

This command will output the segmentation metrics for each one of the tested checkpoints under `/data/seg/ipsc-54_126-r-2560-p-640-sub-8-aug-strd-lac-0_15-fbb`.
An example evaluation over 39 checkpoints is available [here](https://huggingface.co/abhineet123/p2s-video/blob/main/evaluation/ipsc-54_126-r-2560-p-640-sub-8-aug-strd-lac-0_15-fbb.zip).

<a id="generate_summary_plot_s_"></a>
### Generate summary plots
This only applies to live-inference scenarios where segmentation performance over multiple checkpoints needs to be summarized.

Run this command in `~/617/plotting` to generate a couple of Excel files containing consolidated data and plots for the metrics over all of the checkpoints:
```
python concat_metrics_seg.py multi=1 list_paths=/data/seg/ipsc-54_126-r-2560-p-640-sub-8-aug-strd-lac-0_15-fbb list_from_cb=0
```

The files will be created in `/data/seg/#means/ipsc-54_126-r-2560-p-640-sub-8-aug-strd-lac-0_15-fbb.xlsx` and `/data/seg/#medians/ipsc-54_126-r-2560-p-640-sub-8-aug-strd-lac-0_15-fbb.xlsx`, respectively containing the mean and median of the metrics.

Example Excel files are available here:
- [means](https://huggingface.co/abhineet123/p2s-video/blob/main/evaluation/%23means/ipsc-54_126-r-2560-p-640-sub-8-aug-strd-lac-0_15-fbb.xlsx)
- [medians](https://huggingface.co/abhineet123/p2s-video/blob/main/evaluation/%23medians/ipsc-54_126-r-2560-p-640-sub-8-aug-strd-lac-0_15-fbb.xlsx)

<a id="video_object_detection_pipeline_"></a>
# Video Object Detection Pipeline
Following sections demonstrate the complete step-by-step pipeline for creating video object detection training and test sets from the UA-DETRAC dataset, generating tfrecord files from these, training a model, running inference on the trained model, and evaluating the inference outputs against the ground truth.

Assumptions:
- We want to train on the standard UA-DETRAC training set containing the first 60 sequences and test it on the standard test dataset containing the remaining 40 sequences
- We want train on video subsequences made up of _`N=2`_ consecutive frames
- We want to use the 640x640 input size version of the ResNet50 backbone and keep the backbone weights frozen during training
- We want to use the late-fusion video architecture 
- We want to finetune a model with COCO-pretrained weights instead of training from scratch

These assumptions correspond to the primary UA-DETRAC video detection model reported in the [paper](docs/p2s_vid_det_paper.pdf).

<a id="setup_dataset_and_softlink_s__1"></a>
## Setup dataset and softlinks
Download our pre-formatted version of the [UA-DETRAC dataset](https://huggingface.co/datasets/abhineet123/ua_detrac) training and test sets and extract the archives into `/data/DETRAC/Images`:

```
mkdir /data/DETRAC/Images && cd /data/DETRAC/Images

wget https://huggingface.co/datasets/abhineet123/ua_detrac/blob/main/ua_detrac_training_set.zip
unzip ua_detrac_training_set.zip

wget https://huggingface.co/datasets/abhineet123/ua_detrac/blob/main/ua_detrac_test_set.zip
unzip ua_detrac_test_set.zip
```

Create a softlink to `/data` as `~/pix2seq/datasets`:
```
cd ~/pix2seq
ln -s /data datasets
```

Create `~/pix2seq/log` if needed and softlink to it as `~/ipsc/ipsc_data_processing/log/p2s`:
```
mkdir ~/pix2seq/log
cd ~/ipsc/ipsc_data_processing
mkdir log && cd log
ln -s ~/pix2seq/log p2s
```

<a id="setup_pretrained_weight_s__1"></a>
## Setup pretrained weights
Follow the instructions in the [semantic segmentation pipeline](#setup_pretrained_weight_s_).

<a id="generate_json_file_s_"></a>
## Generate JSON files
Run this command in `~/ipsc/ipsc_data_processing` to generate JSON files for training and test sets in the [Youtube VIS 2019](https://youtube-vos.org/dataset/vis/) format:
```
python xml_to_ytvis.py cfg=detrac:0_59:proc-1:len-2:strd-1:gap-1:ign

python xml_to_ytvis.py cfg=detrac:60_99:proc-1:len-2:strd-1:gap-1:ign
```
These commands are available in
[`xml_to_ytvis.md`](https://github.com/abhineet123/ipsc_prediction/blob/master/ipsc_data_processing/cmd/xml_to_ytvis.md)
under
[`len-2 @ 0_59/detrac-->xml_to_ytvis`](https://github.com/abhineet123/ipsc_prediction/blob/master/ipsc_data_processing/cmd/xml_to_ytvis.md#len-2--------0_59detrac--xml_to_ytvis)
and
[`### len-2 @ 60_99/detrac-->xml_to_ytvis`](https://github.com/abhineet123/ipsc_prediction/blob/master/ipsc_data_processing/cmd/xml_to_ytvis.md#len-2--------60_99detrac--xml_to_ytvis).

<a id="samplin_g_"></a>
### sampling
The full test set is far too large for [live inference](#live_inferenc_e_), so we also provide support for sampling small subsets for this purpose.
In our experience, the performance on randomly sampled subsets closely mirrors the performance on the full set.

For example, the following command will sample 40 random subsequences (or 80 frames) from each sequence, thereby reducing the test set size to only 1.6K subsequences:
```
python data/scripts/create_video_tfrecord.py cfg=detrac:60_99:len-2:80_per_seq_random_len_2:strd-2:asi-0
```
Note that this command uses a stride of 2 instead of 1 in the earlier commands so as to generate non-overlapping subsequences.
Using overlapping subsequences can provide redundancy in the detection outputs which can improve the performance somewhat but it is not necessary for live inference.

This command requires that the [sample lists](https://huggingface.co/datasets/abhineet123/ua_detrac/blob/main/detrac_80_per_seq_random_len_2.zip) be downloaded and extracted to `/data/DETRAC`.

```
cd /data/DETRAC
wget https://huggingface.co/datasets/abhineet123/ua_detrac/blob/main/detrac_80_per_seq_random_len_2.zip
unzip detrac_80_per_seq_random_len_2.zip

```

The script that we use to generate such sample lists is available in our [animal detection repo](https://github.com/abhineet123/animal_detection/blob/master/tf_api/csv_to_record.py) along with the [commands for generating these samples](https://github.com/abhineet123/animal_detection/blob/master/tf_api/cmd/csv_to_record.md#detrac--------sampling_based--csv_to_record).

<a id="static_detectio_n_"></a>
### static detection
JSON files for static object detection need to be in the [COCO format](https://cocodataset.org/#format-data) and can be generated using [xml_to_coco.py](https://github.com/abhineet123/ipsc_prediction/blob/master/ipsc_data_processing/xml_to_coco.py).
The corresponding commands are available in [xml_to_coco.md](https://github.com/abhineet123/ipsc_prediction/blob/master/ipsc_data_processing/cmd/xml_to_coco.md#detrac)

<a id="create_tfrecord_file_s__1"></a>
## Create tfrecord files
Run the following commands in `~/pix2seq` to generate tfrecord files from the subpatch datasets:
```
python data/scripts/create_video_tfrecord.py cfg=detrac:0_59:len-2:strd-1:asi-1

python data/scripts/create_video_tfrecord.py cfg=detrac:60_99:len-2:strd-1:asi-1
```
These commands are available in
[`tf_vid-isl`](cmd/tf_vid-isl.md)
under
[`len-2 @ 0_59/detrac-->tf_vid-isl`](cmd/tf_vid-isl.md#len-2--------0_59detrac--tf_vid-isl)
and 
[`len-2 @ 60_99/detrac-->tf_vid-isl`](cmd/tf_vid-isl.md#len-2--------60_99detrac--tf_vid-isl) respectively

<a id="samplin_g__1"></a>
### sampling
Run the following command to generate tfrecord files for the previously generated sampled subset with 80 frames per sequence:
```
python data/scripts/create_video_tfrecord.py cfg=detrac:60_99:len-2:80_per_seq_random_len_2:strd-2:asi-0
```
<a id="train_mode_l__1"></a>
## Train model
Run the following command in `~/pix2seq` to train a video object detection model with:
- 640x640 input-size version of the ResNet-50 backbone 
- late-fusion video architecture
- pretrained weights loaded before training (instead of training from scratch)
- backbone weights kept frozen during training
- data augmentation with source images resized to 1280x1280 followed by random scaling, jittering, cropping to finally extract 640x640 patches for training
- single-machine dual-gpu training over a pair of A100 80GB GPUs using the corresponding maximum batch size of _`B=256`_
```
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,detrac-0_59,batch-256,dbg-0,dyn-1,dist-1,len-2,fbb,jtr,res-1280,lfn
```
This command is available in [`p2s_vid-isl.md`](cmd/p2s_vid-isl.md) under [`len-2-aug-fbb @ detrac-0_59/lfn-->p2s_vid-isl`](cmd/p2s_vid-isl.md#len-2-aug-fbb--------detrac-0_59lfn--p2s_vid-isl).

This will save the checkpoints and tensorboard log files under `~/pix2seq/log/video/resnet_640_detrac-length-2-stride-1-seq-0_59-batch_256-fbb-jtr-res_1280-lfn`.

<a id="pseudo_multi_machine_mode_"></a>
### Pseudo multi-machine mode
We trained this model on a cloud-based VM with 2 x A100 80 GB GPUs and, for some networking-related reason, simply running in single-machine dual-GPU trainingmode as above led to very low GPU usage and therefore very slow training.
We were able to fix this by running in pseudo-multi-machine training mode by using `CUDA_VISIBLE_DEVICES` to hide one GPU each in two different runs to simulate running on two machines with a single A100 each.
The commands for running this way are under [len-2-aug-fbb-self2 @ detrac-0_59/lfn-->p2s_vid-isl](cmd/p2s_vid-isl.md#len-2-aug-fbb-self2--------detrac-0_59lfn--p2s_vid-isl).

<a id="run_inferenc_e__1"></a>
## Run inference
Run the following command in `~/pix2seq` to perform inference on the latest trained checkpoint over the test set and using a single RTX 3090 with batch size 32:
```
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_static_video_det.py --j5=_eval_,vid_det,m-resnet_640_detrac-length-2-stride-1-seq-0_59-batch_256-fbb-jtr-res_1280-lfn,detrac-60_99,batch-32,save-vis-0,dbg-0,dyn-1,dist-0,len-2,vstrd-1,asi-1
```
This will save the predicted detections as sequence-wise csv files under `~/pix2seq/log/video/resnet_640_detrac-length-2-stride-1-seq-0_59-batch_256-fbb-jtr-res_1280-lfn/ckpt-21582-detrac-length-2-stride-1-seq-60_99/csv-batch_32`.

This command is available in [`p2s_vid-isl.md`](cmd/p2s_vid-isl.md) under [`on-60_99 @ len-2-aug-fbb/detrac-0_59/lfn-->p2s_vid-isl`](cmd/p2s_vid-isl.md#on-60_99--------len-2-aug-fbbdetrac-0_59lfn--p2s_vid-isl).

<a id="live_inferenc_e__1"></a>
### Live-inference
A live-inference version of the above command on the previously generated sampled subset is available under [`on-60_99-80_per_seq_random_len_2 @ len-2-aug-fbb/detrac-0_59/lfn-->p2s_vid-isl`](cmd/p2s_vid-isl.md#on-60_99-80_per_seq_random_len_2--------len-2-aug-fbbdetrac-0_59lfn--p2s_vid-isl):
```
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_static_video_det.py --j5=_eval_,vid_det,m-resnet_640_detrac-length-2-stride-1-seq-0_59-batch_256-fbb-jtr-res_1280-lfn,detrac-80_per_seq_random_len_2-60_99,batch-4,save-vis-0,dbg-0,dyn-1,dist-0,len-2,vstrd-2,asi-0,isc
```

<a id="evaluate_detection_performanc_e_"></a>
## Evaluate detection performance
Run the following command in `~/ipsc/ipsc_data_processing` to evaluate the detection performance of the latest checkpoint:
```
python3 eval_det.py cfg=p2s:vid,detrac:60_99:gt-1:ign:nms-1:vnms-1:show-0:proc-1:_in_-resnet_640_detrac-length-2-stride-1-seq-0_59-batch_256-fbb-jtr-res_1280-lfn/ckpt-__var__-detrac-length-2-stride-1-seq-60_99/csv-batch_32:_out_-p2s-lfn-detrac-0_59-len-2-60_99-aug-fbb-strd-1
```
This command is available in [`eval_p2s_vid-isl.md`](https://github.com/abhineet123/ipsc_prediction/blob/master/ipsc_data_processing/cmd/eval_p2s_vid-isl.md) under [`on-60_99-strd-1 @ len-2-aug-fbb/detrac-0_59/lfn-->eval_p2s_vid-isl`](https://github.com/abhineet123/ipsc_prediction/blob/master/ipsc_data_processing/cmd/eval_p2s_vid-isl.md#on-60_99-strd-1--------len-2-aug-fbbdetrac-0_59lfn--eval_p2s_vid-isl)

This command will output the detection metrics for the tested checkpoint under `/data/mAP/p2s-lfn-detrac-0_59-len-2-60_99-aug-fbb-strd-1`.

<a id="live_inferenc_e__2"></a>
### Live-inference
A live-inference version of the above command on the sampled subset is available under [`on-60_99-80_per_seq_random_len_2  @ len-2-aug-fbb/detrac-0_59/lfn-->eval_p2s_vid-isl`](https://github.com/abhineet123/ipsc_prediction/blob/master/ipsc_data_processing/cmd/eval_p2s_vid-isl.md#on-60_99-80_per_seq_random_len_2--------len-2-aug-fbbdetrac-0_59lfn--eval_p2s_vid-isl):
```
python3 eval_det.py cfg=p2s:vid,detrac:80_per_seq_random_len_2:60_99:gt-0:ign:nms-1:show-0:proc-1:_in_-resnet_640_detrac-length-2-stride-1-seq-0_59-batch_256-fbb-jtr-res_1280-lfn/ckpt-__var__-detrac-length-2-stride-2-80_per_seq_random_len_2-seq-60_99/csv-batch_4:_out_-p2s-lfn-detrac-0_59-len-2-60_99-80_per_seq_random_len_2-aug-fbb
```

<a id="model_s_"></a>
# Models
The trained models reported in the papers are available in [this hugging face model repo](https://huggingface.co/abhineet123/p2s-video).

Each model archive should be extracted in the `~/pix2seq` while maintaining the folder structure inside the archive.

For example:
```
cd ~/pix2seq
wget https://huggingface.co/abhineet123/p2s-video/blob/main/IPSC/late_stage/log_seg_resnet_640_resize_2560-54_126-640_640-160_320-rot_15_345_1-sub_8-lac-batch_72-fbb.zip
unzip log_seg_resnet_640_resize_2560-54_126-640_640-160_320-rot_15_345_1-sub_8-lac-batch_72-fbb.zip
```
This will extract the checkpoints to `~/pix2seq/log/seg/resnet_640_resize_2560-54_126-640_640-160_320-rot_15_345_1-sub_8-lac-batch_72-fbb/`.

<a id="semantic_segmentation_"></a>
## Semantic Segmentation
<a id="aris_datase_t_"></a>
### ARIS dataset
- [32 training images](https://huggingface.co/abhineet123/p2s-video/blob/main/ARIS/log_seg_resnet_1024_resize_1280-0_31-1024_1024-64_256-rot_15_345_16-flip-sub_8-lac-617-batch_4-seq3k.zip)
- [24 training images](https://huggingface.co/abhineet123/p2s-video/blob/main/ARIS/log_seg_resnet_640_0_23-640_640-64_256-rot_15_345_8-flip-sub_8-lac-617-batch_12-seq1k_d3_260315_160611.zip)
- [16 training images](https://huggingface.co/abhineet123/p2s-video/blob/main/ARIS/log_seg_resnet_640_0_15-640_640-64_256-rot_15_345_8-flip-sub_8-lac-617-batch_12-seq1k_d3_260315_160437.zip)
- [8 training images](https://huggingface.co/abhineet123/p2s-video/blob/main/ARIS/log_seg_resnet_640_0_7-640_640-64_256-rot_15_345_8-flip-sub_8-lac-617-batch_12-seq1k_d3_260315_160118.zip)
- [4 training images](https://huggingface.co/abhineet123/p2s-video/blob/main/ARIS/log_seg_resnet_640_0_3-640_640-64_256-rot_15_345_8-flip-sub_8-lac-617-batch_12-seq1k.zip)

<a id="ipsc_datase_t_"></a>
### IPSC dataset
- early stage training
    - [static segmentation](https://huggingface.co/abhineet123/p2s-video/blob/main/IPSC/early_stage/log_seg_resnet_640_resize_640-16_53-640_640-640_640-rot_15_345_4-flip-sub_2-2d-lac-batch_8-seq3k.zip)
    - [video segmentation](https://huggingface.co/abhineet123/p2s-video/blob/main/IPSC/early_stage/log_video_seg_resnet_640_resize_2560-16_53-640_640-640_640-length-8-stride-1-sub_8-tac-mc-batch_4-seq3k.zip)
- late stage training
    - [static segmentation](https://huggingface.co/abhineet123/p2s-video/blob/main/IPSC/late_stage/log_seg_resnet_640_resize_2560-54_126-640_640-160_320-rot_15_345_1-sub_8-lac-batch_72-fbb.zip)
    - [video segmentation](https://huggingface.co/abhineet123/p2s-video/blob/main/IPSC/late_stage/log_video_seg_resnet_640_resize_2560-54_126-640_640-640_640-length-8-stride-1-sub_8-tac-mc-batch_8-voc15-seq3k-fbb.zip)
    
<a id="object_detectio_n_"></a>
## Object Detection
<a id="ua_detrac_datase_t_"></a>
### UA-DETRAC dataset
- Early Fusion
    - [N=2](https://huggingface.co/abhineet123/p2s-video/resolve/main/DETRAC/log_video_swin_t_pt_640_detrac-length-2-stride-1-seq-0_59-batch_80-jtr-res_1280-self2-0.zip)
- Middle Fusion
    - [N=2](https://huggingface.co/abhineet123/p2s-video/resolve/main/DETRAC/log_video_resnet_640_detrac-length-2-stride-1-seq-0_59-batch_320-fbb-jtr-res_1280.zip)
    - [N=3](https://huggingface.co/abhineet123/p2s-video/resolve/main/DETRAC/log_video_resnet_640_detrac-length-3-stride-1-seq-0_59-batch_216-fbb-jtr-res_1280-seq640-self2-0.zip)
    - [N=4](https://huggingface.co/abhineet123/p2s-video/resolve/main/DETRAC/log_video_resnet_640_detrac-length-4-stride-1-seq-0_59-batch_160-fbb-jtr-res_1280-seq1k-zexg.zip)
- Late Fusion
    - [N=2](https://huggingface.co/abhineet123/p2s-video/blob/main/DETRAC/log_video_resnet_640_detrac-length-2-stride-1-seq-0_59-batch_256-fbb-jtr-res_1280-lfn-self2-0.zip)
    - [N=3](https://huggingface.co/abhineet123/p2s-video/blob/main/DETRAC/log_video_resnet_640_detrac-length-3-stride-1-seq-0_59-batch_168-fbb-jtr-res_1280-lfn-seq640-self2-0.zip)
    - [N=4](https://huggingface.co/abhineet123/p2s-video/blob/main/DETRAC/log_video_resnet_640_detrac-length-4-stride-1-seq-0_59-batch_96-fbb-jtr-res_1280-seq1k-lfn-zexg.zip)

