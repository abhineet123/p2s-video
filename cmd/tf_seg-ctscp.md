<!-- MarkdownTOC -->

- [val       @ rfm-0](#val___rfm_0_)
    - [orig       @ val](#orig___va_l_)
        - [p-1024       @ orig/val](#p_1024___orig_val_)
            - [lac       @ p-1024/orig/val](#lac___p_1024_orig_va_l_)
            - [lac-2d       @ p-1024/orig/val](#lac_2d___p_1024_orig_va_l_)
            - [rfm       @ p-1024/orig/val](#rfm___p_1024_orig_va_l_)
    - [r-1280_640       @ val](#r_1280_640___va_l_)
        - [p-640       @ r-1280_640/val](#p_640___r_1280_640_val_)
            - [rfm       @ p-640/r-1280_640/val](#rfm___p_640_r_1280_640_val_)
            - [cw       @ p-640/r-1280_640/val](#cw___p_640_r_1280_640_val_)
            - [lac       @ p-640/r-1280_640/val](#lac___p_640_r_1280_640_val_)
            - [mlf-2       @ p-640/r-1280_640/val](#mlf_2___p_640_r_1280_640_val_)
- [train       @ rfm-0](#train___rfm_0_)
    - [orig       @ train](#orig___trai_n_)
        - [p-1024       @ orig/train](#p_1024___orig_train_)
            - [lac       @ p-1024/orig/train](#lac___p_1024_orig_trai_n_)
            - [lac-2d       @ p-1024/orig/train](#lac_2d___p_1024_orig_trai_n_)
            - [rfm       @ p-1024/orig/train](#rfm___p_1024_orig_trai_n_)
            - [cw-2d       @ p-1024/orig/train](#cw_2d___p_1024_orig_trai_n_)
            - [no_starts       @ p-1024/orig/train](#no_starts___p_1024_orig_trai_n_)
            - [cw-no_starts       @ p-1024/orig/train](#cw_no_starts___p_1024_orig_trai_n_)
        - [p-640       @ orig/train](#p_640___orig_train_)
            - [rfm       @ p-640/orig/train](#rfm___p_640_orig_train_)
    - [r-1280_640       @ train](#r_1280_640___trai_n_)
        - [p-640       @ r-1280_640/train](#p_640___r_1280_640_train_)
            - [rfm       @ p-640/r-1280_640/train](#rfm___p_640_r_1280_640_train_)
            - [dm       @ p-640/r-1280_640/train](#dm___p_640_r_1280_640_train_)
            - [dm2       @ p-640/r-1280_640/train](#dm2___p_640_r_1280_640_train_)
            - [cw       @ p-640/r-1280_640/train](#cw___p_640_r_1280_640_train_)
            - [cw-2d       @ p-640/r-1280_640/train](#cw_2d___p_640_r_1280_640_train_)
            - [lac       @ p-640/r-1280_640/train](#lac___p_640_r_1280_640_train_)
            - [no_starts       @ p-640/r-1280_640/train](#no_starts___p_640_r_1280_640_train_)
            - [cw-no_starts       @ p-640/r-1280_640/train](#cw_no_starts___p_640_r_1280_640_train_)
    - [r-1024       @ train](#r_1024___trai_n_)
        - [json       @ r-1024/train](#json___r_1024_train_)
        - [stats       @ r-1024/train](#stats___r_1024_train_)
            - [dm       @ stats/r-1024/train](#dm___stats_r_1024_train_)
            - [dm2       @ stats/r-1024/train](#dm2___stats_r_1024_train_)
        - [rfm       @ r-1024/train](#rfm___r_1024_train_)
    - [r-640       @ train](#r_640___trai_n_)
        - [dm       @ r-640/train](#dm___r_640_trai_n_)
        - [dm2       @ r-640/train](#dm2___r_640_trai_n_)
            - [stof-200       @ dm2/r-640/train](#stof_200___dm2_r_640_trai_n_)
        - [lac       @ r-640/train](#lac___r_640_trai_n_)
            - [vid       @ lac/r-640/train](#vid___lac_r_640_trai_n_)
            - [img       @ lac/r-640/train](#img___lac_r_640_trai_n_)

<!-- /MarkdownTOC -->

<a id="val___rfm_0_"></a>
# val       @ rfm-0-->tf_seg-coco
<a id="orig___va_l_"></a>
## orig       @ val-->tf_seg-ctscp
<a id="p_1024___orig_val_"></a>
### p-1024       @ orig/val-->tf_seg-ctscp
<a id="lac___p_1024_orig_va_l_"></a>
#### lac       @ p-1024/orig/val-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:val:p-1024:sub-8:lac:chk-0:vid-0
<a id="lac_2d___p_1024_orig_va_l_"></a>
#### lac-2d       @ p-1024/orig/val-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:val:p-1024:sub-8:lac:2d:chk-0:vid-0

<a id="rfm___p_1024_orig_va_l_"></a>
#### rfm       @ p-1024/orig/val-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:val:p-1024:rfm:vid-0

<a id="r_1280_640___va_l_"></a>
## r-1280_640       @ val-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:val:r-1280_640:sub-8:lac:chk-0:stats-2:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:val:r-1280_640:sub-4:lac:chk-0:stats-2:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:val:r-1280_640:sub-2:lac:chk-0:stats-2:vid-0
<a id="p_640___r_1280_640_val_"></a>
### p-640       @ r-1280_640/val-->tf_seg-ctscp
<a id="rfm___p_640_r_1280_640_val_"></a>
#### rfm       @ p-640/r-1280_640/val-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:val:r-1280_640:p-640:rfm:vid-0
<a id="cw___p_640_r_1280_640_val_"></a>
#### cw       @ p-640/r-1280_640/val-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:val:r-1280_640:p-640:sub-8:cw:chk-0:stats-2:vid-0
<a id="lac___p_640_r_1280_640_val_"></a>
#### lac       @ p-640/r-1280_640/val-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:val:r-1280_640:p-640:sub-8:lac:chk-0:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:val:r-1280_640:p-640:sub-4:lac:chk-0:stats-2:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:val:r-1280_640:p-640:sub-2:lac:chk-0:stats-2:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:val:r-1280_640:p-640:sub-1:lac:chk-0:stats-2:vid-0
<a id="mlf_2___p_640_r_1280_640_val_"></a>
#### mlf-2       @ p-640/r-1280_640/val-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:val:r-1280_640:p-640:sub-8:lac:chk-0:stats-2:vid-0:mlf-2
python data/scripts/create_seg_tfrecord.py cfg=ctscp:val:r-1280_640:p-640:sub-4:lac:chk-0:stats-2:vid-0:mlf-2
python data/scripts/create_seg_tfrecord.py cfg=ctscp:val:r-1280_640:p-640:sub-2:lac:chk-0:stats-2:vid-0:mlf-2

`dbg`
python data/scripts/create_seg_tfrecord.py cfg=ctscp:val:r-1280_640:p-640:sub-8:chk-0:stats-2:vid-0:mlf-20


<a id="train___rfm_0_"></a>
# train       @ rfm-0-->tf_seg-coco
<a id="orig___trai_n_"></a>
## orig       @ train-->tf_seg-ctscp
<a id="p_1024___orig_train_"></a>
### p-1024       @ orig/train-->tf_seg-ctscp
<a id="lac___p_1024_orig_trai_n_"></a>
#### lac       @ p-1024/orig/train-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:p-1024:sub-8:lac:chk-0:stats-2:vid-0
<a id="lac_2d___p_1024_orig_trai_n_"></a>
#### lac-2d       @ p-1024/orig/train-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:p-1024:sub-8:lac:2d:chk-0:stats-2:vid-0
<a id="rfm___p_1024_orig_trai_n_"></a>
#### rfm       @ p-1024/orig/train-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:p-1024:rfm:vid-0
`seq-0`
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:p-1024:rfm:vid-0:seq-0
<a id="cw_2d___p_1024_orig_trai_n_"></a>
#### cw-2d       @ p-1024/orig/train-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:p-1024:sub-8:cw:2d:chk-0:stats-2:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:p-1024:sub-4:cw:2d:chk-0:stats-2:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:p-1024:sub-2:cw:2d:chk-0:stats-2:vid-0
<a id="no_starts___p_1024_orig_trai_n_"></a>
#### no_starts       @ p-1024/orig/train-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:p-1024:sub-8:no_starts:chk-1:stats-1:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:p-1024:sub-4:no_starts:chk-1:stats-1:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:p-1024:sub-2:no_starts:chk-1:stats-1:vid-0
<a id="cw_no_starts___p_1024_orig_trai_n_"></a>
#### cw-no_starts       @ p-1024/orig/train-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:p-1024:sub-8:cw:no_starts:chk-0:stats-2:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:p-1024:sub-4:cw:no_starts:chk-0:stats-2:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:p-1024:sub-2:cw:no_starts:chk-0:stats-2:vid-0

<a id="p_640___orig_train_"></a>
### p-640       @ orig/train-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:p-640:sub-8:lac:chk-0:stats-2:vid-0

<a id="rfm___p_640_orig_train_"></a>
#### rfm       @ p-640/orig/train-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:p-640:rfm:vid-0

<a id="r_1280_640___trai_n_"></a>
## r-1280_640       @ train-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1280_640:sub-8:lac:chk-0:stats-2:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1280_640:sub-4:lac:chk-0:stats-2:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1280_640:sub-2:lac:chk-0:stats-2:vid-0
<a id="p_640___r_1280_640_train_"></a>
### p-640       @ r-1280_640/train-->tf_seg-ctscp
<a id="rfm___p_640_r_1280_640_train_"></a>
#### rfm       @ p-640/r-1280_640/train-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1280_640:p-640:rfm:vid-0
<a id="dm___p_640_r_1280_640_train_"></a>
#### dm       @ p-640/r-1280_640/train-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1280_640:p-640:sub-8:dm:chk-0:stats-2:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1280_640:p-640:sub-4:dm:chk-0:stats-2:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1280_640:p-640:sub-2:dm:chk-1:stats-1:vid-0
<a id="dm2___p_640_r_1280_640_train_"></a>
#### dm2       @ p-640/r-1280_640/train-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1280_640:p-640:sub-8:dm2:chk-1:stats-1:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1280_640:p-640:sub-4:dm2:chk-1:stats-1:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1280_640:p-640:sub-2:dm2:chk-1:stats-1:vid-0
<a id="cw___p_640_r_1280_640_train_"></a>
#### cw       @ p-640/r-1280_640/train-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1280_640:p-640:sub-8:cw:chk-0:stats-2:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1280_640:p-640:sub-4:cw:chk-0:stats-2:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1280_640:p-640:sub-2:cw:chk-0:stats-2:vid-0
<a id="cw_2d___p_640_r_1280_640_train_"></a>
#### cw-2d       @ p-640/r-1280_640/train-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1280_640:p-640:sub-8:cw:2d:chk-0:stats-2:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1280_640:p-640:sub-4:cw:2d:chk-0:stats-2:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1280_640:p-640:sub-2:cw:2d:chk-0:stats-2:vid-0
<a id="lac___p_640_r_1280_640_train_"></a>
#### lac       @ p-640/r-1280_640/train-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1280_640:p-640:sub-8:lac:chk-0:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1280_640:p-640:sub-4:lac:chk-0:stats-2:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1280_640:p-640:sub-2:lac:chk-0:stats-2:vid-0
<a id="no_starts___p_640_r_1280_640_train_"></a>
#### no_starts       @ p-640/r-1280_640/train-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1280_640:p-640:sub-8:no_starts:chk-1:stats-1:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1280_640:p-640:sub-4:no_starts:chk-1:stats-1:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1280_640:p-640:sub-2:no_starts:chk-1:stats-1:vid-0
<a id="cw_no_starts___p_640_r_1280_640_train_"></a>
#### cw-no_starts       @ p-640/r-1280_640/train-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1280_640:p-640:sub-8:cw:no_starts:chk-0:stats-2:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1280_640:p-640:sub-4:cw:no_starts:chk-0:stats-2:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1280_640:p-640:sub-2:cw:no_starts:chk-0:stats-2:vid-0


<a id="r_1024___trai_n_"></a>
## r-1024       @ train-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1024:sub-8:lac:chk-0
<a id="json___r_1024_train_"></a>
### json       @ r-1024/train-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1024:sub-8:lac:chk-0:json
<a id="stats___r_1024_train_"></a>
### stats       @ r-1024/train-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1024:sub-8:lac:chk-0:stats-2
<a id="dm___stats_r_1024_train_"></a>
#### dm       @ stats/r-1024/train-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1024:sub-8:dm:chk-1:stats-2:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1024:sub-4:dm:chk-1:stats-2:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1024:sub-2:dm:chk-1:stats-2:vid-0
<a id="dm2___stats_r_1024_train_"></a>
#### dm2       @ stats/r-1024/train-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1024:sub-8:dm2:chk-1:stats-2:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1024:sub-4:dm2:chk-1:stats-2:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1024:sub-2:dm2:chk-1:stats-2:vid-0
<a id="rfm___r_1024_train_"></a>
### rfm       @ r-1024/train-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-1024:rfm:mc

<a id="r_640___trai_n_"></a>
## r-640       @ train-->tf_seg-ctscp
<a id="dm___r_640_trai_n_"></a>
### dm       @ r-640/train-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-640:sub-8:dm:chk-1:stats-2:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-640:sub-4:dm:chk-1:stats-2:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-640:sub-2:dm:chk-1:stats-2:vid-0
<a id="dm2___r_640_trai_n_"></a>
### dm2       @ r-640/train-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-640:sub-8:dm2:chk-0:stats-0:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-640:sub-4:dm2:chk-1:stats-2:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-640:sub-2:dm2:chk-1:stats-2:vid-0
<a id="stof_200___dm2_r_640_trai_n_"></a>
#### stof-200       @ dm2/r-640/train-->tf_seg-ctscp
 python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-640:sub-8:dm2:chk-1:stats-0:vid-0:stof-200
<a id="lac___r_640_trai_n_"></a>
### lac       @ r-640/train-->tf_seg-ctscp
<a id="vid___lac_r_640_trai_n_"></a>
#### vid       @ lac/r-640/train-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-640:sub-8:lac:chk-0:stats-2
`dbg`
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-640:sub-8:lac:chk-0:stats-2:seq-0
<a id="img___lac_r_640_trai_n_"></a>
#### img       @ lac/r-640/train-->tf_seg-ctscp
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-640:sub-8:lac:chk-0:vid-0
python data/scripts/create_seg_tfrecord.py cfg=ctscp:val:r-640:sub-8:lac:chk-0:vid-0

python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-640:sub-8:lac:chk-0:stats-2:vid-0
`dbg`
python data/scripts/create_seg_tfrecord.py cfg=ctscp:train:r-640:sub-8:lac:chk-0:stats-2:vid-0:seq-0


