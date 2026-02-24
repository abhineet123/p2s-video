<!-- MarkdownTOC -->

- [panopticapi](#panopticapi_)
    - [train-rfm       @ panopticapi](#train_rfm___panopticap_i_)
    - [val-rfm       @ panopticapi](#val_rfm___panopticap_i_)
        - [end-400       @ val-rfm/panopticapi](#end_400___val_rfm_panopticap_i_)
- [train       @ rfm-0](#train___rfm_0_)
    - [r-640       @ train](#r_640___trai_n_)
        - [p-640       @ r-640/train](#p_640___r_640_trai_n_)
            - [end-100       @ p-640/r-640/train](#end_100___p_640_r_640_trai_n_)
            - [end-2000       @ p-640/r-640/train](#end_2000___p_640_r_640_trai_n_)
        - [p-0-aug       @ r-640/train](#p_0_aug___r_640_trai_n_)
    - [r-1280       @ train](#r_1280___trai_n_)
        - [p-640-sub-8       @ r-1280/train](#p_640_sub_8___r_1280_train_)
        - [p-640-sub-4       @ r-1280/train](#p_640_sub_4___r_1280_train_)
            - [end-100       @ p-640-sub-4/r-1280/train](#end_100___p_640_sub_4_r_1280_train_)
            - [end-2000       @ p-640-sub-4/r-1280/train](#end_2000___p_640_sub_4_r_1280_train_)
            - [end-600       @ p-640-sub-4/r-1280/train](#end_600___p_640_sub_4_r_1280_train_)
- [val       @ rfm-0](#val___rfm_0_)
    - [r-640       @ val](#r_640___va_l_)
        - [p-640       @ r-640/val](#p_640___r_640_va_l_)
            - [end-1000       @ p-640/r-640/val](#end_1000___p_640_r_640_va_l_)
    - [r-1280       @ val](#r_1280___va_l_)
        - [p-640-sub-8       @ r-1280/val](#p_640_sub_8___r_1280_val_)
        - [p-640-sub-4       @ r-1280/val](#p_640_sub_4___r_1280_val_)
            - [end-100       @ p-640-sub-4/r-1280/val](#end_100___p_640_sub_4_r_1280_val_)
            - [end-2000       @ p-640-sub-4/r-1280/val](#end_2000___p_640_sub_4_r_1280_val_)
            - [end-500       @ p-640-sub-4/r-1280/val](#end_500___p_640_sub_4_r_1280_val_)

<!-- /MarkdownTOC -->

<a id="panopticapi_"></a>
# panopticapi
cd datasets/coco
unzip panoptic_annotations_trainval2017.zip

cd annotations/
unzip panoptic_train2017.zip
unzip panoptic_val2017.zip

<a id="train_rfm___panopticap_i_"></a>
## train-rfm       @ panopticapi-->tf_seg-coco
python data/scripts/panopticapi/converters/panoptic2semantic_segmentation.py --input_json_file panoptic_train2017.json --segmentations_folder panoptic_train2017 --output_json_file semantic_train2017.json

python data/scripts/panopticapi/converters/panoptic2semantic_segmentation.py --input_json_file panoptic_train2017.json --segmentations_folder panoptic_train2017 --semantic_seg_folder semantic_train2017

python data/scripts/create_seg_tfrecord.py cfg=coco:rfm

<a id="val_rfm___panopticap_i_"></a>
## val-rfm       @ panopticapi-->tf_seg-coco
python data/scripts/panopticapi/converters/panoptic2semantic_segmentation.py --input_json_file panoptic_val2017.json --segmentations_folder panoptic_val2017 --output_json_file semantic_val2017.json

python data/scripts/panopticapi/converters/panoptic2semantic_segmentation.py --input_json_file panoptic_val2017.json --segmentations_folder panoptic_val2017 --semantic_seg_folder semantic_val2017 

python data/scripts/create_seg_tfrecord.py cfg=coco:val:rfm

<a id="end_400___val_rfm_panopticap_i_"></a>
### end-400       @ val-rfm/panopticapi-->tf_seg-coco
python data/scripts/create_seg_tfrecord.py cfg=coco:val:rfm:end-400

<a id="train___rfm_0_"></a>
# train       @ rfm-0-->tf_seg-coco
<a id="r_640___trai_n_"></a>
## r-640       @ train-->tf_seg-coco
<a id="p_640___r_640_trai_n_"></a>
### p-640       @ r-640/train-->tf_seg-coco
python data/scripts/create_seg_tfrecord.py cfg=coco:r-640:p-640:sub-8:lac:chk-0
<a id="end_100___p_640_r_640_trai_n_"></a>
#### end-100       @ p-640/r-640/train-->tf_seg-coco
python data/scripts/create_seg_tfrecord.py cfg=coco:r-640:p-640:sub-8:lac:end-100
<a id="end_2000___p_640_r_640_trai_n_"></a>
#### end-2000       @ p-640/r-640/train-->tf_seg-coco
python data/scripts/create_seg_tfrecord.py cfg=coco:r-640:p-640:sub-8:lac:end-2000

<a id="p_0_aug___r_640_trai_n_"></a>
### p-0-aug       @ r-640/train-->tf_seg-coco
python data/scripts/create_seg_tfrecord.py cfg=coco:r-640:p-0:rot-15_345_4:sub-8:lac


<a id="r_1280___trai_n_"></a>
## r-1280       @ train-->tf_seg-coco
<a id="p_640_sub_8___r_1280_train_"></a>
### p-640-sub-8       @ r-1280/train-->tf_seg-coco
python data/scripts/create_seg_tfrecord.py cfg=coco:r-1280:p-640:sub-8:lac:chk-0
<a id="p_640_sub_4___r_1280_train_"></a>
### p-640-sub-4       @ r-1280/train-->tf_seg-coco
python data/scripts/create_seg_tfrecord.py cfg=coco:r-1280:p-640:sub-4:lac:stats:json
python data/scripts/create_seg_tfrecord.py cfg=coco:r-1280:p-640:sub-4:stats:json
<a id="end_100___p_640_sub_4_r_1280_train_"></a>
#### end-100       @ p-640-sub-4/r-1280/train-->tf_seg-coco
python data/scripts/create_seg_tfrecord.py cfg=coco:r-1280:p-640:sub-4:lac:end-100
python data/scripts/create_seg_tfrecord.py cfg=coco:r-1280:p-640:sub-4:end-100
<a id="end_2000___p_640_sub_4_r_1280_train_"></a>
#### end-2000       @ p-640-sub-4/r-1280/train-->tf_seg-coco
python data/scripts/create_seg_tfrecord.py cfg=coco:r-1280:p-640:sub-4:end-2000
<a id="end_600___p_640_sub_4_r_1280_train_"></a>
#### end-600       @ p-640-sub-4/r-1280/train-->tf_seg-coco
python data/scripts/create_seg_tfrecord.py cfg=coco:r-1280:p-640:sub-4:end-600

<a id="val___rfm_0_"></a>
# val       @ rfm-0-->tf_seg-coco
<a id="r_640___va_l_"></a>
## r-640       @ val-->tf_seg-coco
<a id="p_640___r_640_va_l_"></a>
### p-640       @ r-640/val-->tf_seg-coco
python data/scripts/create_seg_tfrecord.py cfg=coco:val:r-640:p-640:sub-8:lac:chk-0
`dbg`
python data/scripts/create_seg_tfrecord.py cfg=coco:val:r-640:p-640:sub-8:lac:end-100
<a id="end_1000___p_640_r_640_va_l_"></a>
#### end-1000       @ p-640/r-640/val-->tf_seg-coco
python data/scripts/create_seg_tfrecord.py cfg=coco:val:r-640:p-640:sub-8:lac:end-1000

<a id="r_1280___va_l_"></a>
## r-1280       @ val-->tf_seg-coco
<a id="p_640_sub_8___r_1280_val_"></a>
### p-640-sub-8       @ r-1280/val-->tf_seg-coco
python data/scripts/create_seg_tfrecord.py cfg=coco:val:r-1280:p-640:sub-8:lac:chk-0
<a id="p_640_sub_4___r_1280_val_"></a>
### p-640-sub-4       @ r-1280/val-->tf_seg-coco
python data/scripts/create_seg_tfrecord.py cfg=coco:val:r-1280:p-640:sub-4:lac:stats:json
python data/scripts/create_seg_tfrecord.py cfg=coco:val:r-1280:p-640:sub-4
<a id="end_100___p_640_sub_4_r_1280_val_"></a>
#### end-100       @ p-640-sub-4/r-1280/val-->tf_seg-coco
python data/scripts/create_seg_tfrecord.py cfg=coco:val:r-1280:p-640:sub-4:lac:end-100
python data/scripts/create_seg_tfrecord.py cfg=coco:val:r-1280:p-640:sub-4:end-100
<a id="end_2000___p_640_sub_4_r_1280_val_"></a>
#### end-2000       @ p-640-sub-4/r-1280/val-->tf_seg-coco
python data/scripts/create_seg_tfrecord.py cfg=coco:val:r-1280:p-640:sub-4:end-2000
<a id="end_500___p_640_sub_4_r_1280_val_"></a>
#### end-500       @ p-640-sub-4/r-1280/val-->tf_seg-coco
python data/scripts/create_seg_tfrecord.py cfg=coco:val:r-1280:p-640:sub-4:end-500
