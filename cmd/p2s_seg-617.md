<!-- MarkdownTOC -->

- [0_3](#0_3_)
    - [orig-p-640-sub-8-aug-lac-fbb       @ 0_3](#orig_p_640_sub_8_aug_lac_fbb___0_3_)
            - [on-32_49       @ orig-p-640-sub-8-aug-lac-fbb/0_3](#on_32_49___orig_p_640_sub_8_aug_lac_fbb_0_3_)
    - [orig-p-640-sub-8-aug-lac       @ 0_3](#orig_p_640_sub_8_aug_lac___0_3_)
            - [on-32_49       @ orig-p-640-sub-8-aug-lac/0_3](#on_32_49___orig_p_640_sub_8_aug_lac_0_3_)
- [0_7](#0_7_)
    - [orig-p-640-sub-8-aug-lac-fbb       @ 0_7](#orig_p_640_sub_8_aug_lac_fbb___0_7_)
            - [on-32_49       @ orig-p-640-sub-8-aug-lac-fbb/0_7](#on_32_49___orig_p_640_sub_8_aug_lac_fbb_0_7_)
    - [orig-p-640-sub-8-aug-lac       @ 0_7](#orig_p_640_sub_8_aug_lac___0_7_)
            - [on-32_49       @ orig-p-640-sub-8-aug-lac/0_7](#on_32_49___orig_p_640_sub_8_aug_lac_0_7_)
- [0_15](#0_1_5_)
    - [orig-p-640-sub-8-aug-lac-fbb       @ 0_15](#orig_p_640_sub_8_aug_lac_fbb___0_15_)
            - [on-32_49       @ orig-p-640-sub-8-aug-lac-fbb/0_15](#on_32_49___orig_p_640_sub_8_aug_lac_fbb_0_1_5_)
    - [orig-p-640-sub-8-aug-lac       @ 0_15](#orig_p_640_sub_8_aug_lac___0_15_)
            - [on-32_49       @ orig-p-640-sub-8-aug-lac/0_15](#on_32_49___orig_p_640_sub_8_aug_lac_0_1_5_)
- [0_23](#0_2_3_)
    - [orig-p-640-sub-8-aug-lac-fbb       @ 0_23](#orig_p_640_sub_8_aug_lac_fbb___0_23_)
            - [on-32_49       @ orig-p-640-sub-8-aug-lac-fbb/0_23](#on_32_49___orig_p_640_sub_8_aug_lac_fbb_0_2_3_)
    - [orig-p-640-sub-8-aug-lac       @ 0_23](#orig_p_640_sub_8_aug_lac___0_23_)
            - [on-32_49       @ orig-p-640-sub-8-aug-lac/0_23](#on_32_49___orig_p_640_sub_8_aug_lac_0_2_3_)
- [0_31](#0_3_1_)
    - [orig-p-640-sub-8-aug-lac       @ 0_31](#orig_p_640_sub_8_aug_lac___0_31_)
            - [on-train       @ orig-p-640-sub-8-aug-lac/0_31](#on_train___orig_p_640_sub_8_aug_lac_0_3_1_)
            - [on-32_49       @ orig-p-640-sub-8-aug-lac/0_31](#on_32_49___orig_p_640_sub_8_aug_lac_0_3_1_)
    - [orig-p-640-sub-8-aug-lac-fbb       @ 0_31](#orig_p_640_sub_8_aug_lac_fbb___0_31_)
            - [on-32_49       @ orig-p-640-sub-8-aug-lac-fbb/0_31](#on_32_49___orig_p_640_sub_8_aug_lac_fbb_0_3_1_)
    - [r-640       @ 0_31](#r_640___0_31_)
        - [p-640-sub-4-aug-lac       @ r-640/0_31](#p_640_sub_4_aug_lac___r_640_0_31_)
            - [on-train       @ p-640-sub-4-aug-lac/r-640/0_31](#on_train___p_640_sub_4_aug_lac_r_640_0_31_)
            - [on-32_49       @ p-640-sub-4-aug-lac/r-640/0_31](#on_32_49___p_640_sub_4_aug_lac_r_640_0_31_)
        - [p-640-sub-4-aug-lac-fbb       @ r-640/0_31](#p_640_sub_4_aug_lac_fbb___r_640_0_31_)
            - [on-32_49       @ p-640-sub-4-aug-lac-fbb/r-640/0_31](#on_32_49___p_640_sub_4_aug_lac_fbb_r_640_0_31_)
    - [r-1280       @ 0_31](#r_1280___0_31_)
        - [p-1024-sub-8-aug-lac       @ r-1280/0_31](#p_1024_sub_8_aug_lac___r_1280_0_3_1_)
            - [on-32_49       @ p-1024-sub-8-aug-lac/r-1280/0_31](#on_32_49___p_1024_sub_8_aug_lac_r_1280_0_31_)

<!-- /MarkdownTOC -->

<a id="0_3_"></a>
# 0_3
<a id="orig_p_640_sub_8_aug_lac_fbb___0_3_"></a>
## orig-p-640-sub-8-aug-lac-fbb       @ 0_3-->p2s_seg-617
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,617,seg-0_3:p-640:sub-8:rot-15_345_8:strd-64_256:flip,batch-24,dbg-0,dyn-1,dist-0,ep-10000,gz,pt-1,lac,seq1k,fbb
<a id="on_32_49___orig_p_640_sub_8_aug_lac_fbb_0_3_"></a>
#### on-32_49       @ orig-p-640-sub-8-aug-lac-fbb/0_3-->p2s_seg-617
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_0_3-640_640-64_256-rot_15_345_8-flip-sub_8-lac-617-batch_24-seq1k-fbb,_eval_,batch-4,save-vis-0,dbg-0,dyn-1,617,seg-32_49:p-640:sub-8,lac,seq1k,x99

<a id="orig_p_640_sub_8_aug_lac___0_3_"></a>
## orig-p-640-sub-8-aug-lac       @ 0_3-->p2s_seg-617
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,617,seg-0_3:p-640:sub-8:rot-15_345_8:strd-64_256:flip,batch-12,dbg-0,dyn-1,dist-0,ep-10000,gz,pt-1,lac,seq1k
<a id="on_32_49___orig_p_640_sub_8_aug_lac_0_3_"></a>
#### on-32_49       @ orig-p-640-sub-8-aug-lac/0_3-->p2s_seg-617
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_0_3-640_640-64_256-rot_15_345_8-flip-sub_8-lac-617-batch_12-seq1k,_eval_,batch-4,save-vis-0,dbg-0,dyn-1,617,seg-32_49:p-640:sub-8,lac,seq1k,x99

<a id="0_7_"></a>
# 0_7
<a id="orig_p_640_sub_8_aug_lac_fbb___0_7_"></a>
## orig-p-640-sub-8-aug-lac-fbb       @ 0_7-->p2s_seg-617
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,617,seg-0_7:p-640:sub-8:rot-15_345_8:strd-64_256:flip,batch-24,dbg-0,dyn-1,dist-0,ep-10000,gz,pt-1,lac,seq1k,fbb
<a id="on_32_49___orig_p_640_sub_8_aug_lac_fbb_0_7_"></a>
#### on-32_49       @ orig-p-640-sub-8-aug-lac-fbb/0_7-->p2s_seg-617
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_0_7-640_640-64_256-rot_15_345_8-flip-sub_8-lac-617-batch_24-seq1k-fbb,_eval_,batch-4,save-vis-0,dbg-0,dyn-1,617,seg-32_49:p-640:sub-8,lac,seq1k,x99

<a id="orig_p_640_sub_8_aug_lac___0_7_"></a>
## orig-p-640-sub-8-aug-lac       @ 0_7-->p2s_seg-617
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,617,seg-0_7:p-640:sub-8:rot-15_345_8:strd-64_256:flip,batch-12,dbg-0,dyn-1,dist-0,ep-10000,gz,pt-1,lac,seq1k
<a id="on_32_49___orig_p_640_sub_8_aug_lac_0_7_"></a>
#### on-32_49       @ orig-p-640-sub-8-aug-lac/0_7-->p2s_seg-617
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_0_7-640_640-64_256-rot_15_345_8-flip-sub_8-lac-617-batch_12-seq1k,_eval_,batch-4,save-vis-0,dbg-0,dyn-1,617,seg-32_49:p-640:sub-8,lac,seq1k,x99


<a id="0_1_5_"></a>
# 0_15
<a id="orig_p_640_sub_8_aug_lac_fbb___0_15_"></a>
## orig-p-640-sub-8-aug-lac-fbb       @ 0_15-->p2s_seg-617
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,617,seg-0_15:p-640:sub-8:rot-15_345_8:strd-64_256:flip,batch-24,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,lac,seq1k,fbb
<a id="on_32_49___orig_p_640_sub_8_aug_lac_fbb_0_1_5_"></a>
#### on-32_49       @ orig-p-640-sub-8-aug-lac-fbb/0_15-->p2s_seg-617
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_0_15-640_640-64_256-rot_15_345_8-flip-sub_8-lac-617-batch_24-seq1k-fbb,_eval_,batch-4,save-vis-0,dbg-0,dyn-1,617,seg-32_49:p-640:sub-8,lac,seq1k,grs

<a id="orig_p_640_sub_8_aug_lac___0_15_"></a>
## orig-p-640-sub-8-aug-lac       @ 0_15-->p2s_seg-617
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,617,seg-0_15:p-640:sub-8:rot-15_345_8:strd-64_256:flip,batch-12,dbg-0,dyn-1,dist-0,ep-10000,gz,pt-1,lac,seq1k
<a id="on_32_49___orig_p_640_sub_8_aug_lac_0_1_5_"></a>
#### on-32_49       @ orig-p-640-sub-8-aug-lac/0_15-->p2s_seg-617
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_0_15-640_640-64_256-rot_15_345_8-flip-sub_8-lac-617-batch_12-seq1k,_eval_,batch-4,save-vis-0,dbg-0,dyn-1,617,seg-32_49:p-640:sub-8,lac,seq1k,x99

<a id="0_2_3_"></a>
# 0_23
<a id="orig_p_640_sub_8_aug_lac_fbb___0_23_"></a>
## orig-p-640-sub-8-aug-lac-fbb       @ 0_23-->p2s_seg-617
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,617,seg-0_23:p-640:sub-8:rot-15_345_8:strd-64_256:flip,batch-24,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,lac,seq1k,fbb
<a id="on_32_49___orig_p_640_sub_8_aug_lac_fbb_0_2_3_"></a>
#### on-32_49       @ orig-p-640-sub-8-aug-lac-fbb/0_23-->p2s_seg-617
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_0_23-640_640-64_256-rot_15_345_8-flip-sub_8-lac-617-batch_24-seq1k-fbb,_eval_,batch-4,save-vis-0,dbg-0,dyn-1,617,seg-32_49:p-640:sub-8,lac,seq1k,grs

<a id="orig_p_640_sub_8_aug_lac___0_23_"></a>
## orig-p-640-sub-8-aug-lac       @ 0_23-->p2s_seg-617
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,617,seg-0_23:p-640:sub-8:rot-15_345_8:strd-64_256:flip,batch-12,dbg-0,dyn-1,dist-0,ep-10000,gz,pt-1,lac,seq1k
<a id="on_32_49___orig_p_640_sub_8_aug_lac_0_2_3_"></a>
#### on-32_49       @ orig-p-640-sub-8-aug-lac/0_23-->p2s_seg-617
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_0_23-640_640-64_256-rot_15_345_8-flip-sub_8-lac-617-batch_12-seq1k,_eval_,batch-4,save-vis-0,dbg-0,dyn-1,617,seg-32_49:p-640:sub-8,lac,seq1k,x99

<a id="0_3_1_"></a>
# 0_31
<a id="orig_p_640_sub_8_aug_lac___0_31_"></a>
## orig-p-640-sub-8-aug-lac       @ 0_31-->p2s_seg-617
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,617,seg-0_31:p-640:sub-8:rot-15_345_8:strd-64_256:flip,batch-28,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,lac,seq1k
<a id="on_train___orig_p_640_sub_8_aug_lac_0_3_1_"></a>
#### on-train       @ orig-p-640-sub-8-aug-lac/0_31-->p2s_seg-617
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_0_31-640_640-64_256-rot_15_345_8-flip-sub_8-lac-617-batch_28-seq1k,_eval_,batch-16,save-vis-0,dbg-0,dyn-1,617,seg-0_31:p-640:sub-8,lac,seq1k
<a id="on_32_49___orig_p_640_sub_8_aug_lac_0_3_1_"></a>
#### on-32_49       @ orig-p-640-sub-8-aug-lac/0_31-->p2s_seg-617
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_0_31-640_640-64_256-rot_15_345_8-flip-sub_8-lac-617-batch_28-seq1k,_eval_,batch-16,save-vis-0,dbg-0,dyn-1,617,seg-32_49:p-640:sub-8,lac,seq1k

<a id="orig_p_640_sub_8_aug_lac_fbb___0_31_"></a>
## orig-p-640-sub-8-aug-lac-fbb       @ 0_31-->p2s_seg-617
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,617,seg-0_31:p-640:sub-8:rot-15_345_8:strd-64_256:flip,batch-32,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,lac,seq1k,fbb
<a id="on_32_49___orig_p_640_sub_8_aug_lac_fbb_0_3_1_"></a>
#### on-32_49       @ orig-p-640-sub-8-aug-lac-fbb/0_31-->p2s_seg-617
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_0_31-640_640-64_256-rot_15_345_8-flip-sub_8-lac-617-batch_32-seq1k-fbb,_eval_,batch-4,save-vis-0,dbg-0,dyn-1,617,seg-32_49:p-640:sub-8,lac,seq1k,grs

<a id="r_640___0_31_"></a>
## r-640       @ 0_31-->p2s_seg-617
<a id="p_640_sub_4_aug_lac___r_640_0_31_"></a>
### p-640-sub-4-aug-lac       @ r-640/0_31-->p2s_seg-617
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,617,seg-0_31:r-640:p-640:sub-4:rot-15_345_16:strd-64_256:flip,batch-6,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,lac,seq3k,voc28
`dbg`
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,617,seg-0_31:r-640:p-640:sub-4:rot-15_345_16:strd-64_256:flip,batch-2,dbg-1,dyn-1,dist-0,ep-10000,gz,pt-1,lac,seq3k,voc28
<a id="on_train___p_640_sub_4_aug_lac_r_640_0_31_"></a>
#### on-train       @ p-640-sub-4-aug-lac/r-640/0_31-->p2s_seg-617
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_640-0_31-640_640-64_256-rot_15_345_16-flip-sub_4-lac-617-batch_6-seq3k,_eval_,batch-8,save-vis-0,dbg-0,dyn-1,617,seg-0_31:r-640:p-640:sub-4,lac,seq3k,voc28
<a id="on_32_49___p_640_sub_4_aug_lac_r_640_0_31_"></a>
#### on-32_49       @ p-640-sub-4-aug-lac/r-640/0_31-->p2s_seg-617
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_640-0_31-640_640-64_256-rot_15_345_16-flip-sub_4-lac-617-batch_6-seq3k,_eval_,batch-8,save-vis-0,dbg-0,dyn-1,617,seg-32_49:r-640:p-640:sub-4,lac,seq3k,voc28

<a id="p_640_sub_4_aug_lac_fbb___r_640_0_31_"></a>
### p-640-sub-4-aug-lac-fbb       @ r-640/0_31-->p2s_seg-617
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,617,seg-0_31:r-640:p-640:sub-4:rot-15_345_16:strd-64_256:flip,batch-8,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,lac,seq3k,voc28,fbb
<a id="on_32_49___p_640_sub_4_aug_lac_fbb_r_640_0_31_"></a>
#### on-32_49       @ p-640-sub-4-aug-lac-fbb/r-640/0_31-->p2s_seg-617
python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_640-0_31-640_640-64_256-rot_15_345_16-flip-sub_4-lac-617-batch_8-seq3k-fbb,_eval_,batch-2,save-vis-0,dbg-0,dyn-1,617,seg-32_49:r-640:p-640:sub-4,lac,seq3k,voc28,e5g

<a id="r_1280___0_31_"></a>
## r-1280       @ 0_31-->p2s_seg-617
<a id="p_1024_sub_8_aug_lac___r_1280_0_3_1_"></a>
### p-1024-sub-8-aug-lac       @ r-1280/0_31-->p2s_seg-617
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-1024,617,seg-0_31:r-1280:p-1024:sub-8:rot-15_345_16:strd-64_256:flip,batch-4,dbg-0,dyn-1,dist-1,ep-10000,gz,pt-1,lac,seq3k,voc18
<a id="on_32_49___p_1024_sub_8_aug_lac_r_1280_0_31_"></a>
#### on-32_49       @ p-1024-sub-8-aug-lac/r-1280/0_31-->p2s_seg-617
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_1024_resize_1280-0_31-1024_1024-64_256-rot_15_345_16-flip-sub_8-lac-617-batch_4-seq3k,_eval_,batch-1,save-vis-0,dbg-0,dyn-1,617,seg-32_49:r-1280:p-1024:sub-8,lac,seq3k,voc28,x99

