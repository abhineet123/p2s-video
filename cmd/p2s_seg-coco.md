<!-- MarkdownTOC -->

- [val-rfm](#val_rfm_)
    - [r-640-sub-8-mc-fbb       @ val-rfm](#r_640_sub_8_mc_fbb___val_rf_m_)
- [train-rfm](#train_rfm_)
        - [sub-8-2d-lac-cls_eq-zedg-fbb       @ train-rfm/](#sub_8_2d_lac_cls_eq_zedg_fbb___train_rfm_)
    - [r-640-sub-8-lac-fbb-aug       @ train-rfm](#r_640_sub_8_lac_fbb_aug___train_rf_m_)
            - [on-val       @ r-640-sub-8-lac-fbb-aug/train-rfm](#on_val___r_640_sub_8_lac_fbb_aug_train_rf_m_)
    - [r-640-sub-8-lac-fbb-aug-dz       @ train-rfm](#r_640_sub_8_lac_fbb_aug_dz___train_rf_m_)
- [train-rfm](#train_rfm__1)
    - [r-1280-p-640-sub-8-lac-fbb       @ train-rfm](#r_1280_p_640_sub_8_lac_fbb___train_rf_m_)
        - [eager       @ r-1280-p-640-sub-8-lac-fbb/train-rfm](#eager___r_1280_p_640_sub_8_lac_fbb_train_rfm_)
    - [r-640-p-640-sub-8-lac-fbb       @ train-rfm](#r_640_p_640_sub_8_lac_fbb___train_rf_m_)
- [val](#val_)
    - [r-640-p-640-sub-8-lac-fbb       @ val](#r_640_p_640_sub_8_lac_fbb___va_l_)
    - [r-640-p-640-sub-8-lac-fbb-isc       @ val](#r_640_p_640_sub_8_lac_fbb_isc___va_l_)
            - [on-val       @ r-640-p-640-sub-8-lac-fbb-isc/val](#on_val___r_640_p_640_sub_8_lac_fbb_isc_va_l_)
- [train](#train_)
    - [r-640-p-640-sub-8-lac-fbb       @ train](#r_640_p_640_sub_8_lac_fbb___trai_n_)
            - [on-train-end-2000       @ r-640-p-640-sub-8-lac-fbb/train](#on_train_end_2000___r_640_p_640_sub_8_lac_fbb_trai_n_)
            - [on-val       @ r-640-p-640-sub-8-lac-fbb/train](#on_val___r_640_p_640_sub_8_lac_fbb_trai_n_)
                - [p9-oldest_first       @ on-val/r-640-p-640-sub-8-lac-fbb/train](#p9_oldest_first___on_val_r_640_p_640_sub_8_lac_fbb_train_)
    - [r-1280-p-640-sub-8-lac-fbb       @ train](#r_1280_p_640_sub_8_lac_fbb___trai_n_)
    - [r-640-p-640-sub-5-bac-lac-fbb       @ train](#r_640_p_640_sub_5_bac_lac_fbb___trai_n_)
    - [r-1280-p-640-sub-4-lac-fbb       @ train](#r_1280_p_640_sub_4_lac_fbb___trai_n_)
    - [r-1280-p-640-sub-4-lac-end-100-fbb       @ train](#r_1280_p_640_sub_4_lac_end_100_fbb___trai_n_)
    - [r-1280-p-640-sub-4-mc-fbb       @ train](#r_1280_p_640_sub_4_mc_fbb___trai_n_)
            - [on-train-end-600       @ r-1280-p-640-sub-4-mc-fbb/train](#on_train_end_600___r_1280_p_640_sub_4_mc_fbb_trai_n_)
            - [on-val-end-500       @ r-1280-p-640-sub-4-mc-fbb/train](#on_val_end_500___r_1280_p_640_sub_4_mc_fbb_trai_n_)
            - [on-val       @ r-1280-p-640-sub-4-mc-fbb/train](#on_val___r_1280_p_640_sub_4_mc_fbb_trai_n_)

<!-- /MarkdownTOC -->

<a id="val_rfm_"></a>
# val-rfm

<a id="r_640_sub_8_mc_fbb___val_rf_m_"></a>
## r-640-sub-8-mc-fbb       @ val-rfm-->p2s_seg-coco
CUDA_VISIBLE_DEVICES=0,1 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco_rfm_val,seg-r-640:p-640:sub-8,rfm,batch-8,res-640,dbg-0,dyn-1,dist-1,ep-10000,pt-1,mc,seq3k,fbb,voc18
```
log/seg/resnet_640_semantic_val2017-coco_semantic_val2017-batch_8-res_640-seq2k-fbb-voc18
```

`dbg-1`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco_rfm_val,seg-sub-8,rfm,batch-2,res-640,dbg-1,dyn-1,dist-0,ep-10000,pt-1,lac,seq2k,fbb,voc18

<a id="train_rfm_"></a>
# train-rfm
<a id="sub_8_2d_lac_cls_eq_zedg_fbb___train_rfm_"></a>
### sub-8-2d-lac-cls_eq-zedg-fbb       @ train-rfm/-->p2s_seg-coco
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,seg-54_126:r-640:p-640:sub-4,seq3k,voc500,cvs-360,coco_rfm,rot,flip,batch-128,pt-1,2d,lac,mc,cls_eq-2,fbb,dbg-0,dyn-1,dist-2,zedg


<a id="r_640_sub_8_lac_fbb_aug___train_rf_m_"></a>
## r-640-sub-8-lac-fbb-aug       @ train-rfm-->p2s_seg-coco
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco_rfm_val,seg-sub-8,rfm,batch-4,res-640,rot,crop,flip,dbg-0,dyn-1,dist-0,ep-10000,pt-1,lac,seq2k,fbb,voc18
`dbg-1`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco_rfm_val,seg-sub-8,rfm,batch-2,res-640,rot,crop,flip,dbg-1,dyn-1,dist-0,ep-10000,pt-1,lac,seq2k,fbb,voc18
`dbg-2`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco_rfm_val,seg-sub-8,rfm,batch-2,res-640,flip-0,crop-0,dbg-2,dyn-1,dist-0,ep-10000,pt-1,lac,seq2k,fbb,voc18

watch tail -1 
log/seg/resnet_640_semantic_val2017-coco_semantic_val2017-rfm-batch_4-res_640-rot-crop-flip-seq2k-fbb-voc18
/progress_log.txt

<a id="on_val___r_640_sub_8_lac_fbb_aug_train_rf_m_"></a>
#### on-val       @ r-640-sub-8-lac-fbb-aug/train-rfm-->p2s_seg-coco
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_semantic_val2017-coco_semantic_val2017-rfm-batch_4-res_640-rot-crop-flip-seq2k-fbb-voc18,_eval_,coco_rfm_val,batch-32,save-vis-0,dbg-0,dyn-1,seg-sub-8,rfm,lac,seq2k,voc18,no_vid,d3

<a id="r_640_sub_8_lac_fbb_aug_dz___train_rf_m_"></a>
## r-640-sub-8-lac-fbb-aug-dz       @ train-rfm-->p2s_seg-coco
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco_rfm_val,seg-sub-8,rfm,batch-8,res-640,rot,crop,flip,dbg-0,dyn-1,dist-2,ep-10000,pt-1,lac,seq2k,fbb,voc18,dz

CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco_rfm_val,seg-sub-8,rfm,batch-8,res-640,rot,crop,flip,dbg-0,dyn-1,dist-2,ep-10000,pt-1,lac,seq2k,fbb,voc18,dz

watch date -r 
log/seg/resnet_640_semantic_val2017-coco_semantic_val2017-rfm-batch_8-res_640-rot-crop-flip-seq2k-fbb-voc18-dz
/progress_log.txt -u +"%Y-%m-%d.%H-%M-%S.%3N"
<a id="train_rfm__1"></a>
# train-rfm
<a id="r_1280_p_640_sub_8_lac_fbb___train_rf_m_"></a>
## r-1280-p-640-sub-8-lac-fbb       @ train-rfm-->p2s_seg-coco
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco_rfm,seg-sub-8,rfm,batch-12,res-1280,rot,crop,flip,dbg-0,dyn-1,dist-0,ep-10000,pt-1,lac,seq2k,fbb,voc18
<a id="eager___r_1280_p_640_sub_8_lac_fbb_train_rfm_"></a>
### eager       @ r-1280-p-640-sub-8-lac-fbb/train-rfm-->p2s_seg-coco
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco_rfm,seg-sub-8,rfm,batch-12,res-1280,rot,crop,flip,dbg-1,dyn-1,dist-0,ep-10000,pt-1,lac,seq2k,fbb,eager,voc18
`dbg`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco_rfm,seg-sub-8,rfm,batch-12,res-1280,rot,crop,flip,dbg-0,dyn-1,dist-0,ep-10000,pt-1,lac,seq2k,fbb,eager,voc18
<a id="r_640_p_640_sub_8_lac_fbb___train_rf_m_"></a>
## r-640-p-640-sub-8-lac-fbb       @ train-rfm-->p2s_seg-coco
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco_rfm,seg-sub-8,rfm,batch-12,res-640,rot,flip,dbg-1,dyn-1,dist-0,ep-10000,pt-1,lac,seq2k,fbb,eager,voc18
`dbg`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco_rfm,seg-sub-8,rfm,batch-12,res-640,rot,flip,dbg-1,dyn-1,dist-0,ep-10000,pt-1,lac,seq2k,fbb,eager,voc18


<a id="val_"></a>
# val
<a id="r_640_p_640_sub_8_lac_fbb___va_l_"></a>
## r-640-p-640-sub-8-lac-fbb       @ val-->p2s_seg-coco
CUDA_VISIBLE_DEVICES=0,1 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco_val,seg-r-640:p-640:sub-8,batch-8,dbg-0,dyn-1,dist-1,ep-10000,pt-1,lac,seq2k,fbb,voc18

watch tail -1 
log/seg/resnet_640_resize_640-0_4999-640_640-640_640-sub_8-lac-coco_semantic_val2017-batch_8-seq2k-fbb-voc18
/progress_log.txt

<a id="r_640_p_640_sub_8_lac_fbb_isc___va_l_"></a>
## r-640-p-640-sub-8-lac-fbb-isc       @ val-->p2s_seg-coco
CUDA_VISIBLE_DEVICES=0 NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME==enp3s0 NCCL_P2P_DISABLE=1 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco_val,seg-r-640:p-640:sub-8,batch-64,dbg-0,dyn-1,dist-2,ep-10000,pt-1,lac,seq2k,fbb,voc18,self2-0
CUDA_VISIBLE_DEVICES=1 NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME==enp3s0 NCCL_P2P_DISABLE=1 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco_val,seg-r-640:p-640:sub-8,batch-64,dbg-0,dyn-1,dist-2,ep-10000,pt-1,lac,seq2k,fbb,voc18,self2-1
`dbg`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco_val,seg-r-640:p-640:sub-8,batch-28,dbg-0,dyn-1,dist-0,ep-10000,pt-1,lac,seq2k,fbb,voc18
<a id="on_val___r_640_p_640_sub_8_lac_fbb_isc_va_l_"></a>
#### on-val       @ r-640-p-640-sub-8-lac-fbb-isc/val-->p2s_seg-coco
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_640-0_4999-640_640-640_640-sub_8-lac-coco_semantic_val2017-batch_64-seq2k-fbb-voc18-self2-1,_eval_,coco_val,batch-8,save-vis-0,dbg-0,dyn-1,seg-p-640:r-640:sub-8,lac,seq2k,voc18,isc




<a id="train_"></a>
# train
<a id="r_640_p_640_sub_8_lac_fbb___trai_n_"></a>
## r-640-p-640-sub-8-lac-fbb       @ train-->p2s_seg-coco
CUDA_VISIBLE_DEVICES=0 NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME==enp3s0 NCCL_P2P_DISABLE=1 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco,seg-r-640:p-640:sub-8,batch-64,dbg-0,dyn-1,dist-2,ep-10000,pt-1,lac,seq2k,fbb,voc18,self2-0
CUDA_VISIBLE_DEVICES=1 NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME==enp3s0 NCCL_P2P_DISABLE=1 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco,seg-r-640:p-640:sub-8,batch-64,dbg-0,dyn-1,dist-2,ep-10000,pt-1,lac,seq2k,fbb,voc18,self2-1
`dbg`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco,seg-r-640:p-640:sub-8,batch-32,dbg-0,dyn-1,dist-0,ep-10000,pt-1,lac,seq2k,voc18,fbb
<a id="on_train_end_2000___r_640_p_640_sub_8_lac_fbb_trai_n_"></a>
#### on-train-end-2000       @ r-640-p-640-sub-8-lac-fbb/train-->p2s_seg-coco
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_640-0_118286-640_640-640_640-sub_8-lac-coco_semantic_train2017-batch_64-seq2k-fbb-voc18-self2-0,_eval_,coco,batch-16,save-vis-0,dbg-0,dyn-1,seg-p-640:r-640:sub-8:end-2000,lac,seq2k,voc18,isc
<a id="on_val___r_640_p_640_sub_8_lac_fbb_trai_n_"></a>
#### on-val       @ r-640-p-640-sub-8-lac-fbb/train-->p2s_seg-coco
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_640-0_118286-640_640-640_640-sub_8-lac-coco_semantic_train2017-batch_64-seq2k-fbb-voc18-self2-0,_eval_,coco_val,batch-32,save-vis-0,dbg-0,dyn-1,seg-p-640:r-640:sub-8,lac,seq2k,voc18,isc
<a id="p9_oldest_first___on_val_r_640_p_640_sub_8_lac_fbb_train_"></a>
##### p9-oldest_first       @ on-val/r-640-p-640-sub-8-lac-fbb/train-->p2s_seg-coco
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_640-0_118286-640_640-640_640-sub_8-lac-coco_semantic_train2017-batch_64-seq2k-fbb-voc18-self2-0,_eval_,coco_val,batch-16,save-vis-0,dbg-0,dyn-1,seg-p-640:r-640:sub-8,lac,seq2k,voc18,p9,remote_only,oldest_first

<a id="r_1280_p_640_sub_8_lac_fbb___trai_n_"></a>
## r-1280-p-640-sub-8-lac-fbb       @ train-->p2s_seg-coco
CUDA_VISIBLE_DEVICES=0 NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME==enp3s0 NCCL_P2P_DISABLE=1 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco,seg-r-640:p-640:sub-8,batch-64,dbg-0,dyn-1,dist-2,ep-10000,pt-1,lac,seq2k,fbb,voc18,self2-0
CUDA_VISIBLE_DEVICES=1 NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME==enp3s0 NCCL_P2P_DISABLE=1 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco,seg-r-640:p-640:sub-8,batch-64,dbg-0,dyn-1,dist-2,ep-10000,pt-1,lac,seq2k,fbb,voc18,self2-1
`dbg`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco,seg-r-1280:p-640:sub-8,batch-2,dbg-0,dyn-1,dist-0,ep-10000,pt-1,lac,seq2k,voc18,fbb

<a id="r_640_p_640_sub_5_bac_lac_fbb___trai_n_"></a>
## r-640-p-640-sub-5-bac-lac-fbb       @ train-->p2s_seg-coco
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco,seg-r-640:p-640:sub-5,batch-64,dbg-0,dyn-1,dist-2,ep-10000,pt-1,bac,lac,seq2k,fbb,voc17b,zedg
`dbg`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco,seg-r-640:p-640:sub-5,batch-8,dbg-0,dyn-1,dist-0,ep-10000,pt-1,bac,lac,seq2k,fbb,voc17b

<a id="r_1280_p_640_sub_4_lac_fbb___trai_n_"></a>
## r-1280-p-640-sub-4-lac-fbb       @ train-->p2s_seg-coco
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco,seg-r-1280:p-640:sub-4,batch-24,dbg-0,dyn-1,dist-2,ep-10000,pt-1,lac,seq3k,voc48,fbb,zedg
`dbg`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco,seg-r-1280:p-640:sub-4,batch-3,dbg-0,dyn-1,dist-0,ep-10000,pt-1,lac,seq3k,voc48,fbb

<a id="r_1280_p_640_sub_4_lac_end_100_fbb___trai_n_"></a>
## r-1280-p-640-sub-4-lac-end-100-fbb       @ train-->p2s_seg-coco
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco,seg-r-1280:p-640:sub-4:end-100,batch-32,dbg-0,dyn-1,dist-2,ep-10000,pt-1,lac,seq3k,voc48,fbb,zexg
`dbg`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco,seg-r-1280:p-640:sub-4:end-100,batch-2,dbg-0,dyn-1,dist-0,ep-10000,pt-1,lac,seq3k,voc48,fbb

<a id="r_1280_p_640_sub_4_mc_fbb___trai_n_"></a>
## r-1280-p-640-sub-4-mc-fbb       @ train-->p2s_seg-coco
python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco,seg-r-1280:p-640:sub-4,batch-24,dbg-0,dyn-1,dist-2,ep-10000,pt-1,mc,seq4k,voc28,fbb,zedg
`dbg`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-640,coco,seg-r-1280:p-640:sub-4:end-100,batch-3,dbg-0,dyn-1,dist-0,ep-10000,pt-1,mc,seq4k,voc28,fbb
<a id="on_train_end_600___r_1280_p_640_sub_4_mc_fbb_trai_n_"></a>
#### on-train-end-600       @ r-1280-p-640-sub-4-mc-fbb/train-->p2s_seg-coco
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_1280-0_118286-640_640-640_640-sub_4-mc-coco_semantic_train2017-batch_24-seq4k-voc28-fbb-zedg,_eval_,coco,batch-8,save-vis-0,dbg-0,dyn-1,seg-p-640:r-1280:sub-4:end-600,mc,seq4k,voc28,grs
<a id="on_val_end_500___r_1280_p_640_sub_4_mc_fbb_trai_n_"></a>
#### on-val-end-500       @ r-1280-p-640-sub-4-mc-fbb/train-->p2s_seg-coco
CUDA_VISIBLE_DEVICES=2 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_1280-0_118286-640_640-640_640-sub_4-mc-coco_semantic_train2017-batch_24-seq4k-voc28-fbb-zedg,_eval_,coco_val,batch-8,save-vis-0,dbg-0,dyn-1,seg-p-640:r-1280:sub-4:end-500,mc,seq4k,voc28,grs
<a id="on_val___r_1280_p_640_sub_4_mc_fbb_trai_n_"></a>
#### on-val       @ r-1280-p-640-sub-4-mc-fbb/train-->p2s_seg-coco
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_resize_1280-0_118286-640_640-640_640-sub_4-mc-coco_semantic_train2017-batch_24-seq4k-voc28-fbb-zedg,_eval_,coco_val,batch-16,save-vis-0,dbg-0,dyn-1,seg-p-640:r-1280:sub-4,mc,seq4k,voc28,iter-413994

