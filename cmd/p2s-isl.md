<!-- MarkdownTOC -->

- [resnet-640](#resnet_64_0_)
    - [detrac-non_empty-0_19       @ resnet-640](#detrac_non_empty_0_19___resnet_640_)
        - [on-train       @ detrac-non_empty-0_19/resnet-640](#on_train___detrac_non_empty_0_19_resnet_640_)
        - [on-49_68       @ detrac-non_empty-0_19/resnet-640](#on_49_68___detrac_non_empty_0_19_resnet_640_)
    - [detrac-non_empty-0_9       @ resnet-640](#detrac_non_empty_0_9___resnet_640_)
    - [detrac-non_empty-0_48       @ resnet-640](#detrac_non_empty_0_48___resnet_640_)
        - [on-49_85       @ detrac-non_empty-0_48/resnet-640](#on_49_85___detrac_non_empty_0_48_resnet_640_)
        - [on-49_85-100_per_seq_random       @ detrac-non_empty-0_48/resnet-640](#on_49_85_100_per_seq_random___detrac_non_empty_0_48_resnet_640_)
    - [detrac-0_59-aug-fbb       @ resnet-640](#detrac_0_59_aug_fbb___resnet_640_)
        - [on-0_59-40_per_seq_random       @ detrac-0_59-aug-fbb/resnet-640](#on_0_59_40_per_seq_random___detrac_0_59_aug_fbb_resnet_640_)
        - [on-60_99-40_per_seq_random       @ detrac-0_59-aug-fbb/resnet-640](#on_60_99_40_per_seq_random___detrac_0_59_aug_fbb_resnet_640_)
        - [on-60_99       @ detrac-0_59-aug-fbb/resnet-640](#on_60_99___detrac_0_59_aug_fbb_resnet_640_)
            - [43068       @ on-60_99/detrac-0_59-aug-fbb/resnet-640](#43068___on_60_99_detrac_0_59_aug_fbb_resnet_64_0_)
            - [46560       @ on-60_99/detrac-0_59-aug-fbb/resnet-640](#46560___on_60_99_detrac_0_59_aug_fbb_resnet_64_0_)

<!-- /MarkdownTOC -->
<a id="resnet_64_0_"></a>
# resnet-640 

<a id="detrac_non_empty_0_19___resnet_640_"></a>
## detrac-non_empty-0_19       @ resnet-640-->p2s-isl
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,detrac-non_empty-0_19,batch-3,dbg-0,dyn-1,dist-0
<a id="on_train___detrac_non_empty_0_19_resnet_640_"></a>
### on-train       @ detrac-non_empty-0_19/resnet-640-->p2s-isl
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=eval,m-resnet_640_detrac-non_empty-seq-0_19-batch_18,detrac-non_empty-0_19,batch-48,save-vis-1,dbg-0,dyn-1,dist-0
<a id="on_49_68___detrac_non_empty_0_19_resnet_640_"></a>
### on-49_68       @ detrac-non_empty-0_19/resnet-640-->p2s-isl
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-resnet_640_detrac-non_empty-seq-0_19-batch_18,detrac-non_empty-49_68,batch-16,save-vis-1,dbg-0,dyn-1,dist-0

<a id="detrac_non_empty_0_9___resnet_640_"></a>
## detrac-non_empty-0_9       @ resnet-640-->p2s-isl
python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,detrac-non_empty-0_9,batch-18,dbg-0,dyn-1,dist-0

<a id="detrac_non_empty_0_48___resnet_640_"></a>
## detrac-non_empty-0_48       @ resnet-640-->p2s-isl
python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,detrac-non_empty-0_48,batch-60,dbg-0,dyn-1,dist-1,fbb
<a id="on_49_85___detrac_non_empty_0_48_resnet_640_"></a>
### on-49_85       @ detrac-non_empty-0_48/resnet-640-->p2s-isl
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-resnet_640_detrac-non_empty-seq-0_48-batch_60-fbb,detrac-non_empty-49_85,batch-2,save-vis-0,dbg-0,dyn-1,dist-0,asi-0,grs,iter-180400
<a id="on_49_85_100_per_seq_random___detrac_non_empty_0_48_resnet_640_"></a>
### on-49_85-100_per_seq_random       @ detrac-non_empty-0_48/resnet-640-->p2s-isl
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-resnet_640_detrac-non_empty-seq-0_48-batch_60-fbb,detrac-non_empty-100_per_seq_random-49_85,batch-8,save-vis-0,dbg-0,dyn-1,dist-0,asi-0,grs

<a id="detrac_0_59_aug_fbb___resnet_640_"></a>
## detrac-0_59-aug-fbb       @ resnet-640-->p2s-isl
python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,detrac-0_59,batch-288,dbg-0,dyn-1,dist-2,jtr,res-1280,fbb,exg
`dbg`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,detrac-0_59,batch-48,dbg-0,dyn-1,dist-0,jtr,res-1280,fbb
<a id="on_0_59_40_per_seq_random___detrac_0_59_aug_fbb_resnet_640_"></a>
### on-0_59-40_per_seq_random       @ detrac-0_59-aug-fbb/resnet-640-->p2s-isl
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-resnet_640_detrac-seq-0_59-batch_288-jtr-res_1280-fbb-exg,detrac-40_per_seq_random-0_59,batch-4,save-vis-0,dbg-0,dyn-1,dist-0,asi-0,grs
`dbg`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-resnet_640_detrac-seq-0_59-batch_288-jtr-res_1280-fbb-exg,detrac-100_per_seq_random-0_59,batch-2,save-vis-0,dbg-1,dyn-1,dist-0,asi-0,grs
<a id="on_60_99_40_per_seq_random___detrac_0_59_aug_fbb_resnet_640_"></a>
### on-60_99-40_per_seq_random       @ detrac-0_59-aug-fbb/resnet-640-->p2s-isl
CUDA_VISIBLE_DEVICES=2 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-resnet_640_detrac-seq-0_59-batch_288-jtr-res_1280-fbb-exg,detrac-40_per_seq_random-60_99,batch-4,save-vis-0,dbg-0,dyn-1,dist-0,asi-0,grs
<a id="on_60_99___detrac_0_59_aug_fbb_resnet_640_"></a>
### on-60_99       @ detrac-0_59-aug-fbb/resnet-640-->p2s-isl
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-resnet_640_detrac-seq-0_59-batch_288-jtr-res_1280-fbb-exg,detrac-60_99,batch-64,save-vis-0,dbg-0,dyn-1,dist-0,asi-1
<a id="43068___on_60_99_detrac_0_59_aug_fbb_resnet_64_0_"></a>
#### 43068       @ on-60_99/detrac-0_59-aug-fbb/resnet-640-->p2s-isl
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-resnet_640_detrac-seq-0_59-batch_288-jtr-res_1280-fbb-exg,detrac-60_99,batch-32,save-vis-0,dbg-0,dyn-1,dist-0,asi-0,iter-43068,p9
<a id="46560___on_60_99_detrac_0_59_aug_fbb_resnet_64_0_"></a>
#### 46560       @ on-60_99/detrac-0_59-aug-fbb/resnet-640-->p2s-isl
`latest`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-resnet_640_detrac-seq-0_59-batch_288-jtr-res_1280-fbb-exg,detrac-60_99,batch-32,save-vis-0,dbg-0,dyn-1,dist-0,asi-0,grs,iter-46560

