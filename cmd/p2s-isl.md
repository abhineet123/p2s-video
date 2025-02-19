<!-- MarkdownTOC -->

- [resnet-640](#resnet_64_0_)
    - [detrac-0_19       @ resnet-640](#detrac_0_19___resnet_640_)
        - [on-train       @ detrac-0_19/resnet-640](#on_train___detrac_0_19_resnet_640_)
        - [on-49_68       @ detrac-0_19/resnet-640](#on_49_68___detrac_0_19_resnet_640_)
    - [detrac-0_9       @ resnet-640](#detrac_0_9___resnet_640_)
    - [detrac-0_48       @ resnet-640](#detrac_0_48___resnet_640_)
        - [on-49_85       @ detrac-0_48/resnet-640](#on_49_85___detrac_0_48_resnet_640_)
        - [on-49_85-100_per_seq_random       @ detrac-0_48/resnet-640](#on_49_85_100_per_seq_random___detrac_0_48_resnet_640_)

<!-- /MarkdownTOC -->
<a id="resnet_64_0_"></a>
# resnet-640 
<a id="detrac_0_19___resnet_640_"></a>
## detrac-0_19       @ resnet-640-->p2s-isl
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,detrac-non_empty-0_19,batch-3,dbg-0,dyn-1,dist-0
<a id="on_train___detrac_0_19_resnet_640_"></a>
### on-train       @ detrac-0_19/resnet-640-->p2s-isl
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=eval,m-resnet_640_detrac-non_empty-seq-0_19-batch_18,detrac-non_empty-0_19,batch-48,save-vis-1,dbg-0,dyn-1,dist-0
<a id="on_49_68___detrac_0_19_resnet_640_"></a>
### on-49_68       @ detrac-0_19/resnet-640-->p2s-isl
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-resnet_640_detrac-non_empty-seq-0_19-batch_18,detrac-non_empty-49_68,batch-16,save-vis-1,dbg-0,dyn-1,dist-0

<a id="detrac_0_9___resnet_640_"></a>
## detrac-0_9       @ resnet-640-->p2s-isl
python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,detrac-non_empty-0_9,batch-18,dbg-0,dyn-1,dist-0

<a id="detrac_0_48___resnet_640_"></a>
## detrac-0_48       @ resnet-640-->p2s-isl
python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,detrac-non_empty-0_48,batch-60,dbg-0,dyn-1,dist-1,fbb
<a id="on_49_85___detrac_0_48_resnet_640_"></a>
### on-49_85       @ detrac-0_48/resnet-640-->p2s-isl
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-resnet_640_detrac-non_empty-seq-0_48-batch_60-fbb,detrac-non_empty-49_85,batch-2,save-vis-0,dbg-0,dyn-1,dist-0,asi-0,grs,iter-180400
<a id="on_49_85_100_per_seq_random___detrac_0_48_resnet_640_"></a>
### on-49_85-100_per_seq_random       @ detrac-0_48/resnet-640-->p2s-isl
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-resnet_640_detrac-non_empty-seq-0_48-batch_60-fbb,detrac-non_empty-100_per_seq_random-49_85,batch-8,save-vis-0,dbg-0,dyn-1,dist-0,asi-0,grs

