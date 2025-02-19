<!-- MarkdownTOC -->

- [resnet-640](#resnet_64_0_)
    - [acamp       @ resnet-640](#acamp___resnet_640_)
        - [1k8_vid_entire_seq       @ acamp/resnet-640](#1k8_vid_entire_seq___acamp_resnet_640_)
    - [10k6_vid_entire_seq       @ resnet-640](#10k6_vid_entire_seq___resnet_640_)
    - [detrac-non_empty       @ resnet-640](#detrac_non_empty___resnet_640_)
        - [0_19       @ detrac-non_empty/resnet-640](#0_19___detrac_non_empty_resnet_64_0_)
            - [on-train       @ 0_19/detrac-non_empty/resnet-640](#on_train___0_19_detrac_non_empty_resnet_640_)
            - [on-test       @ 0_19/detrac-non_empty/resnet-640](#on_test___0_19_detrac_non_empty_resnet_640_)
        - [0_9       @ detrac-non_empty/resnet-640](#0_9___detrac_non_empty_resnet_64_0_)
    - [mnist-640-1       @ resnet-640](#mnist_640_1___resnet_640_)
        - [frame-0-1       @ mnist-640-1/resnet-640](#frame_0_1___mnist_640_1_resnet_640_)
    - [mnist-640-5       @ resnet-640](#mnist_640_5___resnet_640_)
            - [on-test       @ mnist-640-5/resnet-640](#on_test___mnist_640_5_resnet_640_)
    - [0_1       @ resnet-640](#0_1___resnet_640_)
        - [val-2_3       @ 0_1/resnet-640](#val_2_3___0_1_resnet_640_)
    - [16_53       @ resnet-640](#16_53___resnet_640_)
        - [on-16_53       @ 16_53/resnet-640](#on_16_53___16_53_resnet_640_)
        - [on-54_126       @ 16_53/resnet-640](#on_54_126___16_53_resnet_640_)
    - [ipsc-0_37       @ resnet-640](#ipsc_0_37___resnet_640_)
        - [on-54_126       @ ipsc-0_37/resnet-640](#on_54_126___ipsc_0_37_resnet_640_)
    - [16_53-jtr-res-1280       @ resnet-640](#16_53_jtr_res_1280___resnet_640_)
        - [on-train       @ 16_53-jtr-res-1280/resnet-640](#on_train___16_53_jtr_res_1280_resnet_64_0_)
            - [acc       @ on-train/16_53-jtr-res-1280/resnet-640](#acc___on_train_16_53_jtr_res_1280_resnet_640_)
        - [on-54_126       @ 16_53-jtr-res-1280/resnet-640](#on_54_126___16_53_jtr_res_1280_resnet_64_0_)
            - [acc       @ on-54_126/16_53-jtr-res-1280/resnet-640](#acc___on_54_126_16_53_jtr_res_1280_resnet_64_0_)
    - [0_37-jtr-res-1280       @ resnet-640](#0_37_jtr_res_1280___resnet_640_)
        - [on-54_126       @ 0_37-jtr-res-1280/resnet-640](#on_54_126___0_37_jtr_res_1280_resnet_640_)
            - [acc       @ on-54_126/0_37-jtr-res-1280/resnet-640](#acc___on_54_126_0_37_jtr_res_1280_resnet_640_)
    - [16_53-buggy       @ resnet-640](#16_53_buggy___resnet_640_)
        - [batch-4-scratch       @ 16_53-buggy/resnet-640](#batch_4_scratch___16_53_buggy_resnet_640_)
            - [on-g2_0_15       @ batch-4-scratch/16_53-buggy/resnet-640](#on_g2_0_15___batch_4_scratch_16_53_buggy_resnet_640_)
            - [on-g2_54_126       @ batch-4-scratch/16_53-buggy/resnet-640](#on_g2_54_126___batch_4_scratch_16_53_buggy_resnet_640_)
        - [dist-1       @ 16_53-buggy/resnet-640](#dist_1___16_53_buggy_resnet_640_)
        - [local       @ 16_53-buggy/resnet-640](#local___16_53_buggy_resnet_640_)
        - [xe       @ 16_53-buggy/resnet-640](#xe___16_53_buggy_resnet_640_)
            - [on-g2_0_15       @ xe/16_53-buggy/resnet-640](#on_g2_0_15___xe_16_53_buggy_resnet_64_0_)
            - [on-g2_54_126       @ xe/16_53-buggy/resnet-640](#on_g2_54_126___xe_16_53_buggy_resnet_64_0_)
        - [gx       @ 16_53-buggy/resnet-640](#gx___16_53_buggy_resnet_640_)
        - [gxe       @ 16_53-buggy/resnet-640](#gxe___16_53_buggy_resnet_640_)
            - [on-g2_0_15       @ gxe/16_53-buggy/resnet-640](#on_g2_0_15___gxe_16_53_buggy_resnet_640_)
            - [on-g2_54_126       @ gxe/16_53-buggy/resnet-640](#on_g2_54_126___gxe_16_53_buggy_resnet_640_)
    - [0_37-buggy       @ resnet-640](#0_37_buggy___resnet_640_)
        - [batch_6       @ 0_37-buggy/resnet-640](#batch_6___0_37_buggy_resnet_64_0_)
        - [gxe       @ 0_37-buggy/resnet-640](#gxe___0_37_buggy_resnet_64_0_)
            - [on-g2_38_53       @ gxe/0_37-buggy/resnet-640](#on_g2_38_53___gxe_0_37_buggy_resnet_64_0_)
    - [pt       @ resnet-640](#pt___resnet_640_)
        - [on-mninstmot       @ pt/resnet-640](#on_mninstmot___pt_resnet_64_0_)
        - [on-g2_0_1       @ pt/resnet-640](#on_g2_0_1___pt_resnet_64_0_)
        - [on-g2_16_53       @ pt/resnet-640](#on_g2_16_53___pt_resnet_64_0_)
            - [dist       @ on-g2_16_53/pt/resnet-640](#dist___on_g2_16_53_pt_resnet_64_0_)
        - [on-g2_54_126       @ pt/resnet-640](#on_g2_54_126___pt_resnet_64_0_)
        - [on-g2_0_15       @ pt/resnet-640](#on_g2_0_15___pt_resnet_64_0_)
- [resnet-1333](#resnet_1333_)
    - [pt       @ resnet-1333](#pt___resnet_133_3_)
        - [on-g2_0_1       @ pt/resnet-1333](#on_g2_0_1___pt_resnet_1333_)
        - [on-g2_16_53       @ pt/resnet-1333](#on_g2_16_53___pt_resnet_1333_)
    - [g2_16_53       @ resnet-1333](#g2_16_53___resnet_133_3_)
- [resnet_c4-640](#resnet_c4_640_)
    - [pt       @ resnet_c4-640](#pt___resnet_c4_64_0_)
        - [on-g2_16_53       @ pt/resnet_c4-640](#on_g2_16_53___pt_resnet_c4_640_)
- [resnet_c4-1024](#resnet_c4_102_4_)
    - [g2_0_37       @ resnet_c4-1024](#g2_0_37___resnet_c4_1024_)
- [resnet_c4_1333](#resnet_c4_133_3_)
    - [pt       @ resnet_c4_1333](#pt___resnet_c4_1333_)
        - [on-g2_0_1       @ pt/resnet_c4_1333](#on_g2_0_1___pt_resnet_c4_133_3_)
        - [on-g2_16_53       @ pt/resnet_c4_1333](#on_g2_16_53___pt_resnet_c4_133_3_)
    - [g2_0_37       @ resnet_c4_1333](#g2_0_37___resnet_c4_1333_)

<!-- /MarkdownTOC -->
<a id="resnet_64_0_"></a>
# resnet-640 

<a id="acamp___resnet_640_"></a>
## acamp       @ resnet-640-->p2s
<a id="1k8_vid_entire_seq___acamp_resnet_640_"></a>
### 1k8_vid_entire_seq       @ acamp/resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,acamp-1k8_vid_entire_seq,batch-9,dbg-0,dyn-1,dist-0

<a id="10k6_vid_entire_seq___resnet_640_"></a>
## 10k6_vid_entire_seq       @ resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,acamp-10k6_vid_entire_seq,batch-9,dbg-0,dyn-1,dist-0


<a id="detrac_non_empty___resnet_640_"></a>
## detrac-non_empty       @ resnet-640-->p2s
<a id="0_19___detrac_non_empty_resnet_64_0_"></a>
### 0_19       @ detrac-non_empty/resnet-640-->p2s
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,detrac-non_empty-0_19,batch-3,dbg-0,dyn-1,dist-0
<a id="on_train___0_19_detrac_non_empty_resnet_640_"></a>
#### on-train       @ 0_19/detrac-non_empty/resnet-640-->p2s
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=eval,m-resnet_640_detrac-non_empty-seq-0_19-batch_18,detrac-non_empty-0_19,batch-48,save-vis-1,dbg-0,dyn-1,dist-0
<a id="on_test___0_19_detrac_non_empty_resnet_640_"></a>
#### on-test       @ 0_19/detrac-non_empty/resnet-640-->p2s
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=eval,m-resnet_640_detrac-non_empty-seq-0_19-batch_18,detrac-non_empty-49_68,batch-48,save-vis-1,dbg-0,dyn-1,dist-0

<a id="0_9___detrac_non_empty_resnet_64_0_"></a>
### 0_9       @ detrac-non_empty/resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,detrac-non_empty-0_9,batch-18,dbg-0,dyn-1,dist-0

<a id="mnist_640_1___resnet_640_"></a>
## mnist-640-1       @ resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,mnist-640-1-12_1000-train,batch-4,pt-1,dbg-1,dyn-1,dist-0
<a id="frame_0_1___mnist_640_1_resnet_640_"></a>
### frame-0-1       @ mnist-640-1/resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py --j5=val-1,mnist-640-1-12_1000-train,frame-0-1,batch-12,train,resnet-640,mnist-640-1-12_1000-train,frame-0-1,batch-6,dbg-1,dyn-1,dist-0,pt-0

<a id="mnist_640_5___resnet_640_"></a>
## mnist-640-5       @ resnet-640-->p2s
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,mnist-640-5-12_1000-train,batch-18,pt-1,dbg-0,dyn-1,dist-0
<a id="on_test___mnist_640_5_resnet_640_"></a>
#### on-test       @ mnist-640-5/resnet-640-->p2s
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=eval,m-resnet_640_mnist_640_5_12_1000_var-train-batch_18,mnist-640-5-12_1000-test,batch-96,save-vis-1,dbg-0,dyn-1,dist-0
`seq-0-5,frame-0-5`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=eval,m-resnet_640_mnist_640_5_12_1000_var-train-batch_18,mnist-640-5-12_1000-test,seq-0-5,frame-0-5,batch-3,save-vis-1,dbg-0,dyn-1,dist-0

<a id="0_1___resnet_640_"></a>
## 0_1       @ resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=val-1,ipsc-0_1,batch-6,train,resnet-640,ipsc-0_1,batch-6,dbg-0,dyn-1,dist-0,pt-1
<a id="val_2_3___0_1_resnet_640_"></a>
### val-2_3       @ 0_1/resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=val-1,ipsc-2_3,batch-18,train,resnet-640,ipsc-0_1,batch-18,dbg-0,dyn-1,dist-0,pt-1

python3 run.py --cfg=configs/config_det_ipsc.py  --j5=val-1,ipsc-2_3,batch-6,train,resnet-640,ipsc-0_1,batch-6,dbg-0,dyn-1,dist-0,pt-1,jtr,res-1280

<a id="16_53___resnet_640_"></a>
## 16_53       @ resnet-640-->p2s
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=val-1,ipsc-54_126_8,batch-24,train,resnet-640,ipsc-16_53,batch-18,dbg-0,dyn-1,dist-0,pt-1
<a id="on_16_53___16_53_resnet_640_"></a>
### on-16_53       @ 16_53/resnet-640-->p2s
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_ext_reorg_roi_g2-frame-16_53-batch_18,_eval_,ipsc-16_53,batch-36,save-vis-1,dbg-0,dyn-1
<a id="on_54_126___16_53_resnet_640_"></a>
### on-54_126       @ 16_53/resnet-640-->p2s
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_ext_reorg_roi_g2-frame-16_53-batch_18,_eval_,ipsc-54_126,batch-36,save-vis-1,dbg-0,dyn-1

<a id="ipsc_0_37___resnet_640_"></a>
## ipsc-0_37       @ resnet-640-->p2s
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=val-1,ipsc-54_126_8,batch-24,train,resnet-640,ipsc-0_37,batch-18,dbg-0,dyn-1,dist-0,pt-1
<a id="on_54_126___ipsc_0_37_resnet_640_"></a>
### on-54_126       @ ipsc-0_37/resnet-640-->p2s
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_ext_reorg_roi_g2-0_37-batch_18,_eval_,ipsc-54_126,batch-36,save-vis-1,dbg-0,dyn-1,dist-0

<a id="16_53_jtr_res_1280___resnet_640_"></a>
## 16_53-jtr-res-1280       @ resnet-640-->p2s
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=val-1,ipsc-54_126_8,batch-24,train,resnet-640,ipsc-16_53,batch-18,dbg-0,dyn-1,dist-0,pt-1,jtr,res-1280
<a id="on_train___16_53_jtr_res_1280_resnet_64_0_"></a>
### on-train       @ 16_53-jtr-res-1280/resnet-640-->p2s
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-batch_18-jtr-res_1280,_eval_,ipsc-16_53,batch-16,save-vis-1
<a id="acc___on_train_16_53_jtr_res_1280_resnet_640_"></a>
#### acc       @ on-train/16_53-jtr-res-1280/resnet-640-->p2s
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-batch_18-jtr-res_1280,_eval_,ipsc-16_53,batch-16,save-vis-1,acc

<a id="on_54_126___16_53_jtr_res_1280_resnet_64_0_"></a>
### on-54_126       @ 16_53-jtr-res-1280/resnet-640-->p2s
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-batch_18-jtr-res_1280,_eval_,ipsc-54_126,batch-16,save-vis-1
<a id="acc___on_54_126_16_53_jtr_res_1280_resnet_64_0_"></a>
#### acc       @ on-54_126/16_53-jtr-res-1280/resnet-640-->p2s
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-batch_18-jtr-res_1280,_eval_,ipsc-54_126,batch-16,save-vis-1,acc

<a id="0_37_jtr_res_1280___resnet_640_"></a>
## 0_37-jtr-res-1280       @ resnet-640-->p2s
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=val-1,ipsc-54_126_8,batch-24,train,resnet-640,ipsc-0_37,batch-18,dbg-0,dyn-1,dist-0,pt-1,jtr,res-1280
<a id="on_54_126___0_37_jtr_res_1280_resnet_640_"></a>
### on-54_126       @ 0_37-jtr-res-1280/resnet-640-->p2s
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_ext_reorg_roi_g2-0_37-batch_18-jtr-res_1280,_eval_,ipsc-54_126,batch-16,save-vis-1
<a id="acc___on_54_126_0_37_jtr_res_1280_resnet_640_"></a>
#### acc       @ on-54_126/0_37-jtr-res-1280/resnet-640-->p2s
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_ext_reorg_roi_g2-0_37-batch_18-jtr-res_1280,_eval_,ipsc-54_126,batch-16,save-vis-1,acc

<a id="16_53_buggy___resnet_640_"></a>
## 16_53-buggy       @ resnet-640-->p2s
<a id="batch_4_scratch___16_53_buggy_resnet_640_"></a>
### batch-4-scratch       @ 16_53-buggy/resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train,resnet-640,ipsc-16_53,batch-4,scratch,dbg-1,dyn-1
<a id="on_g2_0_15___batch_4_scratch_16_53_buggy_resnet_640_"></a>
#### on-g2_0_15       @ batch-4-scratch/16_53-buggy/resnet-640-->p2s
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,m-resnet_640_ext_reorg_roi_g2_16_53-batch_4-scratch,ipsc-0_15,batch-32,save-vis-1
<a id="on_g2_54_126___batch_4_scratch_16_53_buggy_resnet_640_"></a>
#### on-g2_54_126       @ batch-4-scratch/16_53-buggy/resnet-640-->p2s
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,m-resnet_640_ext_reorg_roi_g2_16_53-batch_4-scratch,ipsc-54_126,batch-32,save-vis-1

<a id="dist_1___16_53_buggy_resnet_640_"></a>
### dist-1       @ 16_53-buggy/resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train,resnet-640,ipsc-16_53,batch-12,dist-1,dbg-0,dyn-0

<a id="local___16_53_buggy_resnet_640_"></a>
### local       @ 16_53-buggy/resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train,resnet-640,ipsc-16_53,batch-4,dbg-0,dyn-1,local-0,gpu-0

python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train,resnet-640,ipsc-16_53,batch-4,dbg-0,dyn-1,local-1,gpu-1
<a id="xe___16_53_buggy_resnet_640_"></a>
### xe       @ 16_53-buggy/resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train,resnet-640,ipsc-16_53,batch-32,dbg-0,dyn-1,xe
<a id="on_g2_0_15___xe_16_53_buggy_resnet_64_0_"></a>
#### on-g2_0_15       @ xe/16_53-buggy/resnet-640-->p2s
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,m-resnet_640_ext_reorg_roi_g2_16_53-batch_32-xe,ipsc-0_15,batch-64,save-vis-1,dist-0
<a id="on_g2_54_126___xe_16_53_buggy_resnet_64_0_"></a>
#### on-g2_54_126       @ xe/16_53-buggy/resnet-640-->p2s
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,m-resnet_640_ext_reorg_roi_g2_16_53-batch_32-xe,ipsc-54_126,batch-64,save-vis-1,dist-0
<a id="gx___16_53_buggy_resnet_640_"></a>
### gx       @ 16_53-buggy/resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train,resnet-640,ipsc-16_53,batch-36,dbg-0,dyn-1,gx
<a id="gxe___16_53_buggy_resnet_640_"></a>
### gxe       @ 16_53-buggy/resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train,resnet-640,ipsc-16_53,ep-500,batch-48,dbg-0,dyn-0,gxe
<a id="on_g2_0_15___gxe_16_53_buggy_resnet_640_"></a>
#### on-g2_0_15       @ gxe/16_53-buggy/resnet-640-->p2s
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,m-resnet_640_ext_reorg_roi_g2_16_53-batch_48-gxe,ipsc-0_15,batch-32,save-vis-1,dist-1
<a id="on_g2_54_126___gxe_16_53_buggy_resnet_640_"></a>
#### on-g2_54_126       @ gxe/16_53-buggy/resnet-640-->p2s
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,m-resnet_640_ext_reorg_roi_g2_16_53-batch_48-gxe,ipsc-54_126,batch-32,save-vis-1,dist-1

<a id="0_37_buggy___resnet_640_"></a>
## 0_37-buggy       @ resnet-640-->p2s
<a id="batch_6___ipsc_0_37_resnet_640_"></a>
<a id="batch_6___0_37_buggy_resnet_64_0_"></a>
### batch_6       @ 0_37-buggy/resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train,resnet-640,ipsc-0_37,batch-8,dbg-1,dyn-1
<a id="gxe___0_37_buggy_resnet_64_0_"></a>
### gxe       @ 0_37-buggy/resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train,resnet-640,ipsc-0_37,batch-48,dbg-0,dyn-0,gxe,ep-4000
<a id="on_g2_38_53___gxe_0_37_buggy_resnet_64_0_"></a>
#### on-g2_38_53       @ gxe/0_37-buggy/resnet-640-->p2s
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,m-resnet_640_ext_reorg_roi_g2_0_37-batch_48-gxe,ipsc-38_53,batch-16,save-vis-1,conf-0

<a id="pt___resnet_640_"></a>
## pt       @ resnet-640-->p2s
<a id="on_mninstmot___pt_resnet_64_0_"></a>
### on-mninstmot       @ pt/resnet-640-->p2s
python3 run.py --mode=eval --model_dir=pretrained/resnet_640 --cfg=configs/config_det_mninstmot.py
<a id="on_g2_0_1___pt_resnet_64_0_"></a>
### on-g2_0_1       @ pt/resnet-640-->p2s
``batch-48``  
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,pt,resnet-640,ipsc-0_1,batch-48,save-vis-1,save-csv-1
``batch-2``  
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,pt,resnet-640,ipsc-0_1,batch-2,save-vis-1,save-csv-1
<a id="on_g2_16_53___pt_resnet_64_0_"></a>
### on-g2_16_53       @ pt/resnet-640-->p2s
``batch-64``  
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,pt,resnet-640,ipsc-16_53,batch-64,save-vis-1,save-csv-1
``batch-32``  
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,pt,resnet-640,ipsc-16_53,batch-32,save-vis-1,save-csv-1
<a id="dist___on_g2_16_53_pt_resnet_64_0_"></a>
#### dist       @ on-g2_16_53/pt/resnet-640-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,pt,resnet-640,ipsc-0_1,batch-2,save-vis-1,save-csv-1,dist-1,eager-0

<a id="on_g2_54_126___pt_resnet_64_0_"></a>
### on-g2_54_126       @ pt/resnet-640-->p2s
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,pt,resnet-640,ipsc-54_126,batch-32,save-vis-0,save-csv-1
<a id="on_g2_0_15___pt_resnet_64_0_"></a>
### on-g2_0_15       @ pt/resnet-640-->p2s
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,pt,resnet-640,ipsc-0_15,batch-32,save-vis-0,save-csv-1


<a id="resnet_1333_"></a>
# resnet-1333 
<a id="pt___resnet_133_3_"></a>
## pt       @ resnet-1333-->p2s
<a id="on_g2_0_1___pt_resnet_1333_"></a>
### on-g2_0_1       @ pt/resnet-1333-->p2s
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,pt,resnet-1333,ipsc-0_1,batch-48,save-vis-1,save-csv-1
<a id="on_g2_16_53___pt_resnet_1333_"></a>
### on-g2_16_53       @ pt/resnet-1333-->p2s
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,pt,resnet-1333,ipsc-16_53,batch-24,save-vis-1,save-csv-1

<a id="g2_16_53___resnet_133_3_"></a>
## g2_16_53       @ resnet-1333-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train,resnet-1333,ipsc-16_53,batch-2,dbg-1,dyn-1

<a id="resnet_c4_640_"></a>
# resnet_c4-640 
<a id="pt___resnet_c4_64_0_"></a>
## pt       @ resnet_c4-640-->p2s
<a id="on_g2_16_53___pt_resnet_c4_640_"></a>
### on-g2_16_53       @ pt/resnet_c4-640-->p2s
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,pt,resnet_c4-640,ipsc-16_53,batch-16,save-vis-1,save-csv-1

<a id="resnet_c4_102_4_"></a>
# resnet_c4-1024 
<a id="g2_0_37___resnet_c4_1024_"></a>
## g2_0_37       @ resnet_c4-1024-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train,resnet_c4-1024,ipsc-0_37,batch-6,dbg-0,dyn-0,gxe,ep-1000

<a id="resnet_c4_133_3_"></a>
# resnet_c4_1333 
<a id="pt___resnet_c4_1333_"></a>
## pt       @ resnet_c4_1333-->p2s
<a id="on_g2_0_1___pt_resnet_c4_133_3_"></a>
### on-g2_0_1       @ pt/resnet_c4_1333-->p2s
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,pt,resnet_c4-1333,ipsc-0_1,batch-1,save-vis-1,save-csv-1
<a id="on_g2_16_53___pt_resnet_c4_133_3_"></a>
### on-g2_16_53       @ pt/resnet_c4_1333-->p2s
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,pt,resnet_c4-1333,ipsc-16_53,batch-1,save-vis-1,save-csv-1

<a id="g2_0_37___resnet_c4_1333_"></a>
## g2_0_37       @ resnet_c4_1333-->p2s
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=train,resnet_c4-1333,ipsc-0_37,batch-6,dbg-0,dyn-0,gxe,ep-1000


