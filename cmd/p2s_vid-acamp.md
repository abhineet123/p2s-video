<!-- MarkdownTOC -->

- [mid       @ resnet-640](#mid___resnet_640_)
    - [1k8_vid_entire_seq       @ mid](#1k8_vid_entire_seq___mi_d_)
        - [on-train       @ 1k8_vid_entire_seq/mid](#on_train___1k8_vid_entire_seq_mid_)
        - [on-inv       @ 1k8_vid_entire_seq/mid](#on_inv___1k8_vid_entire_seq_mid_)
    - [10k6_vid_entire_seq       @ mid](#10k6_vid_entire_seq___mi_d_)
        - [on-train       @ 10k6_vid_entire_seq/mid](#on_train___10k6_vid_entire_seq_mi_d_)
        - [on-inv       @ 10k6_vid_entire_seq/mid](#on_inv___10k6_vid_entire_seq_mi_d_)
    - [1k8_vid_entire_seq-aug       @ mid](#1k8_vid_entire_seq_aug___mi_d_)
        - [on-train       @ 1k8_vid_entire_seq-aug/mid](#on_train___1k8_vid_entire_seq_aug_mid_)
        - [on-inv       @ 1k8_vid_entire_seq-aug/mid](#on_inv___1k8_vid_entire_seq_aug_mid_)
    - [10k6_vid_entire_seq-aug       @ mid](#10k6_vid_entire_seq_aug___mi_d_)
        - [on-train       @ 10k6_vid_entire_seq-aug/mid](#on_train___10k6_vid_entire_seq_aug_mi_d_)
        - [on-inv       @ 10k6_vid_entire_seq-aug/mid](#on_inv___10k6_vid_entire_seq_aug_mi_d_)
    - [20k6_5_video-aug       @ mid](#20k6_5_video_aug___mi_d_)
        - [on-train       @ 20k6_5_video-aug/mid](#on_train___20k6_5_video_aug_mid_)
        - [on-inv       @ 20k6_5_video-aug/mid](#on_inv___20k6_5_video_aug_mid_)
    - [1k8_vid_entire_seq-aug-fbb       @ mid](#1k8_vid_entire_seq_aug_fbb___mi_d_)
        - [on-inv       @ 1k8_vid_entire_seq-aug-fbb/mid](#on_inv___1k8_vid_entire_seq_aug_fbb_mid_)
        - [on-inv-2_per_seq       @ 1k8_vid_entire_seq-aug-fbb/mid](#on_inv_2_per_seq___1k8_vid_entire_seq_aug_fbb_mid_)
        - [on-inv-6_per_seq       @ 1k8_vid_entire_seq-aug-fbb/mid](#on_inv_6_per_seq___1k8_vid_entire_seq_aug_fbb_mid_)
        - [on-inv-12_per_seq       @ 1k8_vid_entire_seq-aug-fbb/mid](#on_inv_12_per_seq___1k8_vid_entire_seq_aug_fbb_mid_)
    - [10k6_vid_entire_seq-aug-fbb       @ mid](#10k6_vid_entire_seq_aug_fbb___mi_d_)
        - [on-inv       @ 10k6_vid_entire_seq-aug-fbb/mid](#on_inv___10k6_vid_entire_seq_aug_fbb_mi_d_)
        - [on-inv-2_per_seq       @ 10k6_vid_entire_seq-aug-fbb/mid](#on_inv_2_per_seq___10k6_vid_entire_seq_aug_fbb_mi_d_)
        - [on-inv-12_per_seq       @ 10k6_vid_entire_seq-aug-fbb/mid](#on_inv_12_per_seq___10k6_vid_entire_seq_aug_fbb_mi_d_)
    - [20k6_5_video-aug-fbb       @ mid](#20k6_5_video_aug_fbb___mi_d_)
        - [on-inv       @ 20k6_5_video-aug-fbb/mid](#on_inv___20k6_5_video_aug_fbb_mid_)
        - [on-inv-2_per_seq       @ 20k6_5_video-aug-fbb/mid](#on_inv_2_per_seq___20k6_5_video_aug_fbb_mid_)
        - [on-inv-12_per_seq       @ 20k6_5_video-aug-fbb/mid](#on_inv_12_per_seq___20k6_5_video_aug_fbb_mid_)
    - [1k8_vid_entire_seq-aug-cls_eq-fbb       @ mid](#1k8_vid_entire_seq_aug_cls_eq_fbb___mi_d_)
        - [on-inv-2_per_seq       @ 1k8_vid_entire_seq-aug-cls_eq-fbb/mid](#on_inv_2_per_seq___1k8_vid_entire_seq_aug_cls_eq_fbb_mi_d_)
    - [1k8_vid_entire_seq-aug-cls_eq-fbb-b64       @ mid](#1k8_vid_entire_seq_aug_cls_eq_fbb_b64___mi_d_)
        - [on-inv-2_per_seq       @ 1k8_vid_entire_seq-aug-cls_eq-fbb-b64/mid](#on_inv_2_per_seq___1k8_vid_entire_seq_aug_cls_eq_fbb_b64_mi_d_)
        - [on-inv       @ 1k8_vid_entire_seq-aug-cls_eq-fbb-b64/mid](#on_inv___1k8_vid_entire_seq_aug_cls_eq_fbb_b64_mi_d_)
    - [10k6_vid_entire_seq-aug-cls_eq-fbb       @ mid](#10k6_vid_entire_seq_aug_cls_eq_fbb___mi_d_)
        - [on-inv-2_per_seq       @ 10k6_vid_entire_seq-aug-cls_eq-fbb/mid](#on_inv_2_per_seq___10k6_vid_entire_seq_aug_cls_eq_fbb_mid_)
    - [10k6_vid_entire_seq-aug-cls_eq-fbb-b64       @ mid](#10k6_vid_entire_seq_aug_cls_eq_fbb_b64___mi_d_)
        - [on-inv-2_per_seq       @ 10k6_vid_entire_seq-aug-cls_eq-fbb-b64/mid](#on_inv_2_per_seq___10k6_vid_entire_seq_aug_cls_eq_fbb_b64_mid_)
        - [on-inv       @ 10k6_vid_entire_seq-aug-cls_eq-fbb-b64/mid](#on_inv___10k6_vid_entire_seq_aug_cls_eq_fbb_b64_mid_)
    - [10k6_vid_entire_seq-aug-cls_eq-fbb-b128       @ mid](#10k6_vid_entire_seq_aug_cls_eq_fbb_b128___mi_d_)
        - [on-inv-2_per_seq       @ 10k6_vid_entire_seq-aug-cls_eq-fbb-b128/mid](#on_inv_2_per_seq___10k6_vid_entire_seq_aug_cls_eq_fbb_b128_mi_d_)
    - [10k6_vid_entire_seq-aug-cls_eq       @ mid](#10k6_vid_entire_seq_aug_cls_eq___mi_d_)
        - [on-inv-2_per_seq       @ 10k6_vid_entire_seq-aug-cls_eq/mid](#on_inv_2_per_seq___10k6_vid_entire_seq_aug_cls_eq_mid_)
        - [on-inv       @ 10k6_vid_entire_seq-aug-cls_eq/mid](#on_inv___10k6_vid_entire_seq_aug_cls_eq_mid_)

<!-- /MarkdownTOC -->

<a id="mid___resnet_640_"></a>
# mid       @ resnet-640-->p2s_vid
<a id="1k8_vid_entire_seq___mi_d_"></a>
## 1k8_vid_entire_seq       @ mid-->p2s_vid-acamp
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,acamp-1k8_vid_entire_seq,batch-18,dbg-0,dyn-1,dist-1
<a id="on_train___1k8_vid_entire_seq_mid_"></a>
### on-train       @ 1k8_vid_entire_seq/mid-->p2s_vid-acamp
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_1k8_vid_entire_seq-length-2-stride-1-batch_18,_eval_,vid_det,acamp-1k8_vid_entire_seq,vstrd-1,batch-2,save-vis-0,dbg-0,dyn-1
<a id="on_inv___1k8_vid_entire_seq_mid_"></a>
### on-inv       @ 1k8_vid_entire_seq/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_1k8_vid_entire_seq-length-2-stride-1-batch_18,_eval_,vid_det,acamp-1k8_vid_entire_seq_inv,vstrd-1,batch-48,save-vis-0,dbg-0,dyn-1

<a id="10k6_vid_entire_seq___mi_d_"></a>
## 10k6_vid_entire_seq       @ mid-->p2s_vid-acamp
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,acamp-10k6_vid_entire_seq,batch-18,dbg-0,dyn-1,dist-1
<a id="on_train___10k6_vid_entire_seq_mi_d_"></a>
### on-train       @ 10k6_vid_entire_seq/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_10k610k6_vid_entire_seq_vid_entire_seq-length-2-stride-1-batch_18,_eval_,vid_det,acamp-10k6_vid_entire_seq,vstrd-1,batch-3,save-vis-0,dbg-0,dyn-1
<a id="on_inv___10k6_vid_entire_seq_mi_d_"></a>
### on-inv       @ 10k6_vid_entire_seq/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_10k6_vid_entire_seq-length-2-stride-1-batch_18,_eval_,vid_det,acamp-10k6_vid_entire_seq_inv,vstrd-1,batch-24,save-vis-0,dbg-0,dyn-1


<a id="1k8_vid_entire_seq_aug___mi_d_"></a>
## 1k8_vid_entire_seq-aug       @ mid-->p2s_vid-acamp
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,acamp-1k8_vid_entire_seq,batch-18,dbg-0,dyn-1,dist-1,jtr,res-1280
<a id="on_train___1k8_vid_entire_seq_aug_mid_"></a>
### on-train       @ 1k8_vid_entire_seq-aug/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_1k8_vid_entire_seq-length-2-stride-1-batch_18-jtr-res_1280,_eval_,vid_det,acamp-1k8_vid_entire_seq,vstrd-1,batch-1,save-vis-0,dbg-0,dyn-1
<a id="on_inv___1k8_vid_entire_seq_aug_mid_"></a>
### on-inv       @ 1k8_vid_entire_seq-aug/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_1k8_vid_entire_seq-length-2-stride-1-batch_18-jtr-res_1280,_eval_,vid_det,acamp-1k8_vid_entire_seq_inv,vstrd-1,batch-12,save-vis-0,dbg-0,dyn-1

<a id="10k6_vid_entire_seq_aug___mi_d_"></a>
## 10k6_vid_entire_seq-aug       @ mid-->p2s_vid-acamp
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,acamp-10k6_vid_entire_seq,batch-18,dbg-0,dyn-1,dist-1,jtr,res-1280
`2_per_seq_dbg_bear`
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,acamp-2_per_seq_dbg_bear,batch-1,dbg-1,dyn-1,dist-0,jtr,res-1280
<a id="on_train___10k6_vid_entire_seq_aug_mi_d_"></a>
### on-train       @ 10k6_vid_entire_seq-aug/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_10k6_vid_entire_seq-length-2-stride-1-batch_18-jtr-res_1280,_eval_,vid_det,acamp-10k6_vid_entire_seq,vstrd-1,batch-32,save-vis-0,dbg-0,dyn-1
<a id="on_inv___10k6_vid_entire_seq_aug_mi_d_"></a>
### on-inv       @ 10k6_vid_entire_seq-aug/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_10k6_vid_entire_seq-length-2-stride-1-batch_18-jtr-res_1280,_eval_,vid_det,acamp-10k6_vid_entire_seq_inv,vstrd-1,batch-12,save-vis-0,dbg-0,dyn-1



<a id="20k6_5_video_aug___mi_d_"></a>
## 20k6_5_video-aug       @ mid-->p2s_vid-acamp
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,acamp-20k6_5_video,batch-18,dbg-0,dyn-1,dist-1,jtr,res-1280
<a id="on_train___20k6_5_video_aug_mid_"></a>
### on-train       @ 20k6_5_video-aug/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_20k6_5_video-length-2-stride-1-batch_18-jtr-res_1280,_eval_,vid_det,acamp-20k6_5_video,vstrd-1,batch-4,save-vis-0,dbg-0,dyn-1
<a id="on_inv___20k6_5_video_aug_mid_"></a>
### on-inv       @ 20k6_5_video-aug/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_20k6_5_video-length-2-stride-1-batch_18-jtr-res_1280,_eval_,vid_det,acamp-20k6_5_video_inv,vstrd-1,batch-20,save-vis-0,dbg-0,dyn-1

<a id="1k8_vid_entire_seq_aug_fbb___mi_d_"></a>
## 1k8_vid_entire_seq-aug-fbb       @ mid-->p2s_vid-acamp
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,acamp-1k8_vid_entire_seq,batch-80,dbg-0,dyn-1,dist-1,jtr,res-1280,fbb
<a id="on_inv___1k8_vid_entire_seq_aug_fbb_mid_"></a>
### on-inv       @ 1k8_vid_entire_seq-aug-fbb/mid-->p2s_vid-acamp
`67700`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_1k8_vid_entire_seq-length-2-stride-1-batch_80-jtr-res_1280-fbb,_eval_,vid_det,acamp-1k8_vid_entire_seq_inv,vstrd-1,batch-2,save-vis-0,dbg-0,dyn-1,iter-67700
`97000`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_1k8_vid_entire_seq-length-2-stride-1-batch_80-jtr-res_1280-fbb,_eval_,vid_det,acamp-1k8_vid_entire_seq_inv,vstrd-1,batch-2,save-vis-0,dbg-0,dyn-1,iter-97000
<a id="on_inv_2_per_seq___1k8_vid_entire_seq_aug_fbb_mid_"></a>
### on-inv-2_per_seq       @ 1k8_vid_entire_seq-aug-fbb/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_1k8_vid_entire_seq-length-2-stride-1-batch_80-jtr-res_1280-fbb,_eval_,vid_det,acamp-1k8_vid_entire_seq_inv_2_per_seq,vstrd-1,batch-2,save-vis-0,dbg-1,dyn-1,grs
<a id="on_inv_6_per_seq___1k8_vid_entire_seq_aug_fbb_mid_"></a>
### on-inv-6_per_seq       @ 1k8_vid_entire_seq-aug-fbb/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_1k8_vid_entire_seq-length-2-stride-1-batch_80-jtr-res_1280-fbb,_eval_,vid_det,acamp-1k8_vid_entire_seq_inv_6_per_seq,vstrd-1,batch-8,save-vis-0,dbg-1,dyn-1,grs
<a id="on_inv_12_per_seq___1k8_vid_entire_seq_aug_fbb_mid_"></a>
### on-inv-12_per_seq       @ 1k8_vid_entire_seq-aug-fbb/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_1k8_vid_entire_seq-length-2-stride-1-batch_80-jtr-res_1280-fbb,_eval_,vid_det,acamp-1k8_vid_entire_seq_inv_12_per_seq,vstrd-1,batch-8,save-vis-0,dbg-1,dyn-1,grs


<a id="10k6_vid_entire_seq_aug_fbb___mi_d_"></a>
## 10k6_vid_entire_seq-aug-fbb       @ mid-->p2s_vid-acamp
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,acamp-10k6_vid_entire_seq,batch-72,dbg-0,dyn-1,dist-1,jtr,res-1280,fbb
<a id="on_inv___10k6_vid_entire_seq_aug_fbb_mi_d_"></a>
<a id="on_inv___10k6_vid_entire_seq_aug_fbb_mi_d_"></a>
### on-inv       @ 10k6_vid_entire_seq-aug-fbb/mid-->p2s_vid-acamp
`224100`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_10k6_vid_entire_seq-length-2-stride-1-batch_72-jtr-res_1280-fbb,_eval_,vid_det,acamp-10k6_vid_entire_seq_inv,vstrd-1,batch-2,save-vis-0,dbg-0,dyn-1,iter-224100
`384290`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_10k6_vid_entire_seq-length-2-stride-1-batch_72-jtr-res_1280-fbb,_eval_,vid_det,acamp-10k6_vid_entire_seq_inv,vstrd-1,batch-2,save-vis-0,dbg-0,dyn-1,iter-384290
<a id="on_inv_2_per_seq___10k6_vid_entire_seq_aug_fbb_mi_d_"></a>
### on-inv-2_per_seq       @ 10k6_vid_entire_seq-aug-fbb/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_10k6_vid_entire_seq-length-2-stride-1-batch_72-jtr-res_1280-fbb,_eval_,vid_det,acamp-10k6_vid_entire_seq_inv_2_per_seq,vstrd-1,batch-2,save-vis-0,dbg-0,dyn-1,x99
<a id="on_inv_12_per_seq___10k6_vid_entire_seq_aug_fbb_mi_d_"></a>
### on-inv-12_per_seq       @ 10k6_vid_entire_seq-aug-fbb/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_10k6_vid_entire_seq-length-2-stride-1-batch_72-jtr-res_1280-fbb,_eval_,vid_det,acamp-10k6_vid_entire_seq_inv_12_per_seq,vstrd-1,batch-2,save-vis-0,dbg-0,dyn-1,x99

<a id="20k6_5_video_aug_fbb___mi_d_"></a>
## 20k6_5_video-aug-fbb       @ mid-->p2s_vid-acamp
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,acamp-20k6_5_video,batch-72,dbg-0,dyn-1,dist-1,jtr,res-1280,fbb
<a id="on_inv___20k6_5_video_aug_fbb_mid_"></a>
### on-inv       @ 20k6_5_video-aug-fbb/mid-->p2s_vid-acamp
`303240`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_20k6_5_video-length-2-stride-1-batch_72-jtr-res_1280-fbb,_eval_,vid_det,acamp-20k6_5_video_inv,vstrd-1,batch-2,save-vis-0,dbg-0,dyn-1,iter-303240
`334590`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_20k6_5_video-length-2-stride-1-batch_72-jtr-res_1280-fbb,_eval_,vid_det,acamp-20k6_5_video_inv,vstrd-1,batch-2,save-vis-0,dbg-0,dyn-1,iter-334590
<a id="on_inv_2_per_seq___20k6_5_video_aug_fbb_mid_"></a>
### on-inv-2_per_seq       @ 20k6_5_video-aug-fbb/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_20k6_5_video-length-2-stride-1-batch_72-jtr-res_1280-fbb,_eval_,vid_det,acamp-20k6_5_video_inv_2_per_seq,vstrd-1,batch-2,save-vis-0,dbg-0,dyn-1,e5g
<a id="on_inv_12_per_seq___20k6_5_video_aug_fbb_mid_"></a>
### on-inv-12_per_seq       @ 20k6_5_video-aug-fbb/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_20k6_5_video-length-2-stride-1-batch_72-jtr-res_1280-fbb,_eval_,vid_det,acamp-20k6_5_video_inv_12_per_seq,vstrd-1,batch-8,save-vis-0,dbg-0,dyn-1,e5g


<a id="1k8_vid_entire_seq_aug_cls_eq_fbb___mi_d_"></a>
## 1k8_vid_entire_seq-aug-cls_eq-fbb       @ mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,acamp-1k8_vid_entire_seq,batch-32,dbg-0,dyn-1,dist-0,jtr,res-1280,fbb,cls_eq
<a id="on_inv_2_per_seq___1k8_vid_entire_seq_aug_cls_eq_fbb_mi_d_"></a>
### on-inv-2_per_seq       @ 1k8_vid_entire_seq-aug-cls_eq-fbb/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_1k8_vid_entire_seq-length-2-stride-1-batch_32-jtr-res_1280-fbb-cls_eq,_eval_,vid_det,acamp-1k8_vid_entire_seq_inv_2_per_seq,vstrd-1,batch-4,save-vis-0,dbg-1,dyn-1
<a id="on_inv___1k8_vid_entire_seq_aug_cls_eq_fbb_mi_d_"></a>

<a id="1k8_vid_entire_seq_aug_cls_eq_fbb_b64___mi_d_"></a>
## 1k8_vid_entire_seq-aug-cls_eq-fbb-b64       @ mid-->p2s_vid-acamp
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,acamp-1k8_vid_entire_seq,batch-64,dbg-0,dyn-0,dist-1,jtr,res-1280,fbb,cls_eq
<a id="on_inv_2_per_seq___1k8_vid_entire_seq_aug_cls_eq_fbb_b64_mi_d_"></a>
### on-inv-2_per_seq       @ 1k8_vid_entire_seq-aug-cls_eq-fbb-b64/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_1k8_vid_entire_seq-length-2-stride-1-batch_64-jtr-res_1280-fbb-cls_eq,_eval_,vid_det,acamp-1k8_vid_entire_seq_inv_2_per_seq,vstrd-1,batch-8,save-vis-0,dbg-1,dyn-1,x99
<a id="on_inv___1k8_vid_entire_seq_aug_cls_eq_fbb_b64_mi_d_"></a>
### on-inv       @ 1k8_vid_entire_seq-aug-cls_eq-fbb-b64/mid-->p2s_vid-acamp
`281976`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_1k8_vid_entire_seq-length-2-stride-1-batch_64-jtr-res_1280-fbb-cls_eq,_eval_,vid_det,acamp-1k8_vid_entire_seq_inv,vstrd-1,batch-32,save-vis-0,dbg-1,dyn-1,iter-281976

<a id="10k6_vid_entire_seq_aug_cls_eq_fbb___mi_d_"></a>
## 10k6_vid_entire_seq-aug-cls_eq-fbb       @ mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,acamp-10k6_vid_entire_seq,batch-32,dbg-0,dyn-1,dist-0,jtr,res-1280,fbb,cls_eq
`dbg`
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,acamp-10k6_vid_entire_seq,batch-3,dbg-1,dyn-1,dist-0,jtr,res-1280,fbb,cls_eq
<a id="on_inv_2_per_seq___10k6_vid_entire_seq_aug_cls_eq_fbb_mid_"></a>
### on-inv-2_per_seq       @ 10k6_vid_entire_seq-aug-cls_eq-fbb/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_10k6_vid_entire_seq-length-2-stride-1-batch_32-jtr-res_1280-fbb-cls_eq,_eval_,vid_det,acamp-10k6_vid_entire_seq_inv_2_per_seq,vstrd-1,batch-4,save-vis-0,dbg-0,dyn-1

<a id="10k6_vid_entire_seq_aug_cls_eq_fbb_b64___mi_d_"></a>
## 10k6_vid_entire_seq-aug-cls_eq-fbb-b64       @ mid-->p2s_vid-acamp
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,acamp-10k6_vid_entire_seq,batch-64,dbg-0,dyn-1,dist-1,jtr,res-1280,fbb,cls_eq
<a id="on_inv_2_per_seq___10k6_vid_entire_seq_aug_cls_eq_fbb_b64_mid_"></a>
### on-inv-2_per_seq       @ 10k6_vid_entire_seq-aug-cls_eq-fbb-b64/mid-->p2s_vid-acamp
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_10k6_vid_entire_seq-length-2-stride-1-batch_64-jtr-res_1280-fbb-cls_eq,_eval_,vid_det,acamp-10k6_vid_entire_seq_inv_2_per_seq,vstrd-1,batch-6,save-vis-0,dbg-0,dyn-1,dist-1
<a id="on_inv___10k6_vid_entire_seq_aug_cls_eq_fbb_b64_mid_"></a>
### on-inv       @ 10k6_vid_entire_seq-aug-cls_eq-fbb-b64/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_10k6_vid_entire_seq-length-2-stride-1-batch_64-jtr-res_1280-fbb-cls_eq,_eval_,vid_det,acamp-10k6_vid_entire_seq_inv,vstrd-1,batch-32,save-vis-0,dbg-0,dyn-1,iter-1132942

<a id="10k6_vid_entire_seq_aug_cls_eq_fbb_b128___mi_d_"></a>
## 10k6_vid_entire_seq-aug-cls_eq-fbb-b128       @ mid-->p2s_vid-acamp
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,acamp-10k6_vid_entire_seq,batch-128,dbg-0,dyn-1,dist-2,jtr,res-1280,fbb,cls_eq,px
<a id="on_inv_2_per_seq___10k6_vid_entire_seq_aug_cls_eq_fbb_b128_mi_d_"></a>
### on-inv-2_per_seq       @ 10k6_vid_entire_seq-aug-cls_eq-fbb-b128/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_10k6_vid_entire_seq-length-2-stride-1-batch_128-jtr-res_1280-fbb-cls_eq-px,_eval_,vid_det,acamp-10k6_vid_entire_seq_inv_2_per_seq,vstrd-1,batch-4,save-vis-0,dbg-0,dyn-1,dist-0,p9

<a id="10k6_vid_entire_seq_aug_cls_eq___mi_d_"></a>
## 10k6_vid_entire_seq-aug-cls_eq       @ mid-->p2s_vid-acamp
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,acamp-10k6_vid_entire_seq,batch-18,dbg-0,dyn-1,dist-1,jtr,res-1280,cls_eq
<a id="on_inv_2_per_seq___10k6_vid_entire_seq_aug_cls_eq_mid_"></a>
### on-inv-2_per_seq       @ 10k6_vid_entire_seq-aug-cls_eq/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_10k6_vid_entire_seq-length-2-stride-1-batch_18-jtr-res_1280-cls_eq,_eval_,vid_det,acamp-10k6_vid_entire_seq_inv_2_per_seq,vstrd-1,batch-2,save-vis-0,dbg-0,dyn-1,e5g
<a id="on_inv___10k6_vid_entire_seq_aug_cls_eq_mid_"></a>
### on-inv       @ 10k6_vid_entire_seq-aug-cls_eq/mid-->p2s_vid-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_10k6_vid_entire_seq-length-2-stride-1-batch_18-jtr-res_1280-cls_eq,_eval_,vid_det,acamp-10k6_vid_entire_seq_inv,vstrd-1,batch-12,save-vis-0,dbg-0,dyn-1
