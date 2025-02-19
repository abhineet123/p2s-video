<!-- MarkdownTOC -->

- [resnet-640](#resnet_64_0_)
    - [1k8_vid_entire_seq       @ resnet-640](#1k8_vid_entire_seq___resnet_640_)
        - [on-train       @ 1k8_vid_entire_seq/resnet-640](#on_train___1k8_vid_entire_seq_resnet_64_0_)
        - [on-inv       @ 1k8_vid_entire_seq/resnet-640](#on_inv___1k8_vid_entire_seq_resnet_64_0_)
    - [10k6_vid_entire_seq       @ resnet-640](#10k6_vid_entire_seq___resnet_640_)
        - [on-train       @ 10k6_vid_entire_seq/resnet-640](#on_train___10k6_vid_entire_seq_resnet_640_)
        - [on-inv       @ 10k6_vid_entire_seq/resnet-640](#on_inv___10k6_vid_entire_seq_resnet_640_)
    - [20k6_5_video       @ resnet-640](#20k6_5_video___resnet_640_)
        - [on-train       @ 20k6_5_video/resnet-640](#on_train___20k6_5_video_resnet_64_0_)
        - [on-inv       @ 20k6_5_video/resnet-640](#on_inv___20k6_5_video_resnet_64_0_)
    - [1k8_vid_entire_seq-aug       @ resnet-640](#1k8_vid_entire_seq_aug___resnet_640_)
    - [10k6_vid_entire_seq-aug       @ resnet-640](#10k6_vid_entire_seq_aug___resnet_640_)
        - [on-inv       @ 10k6_vid_entire_seq-aug/resnet-640](#on_inv___10k6_vid_entire_seq_aug_resnet_640_)
    - [20k6_5_video-aug       @ resnet-640](#20k6_5_video_aug___resnet_640_)
    - [1k8_vid_entire_seq-aug-fbb       @ resnet-640](#1k8_vid_entire_seq_aug_fbb___resnet_640_)
        - [on-inv       @ 1k8_vid_entire_seq-aug-fbb/resnet-640](#on_inv___1k8_vid_entire_seq_aug_fbb_resnet_64_0_)
        - [on-inv-2_per_seq       @ 1k8_vid_entire_seq-aug-fbb/resnet-640](#on_inv_2_per_seq___1k8_vid_entire_seq_aug_fbb_resnet_64_0_)
    - [10k6_vid_entire_seq-aug-fbb       @ resnet-640](#10k6_vid_entire_seq_aug_fbb___resnet_640_)
        - [on-inv-2_per_seq       @ 10k6_vid_entire_seq-aug-fbb/resnet-640](#on_inv_2_per_seq___10k6_vid_entire_seq_aug_fbb_resnet_640_)
        - [on-inv       @ 10k6_vid_entire_seq-aug-fbb/resnet-640](#on_inv___10k6_vid_entire_seq_aug_fbb_resnet_640_)
    - [10k6_vid_entire_seq-aug-b96-fbb       @ resnet-640](#10k6_vid_entire_seq_aug_b96_fbb___resnet_640_)
        - [on-inv-2_per_seq       @ 10k6_vid_entire_seq-aug-b96-fbb/resnet-640](#on_inv_2_per_seq___10k6_vid_entire_seq_aug_b96_fbb_resnet_640_)
        - [on-inv       @ 10k6_vid_entire_seq-aug-b96-fbb/resnet-640](#on_inv___10k6_vid_entire_seq_aug_b96_fbb_resnet_640_)
    - [10k6_vid_entire_seq-aug-cls_eq-fbb       @ resnet-640](#10k6_vid_entire_seq_aug_cls_eq_fbb___resnet_640_)
        - [on-inv-2_per_seq       @ 10k6_vid_entire_seq-aug-cls_eq-fbb/resnet-640](#on_inv_2_per_seq___10k6_vid_entire_seq_aug_cls_eq_fbb_resnet_64_0_)
        - [on-inv       @ 10k6_vid_entire_seq-aug-cls_eq-fbb/resnet-640](#on_inv___10k6_vid_entire_seq_aug_cls_eq_fbb_resnet_64_0_)
    - [20k6_5_video-aug-fbb       @ resnet-640](#20k6_5_video_aug_fbb___resnet_640_)
        - [on-inv       @ 20k6_5_video-aug-fbb/resnet-640](#on_inv___20k6_5_video_aug_fbb_resnet_64_0_)
        - [on-inv-2_per_seq       @ 20k6_5_video-aug-fbb/resnet-640](#on_inv_2_per_seq___20k6_5_video_aug_fbb_resnet_64_0_)

<!-- /MarkdownTOC -->
<a id="resnet_64_0_"></a>
# resnet-640 

<a id="1k8_vid_entire_seq___resnet_640_"></a>
## 1k8_vid_entire_seq       @ resnet-640-->p2s-acamp
python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,acamp-1k8_vid_entire_seq,batch-32,dbg-0,dyn-1,dist-1
<a id="on_train___1k8_vid_entire_seq_resnet_64_0_"></a>
### on-train       @ 1k8_vid_entire_seq/resnet-640-->p2s-acamp
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_1k8_vid_entire_seq-batch_32,_eval_,acamp-1k8_vid_entire_seq,batch-4,save-vis-0,dbg-0,dyn-1
<a id="on_inv___1k8_vid_entire_seq_resnet_64_0_"></a>
### on-inv       @ 1k8_vid_entire_seq/resnet-640-->p2s-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_1k8_vid_entire_seq-batch_32,_eval_,acamp-1k8_vid_entire_seq_inv,batch-8,save-vis-0,dbg-0,dyn-1

<a id="10k6_vid_entire_seq___resnet_640_"></a>
## 10k6_vid_entire_seq       @ resnet-640-->p2s-acamp
python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,acamp-10k6_vid_entire_seq,batch-32,dbg-0,dyn-1,dist-1
<a id="on_train___10k6_vid_entire_seq_resnet_640_"></a>
### on-train       @ 10k6_vid_entire_seq/resnet-640-->p2s-acamp
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_10k6_vid_entire_seq-batch_32,_eval_,acamp-10k6_vid_entire_seq,batch-4,save-vis-0,dbg-0,dyn-1
<a id="on_inv___10k6_vid_entire_seq_resnet_640_"></a>
### on-inv       @ 10k6_vid_entire_seq/resnet-640-->p2s-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_10k6_vid_entire_seq-batch_32,_eval_,acamp-10k6_vid_entire_seq_inv,batch-8,save-vis-0,dbg-0,dyn-1

<a id="20k6_5_video___resnet_640_"></a>
## 20k6_5_video       @ resnet-640-->p2s-acamp
python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,acamp-20k6_5_video,batch-32,dbg-0,dyn-1,dist-1
<a id="on_train___20k6_5_video_resnet_64_0_"></a>
### on-train       @ 20k6_5_video/resnet-640-->p2s-acamp
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_20k6_5_video-batch_32,_eval_,acamp-20k6_5_video,batch-4,save-vis-0,dbg-0,dyn-1
<a id="on_inv___20k6_5_video_resnet_64_0_"></a>
### on-inv       @ 20k6_5_video/resnet-640-->p2s-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_20k6_5_video-batch_32,_eval_,acamp-20k6_5_video_inv,batch-12,save-vis-0,dbg-0,dyn-1

<a id="1k8_vid_entire_seq_aug___resnet_640_"></a>
## 1k8_vid_entire_seq-aug       @ resnet-640-->p2s-acamp
python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,acamp-1k8_vid_entire_seq,batch-32,dbg-0,dyn-1,dist-1,jtr,res-1280

<a id="10k6_vid_entire_seq_aug___resnet_640_"></a>
## 10k6_vid_entire_seq-aug       @ resnet-640-->p2s-acamp
python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,acamp-10k6_vid_entire_seq,batch-32,dbg-0,dyn-1,dist-1,jtr,res-1280
<a id="on_inv___10k6_vid_entire_seq_aug_resnet_640_"></a>
### on-inv       @ 10k6_vid_entire_seq-aug/resnet-640-->p2s-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_10k6_vid_entire_seq-batch_32,_eval_,acamp-10k6_vid_entire_seq_inv,batch-8,save-vis-0,dbg-0,dyn-1

<a id="20k6_5_video_aug___resnet_640_"></a>
## 20k6_5_video-aug       @ resnet-640-->p2s-acamp
python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,acamp-20k6_5_video,batch-32,dbg-0,dyn-1,dist-1,jtr,res-1280

<a id="1k8_vid_entire_seq_aug_fbb___resnet_640_"></a>
## 1k8_vid_entire_seq-aug-fbb       @ resnet-640-->p2s-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,acamp-1k8_vid_entire_seq,batch-48,dbg-0,dyn-1,dist-0,jtr,res-1280,fbb
<a id="on_inv___1k8_vid_entire_seq_aug_fbb_resnet_64_0_"></a>
### on-inv       @ 1k8_vid_entire_seq-aug-fbb/resnet-640-->p2s-acamp
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_1k8_vid_entire_seq-batch_48-jtr-res_1280-fbb,_eval_,acamp-1k8_vid_entire_seq_inv,batch-2,save-vis-0,dbg-0,dyn-1,iter-143453
<a id="on_inv_2_per_seq___1k8_vid_entire_seq_aug_fbb_resnet_64_0_"></a>
### on-inv-2_per_seq       @ 1k8_vid_entire_seq-aug-fbb/resnet-640-->p2s-acamp
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_1k8_vid_entire_seq-batch_48-jtr-res_1280-fbb,_eval_,acamp-1k8_vid_entire_seq_inv_2_per_seq,batch-2,save-vis-0,dbg-0,dyn-1,grs-2

<a id="10k6_vid_entire_seq_aug_fbb___resnet_640_"></a>
## 10k6_vid_entire_seq-aug-fbb       @ resnet-640-->p2s-acamp
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,acamp-10k6_vid_entire_seq,batch-48,dbg-0,dyn-1,dist-0,jtr,res-1280,fbb
<a id="on_inv_2_per_seq___10k6_vid_entire_seq_aug_fbb_resnet_640_"></a>
### on-inv-2_per_seq       @ 10k6_vid_entire_seq-aug-fbb/resnet-640-->p2s-acamp
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_10k6_vid_entire_seq-batch_48-jtr-res_1280-fbb,_eval_,acamp-10k6_vid_entire_seq_inv_2_per_seq,batch-2,save-vis-0,dbg-0,dyn-1,grs-2
<a id="on_inv___10k6_vid_entire_seq_aug_fbb_resnet_640_"></a>
### on-inv       @ 10k6_vid_entire_seq-aug-fbb/resnet-640-->p2s-acamp
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_10k6_vid_entire_seq-batch_48-jtr-res_1280-fbb,_eval_,acamp-10k6_vid_entire_seq_inv,batch-8,save-vis-0,dbg-0,dyn-1,iter-513750

<a id="10k6_vid_entire_seq_aug_b96_fbb___resnet_640_"></a>
## 10k6_vid_entire_seq-aug-b96-fbb       @ resnet-640-->p2s-acamp
python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,acamp-10k6_vid_entire_seq,batch-96,dbg-0,dyn-1,dist-1,jtr,res-1280,fbb
<a id="on_inv_2_per_seq___10k6_vid_entire_seq_aug_b96_fbb_resnet_640_"></a>
### on-inv-2_per_seq       @ 10k6_vid_entire_seq-aug-b96-fbb/resnet-640-->p2s-acamp
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_10k6_vid_entire_seq-batch_96-jtr-res_1280-fbb,_eval_,acamp-10k6_vid_entire_seq_inv_2_per_seq,batch-4,save-vis-0,dbg-0,dyn-1
<a id="on_inv___10k6_vid_entire_seq_aug_b96_fbb_resnet_640_"></a>
### on-inv       @ 10k6_vid_entire_seq-aug-b96-fbb/resnet-640-->p2s-acamp
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_10k6_vid_entire_seq-batch_96-jtr-res_1280-fbb,_eval_,acamp-10k6_vid_entire_seq_inv,batch-8,save-vis-0,dbg-0,dyn-1

<a id="10k6_vid_entire_seq_aug_cls_eq_fbb___resnet_640_"></a>
## 10k6_vid_entire_seq-aug-cls_eq-fbb       @ resnet-640-->p2s-acamp
python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,acamp-10k6_vid_entire_seq,batch-128,dbg-0,dyn-1,dist-1,jtr,res-1280,fbb,cls_eq
<a id="on_inv_2_per_seq___10k6_vid_entire_seq_aug_cls_eq_fbb_resnet_64_0_"></a>
### on-inv-2_per_seq       @ 10k6_vid_entire_seq-aug-cls_eq-fbb/resnet-640-->p2s-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_10k6_vid_entire_seq-batch_128-jtr-res_1280-fbb-cls_eq,_eval_,acamp-10k6_vid_entire_seq_inv_2_per_seq,batch-4,save-vis-0,dbg-0,dyn-1,e5g
<a id="on_inv___10k6_vid_entire_seq_aug_cls_eq_fbb_resnet_64_0_"></a>
### on-inv       @ 10k6_vid_entire_seq-aug-cls_eq-fbb/resnet-640-->p2s-acamp
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_10k6_vid_entire_seq-batch_128-jtr-res_1280-fbb-cls_eq,_eval_,acamp-10k6_vid_entire_seq_inv,batch-4,save-vis-0,dbg-0,dyn-1,iter-272958


<a id="20k6_5_video_aug_fbb___resnet_640_"></a>
## 20k6_5_video-aug-fbb       @ resnet-640-->p2s-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,acamp-20k6_5_video,batch-48,dbg-0,dyn-1,dist-0,jtr,res-1280,fbb
<a id="on_inv___20k6_5_video_aug_fbb_resnet_64_0_"></a>
### on-inv       @ 20k6_5_video-aug-fbb/resnet-640-->p2s-acamp
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_20k6_5_video-batch_48-jtr-res_1280-fbb,_eval_,acamp-20k6_5_video_inv,batch-6,save-vis-0,dbg-0,dyn-1,iter-171402
<a id="on_inv_2_per_seq___20k6_5_video_aug_fbb_resnet_64_0_"></a>
### on-inv-2_per_seq       @ 20k6_5_video-aug-fbb/resnet-640-->p2s-acamp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=m-resnet_640_20k6_5_video-batch_48-jtr-res_1280-fbb,_eval_,acamp-20k6_5_video_inv_2_per_seq,batch-2,save-vis-0,dbg-0,dyn-1,grs-2
