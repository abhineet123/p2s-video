<!-- MarkdownTOC -->

- [vit_l-640](#vit_l_640_)
    - [vid_det_all-sampled_eq-aug       @ vit_l-640](#vid_det_all_sampled_eq_aug___vit_l_64_0_)
        - [on-train-ratio_1_10_random       @ vid_det_all-sampled_eq-aug/vit_l-640](#on_train_ratio_1_10_random___vid_det_all_sampled_eq_aug_vit_l_640_)
        - [on-val-16_per_seq_random       @ vid_det_all-sampled_eq-aug/vit_l-640](#on_val_16_per_seq_random___vid_det_all_sampled_eq_aug_vit_l_640_)
- [vit_b-640](#vit_b_640_)
    - [vid_det-sampled_eq-aug-fbb       @ vit_b-640](#vid_det_sampled_eq_aug_fbb___vit_b_64_0_)
        - [on-train-ratio_1_10_random       @ vid_det-sampled_eq-aug-fbb/vit_b-640](#on_train_ratio_1_10_random___vid_det_sampled_eq_aug_fbb_vit_b_640_)
        - [on-val-16_per_seq_random       @ vid_det-sampled_eq-aug-fbb/vit_b-640](#on_val_16_per_seq_random___vid_det_sampled_eq_aug_fbb_vit_b_640_)
        - [on-val       @ vid_det-sampled_eq-aug-fbb/vit_b-640](#on_val___vid_det_sampled_eq_aug_fbb_vit_b_640_)
    - [vid-aug-fbb       @ vit_b-640](#vid_aug_fbb___vit_b_64_0_)
        - [on-train-8_per_seq_random       @ vid-aug-fbb/vit_b-640](#on_train_8_per_seq_random___vid_aug_fbb_vit_b_64_0_)
        - [on-val-16_per_seq_random       @ vid-aug-fbb/vit_b-640](#on_val_16_per_seq_random___vid_aug_fbb_vit_b_64_0_)
        - [on-val       @ vid-aug-fbb/vit_b-640](#on_val___vid_aug_fbb_vit_b_64_0_)
- [resnet-640](#resnet_64_0_)
    - [vid-aug-fbb       @ resnet-640](#vid_aug_fbb___resnet_640_)
        - [on-train-8_per_seq_random       @ vid-aug-fbb/resnet-640](#on_train_8_per_seq_random___vid_aug_fbb_resnet_640_)
        - [on-val-16_per_seq_random       @ vid-aug-fbb/resnet-640](#on_val_16_per_seq_random___vid_aug_fbb_resnet_640_)
        - [on-val-67952       @ vid-aug-fbb/resnet-640](#on_val_67952___vid_aug_fbb_resnet_640_)
        - [on-val-103024       @ vid-aug-fbb/resnet-640](#on_val_103024___vid_aug_fbb_resnet_640_)
    - [vid_det-sampled_eq-aug-fbb       @ resnet-640](#vid_det_sampled_eq_aug_fbb___resnet_640_)
        - [on-train-8_per_seq_random       @ vid_det-sampled_eq-aug-fbb/resnet-640](#on_train_8_per_seq_random___vid_det_sampled_eq_aug_fbb_resnet_64_0_)
        - [on-train-ratio_1_10_random       @ vid_det-sampled_eq-aug-fbb/resnet-640](#on_train_ratio_1_10_random___vid_det_sampled_eq_aug_fbb_resnet_64_0_)
        - [on-val-16_per_seq_random       @ vid_det-sampled_eq-aug-fbb/resnet-640](#on_val_16_per_seq_random___vid_det_sampled_eq_aug_fbb_resnet_64_0_)
        - [on-val-129886       @ vid_det-sampled_eq-aug-fbb/resnet-640](#on_val_129886___vid_det_sampled_eq_aug_fbb_resnet_64_0_)
    - [vid_det-sampled_eq-aug       @ resnet-640](#vid_det_sampled_eq_aug___resnet_640_)
        - [on-train-ratio_1_10_random       @ vid_det-sampled_eq-aug/resnet-640](#on_train_ratio_1_10_random___vid_det_sampled_eq_aug_resnet_64_0_)
        - [on-val-16_per_seq_random       @ vid_det-sampled_eq-aug/resnet-640](#on_val_16_per_seq_random___vid_det_sampled_eq_aug_resnet_64_0_)
    - [vid_det-sampled_eq-aug-isc       @ resnet-640](#vid_det_sampled_eq_aug_isc___resnet_640_)
        - [on-train-ratio_1_10_random       @ vid_det-sampled_eq-aug-isc/resnet-640](#on_train_ratio_1_10_random___vid_det_sampled_eq_aug_isc_resnet_64_0_)
        - [on-val-16_per_seq_random       @ vid_det-sampled_eq-aug-isc/resnet-640](#on_val_16_per_seq_random___vid_det_sampled_eq_aug_isc_resnet_64_0_)
    - [vid_det_all-sampled_eq-aug-fbb       @ resnet-640](#vid_det_all_sampled_eq_aug_fbb___resnet_640_)
        - [on-train-ratio_1_10_random       @ vid_det_all-sampled_eq-aug-fbb/resnet-640](#on_train_ratio_1_10_random___vid_det_all_sampled_eq_aug_fbb_resnet_64_0_)
            - [batch-16       @ on-train-ratio_1_10_random/vid_det_all-sampled_eq-aug-fbb/resnet-640](#batch_16___on_train_ratio_1_10_random_vid_det_all_sampled_eq_aug_fbb_resnet_640_)
        - [on-val-16_per_seq_random       @ vid_det_all-sampled_eq-aug-fbb/resnet-640](#on_val_16_per_seq_random___vid_det_all_sampled_eq_aug_fbb_resnet_64_0_)
            - [batch-16       @ on-val-16_per_seq_random/vid_det_all-sampled_eq-aug-fbb/resnet-640](#batch_16___on_val_16_per_seq_random_vid_det_all_sampled_eq_aug_fbb_resnet_640_)

<!-- /MarkdownTOC -->
<a id="vit_l_640_"></a>
# vit_l-640 
<a id="vid_det_all_sampled_eq_aug___vit_l_64_0_"></a>
## vid_det_all-sampled_eq-aug       @ vit_l-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=0 NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME==enp3s0 NCCL_P2P_DISABLE=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,vit_l-640,pt-1,imgn-vid_det_all-sampled_eq,batch-16,dbg-0,dyn-1,dist-2,jtr,res-1440,fbb,self2-0
CUDA_VISIBLE_DEVICES=1 NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME==enp3s0 NCCL_P2P_DISABLE=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,vit_l-640,pt-1,imgn-vid_det_all-sampled_eq,batch-16,dbg-0,dyn-1,dist-2,jtr,res-1440,fbb,self2-1
`dbg`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,vit_l-640,pt-1,imgn-vid_det_all-sampled_eq,batch-8,dbg-0,dyn-1,dist-0,jtr,res-1440,fbb

watch tail -1 
log/
vit_l_640_imagenet_vid_det_all-sampled_eq-batch_16-jtr-res_1440-fbb-self2-0
/progress_log.txt

<a id="on_train_ratio_1_10_random___vid_det_all_sampled_eq_aug_vit_l_640_"></a>
### on-train-ratio_1_10_random       @ vid_det_all-sampled_eq-aug/vit_l-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-vit_l_640_imagenet_vid_det_all-sampled_eq-batch_16-jtr-res_1440-fbb-self2-0,imgn-vid_det_all-ratio_1_10_random,batch-32,save-vis-0,dbg-0,dyn-1,dist-0,isc
<a id="on_val_16_per_seq_random___vid_det_all_sampled_eq_aug_vit_l_640_"></a>
### on-val-16_per_seq_random       @ vid_det_all-sampled_eq-aug/vit_l-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-vit_l_640_imagenet_vid_det_all-sampled_eq-batch_16-jtr-res_1440-fbb-self2-0,imgn-vid_val-16_per_seq_random,batch-32,save-vis-0,dbg-0,dyn-1,dist-0,isc

<a id="vit_b_640_"></a>
# vit_b-640 
<a id="vid_det_sampled_eq_aug_fbb___vit_b_64_0_"></a>
## vid_det-sampled_eq-aug-fbb       @ vit_b-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=0 NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME==enp3s0 NCCL_P2P_DISABLE=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,vit_b-640,pt-1,imgn-vid_det-sampled_eq,batch-40,dbg-0,dyn-1,dist-2,jtr,res-1440,fbb,self2-0
CUDA_VISIBLE_DEVICES=1 NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME==enp3s0 NCCL_P2P_DISABLE=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,vit_b-640,pt-1,imgn-vid_det-sampled_eq,batch-40,dbg-0,dyn-1,dist-2,jtr,res-1440,fbb,self2-1
`dbg`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,vit_b-640,pt-1,imgn-vid_det-sampled_eq,batch-20,dbg-0,dyn-1,dist-0,jtr,res-1440,fbb
<a id="on_train_ratio_1_10_random___vid_det_sampled_eq_aug_fbb_vit_b_640_"></a>
### on-train-ratio_1_10_random       @ vid_det-sampled_eq-aug-fbb/vit_b-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-vit_b_640_imagenet_vid_det-sampled_eq-batch_40-jtr-res_1440-fbb-self2-0,imgn-vid_det-ratio_1_10_random,batch-32,save-vis-0,dbg-0,dyn-1,dist-0,isc
<a id="on_val_16_per_seq_random___vid_det_sampled_eq_aug_fbb_vit_b_640_"></a>
### on-val-16_per_seq_random       @ vid_det-sampled_eq-aug-fbb/vit_b-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-vit_b_640_imagenet_vid_det-sampled_eq-batch_40-jtr-res_1440-fbb-self2-0,imgn-vid_val-16_per_seq_random,batch-32,save-vis-0,dbg-0,dyn-1,dist-0,isc
<a id="on_val___vid_det_sampled_eq_aug_fbb_vit_b_640_"></a>
### on-val       @ vid_det-sampled_eq-aug-fbb/vit_b-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-vit_b_640_imagenet_vid_det-sampled_eq-batch_40-jtr-res_1440-fbb-self2-0,imgn-vid_val,batch-32,save-vis-0,dbg-0,dyn-1,dist-0,iter-65656


<a id="vid_aug_fbb___vit_b_64_0_"></a>
## vid-aug-fbb       @ vit_b-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=0 NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME==enp3s0 NCCL_P2P_DISABLE=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,vit_b-640,pt-1,imgn-vid,batch-40,dbg-0,dyn-1,dist-2,jtr,res-1440,fbb,self2-0
CUDA_VISIBLE_DEVICES=1 NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME==enp3s0 NCCL_P2P_DISABLE=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,vit_b-640,pt-1,imgn-vid,batch-40,dbg-0,dyn-1,dist-2,jtr,res-1440,fbb,self2-1
<a id="on_train_8_per_seq_random___vid_aug_fbb_vit_b_64_0_"></a>
### on-train-8_per_seq_random       @ vid-aug-fbb/vit_b-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-vit_b_640_imagenet_vid-batch_40-jtr-res_1440-fbb-self2-0,imgn-vid-8_per_seq_random,batch-32,save-vis-0,dbg-0,dyn-1,dist-0,isc
<a id="on_val_16_per_seq_random___vid_aug_fbb_vit_b_64_0_"></a>
### on-val-16_per_seq_random       @ vid-aug-fbb/vit_b-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-vit_b_640_imagenet_vid-batch_40-jtr-res_1440-fbb-self2-0,imgn-vid_val-16_per_seq_random,batch-32,save-vis-0,dbg-0,dyn-1,dist-0,isc
<a id="on_val___vid_aug_fbb_vit_b_64_0_"></a>
### on-val       @ vid-aug-fbb/vit_b-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-vit_b_640_imagenet_vid-batch_40-jtr-res_1440-fbb-self2-0,imgn-vid_val,batch-32,save-vis-0,dbg-0,dyn-1,dist-0,iter-140300


<a id="resnet_64_0_"></a>
# resnet-640 

<a id="vid_aug_fbb___resnet_640_"></a>
## vid-aug-fbb       @ resnet-640-->p2s-imgn
python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,imgn-vid,batch-512,dbg-0,dyn-1,dist-2,jtr,res-1440,fbb,zexg
`dbg`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,imgn-vid,batch-64,dbg-0,dyn-1,dist-0,jtr,res-1440,fbb
<a id="on_train_8_per_seq_random___vid_aug_fbb_resnet_640_"></a>
### on-train-8_per_seq_random       @ vid-aug-fbb/resnet-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-resnet_640_imagenet_vid-batch_512-jtr-res_1440-fbb-zexg,imgn-vid-8_per_seq_random,batch-8,save-vis-0,dbg-0,dyn-1,dist-0,grs
<a id="on_val_16_per_seq_random___vid_aug_fbb_resnet_640_"></a>
### on-val-16_per_seq_random       @ vid-aug-fbb/resnet-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-resnet_640_imagenet_vid-batch_512-jtr-res_1440-fbb-zexg,imgn-vid_val-16_per_seq_random,batch-8,save-vis-0,dbg-0,dyn-1,dist-0,grs
<a id="on_val_67952___vid_aug_fbb_resnet_640_"></a>
### on-val-67952       @ vid-aug-fbb/resnet-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-resnet_640_imagenet_vid-batch_512-jtr-res_1440-fbb-zexg,imgn-vid_val,batch-8,save-vis-0,dbg-0,dyn-1,dist-0,iter-67952
<a id="on_val_103024___vid_aug_fbb_resnet_640_"></a>
### on-val-103024       @ vid-aug-fbb/resnet-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-resnet_640_imagenet_vid-batch_512-jtr-res_1440-fbb-zexg,imgn-vid_val,batch-8,save-vis-0,dbg-0,dyn-1,dist-0,iter-103024


<a id="vid_det_sampled_eq_aug_fbb___resnet_640_"></a>
## vid_det-sampled_eq-aug-fbb       @ resnet-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=0 NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME==enp3s0 NCCL_P2P_DISABLE=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,imgn-vid_det-sampled_eq,batch-448,dbg-0,dyn-1,dist-2,jtr,res-1440,fbb,self2-0
CUDA_VISIBLE_DEVICES=1 NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME==enp3s0 NCCL_P2P_DISABLE=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,imgn-vid_det-sampled_eq,batch-448,dbg-0,dyn-1,dist-2,jtr,res-1440,fbb,self2-1
`dbg`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,imgn-vid_det-sampled_eq,batch-224,dbg-0,dyn-1,dist-0,jtr,res-1440,fbb
<a id="on_train_8_per_seq_random___vid_det_sampled_eq_aug_fbb_resnet_64_0_"></a>
### on-train-8_per_seq_random       @ vid_det-sampled_eq-aug-fbb/resnet-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-resnet_640_imagenet_vid_det-sampled_eq-batch_448-jtr-res_1440-fbb-self2-0,imgn-vid_det-8_per_seq_random,batch-16,save-vis-0,dbg-0,dyn-1,dist-0,isc
<a id="on_train_ratio_1_10_random___vid_det_sampled_eq_aug_fbb_resnet_64_0_"></a>
### on-train-ratio_1_10_random       @ vid_det-sampled_eq-aug-fbb/resnet-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-resnet_640_imagenet_vid_det-sampled_eq-batch_448-jtr-res_1440-fbb-self2-0,imgn-vid_det-ratio_1_10_random,batch-32,save-vis-0,dbg-0,dyn-1,dist-0,isc
<a id="on_val_16_per_seq_random___vid_det_sampled_eq_aug_fbb_resnet_64_0_"></a>
### on-val-16_per_seq_random       @ vid_det-sampled_eq-aug-fbb/resnet-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-resnet_640_imagenet_vid_det-sampled_eq-batch_448-jtr-res_1440-fbb-self2-0,imgn-vid_val-16_per_seq_random,batch-16,save-vis-0,dbg-0,dyn-1,dist-0,isc
<a id="on_val_129886___vid_det_sampled_eq_aug_fbb_resnet_64_0_"></a>
### on-val-129886       @ vid_det-sampled_eq-aug-fbb/resnet-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=2 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-resnet_640_imagenet_vid_det-sampled_eq-batch_448-jtr-res_1440-fbb-self2-0,imgn-vid_val,batch-32,save-vis-0,dbg-0,dyn-1,dist-0,iter-129886

<a id="vid_det_sampled_eq_aug___resnet_640_"></a>
## vid_det-sampled_eq-aug       @ resnet-640-->p2s-imgn
python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,imgn-vid_det-sampled_eq,batch-144,dbg-0,dyn-1,dist-2,jtr,res-1440,zexg
`dbg`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,imgn-vid_det-sampled_eq,batch-18,dbg-0,dyn-1,dist-0,jtr,res-1440
<a id="on_train_ratio_1_10_random___vid_det_sampled_eq_aug_resnet_64_0_"></a>
### on-train-ratio_1_10_random       @ vid_det-sampled_eq-aug/resnet-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-resnet_640_imagenet_vid_det-sampled_eq-batch_144-jtr-res_1440-zexg,imgn-vid_det-ratio_1_10_random,batch-32,save-vis-0,dbg-0,dyn-1,dist-0,grs
<a id="on_val_16_per_seq_random___vid_det_sampled_eq_aug_resnet_64_0_"></a>
### on-val-16_per_seq_random       @ vid_det-sampled_eq-aug/resnet-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-resnet_640_imagenet_vid_det-sampled_eq-batch_144-jtr-res_1440-zexg,imgn-vid_val-16_per_seq_random,batch-16,save-vis-0,dbg-0,dyn-1,dist-0,grs

<a id="vid_det_sampled_eq_aug_isc___resnet_640_"></a>
## vid_det-sampled_eq-aug-isc       @ resnet-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=0 NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME==enp3s0 NCCL_P2P_DISABLE=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,imgn-vid_det-sampled_eq,batch-144,dbg-0,dyn-1,dist-2,jtr,res-1440,self2-0
CUDA_VISIBLE_DEVICES=1 NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME==enp3s0 NCCL_P2P_DISABLE=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,imgn-vid_det-sampled_eq,batch-144,dbg-0,dyn-1,dist-2,jtr,res-1440,self2-1
`dbg`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,imgn-vid_det-sampled_eq,batch-72,dbg-0,dyn-1,dist-0,jtr,res-1440
<a id="on_train_ratio_1_10_random___vid_det_sampled_eq_aug_isc_resnet_64_0_"></a>
### on-train-ratio_1_10_random       @ vid_det-sampled_eq-aug-isc/resnet-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-resnet_640_imagenet_vid_det-sampled_eq-batch_144-jtr-res_1440-self2-0,imgn-vid_det-ratio_1_10_random,batch-32,save-vis-0,dbg-0,dyn-1,dist-0,isc
<a id="on_val_16_per_seq_random___vid_det_sampled_eq_aug_isc_resnet_64_0_"></a>
### on-val-16_per_seq_random       @ vid_det-sampled_eq-aug-isc/resnet-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-resnet_640_imagenet_vid_det-sampled_eq-batch_144-jtr-res_1440-self2-0,imgn-vid_val-16_per_seq_random,batch-32,save-vis-0,dbg-0,dyn-1,dist-0,isc


<a id="vid_det_all_sampled_eq_aug_fbb___resnet_640_"></a>
## vid_det_all-sampled_eq-aug-fbb       @ resnet-640-->p2s-imgn
python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,imgn-vid_det_all-sampled_eq,batch-384,dbg-0,dyn-1,dist-2,jtr,res-1440,fbb,zeg
`dbg`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=train,resnet-640,pt-1,imgn-vid_det_all-sampled_eq,batch-64,dbg-0,dyn-1,dist-0,jtr,res-1440,fbb
<a id="on_train_ratio_1_10_random___vid_det_all_sampled_eq_aug_fbb_resnet_64_0_"></a>
### on-train-ratio_1_10_random       @ vid_det_all-sampled_eq-aug-fbb/resnet-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-resnet_640_imagenet_vid_det_all-sampled_eq-batch_384-jtr-res_1440-fbb-zeg,imgn-vid_det_all-ratio_1_10_random,batch-32,save-vis-0,dbg-0,dyn-1,dist-0,grs
<a id="batch_16___on_train_ratio_1_10_random_vid_det_all_sampled_eq_aug_fbb_resnet_640_"></a>
#### batch-16       @ on-train-ratio_1_10_random/vid_det_all-sampled_eq-aug-fbb/resnet-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-resnet_640_imagenet_vid_det_all-sampled_eq-batch_384-jtr-res_1440-fbb-zeg,imgn-vid_det_all-ratio_1_10_random,batch-16,save-vis-0,dbg-0,dyn-1,dist-0,grs
<a id="on_val_16_per_seq_random___vid_det_all_sampled_eq_aug_fbb_resnet_64_0_"></a>
### on-val-16_per_seq_random       @ vid_det_all-sampled_eq-aug-fbb/resnet-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-resnet_640_imagenet_vid_det_all-sampled_eq-batch_384-jtr-res_1440-fbb-zeg,imgn-vid_val-16_per_seq_random,batch-32,save-vis-0,dbg-0,dyn-1,dist-0,grs
<a id="batch_16___on_val_16_per_seq_random_vid_det_all_sampled_eq_aug_fbb_resnet_640_"></a>
#### batch-16       @ on-val-16_per_seq_random/vid_det_all-sampled_eq-aug-fbb/resnet-640-->p2s-imgn
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py --j5=_eval_,m-resnet_640_imagenet_vid_det_all-sampled_eq-batch_384-jtr-res_1440-fbb-zeg,imgn-vid_val-16_per_seq_random,batch-16,save-vis-0,dbg-0,dyn-1,dist-0,grs

