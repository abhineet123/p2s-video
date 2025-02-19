<!-- MarkdownTOC -->

- [swin-t](#swin_t_)
    - [ipsc-16_53-len-2       @ swin-t](#ipsc_16_53_len_2___swin_t_)
    - [ipsc-16_53-len-6       @ swin-t](#ipsc_16_53_len_6___swin_t_)
    - [gram-0_1       @ swin-t](#gram_0_1___swin_t_)
        - [len-2       @ gram-0_1/swin-t](#len_2___gram_0_1_swin_t_)
    - [mnist-640-1-12_1000       @ swin-t](#mnist_640_1_12_1000___swin_t_)
    - [len-2       @ swin-t](#len_2___swin_t_)
            - [on-test       @ len-2/swin-t](#on_test___len_2_swin_t_)
        - [len-3       @ len-2/swin-t](#len_3___len_2_swin_t_)
            - [on-train       @ len-3/len-2/swin-t](#on_train___len_3_len_2_swin_t_)
            - [on-test       @ len-3/len-2/swin-t](#on_test___len_3_len_2_swin_t_)
    - [mnist-640-5-12_1000       @ swin-t](#mnist_640_5_12_1000___swin_t_)
    - [len-2       @ swin-t](#len_2___swin_t__1)
    - [len-3       @ swin-t](#len_3___swin_t_)
    - [len-4       @ swin-t](#len_4___swin_t_)
- [swin-s](#swin_s_)
    - [mnist-640-1-12_1000       @ swin-s](#mnist_640_1_12_1000___swin_s_)
        - [len-2       @ mnist-640-1-12_1000/swin-s](#len_2___mnist_640_1_12_1000_swin_s_)
- [resnet-640-lfn](#resnet_640_lf_n_)
    - [gram-0_1       @ resnet-640-lfn](#gram_0_1___resnet_640_lfn_)
        - [len-9       @ gram-0_1/resnet-640-lfn](#len_9___gram_0_1_resnet_640_lf_n_)
            - [on-train       @ len-9/gram-0_1/resnet-640-lfn](#on_train___len_9_gram_0_1_resnet_640_lf_n_)
        - [len-14-0_2000       @ gram-0_1/resnet-640-lfn](#len_14_0_2000___gram_0_1_resnet_640_lf_n_)
            - [on-3000_5000       @ len-14-0_2000/gram-0_1/resnet-640-lfn](#on_3000_5000___len_14_0_2000_gram_0_1_resnet_640_lf_n_)
        - [len-16-0_2000       @ gram-0_1/resnet-640-lfn](#len_16_0_2000___gram_0_1_resnet_640_lf_n_)
    - [detrac-non_empty       @ resnet-640-lfn](#detrac_non_empty___resnet_640_lfn_)
        - [0_19-jtr       @ detrac-non_empty/resnet-640-lfn](#0_19_jtr___detrac_non_empty_resnet_640_lf_n_)
            - [on-train       @ 0_19-jtr/detrac-non_empty/resnet-640-lfn](#on_train___0_19_jtr_detrac_non_empty_resnet_640_lfn_)
                - [strd-2       @ on-train/0_19-jtr/detrac-non_empty/resnet-640-lfn](#strd_2___on_train_0_19_jtr_detrac_non_empty_resnet_640_lf_n_)
        - [0_19-len-9       @ detrac-non_empty/resnet-640-lfn](#0_19_len_9___detrac_non_empty_resnet_640_lf_n_)
    - [ipsc-16_53       @ resnet-640-lfn](#ipsc_16_53___resnet_640_lfn_)
        - [on-16_53       @ ipsc-16_53/resnet-640-lfn](#on_16_53___ipsc_16_53_resnet_640_lf_n_)
        - [on-54_126       @ ipsc-16_53/resnet-640-lfn](#on_54_126___ipsc_16_53_resnet_640_lf_n_)
    - [mnist-640-5-12_1000       @ resnet-640-lfn](#mnist_640_5_12_1000___resnet_640_lfn_)
        - [len-4       @ mnist-640-5-12_1000/resnet-640-lfn](#len_4___mnist_640_5_12_1000_resnet_640_lfn_)
        - [len-6       @ mnist-640-5-12_1000/resnet-640-lfn](#len_6___mnist_640_5_12_1000_resnet_640_lfn_)
        - [len-9       @ mnist-640-5-12_1000/resnet-640-lfn](#len_9___mnist_640_5_12_1000_resnet_640_lfn_)
            - [msl-2048       @ len-9/mnist-640-5-12_1000/resnet-640-lfn](#msl_2048___len_9_mnist_640_5_12_1000_resnet_640_lfn_)
- [resnet-640](#resnet_64_0_)
    - [detrac       @ resnet-640](#detrac___resnet_640_)
        - [0_19       @ detrac/resnet-640](#0_19___detrac_resnet_64_0_)
            - [on-train       @ 0_19/detrac/resnet-640](#on_train___0_19_detrac_resnet_640_)
                - [strd-1       @ on-train/0_19/detrac/resnet-640](#strd_1___on_train_0_19_detrac_resnet_64_0_)
                - [strd-2       @ on-train/0_19/detrac/resnet-640](#strd_2___on_train_0_19_detrac_resnet_64_0_)
            - [on-49_68       @ 0_19/detrac/resnet-640](#on_49_68___0_19_detrac_resnet_640_)
                - [strd-1       @ on-49_68/0_19/detrac/resnet-640](#strd_1___on_49_68_0_19_detrac_resnet_64_0_)
                - [strd-2       @ on-49_68/0_19/detrac/resnet-640](#strd_2___on_49_68_0_19_detrac_resnet_64_0_)
        - [0_9       @ detrac/resnet-640](#0_9___detrac_resnet_64_0_)
            - [on-train       @ 0_9/detrac/resnet-640](#on_train___0_9_detrac_resnet_64_0_)
                - [strd-2       @ on-train/0_9/detrac/resnet-640](#strd_2___on_train_0_9_detrac_resnet_640_)
            - [on-49_68       @ 0_9/detrac/resnet-640](#on_49_68___0_9_detrac_resnet_64_0_)
                - [strd-2       @ on-49_68/0_9/detrac/resnet-640](#strd_2___on_49_68_0_9_detrac_resnet_640_)
    - [ipsc-0_4       @ resnet-640](#ipsc_0_4___resnet_640_)
        - [batch-3       @ ipsc-0_4/resnet-640](#batch_3___ipsc_0_4_resnet_64_0_)
            - [on-g2_0_4       @ batch-3/ipsc-0_4/resnet-640](#on_g2_0_4___batch_3_ipsc_0_4_resnet_64_0_)
        - [batch-6       @ ipsc-0_4/resnet-640](#batch_6___ipsc_0_4_resnet_64_0_)
            - [on-g2_5_9       @ batch-6/ipsc-0_4/resnet-640](#on_g2_5_9___batch_6_ipsc_0_4_resnet_64_0_)
            - [on-g2_0_4       @ batch-6/ipsc-0_4/resnet-640](#on_g2_0_4___batch_6_ipsc_0_4_resnet_64_0_)
    - [ipsc-5_9       @ resnet-640](#ipsc_5_9___resnet_640_)
        - [batch-8       @ ipsc-5_9/resnet-640](#batch_8___ipsc_5_9_resnet_64_0_)
            - [on-g2_0_4       @ batch-8/ipsc-5_9/resnet-640](#on_g2_0_4___batch_8_ipsc_5_9_resnet_64_0_)
            - [on-5_9       @ batch-8/ipsc-5_9/resnet-640](#on_5_9___batch_8_ipsc_5_9_resnet_64_0_)
        - [fg-4       @ ipsc-5_9/resnet-640](#fg_4___ipsc_5_9_resnet_64_0_)
    - [ipsc-16_53-len-6       @ resnet-640](#ipsc_16_53_len_6___resnet_640_)
        - [on-train       @ ipsc-16_53-len-6/resnet-640](#on_train___ipsc_16_53_len_6_resnet_64_0_)
        - [on-54_126       @ ipsc-16_53-len-6/resnet-640](#on_54_126___ipsc_16_53_len_6_resnet_64_0_)
    - [ipsc-16_53-jtr-res-1280       @ resnet-640](#ipsc_16_53_jtr_res_1280___resnet_640_)
    - [ipsc-16_53-jtr-res-1280-len-6       @ resnet-640](#ipsc_16_53_jtr_res_1280_len_6___resnet_640_)
    - [ipsc-16_53       @ resnet-640](#ipsc_16_53___resnet_640_)
        - [on-train       @ ipsc-16_53/resnet-640](#on_train___ipsc_16_53_resnet_64_0_)
        - [on-54_126       @ ipsc-16_53/resnet-640](#on_54_126___ipsc_16_53_resnet_64_0_)
        - [on-0_15       @ ipsc-16_53/resnet-640](#on_0_15___ipsc_16_53_resnet_64_0_)
    - [ipsc-0_37       @ resnet-640](#ipsc_0_37___resnet_640_)
        - [on-54_126       @ ipsc-0_37/resnet-640](#on_54_126___ipsc_0_37_resnet_640_)
            - [strd-1       @ on-54_126/ipsc-0_37/resnet-640](#strd_1___on_54_126_ipsc_0_37_resnet_640_)
            - [strd-2       @ on-54_126/ipsc-0_37/resnet-640](#strd_2___on_54_126_ipsc_0_37_resnet_640_)
    - [ipsc-0_37-fg-4       @ resnet-640](#ipsc_0_37_fg_4___resnet_640_)
    - [mnist-640-1-12_1000       @ resnet-640](#mnist_640_1_12_1000___resnet_640_)
        - [len-2       @ mnist-640-1-12_1000/resnet-640](#len_2___mnist_640_1_12_1000_resnet_640_)
            - [on-train       @ len-2/mnist-640-1-12_1000/resnet-640](#on_train___len_2_mnist_640_1_12_1000_resnet_640_)
            - [on-test       @ len-2/mnist-640-1-12_1000/resnet-640](#on_test___len_2_mnist_640_1_12_1000_resnet_640_)
        - [len-3       @ mnist-640-1-12_1000/resnet-640](#len_3___mnist_640_1_12_1000_resnet_640_)
            - [swin-t       @ len-3/mnist-640-1-12_1000/resnet-640](#swin_t___len_3_mnist_640_1_12_1000_resnet_640_)
        - [len-9       @ mnist-640-1-12_1000/resnet-640](#len_9___mnist_640_1_12_1000_resnet_640_)
    - [mnist-640-5-12_1000       @ resnet-640](#mnist_640_5_12_1000___resnet_640_)
        - [len-2       @ mnist-640-5-12_1000/resnet-640](#len_2___mnist_640_5_12_1000_resnet_640_)
            - [on-train       @ len-2/mnist-640-5-12_1000/resnet-640](#on_train___len_2_mnist_640_5_12_1000_resnet_640_)
            - [on-test       @ len-2/mnist-640-5-12_1000/resnet-640](#on_test___len_2_mnist_640_5_12_1000_resnet_640_)
        - [len-9       @ mnist-640-5-12_1000/resnet-640](#len_9___mnist_640_5_12_1000_resnet_640_)

<!-- /MarkdownTOC -->
<a id="swin_t_"></a>
# swin-t 
<a id="ipsc_16_53_len_2___swin_t_"></a>
## ipsc-16_53-len-2       @ swin-t-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-16_53,len-2,strd-1,batch-6,dbg-0,dyn-1,dist-0,ep-10000,ckpt_ep-1,swin-t
<a id="ipsc_16_53_len_6___swin_t_"></a>
## ipsc-16_53-len-6       @ swin-t-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-16_53,len-2,strd-1,batch-3,dbg-0,dyn-1,dist-0,ep-10000,ckpt_ep-1,swin-t
<a id="gram_0_1___swin_t_"></a>
## gram-0_1       @ swin-t-->p2s_vid
<a id="len_2___gram_0_1_swin_t_"></a>
### len-2       @ gram-0_1/swin-t-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,gram-0_1,len-2,batch-3,dbg-1,dyn-1,dist-0,ep-10000,ckpt_ep-1,swin-t
<a id="mnist_640_1_12_1000___swin_t_"></a>
## mnist-640-1-12_1000       @ swin-t-->p2s_vid
<a id="len_2___swin_t_"></a>
## len-2       @ swin-t-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,mnist-640-1-12_1000-train,batch-4,dbg-0,dyn-1,dist-0,ep-10000,ckpt_ep-1,swin-t
<a id="on_test___len_2_swin_t_"></a>
#### on-test       @ len-2/swin-t-->p2s_vid
`strd-2`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py --j5=m-swin_t_640_mnist_640_1_12_1000_var-length-2-stride-1-batch_4,_eval_,vid_det,mnist-640-1-12_1000-test,len-2,strd-2,batch-36,save-vis-1,dbg-1,dyn-1,dist-0

<a id="len_3___len_2_swin_t_"></a>
### len-3       @ len-2/swin-t-->p2s_vid
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,mnist-640-1-12_1000,len-3,batch-4,dbg-1,dyn-1,dist-0,ep-10000,ckpt_ep-1,swin-t
<a id="on_train___len_3_len_2_swin_t_"></a>
#### on-train       @ len-3/len-2/swin-t-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=m-swin_t_640_mnist_640_1_12_1000_var-length-3-stride-1-batch_4,_eval_,vid_det,mnist-640-1-12_1000-train,batch-32,save-vis-1,dbg-0,dyn-1,dist-0
<a id="on_test___len_3_len_2_swin_t_"></a>
#### on-test       @ len-3/len-2/swin-t-->p2s_vid
`strd-3`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=m-swin_t_640_mnist_640_1_12_1000_var-length-3-stride-1-batch_4,_eval_,vid_det,mnist-640-1-12_1000-test,len-3,strd-3,batch-36,save-vis-1,dbg-1,dyn-1,dist-0

<a id="mnist_640_5_12_1000___swin_t_"></a>
## mnist-640-5-12_1000       @ swin-t-->p2s_vid
<a id="len_2___swin_t__1"></a>
## len-2       @ swin-t-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,mnist-640-5-12_1000,batch-8,dbg-0,dyn-1,dist-1,ep-10000,ckpt_ep-1,swin-t
<a id="len_3___swin_t_"></a>
## len-3       @ swin-t-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,mnist-640-5-12_1000,batch-3,dbg-1,dyn-1,dist-0,ep-10000,ckpt_ep-1,swin-t
<a id="len_4___swin_t_"></a>
## len-4       @ swin-t-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,mnist-640-5-12_1000,batch-3,dbg-1,dyn-1,dist-0,ep-10000,ckpt_ep-1,swin-t

<a id="swin_s_"></a>
# swin-s 
<a id="mnist_640_1_12_1000___swin_s_"></a>
## mnist-640-1-12_1000       @ swin-s-->p2s_vid
<a id="len_2___mnist_640_1_12_1000_swin_s_"></a>
### len-2       @ mnist-640-1-12_1000/swin-s-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-0,mnist-640-1-12_1000,batch-3,dbg-1,dyn-1,ep-10000,ckpt_ep-1,swin-s

<a id="resnet_640_lf_n_"></a>
# resnet-640-lfn 
<a id="gram_0_1___resnet_640_lfn_"></a>
## gram-0_1       @ resnet-640-lfn-->p2s_vid
<a id="len_9___gram_0_1_resnet_640_lf_n_"></a>
### len-9       @ gram-0_1/resnet-640-lfn-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,gram-0_1,len-9,batch-2,dbg-0,dyn-1,dist-0,ep-10000,ckpt_ep-1,lfn
<a id="on_train___len_9_gram_0_1_resnet_640_lf_n_"></a>
#### on-train       @ len-9/gram-0_1/resnet-640-lfn-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=vid_det,m-resnet_640_gram_only-length-9-stride-1-seq-0_1-batch_2-lfn,_eval_,gram-0_1,len-9,strd-9,batch-2,save-vis-1,dbg-0,dyn-1,dist-0

<a id="len_14_0_2000___gram_0_1_resnet_640_lf_n_"></a>
### len-14-0_2000       @ gram-0_1/resnet-640-lfn-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,gram-0_1-0_2000,len-14,batch-2,dbg-0,dyn-1,dist-1,ep-10000,ckpt_ep-1,lfn
<a id="on_3000_5000___len_14_0_2000_gram_0_1_resnet_640_lf_n_"></a>
#### on-3000_5000       @ len-14-0_2000/gram-0_1/resnet-640-lfn-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=vid_det,m-resnet_640_gram_only-0_2000-length-14-stride-1-seq-0_1-batch_2-lfn,_eval_,gram-0_1-3000_5000,len-14,strd-14,batch-1,save-vis-1,dbg-0,dyn-1,dist-0

<a id="len_16_0_2000___gram_0_1_resnet_640_lf_n_"></a>
### len-16-0_2000       @ gram-0_1/resnet-640-lfn-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,gram-0_1-0_2000,len-16,batch-2,dbg-0,dyn-1,dist-1,ep-10000,ckpt_ep-1,lfn

<a id="detrac_non_empty___resnet_640_lfn_"></a>
## detrac-non_empty       @ resnet-640-lfn-->p2s_vid
<a id="0_19_jtr___detrac_non_empty_resnet_640_lf_n_"></a>
### 0_19-jtr       @ detrac-non_empty/resnet-640-lfn-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,detrac-non_empty-0_19,len-6,batch-6,dbg-0,dyn-1,dist-1,ep-10000,ckpt_ep-1,lfn,jtr
<a id="on_train___0_19_jtr_detrac_non_empty_resnet_640_lfn_"></a>
#### on-train       @ 0_19-jtr/detrac-non_empty/resnet-640-lfn-->p2s_vid
<a id="strd_2___on_train_0_19_jtr_detrac_non_empty_resnet_640_lf_n_"></a>
##### strd-2       @ on-train/0_19-jtr/detrac-non_empty/resnet-640-lfn-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_19-batch_18,detrac-non_empty-0_19,strd-2,batch-12,save-vis-1,dbg-0,dyn-1,dist-0

<a id="0_19_len_9___detrac_non_empty_resnet_640_lf_n_"></a>
### 0_19-len-9       @ detrac-non_empty/resnet-640-lfn-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,detrac-non_empty-0_19,len-9,batch-2,dbg-0,dyn-1,dist-1,ep-10000,ckpt_ep-1,lfn

<a id="ipsc_16_53___resnet_640_lfn_"></a>
## ipsc-16_53       @ resnet-640-lfn-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-16_53,len-3,strd-1,batch-6,dbg-0,dyn-1,dist-0,ep-10000,ckpt_ep-1,lfn
<a id="on_16_53___ipsc_16_53_resnet_640_lf_n_"></a>
### on-16_53       @ ipsc-16_53/resnet-640-lfn-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-3-stride-1-batch_6-lfn,_eval_,vid_det,ipsc-16_53,len-3,strd-3,batch-24,save-vis-1,dbg-0,dyn-1
<a id="on_54_126___ipsc_16_53_resnet_640_lf_n_"></a>
### on-54_126       @ ipsc-16_53/resnet-640-lfn-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-3-stride-1-batch_6-lfn,_eval_,vid_det,ipsc-54_126,len-3,strd-3,batch-24,save-vis-1,dbg-0,dyn-1


<a id="mnist_640_5_12_1000___resnet_640_lfn_"></a>
## mnist-640-5-12_1000       @ resnet-640-lfn-->p2s_vid
<a id="len_4___mnist_640_5_12_1000_resnet_640_lfn_"></a>
### len-4       @ mnist-640-5-12_1000/resnet-640-lfn-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,mnist-640-5-12_1000-train,len-4,strd-1,batch-4,dbg-0,dyn-1,dist-0,ep-10000,ckpt_ep-1,lfn,jtr
<a id="len_6___mnist_640_5_12_1000_resnet_640_lfn_"></a>
### len-6       @ mnist-640-5-12_1000/resnet-640-lfn-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,mnist-640-5-12_1000-train,len-6,strd-1,batch-3,dbg-0,dyn-1,dist-0,ep-10000,ckpt_ep-1,lfn
<a id="len_9___mnist_640_5_12_1000_resnet_640_lfn_"></a>
### len-9       @ mnist-640-5-12_1000/resnet-640-lfn-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,mnist-640-5-12_1000-train,len-9,strd-1,batch-2,dbg-0,dyn-1,dist-0,ep-10000,ckpt_ep-1,lfn
<a id="msl_2048___len_9_mnist_640_5_12_1000_resnet_640_lfn_"></a>
#### msl-2048       @ len-9/mnist-640-5-12_1000/resnet-640-lfn-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-0,mnist-640-5-12_1000-train,len-9,strd-1,batch-2,dbg-0,dyn-1,dist-0,ep-10000,ckpt_ep-1,lfn,msl-2048


<a id="resnet_64_0_"></a>
# resnet-640 
<a id="detrac___resnet_640_"></a>
## detrac       @ resnet-640-->p2s_vid
<a id="0_19___detrac_resnet_64_0_"></a>
### 0_19       @ detrac/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,detrac-non_empty-0_19,batch-3,dbg-0,dyn-1,dist-1,ep-10000,ckpt_ep-1

<a id="on_train___0_19_detrac_resnet_640_"></a>
#### on-train       @ 0_19/detrac/resnet-640-->p2s_vid
<a id="strd_1___on_train_0_19_detrac_resnet_64_0_"></a>
##### strd-1       @ on-train/0_19/detrac/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_19-batch_18,detrac-non_empty-0_19,batch-48,save-vis-1,dbg-0,dyn-1,dist-0
<a id="strd_2___on_train_0_19_detrac_resnet_64_0_"></a>
##### strd-2       @ on-train/0_19/detrac/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_19-batch_18,detrac-non_empty-0_19,strd-2,batch-12,save-vis-1,dbg-0,dyn-1,dist-0

<a id="on_49_68___0_19_detrac_resnet_640_"></a>
#### on-49_68       @ 0_19/detrac/resnet-640-->p2s_vid
<a id="strd_1___on_49_68_0_19_detrac_resnet_64_0_"></a>
##### strd-1       @ on-49_68/0_19/detrac/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_19-batch_18,detrac-non_empty-49_68,batch-48,save-vis-1,dbg-0,dyn-1,dist-0
<a id="strd_2___on_49_68_0_19_detrac_resnet_64_0_"></a>
##### strd-2       @ on-49_68/0_19/detrac/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_19-batch_18,eval,vid_det,detrac-non_empty-49_68,strd-2,batch-12,save-vis-1,dbg-0,dyn-1,dist-0

<a id="0_9___detrac_resnet_64_0_"></a>
### 0_9       @ detrac/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,detrac-non_empty-0_9,batch-18,dbg-0,dyn-1,dist-1,ep-10000,ckpt_ep-1
<a id="on_train___0_9_detrac_resnet_64_0_"></a>
#### on-train       @ 0_9/detrac/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_9-batch_18,detrac-non_empty-0_9,batch-48,save-vis-1,dbg-0,dyn-1,dist-0
<a id="strd_2___on_train_0_9_detrac_resnet_640_"></a>
##### strd-2       @ on-train/0_9/detrac/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_9-batch_18,detrac-non_empty-0_9,strd-2,batch-48,save-vis-1,dbg-0,dyn-1,dist-0
<a id="on_49_68___0_9_detrac_resnet_64_0_"></a>
#### on-49_68       @ 0_9/detrac/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_9-batch_18,detrac-non_empty-49_68,batch-1,save-vis-1,dbg-0,dyn-1,dist-0
<a id="strd_2___on_49_68_0_9_detrac_resnet_640_"></a>
##### strd-2       @ on-49_68/0_9/detrac/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=m-resnet_640_detrac-length-2-stride-1-non_empty-seq-0_9-batch_18,eval,vid_det,detrac-non_empty-49_68,strd-2,batch-48,save-vis-1,dbg-0,dyn-1,dist-0

<a id="ipsc_0_4___resnet_640_"></a>
## ipsc-0_4       @ resnet-640-->p2s_vid
<a id="batch_3___ipsc_0_4_resnet_64_0_"></a>
### batch-3       @ ipsc-0_4/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-0,ipsc-g2_0_4,batch-3,dbg-1,dyn-1,ep-4000
<a id="on_g2_0_4___batch_3_ipsc_0_4_resnet_64_0_"></a>
#### on-g2_0_4       @ batch-3/ipsc-0_4/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_ext_reorg_roi_g2_0_4-length-2-stride-1-batch_3,ipsc-g2_0_4,batch-1,save-vis-1,dbg-1,dyn-1

<a id="batch_6___ipsc_0_4_resnet_64_0_"></a>
### batch-6       @ ipsc-0_4/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-0,ipsc-g2_0_4,batch-6,dbg-0,dyn-1,ep-4000,dist-1
<a id="on_g2_5_9___batch_6_ipsc_0_4_resnet_64_0_"></a>
#### on-g2_5_9       @ batch-6/ipsc-0_4/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=eval,vid_det,m-resnet_640_ext_reorg_roi_g2_0_4-length-2-stride-1-batch_6,ipsc-g2_5_9,batch-3,save-vis-1,dbg-1,dyn-1
<a id="on_g2_0_4___batch_6_ipsc_0_4_resnet_64_0_"></a>
#### on-g2_0_4       @ batch-6/ipsc-0_4/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_ext_reorg_roi_g2_0_4-length-2-stride-1-batch_6,ipsc-g2_0_4,batch-1,save-vis-1,dbg-1,dyn-1

<a id="ipsc_5_9___resnet_640_"></a>
## ipsc-5_9       @ resnet-640-->p2s_vid
<a id="batch_8___ipsc_5_9_resnet_64_0_"></a>
### batch-8       @ ipsc-5_9/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-g2_5_9,batch-3,dbg-1,dyn-1,ep-1000000,dist-0,ckpt_ep-20
<a id="on_g2_0_4___batch_8_ipsc_5_9_resnet_64_0_"></a>
#### on-g2_0_4       @ batch-8/ipsc-5_9/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py  --j5=eval,vid_det,m-resnet_640_ext_reorg_roi_g2_5_9-length-2-stride-1-batch_8,ipsc-g2_0_4,batch-4,save-vis-1,dbg-1,dyn-1
**12094**
python3 run.py --cfg=configs/config_video_det.py  --j5=eval,vid_det,m-resnet_640_ext_reorg_roi_g2_5_9-length-2-stride-1-batch_8,ipsc-12094_0_4,batch-4,save-vis-1,dbg-1,dyn-1
**12094_short**
python3 run.py --cfg=configs/config_video_det.py  --j5=eval,vid_det,m-resnet_640_ext_reorg_roi_g2_5_9-length-2-stride-1-batch_8,ipsc-12094_short_0_4,batch-4,save-vis-1,dbg-1,dyn-1
<a id="on_5_9___batch_8_ipsc_5_9_resnet_64_0_"></a>
#### on-5_9       @ batch-8/ipsc-5_9/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py  --j5=eval,vid_det,m-resnet_640_ext_reorg_roi_g2_5_9-length-2-stride-1-batch_8,ipsc-g2_5_9,batch-4,save-vis-1,dbg-1,dyn-1

<a id="fg_4___ipsc_5_9_resnet_64_0_"></a>
### fg-4       @ ipsc-5_9/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-g2_5_9,batch-4,dbg-1,dyn-1,ep-10000,dist-1,ckpt_ep-20,fg-4

<a id="ipsc_16_53_len_6___resnet_640_"></a>
## ipsc-16_53-len-6       @ resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-16_53,len-6,strd-1,batch-3,dbg-0,dyn-1,dist-0,ep-10000,ckpt_ep-1
<a id="on_train___ipsc_16_53_len_6_resnet_64_0_"></a>
### on-train       @ ipsc-16_53-len-6/resnet-640-->p2s_vid
<a id="strd_1___on_54_126_ipsc_16_53_resnet_64_0_"></a>
`strd-1` 
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-6-stride-1-batch_3,_eval_,vid_det,ipsc-16_53,len-6,strd-1,batch-4,save-vis-1,dbg-0,dyn-1
<a id="strd_2___on_54_126_ipsc_16_53_resnet_64_0_"></a>
`strd-6` 
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-6-stride-1-batch_3,_eval_,vid_det,ipsc-16_53,len-6,strd-6,batch-4,save-vis-1,dbg-0,dyn-1

<a id="on_54_126___ipsc_16_53_len_6_resnet_64_0_"></a>
### on-54_126       @ ipsc-16_53-len-6/resnet-640-->p2s_vid
<a id="strd_1___on_54_126_ipsc_16_53_resnet_64_0_"></a>
`strd-1` 
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-6-stride-1-batch_3,_eval_,vid_det,ipsc-54_126,len-6,strd-1,batch-12,save-vis-1,dbg-0,dyn-1
<a id="strd_2___on_54_126_ipsc_16_53_resnet_64_0_"></a>
`strd-6` 
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-6-stride-1-batch_3,_eval_,vid_det,ipsc-54_126,len-6,strd-6,batch-12,save-vis-1,dbg-0,dyn-1

<a id="ipsc_16_53_jtr_res_1280___resnet_640_"></a>
## ipsc-16_53-jtr-res-1280       @ resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=_val_,ipsc-54_126,batch-2,len-2,strd-2,sample-8,_train_,resnet-640,vid_det,pt-1,ipsc-16_53,len-2,strd-1,batch-18,dbg-0,dyn-1,dist-1,ep-10000,ckpt_ep-1,jtr,res-1280

<a id="ipsc_16_53_jtr_res_1280_len_6___resnet_640_"></a>
## ipsc-16_53-jtr-res-1280-len-6       @ resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=_val_,ipsc-54_126,batch-2,len-6,strd-6,sample-4,_train_,resnet-640,vid_det,pt-1,ipsc-16_53,len-6,strd-1,batch-6,dbg-0,dyn-1,dist-1,ep-10000,ckpt_ep-1,jtr,res-1280

<a id="ipsc_16_53___resnet_640_"></a>
## ipsc-16_53       @ resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-16_53,len-2,strd-1,batch-18,dbg-0,dyn-1,dist-1,ep-10000,ckpt_ep-1
<a id="on_train___ipsc_16_53_resnet_64_0_"></a>
### on-train       @ ipsc-16_53/resnet-640-->p2s_vid
`strd-1` 
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-2-stride-1-batch_18,_eval_,vid_det,ipsc-16_53,len-2,strd-1,batch-18,save-vis-1,dbg-0,dyn-1
`strd-2` 
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-2-stride-1-batch_18,_eval_,vid_det,ipsc-16_53,len-2,strd-2,batch-18,save-vis-1,dbg-0,dyn-1

<a id="on_54_126___ipsc_16_53_resnet_64_0_"></a>
### on-54_126       @ ipsc-16_53/resnet-640-->p2s_vid
`strd-1` 
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-2-stride-1-batch_18,_eval_,vid_det,ipsc-54_126,len-2,strd-1,batch-36,save-vis-1,dbg-0,dyn-1
`strd-2` 
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-2-stride-1-batch_18,_eval_,vid_det,ipsc-54_126,len-2,strd-2,batch-36,save-vis-1,dbg-0,dyn-1

<a id="on_0_15___ipsc_16_53_resnet_64_0_"></a>
### on-0_15       @ ipsc-16_53/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-16_53-length-2-stride-1-batch_18,_eval_,vid_det,ipsc-0_15,len-2,strd-1,batch-36,save-vis-1,dbg-0,dyn-1

<a id="ipsc_0_37___resnet_640_"></a>
## ipsc-0_37       @ resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-0_37,len-2,strd-1,batch-18,dbg-0,dyn-1,dist-1,ep-10000,ckpt_ep-1
<a id="on_54_126___ipsc_0_37_resnet_640_"></a>
### on-54_126       @ ipsc-0_37/resnet-640-->p2s_vid
<a id="strd_1___on_54_126_ipsc_0_37_resnet_640_"></a>
#### strd-1       @ on-54_126/ipsc-0_37/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-0_37-length-2-stride-1-batch_18,_eval_,vid_det,ipsc-54_126,len-2,strd-1,batch-36,save-vis-1,dbg-0,dyn-1
<a id="strd_2___on_54_126_ipsc_0_37_resnet_640_"></a>
#### strd-2       @ on-54_126/ipsc-0_37/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py  --j5=m-resnet_640_ext_reorg_roi_g2-0_37-length-2-stride-1-batch_18,_eval_,vid_det,ipsc-54_126,len-2,strd-2,batch-36,save-vis-1,dbg-0,dyn-1

<a id="ipsc_0_37_fg_4___resnet_640_"></a>
## ipsc-0_37-fg-4       @ resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,ipsc-g2_0_37,len-2,strd-1,fg-4,batch-16,dbg-0,dyn-1,dist-1,ep-10000,ckpt_ep-1

<a id="mnist_640_1_12_1000___resnet_640_"></a>
## mnist-640-1-12_1000       @ resnet-640-->p2s_vid
<a id="len_2___mnist_640_1_12_1000_resnet_640_"></a>
### len-2       @ mnist-640-1-12_1000/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-0,mnist-640-1-12_1000,batch-3,dbg-1,dyn-1,ep-10000,ckpt_ep-1
<a id="on_train___len_2_mnist_640_1_12_1000_resnet_640_"></a>
#### on-train       @ len-2/mnist-640-1-12_1000/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_mnist_640_1_12_1000_var-length-2-stride-1-train-batch_18,mnist-640-1-12_1000,batch-1,save-vis-1,dbg-1,dyn-1,dist-0,suffix-train
<a id="on_test___len_2_mnist_640_1_12_1000_resnet_640_"></a>
#### on-test       @ len-2/mnist-640-1-12_1000/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_mnist_640_1_12_1000_var-length-2-stride-1-train-batch_18,mnist-640-1-12_1000,batch-48,save-vis-1,dbg-1,dyn-1,dist-0,suffix-test
`strd-2`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_mnist_640_1_12_1000_var-length-2-stride-1-train-batch_18,mnist-640-1-12_1000,strd-2,batch-48,save-vis-1,dbg-0,dyn-1,dist-0,suffix-test

<a id="len_3___mnist_640_1_12_1000_resnet_640_"></a>
### len-3       @ mnist-640-1-12_1000/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,mnist-640-1-12_1000,len-3,batch-12,dbg-0,dyn-1,dist-1,ep-10000,ckpt_ep-1
<a id="swin_t___len_3_mnist_640_1_12_1000_resnet_640_"></a>
#### swin-t       @ len-3/mnist-640-1-12_1000/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,mnist-640-1-12_1000,len-3,batch-4,dbg-1,dyn-1,dist-0,ep-10000,ckpt_ep-1,swin-t

<a id="len_9___mnist_640_1_12_1000_resnet_640_"></a>
### len-9       @ mnist-640-1-12_1000/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,mnist-640-1-12_1000,len-9,batch-4,dbg-0,dyn-1,dist-1,ep-10000,ckpt_ep-1

<a id="mnist_640_5_12_1000___resnet_640_"></a>
## mnist-640-5-12_1000       @ resnet-640-->p2s_vid
<a id="len_2___mnist_640_5_12_1000_resnet_640_"></a>
### len-2       @ mnist-640-5-12_1000/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,mnist-640-5-12_1000,batch-18,dbg-0,dyn-1,dist-1,ep-10000,ckpt_ep-1
<a id="on_train___len_2_mnist_640_5_12_1000_resnet_640_"></a>
#### on-train       @ len-2/mnist-640-5-12_1000/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_mnist_640_5_12_1000_var-length-2-stride-1-train-batch_18,mnist-640-5-12_1000,batch-36,save-vis-1,dbg-0,dyn-1,dist-0,suffix-train
<a id="on_test___len_2_mnist_640_5_12_1000_resnet_640_"></a>
#### on-test       @ len-2/mnist-640-5-12_1000/resnet-640-->p2s_vid
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_mnist_640_5_12_1000_var-length-2-stride-1-train-batch_18,mnist-640-5-12_1000,batch-36,save-vis-1,dbg-0,dyn-1,dist-0,suffix-test
`strd-2`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_video_det.py --j5=eval,vid_det,m-resnet_640_mnist_640_5_12_1000_var-length-2-stride-1-train-batch_18,mnist-640-5-12_1000,strd-2,batch-48,save-vis-1,dbg-0,dyn-1,dist-0,suffix-test

<a id="len_9___mnist_640_5_12_1000_resnet_640_"></a>
### len-9       @ mnist-640-5-12_1000/resnet-640-->p2s_vid
python3 run.py --cfg=configs/config_video_det.py --j5=train,resnet-640,vid_det,pt-1,mnist-640-5-12_1000,len-9,batch-4,dbg-0,dyn-1,dist-1,ep-10000,ckpt_ep-1
