<!-- MarkdownTOC -->

- [train](#train_)
    - [p-1024-sub-8-lac-fbb       @ train](#p_1024_sub_8_lac_fbb___trai_n_)
    - [p-1024-sub-4-lac-bac-fbb       @ train](#p_1024_sub_4_lac_bac_fbb___trai_n_)
    - [p-1024-res-640-sub-4-lac-bac-fbb       @ train](#p_1024_res_640_sub_4_lac_bac_fbb___trai_n_)
        - [on-val-put       @ p-1024-res-640-sub-4-lac-bac-fbb/train](#on_val_put___p_1024_res_640_sub_4_lac_bac_fbb_train_)
        - [on-val       @ p-1024-res-640-sub-4-lac-bac-fbb/train](#on_val___p_1024_res_640_sub_4_lac_bac_fbb_train_)
    - [r-640-sub-8-dm2-fbb       @ train](#r_640_sub_8_dm2_fbb___trai_n_)
        - [on-val-put       @ r-640-sub-8-dm2-fbb/train](#on_val_put___r_640_sub_8_dm2_fbb_trai_n_)
        - [on-val       @ r-640-sub-8-dm2-fbb/train](#on_val___r_640_sub_8_dm2_fbb_trai_n_)
        - [on-train       @ r-640-sub-8-dm2-fbb/train](#on_train___r_640_sub_8_dm2_fbb_trai_n_)
    - [r-640-sub-8-lac-fbb       @ train](#r_640_sub_8_lac_fbb___trai_n_)
        - [hp8470p_put       @ r-640-sub-8-lac-fbb/train](#hp8470p_put___r_640_sub_8_lac_fbb_trai_n_)
        - [on-val       @ r-640-sub-8-lac-fbb/train](#on_val___r_640_sub_8_lac_fbb_trai_n_)
        - [on-train       @ r-640-sub-8-lac-fbb/train](#on_train___r_640_sub_8_lac_fbb_trai_n_)
    - [r-1280_640-p-640-sub-8-lac       @ train](#r_1280_640_p_640_sub_8_lac___trai_n_)
    - [r-1280_640-p-640-sub-8-lac-fbb       @ train](#r_1280_640_p_640_sub_8_lac_fbb___trai_n_)
        - [on-val-put       @ r-1280_640-p-640-sub-8-lac-fbb/train](#on_val_put___r_1280_640_p_640_sub_8_lac_fbb_train_)
        - [on-val       @ r-1280_640-p-640-sub-8-lac-fbb/train](#on_val___r_1280_640_p_640_sub_8_lac_fbb_train_)
        - [on-train       @ r-1280_640-p-640-sub-8-lac-fbb/train](#on_train___r_1280_640_p_640_sub_8_lac_fbb_train_)
    - [r-1280_640-p-640-sub-5-lac       @ train](#r_1280_640_p_640_sub_5_lac___trai_n_)
        - [on-val-put       @ r-1280_640-p-640-sub-5-lac/train](#on_val_put___r_1280_640_p_640_sub_5_lac_train_)
        - [on-val       @ r-1280_640-p-640-sub-5-lac/train](#on_val___r_1280_640_p_640_sub_5_lac_train_)
        - [on-train       @ r-1280_640-p-640-sub-5-lac/train](#on_train___r_1280_640_p_640_sub_5_lac_train_)
    - [r-1280_640-p-640-sub-5-lac-fbb       @ train](#r_1280_640_p_640_sub_5_lac_fbb___trai_n_)
        - [on-val-put       @ r-1280_640-p-640-sub-5-lac-fbb/train](#on_val_put___r_1280_640_p_640_sub_5_lac_fbb_train_)
        - [on-val       @ r-1280_640-p-640-sub-5-lac-fbb/train](#on_val___r_1280_640_p_640_sub_5_lac_fbb_train_)
        - [on-train       @ r-1280_640-p-640-sub-5-lac-fbb/train](#on_train___r_1280_640_p_640_sub_5_lac_fbb_train_)
    - [r-1280_640-p-640-sub-8-lac-bac-fbb       @ train](#r_1280_640_p_640_sub_8_lac_bac_fbb___trai_n_)
        - [hp8470p_put       @ r-1280_640-p-640-sub-8-lac-bac-fbb/train](#hp8470p_put___r_1280_640_p_640_sub_8_lac_bac_fbb_train_)
    - [r-1280_640-p-640-sub-2-lac-fbb       @ train](#r_1280_640_p_640_sub_2_lac_fbb___trai_n_)
    - [r-1280_640-p-640-sub-2-lac-2d-fbb       @ train](#r_1280_640_p_640_sub_2_lac_2d_fbb___trai_n_)
- [train-rfm](#train_rfm_)
    - [p-1024-sub-8-lac-fbb       @ train-rfm](#p_1024_sub_8_lac_fbb___train_rf_m_)
        - [on-train       @ p-1024-sub-8-lac-fbb/train-rfm](#on_train___p_1024_sub_8_lac_fbb_train_rfm_)
        - [on-val       @ p-1024-sub-8-lac-fbb/train-rfm](#on_val___p_1024_sub_8_lac_fbb_train_rfm_)
    - [p-1024-sub-8-lac-2d-fbb       @ train-rfm](#p_1024_sub_8_lac_2d_fbb___train_rf_m_)
    - [p-1024-sub-8-mhd-fbb       @ train-rfm](#p_1024_sub_8_mhd_fbb___train_rf_m_)
    - [r-1280_640       @ train-rfm](#r_1280_640___train_rf_m_)
        - [p-640-sub-8-mhd-fbb       @ r-1280_640/train-rfm](#p_640_sub_8_mhd_fbb___r_1280_640_train_rfm_)
            - [on-val       @ p-640-sub-8-mhd-fbb/r-1280_640/train-rfm](#on_val___p_640_sub_8_mhd_fbb_r_1280_640_train_rfm_)
        - [p-640-sub-8-mhd-1241-fbb       @ r-1280_640/train-rfm](#p_640_sub_8_mhd_1241_fbb___r_1280_640_train_rfm_)
            - [on-val       @ p-640-sub-8-mhd-1241-fbb/r-1280_640/train-rfm](#on_val___p_640_sub_8_mhd_1241_fbb_r_1280_640_train_rf_m_)
        - [p-640-sub-8-mhd-1241-fbb-b64       @ r-1280_640/train-rfm](#p_640_sub_8_mhd_1241_fbb_b64___r_1280_640_train_rfm_)
            - [on-val       @ p-640-sub-8-mhd-1241-fbb-b64/r-1280_640/train-rfm](#on_val___p_640_sub_8_mhd_1241_fbb_b64_r_1280_640_train_rf_m_)
        - [p-640-sub-8-mhd-1241-fbb-b54-zeg       @ r-1280_640/train-rfm](#p_640_sub_8_mhd_1241_fbb_b54_zeg___r_1280_640_train_rfm_)
            - [on-val       @ p-640-sub-8-mhd-1241-fbb-b54-zeg/r-1280_640/train-rfm](#on_val___p_640_sub_8_mhd_1241_fbb_b54_zeg_r_1280_640_train_rf_m_)
        - [p-640-sub-8-mhd-1241-fbb-b16       @ r-1280_640/train-rfm](#p_640_sub_8_mhd_1241_fbb_b16___r_1280_640_train_rfm_)
        - [p-640-sub-8-mhd-1241-fbb-b9       @ r-1280_640/train-rfm](#p_640_sub_8_mhd_1241_fbb_b9___r_1280_640_train_rfm_)
        - [p-640-sub-8-mhd-fbb-no_aug       @ r-1280_640/train-rfm](#p_640_sub_8_mhd_fbb_no_aug___r_1280_640_train_rfm_)
            - [on-val       @ p-640-sub-8-mhd-fbb-no_aug/r-1280_640/train-rfm](#on_val___p_640_sub_8_mhd_fbb_no_aug_r_1280_640_train_rf_m_)
        - [p-640-sub-4-mhd-fbb       @ r-1280_640/train-rfm](#p_640_sub_4_mhd_fbb___r_1280_640_train_rfm_)

<!-- /MarkdownTOC -->

<a id="train_"></a>
# train
<a id="p_1024_sub_8_lac_fbb___trai_n_"></a>
## p-1024-sub-8-lac-fbb       @ train-->p2s_seg-ctscp
python3 run.py --cfg=configs/config_seg.py  --j5=_train_,resnet-1024,ctscp-train,seg-p-1024:sub-8,lac,batch-32,dbg-0,dyn-1,dist-2,pt-1,seq3k,voc20,fbb,gdez
`single gpu`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=_train_,resnet-1024,ctscp-train,seg-p-1024:sub-8,lac,batch-4,dbg-0,dyn-1,dist-0,pt-1,seq3k,voc20,fbb

<a id="p_1024_sub_4_lac_bac_fbb___trai_n_"></a>
## p-1024-sub-4-lac-bac-fbb       @ train-->p2s_seg-ctscp
`single gpu`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=_train_,resnet-1024,ctscp-train,seg-p-1024:sub-4,lac,bac,batch-2,dbg-0,dyn-1,dist-0,pt-1,seq4k,voc5248,fbb

<a id="p_1024_res_640_sub_4_lac_bac_fbb___trai_n_"></a>
## p-1024-res-640-sub-4-lac-bac-fbb       @ train-->p2s_seg-ctscp
python3 run.py --cfg=configs/config_seg.py  --j5=_train_,resnet-640,ctscp-train,res-640,seg-p-1024:sub-4,lac,bac,batch-24,dbg-0,dyn-1,dist-2,pt-1,seq4k,voc5248,fbb,gdez
`single gpu`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=_train_,resnet-640,ctscp-train,res-640,seg-p-1024:sub-4,lac,bac,batch-4,dbg-0,dyn-1,dist-0,pt-1,seq4k,voc5248,fbb
```
watch tail -1 log/seg/resnet_640_ctscp-train-1024_1024-1024_1024-sub_4-bac-lac-res_640-batch_24-seq4k-voc5248-fbb-gdez/progress_log.txt
```
<a id="on_val_put___p_1024_res_640_sub_4_lac_bac_fbb_train_"></a>
### on-val-put       @ p-1024-res-640-sub-4-lac-bac-fbb/train-->p2s_seg-ctscp
`hp8470p_put`
CUDA_VISIBLE_DEVICES= python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_ctscp-train-1024_1024-1024_1024-sub_4-bac-lac-res_640-batch_24-seq4k-voc5248-fbb-gdez,_eval_,ctscp-val,batch-2,save-vis-0,dbg-0,dyn-1,seg-p-1024:sub-4,no_vid,logits,hp8470p_put-6
<a id="on_val___p_1024_res_640_sub_4_lac_bac_fbb_train_"></a>
### on-val       @ p-1024-res-640-sub-4-lac-bac-fbb/train-->p2s_seg-ctscp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-sub_5-lac-batch_32-seq3k-voc19b-gdez,_eval_,ctscp-val,batch-16,save-vis-0,dbg-0,dyn-1,seg-r-1280_640:p-640:sub-5,no_vid,logits,8470p


<a id="r_640_sub_8_dm2_fbb___trai_n_"></a>
## r-640-sub-8-dm2-fbb       @ train-->p2s_seg-ctscp
NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=enp6s0 python3 run.py --cfg=configs/config_seg.py  --j5=_train_,resnet-640,ctscp-train,seg-r-640:sub-8,dm2,batch-48,dbg-0,dyn-1,dist-2,pt-1,seq3k,voc7k,fbb,gdez
NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=eno1 python3 run.py --cfg=configs/config_seg.py  --j5=_train_,resnet-640,ctscp-train,seg-r-640:sub-8,dm2,batch-48,dbg-0,dyn-1,dist-2,pt-1,seq3k,voc7k,fbb,gdez
NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=enp0s25 python3 run.py --cfg=configs/config_seg.py  --j5=_train_,resnet-640,ctscp-train,seg-r-640:sub-8,dm2,batch-48,dbg-0,dyn-1,dist-2,pt-1,seq3k,voc7k,fbb,gdez
NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=enp13s0 python3 run.py --cfg=configs/config_seg.py  --j5=_train_,resnet-640,ctscp-train,seg-r-640:sub-8,dm2,batch-48,dbg-0,dyn-1,dist-2,pt-1,seq3k,voc7k,fbb,gdez
```
watch tail -1 log/seg/resnet_640_ctscp-train-resize_640-sub_8-dm2-mc-batch_48-seq3k-voc7k-fbb-gdez/progress_log.txt                                                            
```
`single gpu`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=_train_,resnet-640,ctscp-train,seg-r-640:sub-8,dm2,batch-6,dbg-0,dyn-1,dist-0,pt-1,seq3k,voc7k,fbb
`dbg`
python3 run.py --cfg=configs/config_seg.py  --j5=_train_,resnet-640,ctscp-train,seg-r-640:sub-8,dm2,batch-1,dbg-1,dyn-1,dist-0,pt-1,seq3k,voc7k,fbb
<a id="on_val_put___r_640_sub_8_dm2_fbb_trai_n_"></a>
### on-val-put       @ r-640-sub-8-dm2-fbb/train-->p2s_seg-ctscp
`x99_put`
CUDA_VISIBLE_DEVICES= python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_ctscp-train-resize_640-sub_8-dm2-mc-batch_48-seq3k-voc7k-fbb-gdez,_eval_,ctscp-val,batch-2,save-vis-0,dbg-0,dyn-1,seg-r-640:sub-8,no_vid,logits,x99_put-4
`hp8470p_put`
CUDA_VISIBLE_DEVICES= python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_ctscp-train-resize_640-sub_8-dm2-mc-batch_48-seq3k-voc7k-fbb-gdez,_eval_,ctscp-val,batch-2,save-vis-0,dbg-0,dyn-1,seg-r-640:sub-8,no_vid,logits,hp8470p_put-4
<a id="on_val___r_640_sub_8_dm2_fbb_trai_n_"></a>
### on-val       @ r-640-sub-8-dm2-fbb/train-->p2s_seg-ctscp
`local`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_ctscp-train-resize_640-sub_8-dm2-mc-batch_48-seq3k-voc7k-fbb-gdez,_eval_,ctscp-val,batch-48,save-vis-0,dbg-0,dyn-1,seg-r-640:sub-8,no_vid,logits
<a id="on_train___r_640_sub_8_dm2_fbb_trai_n_"></a>
### on-train       @ r-640-sub-8-dm2-fbb/train-->p2s_seg-ctscp
`local`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_ctscp-train-resize_640-sub_8-dm2-mc-batch_48-seq3k-voc7k-fbb-gdez,_eval_,ctscp-train,batch-48,save-vis-0,dbg-0,dyn-1,seg-r-640:sub-8,no_vid,logits


<a id="r_640_sub_8_lac_fbb___trai_n_"></a>
## r-640-sub-8-lac-fbb       @ train-->p2s_seg-ctscp
python3 run.py --cfg=configs/config_seg.py  --j5=_train_,resnet-640,ctscp-train,seg-r-640:sub-8,lac,batch-40,dbg-0,dyn-1,dist-2,pt-1,seq3k,voc10,fbb,gdez
```
watch tail -1 log/seg/resnet_640_ctscp-train-resize_640-sub_8-lac-batch_40-seq3k-voc20-fbb-gdez/progress_log.txt                                                                
```
`single gpu`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=_train_,resnet-640,ctscp-train,seg-r-640:sub-8,lac,batch-5,dbg-0,dyn-1,dist-0,pt-1,seq3k,voc10,fbb
<a id="hp8470p_put___r_640_sub_8_lac_fbb_trai_n_"></a>
### hp8470p_put       @ r-640-sub-8-lac-fbb/train-->p2s_seg-ctscp
CUDA_VISIBLE_DEVICES= python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_ctscp-train-resize_640-sub_8-lac-batch_40-seq3k-voc20-fbb-gdez,_eval_,ctscp-val,batch-2,save-vis-0,dbg-0,dyn-1,seg-r-640:sub-8,no_vid,logits,hp8470p_put-4
<a id="on_val___r_640_sub_8_lac_fbb_trai_n_"></a>
### on-val       @ r-640-sub-8-lac-fbb/train-->p2s_seg-ctscp
`local`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_ctscp-train-resize_640-sub_8-lac-batch_40-seq3k-voc20-fbb-gdez,_eval_,ctscp-val,batch-32,save-vis-0,dbg-0,dyn-1,seg-r-640:sub-8,no_vid,logits
<a id="on_train___r_640_sub_8_lac_fbb_trai_n_"></a>
### on-train       @ r-640-sub-8-lac-fbb/train-->p2s_seg-ctscp
`local`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_ctscp-train-resize_640-sub_8-lac-batch_40-seq3k-voc20-fbb-gdez,_eval_,ctscp-train,batch-32,save-vis-0,dbg-0,dyn-1,seg-r-640:sub-8,no_vid,logits

<a id="r_1280_640_p_640_sub_8_lac___trai_n_"></a>
## r-1280_640-p-640-sub-8-lac       @ train-->p2s_seg-ctscp
python3 run.py --cfg=configs/config_seg.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-8,lac,batch-80,dbg-0,dyn-1,dist-2,pt-1,seq2k,voc8192,gdez
`single gpu`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-8,lac,batch-8,dbg-0,dyn-1,dist-0,pt-1,seq2k,voc8192

<a id="r_1280_640_p_640_sub_8_lac_fbb___trai_n_"></a>
## r-1280_640-p-640-sub-8-lac-fbb       @ train-->p2s_seg-ctscp
python3 run.py --cfg=configs/config_seg.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-8,lac,batch-80,dbg-0,dyn-1,dist-2,pt-1,seq2k,voc8192,fbb,gdez
```
watch tail -1 log/seg/resnet_640_ctscp-train-resize_1280x640-640_640-640_640-sub_8-lac-batch_80-seq2k-voc8192-fbb-gdez/progress_log.txt                                   
```
`single gpu`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-8,lac,batch-10,dbg-0,dyn-1,dist-0,pt-1,seq2k,voc8192,fbb
<a id="on_val_put___r_1280_640_p_640_sub_8_lac_fbb_train_"></a>
### on-val-put       @ r-1280_640-p-640-sub-8-lac-fbb/train-->p2s_seg-ctscp
`hp8470p_put`
CUDA_VISIBLE_DEVICES= python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-sub_8-lac-batch_80-seq2k-voc8192-fbb-gdez,_eval_,ctscp-val,batch-2,save-vis-0,dbg-0,dyn-1,seg-r-1280_640:p-640:sub-8,no_vid,logits,hp8470p_put-4
`x99_put`
CUDA_VISIBLE_DEVICES= python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-sub_8-lac-batch_80-seq2k-voc8192-fbb-gdez,_eval_,ctscp-val,batch-2,save-vis-0,dbg-0,dyn-1,seg-r-1280_640:p-640:sub-8,no_vid,logits,x99_put-4
<a id="on_val___r_1280_640_p_640_sub_8_lac_fbb_train_"></a>
### on-val       @ r-1280_640-p-640-sub-8-lac-fbb/train-->p2s_seg-ctscp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-sub_8-lac-batch_80-seq2k-voc8192-fbb-gdez,_eval_,ctscp-val,batch-16,save-vis-0,dbg-0,dyn-1,seg-r-1280_640:p-640:sub-8,no_vid,logits,grs
`local`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-sub_8-lac-batch_80-seq2k-voc8192-fbb-gdez,_eval_,ctscp-val,batch-16,save-vis-0,dbg-0,dyn-1,seg-r-1280_640:p-640:sub-8,no_vid,logits
<a id="on_train___r_1280_640_p_640_sub_8_lac_fbb_train_"></a>
### on-train       @ r-1280_640-p-640-sub-8-lac-fbb/train-->p2s_seg-ctscp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-sub_8-lac-batch_80-seq2k-voc8192-fbb-gdez,_eval_,ctscp-train,batch-32,save-vis-0,dbg-0,dyn-1,seg-r-1280_640:p-640:sub-8,no_vid,logits

<a id="r_1280_640_p_640_sub_5_lac___trai_n_"></a>
## r-1280_640-p-640-sub-5-lac       @ train-->p2s_seg-ctscp
python3 run.py --cfg=configs/config_seg.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-5,lac,batch-32,dbg-0,dyn-1,dist-2,pt-1,seq3k,voc19b,gdez
```
watch tail -1 log/seg/resnet_640_ctscp-train-resize_1280x640-640_640-640_640-sub_5-lac-batch_32-seq3k-voc19b-gdez/progress_log.txt
```
`single gpu`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-5,lac,batch-4,dbg-0,dyn-1,dist-0,pt-1,seq3k,voc19b
<a id="on_val_put___r_1280_640_p_640_sub_5_lac_train_"></a>
### on-val-put       @ r-1280_640-p-640-sub-5-lac/train-->p2s_seg-ctscp
`hp8470p_put`
CUDA_VISIBLE_DEVICES= python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-sub_5-lac-batch_32-seq3k-voc19b-gdez,_eval_,ctscp-val,batch-2,save-vis-0,dbg-0,dyn-1,seg-r-1280_640:p-640:sub-5,no_vid,logits,hp8470p_put-6
<a id="on_val___r_1280_640_p_640_sub_5_lac_train_"></a>
### on-val       @ r-1280_640-p-640-sub-5-lac/train-->p2s_seg-ctscp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-sub_5-lac-batch_32-seq3k-voc19b-gdez,_eval_,ctscp-val,batch-16,save-vis-0,dbg-0,dyn-1,seg-r-1280_640:p-640:sub-5,no_vid,logits,8470p
<a id="on_train___r_1280_640_p_640_sub_5_lac_train_"></a>
### on-train       @ r-1280_640-p-640-sub-5-lac/train-->p2s_seg-ctscp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-sub_5-lac-batch_32-seq3k-voc19b-gdez,_eval_,ctscp-train,batch-16,save-vis-0,dbg-0,dyn-1,seg-r-1280_640:p-640:sub-5,no_vid,logits

<a id="r_1280_640_p_640_sub_5_lac_fbb___trai_n_"></a>
## r-1280_640-p-640-sub-5-lac-fbb       @ train-->p2s_seg-ctscp
python3 run.py --cfg=configs/config_seg.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-5,lac,batch-40,dbg-0,dyn-1,dist-2,pt-1,seq3k,voc19b,fbb,gdez
```
watch tail -1 log/seg/resnet_640_ctscp-train-resize_1280x640-640_640-640_640-sub_5-lac-batch_40-seq3k-voc19b-fbb-gdez/progress_log.txt
```
`single gpu`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-5,lac,batch-5,dbg-0,dyn-1,dist-0,pt-1,seq3k,voc19b,fbb
<a id="on_val_put___r_1280_640_p_640_sub_5_lac_fbb_train_"></a>
### on-val-put       @ r-1280_640-p-640-sub-5-lac-fbb/train-->p2s_seg-ctscp
`hp8470p_put`
CUDA_VISIBLE_DEVICES= python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-sub_5-lac-batch_40-seq3k-voc19b-fbb-gdez,_eval_,ctscp-val,batch-2,save-vis-0,dbg-0,dyn-1,seg-r-1280_640:p-640:sub-5,no_vid,logits,hp8470p_put-4
<a id="on_val___r_1280_640_p_640_sub_5_lac_fbb_train_"></a>
### on-val       @ r-1280_640-p-640-sub-5-lac-fbb/train-->p2s_seg-ctscp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-sub_5-lac-batch_40-seq3k-voc19b-fbb-gdez,_eval_,ctscp-val,batch-16,save-vis-0,dbg-0,dyn-1,seg-r-1280_640:p-640:sub-5,no_vid,logits,8470p
<a id="on_train___r_1280_640_p_640_sub_5_lac_fbb_train_"></a>
### on-train       @ r-1280_640-p-640-sub-5-lac-fbb/train-->p2s_seg-ctscp
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-sub_5-lac-batch_40-seq3k-voc19b-fbb-gdez,_eval_,ctscp-train,batch-32,save-vis-0,dbg-0,dyn-1,seg-r-1280_640:p-640:sub-5,no_vid,logits


<a id="r_1280_640_p_640_sub_8_lac_bac_fbb___trai_n_"></a>
## r-1280_640-p-640-sub-8-lac-bac-fbb       @ train-->p2s_seg-ctscp
python3 run.py --cfg=configs/config_seg.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-8,lac,bac,batch-256,dbg-0,dyn-1,dist-0,pt-1,seq1k,voc2k,fbb,gdez
`single gpu`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-8,lac,bac,batch-32,dbg-0,dyn-1,dist-0,pt-1,seq1k,voc2k,fbb
```
watch tail -1 log/seg/resnet_640_ctscp-train-resize_1280x640-640_640-640_640-sub_8-bac-lac-batch_256-seq1k-voc2k-fbb-gdez/progress_log.txt
```
<a id="hp8470p_put___r_1280_640_p_640_sub_8_lac_bac_fbb_train_"></a>
### hp8470p_put       @ r-1280_640-p-640-sub-8-lac-bac-fbb/train-->p2s_seg-ctscp
CUDA_VISIBLE_DEVICES= python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-sub_8-bac-lac-batch_256-seq1k-voc2k-fbb-gdez,_eval_,ctscp-val,batch-2,save-vis-0,dbg-0,dyn-1,seg-r-1280_640:p-640:sub-8,no_vid,logits,hp8470p_put-6

<a id="r_1280_640_p_640_sub_2_lac_fbb___trai_n_"></a>
## r-1280_640-p-640-sub-2-lac-fbb       @ train-->p2s_seg-ctscp
`gives OOM`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-2,lac,batch-1,dbg-0,dyn-1,dist-0,pt-1,seq8k,voc109k,fbb
<a id="r_1280_640_p_640_sub_2_lac_2d_fbb___trai_n_"></a>
## r-1280_640-p-640-sub-2-lac-2d-fbb       @ train-->p2s_seg-ctscp
`gives OOM`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-2,lac,2d,batch-1,dbg-0,dyn-1,dist-0,pt-1,seq12k,voc7k,fbb
<a id="train_rfm_"></a>
# train-rfm
<a id="p_1024_sub_8_lac_fbb___train_rf_m_"></a>
## p-1024-sub-8-lac-fbb       @ train-rfm-->p2s_seg-ctscp
python3 run.py --cfg=configs/config_seg.py  --j5=_train_,resnet-1024,ctscp-train,seg-p-1024:sub-8,rfm,rot,flip,batch-32,dbg-0,dyn-1,dist-2,pt-1,lac,mc,seq3k,fbb,voc20,cls_eq-1,zedg

CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=_train_,resnet-1024,ctscp-train,seg-p-1024:sub-8,rfm,rot,flip,batch-8,dbg-0,dyn-1,dist-0,pt-1,lac,mc,seq3k,fbb,voc20,cls_eq-1
```
watch tail -1 log/seg/resnet_1024_train-1024_1024-1024_1024-ctscp_train-rfm-rot-flip-batch_32-seq3k-fbb-voc20-cls_eq_1-zedg/progress_log.txt
```
`dbg`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=_train_,resnet-1024,ctscp-train,seg-p-1024:sub-8,rfm,rot,flip,batch-2,dbg-1,dyn-1,dist-0,pt-1,lac,mc,seq3k,fbb,voc20,cls_eq-1
<a id="on_train___p_1024_sub_8_lac_fbb_train_rfm_"></a>
### on-train       @ p-1024-sub-8-lac-fbb/train-rfm-->p2s_seg-ctscp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_1024_train-1024_1024-1024_1024-ctscp_train-rfm-rot-flip-batch_32-seq3k-fbb-voc20-cls_eq_1-zedg,_eval_,ctscp-train,batch-32,save-vis-0,dbg-0,dyn-1,seg-p-1024:sub-8,rfm,grs,no_vid,logits
`local`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_1024_train-1024_1024-1024_1024-ctscp_train-rfm-rot-flip-batch_32-seq3k-fbb-voc20-cls_eq_1-zedg,_eval_,ctscp-train,batch-32,save-vis-0,dbg-0,dyn-1,seg-p-1024:sub-8,rfm,no_vid,logits
`seq-0`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_1024_train-1024_1024-1024_1024-ctscp_train-rfm-rot-flip-batch_32-seq3k-fbb-voc20-cls_eq_1-zedg,_eval_,ctscp-train,batch-32,save-vis-0,dbg-0,dyn-1,seg-p-1024:sub-8:seq-0,rfm,no_vid,logits
`dbg`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_1024_train-1024_1024-1024_1024-ctscp_train-rfm-rot-flip-batch_32-seq3k-fbb-voc20-cls_eq_1-zedg,_eval_,ctscp-train,batch-2,save-vis-1,dbg-0,dyn-1,seg-p-1024:sub-8,rfm,grs,no_vid,logits
<a id="on_val___p_1024_sub_8_lac_fbb_train_rfm_"></a>
### on-val       @ p-1024-sub-8-lac-fbb/train-rfm-->p2s_seg-ctscp
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_1024_train-1024_1024-1024_1024-ctscp_train-rfm-rot-flip-batch_32-seq3k-fbb-voc20-cls_eq_1-zedg,_eval_,ctscp-val,batch-32,save-vis-0,dbg-0,dyn-1,seg-p-1024:sub-8,rfm,grs,no_vid,logits
`local`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_1024_train-1024_1024-1024_1024-ctscp_train-rfm-rot-flip-batch_32-seq3k-fbb-voc20-cls_eq_1-zedg,_eval_,ctscp-val,batch-32,save-vis-0,dbg-0,dyn-1,seg-p-1024:sub-8,rfm,no_vid,logits
```
resnet_1024_train-1024_1024-1024_1024-ctscp_train-rfm-rot-flip-batch_32-seq3k-fbb-voc20-cls_eq_1-zedg/ckpt-*-ctscp-val-1024_1024-1024_1024-sub_8-lac/masks-batch_32

resnet_1024_train-1024_1024-1024_1024-ctscp_train-rfm-rot-flip-batch_32-seq3k-fbb-voc20-cls_eq_1-zedg/ckpt-*-val-1024_1024-1024_1024-sub_8-lac/masks-ctscp-batch_32

```
`defer`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_1024_train-1024_1024-1024_1024-ctscp_train-rfm-rot-flip-batch_32-seq3k-fbb-voc20-cls_eq_1-zedg,_eval_,ctscp-val,batch-32,save-vis-0,dbg-0,dyn-1,seg-p-1024:sub-8,lac,seq3k,voc20,rfm,grs,defer
`dbg`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=m-resnet_1024_train-1024_1024-1024_1024-ctscp_train-rfm-rot-flip-batch_32-seq3k-fbb-voc20-cls_eq_1-zedg,_eval_,ctscp-val,batch-2,save-vis-0,dbg-1,dyn-1,seg-p-1024:sub-8,rfm,grs,no_vid,logits


<a id="p_1024_sub_8_lac_2d_fbb___train_rf_m_"></a>
## p-1024-sub-8-lac-2d-fbb       @ train-rfm-->p2s_seg-ctscp
CUDA_VISIBLE_DEVICES=0,1 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-1024,ctscp,seg-p-1024:sub-8,rfm,batch-32,dbg-0,dyn-1,dist-1,pt-1,2d,lac,mc,seq5k,voc3k,fbb,cls_eq-1
`dbg`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg.py  --j5=train,resnet-1024,ctscp,seg-p-1024:sub-8,rfm,batch-4,dbg-0,dyn-1,dist-1,pt-1,2d,lac,mc,seq5k,voc3k,fbb,cls_eq-1

<a id="p_1024_sub_8_mhd_fbb___train_rf_m_"></a>
## p-1024-sub-8-mhd-fbb       @ train-rfm-->p2s_seg-ctscp
python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-1024,ctscp-train,seg-p-1024:sub-8,mhd,rfm,rot,flip,batch-32,dbg-0,dyn-1,dist-2,pt-1,seq15,voc_xyl-131,voc_c-22,fbb,zedg
`single gpu`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-1024,ctscp-train,seg-p-1024:sub-8,mhd,rfm,rot,flip,batch-4,dbg-0,dyn-1,dist-0,pt-1,seq15,voc_xyl-131,voc_c-22,fbb
`smha (separate mha)`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-1024,ctscp-train,seg-p-1024:sub-8,mhd,smha,rfm,rot,flip,batch-4,dbg-0,dyn-1,dist-0,pt-1,seq15,voc_xyl-131,voc_c-22,fbb
`seq-512`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-1024,ctscp-train,seg-p-1024:sub-8,mhd,rfm,rot,flip,batch-6,dbg-0,dyn-1,dist-0,pt-1,voc_xyl-131,voc_c-22,fbb
`dbg`
python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-1024,ctscp-train,seg-p-1024:sub-8,mhd,rfm,rot,flip,batch-1,dbg-1,dyn-1,dist-0,pt-1,seq15,voc_xyl-131,voc_c-22,fbb

<a id="r_1280_640___train_rf_m_"></a>
## r-1280_640       @ train-rfm-->p2s_seg-ctscp
<a id="p_640_sub_8_mhd_fbb___r_1280_640_train_rfm_"></a>
### p-640-sub-8-mhd-fbb       @ r-1280_640/train-rfm-->p2s_seg-ctscp
NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=enp6s0 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-8,mhd,rfm,rot,flip,batch-72,dbg-0,dyn-1,dist-2,pt-1,seq1k,voc_xyl-86,voc_c-25,fbb,gdez
NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=eno1 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-8,mhd,rfm,rot,flip,batch-72,dbg-0,dyn-1,dist-2,pt-1,seq1k,voc_xyl-86,voc_c-25,fbb,gdez
NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=enp0s25 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-8,mhd,rfm,rot,flip,batch-72,dbg-0,dyn-1,dist-2,pt-1,seq1k,voc_xyl-86,voc_c-25,fbb,gdez
NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=enp13s0 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-8,mhd,rfm,rot,flip,batch-72,dbg-0,dyn-1,dist-2,pt-1,seq1k,voc_xyl-86,voc_c-25,fbb,gdez

```
watch tail -1
log/seg/resnet_640_ctscp-train-resize_1280x640-640_640-640_640-mhd-rfm-rot-flip-batch_72-seq1k-fbb-zedg
/progress_log.txt     

mv log/seg/resnet_640_ctscp-train-resize_1280x640-640_640-640_640-mhd-rfm-rot-flip-batch_72-seq1k-fbb-zedg log/seg/resnet_640_ctscp-train-resize_1280x640-640_640-640_640-mhd-rfm-rot-flip-batch_72-seq1k-fbb-gdez     
```

`single gpu`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-8,mhd,rfm,rot,flip,batch-9,dbg-0,dyn-1,dist-0,pt-1,seq1k,voc_xyl-86,voc_c-25,fbb
`dbg`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-8,mhd,rfm,rot,flip,batch-1,dbg-1,dyn-1,dist-0,pt-1,seq1k,voc_xyl-86,voc_c-25,fbb
```
watch tail -1 
log/seg/resnet_640_ctscp-train-resize_1280x640-640_640-640_640-mhd-rfm-rot-flip-batch_9-seq1k-fbb
/progress_log.txt
```
<a id="on_val___p_640_sub_8_mhd_fbb_r_1280_640_train_rfm_"></a>
#### on-val       @ p-640-sub-8-mhd-fbb/r-1280_640/train-rfm-->p2s_seg-ctscp
CUDA_VISIBLE_DEVICES= python3 run.py --cfg=configs/config_seg_mhd.py  --j5=m-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-mhd-rfm-rot-flip-batch_72-seq1k-fbb-gdez,_eval_,ctscp-val,batch-2,save-vis-0,dbg-0,dyn-1,seg-r-1280_640:p-640:sub-8,mhd,rfm,no_vid,logits,x99_put
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=m-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-mhd-rfm-rot-flip-batch_72-seq1k-fbb-gdez,_eval_,ctscp-val,batch-32,save-vis-0,dbg-0,dyn-1,seg-r-1280_640:p-640:sub-8,mhd,rfm,no_vid,logits


<a id="p_640_sub_8_mhd_1241_fbb___r_1280_640_train_rfm_"></a>
### p-640-sub-8-mhd-1241-fbb       @ r-1280_640/train-rfm-->p2s_seg-ctscp
NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=enp6s0 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-8,mhd-1241,rfm,rot,flip,batch-72,dbg-0,dyn-1,dist-2,pt-1,seq1k,voc_xyl-86,voc_c-25,fbb,gdez
NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=eno1 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-8,mhd-1241,rfm,rot,flip,batch-72,dbg-0,dyn-1,dist-2,pt-1,seq1k,voc_xyl-86,voc_c-25,fbb,gdez
NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=enp0s25 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-8,mhd-1241,rfm,rot,flip,batch-72,dbg-0,dyn-1,dist-2,pt-1,seq1k,voc_xyl-86,voc_c-25,fbb,gdez
NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=enp13s0 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-8,mhd-1241,rfm,rot,flip,batch-72,dbg-0,dyn-1,dist-2,pt-1,seq1k,voc_xyl-86,voc_c-25,fbb,gdez
```
watch tail -1 log/seg/resnet_640_ctscp-train-resize_1280x640-640_640-640_640-mhd_1241-rfm-rot-flip-batch_72-seq1k-fbb-gdez/progress_log.txt     
```
`single gpu`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-8,mhd-1241,rfm,rot,flip,batch-9,dbg-0,dyn-1,dist-0,pt-1,seq1k,voc_xyl-86,voc_c-25,fbb
`dbg`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-8,mhd-1241,rfm,rot,flip,batch-1,dbg-1,dyn-1,dist-0,pt-1,seq1k,voc_xyl-86,voc_c-25,fbb
<a id="on_val___p_640_sub_8_mhd_1241_fbb_r_1280_640_train_rf_m_"></a>
#### on-val       @ p-640-sub-8-mhd-1241-fbb/r-1280_640/train-rfm-->p2s_seg-ctscp
CUDA_VISIBLE_DEVICES= python3 run.py --cfg=configs/config_seg_mhd.py  --j5=m-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-mhd_1241-rfm-rot-flip-batch_72-seq1k-fbb-gdez,_eval_,ctscp-val,batch-2,save-vis-0,dbg-0,dyn-1,seg-r-1280_640:p-640:sub-8,mhd,rfm,no_vid,logits,x99_put
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=m-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-mhd_1241-rfm-rot-flip-batch_72-seq1k-fbb-gdez,_eval_,ctscp-val,batch-32,save-vis-0,dbg-0,dyn-1,seg-r-1280_640:p-640:sub-8,mhd,rfm,no_vid,logits


<a id="p_640_sub_8_mhd_1241_fbb_b64___r_1280_640_train_rfm_"></a>
### p-640-sub-8-mhd-1241-fbb-b64       @ r-1280_640/train-rfm-->p2s_seg-ctscp
NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=enp6s0 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-8,mhd-1241,rfm,rot,flip,batch-64,dbg-0,dyn-1,dist-2,pt-1,seq1k,voc_xyl-86,voc_c-25,fbb,zedg
NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=eno1 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-8,mhd-1241,rfm,rot,flip,batch-64,dbg-0,dyn-1,dist-2,pt-1,seq1k,voc_xyl-86,voc_c-25,fbb,zedg
NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=enp0s25 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-8,mhd-1241,rfm,rot,flip,batch-64,dbg-0,dyn-1,dist-2,pt-1,seq1k,voc_xyl-86,voc_c-25,fbb,zedg
NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=enp13s0 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-8,mhd-1241,rfm,rot,flip,batch-64,dbg-0,dyn-1,dist-2,pt-1,seq1k,voc_xyl-86,voc_c-25,fbb,zedg
```
watch tail -1 log/seg/resnet_640_ctscp-train-resize_1280x640-640_640-640_640-mhd_1241-rfm-rot-flip-batch_64-seq1k-fbb-zedg/progress_log.txt                                     
```
<a id="on_val___p_640_sub_8_mhd_1241_fbb_b64_r_1280_640_train_rf_m_"></a>
#### on-val       @ p-640-sub-8-mhd-1241-fbb-b64/r-1280_640/train-rfm-->p2s_seg-ctscp
CUDA_VISIBLE_DEVICES= python3 run.py --cfg=configs/config_seg_mhd.py  --j5=m-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-mhd_1241-rfm-rot-flip-batch_64-seq1k-fbb-zedg,_eval_,ctscp-val,batch-2,save-vis-0,dbg-0,dyn-1,seg-r-1280_640:p-640:sub-8,mhd,rfm,no_vid,logits,x99_put
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=m-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-mhd_1241-rfm-rot-flip-batch_64-seq1k-fbb-zedg,_eval_,ctscp-val,batch-32,save-vis-0,dbg-0,dyn-1,seg-r-1280_640:p-640:sub-8,mhd,rfm,no_vid,logits

<a id="p_640_sub_8_mhd_1241_fbb_b54_zeg___r_1280_640_train_rfm_"></a>
### p-640-sub-8-mhd-1241-fbb-b54-zeg       @ r-1280_640/train-rfm-->p2s_seg-ctscp
NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=enp6s0 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-8,mhd-1241,rfm,rot,flip,batch-54,dbg-0,dyn-1,dist-2,pt-1,seq1k,voc_xyl-86,voc_c-25,fbb,zeg
NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=enp0s25 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-8,mhd-1241,rfm,rot,flip,batch-54,dbg-0,dyn-1,dist-2,pt-1,seq1k,voc_xyl-86,voc_c-25,fbb,zeg
NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=enp13s0 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-8,mhd-1241,rfm,rot,flip,batch-54,dbg-0,dyn-1,dist-2,pt-1,seq1k,voc_xyl-86,voc_c-25,fbb,zeg
```
watch tail -1 log/seg/resnet_640_ctscp-train-resize_1280x640-640_640-640_640-mhd_1241-rfm-rot-flip-batch_54-seq1k-fbb-zeg/progress_log.txt     

log/seg/resnet_640_ctscp-train-resize_1280x640-640_640-640_640-mhd_1241-rfm-rot-flip-batch_54-seq1k-fbb-zeg 
```
<a id="on_val___p_640_sub_8_mhd_1241_fbb_b54_zeg_r_1280_640_train_rf_m_"></a>
#### on-val       @ p-640-sub-8-mhd-1241-fbb-b54-zeg/r-1280_640/train-rfm-->p2s_seg-ctscp
`d3`
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=m-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-mhd_1241-rfm-rot-flip-batch_54-seq1k-fbb-zeg,_eval_,ctscp-val,batch-32,save-vis-0,dbg-0,dyn-1,seg-r-1280_640:p-640:sub-8,mhd,rfm,no_vid,grs
`x99_put`
CUDA_VISIBLE_DEVICES= python3 run.py --cfg=configs/config_seg_mhd.py  --j5=m-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-mhd_1241-rfm-rot-flip-batch_54-seq1k-fbb-zeg,_eval_,ctscp-val,batch-2,save-vis-0,dbg-0,dyn-1,seg-r-1280_640:p-640:sub-8,mhd,rfm,no_vid,x99_put
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=m-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-mhd_1241-rfm-rot-flip-batch_54-seq1k-fbb-zeg,_eval_,ctscp-val,batch-32,save-vis-0,dbg-0,dyn-1,seg-r-1280_640:p-640:sub-8,mhd,rfm,no_vid
`dbg`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=m-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-mhd_1241-rfm-rot-flip-batch_54-seq1k-fbb-zeg,_eval_,ctscp-val,batch-1,save-vis-0,dbg-1,dyn-1,seg-r-1280_640:p-640:sub-8,mhd,rfm,no_vid


<a id="p_640_sub_8_mhd_1241_fbb_b16___r_1280_640_train_rfm_"></a>
### p-640-sub-8-mhd-1241-fbb-b16       @ r-1280_640/train-rfm-->p2s_seg-ctscp
python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-8,mhd-1241,rfm,rot,flip,batch-16,dbg-0,dyn-1,dist-1,pt-1,seq1k,voc_xyl-86,voc_c-25,fbb

<a id="p_640_sub_8_mhd_1241_fbb_b9___r_1280_640_train_rfm_"></a>
### p-640-sub-8-mhd-1241-fbb-b9       @ r-1280_640/train-rfm-->p2s_seg-ctscp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-8,mhd-1241,rfm,rot,flip,batch-9,dbg-0,dyn-1,dist-0,pt-1,seq1k,voc_xyl-86,voc_c-25,fbb

<a id="p_640_sub_8_mhd_fbb_no_aug___r_1280_640_train_rfm_"></a>
### p-640-sub-8-mhd-fbb-no_aug       @ r-1280_640/train-rfm-->p2s_seg-ctscp
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-8,mhd,rfm,batch-9,dbg-0,dyn-1,dist-0,pt-1,seq1k,voc_xyl-86,voc_c-25,fbb
```
watch tail -1 log/seg/resnet_640_ctscp-train-resize_1280x640-640_640-640_640-mhd-rfm-batch_9-seq1k-fbb/progress_log.txt
```
<a id="on_val___p_640_sub_8_mhd_fbb_no_aug_r_1280_640_train_rf_m_"></a>
#### on-val       @ p-640-sub-8-mhd-fbb-no_aug/r-1280_640/train-rfm-->p2s_seg-ctscp
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=m-resnet_640_ctscp-train-resize_1280x640-640_640-640_640-mhd-rfm-batch_9-seq1k-fbb/,_eval_,ctscp-val,batch-2,save-vis-0,dbg-0,dyn-1,seg-r-1280_640:p-640:sub-8,mhd,rfm,no_vid


<a id="p_640_sub_4_mhd_fbb___r_1280_640_train_rfm_"></a>
### p-640-sub-4-mhd-fbb       @ r-1280_640/train-rfm-->p2s_seg-ctscp
python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-4,mhd,rfm,rot,flip,batch-32,dbg-0,dyn-1,dist-2,pt-1,seq2k,voc_xyl-166,voc_c-25,fbb,zedg
`single gpu`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-4,mhd,rfm,rot,flip,batch-6,dbg-0,dyn-1,dist-0,pt-1,seq2k,voc_xyl-166,voc_c-25,fbb
`sxyl`
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-4,mhd,rfm,rot,flip,batch-6,dbg-0,dyn-1,dist-0,pt-1,seq2k,voc_xyl-166,voc_c-25,fbb,sxyl
`dbg`
python3 run.py --cfg=configs/config_seg_mhd.py  --j5=_train_,resnet-640,ctscp-train,seg-r-1280_640:p-640:sub-4,mhd,rfm,batch-1,dbg-1,dyn-1,dist-0,pt-1,seq2k,voc_xyl-166,voc_c-25,fbb