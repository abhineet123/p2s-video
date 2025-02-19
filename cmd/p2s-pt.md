<!-- MarkdownTOC -->

- [resnet-640](#resnet_64_0_)
    - [on-mninstmot       @ resnet-640](#on_mninstmot___resnet_640_)
    - [on-g2_0_1       @ resnet-640](#on_g2_0_1___resnet_640_)
    - [on-g2_16_53       @ resnet-640](#on_g2_16_53___resnet_640_)
        - [dist       @ on-g2_16_53/resnet-640](#dist___on_g2_16_53_resnet_640_)
    - [on-g2_54_126       @ resnet-640](#on_g2_54_126___resnet_640_)
    - [on-g2_0_15       @ resnet-640](#on_g2_0_15___resnet_640_)
- [resnet-1333](#resnet_1333_)
    - [on-g2_0_1       @ resnet-1333](#on_g2_0_1___resnet_133_3_)
    - [on-g2_16_53       @ resnet-1333](#on_g2_16_53___resnet_133_3_)
- [resnet_c4-640](#resnet_c4_640_)
    - [on-g2_16_53       @ resnet_c4-640](#on_g2_16_53___resnet_c4_64_0_)
- [resnet_c4_1333](#resnet_c4_133_3_)
    - [on-g2_0_1       @ resnet_c4_1333](#on_g2_0_1___resnet_c4_1333_)
    - [on-g2_16_53       @ resnet_c4_1333](#on_g2_16_53___resnet_c4_1333_)

<!-- /MarkdownTOC -->
<a id="resnet_64_0_"></a>
# resnet-640 
<a id="on_mninstmot___resnet_640_"></a>
## on-mninstmot       @ resnet-640-->p2s-pt
python3 run.py --mode=eval --model_dir=pretrained/resnet_640 --cfg=configs/config_det_mninstmot.py
<a id="on_g2_0_1___resnet_640_"></a>
## on-g2_0_1       @ resnet-640-->p2s-pt
``batch-48``  
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,pt,resnet-640,ipsc-0_1,batch-48,save-vis-1,save-csv-1
``batch-2``  
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,pt,resnet-640,ipsc-0_1,batch-2,save-vis-1,save-csv-1
<a id="on_g2_16_53___resnet_640_"></a>
## on-g2_16_53       @ resnet-640-->p2s-pt
``batch-64``  
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,pt,resnet-640,ipsc-16_53,batch-64,save-vis-1,save-csv-1
``batch-32``  
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,pt,resnet-640,ipsc-16_53,batch-32,save-vis-1,save-csv-1
<a id="dist___on_g2_16_53_resnet_640_"></a>
### dist       @ on-g2_16_53/resnet-640-->p2s-pt
python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,pt,resnet-640,ipsc-0_1,batch-2,save-vis-1,save-csv-1,dist-1,eager-0

<a id="on_g2_54_126___resnet_640_"></a>
## on-g2_54_126       @ resnet-640-->p2s-pt
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,pt,resnet-640,ipsc-54_126,batch-32,save-vis-0,save-csv-1
<a id="on_g2_0_15___resnet_640_"></a>
## on-g2_0_15       @ resnet-640-->p2s-pt
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,pt,resnet-640,ipsc-0_15,batch-32,save-vis-0,save-csv-1


<a id="resnet_1333_"></a>
# resnet-1333 
<a id="on_g2_0_1___resnet_133_3_"></a>
## on-g2_0_1       @ resnet-1333-->p2s-pt
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,pt,resnet-1333,ipsc-0_1,batch-48,save-vis-1,save-csv-1
<a id="on_g2_16_53___resnet_133_3_"></a>
## on-g2_16_53       @ resnet-1333-->p2s-pt
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,pt,resnet-1333,ipsc-16_53,batch-24,save-vis-1,save-csv-1

<a id="resnet_c4_640_"></a>
# resnet_c4-640 
<a id="on_g2_16_53___resnet_c4_64_0_"></a>
## on-g2_16_53       @ resnet_c4-640-->p2s-pt
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,pt,resnet_c4-640,ipsc-16_53,batch-16,save-vis-1,save-csv-1

<a id="resnet_c4_133_3_"></a>
# resnet_c4_1333 
<a id="on_g2_0_1___resnet_c4_1333_"></a>
## on-g2_0_1       @ resnet_c4_1333-->p2s-pt
CUDA_VISIBLE_DEVICES=1 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,pt,resnet_c4-1333,ipsc-0_1,batch-1,save-vis-1,save-csv-1
<a id="on_g2_16_53___resnet_c4_1333_"></a>
## on-g2_16_53       @ resnet_c4_1333-->p2s-pt
CUDA_VISIBLE_DEVICES=0 python3 run.py --cfg=configs/config_det_ipsc.py  --j5=eval,pt,resnet_c4-1333,ipsc-16_53,batch-1,save-vis-1,save-csv-1

