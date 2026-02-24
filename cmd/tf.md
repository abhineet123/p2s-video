<!-- MarkdownTOC -->

- [imgn       @ tfrecord](#imgn___tfrecord_)
    - [vid       @ imgn](#vid___imgn_)
        - [8_per_seq_random       @ vid/imgn](#8_per_seq_random___vid_imgn_)
    - [vid_val       @ imgn](#vid_val___imgn_)
        - [16_per_seq_random       @ vid_val/imgn](#16_per_seq_random___vid_val_imgn_)
    - [vid_det       @ imgn](#vid_det___imgn_)
        - [sampled_eq       @ vid_det/imgn](#sampled_eq___vid_det_imgn_)
        - [8_per_seq_random       @ vid_det/imgn](#8_per_seq_random___vid_det_imgn_)
        - [ratio_1_10_random       @ vid_det/imgn](#ratio_1_10_random___vid_det_imgn_)
    - [vid_det_all       @ imgn](#vid_det_all___imgn_)
        - [sampled_eq       @ vid_det_all/imgn](#sampled_eq___vid_det_all_imgn_)
        - [ratio_1_10_random       @ vid_det_all/imgn](#ratio_1_10_random___vid_det_all_imgn_)
- [detrac-non_empty       @ tfrecord](#detrac_non_empty___tfrecord_)
    - [0_19       @ detrac-non_empty](#0_19___detrac_non_empty_)
    - [0_9       @ detrac-non_empty](#0_9___detrac_non_empty_)
    - [0_48       @ detrac-non_empty](#0_48___detrac_non_empty_)
    - [49_68       @ detrac-non_empty](#49_68___detrac_non_empty_)
    - [49_85       @ detrac-non_empty](#49_85___detrac_non_empty_)
        - [100_per_seq_random       @ 49_85/detrac-non_empty](#100_per_seq_random___49_85_detrac_non_empty_)
- [detrac       @ tfrecord](#detrac___tfrecord_)
    - [0_59       @ detrac](#0_59___detrac_)
        - [100_per_seq_random       @ 0_59/detrac](#100_per_seq_random___0_59_detra_c_)
        - [40_per_seq_random       @ 0_59/detrac](#40_per_seq_random___0_59_detra_c_)
    - [60_99       @ detrac](#60_99___detrac_)
        - [100_per_seq_random       @ 60_99/detrac](#100_per_seq_random___60_99_detrac_)
        - [40_per_seq_random       @ 60_99/detrac](#40_per_seq_random___60_99_detrac_)
- [ipsc       @ tfrecord](#ipsc___tfrecord_)
    - [0_1       @ ipsc](#0_1___ipsc_)
    - [2_3       @ ipsc](#2_3___ipsc_)
    - [16_53       @ ipsc](#16_53___ipsc_)
    - [0_37       @ ipsc](#0_37___ipsc_)
    - [54_126       @ ipsc](#54_126___ipsc_)
    - [0_1       @ ipsc](#0_1___ipsc__1)
    - [0_15       @ ipsc](#0_15___ipsc_)
    - [38_53       @ ipsc](#38_53___ipsc_)
- [acamp](#acamp_)
    - [1k8_vid_entire_seq       @ acamp](#1k8_vid_entire_seq___acam_p_)
        - [inv       @ 1k8_vid_entire_seq/acamp](#inv___1k8_vid_entire_seq_acamp_)
            - [2_per_seq       @ inv/1k8_vid_entire_seq/acamp](#2_per_seq___inv_1k8_vid_entire_seq_acamp_)
    - [10k6_vid_entire_seq       @ acamp](#10k6_vid_entire_seq___acam_p_)
        - [inv       @ 10k6_vid_entire_seq/acamp](#inv___10k6_vid_entire_seq_acam_p_)
            - [2_per_seq       @ inv/10k6_vid_entire_seq/acamp](#2_per_seq___inv_10k6_vid_entire_seq_acam_p_)
    - [20k6_5_video       @ acamp](#20k6_5_video___acam_p_)
        - [inv       @ 20k6_5_video/acamp](#inv___20k6_5_video_acamp_)
            - [2_per_seq       @ inv/20k6_5_video/acamp](#2_per_seq___inv_20k6_5_video_acamp_)

<!-- /MarkdownTOC -->

<a id="imgn___tfrecord_"></a>
# imgn       @ tfrecord-->p2s_setup
<a id="vid___imgn_"></a>
## vid       @ imgn-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=imgn:vid
<a id="8_per_seq_random___vid_imgn_"></a>
### 8_per_seq_random       @ vid/imgn-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=imgn:8_per_seq_random

<a id="vid_val___imgn_"></a>
## vid_val       @ imgn-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=imgn:vid_val
<a id="16_per_seq_random___vid_val_imgn_"></a>
### 16_per_seq_random       @ vid_val/imgn-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=imgn:vid_val:16_per_seq_random

<a id="vid_det___imgn_"></a>
## vid_det       @ imgn-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=imgn:vid_det
<a id="sampled_eq___vid_det_imgn_"></a>
### sampled_eq       @ vid_det/imgn-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=imgn:vid_det:sampled_eq
<a id="8_per_seq_random___vid_det_imgn_"></a>
### 8_per_seq_random       @ vid_det/imgn-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=imgn:vid_det:8_per_seq_random
<a id="ratio_1_10_random___vid_det_imgn_"></a>
### ratio_1_10_random       @ vid_det/imgn-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=imgn:vid_det:ratio_1_10_random

<a id="vid_det_all___imgn_"></a>
## vid_det_all       @ imgn-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=imgn:vid_det_all
<a id="sampled_eq___vid_det_all_imgn_"></a>
### sampled_eq       @ vid_det_all/imgn-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=imgn:vid_det_all:sampled_eq
<a id="ratio_1_10_random___vid_det_all_imgn_"></a>
### ratio_1_10_random       @ vid_det_all/imgn-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=imgn:vid_det_all:ratio_1_10_random

<a id="detrac_non_empty___tfrecord_"></a>
# detrac-non_empty       @ tfrecord-->p2s_setup
<a id="0_19___detrac_non_empty_"></a>
## 0_19       @ detrac-non_empty-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=detrac:non_empty-0_19
<a id="0_9___detrac_non_empty_"></a>
## 0_9       @ detrac-non_empty-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=detrac:non_empty-0_9
<a id="0_48___detrac_non_empty_"></a>
## 0_48       @ detrac-non_empty-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=detrac:non_empty-0_48
<a id="49_68___detrac_non_empty_"></a>
## 49_68       @ detrac-non_empty-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=detrac:non_empty-49_68
<a id="49_85___detrac_non_empty_"></a>
## 49_85       @ detrac-non_empty-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=detrac:non_empty-49_85
<a id="100_per_seq_random___49_85_detrac_non_empty_"></a>
### 100_per_seq_random       @ 49_85/detrac-non_empty-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=detrac:non_empty-49_85:100_per_seq_random

<a id="detrac___tfrecord_"></a>
# detrac       @ tfrecord-->p2s_setup
<a id="0_59___detrac_"></a>
## 0_59       @ detrac-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=detrac:0_59
<a id="100_per_seq_random___0_59_detra_c_"></a>
### 100_per_seq_random       @ 0_59/detrac-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=detrac:0_59:100_per_seq_random
<a id="40_per_seq_random___0_59_detra_c_"></a>
### 40_per_seq_random       @ 0_59/detrac-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=detrac:0_59:40_per_seq_random

<a id="60_99___detrac_"></a>
## 60_99       @ detrac-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=detrac:60_99
<a id="100_per_seq_random___60_99_detrac_"></a>
### 100_per_seq_random       @ 60_99/detrac-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=detrac:60_99:100_per_seq_random
<a id="40_per_seq_random___60_99_detrac_"></a>
### 40_per_seq_random       @ 60_99/detrac-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=detrac:60_99:40_per_seq_random




<a id="ipsc___tfrecord_"></a>
# ipsc       @ tfrecord-->p2s_setup
<a id="0_1___ipsc_"></a>
## 0_1       @ ipsc-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=ipsc:0_1:mask
<a id="2_3___ipsc_"></a>
## 2_3       @ ipsc-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=ipsc:2_3
<a id="16_53___ipsc_"></a>
## 16_53       @ ipsc-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=ipsc:16_53
<a id="0_37___ipsc_"></a>
## 0_37       @ ipsc-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=ipsc:0_37
<a id="38_53___ipsc_"></a>
<a id="54_126___ipsc_"></a>
## 54_126       @ ipsc-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=ipsc:54_126
python3 data/scripts/create_ipsc_tfrecord.py cfg=ipsc:54_126:strd-5
python3 data/scripts/create_ipsc_tfrecord.py cfg=ipsc:54_126:strd-8

<a id="0_1___ipsc__1"></a>
## 0_1       @ ipsc-->tf
python data/scripts/create_ipsc_tfrecord.py --ann_file=ext_reorg_roi_g2_0_1.json --n_proc=0
<a id="0_15___ipsc_"></a>
## 0_15       @ ipsc-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=ipsc:0_15
<a id="38_53___ipsc_"></a>
## 38_53       @ ipsc-->tf
python3 data/scripts/create_ipsc_tfrecord.py --ann_file=ext_reorg_roi_g2_38_53.json --n_proc=0

<a id="acamp_"></a>
# acamp
<a id="1k8_vid_entire_seq___acam_p_"></a>
## 1k8_vid_entire_seq       @ acamp-->tf
python data/scripts/create_ipsc_tfrecord.py cfg=acamp:1k8_vid_entire_seq
<a id="inv___1k8_vid_entire_seq_acamp_"></a>
### inv       @ 1k8_vid_entire_seq/acamp-->tf
python data/scripts/create_ipsc_tfrecord.py cfg=acamp:1k8_vid_entire_seq_inv
<a id="2_per_seq___inv_1k8_vid_entire_seq_acamp_"></a>
#### 2_per_seq       @ inv/1k8_vid_entire_seq/acamp-->tf
python data/scripts/create_ipsc_tfrecord.py cfg=acamp:1k8_vid_entire_seq_inv_2_per_seq

<a id="10k6_vid_entire_seq___acam_p_"></a>
## 10k6_vid_entire_seq       @ acamp-->tf
python data/scripts/create_ipsc_tfrecord.py cfg=acamp:10k6_vid_entire_seq
<a id="inv___10k6_vid_entire_seq_acam_p_"></a>
### inv       @ 10k6_vid_entire_seq/acamp-->tf
python data/scripts/create_ipsc_tfrecord.py cfg=acamp:10k6_vid_entire_seq_inv
<a id="2_per_seq___inv_10k6_vid_entire_seq_acam_p_"></a>
#### 2_per_seq       @ inv/10k6_vid_entire_seq/acamp-->tf
python data/scripts/create_ipsc_tfrecord.py cfg=acamp:10k6_vid_entire_seq_inv_2_per_seq

<a id="20k6_5_video___acam_p_"></a>
## 20k6_5_video       @ acamp-->tf
python data/scripts/create_ipsc_tfrecord.py cfg=acamp:20k6_5_video
<a id="inv___20k6_5_video_acamp_"></a>
### inv       @ 20k6_5_video/acamp-->tf
python data/scripts/create_ipsc_tfrecord.py cfg=acamp:20k6_5_video_inv
<a id="2_per_seq___inv_20k6_5_video_acamp_"></a>
#### 2_per_seq       @ inv/20k6_5_video/acamp-->tf
python data/scripts/create_ipsc_tfrecord.py cfg=acamp:20k6_5_video_inv_2_per_seq


