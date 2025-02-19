<!-- MarkdownTOC -->

- [detrac       @ tfrecord](#detrac___tfrecord_)
    - [0_19       @ detrac](#0_19___detrac_)
    - [0_9       @ detrac](#0_9___detrac_)
    - [0_48       @ detrac](#0_48___detrac_)
    - [49_68       @ detrac](#49_68___detrac_)
    - [49_85       @ detrac](#49_85___detrac_)
        - [100_per_seq_random       @ 49_85/detrac](#100_per_seq_random___49_85_detrac_)
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
<a id="detrac___tfrecord_"></a>
# detrac       @ tfrecord-->p2s_setup
<a id="0_19___detrac_"></a>
## 0_19       @ detrac-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=detrac:non_empty-0_19
<a id="0_9___detrac_"></a>
## 0_9       @ detrac-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=detrac:non_empty-0_9
<a id="0_48___detrac_"></a>
## 0_48       @ detrac-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=detrac:non_empty-0_48
<a id="49_68___detrac_"></a>
## 49_68       @ detrac-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=detrac:non_empty-49_68
<a id="49_85___detrac_"></a>
## 49_85       @ detrac-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=detrac:non_empty-49_85
<a id="100_per_seq_random___49_85_detrac_"></a>
### 100_per_seq_random       @ 49_85/detrac-->tf
python3 data/scripts/create_ipsc_tfrecord.py cfg=detrac:non_empty-49_85:100_per_seq_random

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


