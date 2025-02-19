<!-- MarkdownTOC -->

- [detrac       @ tfrecord](#detrac___tfrecord_)
    - [0_19       @ detrac](#0_19___detrac_)
    - [0_9       @ detrac](#0_9___detrac_)
    - [49_68       @ detrac](#49_68___detrac_)
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
    - [10k6_vid_entire_seq       @ acamp](#10k6_vid_entire_seq___acam_p_)
    - [20k6_5_video       @ acamp](#20k6_5_video___acam_p_)
    - [1k8_vid_entire_seq_inv       @ acamp](#1k8_vid_entire_seq_inv___acam_p_)
    - [10k6_vid_entire_seq_inv       @ acamp](#10k6_vid_entire_seq_inv___acam_p_)
    - [20k6_5_video_inv       @ acamp](#20k6_5_video_inv___acam_p_)

<!-- /MarkdownTOC -->
<a id="detrac___tfrecord_"></a>
# detrac       @ tfrecord-->p2s_setup
<a id="0_19___detrac_"></a>
## 0_19       @ detrac-->tf-ipsc
python3 data/scripts/create_ipsc_tfrecord.py cfg=detrac:non_empty-0_19
<a id="0_9___detrac_"></a>
## 0_9       @ detrac-->tf-ipsc
python3 data/scripts/create_ipsc_tfrecord.py cfg=detrac:non_empty-0_9
<a id="49_68___detrac_"></a>
## 49_68       @ detrac-->tf-ipsc
python3 data/scripts/create_ipsc_tfrecord.py cfg=detrac:non_empty-49_68

<a id="ipsc___tfrecord_"></a>
# ipsc       @ tfrecord-->p2s_setup
<a id="0_1___ipsc_"></a>
## 0_1       @ ipsc-->tf-ipsc
python3 data/scripts/create_ipsc_tfrecord.py cfg=ipsc:0_1:mask
<a id="2_3___ipsc_"></a>
## 2_3       @ ipsc-->tf-ipsc
python3 data/scripts/create_ipsc_tfrecord.py cfg=ipsc:2_3
<a id="16_53___ipsc_"></a>
## 16_53       @ ipsc-->tf-ipsc
python3 data/scripts/create_ipsc_tfrecord.py cfg=ipsc:16_53
<a id="0_37___ipsc_"></a>
## 0_37       @ ipsc-->tf-ipsc
python3 data/scripts/create_ipsc_tfrecord.py cfg=ipsc:0_37
<a id="38_53___ipsc_"></a>
<a id="54_126___ipsc_"></a>
## 54_126       @ ipsc-->tf-ipsc
python3 data/scripts/create_ipsc_tfrecord.py cfg=ipsc:54_126
python3 data/scripts/create_ipsc_tfrecord.py cfg=ipsc:54_126:strd-5
python3 data/scripts/create_ipsc_tfrecord.py cfg=ipsc:54_126:strd-8

<a id="0_1___ipsc__1"></a>
## 0_1       @ ipsc-->tf-ipsc
python data/scripts/create_ipsc_tfrecord.py --ann_file=ext_reorg_roi_g2_0_1.json --n_proc=0
<a id="0_15___ipsc_"></a>
## 0_15       @ ipsc-->tf-ipsc
python3 data/scripts/create_ipsc_tfrecord.py cfg=ipsc:0_15
<a id="38_53___ipsc_"></a>
## 38_53       @ ipsc-->tf-ipsc
python3 data/scripts/create_ipsc_tfrecord.py --ann_file=ext_reorg_roi_g2_38_53.json --n_proc=0

<a id="acamp_"></a>
# acamp
<a id="1k8_vid_entire_seq___acam_p_"></a>
## 1k8_vid_entire_seq       @ acamp-->tf-ipsc
python data/scripts/create_ipsc_tfrecord.py cfg=acamp:1k8_vid_entire_seq

<a id="10k6_vid_entire_seq___acam_p_"></a>
## 10k6_vid_entire_seq       @ acamp-->tf-ipsc
python data/scripts/create_ipsc_tfrecord.py cfg=acamp:10k6_vid_entire_seq

<a id="20k6_5_video___acam_p_"></a>
## 20k6_5_video       @ acamp-->tf-ipsc
python data/scripts/create_ipsc_tfrecord.py cfg=acamp:20k6_5_video

<a id="1k8_vid_entire_seq_inv___acam_p_"></a>
## 1k8_vid_entire_seq_inv       @ acamp-->tf-ipsc
python data/scripts/create_ipsc_tfrecord.py cfg=acamp:1k8_vid_entire_seq_inv
<a id="10k6_vid_entire_seq_inv___acam_p_"></a>
## 10k6_vid_entire_seq_inv       @ acamp-->tf-ipsc
python data/scripts/create_ipsc_tfrecord.py cfg=acamp:10k6_vid_entire_seq_inv
<a id="20k6_5_video_inv___acam_p_"></a>
## 20k6_5_video_inv       @ acamp-->tf-ipsc
python data/scripts/create_ipsc_tfrecord.py cfg=acamp:20k6_5_video_inv

