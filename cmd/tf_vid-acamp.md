<!-- MarkdownTOC -->

- [1k8_vid_entire_seq       @ acamp](#1k8_vid_entire_seq___acam_p_)
    - [inv       @ 1k8_vid_entire_seq](#inv___1k8_vid_entire_seq_)
        - [2_per_seq       @ inv/1k8_vid_entire_seq](#2_per_seq___inv_1k8_vid_entire_seq_)
        - [6_per_seq       @ inv/1k8_vid_entire_seq](#6_per_seq___inv_1k8_vid_entire_seq_)
        - [12_per_seq       @ inv/1k8_vid_entire_seq](#12_per_seq___inv_1k8_vid_entire_seq_)
- [10k6_vid_entire_seq       @ acamp](#10k6_vid_entire_seq___acam_p_)
    - [inv       @ 10k6_vid_entire_seq](#inv___10k6_vid_entire_se_q_)
        - [2_per_seq       @ inv/10k6_vid_entire_seq](#2_per_seq___inv_10k6_vid_entire_se_q_)
        - [12_per_seq       @ inv/10k6_vid_entire_seq](#12_per_seq___inv_10k6_vid_entire_se_q_)
- [20k6_5_video       @ acamp](#20k6_5_video___acam_p_)
    - [inv       @ 20k6_5_video](#inv___20k6_5_video_)
        - [2_per_seq       @ inv/20k6_5_video](#2_per_seq___inv_20k6_5_video_)
        - [12_per_seq       @ inv/20k6_5_video](#12_per_seq___inv_20k6_5_video_)
- [2_per_seq_dbg_bear       @ acamp](#2_per_seq_dbg_bear___acam_p_)

<!-- /MarkdownTOC -->

<a id="1k8_vid_entire_seq___acam_p_"></a>
# 1k8_vid_entire_seq       @ acamp-->tf_vid-acamp
python data/scripts/create_video_tfrecord.py cfg=acamp:1k8_vid_entire_seq:len-2:strd-1:asi-2
python data/scripts/create_video_tfrecord.py cfg=acamp:1k8_vid_entire_seq:len-2:strd-2
<a id="inv___1k8_vid_entire_seq_"></a>
## inv       @ 1k8_vid_entire_seq-->tf_vid-acamp
python data/scripts/create_video_tfrecord.py cfg=acamp:1k8_vid_entire_seq_inv:len-2:strd-1:asi-2
<a id="2_per_seq___inv_1k8_vid_entire_seq_"></a>
### 2_per_seq       @ inv/1k8_vid_entire_seq-->tf_vid-acamp
python data/scripts/create_video_tfrecord.py cfg=acamp:1k8_vid_entire_seq_inv_2_per_seq:len-2:strd-1:asi
<a id="6_per_seq___inv_1k8_vid_entire_seq_"></a>
### 6_per_seq       @ inv/1k8_vid_entire_seq-->tf_vid-acamp
python data/scripts/create_video_tfrecord.py cfg=acamp:1k8_vid_entire_seq_inv_6_per_seq:len-2:strd-1:asi
<a id="12_per_seq___inv_1k8_vid_entire_seq_"></a>
### 12_per_seq       @ inv/1k8_vid_entire_seq-->tf_vid-acamp
python data/scripts/create_video_tfrecord.py cfg=acamp:1k8_vid_entire_seq_inv_12_per_seq:len-2:strd-1:asi

<a id="10k6_vid_entire_seq___acam_p_"></a>
# 10k6_vid_entire_seq       @ acamp-->tf_vid-acamp
python data/scripts/create_video_tfrecord.py cfg=acamp:10k6_vid_entire_seq:len-2:strd-1:asi
<a id="inv___10k6_vid_entire_se_q_"></a>
## inv       @ 10k6_vid_entire_seq-->tf_vid-acamp
python data/scripts/create_video_tfrecord.py cfg=acamp:10k6_vid_entire_seq_inv:len-2:strd-1:asi
<a id="2_per_seq___inv_10k6_vid_entire_se_q_"></a>
### 2_per_seq       @ inv/10k6_vid_entire_seq-->tf_vid-acamp
python data/scripts/create_video_tfrecord.py cfg=acamp:10k6_vid_entire_seq_inv_2_per_seq:len-2:strd-1:asi
<a id="12_per_seq___inv_10k6_vid_entire_se_q_"></a>
### 12_per_seq       @ inv/10k6_vid_entire_seq-->tf_vid-acamp
python data/scripts/create_video_tfrecord.py cfg=acamp:10k6_vid_entire_seq_inv_12_per_seq:len-2:strd-1:asi

<a id="20k6_5_video___acam_p_"></a>
# 20k6_5_video       @ acamp-->tf_vid-acamp
python data/scripts/create_video_tfrecord.py cfg=acamp:20k6_5_video:len-2:strd-1:asi
<a id="inv___20k6_5_video_"></a>
## inv       @ 20k6_5_video-->tf_vid-acamp
python data/scripts/create_video_tfrecord.py cfg=acamp:20k6_5_video_inv:len-2:strd-1:asi
<a id="2_per_seq___inv_20k6_5_video_"></a>
### 2_per_seq       @ inv/20k6_5_video-->tf_vid-acamp
python data/scripts/create_video_tfrecord.py cfg=acamp:20k6_5_video_inv_2_per_seq:len-2:strd-1:asi
<a id="12_per_seq___inv_20k6_5_video_"></a>
### 12_per_seq       @ inv/20k6_5_video-->tf_vid-acamp
python data/scripts/create_video_tfrecord.py cfg=acamp:20k6_5_video_inv_12_per_seq:len-2:strd-1:asi

<a id="2_per_seq_dbg_bear___acam_p_"></a>
# 2_per_seq_dbg_bear       @ acamp-->tf_vid-acamp
python data/scripts/create_video_tfrecord.py cfg=acamp:2_per_seq_dbg_bear:len-2:strd-1
