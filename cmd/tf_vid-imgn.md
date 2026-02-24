<!-- MarkdownTOC -->

- [vid       @ imgn](#vid___imgn_)
    - [len-2       @ vid](#len_2___vi_d_)
        - [4_per_seq_random_len_2       @ len-2/vid](#4_per_seq_random_len_2___len_2_vi_d_)
- [vid_val       @ imgn](#vid_val___imgn_)
    - [len-2       @ vid_val](#len_2___vid_va_l_)
        - [8_per_seq_random_len_2       @ len-2/vid_val](#8_per_seq_random_len_2___len_2_vid_va_l_)

<!-- /MarkdownTOC -->

<a id="vid___imgn_"></a>
# vid       @ imgn-->tf_vid-imgn
<a id="len_2___vi_d_"></a>
## len-2       @ vid-->tf_vid-imgn
python data/scripts/create_video_tfrecord.py cfg=imgn:vid:len-2:strd-1:asi-0
python data/scripts/create_video_tfrecord.py cfg=imgn:vid:len-2:strd-1:asi-0:vis
<a id="4_per_seq_random_len_2___len_2_vi_d_"></a>
### 4_per_seq_random_len_2       @ len-2/vid-->tf_vid-imgn
python data/scripts/create_video_tfrecord.py cfg=imgn:vid:4_per_seq_random_len_2:len-2:strd-2:asi-0


<a id="vid_val___imgn_"></a>
# vid_val       @ imgn-->tf_vid-imgn
<a id="len_2___vid_va_l_"></a>
## len-2       @ vid_val-->tf_vid-imgn
python data/scripts/create_video_tfrecord.py cfg=imgn:vid_val:len-2:strd-1:asi-1
<a id="8_per_seq_random_len_2___len_2_vid_va_l_"></a>
### 8_per_seq_random_len_2       @ len-2/vid_val-->tf_vid-imgn
python data/scripts/create_video_tfrecord.py cfg=imgn:vid_val:8_per_seq_random_len_2:len-2:strd-2:asi-0
