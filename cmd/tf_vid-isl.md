<!-- MarkdownTOC -->

- [gram       @ tfrecord](#gram___tfrecord_)
    - [0_1       @ gram](#0_1___gram_)
        - [len-9       @ 0_1/gram](#len_9___0_1_gram_)
        - [len-14       @ 0_1/gram](#len_14___0_1_gram_)
            - [0_2000       @ len-14/0_1/gram](#0_2000___len_14_0_1_gra_m_)
            - [3000_5000       @ len-14/0_1/gram](#3000_5000___len_14_0_1_gra_m_)
        - [len-16       @ 0_1/gram](#len_16___0_1_gram_)
- [idot       @ gram](#idot___gram_)
    - [8_8       @ idot](#8_8___idot_)
- [detrac](#detra_c_)
    - [0_0       @ detrac](#0_0___detrac_)
    - [0_19       @ detrac](#0_19___detrac_)
        - [strd-2       @ 0_19/detrac](#strd_2___0_19_detra_c_)
        - [len-3       @ 0_19/detrac](#len_3___0_19_detra_c_)
        - [len-4       @ 0_19/detrac](#len_4___0_19_detra_c_)
        - [len-6       @ 0_19/detrac](#len_6___0_19_detra_c_)
        - [len-8       @ 0_19/detrac](#len_8___0_19_detra_c_)
        - [len-9       @ 0_19/detrac](#len_9___0_19_detra_c_)
        - [len-64       @ 0_19/detrac](#len_64___0_19_detra_c_)
    - [0_9       @ detrac](#0_9___detrac_)
        - [strd-2       @ 0_9/detrac](#strd_2___0_9_detrac_)
    - [49_68       @ detrac](#49_68___detrac_)
        - [len-2       @ 49_68/detrac](#len_2___49_68_detrac_)
        - [len-6       @ 49_68/detrac](#len_6___49_68_detrac_)
        - [len-9       @ 49_68/detrac](#len_9___49_68_detrac_)
        - [len-64       @ 49_68/detrac](#len_64___49_68_detrac_)
    - [0_48       @ detrac](#0_48___detrac_)
        - [len-2       @ 0_48/detrac](#len_2___0_48_detra_c_)
        - [len-4       @ 0_48/detrac](#len_4___0_48_detra_c_)
        - [len-8       @ 0_48/detrac](#len_8___0_48_detra_c_)
        - [len-12       @ 0_48/detrac](#len_12___0_48_detra_c_)
        - [len-16       @ 0_48/detrac](#len_16___0_48_detra_c_)
        - [len-24       @ 0_48/detrac](#len_24___0_48_detra_c_)
        - [len-28       @ 0_48/detrac](#len_28___0_48_detra_c_)
        - [len-32       @ 0_48/detrac](#len_32___0_48_detra_c_)
        - [len-40       @ 0_48/detrac](#len_40___0_48_detra_c_)
        - [len-48       @ 0_48/detrac](#len_48___0_48_detra_c_)
        - [len-56       @ 0_48/detrac](#len_56___0_48_detra_c_)
        - [len-64       @ 0_48/detrac](#len_64___0_48_detra_c_)
    - [49_85-strd-1       @ detrac](#49_85_strd_1___detrac_)
        - [len-2       @ 49_85-strd-1/detrac](#len_2___49_85_strd_1_detra_c_)
        - [len-4       @ 49_85-strd-1/detrac](#len_4___49_85_strd_1_detra_c_)
        - [len-8       @ 49_85-strd-1/detrac](#len_8___49_85_strd_1_detra_c_)
        - [len-16       @ 49_85-strd-1/detrac](#len_16___49_85_strd_1_detra_c_)
        - [len-32       @ 49_85-strd-1/detrac](#len_32___49_85_strd_1_detra_c_)
        - [len-40       @ 49_85-strd-1/detrac](#len_40___49_85_strd_1_detra_c_)
        - [len-48       @ 49_85-strd-1/detrac](#len_48___49_85_strd_1_detra_c_)
        - [len-56       @ 49_85-strd-1/detrac](#len_56___49_85_strd_1_detra_c_)
        - [len-64       @ 49_85-strd-1/detrac](#len_64___49_85_strd_1_detra_c_)
    - [49_85-strd-same       @ detrac](#49_85_strd_same___detrac_)
        - [len-2       @ 49_85-strd-same/detrac](#len_2___49_85_strd_same_detrac_)
            - [80_per_seq_random_len_2       @ len-2/49_85-strd-same/detrac](#80_per_seq_random_len_2___len_2_49_85_strd_same_detrac_)
        - [len-4       @ 49_85-strd-same/detrac](#len_4___49_85_strd_same_detrac_)
            - [80_per_seq_random_len_4       @ len-4/49_85-strd-same/detrac](#80_per_seq_random_len_4___len_4_49_85_strd_same_detrac_)
        - [len-8       @ 49_85-strd-same/detrac](#len_8___49_85_strd_same_detrac_)
            - [80_per_seq_random_len_8       @ len-8/49_85-strd-same/detrac](#80_per_seq_random_len_8___len_8_49_85_strd_same_detrac_)
        - [len-12       @ 49_85-strd-same/detrac](#len_12___49_85_strd_same_detrac_)
        - [len-16       @ 49_85-strd-same/detrac](#len_16___49_85_strd_same_detrac_)
            - [256_per_seq_random_len_16       @ len-16/49_85-strd-same/detrac](#256_per_seq_random_len_16___len_16_49_85_strd_same_detra_c_)
        - [len-32       @ 49_85-strd-same/detrac](#len_32___49_85_strd_same_detrac_)
            - [512_per_seq_random_len_32       @ len-32/49_85-strd-same/detrac](#512_per_seq_random_len_32___len_32_49_85_strd_same_detra_c_)
        - [len-40       @ 49_85-strd-same/detrac](#len_40___49_85_strd_same_detrac_)
        - [len-48       @ 49_85-strd-same/detrac](#len_48___49_85_strd_same_detrac_)
        - [len-56       @ 49_85-strd-same/detrac](#len_56___49_85_strd_same_detrac_)
        - [len-64       @ 49_85-strd-same/detrac](#len_64___49_85_strd_same_detrac_)

<!-- /MarkdownTOC -->

<a id="gram___tfrecord_"></a>
# gram       @ tfrecord-->p2s_setup
<a id="0_1___gram_"></a>
## 0_1       @ gram-->tf_vid-isl
<a id="len_9___0_1_gram_"></a>
### len-9       @ 0_1/gram-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=gram:0_1:len-9:strd-1
python data/scripts/create_video_tfrecord.py cfg=gram:0_1:len-9:strd-9

<a id="len_14___0_1_gram_"></a>
### len-14       @ 0_1/gram-->tf_vid-isl
<a id="0_2000___len_14_0_1_gra_m_"></a>
#### 0_2000       @ len-14/0_1/gram-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=gram:0_1:frame-0_2000:len-14:strd-1
<a id="3000_5000___len_14_0_1_gra_m_"></a>
#### 3000_5000       @ len-14/0_1/gram-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=gram:0_1:frame-3000_5000:len-14:strd-14

<a id="len_16___0_1_gram_"></a>
### len-16       @ 0_1/gram-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=gram:0_1:frame-0_2000:len-16:strd-1

<a id="idot___gram_"></a>
# idot       @ gram-->p2s_vid_tfrecord
python data/scripts/create_video_tfrecord.py cfg=idot:len-9:strd-1
<a id="8_8___idot_"></a>
## 8_8       @ idot-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=idot:8_8:len-2:strd-10:fg-10

python data/scripts/create_video_tfrecord.py cfg=idot:8_8:len-3:strd-20:fg-10

python data/scripts/create_video_tfrecord.py cfg=idot:8_8:len-6:strd-50:fg-10

<a id="detra_c_"></a>
# detrac
<a id="0_0___detrac_"></a>
## 0_0       @ detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_0:len-2:strd-1

<a id="0_19___detrac_"></a>
## 0_19       @ detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:len-2:strd-1

<a id="strd_2___0_19_detra_c_"></a>
### strd-2       @ 0_19/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:len-2:strd-2
<a id="len_3___0_19_detra_c_"></a>
### len-3       @ 0_19/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:len-3:strd-1
<a id="len_4___0_19_detra_c_"></a>
### len-4       @ 0_19/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:len-4:strd-1
<a id="len_6___0_19_detra_c_"></a>
### len-6       @ 0_19/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:len-6:strd-1
<a id="len_8___0_19_detra_c_"></a>
### len-8       @ 0_19/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:len-8:strd-1
<a id="len_9___0_19_detra_c_"></a>
### len-9       @ 0_19/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:len-9:strd-1
<a id="len_32___0_19_detra_c_"></a>
<a id="len_64___0_19_detra_c_"></a>
### len-64       @ 0_19/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_19:len-64:strd-1:asi-0

<a id="0_9___detrac_"></a>
## 0_9       @ detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_9:len-2:strd-1
<a id="strd_2___0_9_detrac_"></a>
### strd-2       @ 0_9/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_9:len-2:strd-2

<a id="49_68___detrac_"></a>
## 49_68       @ detrac-->tf_vid-isl
<a id="len_2___49_68_detrac_"></a>
### len-2       @ 49_68/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_68:len-2:strd-1:asi-2
`strd-2` 
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_68:len-2:strd-2

<a id="len_6___49_68_detrac_"></a>
### len-6       @ 49_68/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_68:len-6:strd-1:asi

<a id="len_9___49_68_detrac_"></a>
### len-9       @ 49_68/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_68:len-9:strd-1:asi-2

<a id="len_64___49_68_detrac_"></a>
### len-64       @ 49_68/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_68:len-64:strd-1:asi-2

<a id="0_48___detrac_"></a>
## 0_48       @ detrac-->tf_vid-isl
<a id="len_2___0_48_detra_c_"></a>
### len-2       @ 0_48/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_48:len-2:strd-1:asi-0
<a id="len_4___0_48_detra_c_"></a>
### len-4       @ 0_48/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_48:len-4:strd-1:asi-0
<a id="len_8___0_48_detra_c_"></a>
### len-8       @ 0_48/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_48:len-8:strd-1:asi-0
<a id="len_12___0_48_detra_c_"></a>
### len-12       @ 0_48/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_48:len-12:strd-1:asi-0
<a id="len_16___0_48_detra_c_"></a>
### len-16       @ 0_48/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_48:len-16:strd-1:asi-0
<a id="len_24___0_48_detra_c_"></a>
### len-24       @ 0_48/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_48:len-24:strd-1:asi-0
<a id="len_28___0_48_detra_c_"></a>
### len-28       @ 0_48/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_48:len-28:strd-1:asi-0
<a id="len_32___0_48_detra_c_"></a>
### len-32       @ 0_48/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_48:len-32:strd-1:asi-0
<a id="len_40___0_48_detra_c_"></a>
### len-40       @ 0_48/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_48:len-40:strd-1:asi-0
<a id="len_48___0_48_detra_c_"></a>
### len-48       @ 0_48/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_48:len-48:strd-1:asi-0
<a id="len_56___0_48_detra_c_"></a>
### len-56       @ 0_48/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_48:len-56:strd-1:asi-0
<a id="len_64___0_48_detra_c_"></a>
### len-64       @ 0_48/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-0_48:len-64:strd-1:asi-0

<a id="49_85_strd_1___detrac_"></a>
## 49_85-strd-1       @ detrac-->tf_vid-isl
<a id="len_2___49_85_strd_1_detra_c_"></a>
### len-2       @ 49_85-strd-1/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_85:len-2:strd-1:asi:strds-2
<a id="len_4___49_85_strd_1_detra_c_"></a>
### len-4       @ 49_85-strd-1/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_85:len-4:strd-1:asi:strds-4
<a id="len_8___49_85_strd_1_detra_c_"></a>
### len-8       @ 49_85-strd-1/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_85:len-8:strd-1:asi:strds-8
<a id="len_16___49_85_strd_1_detra_c_"></a>
### len-16       @ 49_85-strd-1/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_85:len-16:strd-1:asi:strds-16
<a id="len_32___49_85_strd_1_detra_c_"></a>
### len-32       @ 49_85-strd-1/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_85:len-32:strd-1:asi:strds-32
<a id="len_40___49_85_strd_1_detra_c_"></a>
### len-40       @ 49_85-strd-1/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_85:len-40:strd-1:asi-0
<a id="len_48___49_85_strd_1_detra_c_"></a>
### len-48       @ 49_85-strd-1/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_85:len-48:strd-1:asi-0
<a id="len_56___49_85_strd_1_detra_c_"></a>
### len-56       @ 49_85-strd-1/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_85:len-56:strd-1:asi-0
<a id="len_64___49_85_strd_1_detra_c_"></a>
### len-64       @ 49_85-strd-1/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_85:len-64:strd-1:asi-0

<a id="49_85_strd_same___detrac_"></a>
## 49_85-strd-same       @ detrac-->tf_vid-isl
<a id="len_2___49_85_strd_same_detrac_"></a>
### len-2       @ 49_85-strd-same/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_85:len-2:strd-2:asi-0
<a id="80_per_seq_random_len_2___len_2_49_85_strd_same_detrac_"></a>
#### 80_per_seq_random_len_2       @ len-2/49_85-strd-same/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_85:len-2:80_per_seq_random_len_2:strd-2:asi-0

<a id="len_4___49_85_strd_same_detrac_"></a>
### len-4       @ 49_85-strd-same/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_85:len-4:strd-4:asi-0
<a id="80_per_seq_random_len_4___len_4_49_85_strd_same_detrac_"></a>
#### 80_per_seq_random_len_4       @ len-4/49_85-strd-same/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_85:len-4:80_per_seq_random_len_4:strd-4:asi-0

<a id="len_8___49_85_strd_same_detrac_"></a>
### len-8       @ 49_85-strd-same/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_85:len-8:strd-8:asi-0
<a id="80_per_seq_random_len_8___len_8_49_85_strd_same_detrac_"></a>
#### 80_per_seq_random_len_8       @ len-8/49_85-strd-same/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_85:len-8:80_per_seq_random_len_8:strd-8:asi-0

<a id="len_12___49_85_strd_same_detrac_"></a>
### len-12       @ 49_85-strd-same/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_85:len-12:strd-12:asi-0

<a id="len_16___49_85_strd_same_detrac_"></a>
### len-16       @ 49_85-strd-same/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_85:len-16:strd-16:asi
<a id="256_per_seq_random_len_16___len_16_49_85_strd_same_detra_c_"></a>
#### 256_per_seq_random_len_16       @ len-16/49_85-strd-same/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_85:len-16:256_per_seq_random_len_16:strd-16:asi-0

<a id="len_32___49_85_strd_same_detrac_"></a>
### len-32       @ 49_85-strd-same/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_85:len-32:strd-32:asi-0
<a id="512_per_seq_random_len_32___len_32_49_85_strd_same_detra_c_"></a>
#### 512_per_seq_random_len_32       @ len-32/49_85-strd-same/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_85:len-32:512_per_seq_random_len_32:strd-32:asi-0

<a id="len_40___49_85_strd_same_detrac_"></a>
### len-40       @ 49_85-strd-same/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_85:len-40:strd-40:asi-0
<a id="len_48___49_85_strd_same_detrac_"></a>
### len-48       @ 49_85-strd-same/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_85:len-48:strd-48:asi-0
<a id="len_56___49_85_strd_same_detrac_"></a>
### len-56       @ 49_85-strd-same/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_85:len-56:strd-56:asi-0
<a id="len_64___49_85_strd_same_detrac_"></a>
### len-64       @ 49_85-strd-same/detrac-->tf_vid-isl
python data/scripts/create_video_tfrecord.py cfg=detrac:non_empty-49_85:len-64:strd-64:asi-0
