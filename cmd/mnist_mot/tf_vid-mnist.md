<!-- MarkdownTOC -->

- [640-1       @ mnist](#640_1___mnis_t_)
    - [len-2       @ 640-1](#len_2___640_1_)
    - [test       @ 640-1](#test___640_1_)
        - [strd-2       @ test/640-1](#strd_2___test_640_1_)
    - [len-3       @ 640-1](#len_3___640_1_)
    - [test       @ 640-1](#test___640_1__1)
        - [strd-3       @ test/640-1](#strd_3___test_640_1_)
    - [len-9       @ 640-1](#len_9___640_1_)
- [640-3       @ mnist](#640_3___mnis_t_)
- [640-5       @ mnist](#640_5___mnis_t_)
    - [len-2       @ 640-5](#len_2___640_5_)
    - [len-3       @ 640-5](#len_3___640_5_)
    - [len-4       @ 640-5](#len_4___640_5_)
    - [len-6       @ 640-5](#len_6___640_5_)
    - [len-9       @ 640-5](#len_9___640_5_)

<!-- /MarkdownTOC -->
<a id="640_1___mnis_t_"></a>
# 640-1       @ mnist-->p2s_vid_tf_mnist
<a id="len_2___640_1_"></a>
## len-2       @ 640-1-->p2s_vid_tf_mnist
python data/scripts/create_video_tfrecord.py cfg=mnist:640-1:12_1000:gz:len-2:strd-1:train
<a id="test___640_1_"></a>
## test       @ 640-1-->p2s_vid_tf_mnist
python data/scripts/create_video_tfrecord.py cfg=mnist:640-1:12_1000:gz:len-2:strd-1:test
<a id="strd_2___test_640_1_"></a>
### strd-2       @ test/640-1-->p2s_vid_tf_mnist
python data/scripts/create_video_tfrecord.py cfg=mnist:640-1:12_1000:gz:len-2:strd-2:test

<a id="len_3___640_1_"></a>
## len-3       @ 640-1-->p2s_vid_tf_mnist
python data/scripts/create_video_tfrecord.py cfg=mnist:640-1:12_1000:gz:len-3:strd-1:proc-12
<a id="test___640_1__1"></a>
## test       @ 640-1-->p2s_vid_tf_mnist
python data/scripts/create_video_tfrecord.py cfg=mnist:640-1:12_1000:gz:len-3:strd-1:test
<a id="strd_3___test_640_1_"></a>
### strd-3       @ test/640-1-->p2s_vid_tf_mnist
python data/scripts/create_video_tfrecord.py cfg=mnist:640-1:12_1000:gz:len-3:strd-3:test

<a id="len_9___640_1_"></a>
## len-9       @ 640-1-->p2s_vid_tf_mnist
python data/scripts/create_video_tfrecord.py cfg=mnist:640-1:12_1000:gz:len-9:strd-1:proc-12:train

<a id="640_3___mnis_t_"></a>
# 640-3       @ mnist-->p2s_vid_tf_mnist
python data/scripts/create_video_tfrecord.py cfg=mnist:640-3:12_1000:gz:len-2:strd-1:proc-6

<a id="640_5___mnis_t_"></a>
# 640-5       @ mnist-->p2s_vid_tf_mnist
<a id="len_2___640_5_"></a>
## len-2       @ 640-5-->p2s_vid_tf_mnist
python data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-2:strd-1:proc-12:train

python data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-2:strd-1:proc-12:test
python data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-2:strd-2:proc-12:test

<a id="len_3___640_5_"></a>
## len-3       @ 640-5-->p2s_vid_tf_mnist
python data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-3:strd-1:proc-12:train

python data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-3:strd-1:proc-12:test
python data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-3:strd-3:proc-12:test

<a id="len_4___640_5_"></a>
## len-4       @ 640-5-->p2s_vid_tf_mnist
python data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-4:strd-1:proc-12:train

python data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-4:strd-1:proc-12:test
python data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-4:strd-4:proc-12:test
<a id="len_6___640_5_"></a>
## len-6       @ 640-5-->p2s_vid_tf_mnist
python data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-6:strd-1:proc-12:train

python data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-6:strd-1:proc-12:test
python data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-6:strd-6:proc-12:test

<a id="len_9___640_5_"></a>
## len-9       @ 640-5-->p2s_vid_tf_mnist
python data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-9:strd-1:proc-12:train

python data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-9:strd-1:proc-12:test
python data/scripts/create_video_tfrecord.py cfg=mnist:640-5:12_1000:gz:len-9:strd-9:proc-12:test
