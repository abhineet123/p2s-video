<!-- MarkdownTOC -->

- [640-1       @ mnist](#640_1___mnis_t_)
    - [frame-0_1       @ 640-1](#frame_0_1___640_1_)
    - [frame-2_3       @ 640-1](#frame_2_3___640_1_)
- [640-3       @ mnist](#640_3___mnis_t_)
- [640-5       @ mnist](#640_5___mnis_t_)

<!-- /MarkdownTOC -->
<a id="640_1___mnis_t_"></a>
# 640-1       @ mnist-->p2s_tf_mnist
python3 data/scripts/create_ipsc_tfrecord.py cfg=mnist:640-1:12_1000:train:gz
<a id="frame_0_1___640_1_"></a>
## frame-0_1       @ 640-1-->p2s_tf_mnist
python3 data/scripts/create_ipsc_tfrecord.py cfg=mnist:640-1:12_1000:train:frame-0_1:gz
<a id="frame_2_3___640_1_"></a>
## frame-2_3       @ 640-1-->p2s_tf_mnist
python3 data/scripts/create_ipsc_tfrecord.py cfg=mnist:640-1:12_1000:train:frame-2_3:gz
python3 data/scripts/create_ipsc_tfrecord.py cfg=mnist:640-1:12_1000:test:frame-2_3:gz
<a id="640_3___mnis_t_"></a>
# 640-3       @ mnist-->p2s_tf_mnist
python3 data/scripts/create_ipsc_tfrecord.py cfg=mnist:640-3:12_1000:train:gz
<a id="640_5___mnis_t_"></a>
# 640-5       @ mnist-->p2s_tf_mnist
python3 data/scripts/create_ipsc_tfrecord.py cfg=mnist:640-5:12_1000:train:gz
python3 data/scripts/create_ipsc_tfrecord.py cfg=mnist:640-5:12_1000:test:gz

python3 data/scripts/create_ipsc_tfrecord.py cfg=mnist:640-5:12_1000:test:seq-0_5:frame-0_5:gz
