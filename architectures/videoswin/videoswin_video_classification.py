import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from decord import VideoReader

from huggingface_hub import hf_hub_download
from mpl_toolkits.axes_grid1 import ImageGrid
warnings.simplefilter(action="ignore")
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DEVICE = 'GPU' # 'GPU', 'TPU'

if DEVICE == 'TPU':
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu="local")
    strategy = tf.distribute.TPUStrategy(tpu)
    keras.mixed_precision.set_global_policy("mixed_bfloat16")
else:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.optimizer.set_jit(True)
    keras.mixed_precision.set_global_policy("mixed_float16")
    [tf.config.experimental.set_memory_growth(pd, True) for pd in physical_devices]
    strategy = tf.distribute.MirroredStrategy()

HOME = os.getcwd()

"""## Utils"""

def denormalize(z):
    mean = np.array([123.675, 116.28, 103.53])
    variance = np.array([np.square(58.395), np.square(57.12), np.square(57.375)])
    std = np.sqrt(variance)
    x = (z * std) + mean
    x = x.clip(0, 255)
    return x

def show_sequence(seq, sample=8, title=''):
    '''Plot a sequence of images horizontally'''
    # clip is [T, H, W, 3]
    assert (seq.shape[-1] == 3)
    fig = plt.figure(figsize=(20, 2.5))
    fig.suptitle(title, fontsize=16)
    grid = ImageGrid(fig, 111, nrows_ncols=(1, sample), axes_pad=0.1)
    for ax, img in zip(grid, denormalize(seq)):
        ax.imshow(img.astype('uint8'))
        ax.set_axis_off()
    plt.show()
    return

hf_dataset_identifier = "innat/KineticsTop5"
filename = "kinetics_top5.zip" if DEVICE == 'TPU' else "kinetics_top5_tiny.zip"
file_path = hf_hub_download(
    repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset"
)
train_set = f'/data/KineticsTop5/train'
test_set = f'/data/KineticsTop5/test'
class_folders = os.listdir(train_set)
label2id = {label:i for i, label in enumerate(class_folders)}
id2label = {v:k for k, v in label2id.items()}

def create_dataframe(path):
    data = []

    for class_name in os.listdir(path):
        class_dir = os.path.join(path, class_name)
        if os.path.isdir(class_dir) and class_name in label2id:
            for video_file in os.listdir(class_dir):
                video_path = os.path.join(class_dir, video_file)
                data.append({
                    'video_path': os.path.abspath(video_path),
                    'label': label2id[class_name],
                    'class_name': class_name
                })

    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)
    return df

train_df = create_dataframe(train_set)
test_df = create_dataframe(test_set)
train_df.shape, test_df.shape

test_df.head()

"""# Data Loader

Reading the video data from dataloader in training time can be challenging. We can leverage [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) format, but due to the cost of disk space, we will skip that. For now, we will be using [`decord`](https://github.com/dmlc/decord) for video reader. It supports both GPUs and TPUs.


"""

input_size=224
num_frame=32
sampling_rate=2
batch_size=4 * strategy.num_replicas_in_sync

def read_video(file_path):
    vr = VideoReader(file_path.numpy().decode('utf-8'))
    frames = vr.get_batch(range(len(vr))).asnumpy()
    return format_frames(
        frames,
        output_size=(input_size, input_size)
    )

def format_frames(frame, output_size):
    frame = tf.image.convert_image_dtype(frame, tf.uint8)
    frame = tf.image.resize(frame, size=list(output_size))
    return frame

def load_video(file_path, label):
    video = tf.py_function(func=read_video, inp=[file_path], Tout=tf.float32)
    video.set_shape([None, None, None, 3])
    return video, tf.cast(label, dtype=tf.float32)

"""The raw **Kinetics** data comes with variable size of **input frame** and **frame size**. We will resize the video to the target shape (i.e. `224`). For **VideoSwin** model, the clip length is `32`. We will use frame rate `2` which is the stride or step from a frame to next frame. We will also take `1` clip from a video. However, in literature for evaluation, it's common to take `num_clips > 1` from each clip and average the results for final prediction For **VideoSwin**, check the official [note](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/db018fb8896251711791386bbd2127562fd8d6a6/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py#L48-L52) on this. But here for simplicity cause, we will skip that.


Sampling strategy for frame selection is another important factor which might need another attention. It can significantly differs the reported scores. In this notebook, we will be using [uniform sampling](https://gist.github.com/innat/205075992360d8d7a241c7f1013866a8) for frame selection.
"""

def uniform_temporal_subsample(
    x, num_samples, clip_idx, total_clips, frame_rate=1, temporal_dim=-4
):
    t = tf.shape(x)[temporal_dim]
    max_offset = t - num_samples * frame_rate
    step = max_offset // total_clips
    offset = clip_idx * step
    indices = tf.linspace(
        tf.cast(offset, tf.float32),
        tf.cast(offset + (num_samples-1) * frame_rate, tf.float32),
        num_samples
    )
    indices = tf.clip_by_value(indices, 0, tf.cast(t - 1, tf.float32))
    indices = tf.cast(tf.round(indices), tf.int32)
    return tf.gather(x, indices, axis=temporal_dim)


def clip_generator(
    image, num_frames=32, frame_rate=1, num_clips=1, crop_size=224
):
    clips_list = []
    for i in range(num_clips):
        frame = uniform_temporal_subsample(
            image, num_frames, i, num_clips, frame_rate=frame_rate, temporal_dim=0
        )
        clips_list.append(frame)

    video = tf.stack(clips_list)
    video = tf.reshape(
        video, [num_clips*num_frames, crop_size, crop_size, 3]
    )
    return video

processing_model = keras.Sequential(
    [
        layers.Normalization(
            mean=[123.675, 116.28, 103.53],
            variance=[np.square(58.395), np.square(57.12), np.square(57.375)]
        )
    ]
)

def create_dataloader(df, batch_size, shuffle=True, drop_reminder=True):
    ds = tf.data.Dataset.from_tensor_slices(
        (df['video_path'].values, df['label'].values)
    )
    ds = ds.repeat()
    ds = ds.shuffle(8 * batch_size) if shuffle else ds
    ds = ds.map(load_video, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(
        lambda x, y: (clip_generator(x, num_frame, sampling_rate, num_clips=1), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(batch_size, drop_remainder=drop_reminder)
    ds = ds.map(lambda x, y: (processing_model(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = create_dataloader(
    train_df, batch_size, shuffle=True
)

test_ds = create_dataloader(
    test_df, batch_size, shuffle=False
)

"""**Visualize**"""

temp_ds = create_dataloader(
    test_df, 8, shuffle=False
)
x, y = next(iter(temp_ds))

for i in range(len(x)):
    show_sequence(
        x.numpy()[i], sample=5, title=id2label[y[i].numpy()]
    )

"""# Model

As mentioned, we are using a custom **Kinetics-5** dataset. We will be using **VideoSwin** model with **Something-Something-V2** pretrained checkpoint. We will freeze the backbone and only train the custom head for downstream task.
"""

backbone_name = "TFVideoSwinB_SSV2_K400_P244_W1677_32x224"

# !wget -f'https://github.com/innat/VideoMAE/releases/download/v1.1/{backbone_name}.zip'
# !unzip -q f'/content/{backbone_name}.zip'

with strategy.scope():
    # fine-tuned model on kinetics-top5
    finetune_videoswin = keras.models.load_model(
        f'/content/data/{backbone_name}', compile=False
    )
    finetune_videoswin.trainable = False

    # downstream model
    model = keras.Sequential([
        finetune_videoswin,
        layers.Dense(
            len(class_folders), dtype='float32', activation=None
        )
    ])

    # compile the model
    model.compile(
        optimizer=keras.optimizers.AdamW(
          learning_rate=1e-3,
          weight_decay=1e-4
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
          keras.metrics.SparseCategoricalAccuracy(name='acc@1'),
          keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="acc@3")
        ]
    )

history = model.fit(
    train_ds,
    validation_data=test_ds,
    steps_per_epoch=len(train_df) // batch_size,
    validation_steps=len(test_df) // batch_size,
    epochs=5,
)

"""# Inference"""

test_sample = 'squat/6_jT33BqtK8.mp4' #@param ["opening_bottle/0npur3PQMHA.mp4", "reading_newspaper/26N2XO29YU8.mp4", "reading_book/BFYBAg9fSL4.mp4", "sneezing/6k-Uob0qzW4.mp4", "squat/6_jT33BqtK8.mp4"]

def read_video(file_path):
    vr = VideoReader(file_path)
    frames = vr.get_batch(range(len(vr))).asnumpy()
    return format_frames(
        frames,
        output_size=(input_size, input_size)
    )

def format_frames(frame, output_size):
    frame = tf.image.convert_image_dtype(frame, tf.uint8)
    frame = tf.image.resize(frame, size=list(output_size))
    return frame

sample_ds = read_video(os.path.join('/kaggle/working/data/test', test_sample))
sample_ds = clip_generator(sample_ds, num_frame, sampling_rate, num_clips=1)
sample_ds = processing_model(sample_ds)
sample_ds.shape

y_pred = model(sample_ds, training=False)
y_pred = y_pred.numpy().argmax(-1)[0]
y_pred

show_sequence(
    sample_ds.numpy().squeeze(), sample=5, title='predicted: ' + str(id2label[y_pred])
)