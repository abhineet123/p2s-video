# coding=utf-8
# Copyright 2022 The Pix2Seq Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utility functions for decoding example into features and labels."""

import tensorflow as tf


def get_feature_map_for_image():
    return {
        'image/encoded': tf.io.FixedLenFeature((), tf.string),
        'image/source_id': tf.io.FixedLenFeature((), tf.string, ''),
        'image/height': tf.io.FixedLenFeature((), tf.int64, -1),
        'image/width': tf.io.FixedLenFeature((), tf.int64, -1),
        'image/filename': tf.io.FixedLenFeature((), tf.string, ''),
        'image/n_runs': tf.io.FixedLenFeature((), tf.int64, -1),
    }


def get_feature_map_for_object_detection():
    return {
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
        'image/object/area': tf.io.VarLenFeature(tf.float32),
        'image/object/is_crowd': tf.io.VarLenFeature(tf.int64),
        'image/object/score': tf.io.VarLenFeature(tf.float32),
    }


def get_feature_map_for_video(length):
    return {
        'video/source_id': tf.io.FixedLenFeature((), tf.int64, -1),
        'video/height': tf.io.FixedLenFeature((), tf.int64, -1),
        'video/width': tf.io.FixedLenFeature((), tf.int64, -1),
        'video/size': tf.io.FixedLenFeature((2,), tf.int64, (-1, -1)),
        'video/num_frames': tf.io.FixedLenFeature((), tf.int64, -1),
        'video/file_names': tf.io.FixedLenFeature((length,), tf.string),
        'video/file_ids': tf.io.FixedLenFeature((length,), tf.int64),
        # 'video/file_names': tf.io.VarLenFeature(tf.string),
        # 'video/file_ids': tf.io.VarLenFeature(tf.int64),
    }


def get_feature_map_for_video_detection(vid_len):
    feat_dict = {
        'video/object/class/text': tf.io.VarLenFeature(tf.string),
        'video/object/class/label': tf.io.VarLenFeature(tf.int64),
        'video/object/is_crowd': tf.io.VarLenFeature(tf.int64),
        'video/object/score': tf.io.VarLenFeature(tf.float32),
    }

    for _id in range(vid_len):
        frame_feat_dict = {
            f'video/object/bbox/xmin-{_id}': tf.io.VarLenFeature(tf.float32),
            f'video/object/bbox/xmax-{_id}': tf.io.VarLenFeature(tf.float32),
            f'video/object/bbox/ymin-{_id}': tf.io.VarLenFeature(tf.float32),
            f'video/object/bbox/ymax-{_id}': tf.io.VarLenFeature(tf.float32),
            f'video/object/area-{_id}': tf.io.VarLenFeature(tf.float32),
        }
        feat_dict.update(frame_feat_dict)
    return feat_dict



def get_feature_map_for_instance_segmentation():
    return {
        'image/object/segmentation':
            tf.io.RaggedFeature(
                value_key='image/object/segmentation_v',
                dtype=tf.float32,
                partitions=[
                    tf.io.RaggedFeature.RowSplits('image/object/segmentation_sep')  # pytype: disable=attribute-error
                ]),
    }


def get_feature_map_for_keypoint_detection():
    return {
        'image/object/keypoints':
            tf.io.RaggedFeature(
                value_key='image/object/keypoints_v',
                dtype=tf.float32,
                partitions=[
                    tf.io.RaggedFeature.RowSplits('image/object/keypoints_sep')  # pytype: disable=attribute-error
                ]),
        'image/object/num_keypoints':
            tf.io.VarLenFeature(tf.int64),
    }


def get_feature_map_for_captioning():
    return {
        'image/caption': tf.io.VarLenFeature(tf.string),
    }


def decode_image(example, key='image/encoded'):
    """Decodes the image and set its static shape."""
    image = tf.io.decode_image(example[key], channels=3)
    image.set_shape([None, None, 3])
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def decode_mask(example, key='mask/encoded'):
    """Decodes the image and set its static shape."""
    mask = tf.io.decode_image(example[key], channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.uint8)

    """tf.image operations only seem to support 3D or greater images"""
    # mask = tf.expand_dims(mask, axis=-1)
    mask.set_shape([None, None, 1])
    return mask


def decode_boxes(example):
    """Concat box coordinates in the format of [ymin, xmin, ymax, xmax]."""
    xmin = example['image/object/bbox/xmin']
    xmax = example['image/object/bbox/xmax']
    ymin = example['image/object/bbox/ymin']
    ymax = example['image/object/bbox/ymax']
    return tf.stack([ymin, xmin, ymax, xmax], axis=-1)


def decode_areas(example):
    xmin = example['image/object/bbox/xmin']
    xmax = example['image/object/bbox/xmax']
    ymin = example['image/object/bbox/ymin']
    ymax = example['image/object/bbox/ymax']
    height = tf.cast(example['image/height'], dtype=tf.float32)
    width = tf.cast(example['image/width'], dtype=tf.float32)
    return tf.cond(
        tf.greater(tf.shape(example['image/object/area'])[0], 0),
        lambda: example['image/object/area'],
        lambda: (xmax - xmin) * (ymax - ymin) * height * width)


def decode_video_boxes(example, num_frames):
    bboxes = []
    for _id in range(num_frames):
        xmin = example[f'video/object/bbox/xmin-{_id}']
        xmax = example[f'video/object/bbox/xmax-{_id}']
        ymin = example[f'video/object/bbox/ymin-{_id}']
        ymax = example[f'video/object/bbox/ymax-{_id}']

        bboxes += [ymin, xmin, ymax, xmax]

    bboxes = tf.stack(bboxes, axis=-1)
    return bboxes


def decode_video_areas(example, num_frames):
    areas_list = []
    for _id in range(num_frames):
        area = example[f'video/object/area-{_id}']
        areas_list.append(area)
    return tf.stack(areas_list, axis=-1)


def decode_is_crowd(example):
    return tf.cond(
        tf.greater(tf.shape(example['image/object/is_crowd'])[0], 0),
        lambda: tf.cast(example['image/object/is_crowd'], dtype=tf.bool),
        lambda: tf.zeros_like(example['image/object/class/label'], dtype=tf.bool)
    )


def decode_scores(example):
    return tf.cond(
        tf.greater(tf.shape(example['image/object/score'])[0], 0),
        lambda: example['image/object/score'],
        lambda: tf.ones_like(example['image/object/class/label'],  # pylint: disable=g-long-lambda
                             dtype=tf.float32)
    )
