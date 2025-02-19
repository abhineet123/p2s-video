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
"""Helper functions for creating TFRecord datasets."""

import hashlib
import io
import itertools
import multiprocessing
import os.path
from tqdm import tqdm

from absl import logging
import numpy as np
from PIL import Image

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

LOG_EVERY = 100


def convert_to_feature(value, value_type=None):
    """Converts the given python object to a tf.train.Feature.

    Args:
      value: int, float, bytes or a list of them.
      value_type: optional, if specified, forces the feature to be of the given
        type. Otherwise, type is inferred automatically. Can be one of
        ['bytes', 'int64', 'float', 'bytes_list', 'int64_list', 'float_list']

    Returns:
      feature: A tf.train.Feature object.
    """

    if value_type is None:

        element = value[0] if isinstance(value, list) else value

        # if isinstance(element, str):
        #     value_type = 'bytes'

        if isinstance(element, bytes):
            value_type = 'bytes'

        elif isinstance(element, (int, np.integer)):
            value_type = 'int64'

        elif element is None or isinstance(element, (float, np.floating)):
            value_type = 'float'
        else:
            raise ValueError('Cannot convert type {} to feature'.
                             format(type(element)))

        if isinstance(value, list):
            value_type = value_type + '_list'

    if value_type == 'int64':
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    elif value_type == 'int64_list':
        value = np.asarray(value).astype(np.int64).reshape(-1)
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    elif value_type == 'float':
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    elif value_type == 'float_list':
        value = np.asarray(value).astype(np.float32).reshape(-1)
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    elif value_type == 'bytes':
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    elif value_type == 'bytes_list':
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    else:
        raise ValueError('Unknown value_type parameter - {}'.format(value_type))


def video_info_to_feature_dict(height, width, file_ids, video_id, file_paths):
    img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]

    for file_path in file_paths:
        # realpath = os.path.realpath(file_path)

        assert os.path.exists(file_path), f"file_path does not exist: {file_path}"

        """
        imghdr is annoyingly buggy and now also deprecated
        https://stackoverflow.com/questions/36870661/imghdr-python-cant-detec-type-of-some-images-image-extension
        """
        # import imghdr
        # img_type = imghdr.what(file_path)

        from PIL import Image

        img_type = Image.open(file_path).format.lower()
        # print(f'img_type: {img_type}')

        if img_type is None:
            raise AssertionError(f"{file_path} is not an image")
        elif img_type not in img_type_accepted_by_tf:
            raise AssertionError(f"{file_path} is a {img_type}, not accepted by TensorFlow")

    file_paths = [file_path.encode('utf8') for file_path in file_paths]

    return {
        'video/height': convert_to_feature(height),
        'video/width': convert_to_feature(width),
        'video/size': convert_to_feature((height, width), value_type='int64_list'),
        'video/file_names': convert_to_feature(file_paths),
        'video/file_ids': convert_to_feature(file_ids),
        'video/num_frames': convert_to_feature(len(file_paths)),
        'video/source_id': convert_to_feature(video_id),
    }


def image_info_to_feature_dict(height, width, filename, image_id,
                               encoded_str, encoded_format):
    """Convert image information to a dict of features."""

    key = hashlib.sha256(encoded_str).hexdigest()

    return {
        'image/height': convert_to_feature(height),
        'image/width': convert_to_feature(width),
        'image/filename': convert_to_feature(filename.encode('utf8')),
        'image/source_id': convert_to_feature(str(image_id).encode('utf8')),
        'image/key/sha256': convert_to_feature(key.encode('utf8')),
        'image/encoded': convert_to_feature(encoded_str),
        'image/format': convert_to_feature(encoded_format.encode('utf8')),
    }

def video_seg_info_to_feature_dict(vid_id, height, width, vid_path, mask_vid_path,
                                   vid_len, seq):


    video_feature_dict = {
        'video/id': convert_to_feature(vid_id),
        'video/height': convert_to_feature(height),
        'video/width': convert_to_feature(width),
        'video/path': convert_to_feature(vid_path.encode('utf8')),
        'video/mask_path': convert_to_feature(mask_vid_path.encode('utf8')),
        'video/num_frames': convert_to_feature(vid_len),
        'video/seq': convert_to_feature(seq.encode('utf8')),
    }
    return video_feature_dict

def video_seg_frame_info_to_feature_dict(
        _id,
        image_id,
        frame_id,
        filename, encoded_img, encoded_format):
    key = hashlib.sha256(encoded_img).hexdigest()

    video_frame_feature_dict = {
        f'video/frame-{_id}/filename': convert_to_feature(filename.encode('utf8')),
        f'video/frame-{_id}/image_id': convert_to_feature(str(image_id).encode('utf8')),
        f'video/frame-{_id}/frame_id': convert_to_feature(int(frame_id)),
        f'video/frame-{_id}/key/sha256': convert_to_feature(key.encode('utf8')),
        f'video/frame-{_id}/encoded': convert_to_feature(encoded_img),
        f'video/frame-{_id}/format': convert_to_feature(encoded_format.encode('utf8')),
    }
    return video_frame_feature_dict


def read_image(image_path):
    pil_image = Image.open(image_path)
    return np.asarray(pil_image)


def encode_mask_as_png(mask):
    pil_image = Image.fromarray(mask)
    output_io = io.BytesIO()
    pil_image.save(output_io, format='PNG')
    return output_io.getvalue()


def write_tf_record_dataset(output_path,
                            annotation_iterator,
                            process_func,
                            num_shards,
                            iter_len,
                            multiple_processes=None,
                            unpack_arguments=True):

    writers = [
        tf.io.TFRecordWriter(
            output_path + '-%05d-of-%05d.tfrecord' % (i, num_shards))
        for i in range(num_shards)
    ]
    total_num_annotations_skipped = 0

    if multiple_processes <= 1:
        for idx, annotations_iter_ in tqdm(enumerate(annotation_iterator), total=iter_len):
            tf_example, num_annotations_skipped = process_func(*annotations_iter_)
            writers[idx % num_shards].write(tf_example.SerializeToString())
            total_num_annotations_skipped += num_annotations_skipped
    else:
        if multiple_processes is None or multiple_processes > 0:

            pool = multiprocessing.Pool(processes=multiple_processes)
            if unpack_arguments:
                tf_example_iterator = pool.starmap(process_func, annotation_iterator)
            else:
                tf_example_iterator = pool.i6map(process_func, annotation_iterator)
        else:
            if unpack_arguments:
                tf_example_iterator = itertools.starmap(process_func, annotation_iterator)
            else:
                tf_example_iterator = map(process_func, annotation_iterator)

        for idx, (tf_example, num_annotations_skipped) in tqdm(enumerate(
                tf_example_iterator), total=iter_len):
            # if idx % LOG_EVERY == 0:
            #     logging.info('On image %d', idx)

            total_num_annotations_skipped += num_annotations_skipped
            writers[idx % num_shards].write(tf_example.SerializeToString())

        if multiple_processes is None or multiple_processes > 0:
            pool.close()
            pool.join()

    for writer in writers:
        writer.close()

    logging.info('Finished writing, skipped %d annotations.',
                 total_num_annotations_skipped)
    return total_num_annotations_skipped


def check_and_make_dir(directory):
    """Creates the directory if it doesn't exist."""
    if not tf.io.gfile.isdir(directory):
        tf.io.gfile.makedirs(directory)
