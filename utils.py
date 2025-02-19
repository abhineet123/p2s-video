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
"""General utils (used across different modules)."""

from concurrent import futures
import copy
import functools
import json
import logging
import math
import operator
import os
import time
import einops
import matplotlib
import matplotlib.cm
import numpy as np
from tqdm import tqdm
import shutil

import vocab
import tensorflow as tf

import subprocess


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


def check_ckpt_vars(cfg, trainer):
    cur_step_ = trainer.optimizer.iterations.numpy()
    ckpt_vars_pt = trainer.ckpt_vars_p

    if cur_step_ != 1 or ckpt_vars_pt is None:
        return

    name_to_shape_pt = trainer.name_to_shape_p
    trainer.checkpoint_manager.save(cur_step_)
    ckpt_vars, name_to_shape = save_ckpt_vars(cfg.model_dir)

    ckpt_names_pt = set(k for k in ckpt_vars_pt['name'] if 'optimizer' not in k)
    ckpt_names = set(k for k in ckpt_vars['name'] if 'optimizer' not in k)

    ckpt_names_file = os.path.join(cfg.model_dir, 'ckpt_names.txt')
    with open(ckpt_names_file, 'w') as fid:
        fid.write('\n'.join(ckpt_names))

    ckpt_names_pt_file = os.path.join(cfg.model_dir, 'ckpt_names_pt.txt')
    with open(ckpt_names_pt_file, 'w') as fid:
        fid.write('\n'.join(ckpt_names_pt))

    unmatched_names_model = ckpt_names - ckpt_names_pt
    unmatched_names_pt = ckpt_names_pt - ckpt_names

    matched_names = ckpt_names.intersection(ckpt_names_pt)

    if unmatched_names_model:
        unmatched_names_model_file = os.path.join(cfg.model_dir, 'unmatched_names_model.txt')
        with open(unmatched_names_model_file, 'w') as fid:
            fid.write('\n'.join(unmatched_names_model))

    if unmatched_names_pt:
        unmatched_names_pt_file = os.path.join(cfg.model_dir, 'unmatched_names_pt.txt')
        with open(unmatched_names_pt_file, 'w') as fid:
            fid.write('\n'.join(unmatched_names_pt))

    unmatched_shapes = {
        name: (name_to_shape_pt[name], name_to_shape[name])
        for name in matched_names
        if name_to_shape_pt[name] != name_to_shape[name]
    }

    if unmatched_shapes:
        names = list(unmatched_shapes.keys())
        unmatched_shapes_dict = dict(
            name=names,
            shapes=[unmatched_shapes[name] for name in names]
        )

        import pandas as pd
        unmatched_shapes_df = pd.DataFrame.from_dict(unmatched_shapes_dict)

        unmatched_shapes_csv = os.path.join(cfg.model_dir, 'unmatched_shapes.csv')
        print(f'saving unmatched_shapes to {unmatched_shapes_csv}')
        unmatched_shapes_df.to_csv(
            unmatched_shapes_csv,
            index=False,
        )


def json_serializable(val):
    try:
        json.dumps(val)
        return True
    except TypeError:
        return False


def tf_float32(t):
    return tf.cast(t, tf.float32)


def flatten_vid(t):
    """merge vid dim with batch dim"""
    shape_list = shape_as_list(t)
    inner_dims = shape_list[2:]
    new_bsz = shape_list[0] * shape_list[1]
    out_shape = [new_bsz, ] + inner_dims
    return tf.reshape(t, out_shape)


def unflatten_vid(t, vid_len):
    """split vid dim from combined vid+batch dim"""
    shape_list = shape_as_list(t)
    inner_dims = shape_list[1:]
    new_bsz = shape_list[0] // vid_len
    out_shape = [new_bsz, vid_len] + inner_dims
    return tf.reshape(t, out_shape)


def flatten_batch_dims(t, out_rank):
    """Merge first few dims to have out_rank."""
    if t.shape.rank == out_rank:
        return t
    if t.shape.rank < out_rank:
        raise ValueError('Tensor has rank %d. Expected at least %d' %
                         (t.shape.rank, out_rank))
    shape_list = shape_as_list(t)
    in_rank = len(shape_list)
    split = in_rank - out_rank + 1
    inner_dims = shape_list[split:]
    new_bsz = functools.reduce(operator.mul, shape_list[:split])
    out_shape = [new_bsz] + inner_dims
    return tf.reshape(t, out_shape)


def flatten_non_batch_dims(t, out_rank):
    """Merge last few dims to have out_rank."""
    if t.shape.rank == out_rank:
        return t
    if t.shape.rank < out_rank:
        raise ValueError('Tensor has rank %d. Expected at least %d' %
                         (t.shape.rank, out_rank))
    shape_list = shape_as_list(t)
    split = out_rank - 1
    inner_dims = shape_list[:split]
    new_last_dim = functools.reduce(operator.mul, shape_list[split:])
    out_shape = inner_dims + [new_last_dim]
    return tf.reshape(t, out_shape)


def int2bits(x, n, out_dtype=None):
    """Convert an integer x in (...) into bits in (..., n)."""
    x = tf.bitwise.right_shift(tf.expand_dims(x, -1), tf.range(n))
    x = tf.math.mod(x, 2)
    if out_dtype and out_dtype != x.dtype:
        x = tf.cast(x, out_dtype)
    return x


def bits2int(x, out_dtype):
    """Converts bits x in (..., n) into an integer in (...)."""
    x = tf.cast(x, out_dtype)
    x = tf.math.reduce_sum(x * (2 ** tf.range(tf.shape(x)[-1])), -1)
    return x


def images2patches(images, patch_size, dividers=(1,), keep_spatial=True):
    """Extract patches of same size in multiple resolutions."""
    _, height, width, _ = shape_as_list(images)  # (bsz, h, w, c)
    patches_list = []
    for d in dividers:
        inputs = images
        if d > 1:
            inputs = tf.image.resize(
                inputs, [height // d, width // d], method='bicubic', antialias=True)
        patches = tf.image.extract_patches(
            inputs,
            sizes=[1, patch_size, patch_size, 1],
            strides=[1, patch_size, patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID')
        if not keep_spatial:
            pshape = shape_as_list(patches)  # (bsz, #p1, #p2, p_size1*p_size2*3)
            patches = tf.reshape(patches, [pshape[0], pshape[1] * pshape[2], pshape[3]])
        patches_list.append(patches)
    return patches_list


def patches2images(patches_list, patch_size):
    """Putting list of patches back into images."""
    images_list = []
    for patches in patches_list:
        if patches.shape.ndims == 3:
            bsz, n_patches, d = shape_as_list(patches)  # (bsz, n_patches, d)
            s = tf.cast(tf.math.sqrt(tf.cast(n_patches, tf.float32)), tf.int32)
            patches = tf.reshape(patches, [bsz, s, s, d])
        else:
            assert patches.shape.ndims == 4  # (bsz, h, w, d)
        if patch_size > 1:
            images = tf.nn.depth_to_space(patches, patch_size)
        else:
            images = patches
        images_list.append(images)
    return images_list


def extract_patches(images, patch_size, patch_ordering='default'):
    """Extra patches from images and return a sequence of patch tokens."""
    tokens = tf.image.extract_patches(
        images,
        sizes=[1, *patch_size, 1],
        strides=[1, *patch_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID',
    )
    if patch_ordering == 'default':
        tokens = einops.rearrange(tokens, 'b h w c -> b (h w) c')
    elif patch_ordering == 'snake':
        tokens_list = []
        for i in range(tokens.shape[1]):
            if i % 2 == 0:
                tokens_list.append(tokens[:, i])
            else:
                tokens_list.append(tokens[:, i][:, ::-1])
        tokens = tf.stack(tokens_list, axis=1)
        tokens = einops.rearrange(tokens, 'b h w c -> b (h w) c')
    else:
        raise ValueError(f'Unknown patch_ordering {patch_ordering}')
    return tokens


def tokens2images(tokens, patch_size, image_height, image_width):
    """Tokens from `extract_patches` put back into images."""
    tokens = einops.rearrange(
        tokens,
        'b (h w) c -> b h w c',
        h=image_height // patch_size[0],
        w=image_width // patch_size[1],
    )
    return tf.nn.depth_to_space(tokens, patch_size[0])


def split_image_into_sub_images(x, num_sub_images_y, num_sub_images_x):
    """Split images into sub-images."""
    sub_image_shape_x = x.shape[1] // num_sub_images_x
    sub_image_shape_y = x.shape[2] // num_sub_images_y

    sub_images = []
    for i in range(num_sub_images_x):
        for j in range(num_sub_images_y):
            sub_image = x[
                        :,
                        i * sub_image_shape_x: (i + 1) * sub_image_shape_x,
                        j * sub_image_shape_y: (j + 1) * sub_image_shape_y,
                        :,
                        ]
            sub_images.append(sub_image)
    return sub_images


def images2subimages2tokens(images, sub_image_size, patch_size):
    """Split images into sub images, and returns them as patch tokens."""
    _, h, w, _ = images.shape
    sub_ratio_h, sub_ratio_w = h // sub_image_size[0], w // sub_image_size[1]
    images_seq = split_image_into_sub_images(images, sub_ratio_h, sub_ratio_w)
    tokens = [extract_patches(images_, patch_size) for images_ in images_seq]
    tokens = tf.stack(tokens, 1)
    return tokens  # (b, num_sub_images, num_patches, d)


def images2glimpses2tokens(
        images, sub_image_size, patch_size, mini_x=0, shuffle=False
):
    """Similar to images2subimages2tokens, with extra downsampling & shuffling."""
    bsz = tf.shape(images)[0]
    _, h, w, _ = images.shape
    sub_ratio_h, sub_ratio_w = h // sub_image_size[0], w // sub_image_size[1]
    images_sub = split_image_into_sub_images(images, sub_ratio_h, sub_ratio_w)
    if mini_x > 0:
        images_mini = tf.image.resize(images, [h // sub_ratio_h, w // sub_ratio_w])
        images_seq = [images_mini] * mini_x + images_sub
    else:
        images_seq = images_sub
    tokens = [extract_patches(images_, patch_size) for images_ in images_seq]
    tokens = tf.stack(tokens, 1)
    idx = tf.tile(
        tf.expand_dims(tf.range(sub_ratio_h * sub_ratio_w + mini_x), 0), [bsz, 1]
    )
    idx = tf.vectorized_map(tf.random.shuffle, idx) if shuffle else idx
    tokens = tf.gather(tokens, idx, axis=1, batch_dims=1)
    idx_hot = tf.one_hot(idx, sub_ratio_h * sub_ratio_w + mini_x)
    return tokens, tf.expand_dims(idx_hot, 2)  # (b, t, tokens, d), (b, t, 1, d)


def combine_sub_images(sub_images, num_sub_images_y, num_sub_images_x):
    combined_rows = []
    for i in range(0, num_sub_images_x * num_sub_images_y, num_sub_images_y):
        combined_rows.append(
            tf.concat(sub_images[i: i + num_sub_images_y], axis=2)
        )
    combined_image = tf.concat(combined_rows, axis=1)
    return combined_image


def tokens2subimages2images(tokens, sub_image_size, patch_size, image_size):
    """Tokens (b, t, n, d) from images2subimages2tokens put back to images."""
    num_sub_images_y = image_size[0] // sub_image_size[0]
    num_sub_images_x = image_size[1] // sub_image_size[1]
    sub_images = [
        tokens2images(
            tokens[:, i, :, :], patch_size, sub_image_size[0], sub_image_size[1]
        )
        for i in range(tokens.shape[1])
    ]
    images = combine_sub_images(sub_images, num_sub_images_y, num_sub_images_x)
    return images


def reduce_non_leading_dims(x, reduce_op=tf.reduce_sum, keepdims=True):
    return reduce_op(x, range(1, x.shape.ndims), keepdims=keepdims)


def tile_along_batch(t_in, factor):
    """Tile tensor in the first/batch dimension."""
    if factor == 1:
        return t_in
    t_out = tf.expand_dims(t_in, 1)
    multiples = [1] * t_out.shape.rank
    multiples[1] = factor
    t_out = tf.tile(t_out, multiples)
    shape = shape_as_list(t_out)
    t_out = tf.reshape(t_out, [shape[0] * shape[1]] + shape[2:])
    return t_out


def shape_as_list(t):
    # Assumes rank of `t` is statically known.
    shape = t.shape.as_list()
    dynamic_shape = tf.shape(t)
    return [
        shape[i] if shape[i] is not None else dynamic_shape[i]
        for i in range(len(shape))
    ]


def pad_to_max_len(data, max_len, dim, padding_token=0):
    """Pad the data tensor to max length on dim."""
    shape = shape_as_list(data)

    if shape[dim] >= max_len:
        return data

    padding_shape, new_shape = copy.copy(shape), copy.copy(shape)
    padding_shape[dim] = max_len - padding_shape[dim]
    new_shape[dim] = max_len
    paddings = tf.fill(padding_shape, tf.cast(padding_token, dtype=data.dtype))
    data_res = tf.reshape(tf.concat([data, paddings], axis=dim), new_shape)
    return data_res


# import pandas as pd
# import inspect
# np_dict = dict()


# def add_name(var_dict):
#     var_dict['np_dict'] = np_dict
#     for name, val in var_dict.items():
#         if callable(getattr(val, "numpy", None)):
#             setattr(val, "_pycharm_name_", name)

#
# def to_numpy(self):
#     try:
#         name = self._pycharm_name_
#     except AttributeError:
#         # name = 'self'
#         dtype = str(self.dtype)
#         shape = 'x'.join(str(x) for x in self.shape)
#         name = f'{dtype}_{shape}'
#
#         name_id = 0
#         while name in np_dict.keys():
#             name = f'{name}_{name_id}'
#             name_id += 1
#
#     np_dict[name] = self.numpy()
#     setattr(self, "numpy_arr", np_dict[name])
#     return np_dict[name]


def to_numpy(self):
    arr = self.numpy()
    setattr(self, "a___", arr)
    return arr


# def to_numpy(self, var_dict):
#     np_dict.clear()
#     for name, val in var_dict.items():
#         try:
#             np_dict[name.name] = val.numpy()
#             return np_dict[name]
#         except AttributeError:
#             try:
#                 np_dict[name] = val.cpu().numpy()
#             except AttributeError:
#                 continue
#     return np_dict

def quantize(coordinates, bins):
    """Quantization of (normalized) coordinates in [0, 1]."""
    coordinates = tf.cast(tf.round(coordinates * (bins - 1)), tf.int64)
    coordinates = tf.clip_by_value(coordinates, 0, bins - 1)
    return coordinates


def dequantize(boxes, bins):
    """Dequantization of discrete tokens of coordinates in [0, bins-1]."""
    boxes = tf.cast(boxes, tf.float32)
    boxes = boxes / (bins - 1)
    return boxes


def dequantize_video(boxes, bins):
    """Dequantization of discrete tokens of coordinates in [0, bins-1]."""
    boxes = tf.cast(boxes, tf.float32)
    boxes = boxes / (bins - 1)
    return boxes


def yx2xy(seq):
    x = np.asarray(seq[1::2]).reshape([-1, 1])
    y = np.asarray(seq[::2]).reshape([-1, 1])
    return np.concatenate([x, y], axis=-1).reshape([-1]).tolist()


def scale_points(points, scale):
    """Scales points.

    Args:
      points: Tensor with shape [num_points * 2], [batch, num_points * 2] or
        [batch, instances, num_points * 2] where points are organized in
        (y, x) format.
      scale: Tensor with shape [2] or [batch, 2].

    Returns:
      Tensor with same shape as points.
    """
    points_orig = points
    orig_shape = tf.shape(points)
    coords_len = points.shape[-1]
    if points.shape.rank == 1:
        points = tf.reshape(points, [coords_len // 2, 2])
    elif points.shape.rank == 2:
        points = tf.reshape(points, [-1, coords_len // 2, 2])
    else:
        points = tf.reshape(points, [-1, orig_shape[1], coords_len // 2, 2])
        scale = tf.expand_dims(scale, -2)
    points = points * scale
    points = tf.reshape(points, orig_shape)
    points = preserve_reserved_tokens(points, points_orig)
    return points


def preserve_reserved_tokens(points, points_orig) -> object:
    """Preserve reserved tokens in points according to points_orig."""
    return replace_reserved_tokens(points, points_orig, dict(zip(vocab.FLOATS,
                                                                 vocab.FLOATS)))


def boolean_mask(x, mask):
    masked_x = []
    for x_, mask_ in zip(x, mask):
        masked_x_ = tf.boolean_mask(x_, mask_)
        masked_x.append(masked_x_)
    return masked_x


def replace_reserved_tokens(seq, ref_seq, replacements):
    for key, replacement in replacements.items():
        seq = tf.where(
            tf.equal(ref_seq, key), tf.constant(replacement, seq.dtype), seq)
    return seq


def restore_from_checkpoint(model_dir, allow_partial, ignore_missing=False, **kwargs):
    """Restores the latest ckpt.

    Args:
      model_dir: directory of checkpoint to be restored.
      allow_partial: whether to allow partially restoring the checkpoint.
      **kwargs: arguments for `tf.train.Checkpoint` so it knows what to restore,
        e.g., `model=model, global_step=global_step, optimizer=optimizer`.

    Returns:
      latest_ckpt: The full path to the latest checkpoint or None if no
        checkpoint was found.
      checkpoint object
      verify_restored: function for verification
    """
    verify_restored = verify_existing = None
    checkpoint = tf.train.Checkpoint(**kwargs)
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    ckpt_vars = None
    name_to_shape = None
    if latest_ckpt:
        ckpt_vars, ckpt_vars_dict, name_to_shape = save_ckpt_vars(model_dir, latest_ckpt)
        logging.info('Restoring from latest checkpoint: %s', latest_ckpt)
        if allow_partial:
            status = checkpoint.restore(latest_ckpt).expect_partial()
        else:
            status = checkpoint.restore(latest_ckpt)
        verify_restored = status.assert_consumed
        verify_existing = status.assert_existing_objects_matched
    elif not ignore_missing:
        raise AssertionError(f'No checkpoint found in {model_dir}')

    return (latest_ckpt, checkpoint, verify_restored, verify_existing,
            ckpt_vars, name_to_shape)


def save_ckpt_vars(model_dir, latest_ckpt=None):
    if latest_ckpt is None:
        latest_ckpt = tf.train.latest_checkpoint(model_dir)

    ckpt_vars = tf.train.list_variables(latest_ckpt)
    ckpt_dict = dict(
        name=[ckpt_var[0] for ckpt_var in ckpt_vars],
        shape=[ckpt_var[1] for ckpt_var in ckpt_vars]
    )

    import pandas as pd
    ckpt_df = pd.DataFrame.from_dict(ckpt_dict)

    ckpt_vars_csv = latest_ckpt + "-ckpt_vars.csv"
    print(f'saving ckpt_vars to {ckpt_vars_csv}')
    ckpt_df.to_csv(
        ckpt_vars_csv,
        # sep='\t',
        index=False,
        # lineterminator='\n'
    )
    name_to_shape = {name: shape for name, shape in
                     zip(ckpt_dict['name'], ckpt_dict['shape'])}
    return ckpt_vars, ckpt_dict, name_to_shape


def check_checkpoint_restored(strict_verifiers, loose_verifiers=()):
    """Verification after model variables built."""
    strict_verifiers_new = []
    for strict_verifier in strict_verifiers:  # Stop exp from running.
        if strict_verifier:
            strict_verifier()
            strict_verifier = None
        strict_verifiers_new.append(strict_verifier)
    loose_verifiers_new = []
    for loose_verifier in loose_verifiers:  # Give warning in the log.
        if loose_verifier:
            try:
                loose_verifier()
            except AssertionError as e:
                print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                print('+++++++++++Checkpoint verification msg begin+++++++++++')
                print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                print(e)
                print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                print('+++++++++++Checkpoint verification msg ends+++++++++++')
                print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            loose_verifier = None
        loose_verifiers_new.append(loose_verifier)
    return strict_verifiers_new, loose_verifiers_new


def build_strategy(dist, use_tpu, master, training):
    """Returns a tf.distribute.Strategy."""
    if use_tpu:
        cluster = tf.distribute.cluster_resolver.TPUClusterResolver(master)
        tf.config.experimental_connect_to_cluster(cluster)
        topology = tf.tpu.experimental.initialize_tpu_system(cluster)
        logging.info('Topology:')
        logging.info('num_tasks: %d', topology.num_tasks)
        logging.info('num_tpus_per_task: %d', topology.num_tpus_per_task)
        strategy = tf.distribute.TPUStrategy(cluster)
    elif dist == 0:
        strategy = tf.distribute.get_strategy()
    elif dist == 1:  # For (multiple) GPUs.
        cross_device_ops = None  # tf.distribute.NcclAllReduce() by default
        # if the default cross_device_ops fails, try either of the following two
        # by uncommenting it.
        if not training:
            """default NcclAllReduce does not allow gather operation with 
            string tensors returned from postprocess_tpu"""
            # cross_device_ops = tf.distribute.HierarchicalCopyAllReduce()
            cross_device_ops = tf.distribute.ReductionToOneDevice()
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops)
        logging.info('Running using MirroredStrategy on %d replicas',
                     strategy.num_replicas_in_sync)
    elif dist == 2:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
    else:
        raise AssertionError(f'invalid dist: {dist}')

    return strategy


def get_train_steps(dataset, train_steps, train_epochs, train_batch_size):
    """Determine the number of training steps."""
    if train_steps:
        return train_steps
    num_train_examples = dataset.num_train_examples
    train_steps = num_train_examples * train_epochs // train_batch_size + 1
    return train_steps


def get_val_steps(dataset, eval_steps, eval_batch_size):
    """Determine the number of val steps."""
    num_eval_examples = dataset.num_eval_examples
    if not eval_steps and num_eval_examples:
        eval_steps = int(math.floor(num_eval_examples / eval_batch_size))
    return eval_steps


def get_eval_steps(dataset, eval_steps, eval_batch_size):
    """Determine the number of eval steps."""
    num_eval_examples = dataset.num_eval_examples
    if not eval_steps and num_eval_examples:
        # if (num_eval_examples % eval_batch_size != 0):
        #     raise ValueError('Only divisible eval batch sizes are currently supported.')
        eval_steps = int(math.ceil(num_eval_examples / eval_batch_size))
    return eval_steps

    # TODO(b/181662974): Revert this and support non-even batch sizes.
    # return eval_steps or int(
    #     math.ceil(num_eval_examples / eval_batch_size))
    # return eval_steps or (int(
    #     math.floor(num_eval_examples /
    #                eval_batch_size)) if num_eval_examples else None)


def get_checkpoint_steps(dataset, checkpoint_steps,
                         checkpoint_epochs, train_batch_size):
    """Determine the number of checkpoint steps."""
    num_train_examples = dataset.num_train_examples
    return checkpoint_steps or checkpoint_epochs * int(
        round(num_train_examples / train_batch_size))


def count_params(model, verbose=False):
    """Count parameters in `tf.keras.models.Model`."""
    if verbose:
        logging.info('Trainable variables:')
    total_params = 0
    for var in model.trainable_variables:
        if verbose:
            logging.info('%s\t%s', var.name, var.shape)
        total_params += np.prod(var.shape)
    if verbose:
        logging.info('Total number of parameters: {:,}'.format(total_params))
    return total_params


def merge_list_of_dict(list_of_dict):
    """Merge a list of dictionary (with shared keys) into a single dictionary."""
    if len(list_of_dict) == 1:
        return list_of_dict[0]
    dict_new = {}
    for key in list_of_dict[0].keys():
        dict_new[key] = tf.stack([x[key] for x in list_of_dict])
    return dict_new


def log_cfg(cfg):
    """Get the config and log it."""
    # logging.info('Config: %s', config)
    # Log config to the model directory.
    train_config_filepath = os.path.join(cfg.model_dir, 'config.json')
    eval_config_filepath = os.path.join(cfg.model_dir, f'config_{cfg.eval.tag}.json')

    config_filepath = train_config_filepath if cfg.training else eval_config_filepath
    if not tf.io.gfile.exists(config_filepath):
        tf.io.gfile.makedirs(cfg.model_dir)
        with tf.io.gfile.GFile(config_filepath, 'w') as f:
            f.write(cfg.to_json(indent=2, sort_keys=True))


def colorize(images, vmin=None, vmax=None, cmap=None):
    """Convert grayscaled images into colored images.

    Args:
      images: grayscale image tensor of shape (h, w) or (bsz, h, w).
      vmin: the minimum value of the range used for normalization.
        (Default: minimum of images)
      vmax: the maximum value of the range used for normalization.
        (Default: maximum of images)
      cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: 'gray')

    Returns:
      a colored image tensor of shape (h, w) or (bsz, h, w).
    """
    vmin = tf.reduce_min(images) if vmin is None else vmin
    vmax = tf.reduce_max(images) if vmax is None else vmax
    images = (images - vmin) / (vmax - vmin)
    images = tf.squeeze(images)  # squeeze last dim if it exists

    indices = tf.cast(tf.round(images * 255), tf.int32)
    cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'viridis')
    colors = tf.constant(cm.colors, dtype=tf.float32)
    return tf.gather(colors, indices)


def copy_dir(source_dir, destination_dir):
    """Copy files in the source directory to a destination directory."""
    logging.info('Copying files from %s to %s', source_dir, destination_dir)
    start = time.time()
    if tf.io.gfile.exists(destination_dir):
        logging.info('Removing existing files at %s', destination_dir)
        tf.io.gfile.rmtree(destination_dir)
    tf.io.gfile.makedirs(destination_dir)
    filenames = tf.io.gfile.listdir(source_dir)
    copy_fns = [
        lambda filename=filename: tf.io.gfile.copy(
            os.path.join(source_dir, filename),
            os.path.join(destination_dir, filename)) for filename in filenames
    ]
    run_in_parallel(copy_fns)
    logging.info('Copying %d files took %.2f seconds', len(filenames),
                 time.time() - start)


def run_in_parallel(fns):
    with futures.ThreadPoolExecutor() as executor:
        tasks = [executor.submit(fn) for fn in fns]
        for task in tasks:
            task.result()


def linux_path(*args, **kwargs):
    return os.path.join(*args, **kwargs).replace(os.sep, '/')


def connect_to_remote(info_file, remote, proxy):
    remote_info = read_remote_info(info_file)
    name0, name1, dst, ecr, key, port = remote_info[remote]

    username, server = dst.split('@')
    print(f'connecting to remote {remote}')

    import paramiko
    ssh = paramiko.SSHClient()

    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(server, username=username, port=port, timeout=5)

    ssh_main = None

    if proxy:
        print(f'connecting to proxy remote {proxy}')

        _, _, proxy_dst, _, _, proxy_port = remote_info[proxy]
        proxy_username, proxy_server = proxy_dst.split('@')

        vmtransport = ssh.get_transport()
        dest_addr = (proxy_server, proxy_port)  # edited#
        local_addr = (server, port)  # edited#
        try:
            vmchannel = vmtransport.open_channel(
                "direct-tcpip", dest_addr, local_addr)
        except paramiko.ssh_exception.ChannelException as e:
            print(f'\n\nfailed to open_channel:\n{e}\n')
            return None

        proxy_ssh = paramiko.SSHClient()
        proxy_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        proxy_ssh.connect(proxy_server, username=proxy_username, sock=vmchannel)

        ssh_main = ssh
        ssh = proxy_ssh
    return ssh, ssh_main


def get_remote_config(checkpoint_dir, info_file, remote, proxy):
    checkpoint_dir = os.path.abspath(checkpoint_dir)

    ret = connect_to_remote(info_file, remote, proxy)
    if ret is None:
        return None
    ssh, ssh_main = ret

    sftp = ssh.open_sftp()
    config_path = linux_path(checkpoint_dir, 'config.json')
    os.makedirs(checkpoint_dir, exist_ok=True)
    try:
        sftp.get(config_path, config_path)
    except FileNotFoundError:
        print(f'sftp file not found: {config_path}')

    ssh.close()
    if ssh_main is not None:
        ssh_main.close()


def sleep_with_pbar(hrs=0, mins=0, start=None, min_sleep=None):
    """
    :param min_sleep: minimum sleep time in minutes
    """
    sleep_mins = int(hrs * 60 + mins)

    if start is not None:
        sleep_secs = int(sleep_mins * 60)
        sleep_secs = start + sleep_secs - time.time()
        sleep_mins = int(max(0, sleep_secs // 60))

    if min_sleep is not None and sleep_mins < min_sleep:
        sleep_mins = min_sleep

    for _ in tqdm(range(sleep_mins), desc='sleeping', ncols=60):
        time.sleep(60)


def get_name(ckpt):
    return os.path.splitext(os.path.basename(ckpt))[0]


def get_ckpt_iter(k):
    return int(get_name(k).replace('ckpt-', ''))


def is_ckpt(k):
    return k.startswith('ckpt-') and (
        k.endswith('.data-00000-of-00001')
        # or
        # k.endswith('.index')
    )


def get_local_ckpt(checkpoint_dir, excluded_ckpts, ckpt_iter, create_copy):
    local_ckpts = [k for k in os.listdir(checkpoint_dir)
                   if os.path.isfile(linux_path(checkpoint_dir, k))
                   and is_ckpt(k) and get_name(k) not in excluded_ckpts]

    # eval_dirs = [k for k in os.listdir(checkpoint_dir)
    #              if k.startswith('ckpt-') and os.path.isdir(linux_path(checkpoint_dir, k))]
    # local_ckpts = [ckpt for ckpt in local_ckpts if
    #                not any(eval_dir.startswith(get_name(ckpt))
    #                        for eval_dir in eval_dirs)]
    if not local_ckpts:
        if ckpt_iter > 0:
            raise AssertionError(f'ckpt {ckpt_iter} not found in {checkpoint_dir}')
        return None

    if ckpt_iter > 0:
        local_ckpt_iters = list(map(get_ckpt_iter, local_ckpts))
        ckpt_idx = local_ckpt_iters.index(ckpt_iter)
    else:
        local_ckpts.sort(reverse=True, key=get_ckpt_iter)
        ckpt_idx = 0

    ckpt = local_ckpts[ckpt_idx]

    ckpt_info_file = linux_path(checkpoint_dir, 'checkpoint')
    ckpt_name = get_name(ckpt)
    with open(ckpt_info_file, 'w') as f:
        f.write(f'model_checkpoint_path: "{ckpt_name}"')

    ckpt_path = linux_path(checkpoint_dir, ckpt_name)
    if create_copy:
        ckpt_copy_dir = linux_path(checkpoint_dir, '#ckpt_copies')
        os.makedirs(ckpt_copy_dir, exist_ok=True)

        print(f'copying {ckpt_name} to {ckpt_copy_dir}')

        ckpt_idx = f'{ckpt_name}.index'
        files_to_transfer = [ckpt, ckpt_idx]
        for f in files_to_transfer:
            src_path = linux_path(checkpoint_dir, f)
            dst_path = linux_path(ckpt_copy_dir, f)
            print(f'{src_path} --> {dst_path}')
            shutil.copyfile(src_path, dst_path)

    return ckpt_path


def get_remote_ckpt(checkpoint_dir, info_file, remote, proxy):
    ret = connect_to_remote(info_file, remote, proxy)
    if ret is None:
        return None
    ssh, ssh_main = ret

    # key_file =  linux_path(os.path.expanduser('~'), '.ssh', 'id_rsa')
    # k = paramiko.RSAKey.from_private_key_file(key_file)
    # ssh.connect(hostname=server, username=username, port=port, pkey=k, timeout=5)

    cmd_to_execute = f'ls {checkpoint_dir}'
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(cmd_to_execute)

    ssh_stdout.channel.recv_exit_status()
    ssh_stdout_lines = ssh_stdout.readlines()

    remote_files = [k.strip() for k in ssh_stdout_lines]
    remote_ckpts = [k for k in remote_files if is_ckpt(k)]
    local_ckpts = [k for k in os.listdir(checkpoint_dir) if is_ckpt(k)]
    new_remote_ckpts = [ckpt for ckpt in remote_ckpts if ckpt not in local_ckpts]

    # new_remote_ckpts_str = '\n'.join(new_remote_ckpts)
    # print(f'\nnew_remote_ckpts:\n{new_remote_ckpts_str}\n')

    new_remote_ckpts.sort()

    ckpt_path = None

    if new_remote_ckpts:
        ckpt = new_remote_ckpts[-1]
        ckpt_name = os.path.splitext(os.path.basename(ckpt))[0]
        ckpt_path = linux_path(checkpoint_dir, ckpt_name)

        ckpt_idx = f'{ckpt_name}.index'
        files_to_transfer = [ckpt, ckpt_idx]

        files_to_transfer_str = '\n'.join(files_to_transfer)
        print(f'\nfiles_to_transfer:\n{files_to_transfer_str}\n')

        # files_to_transfer.append('checkpoint')
        sftp = ssh.open_sftp()
        for file in files_to_transfer:
            # print(f'transferring {file}')
            file_path = linux_path(checkpoint_dir, file)
            try:
                sftp.get(file_path, file_path)
            except FileNotFoundError:
                return None

        ckpt_info_file = linux_path(checkpoint_dir, 'checkpoint')
        with open(ckpt_info_file, 'w') as f:
            f.write(f'model_checkpoint_path: "{ckpt_name}"')
    ssh.close()
    if ssh_main is not None:
        ssh_main.close()
    return ckpt_path


def read_remote_info(info_file):
    info_path = linux_path(os.path.expanduser('~'), info_file)

    info_data = open(info_path, 'r').readlines()
    info_data = [k.strip() for k in info_data]

    dst_info = {}
    for datum in info_data:
        info = datum.split(' ')
        name0, name1, dst = info[:3]
        port = 22
        ecr = key = None
        if len(info) > 3:
            port = int(info[3])
        if len(info) > 4:
            ecr = info[4]
        if len(info) > 5:
            key = info[5]
        dst_info[name0] = [name0, name1, dst, ecr, key, port]
    return dst_info
