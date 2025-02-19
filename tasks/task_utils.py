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
"""Common task utils."""

import json
from typing import Optional, Any, Dict

import cv2

import utils
import vocab
import tensorflow as tf

import numpy as np
import scipy

from PIL import Image

from tasks.visualization import vis_utils
from eval_utils import col_bgr, resize_ar


def split_runs(run_ids, starts, lengths, max_length):
    """divide over-long runs into segments"""
    new_starts_ = []
    new_lengths_ = []
    # starts_, lengths_ = list(starts), list(lengths)
    for _id in run_ids:
        start, length = starts[_id], lengths[_id]

        new_starts_.append(start)
        new_lengths_.append(max_length)

        residual_length = length - max_length
        start_ = start
        length_ = max_length
        while True:
            start_ += length_
            new_starts_.append(start_)

            length_ = min(residual_length, max_length)
            new_lengths_.append(length_)

            residual_length -= length_
            if residual_length <= 0:
                break

        # if _id < len(lengths) - 1:
        #     cmb = np.stack((starts, lengths), axis=1)
        #     cmb_new = np.stack((new_starts_, new_lengths_), axis=1)
        #     print()

    valid_starts = [v for i, v in enumerate(starts) if i not in run_ids]
    valid_lengths = [v for i, v in enumerate(lengths) if i not in run_ids]

    valid_starts += new_starts_
    valid_lengths += new_lengths_

    starts, lengths = np.asarray(valid_starts), np.asarray(valid_lengths)

    sort_idx = np.argsort(starts)
    starts = starts[sort_idx]
    lengths = lengths[sort_idx]

    return starts, lengths


def split_runs_tf(run_ids, starts, lengths, max_length):
    """divide over-long runs into segments"""
    new_starts_ = []
    new_lengths_ = []
    # starts_, lengths_ = list(starts), list(lengths)
    for _id in run_ids:
        start, length = starts[_id], lengths[_id]

        new_starts_.append(start)
        new_lengths_.append(max_length)

        residual_length = length - max_length
        start_ = start
        length_ = max_length
        while True:
            start_ += length_
            new_starts_.append(start_)

            length_ = min(residual_length, max_length)
            new_lengths_.append(length_)

            residual_length -= length_
            if residual_length <= 0:
                break

        # if _id < len(lengths) - 1:
        #     cmb = np.stack((starts, lengths), axis=1)
        #     cmb_new = np.stack((new_starts_, new_lengths_), axis=1)
        #     print()

    valid_starts = [v for i, v in enumerate(starts) if i not in run_ids]
    valid_lengths = [v for i, v in enumerate(lengths) if i not in run_ids]

    valid_starts += new_starts_
    valid_lengths += new_lengths_

    starts, lengths = tf.experimental.numpy.asarray(valid_starts), tf.experimental.numpy.asarray(valid_lengths)

    sort_idx = tf.experimental.numpy.argsort(starts)
    starts = starts[sort_idx]
    lengths = lengths[sort_idx]

    return starts, lengths


def read_frame(vid_reader, frame_id, vid_path=None):
    vid_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    assert vid_reader.get(cv2.CAP_PROP_POS_FRAMES) == frame_id, "Failed to set frame index in video"
    ret, image = vid_reader.read()
    if not ret:
        msg = f'Frame {frame_id} could not be read'
        if vid_path is not None:
            msg = f'{vid_path} : {msg}'
        raise AssertionError(msg)
    return image


def load_video(vid_path, seq=''):
    vid_reader = cv2.VideoCapture()
    if not vid_reader.open(vid_path):
        raise AssertionError(f'Video file could not be opened: {vid_path}')

    num_frames = int(vid_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_height = int(vid_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_width = int(vid_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    if seq:
        print(f'\n{seq}: loaded {vid_width}x{vid_height} video with {num_frames} frames from {vid_path}')

    return vid_reader, vid_width, vid_height, num_frames


def mask_to_binary(mask):
    return (mask > 0).astype(np.uint8) * 255


def vid_mask_to_gs(vid_mask, check=True, copy=False):
    if len(vid_mask.shape) == 4:
        vid_mask_b = vid_mask[..., 0].squeeze()
        if check:
            vid_mask_g = vid_mask[..., 1].squeeze()
            vid_mask_r = vid_mask[..., 2].squeeze()

            assert np.array_equal(vid_mask_b, vid_mask_g), "mask_b and mask_g mismatch"
            assert np.array_equal(vid_mask_b, vid_mask_r), "mask_b and mask_r mismatch"

        vid_mask_gs = np.copy(vid_mask_b) if copy else vid_mask_b
    else:
        vid_mask_gs = np.copy(vid_mask) if copy else vid_mask
    return vid_mask_gs


def mask_to_gs(mask, check=True, copy=True):
    if len(mask.shape) == 2:
        return mask.copy() if copy else mask
    mask_b = mask[..., 0].squeeze()
    if check:
        mask_g = mask[..., 1].squeeze()
        mask_r = mask[..., 2].squeeze()
        assert np.array_equal(mask_b, mask_g), "mask_b and mask_g mismatch"
        assert np.array_equal(mask_b, mask_r), "mask_b and mask_r mismatch"
    return mask_b.copy() if copy else mask_b


def check_instance_wise_rle_tokens(
        image, mask, mask_sub, rle_tokens, n_classes,
        starts_2d,
        starts_offset, lengths_offset, class_offset,
        max_length,
        subsample,
        flat_order,
        class_to_col, is_vis):
    mask_gt = mask_to_gs(mask, copy=True)
    mask_gt_sub = mask_to_gs(mask_sub, copy=True)

    if len(rle_tokens) == 0:
        assert np.all(mask_gt == 0), "non-zero mask found for empty rle_tokens"
        return

    if is_vis:
        mask_vis_to_id(mask_gt, n_classes)
        mask_vis_to_id(mask_gt_sub, n_classes)

    n_rows, n_cols = mask_gt.shape

    if subsample > 1:
        max_length = int(max_length / subsample)
        n_rows, n_cols = int(n_rows / subsample), int(n_cols / subsample)

    mask_rec, instance_info, rle_rec_cmp = mask_from_instance_wise_tokens(
        rle_tokens, (n_rows, n_cols),
        n_classes,
        allow_extra=False,
        starts_offset=starts_offset,
        lengths_offset=lengths_offset,
        class_offset=class_offset,
        starts_2d=starts_2d,
        flat_order=flat_order,
        ignore_invalid=False,

    )
    starts, lengths, class_ids = rle_rec_cmp
    assert np.all(lengths <= max_length), f"run length cannot be > {max_length}"
    assert np.all(lengths > 0), "run length cannot be 0"

    n_rows, n_cols = mask_gt_sub.shape
    n_pix = n_rows * n_cols

    assert np.all(starts <= n_pix - 1), f"starts cannot be > {n_pix - 1}"

    if len(rle_rec_cmp) == 3:
        class_ids = rle_rec_cmp[2]
        assert 0 not in class_ids, "class_ids must be non-zero"
        assert np.all(np.asarray(class_ids) <= n_classes), "class_ids must be <= n_classes"

    if not np.array_equal(mask_gt_sub, mask_rec):
        print("mask_rec mismatch")

        # if subsample > 1:
        #     mask_rec = resize_mask(mask_rec, mask.shape, n_classes, is_vis=1)

        # import eval_utils
        mask_gt_vis = mask_id_to_vis_bgr(mask_gt_sub, class_to_col)
        mask_rec_vis = mask_id_to_vis_bgr(mask_rec, class_to_col)

        mask_gt_vis_res = resize_mask(mask_gt_vis, image.shape)
        mask_rec_vis_res = resize_mask(mask_rec_vis, image.shape)

        masks_all = np.concatenate([image, mask_gt_vis_res, mask_rec_vis_res], axis=1)
        # vis_txt = ' '.join(vis_txt)
        # masks_all = eval_utils.annotate(masks_all, vis_txt)
        # cv2.imshow('mask_gt_vis', mask_gt_vis)
        # cv2.imshow('mask_rec_vis', mask_rec_vis)

        masks_all = resize_ar(masks_all, 640, 480)

        cv2.imshow('mask_gt_vis', mask_gt_vis)
        cv2.imshow('mask_rec_vis', mask_rec_vis)
        cv2.imshow('masks_all', masks_all)
        k = cv2.waitKey(0)
        if k == 27:
            exit()

def check_class_wise_rle_tokens(
        image, mask, mask_sub, rle_tokens, n_classes,
        starts_2d,
        starts_offset, lengths_offset, class_offset,
        max_length,
        subsample,
        flat_order,
        class_to_col, is_vis):
    mask_gt = mask_to_gs(mask, copy=True)
    mask_gt_sub = mask_to_gs(mask_sub, copy=True)

    if len(rle_tokens) == 0:
        assert np.all(mask_gt == 0), "non-zero mask found for empty rle_tokens"
        return

    if is_vis:
        mask_vis_to_id(mask_gt, n_classes)
        mask_vis_to_id(mask_gt_sub, n_classes)

    n_rows, n_cols = mask_gt.shape

    if subsample > 1:
        max_length = int(max_length / subsample)
        n_rows, n_cols = int(n_rows / subsample), int(n_cols / subsample)

    mask_rec, rle_rec_cmp = mask_from_class_wise_tokens(
        rle_tokens, (n_rows, n_cols),
        n_classes,
        allow_extra=False,
        starts_offset=starts_offset,
        lengths_offset=lengths_offset,
        class_offset=class_offset,
        starts_2d=starts_2d,
        flat_order=flat_order,
        ignore_invalid=False,

    )
    starts, lengths, class_ids = rle_rec_cmp
    assert np.all(lengths <= max_length), f"run length cannot be > {max_length}"
    assert np.all(lengths > 0), "run length cannot be 0"

    n_rows, n_cols = mask_gt_sub.shape
    n_pix = n_rows * n_cols
    assert np.all(starts <= n_pix - 1), f"starts cannot be > {n_pix - 1}"

    if len(rle_rec_cmp) == 3:
        class_ids = rle_rec_cmp[2]
        assert 0 not in class_ids, "class_ids must be non-zero"
        assert np.all(np.asarray(class_ids) <= n_classes), "class_ids must be <= n_classes"

    if not np.array_equal(mask_gt_sub, mask_rec):
        print("mask_rec mismatch")

        # if subsample > 1:
        #     mask_rec = resize_mask(mask_rec, mask.shape, n_classes, is_vis=1)

        # import eval_utils
        mask_gt_vis = mask_id_to_vis_bgr(mask_gt_sub, class_to_col)
        mask_rec_vis = mask_id_to_vis_bgr(mask_rec, class_to_col)

        mask_gt_vis_res = resize_mask(mask_gt_vis, image.shape)
        mask_rec_vis_res = resize_mask(mask_rec_vis, image.shape)

        masks_all = np.concatenate([image, mask_gt_vis_res, mask_rec_vis_res], axis=1)
        # vis_txt = ' '.join(vis_txt)
        # masks_all = eval_utils.annotate(masks_all, vis_txt)
        # cv2.imshow('mask_gt_vis', mask_gt_vis)
        # cv2.imshow('mask_rec_vis', mask_rec_vis)

        masks_all = resize_ar(masks_all, 640, 480)

        cv2.imshow('mask_gt_vis', mask_gt_vis)
        cv2.imshow('mask_rec_vis', mask_rec_vis)
        cv2.imshow('masks_all', masks_all)
        k = cv2.waitKey(0)
        if k == 27:
            exit()


def check_rle(config, mask_vid_paths, images, img_ids, frame_ids, rles, training, class_id_to_col):
    if isinstance(mask_vid_paths, str):
        images = np.expand_dims(images, axis=0)
        img_ids = np.expand_dims(img_ids, axis=0)
        frame_ids = np.expand_dims(frame_ids, axis=0)
        rles = np.expand_dims(rles, axis=0)

        mask_vid_paths = [mask_vid_paths, ]

    batch_size = frame_ids.shape[0]
    if training:
        mode_cfg = config.dataset.train
    else:
        mode_cfg = config.dataset.eval

    max_length = mode_cfg.max_length
    subsample = mode_cfg.subsample

    assert max_length > 0, "max_length must be > 0"
    # assert subsample >= 1, "subsample must be >= 1"

    rle_from_mask = config.dataset.rle_from_mask
    instance_wise = config.dataset.instance_wise
    class_wise = config.dataset.class_wise
    multi_class = config.dataset.multi_class

    n_classes = len(class_id_to_col)

    if not multi_class:
        assert n_classes == 2, "n_classes must be 2 for no multi_class"
    else:
        assert n_classes > 2, "n_classes must be > 2 for multi_class"

    flat_order = config.dataset.flat_order
    length_as_class = config.dataset.length_as_class
    starts_2d = config.dataset.starts_2d

    starts_offset = config.model.coord_vocab_shift
    lengths_offset = config.model.len_vocab_shift
    class_offset = config.model.class_vocab_shift

    if length_as_class:
        n_lac_classes = int(max_length / subsample) * (n_classes - 1)
        if starts_offset < n_lac_classes:
            print(f'setting starts_offset to {n_lac_classes}')
            starts_offset = n_lac_classes
    masks = []
    masks_sub = []
    for batch_id in range(batch_size):
        mask_vid_path = mask_vid_paths[batch_id]
        if isinstance(mask_vid_path, bytes):
            mask_vid_path = mask_vid_path.decode('utf-8')

        rle = rles[batch_id]
        img_id = img_ids[batch_id]
        image = images[batch_id]
        frame_id = frame_ids[batch_id]
        vid_reader, vid_width, vid_height, num_frames = load_video(mask_vid_path)

        n_rows, n_cols = vid_height, vid_width
        n_rows_sub, n_cols_sub = int(n_rows / subsample), int(n_cols / subsample)

        mask = read_frame(vid_reader, frame_id - 1, mask_vid_path)

        if not multi_class:
            mask = mask_to_binary(mask)
        mask = mask_to_gs(mask)

        if subsample > 1:
            if rle_from_mask:
                """tf resizes images differently from opencv so 
                rle generated from tf-resized mask can only be checked against tf-resized mask"""
                mask_tf = tf.expand_dims(tf.convert_to_tensor(mask), axis=2)
                # scale = 1. / subsample
                # input_size = tf.cast(tf.shape(image)[1:3], tf.float32)
                # scaled_size = tf.cast(tf.multiply(input_size, scale), tf.int32)
                # mask_sub_tf = tf.image.resize(
                #     mask_tf, tf.cast(scaled_size, tf.int32), method="nearest", antialias=False)

                mask_sub_tf = tf.image.resize(
                    mask_tf, (n_rows_sub, n_cols_sub), method="nearest", antialias=False)
                mask_sub = mask_sub_tf.numpy().squeeze()
            else:
                mask_sub = resize_mask(mask, (n_rows_sub, n_cols_sub))
                # mask_sub = subsample_mask(mask, subsample, n_classes, is_vis=1)
        else:
            mask_sub = np.copy(mask)

        masks.append(mask)
        masks_sub.append(mask_sub)

        rle_stripped = rle[rle != vocab.PADDING_TOKEN]
        if rle_stripped.size == 0:
            print(f'\n{img_id}: mask is empty\n')

        if instance_wise:
            check_instance_wise_rle_tokens(
                image, mask, mask_sub,
                rle_stripped, n_classes,
                starts_2d=starts_2d,
                starts_offset=starts_offset,
                lengths_offset=lengths_offset,
                class_offset=class_offset,
                max_length=max_length,
                subsample=subsample,
                flat_order=flat_order,
                class_to_col=class_id_to_col,
                is_vis=True)
        else:
            check_rle_tokens(
                image, mask, mask_sub,
                rle_stripped,
                n_classes=n_classes,
                length_as_class=length_as_class,
                starts_2d=starts_2d,
                starts_offset=starts_offset,
                lengths_offset=lengths_offset,
                class_offset=class_offset,
                max_length=max_length,
                subsample=subsample,
                class_to_col=class_id_to_col,
                multi_class=multi_class,
                flat_order=flat_order,
                is_vis=True,
            )
    return masks, masks_sub


def check_rle_tokens(
        image, mask, mask_sub, rle_tokens, n_classes,
        length_as_class,
        starts_2d,
        starts_offset, lengths_offset, class_offset,
        max_length, subsample, multi_class,
        flat_order,
        class_to_col, is_vis):
    mask_gt = mask_to_gs(mask, copy=True)
    mask_gt_sub = mask_to_gs(mask_sub, copy=True)

    if len(rle_tokens) == 0:
        assert np.all(mask_gt_sub == 0), "non-zero mask found for empty rle_tokens"
        return

    if is_vis:
        mask_vis_to_id(mask_gt, n_classes)
        mask_vis_to_id(mask_gt_sub, n_classes)

    n_rows, n_cols = mask_gt.shape

    if subsample > 1:
        max_length = int(max_length / subsample)
        n_rows, n_cols = int(n_rows / subsample), int(n_cols / subsample)

    mask_rec, rle_rec_cmp = mask_from_tokens(
        rle_tokens,
        (n_rows, n_cols),
        allow_extra=False,
        length_as_class=length_as_class,
        max_length=max_length,
        starts_offset=starts_offset,
        lengths_offset=lengths_offset,
        class_offset=class_offset,
        starts_2d=starts_2d,
        multi_class=multi_class,
        flat_order=flat_order,
    )

    starts, lengths = rle_rec_cmp[:2]
    assert np.all(lengths <= max_length), f"run length cannot be > {max_length}"
    assert np.all(lengths > 0), "run length cannot be 0"

    n_rows, n_cols = mask_gt_sub.shape
    n_pix = n_rows * n_cols
    assert np.all(starts <= n_pix - 1), f"starts cannot be > {n_pix - 1}"

    if len(rle_rec_cmp) == 3:
        class_ids = rle_rec_cmp[2]
        assert 0 not in class_ids, "class_ids must be non-zero"
        assert np.all(np.asarray(class_ids) <= n_classes), "class_ids must be <= n_classes"

    if not np.array_equal(mask_gt_sub, mask_rec):
        print("mask_rec mismatch")

        # if subsample > 1:
        #     mask_rec = resize_mask(mask_rec, mask.shape, n_classes, is_vis=1)

        # import eval_utils
        mask_gt_vis = mask_id_to_vis_bgr(mask_gt_sub, class_to_col)
        mask_rec_vis = mask_id_to_vis_bgr(mask_rec, class_to_col)

        mask_gt_vis = resize_mask(mask_gt_vis, image.shape)
        mask_rec_vis = resize_mask(mask_rec_vis, image.shape)

        masks_all = np.concatenate([image, mask_gt_vis, mask_rec_vis], axis=1)
        # vis_txt = ' '.join(vis_txt)
        # masks_all = eval_utils.annotate(masks_all, vis_txt)
        # cv2.imshow('mask_gt_vis', mask_gt_vis)
        # cv2.imshow('mask_rec_vis', mask_rec_vis)

        masks_all = resize_ar(masks_all, 640, 480)

        cv2.imshow('masks_all', masks_all)
        cv2.imshow('mask_gt_vis', mask_gt_vis)
        cv2.imshow('mask_rec_vis', mask_rec_vis)
        k = cv2.waitKey(0)
        if k == 27:
            exit()

        # print('masks match !')


def check_individual_vid_masks(video, vid_mask_gt_sub, vid_mask_rec, class_id_to_col, n_classes):
    for vid_id, (image, mask_gt_sub, mask_rec) in enumerate(zip(video, vid_mask_gt_sub, vid_mask_rec)):
        if not np.array_equal(mask_gt_sub, mask_rec):
            print(f"mask {vid_id} mismatch")

            mask_gt_sub_vis = mask_id_to_vis_bgr(mask_gt_sub, class_id_to_col)
            mask_rec_vis = mask_id_to_vis_bgr(mask_rec, class_id_to_col)

            mask_gt_sub_vis = resize_mask(mask_gt_sub_vis, image.shape)
            mask_rec_vis = resize_mask(mask_rec_vis, image.shape)

            masks_all = np.concatenate([image, mask_gt_sub_vis, mask_rec_vis], axis=1)
            masks_all = cv2.resize(masks_all, (960, 540))

            cv2.imshow(f'masks {vid_id}', masks_all)
            k = cv2.waitKey(0)
            if k == 27:
                exit()
        else:
            print(f"mask {vid_id} matches")


def check_video_rle_ranges(rle_tokens, n_runs, multi_class, time_as_class, length_as_class,
                           vocab_size, starts_offset, lengths_offset, class_offset):
    has_class_tokens = (multi_class or time_as_class) and not length_as_class

    n_rle_tokens = len(rle_tokens)
    n_tokens_per_run = 3 if has_class_tokens else 2
    assert n_rle_tokens % n_tokens_per_run == 0, f"rle_tokens length must be divisible by {n_tokens_per_run}"

    assert (n_runs == 0 and n_rle_tokens == 0) or (n_runs > 0 and n_rle_tokens > 0), \
        "n_runs and n_rle_tokens must both either be zero or non-zero"

    starts_tokens = np.array(rle_tokens[0:][::n_tokens_per_run], dtype=np.int64)
    max_starts = np.amax(starts_tokens)
    min_starts = np.amin(starts_tokens)
    assert max_starts < vocab_size, "max_starts exceeds vocab_size"
    assert min_starts > starts_offset, "starts_offset exceeds min_starts"
    # assert max_starts < starts_bins + starts_offset, "max_starts exceeds starts_bins + starts_offset"

    lengths_tokens = np.array(rle_tokens[1:][::n_tokens_per_run], dtype=np.int64)
    max_lengths_tokens = np.amax(lengths_tokens)
    min_lengths_tokens = np.amin(lengths_tokens)
    assert max_lengths_tokens < starts_offset, "max_lengths_tokens exceeds starts_offset"
    assert min_lengths_tokens > lengths_offset, "lengths_offset exceeds min_lengths_tokens"
    # assert max_lengths_tokens < lengths_bins + lengths_offset, "max_lengths_tokens exceeds lengths_bins
    # + lengths_offset"

    if has_class_tokens:
        class_tokens = np.array(rle_tokens[2:][::n_tokens_per_run], dtype=np.int64)
        max_class_tokens = np.amax(class_tokens)
        min_class_tokens = np.amin(class_tokens)
        assert max_class_tokens < starts_offset, "max_class_tokens exceeds starts_offset"
        assert min_class_tokens > class_offset, "class_offset exceeds min_class_tokens"

        if not time_as_class:
            assert max_class_tokens < lengths_offset, "max_class_tokens exceeds lengths_offset"


def check_video_rle_tokens(
        video, vid_mask, vid_mask_sub,
        rle_tokens, n_classes,
        time_as_class,
        length_as_class,
        starts_offset, lengths_offset, class_offset,
        max_length, subsample, multi_class,
        flat_order,
        class_id_to_name,
        class_id_to_col,
        is_vis,
        tac_mask_sub,
        tac_id_to_col,
        allow_overlap,
        allow_extra,
):
    vid_mask_gt = vid_mask_to_gs(vid_mask, copy=True)
    vid_len, n_rows, n_cols = vid_mask_gt.shape

    if subsample > 1:
        max_length = int(max_length / subsample)
        n_rows, n_cols = int(n_rows / subsample), int(n_cols / subsample)

    if not length_as_class:
        assert allow_overlap or starts_offset > lengths_offset + max_length, ("len_token_range overlaps "
                                                                              "starts_token_range")

    if multi_class or time_as_class or length_as_class:
        n_classes_ = n_classes
        if time_as_class:
            n_classes_ = n_classes_ ** vid_len
        if length_as_class:
            n_total_classes = max_length * (n_classes_ - 1)
        else:
            n_total_classes = n_classes_

        assert allow_overlap or starts_offset > class_offset + n_total_classes, ("class_token_range overlaps "
                                                                                 "starts_token_range")

        if not length_as_class:
            assert allow_overlap or lengths_offset > class_offset + n_total_classes, ("class_token_range overlaps "
                                                                                      "len_token_range")

    if len(rle_tokens) == 0:
        assert np.all(vid_mask_gt == 0), "non-zero mask found for empty rle_tokens"

        # print('empty masks match !')
        return

    if is_vis:
        vid_mask_gt = mask_vis_to_id(vid_mask_gt, n_classes, copy=True)
        vid_mask_sub = mask_vis_to_id(vid_mask_sub, n_classes, copy=True)

    if subsample > 1:
        vid_mask_gt_sub = subsample_vid_mask(vid_mask_gt, subsample, n_classes, is_vis=0)
    else:
        vid_mask_gt_sub = vid_mask_gt

    if time_as_class and tac_mask_sub is None:
        tac_mask_sub = vid_mask_to_tac(video, vid_mask_gt_sub, n_classes, class_id_to_col, check=True)
        tac_id_to_col, tac_id_to_name = get_tac_info(vid_len, class_id_to_name)

    # rle_len = len(rle_tokens)
    vid_mask_rec, tac_mask_rec, rle_rec_cmp = vid_mask_from_tokens(
        rle_tokens,
        allow_extra=allow_extra,
        vid_len=vid_len,
        shape=(n_rows, n_cols),
        length_as_class=length_as_class,
        max_length=max_length,
        starts_offset=starts_offset,
        lengths_offset=lengths_offset,
        class_offset=class_offset,
        starts_2d=False,
        multi_class=multi_class,
        flat_order=flat_order,
        time_as_class=time_as_class,
        n_classes=n_classes,
        ignore_invalid=False,
    )

    starts, lengths = rle_rec_cmp[:2]
    assert np.all(lengths <= max_length), f"run length cannot be > {max_length}"
    assert np.all(lengths > 0), "run length cannot be 0"

    n_rows, n_cols = vid_mask_gt_sub.shape[1:]
    if time_as_class:
        n_pix = n_rows * n_cols
    else:
        n_pix = vid_len * n_rows * n_cols

    assert np.all(starts <= n_pix - 1), f"starts cannot be > {n_pix - 1}"

    if len(rle_rec_cmp) == 3:
        class_ids = rle_rec_cmp[2]
        assert 0 not in class_ids, "class_ids must be non-zero"

        if time_as_class:
            n_classes_ = int(n_classes ** vid_len)
        else:
            n_classes_ = n_classes

        assert np.all(np.asarray(class_ids) <= n_classes_), f"class_ids must be <= {n_classes_}"

    if not np.array_equal(vid_mask_gt_sub, vid_mask_sub):
        print('vid_mask_gt_sub mismatch')
        check_individual_vid_masks(video, vid_mask_gt_sub, vid_mask_sub,
                                   class_id_to_col, n_classes)
        raise AssertionError()
    # else:
    #     print('vid_mask_gt_sub matches')

    if not np.array_equal(vid_mask_gt_sub, vid_mask_rec):
        print("vid_mask_rec mismatch")
        if time_as_class and tac_mask_sub is not None:
            if not np.array_equal(tac_mask_sub, tac_mask_rec):
                print("tac_masks mismatch")
                tac_mask_sub_vis = mask_id_to_vis_bgr(tac_mask_sub, tac_id_to_col)
                tac_mask_rec_vis = mask_id_to_vis_bgr(tac_mask_rec, tac_id_to_col)
                tac_mask_sub_vis = resize_mask(tac_mask_sub_vis, (640, 640))
                tac_mask_rec_vis = resize_mask(tac_mask_rec_vis, (640, 640))

                masks_all = np.concatenate([tac_mask_sub_vis, tac_mask_rec_vis], axis=1)
                masks_all = cv2.resize(masks_all, (960, 540))
                cv2.imshow(f'tac_mask', masks_all)
                k = cv2.waitKey(0)
                if k == 27:
                    exit()
            else:
                vid_mask_gt_rec = vid_mask_from_tac(tac_mask_sub, vid_len, n_classes)
                if not np.array_equal(vid_mask_gt_rec, vid_mask_gt_sub):
                    print("vid_mask_gt_rec mismatch")

                print("\ntac_masks match !\n")

        check_individual_vid_masks(video, vid_mask_gt_sub, vid_mask_rec, class_id_to_col, n_classes)

        # if subsample > 1:
        #     mask_rec = resize_mask(mask_rec, mask.shape, n_classes, is_vis=1)

        # import eval_utils
        vid_mask_gt_vis = vid_mask_id_to_vis_rgb(vid_mask_gt, class_id_to_col)
        vid_mask_rec_vis = vid_mask_id_to_vis_rgb(vid_mask_rec, class_id_to_col)

        vid_mask_rec_vis_ = concat_vid(vid_mask_rec_vis, axis=0, border=1)
        cv2.imshow('vid_mask_rec_vis', vid_mask_rec_vis_)

        vid_mask_gt_vis = resize_vid(vid_mask_gt_vis, video.shape[1:3])
        vid_mask_rec_vis = resize_vid(vid_mask_rec_vis, video.shape[1:3])

        vid_mask_gt_vis = concat_vid(vid_mask_gt_vis, axis=0, border=1)
        vid_mask_rec_vis = concat_vid(vid_mask_rec_vis, axis=0, border=1)
        vid_vis = concat_vid(video, axis=0, border=1)

        vid_masks_all = np.concatenate([vid_vis, vid_mask_gt_vis, vid_mask_rec_vis], axis=1)

        vid_masks_all = cv2.resize(vid_masks_all, (960, 540))
        cv2.imshow('vid_masks_all', vid_masks_all)
        k = cv2.waitKey(1)
        if k == 27:
            exit()
    # else:
    #     print('\nvid masks match !\n')


def interleave_rle(rle_cmps):
    rle = [int(item) for sublist in zip(*rle_cmps) for item in sublist]
    return rle


def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def subsample_vid_mask(vid_mask, factor: int, n_classes, is_vis):
    masks = [subsample_mask(mask, factor, n_classes, is_vis) for mask in vid_mask]
    masks = np.stack(masks, axis=0)
    return masks


def sub_and_super_sample_mask(mask, factor, n_classes, is_vis=1, shape=None,
                              check=False, max_diff_rate=5):
    assert len(mask.shape) == 2, "only grayscale masks are supported"
    if is_vis:
        mask = np.copy(mask)
        mask_vis_to_id(mask, n_classes)

    n_rows, n_cols = mask.shape
    if shape is not None:
        assert factor is None, "factor must be None for custom shape"
        n_rows_out, n_cols_out = shape
        factor_x, factor_y = n_cols_out / n_cols, n_rows_out / n_rows
    else:
        if isinstance(factor, (tuple, list)):
            factor_x, factor_y = factor
        else:
            factor_x = factor_y = factor
        n_rows_out, n_cols_out = int(n_rows * factor_y), int(n_cols * factor_x)

    mask_out = np.zeros_like(mask, shape=(n_rows_out, n_cols_out))

    for row_id in range(n_rows_out):
        for col_id in range(n_cols_out):
            mask_out[row_id, col_id] = mask[int(row_id / factor_y), int(col_id / factor_x)]

    if is_vis:
        mask_id_to_vis(mask_out, n_classes)

    if check:
        mask_rec = sub_and_super_sample_mask(
            mask_out, n_classes=n_classes, factor=None, shape=mask.shape, is_vis=is_vis)
        labels_diff_rate = check_mask_diff(mask_rec, mask, max_diff_rate)
        return mask_out, labels_diff_rate
    return mask_out


def supersample_mask(mask, factor, n_classes, is_vis=1, shape=None,
                     check=False, max_diff_rate=5):
    assert len(mask.shape) == 2, "only grayscale masks are supported"
    if is_vis:
        mask = np.copy(mask)
        mask_vis_to_id(mask, n_classes)

    n_rows, n_cols = mask.shape
    if shape is not None:
        assert factor is None, "factor must be None for custom shape"
        n_rows_out, n_cols_out = shape
        assert n_rows_out >= n_rows and n_cols_out >= n_cols, \
            "target size must be larger than source size for super sampling"
        factor_x, factor_y = n_cols_out / n_cols, n_rows_out / n_rows
    else:
        if isinstance(factor, (tuple, list)):
            factor_x, factor_y = factor
        else:
            factor_x = factor_y = factor
        n_rows_out, n_cols_out = int(n_rows * factor_y), int(n_cols * factor_x)

    mask_out = np.zeros_like(mask, shape=(n_rows_out, n_cols_out))

    for row_id in range(n_rows_out):
        for col_id in range(n_cols_out):
            mask_out[row_id, col_id] = mask[int(row_id / factor_y), int(col_id / factor_x)]

    if is_vis:
        mask_id_to_vis(mask_out, n_classes)

    if check:
        mask_rec = supersample_mask(
            mask_out, n_classes=n_classes, factor=None, shape=mask.shape, is_vis=is_vis)
        labels_diff_rate = check_mask_diff(mask_rec, mask, max_diff_rate)
        return mask_out, labels_diff_rate
    return mask_out


def subsample_mask(mask, factor, n_classes, is_vis=1, shape=None,
                   check=False, max_diff_rate=5):
    assert len(mask.shape) == 2, "only grayscale masks are supported"
    if is_vis:
        mask = mask_vis_to_id(mask, n_classes, copy=True)

    n_rows, n_cols = mask.shape

    if shape is not None:
        assert factor is None, "factor must be None for custom shape"
        n_rows_out, n_cols_out = shape
        assert n_rows_out <= n_rows and n_cols_out <= n_cols, \
            "target size must be smaller than source size for subsampling"

        factor_x, factor_y = n_cols / n_cols_out, n_rows / n_rows_out
    else:
        if isinstance(factor, (tuple, list)):
            factor_x, factor_y = factor
        else:
            factor_x = factor_y = factor
        n_rows_out, n_cols_out = int(n_rows / factor_y), int(n_cols / factor_x)

    mask_out = np.zeros_like(mask, shape=(n_rows_out, n_cols_out))

    for class_id in range(1, n_classes):
        y, x = np.nonzero(mask == class_id)
        out_x, out_y = (x // factor_x).astype(np.int64), (y // factor_y).astype(np.int64)
        mask_out[out_y, out_x] = class_id
    if is_vis:
        mask_id_to_vis(mask_out, n_classes)
    if check:
        mask_rec = supersample_mask(
            mask_out, n_classes=n_classes, factor=None, shape=mask.shape, is_vis=is_vis)
        labels_diff_rate = check_mask_diff(mask_rec, mask, max_diff_rate)
        return mask_out, labels_diff_rate

    return mask_out


def resize_video_mask(vid_mask, shape):
    masks = [resize_mask(mask, shape) for mask in vid_mask]
    masks = np.stack(masks, axis=0)
    return masks


def resize_mask(mask, shape, check=False, max_diff_rate=5):
    n_rows, n_cols = shape[:2]
    mask_out = mask

    # if not is_vis:
    #     assert n_classes is not None, "n_classes must be provided if mask is not vis"
    #     mask_out = mask_id_to_vis(mask, n_classes, copy=True)

    mask_out = cv2.resize(mask_out, (n_cols, n_rows), interpolation=cv2.INTER_NEAREST)

    # if not is_vis:
    #     mask_vis_to_id(mask_out, n_classes)

    if check:
        mask_rec = resize_mask(mask_out, shape=mask.shape)
        labels_diff_rate = check_mask_diff(mask_rec, mask, max_diff_rate)
        return mask_out, labels_diff_rate

    return mask_out


def mask_id_to_vis(mask_id, n_classes, to_rgb=0, copy=False,
                   return_class_id_to_col=False, class_id_to_col=None):
    if to_rgb or copy:
        mask_vis = np.zeros_like(mask_id)
    else:
        mask_vis = mask_id

    if class_id_to_col is None:
        if n_classes is None:
            unique_ids = np.unique(mask_id)
            n_classes = len(unique_ids)
            assert np.amax(unique_ids) == n_classes - 1, \
                "n_classes can be inferred only if all classes occur in the image"

        if n_classes == 3:
            class_id_to_col = {
                0: 0,
                1: 128,
                2: 255,
            }
        elif n_classes == 2:
            class_id_to_col = {
                0: 0,
                1: 255,
                2: 255,
            }
        else:
            cols = get_class_cols_gs(n_classes)
            class_id_to_col = {
                class_id: cols[class_id] for class_id in range(n_classes)
            }

    for class_id, class_col in class_id_to_col.items():
        mask_vis[mask_id == class_id] = class_col

    if to_rgb and len(mask_vis.shape) == 2:
        mask_vis = np.stack((mask_vis,) * 3, axis=2)

    if return_class_id_to_col:
        return mask_vis, class_id_to_col

    if to_rgb or copy:
        return mask_vis


def mask_vis_to_id(mask_vis, n_classes=None, copy=False, check=False,
                   max_diff_rate=0.1, precise=False, spurious_mids=False,
                   col_to_id=None):
    if col_to_id is not None:
        mask_id = np.zeros_like(mask_vis)
        for col, _id in col_to_id.items():
            mask_id[mask_vis == col] = _id
        return mask_id
    else:

        assert n_classes is not None, "n_classes and col_to_id cannot both be None"

        if spurious_mids or copy or check:
            mask_id = np.zeros_like(mask_vis)
        else:
            mask_id = mask_vis

        if n_classes == 3:
            if precise:
                mask_id[np.logical_and(mask_vis != 128, mask_vis != 255)] = 0
                mask_id[mask_vis == 128] = 1
                mask_id[mask_vis == 255] = 2
            else:
                mask_id[mask_vis < 64] = 0
                mask_id[np.logical_and(mask_vis >= 64, mask_vis < 192)] = 1
                mask_id[mask_vis >= 192] = 2

            if spurious_mids:
                remove_spurious_mids_with_edges(mask_id, max_iters=1)
                remove_spurious_mids_with_cc(mask_id, min_area=5)

        elif n_classes == 2:
            if precise:
                mask_id[mask_vis != 0] = 1
            else:
                mask_id[mask_vis < 128] = 0
                mask_id[mask_vis >= 128] = 1
        else:
            cols = get_class_cols_gs(n_classes)
            for class_id in range(1, n_classes):
                mask_vis[mask_id == cols[class_id]] = class_id

    if check:
        mask_vis_rec = mask_id_to_vis(mask_id, n_classes, copy=True)
        labels_diff_rate = check_mask_diff(mask_vis_rec, mask_vis, max_diff_rate)
        return mask_id, labels_diff_rate

    return mask_id


def check_mask_diff(mask_vis_rec, mask_vis, max_diff_rate):
    diff_bool = mask_vis_rec != mask_vis
    n_diff = np.count_nonzero(diff_bool)
    labels_diff_rate = float(n_diff) / mask_vis.size * 100.
    if labels_diff_rate > max_diff_rate:
        print(f"\nlabels_diff_rate is too high: {labels_diff_rate:.3f}\n")
        diff_img = diff_bool.astype(np.uint8) * 255
        cv2.imshow('mask_vis', mask_vis)
        cv2.imshow('mas6k_vis_rec', mask_vis_rec)
        cv2.imshow('diff_img', diff_img)
        cv2.waitKey(0)
        raise AssertionError()
    return labels_diff_rate


def remove_spurious_mids_with_cc(mask_id, min_area=5):
    mids_bool = (mask_id == 1).astype(np.uint8) * 255
    connectivity = 4
    output = cv2.connectedComponentsWithStats(mids_bool, connectivity, cv2.CV_32S)
    labels = output[1]
    stats = output[2]

    areas = stats[:, cv2.CC_STAT_AREA]
    invalid_labels = np.where(areas < min_area)[0]
    # labels_filtered = np.copy(labels)
    for invalid_label in invalid_labels:
        spurious_bool = labels == invalid_label
        mask_id[spurious_bool] = 0
        # labels_filtered[spurious_bool] = 0

    # num_labels = output[0]
    # num_labels_f = num_labels - len(invalid_labels)
    # labels_filtered_img = labels_filtered.astype(np.float32) / (num_labels_f - 1)
    # labels_img = labels.astype(np.float32) / (num_labels - 1)
    # cv2.imshow('mids_bool', mids_bool)
    # cv2.imshow('labels_img', labels_img)
    # cv2.imshow('labels_filtered_img', labels_filtered_img)
    # cv2.waitKey(1)
    # centroids = output[3]


def remove_spurious_mids_with_edges(mask_id, max_iters=0):
    from scipy.signal import convolve2d as conv2

    iter_id = 0

    highs_bool = mask_id == 2
    mids_bool = mask_id == 1

    while True:
        iter_id += 1

        highs_img = highs_bool.astype(np.uint8) * 255

        sobel_y = -np.array(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        )
        sobel_y_img = conv2(highs_img, sobel_y, 'same')
        sobel_x = -np.array(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        )
        sobel_x_img = conv2(highs_img, sobel_x, 'same')

        sobel_img = sobel_y_img ** 2 + sobel_x_img ** 2

        edge_bool = sobel_img > 0

        spurious_bool = np.logical_and(mids_bool, edge_bool)

        n_spurious = np.count_nonzero(spurious_bool)
        if n_spurious == 0:
            break

        mask_id[spurious_bool] = 2

        highs_bool[spurious_bool] = True
        mids_bool[spurious_bool] = False

        if iter_id >= max_iters > 0:
            break

        # mids_img = mids_bool.astype(np.uint8) * 255
        # edge_img = edge_bool.astype(np.uint8) * 255
        # spurious_img = spurious_bool.astype(np.uint8) * 255
        # edge_img = resize_ar(edge_img, 640)
        # mids_img = resize_ar(mids_img, 640)
        # highs_img = resize_ar(highs_img, 640)
        # spurious_img = resize_ar(spurious_img, 640)
        # spurious_img = vis_utils.annotate(spurious_img, f"iter {iter_id}: {n_spurious}")
        # cv2.imshow('edge_img', edge_img)
        # cv2.imshow('mids_img', mids_img)
        # cv2.imshow('highs_img', highs_img)
        # cv2.imshow('spurious_img', spurious_img)
        # cv2.waitKey(0)


def blend_mask(mask, image, class_id_to_col, alpha=0.5):
    n_classes = len(class_id_to_col)
    """ignore class id 0 for background"""
    vis_image = np.copy(image)

    for class_id in range(1, n_classes):
        class_col = class_id_to_col[class_id]
        if isinstance(class_col, str):
            class_col = col_bgr[class_col]
        class_mask_binary = (mask == class_id)
        vis_image[class_mask_binary] = vis_image[class_mask_binary] * (1 - alpha) + np.asarray(class_col) * alpha
    return vis_image


def vid_mask_id_to_vis_rgb(vid_mask, class_id_to_col):
    masks = [mask_id_to_vis_bgr(mask, class_id_to_col) for mask in vid_mask]
    vid_mask_rgb = np.stack(masks, axis=0)
    return vid_mask_rgb


def mask_id_to_vis_bgr(mask, class_id_to_col):
    mask_bgr = np.stack((mask,) * 3, axis=2).astype(np.uint8)

    n_classes = len(class_id_to_col)
    for class_id in range(n_classes):
        class_col = class_id_to_col[class_id]
        if isinstance(class_col, str):
            class_col = col_bgr[class_col]
        mask_bgr[mask == class_id] = class_col
    return mask_bgr


def mask_vis_bgr_to_id(mask, class_id_to_col, check=0):
    mask_id = np.zeros_like(mask)
    mask_id = mask_id[..., 0].squeeze()

    n_classes = len(class_id_to_col)
    if n_classes == 2:
        """deal with annoying BGR masks when MC is to be disabled"""
        pix_mask = np.any(mask > 0, axis=2)
        mask_id[pix_mask] = 1
    else:
        mask_b = mask[..., 0].squeeze()
        mask_g = mask[..., 1].squeeze()
        mask_r = mask[..., 2].squeeze()

        for class_id, class_col in class_id_to_col.items():
            if isinstance(class_col, str):
                class_col = col_bgr[class_col]
            b, g, r = class_col
            pix_mask = np.logical_and.reduce([
                mask_b == b,
                mask_g == g,
                mask_r == r,
            ])
            mask_id[pix_mask] = class_id
    if check:
        mask_rec = mask_id_to_vis_bgr(mask_id, class_id_to_col)
        if not np.array_equal(mask, mask_rec):
            print("mask_vis_bgr_to_id mask_rec mismatch")
            mask_ = cv2.resize(mask, (640, 640))
            mask_rec_ = cv2.resize(mask_rec, (640, 640))
            cv2.imshow('mask', mask_)
            cv2.imshow('mask_rec', mask_rec_)
            cv2.waitKey(0)

    # mask_gs = mask_to_gs(mask_id)
    return mask_id


def get_class_cols_gs(n_classes):
    min_col, max_col = 0, 255
    n_col_levels = int(n_classes + 1)
    col_range = max_col - min_col
    assert n_col_levels <= col_range, "n_col_levels exceeds col_range"
    cols = [int(x) for x in np.linspace(
        min_col, max_col,
        n_col_levels, dtype=int)]
    return cols


def get_cols_rgb(n_cols, min_col=100, max_col=200):
    n_col_levels = int(n_cols ** (1. / 3) + 1)
    col_range = max_col - min_col
    assert n_col_levels <= col_range, "n_col_levels exceeds col_range"
    col_levels = [int(x) for x in np.linspace(
        min_col, max_col,
        n_col_levels, dtype=int)]
    import itertools
    cols = list(itertools.product(col_levels, repeat=3))

    return cols


def resize_vid(vid, shape):
    n_rows, n_cols = shape
    vid_out = []
    for img in vid:
        img = cv2.resize(img, (n_rows, n_cols))
        vid_out.append(img)
    vid_out = np.stack(vid_out, axis=0)
    return vid_out


def concat_with_boder(img1, img2, axis, border):
    # if axis == 0:
    #     border_img = np.full((border_size, img1.shape[1], 3), 255, dtype=np.uint8)
    # elif axis == 1:
    #     border_img = np.full((img1.shape[0], border_size, 3), 255, dtype=np.uint8)
    # else:
    #     raise AssertionError('invalid axis')
    # img = np.concatenate((img1, border_img, img2), axis=axis)

    img1 = cv2.copyMakeBorder(img1, border, border, border, border, cv2.BORDER_CONSTANT, None, (255, 255, 255))
    img2 = cv2.copyMakeBorder(img2, border, border, border, border, cv2.BORDER_CONSTANT, None, (255, 255, 255))
    img = np.concatenate((img1, img2), axis=axis)

    return img


def concat_vid(vid, axis, border):
    vid_out = []
    for img_id, img in enumerate(vid):
        if border:
            img = cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT, None, (255, 255, 255))

        vid_out.append(img)
        # if border and img_id < vid.shape[0] - 1:
        #     if axis == 0:
        #         border_img = np.full((border, img.shape[1], 3), 255, dtype=np.uint8)
        #     elif axis == 1:
        #         border_img = np.full((img.shape[0], border, 3), 255, dtype=np.uint8)
        #     else:
        #         raise AssertionError('invalid axis')
        #     vid_out.append(border_img)

    vid_out = np.concatenate(vid_out, axis=axis)
    return vid_out


def get_tac_info(vid_len, class_id_to_name):
    n_classes = len(class_id_to_name)

    cols = (
        'black',
        'deep_sky_blue', 'yellow', 'forest_green', 'cyan', 'magenta',
        'orange', 'maroon', 'peach_puff', 'dark_orange', 'slate_gray',
        'pale_turquoise', 'green_yellow', 'indian_red', 'sienna', 'dark_orange',
        'hot_pink', 'deep_pink', 'orchid', 'blue_violet', 'tomato',
        'orange_red', 'brown', 'peru', 'gold', 'chartreuse',
        'steel_blue', 'gray',
    )
    if n_classes == 2:
        if vid_len == 2:
            cols = (
                'black',  # 0, 0, 0
                'red',  # 0, 1, 0
                'green',  # 0, 0, 1
                # 'royal_blue',  # 0, 0, 1
                # 'sky_blue',  # 0, 0, 1
                # 'deep_sky_blue',  # 0, 0, 1
                'cyan',  # 0, 1, 1
            )
        elif vid_len == 3:
            cols = (
                'black',  # 0, 0, 0
                'deep_sky_blue',  # 0, 0, 1
                'forest_green',  # 0, 1, 0
                'cyan',  # 0, 1, 1
                'maroon',  # 1, 0, 0
                'magenta',  # 1, 0, 1
                'yellow',  # 1, 1, 1
                'slate_gray',  # 1, 1, 1
            )
    elif n_classes == 3:
        if vid_len == 2:
            cols = (
                'black',  # 00
                'deep_sky_blue',  # 10
                'magenta',  # 20
                'orange',  # 01
                'forest_green',  # 11
                'maroon',  # 21
                'cyan',  # 02
                'slate_gray',  # 12
                'yellow',  # 22
            )
    cols = [col_bgr[col] for col in cols]

    n_cols = int(n_classes ** vid_len)

    if n_cols > len(cols):

        invalid_cols = [
            'white',
            'black',
            'red',
            'green',
            'purple',
        ]

        import itertools

        n_col_levels = int(n_cols ** (1. / 3) + 1)
        col_levels = [int(x) for x in np.linspace(100, 200, n_col_levels + 1, dtype=int)]
        # col_levels = list(range(100, 201))
        all_cols = list(itertools.product(col_levels, repeat=3))

        # all_cols = list(col_bgr.keys())
        # all_cols = [col_bgr[col] for col in all_cols]

        for col in invalid_cols:
            try:
                all_cols.remove(col_bgr[col])
            except ValueError:
                pass

        assert len(all_cols) > n_cols - 1, "too few all_cols"

        # cols = [(0, 0, 0), (255, 255, 255)]
        # for col_id in range(n_cols - 1):
        #     sorted(all_cols, key=lambda x: get_min_col_diff(cols, x))
        #     new_col = all_cols.pop(-1)
        #     cols.append(new_col)
        # cols.remove((255, 255, 255))

        import random
        random.shuffle(cols)
        cols = all_cols
        sample = len(cols) // (n_cols - 1)
        cols = cols[::sample]
        cols.insert(0, (0, 0, 0))

    tac_id_to_col = {
        col_id: cols[col_id] for col_id in range(n_cols)
    }

    import itertools
    # class_names_str = [str(class_id) for class_id in range(n_classes)]
    class_names_str = [class_id_to_name[class_id] if class_id > 0 else 'bkg'
                       for class_id in range(n_classes)]
    tac_names = list(itertools.product(class_names_str, repeat=vid_len))
    tac_id_to_name = {
        tac_id: '-'.join(col_name[::-1]) for tac_id, col_name in enumerate(tac_names)
        # tac_id: ''.join(col_name) for tac_id, col_name in enumerate(tac_names)
    }

    return tac_id_to_col, tac_id_to_name


def vis_tac_run_pix(start, length, tac_mask_flat, vid_mask, col, tac_id_to_col, class_id_to_col,
                    flat_order):
    n_classes = len(class_id_to_col)
    vid_len, n_rows, n_cols = vid_mask.shape[:3]
    flat_ids = range(start, start + length)
    row_ids, col_ids = np.unravel_index(flat_ids, (n_rows, n_cols), order=flat_order)
    tac_class_ids = tac_mask_flat[start:start + length]

    vid_class_ids = [tac_to_vid_class_ids(tac_id, vid_len, n_classes)
                     for tac_id in tac_class_ids]
    vid_mask = np.copy(vid_mask)

    from collections import defaultdict
    img_to_run_pixs = defaultdict(list)

    for row_id, col_id, vid_class_id, tac_id in zip(row_ids, col_ids, vid_class_ids, tac_class_ids):
        for _id, class_id in enumerate(vid_class_id):
            if class_id == 0:
                continue
            col_ = col if col is not None else tac_id_to_col[tac_id]
            # col_ = col if col is not None else class_id_to_col[class_id]
            if isinstance(col_, str):
                col_ = col_bgr[col_]
            vid_mask[_id, row_id, col_id] = col_
            img_to_run_pixs[_id].append((row_id, col_id))

    return vid_mask, img_to_run_pixs


def vis_video_run_pix(start, length, vid_mask_sub_flat, vid_mask_sub_binary_vis,
                      temp_col, flat_order):
    vid_len, n_rows, n_cols = vid_mask_sub_binary_vis.shape[:3]
    vid_mask_bool_flat = np.zeros_like(vid_mask_sub_flat, dtype=bool)
    vid_mask_bool_flat[start:start + length] = True
    vid_mask_bool = np.reshape(vid_mask_bool_flat, vid_mask_sub_binary_vis.shape[:3],
                               order=flat_order)

    from collections import defaultdict
    img_to_run_pixs = defaultdict(list)
    for run_pix in range(start, start + length):
        _id, run_y, run_x = np.unravel_index([run_pix, ], (vid_len, n_rows, n_cols), order=flat_order)
        _id, run_y, run_x = int(_id[0]), int(run_y[0]), int(run_x[0])
        img_to_run_pixs[_id].append((run_y, run_x))
    vid_mask_sub_binary_vis = np.copy(vid_mask_sub_binary_vis)
    if isinstance(temp_col, str):
        temp_col = col_bgr[temp_col]

    vid_mask_sub_binary_vis[vid_mask_bool] = temp_col

    return vid_mask_sub_binary_vis, img_to_run_pixs


def get_min_col_diff(cols, _col2):
    col_diffs = [np.sum(np.fabs(np.asarray(_col1, dtype=np.float32) - np.asarray(_col2, dtype=np.float32))) / 3.0
                 for _col1 in cols]
    mean_col_diff = np.amin(col_diffs)
    return mean_col_diff


def get_mean_col_diff(cols, _col2):
    col_diffs = [np.sum(np.fabs(np.asarray(_col1, dtype=np.float32) - np.asarray(_col2, dtype=np.float32))) / 3.0
                 for _col1 in cols]
    mean_col_diff = np.mean(col_diffs)
    return mean_col_diff


def vis_run(start, length, class_id, run_txt,
            mask_sub_vis, mask_bool,
            class_id_to_col,
            text_img, text_x, text_y, font_size,
            vis_size, vis_image,
            n_rows, n_cols,
            resize_x, resize_y,
            flat_order, eos, arrow,
            frg_col,
            col):
    if col is None:
        col = class_id_to_col[class_id]

    if isinstance(col, str):
        col = col_bgr[col]

    # col_id = run_id % len(cols)
    # col = cols[col_id]
    mask_sub_vis = np.copy(mask_sub_vis)
    mask_sub_vis[mask_bool] = col

    text_img, text_x, text_y, text_bb = vis_utils.write_text(text_img, run_txt, text_x, text_y, col,
                                                             wait=100, bb=1, show=0, font_size=font_size)

    if eos:
        text_img, _, _ = vis_utils.write_text(text_img, 'EOS', text_x, text_y, frg_col,
                                              show=0, font_size=font_size)

    vis_mask_ = cv2.resize(mask_sub_vis, (vis_size, vis_size))
    vis_image_ = cv2.resize(vis_image, (vis_size, vis_size))

    # vis_mask_, text_bb = vis_utils.write_text(vis_mask_, run_txt, 5, 5, col, font_size=24, show=0, bb=1)

    run_center = int(start + length / 2)
    run_y, run_x = np.unravel_index([run_center, ], (n_rows, n_cols), order=flat_order)
    run_y, run_x = int(run_y[0]), int(run_x[0])

    run_x, run_y = int(run_x * resize_x), int(run_y * resize_y)
    """vis_mask_ to the right of vis_image_"""
    run_x += vis_size
    seg_rle_vis = np.concatenate((vis_image_, vis_mask_), axis=1)

    left, top, right, bottom, multiline = text_bb
    text_bb_x, text_bb_y = int((left + right) / 2), int(bottom)
    """text_img is to the right of vis_mask_"""
    text_bb_x += int(vis_size * 2)
    text_bb_y += 10
    seg_rle_vis = np.concatenate((seg_rle_vis, text_img), axis=1)

    if arrow:
        seg_rle_vis = cv2.arrowedLine(seg_rle_vis, (run_x, run_y), (text_bb_x, text_bb_y), col,
                                      2, tipLength=0.01)
    return seg_rle_vis, mask_sub_vis, (text_img, text_x, text_y)


def vis_video_run_txt(img_to_run_pixs, run_txt, vid_mask_vis, vid_vis,
                      text_info, font_size, vis_size,
                      time_as_class, tac_mask_sub, tac_mask,
                      tac_id_to_name, tac_id_to_col,
                      class_id_to_name, class_id_to_col,
                      col, eos, eos_col, arrow):
    assert vid_mask_vis.shape[0] == vid_vis.shape[0], "vid_mask_vis shape mismatch"

    vid_len, n_rows, n_cols, _ = vid_mask_vis.shape

    vid_masks_vis_ = resize_vid(vid_mask_vis, (vis_size, vis_size))
    vid_vis_ = resize_vid(vid_vis, (vis_size, vis_size))

    vid_masks_vis_ = concat_vid(vid_masks_vis_, axis=1, border=1)
    vid_vis_ = concat_vid(vid_vis_, axis=1, border=1)
    tac_mask_cat = None
    if time_as_class:
        vid_masks_vis_ = label_video_mask(vid_masks_vis_, class_id_to_name, class_id_to_col)

        tac_mask_sub_rgb = mask_id_to_vis_bgr(tac_mask_sub, tac_id_to_col)
        tac_mask_sub_rgb = resize_mask(tac_mask_sub_rgb, (vis_size, vis_size))
        # tac_mask_cat = tac_mask_sub_rgb
        tac_mask_rgb = mask_id_to_vis_bgr(tac_mask, tac_id_to_col)
        tac_mask_rgb = resize_mask(tac_mask_rgb, (vis_size, vis_size))

        # tac_mask_cat = concat_with_boder(tac_mask_rgb, tac_mask_sub_rgb, axis=1, border=1)
        tac_mask_cat = np.concatenate((tac_mask_rgb, tac_mask_sub_rgb), axis=1)
        tac_mask_cat = label_tac_mask(tac_mask_cat, tac_id_to_name, tac_id_to_col)

    vis_image_cat = np.concatenate((vid_vis_, vid_masks_vis_), axis=0)
    # if time_as_class:
    #     vis_image_cat = np.concatenate((vis_image_cat, tac_mask_cat), axis=1)

    text_img, text_x, text_y = text_info

    if text_img is None:
        if time_as_class:
            text_img_h, text_img_w = tac_mask_cat.shape[:2]
        else:
            overall_h = vis_image_cat.shape[0]
            text_img_h = overall_h

            # overall_w = int(18 * overall_h / 9)
            # text_img_w = overall_w - vis_image_cat.shape[1]
            text_img_w = vis_size
        text_img = np.full((text_img_h, text_img_w, 3), (0, 0, 0), dtype=np.uint8)

    text_img, text_x, text_y, text_bb = vis_utils.write_text(text_img, run_txt, text_x, text_y, col,
                                                             wait=100, bb=1, show=0, font_size=font_size)

    if eos:
        text_img, _, _ = vis_utils.write_text(
            text_img, 'EOS', text_x, text_y, eos_col, show=0, font_size=font_size)

    if time_as_class:
        text_img_vis = concat_with_boder(tac_mask_cat, text_img, 0, 1)
    else:
        text_img_vis = text_img

    vis_image_cat = concat_with_boder(vis_image_cat, text_img_vis, 1, 1)

    if arrow:
        left, top, right, bottom, multiline = text_bb
        text_bb_x, text_bb_y = int((left + right) / 2), int(bottom)
        """text_img is to the right of vid_vis_ and vid_masks_vis_"""
        text_bb_x += int(vis_size * vid_len)
        if time_as_class:
            text_bb_y += tac_mask_cat.shape[0]

        text_bb_y += 10

        resize_y, resize_x = float(vis_size) / n_rows, float(vis_size) / n_cols
        n_vis_imgs = len(img_to_run_pixs)
        for k, (img_id_, run_pixs_) in enumerate(img_to_run_pixs.items()):
            run_xs_ = [run_pix[1] for run_pix in run_pixs_]
            run_ys_ = [run_pix[0] for run_pix in run_pixs_]
            mean_run_ys_ = int(np.mean(run_ys_))
            mean_run_xs_ = int(np.mean(run_xs_))

            mean_run_xs_, mean_run_ys_ = int(mean_run_xs_ * resize_x), int(mean_run_ys_ * resize_y)

            """vis_vid_masks_ to the right of vis_vid_images_"""
            mean_run_xs_ += img_id_ * vis_size
            mean_run_ys_ += vis_size

            vis_image_cat = cv2.arrowedLine(vis_image_cat,
                                            (text_bb_x, text_bb_y),
                                            (mean_run_xs_, mean_run_ys_),
                                            col, 2, tipLength=0.02)

    return vis_image_cat, vid_mask_vis, (text_img, text_x, text_y)


def vis_video_and_masks(vid_vis, vid_mask, vid_mask_sub, vis_size,
                        class_id_to_col, class_id_to_name,
                        image_ids, time_as_class):
    n_classes = len(class_id_to_col)
    vid_len = vid_vis.shape[0]

    vis_images = []
    mask_sub_vis_ = []
    mask_sub_rgb_ = []
    mask_rgb_ = []
    vid_mask_sub_binary_vis = []

    for frame_id in range(vid_len):
        image = vid_vis[frame_id, ...]
        image_id = image_ids[frame_id]

        mask = vid_mask[frame_id, ...]
        mask_sub = vid_mask_sub[frame_id, ...]

        mask_rgb = mask_id_to_vis_bgr(mask, class_id_to_col)
        mask_sub_rgb = mask_id_to_vis_bgr(mask_sub, class_id_to_col)

        if time_as_class:
            mask_sub_binary_vis = mask_id_to_vis_bgr(mask_sub, class_id_to_col)
        else:
            mask_sub_binary_vis = mask_id_to_vis(mask_sub, n_classes=n_classes, to_rgb=1, copy=True)
            mask_sub_binary_vis[mask_sub_binary_vis > 0] = 255

        vid_mask_sub_binary_vis.append(mask_sub_binary_vis)

        vis_image = blend_mask(mask, image, class_id_to_col, alpha=0.25)
        vis_image = cv2.resize(vis_image, (vis_size, vis_size))

        file_txt = f'{image_id}'
        vis_image, _, _ = vis_utils.write_text(vis_image, file_txt, 5, 5, (255, 255, 255), font_size=24)

        vis_images.append(vis_image)
        mask_sub_vis_.append(cv2.resize(mask_sub_binary_vis, (vis_size, vis_size)))
        mask_sub_rgb_.append(cv2.resize(mask_sub_rgb, (vis_size, vis_size)))
        mask_rgb_.append(cv2.resize(mask_rgb, (vis_size, vis_size)))

    vid_vis = np.stack(vis_images, axis=0)
    vid_mask_sub_binary_vis = np.stack(vid_mask_sub_binary_vis, axis=0)

    vis_images_ = np.concatenate(vis_images, axis=1)
    mask_sub_vis_ = np.concatenate(mask_sub_vis_, axis=1)
    mask_sub_rgb_ = np.concatenate(mask_sub_rgb_, axis=1)
    mask_rgb_ = np.concatenate(mask_rgb_, axis=1)

    # cv2.imshow('vid_vis', vid_vis)
    cv2.imshow('vis_images_', vis_images_)
    # cv2.imshow('mask_sub_vis_', mask_sub_vis_)
    # cv2.imshow('mask_sub_vis_', mask_sub_vis_)
    # cv2.imshow('mask_sub_rgb', mask_sub_rgb_)
    # cv2.imshow('mask_rgb', mask_rgb_)
    cv2.waitKey(1)

    # cv2.waitKey(0 if n_runs > 0 else 10)
    return vid_vis, vid_mask_sub_binary_vis


def label_video_mask(vid_masks_vis_, class_id_to_name, class_id_to_col):
    text_x = 5
    text_y = 2
    font_size = 24
    # vid_masks_vis_, _, _ = vis_utils.write_text(
    #     vid_masks_vis_, f'Image mask with classes:', text_x, text_y,
    #     (255, 255, 255),
    #     wait=100, bb=0, show=0, font_size=font_size)
    # text_y += font_size
    for class_id, class_col in class_id_to_col.items():
        if class_id == 0:
            continue
        class_name = class_id_to_name[class_id]
        if isinstance(class_col, str):
            from eval_utils import col_bgr
            class_col = col_bgr[class_col]

        vid_masks_vis_, text_x, text_y = vis_utils.write_text(
            vid_masks_vis_, f'{class_name}   ', text_x, text_y,
            class_col,
            wait=100, bb=0, show=0, font_size=font_size)
        # text_y += font_size

    return vid_masks_vis_


def label_tac_mask(tac_mask_cat, tac_id_to_name, tac_id_to_col):
    font_size = 24
    text_x = 5
    text_y = 2
    # tac_mask_cat, _, _ = vis_utils.write_text(
    #     tac_mask_cat, f'Time-as-Class video mask with classes:', text_x, text_y,
    #     (255, 255, 255),
    #     wait=100, bb=0, show=0, font_size=font_size)
    # text_y += font_size
    for rle_id, rle_col in tac_id_to_col.items():
        if rle_id == 0:
            continue
        rle_name = tac_id_to_name[rle_id]
        tac_mask_cat, text_x, text_y = vis_utils.write_text(
            tac_mask_cat, f'{rle_name}   ', text_x, text_y,
            rle_col,
            wait=100, bb=0, show=0, font_size=font_size)
        # text_y += font_size
    return tac_mask_cat


def vis_video_rle(rle_cmp, class_id_to_col, class_id_to_name, image_ids,
                  vid_vis, vid_mask, vid_mask_sub, time_as_class,
                  length_as_class, max_length, flat_order,
                  tac_id_to_name, tac_id_to_col, pad_tokens):
    n_classes = len(class_id_to_col)

    starts, lengths = rle_cmp[:2]
    class_ids = None

    if len(rle_cmp) == 3:
        class_ids = rle_cmp[2]

    n_runs = len(starts)

    text_bkg_col = (0, 0, 0)
    # text_bkg_col = (50, 50, 50)
    frg_col = (255, 255, 255)
    temp_col = col_bgr['medium_purple']
    vis_size = 480
    font_size = 24
    _pause = 1

    vid_len, n_rows, n_cols, _ = vid_vis.shape
    assert vid_len == len(image_ids), "vid_len mismatch"

    if time_as_class:
        vid_mask_ = vid_mask_from_tac(vid_mask, vid_len, n_classes)
        vid_mask_sub_ = vid_mask_from_tac(vid_mask_sub, vid_len, n_classes)
        tac_mask = vid_mask
        tac_mask_sub = vid_mask_sub
        tac_mask_sub_flat = tac_mask_sub.flatten(order=flat_order)

        # cv2.imshow('tac_masks', tac_mask_cat)
        # cv2.waitKey(10)
    else:
        vid_mask_ = vid_mask
        vid_mask_sub_ = vid_mask_sub
        tac_mask = tac_mask_sub = tac_mask_sub_flat = None

    vid_vis, vid_mask_sub_binary_vis = vis_video_and_masks(
        vid_vis, vid_mask_, vid_mask_sub_,
        vis_size, class_id_to_col, class_id_to_name, image_ids, time_as_class)

    text_img = None
    text_x = text_y = 5

    vid_mask_sub_flat = vid_mask_sub_.flatten(order=flat_order)

    if pad_tokens:
        max_token_len = max(
            len(str(np.amax(starts))),
            len(str(np.amax(lengths)))
        )
        fmt = f'0{max_token_len}d'
    else:
        fmt = f'd'

    win_name = 'vid_rle_vis'
    # cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)

    for run_id, (start, length) in enumerate(zip(starts, lengths)):
        text_info = (text_img, text_x, text_y)
        run_txt = f'{int(start):{fmt}}, {int(length):{fmt}}, '

        if class_ids is not None:
            class_id = class_ids[run_id]
            if time_as_class:
                class_name = tac_id_to_name[class_id]
            else:
                class_name = class_id_to_name[class_id]
            run_txt = f'{run_txt}{class_name}, '
        else:
            class_id = 1

        if length_as_class:
            class_id = (length - 1) // max_length + 1
            length = (length - 1) % max_length + 1
            if time_as_class:
                class_name = tac_id_to_name[class_id]
                run_txt = f'{int(start)}, {class_name}-{length}, '
            else:
                class_name = class_id_to_name[class_id]
                run_txt = f'{int(start)}, {class_name}{length}, '

        if time_as_class:
            vid_mask_sub_binary_vis, img_to_run_pixs = vis_tac_run_pix(
                start, length,
                tac_mask_sub_flat, vid_mask_sub_binary_vis,
                temp_col, tac_id_to_col, class_id_to_col, flat_order,
            )
        else:
            vid_mask_sub_binary_vis, img_to_run_pixs = vis_video_run_pix(
                start, length,
                vid_mask_sub_flat, vid_mask_sub_binary_vis,
                temp_col, flat_order,
            )
        eos = run_id == n_runs - 1

        vid_rle_vis, _, _ = vis_video_run_txt(
            img_to_run_pixs, run_txt, vid_mask_sub_binary_vis,
            vid_vis, text_info, font_size, vis_size,
            time_as_class, tac_mask_sub, tac_mask,
            tac_id_to_name, tac_id_to_col,
            class_id_to_name, class_id_to_col,
            temp_col, eos, frg_col, arrow=1)

        vid_rle_vis = resize_ar(vid_rle_vis, int(1920 / 1.25), int(1080 / 1.25))
        cv2.imshow(win_name, vid_rle_vis)
        # cv2.imshow('tac_mask_cat', tac_mask_cat)
        k = cv2.waitKey(0 if _pause else 10)
        if k == 27:
            return
        elif k == 32:
            _pause = 1 - _pause
        elif k == ord('q'):
            exit()

        if time_as_class:
            vid_mask_sub_binary_vis, _ = vis_tac_run_pix(
                start, length,
                tac_mask_sub_flat, vid_mask_sub_binary_vis,
                None, tac_id_to_col, class_id_to_col, flat_order)
            col = tac_id_to_col[class_id]
        else:
            col = class_id_to_col[class_id]
            vid_mask_sub_binary_vis, _ = vis_video_run_pix(
                start, length,
                vid_mask_sub_flat, vid_mask_sub_binary_vis,
                col, flat_order,
            )
        if isinstance(col, str):
            col = col_bgr[col]

        vid_rle_vis, vid_mask_sub_binary_vis, text_info = vis_video_run_txt(
            img_to_run_pixs, run_txt, vid_mask_sub_binary_vis,
            vid_vis, text_info, font_size, vis_size,
            time_as_class, tac_mask_sub, tac_mask,
            tac_id_to_name, tac_id_to_col,
            class_id_to_name, class_id_to_col,
            col, eos, frg_col, arrow=0)
        text_img, text_x, text_y = text_info

    if n_runs > 0:
        vid_rle_vis = resize_ar(vid_rle_vis, int(1920 / 1.25), int(1080 / 1.25))
        cv2.imshow('vid_rle_vis', vid_rle_vis)
        k = cv2.waitKey(0)
        if k == 27:
            return
        elif k == 32:
            _pause = 1 - _pause
        elif k == ord('q'):
            exit()


def rle_to_lac(rle_cmp, max_length):
    starts, lengths, class_ids = rle_cmp

    lac = np.asarray([max_length * (class_id - 1) + length
                      for class_id, length in zip(class_ids, lengths)], dtype=np.int64)

    lengths_rec = lengths_from_lac(lac, max_length)
    class_ids_rec = class_ids_from_lac(lac, max_length)

    assert np.array_equal(lengths_rec, lengths), "lengths_rec mismatch"
    assert np.array_equal(class_ids_rec, class_ids), "class_ids_rec mismatch"

    rle_cmp = [starts, lac]
    return rle_cmp


def lengths_from_lac(lac, max_length):
    # lengths = np.asarray([(lac_id - 1) % max_length + 1
    #                       for lac_id in lac], dtype=np.int64)
    lengths = (lac - 1) % max_length + 1
    return lengths


def class_ids_from_lac(lac, max_length):
    # class_ids = np.asarray([(lac_id - 1) // max_length + 1
    #                         for lac_id in lac], dtype=np.int64)
    class_ids = (lac - 1) // max_length + 1

    return class_ids


def rle_from_lac(lac, max_length):
    assert np.all(lac > 0), "lac must be > 0"

    lengths = lengths_from_lac(lac, max_length)
    class_ids = class_ids_from_lac(lac, max_length)

    # rle_cmp.append(lengths)
    # rle_cmp.append(class_ids)
    return lengths, class_ids


def vis_rle(rle_cmp, length_as_class, max_length,
            class_id_to_col, class_id_to_name,
            image, mask, mask_sub, flat_order):
    starts, lengths = rle_cmp[:2]
    class_ids = None

    if len(rle_cmp) == 3:
        class_ids = rle_cmp[2]

    n_runs = len(starts)
    n_classes = len(class_id_to_col)
    # cols = get_cols(n_runs)

    mask_rgb = mask_id_to_vis_bgr(mask, class_id_to_col)
    mask_sub_rgb = mask_id_to_vis_bgr(mask_sub, class_id_to_col)

    mask_sub_vis = mask_id_to_vis(mask_sub, n_classes=n_classes, to_rgb=1, copy=True)
    mask_sub_vis[mask_sub_vis > 0] = 255

    vis_image = blend_mask(mask, image, class_id_to_col, alpha=0.25)

    text_x = text_y = 5
    # col = (0, 255, 0)
    bkg_col = (0, 0, 0)
    frg_col = (255, 255, 255)
    temp_col = col_bgr['medium_purple']
    # mask_col = (0, 255, 0)
    vis_size = 640
    font_size = 24
    n_rows, n_cols = mask_sub.shape
    resize_y, resize_x = float(vis_size) / n_rows, float(vis_size) / n_cols

    # mask_sub_vis_ = cv2.resize(mask_sub_vis, (vis_size, vis_size))
    # mask_sub_rgb_ = cv2.resize(mask_sub_rgb, (vis_size, vis_size))
    # mask_rgb_ = cv2.resize(mask_rgb, (vis_size, vis_size))
    # cv2.imshow('mask_sub_vis_', mask_sub_vis_)
    # cv2.imshow('mask_sub_rgb', mask_sub_rgb_)
    # cv2.imshow('mask_rgb', mask_rgb_)
    # cv2.waitKey(0)
    # return

    text_img = np.full((vis_size, vis_size, 3), bkg_col, dtype=np.uint8)
    mask_flat = mask_sub.flatten(order=flat_order)
    _pause = 1

    # from datetime import datetime
    # time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    # out_vis_path = f'vis_rle_{time_stamp}.mp4'
    # vid_writer = vis_utils.get_video_writer(out_vis_path)

    for run_id, (start, length) in enumerate(zip(starts, lengths)):
        # run_y, run_x = np.unravel_index([start, start + length], (n_rows, n_cols), order=flat_order)
        # mask_sub_rgb[run_y[0]:run_y[1], run_x[0]:run_x[1], :] = col

        # seg_rle_vis = resize_ar(seg_rle_vis, int(1920/1.25), int(1050/1.25))
        # vis_utils.write_frames_to_videos(vid_writer, seg_rle_vis)

        run_txt = f'{int(start)}, {int(length)}, '
        if class_ids is not None:
            class_id = class_ids[run_id]
            class_name = class_id_to_name[class_id]
            run_txt = f'{run_txt}{class_name}, '
        else:
            class_id = 1

        if length_as_class:
            class_id = (length - 1) // max_length + 1
            length = (length - 1) % max_length + 1
            class_name = class_id_to_name[class_id]
            run_txt = f'{int(start)}, {class_name}{length}, '

        mask_bool_flat = np.zeros_like(mask_flat, dtype=bool)
        mask_bool_flat[start:start + length] = True
        mask_bool = np.reshape(mask_bool_flat, (n_rows, n_cols), order=flat_order)

        eos = run_id == n_runs - 1

        seg_rle_vis, _, _ = vis_run(
            start, length, class_id, run_txt,
            mask_sub_vis, mask_bool,
            class_id_to_col,
            text_img, text_x, text_y, font_size,
            vis_size, vis_image,
            n_rows, n_cols,
            resize_x, resize_y,
            flat_order,
            arrow=True,
            eos=eos,
            frg_col=frg_col,
            col=temp_col)

        cv2.imshow('seg_rle_vis', seg_rle_vis)
        k = cv2.waitKey(0 if _pause else 100)
        if k == 27:
            exit()
        elif k == 32:
            _pause = 1 - _pause

        seg_rle_vis, mask_sub_vis, text_info = vis_run(
            start, length, class_id, run_txt,
            mask_sub_vis, mask_bool,
            class_id_to_col,
            text_img, text_x, text_y, font_size,
            vis_size, vis_image,
            n_rows, n_cols,
            resize_x, resize_y,
            flat_order,
            arrow=False,
            eos=eos,
            frg_col=frg_col,
            col=None)
        text_img, text_x, text_y = text_info

    cv2.waitKey(0)
    # vis_utils.close_video_writers(vid_writer)


def construct_rle(starts_rows, starts_cols, lengths, shape, starts_2d,
                  starts_offset, lengths_offset, flat_order):
    lengths += lengths_offset
    if starts_2d:
        starts_rows += 1 + starts_offset
        starts_cols += 1 + starts_offset
        rle = [item for sublist in zip(starts_rows, starts_cols, lengths) for item in sublist]
    else:
        starts = np.ravel_multi_index((starts_rows, starts_cols), shape, order=flat_order)

        starts += 1 + starts_offset + 1
        rle = [int(item) for sublist in zip(starts, lengths) for item in sublist]
    return rle


def deconstruct_rle(rle, shape, starts_2d, starts_offset, lengths_offset, flat_order):
    if starts_2d:
        start_rows, start_cols, lengths = [
            np.asarray(x, dtype=int) for x in (rle[0:][::3], rle[1:][::3], rle[2:][::3])]
        start_rows -= 1
        start_cols -= 1

        start_rows -= (starts_offset + 1)
        start_cols -= (starts_offset + 1)
    else:
        starts, lengths = [np.asarray(x, dtype=int) for x in (rle[0:][::2], rle[1:][::2])]
        starts -= (starts_offset + 1)
        start_rows, start_cols = np.unravel_index(starts, shape, order=flat_order)

    lengths -= lengths_offset

    return start_rows, start_cols, lengths


def supersample_rle(starts_sub, lengths_sub, subsample, shape, max_length, flat_order):
    n_rows, n_cols = shape
    n_rows_sub, n_cols_sub = int(n_rows / subsample), int(n_cols / subsample)
    max_length_sub = int(max_length / subsample)

    starts_rows_sub, starts_cols_sub = np.unravel_index(starts_sub, (n_rows_sub, n_cols_sub),
                                                        order=flat_order)

    starts_rows = starts_rows_sub / (n_rows_sub - 1) * (n_rows - 1)
    starts_cols = starts_cols_sub / (n_cols_sub - 1) * (n_cols - 1)

    starts_rows = starts_rows.astype(np.int64)
    starts_cols = starts_cols.astype(np.int64)

    lengths = (lengths_sub - 1).astype(np.float64) / (max_length_sub - 1) * (max_length - 1) + 1
    lengths = lengths.astype(np.int64)

    starts = np.ravel_multi_index((starts_rows, starts_cols), shape, order=flat_order)

    return starts, lengths


def subsample_rle(starts, lengths, subsample, shape, max_length, flat_order):
    if len(starts) == 0:
        return starts, lengths

    n_rows, n_cols = shape
    max_length_sub = int(max_length / subsample)
    n_rows_sub, n_cols_sub = int(n_rows / subsample), int(n_cols / subsample)

    starts_rows, starts_cols = np.unravel_index(starts, (n_rows, n_cols), order=flat_order)

    starts_rows_norm, starts_cols_norm = (starts_rows.astype(np.float64) / (n_rows - 1),
                                          starts_cols.astype(np.float64) / (n_cols - 1))

    """
    lengths goes from 1 to max_length so 1 must be subtracted before normalizing so lengths_norm starts from 0
    and un-normalizing works correctly    
    """
    lengths_norm = (lengths.astype(np.float64) - 1) / (max_length - 1)

    lengths = ((lengths_norm * (max_length_sub - 1)) + 1).astype(np.int64)
    starts_rows = (starts_rows_norm * (n_rows_sub - 1)).astype(np.int64)
    starts_cols = (starts_cols_norm * (n_cols_sub - 1)).astype(np.int64)

    rle_all = list(zip(starts_rows, starts_cols, lengths))
    rle_unique = remove_duplicates(rle_all)

    starts_rows, starts_cols, lengths = zip(*rle_unique)

    starts_rows = np.asarray(starts_rows)
    starts_cols = np.asarray(starts_cols)
    lengths = np.asarray(lengths)

    starts = np.ravel_multi_index((starts_rows, starts_cols), (n_rows_sub, n_cols_sub),
                                  order=flat_order)

    return starts, lengths


def rle_to_tokens(rle_cmp, shape, length_as_class,
                  starts_offset, lengths_offset, class_offset,
                  starts_2d, flat_order):
    starts, lengths = rle_cmp[:2]

    if len(starts) == 0:
        return []

    if length_as_class:
        assert len(rle_cmp) == 2, "rle must have 2 components with length_as_class enabled"
        lengths += class_offset
    else:
        lengths += lengths_offset

    if starts_2d:
        starts_rows, starts_cols = np.unravel_index(starts, shape, order=flat_order)
        starts_rows += (starts_offset + 1)
        starts_cols += (starts_offset + 1)
        rle_tokens_cmp = [starts_rows, starts_cols, lengths]
    else:
        starts += (starts_offset + 1)
        rle_tokens_cmp = [starts, lengths]

    if len(rle_cmp) == 3:
        class_ids = np.asarray(rle_cmp[2])
        class_ids += class_offset
        rle_tokens_cmp.append(class_ids)
    else:
        assert len(rle_cmp) == 2, "rle_cmp length must be 2 or 3"

    rle_tokens = [int(item) for sublist in zip(*rle_tokens_cmp) for item in sublist]

    return rle_tokens


def rle_to_tokens_tf(rle_cmp, shape, length_as_class,
                     starts_offset, lengths_offset, class_offset,
                     starts_2d, flat_order):
    starts, lengths = rle_cmp[:2]

    if len(starts) == 0:
        return []

    if length_as_class:
        assert len(rle_cmp) == 2, "rle must have 2 components with length_as_class enabled"
        lengths = lengths + class_offset
    else:
        lengths = lengths + lengths_offset

    if starts_2d:
        starts_rows, starts_cols = tf.experimental.numpy.unravel_index(starts, shape, order=flat_order)
        starts_rows += (starts_offset + 1)
        starts_cols += (starts_offset + 1)
        rle_tokens_cmp = [starts_rows, starts_cols, lengths]
    else:
        starts += (starts_offset + 1)
        rle_tokens_cmp = [starts, lengths]

    if len(rle_cmp) == 3:
        class_ids = rle_cmp[2] + class_offset
        rle_tokens_cmp.append(class_ids)
    else:
        assert len(rle_cmp) == 2, "rle_cmp length must be 2 or 3"

    rle_tokens = tf.experimental.numpy.asarray([int(item) for sublist in zip(*rle_tokens_cmp) for item in sublist])

    return rle_tokens


def mask_from_logits(
        rle_logits,
        shape,
        max_length,
        n_classes,
        starts_offset, lengths_offset, class_offset,
        length_as_class,
        starts_2d,
        flat_order,
        multi_class,
        max_seq_len,
        vocab_size,
        allow_overlap,
):
    rle_cmp = rle_from_logits(
        rle_logits=rle_logits,
        shape=shape,
        max_length=max_length,
        n_classes=n_classes,
        starts_offset=starts_offset,
        lengths_offset=lengths_offset,
        class_offset=class_offset,
        time_as_class=False,
        length_as_class=length_as_class,
        flat_order=flat_order,
        starts_2d=starts_2d,
        multi_class=multi_class,
        vid_len=1,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        allow_overlap=allow_overlap,
    )
    starts, lengths = rle_cmp[:2]
    if len(rle_cmp) == 3:
        class_ids = rle_cmp[2]
    else:
        class_ids = [1, ] * len(starts)

    mask = rle_to_mask(
        starts, lengths, class_ids,
        shape,
    )
    return mask, rle_cmp


def vid_mask_from_logits(
        rle_logits,
        shape,
        max_length,
        n_classes,
        starts_offset, lengths_offset, class_offset,
        time_as_class,
        length_as_class,
        starts_2d,
        flat_order,
        multi_class,
        vid_len,
        max_seq_len,
        vocab_size,
        allow_overlap,
):
    rle_cmp = rle_from_logits(
        rle_logits,
        shape,
        max_length,
        n_classes,
        starts_offset, lengths_offset, class_offset,
        time_as_class,
        length_as_class,
        starts_2d,
        flat_order,
        multi_class,
        vid_len,
        max_seq_len,
        vocab_size,
        allow_overlap,
    )
    starts, lengths = rle_cmp[:2]
    if len(rle_cmp) == 3:
        class_ids = rle_cmp[2]
    else:
        class_ids = [1, ] * len(starts)

    mask, tac_mask = rle_to_vid_mask(
        starts, lengths, class_ids, shape,
        vid_len, time_as_class, n_classes,
    )
    return mask, tac_mask, rle_cmp


def selective_argmax(arr, idx_range):
    start_idx, end_idx = idx_range
    return start_idx + np.argmax(arr[:, start_idx:end_idx], axis=1)


def rle_from_logits(
        rle_logits,
        shape,
        max_length,
        n_classes,
        starts_offset, lengths_offset, class_offset,
        time_as_class,
        length_as_class,
        starts_2d,
        flat_order,
        multi_class,
        vid_len,
        max_seq_len,
        vocab_size,
        allow_overlap,
):
    """generate RLE for both static and video masks"""

    # assert not starts_2d, "starts_2d is not supported yet"

    max_seq_len_, vocab_size_ = rle_logits.shape
    assert max_seq_len_ == max_seq_len, "max_seq_len mismatch"
    assert vocab_size_ == vocab_size, "vocab_size mismatch"

    n_rows, n_cols = shape

    rle_tokens_raw = np.argmax(rle_logits, axis=1).squeeze()
    n_tokens_raw = len(rle_tokens_raw)

    """
    index of the first non-zero (non-padding) token from end
    EOS is the next token to this one, i.e. the last continuous zero token from end
    """
    rle_tokens_inv = rle_tokens_raw[::-1]
    eos_idxs = np.nonzero(rle_tokens_inv)[0]
    if len(eos_idxs) == 0:
        eos_idx = 0
    else:
        eos_idx = eos_idxs[0]
        eos_idx = n_tokens_raw - eos_idx

    rle_logits_non_padding = rle_logits[:eos_idx, :]
    seq_len = rle_logits_non_padding.shape[0]

    if starts_2d:
        assert n_rows == n_cols, "n_rows and n_cols must be same for starts_2d"
        starts_bins = n_rows
    else:
        starts_bins = n_rows * n_cols

    if vid_len == 1:
        assert not time_as_class, "vid_len must be > 1 for time_as_class"

    if time_as_class:
        assert vid_len > 1, "vid_len must be > 1 for time_as_class"

    if vid_len > 1 and not time_as_class:
        starts_bins *= vid_len

    assert vocab_size >= starts_offset + starts_bins, "invalid vocab_size"

    has_class_tokens = (time_as_class or multi_class) and not length_as_class

    n_tokens_per_run = 3 if has_class_tokens else 2
    if starts_2d:
        n_tokens_per_run += 1

    if seq_len < n_tokens_per_run:
        rle_cmp = [[], [], []] if has_class_tokens else [[], []]
        return rle_cmp

    if seq_len % n_tokens_per_run != 0:
        n_extra_tokens = seq_len % n_tokens_per_run
        rle_logits_non_padding = rle_logits_non_padding[:-n_extra_tokens, :]

    rle_id = 0
    starts_token_range = [starts_offset, starts_offset + starts_bins]
    if starts_2d:
        starts_rows_logits = rle_logits_non_padding[rle_id::n_tokens_per_run, :]
        starts_rows_tokens = selective_argmax(starts_rows_logits, starts_token_range)
        starts_rows = starts_rows_tokens - starts_offset - 1
        rle_id += 1

        starts_cols_logits = rle_logits_non_padding[rle_id::n_tokens_per_run, :]
        starts_cols_tokens = selective_argmax(starts_cols_logits, starts_token_range)
        starts_cols = starts_cols_tokens - starts_offset - 1
        rle_id += 1

        assert len(starts_rows) == len(starts_cols), "starts_rows-starts_cols len mismatch"

        starts = np.ravel_multi_index((starts_rows, starts_cols), shape, order=flat_order)
    else:
        starts_logits = rle_logits_non_padding[rle_id::n_tokens_per_run, :]
        starts_token_range = [starts_offset, starts_offset + starts_bins]
        starts_tokens = selective_argmax(starts_logits, starts_token_range)
        starts = starts_tokens - starts_offset - 1
        rle_id += 1

    rle_cmp = [starts, ]

    if not length_as_class:
        assert allow_overlap or starts_offset >= lengths_offset + max_length, ("len_token_range overlaps "
                                                                               "starts_token_range")

        len_token_range = [lengths_offset, lengths_offset + max_length]
        len_logits = rle_logits_non_padding[rle_id::n_tokens_per_run, :]
        len_tokens = selective_argmax(len_logits, len_token_range)
        lengths = len_tokens - lengths_offset

        starts_ = rle_cmp[0]
        assert len(lengths) == len(starts_), "lengths-starts len mismatch"
        rle_cmp.append(lengths)
        rle_id += 1

    if multi_class or time_as_class or length_as_class:
        n_classes_ = n_classes
        if time_as_class:
            n_classes_ = n_classes_ ** vid_len
        if length_as_class:
            n_total_classes = max_length * (n_classes_ - 1)
        else:
            n_total_classes = n_classes_

        assert allow_overlap or starts_offset >= class_offset + n_total_classes, ("class_token_range overlaps "
                                                                                  "starts_token_range")

        if not length_as_class:
            assert allow_overlap or lengths_offset >= class_offset + n_total_classes, ("class_token_range overlaps "
                                                                                       "len_token_range")

        class_token_range = [class_offset, class_offset + n_total_classes]
        class_logits = rle_logits_non_padding[rle_id::n_tokens_per_run, :]
        class_tokens = selective_argmax(class_logits, class_token_range)
        class_ids = class_tokens - class_offset

        starts_ = rle_cmp[0]
        assert len(class_ids) == len(starts_), "class_ids-starts len mismatch"

        rle_cmp.append(class_ids)

    if length_as_class:
        assert len(rle_cmp) == 2, "rle_cmp len must be 2 for length_as_class"
        starts, lac = rle_cmp
        lengths, class_ids = rle_from_lac(lac, max_length)
        rle_cmp = [starts, lengths, class_ids]

    return rle_cmp


def mask_from_instance_wise_tokens(
        rle_tokens,
        shape,
        n_classes,
        allow_extra,
        starts_offset, lengths_offset, class_offset,
        starts_2d, flat_order,
        ignore_invalid=False,
        rle_logits=None,
):
    mask = np.zeros(tuple(shape), dtype=np.uint8)
    instance_mask_cmb = np.zeros(tuple(shape), dtype=np.uint8)

    # masks, classes, scores for all instances
    instance_info = [[], [], [], None]

    # starts, lengths, class_ids
    rle_cmp = [[], [], []]

    if len(rle_tokens) == 0:
        return mask, instance_info, rle_cmp

    class_tokens = np.asarray(list(range(1, n_classes)))
    class_tokens += class_offset

    inst_end_idxs = np.nonzero(np.in1d(rle_tokens, class_tokens))[0]

    inst_start_idx = 0
    n_instances = len(inst_end_idxs)
    assert n_instances <= 255, "uint8 can only handle upto 255 instances"

    for instance_id in range(n_instances):
        inst_end_idx = inst_end_idxs[instance_id]
        inst_rle_tokens = rle_tokens[inst_start_idx:inst_end_idx]
        inst_class_token = rle_tokens[inst_end_idx]
        inst_class_id = inst_class_token - class_offset

        if rle_logits is None:
            instance_score = 1
        else:
            logits_ = rle_logits[inst_end_idx, :]
            logits_norm_ = scipy.special.softmax(logits_)
            instance_score = float(logits_norm_[inst_class_token])

        instance_rle_cmp = rle_from_tokens(
            inst_rle_tokens,
            shape,
            allow_extra=allow_extra,
            length_as_class=False,
            starts_offset=starts_offset,
            lengths_offset=lengths_offset,
            class_offset=class_offset,
            starts_2d=starts_2d,
            multi_class=False,
            flat_order=flat_order,
            ignore_invalid=ignore_invalid,
        )
        starts, lengths = instance_rle_cmp
        class_ids = [1, ] * len(starts)

        instance_mask = rle_to_mask(
            starts, lengths, class_ids,
            shape,
        )
        instance_mask = instance_mask.astype(bool)
        instance_info[0].append(instance_mask)
        instance_info[1].append(inst_class_id)
        instance_info[2].append(instance_score)

        instance_mask_cmb[instance_mask] = instance_id + 1
        mask[instance_mask] = inst_class_id

        rle_cmp[0] += list(starts)
        rle_cmp[1] += list(lengths)
        rle_cmp[2] += list(class_ids)

        inst_start_idx = inst_end_idx + 1

    rle_cmp = [np.asarray(k) for k in rle_cmp]
    instance_info[3] = instance_mask_cmb
    return mask, instance_info, rle_cmp

def mask_from_class_wise_tokens(
        rle_tokens,
        shape,
        n_classes,
        allow_extra,
        starts_offset, lengths_offset, class_offset,
        starts_2d, flat_order,
        ignore_invalid=False,
):
    mask = np.zeros(tuple(shape), dtype=np.uint8)
    # starts, lengths, class_ids
    rle_cmp = [[], [], []]

    if len(rle_tokens) == 0:
        return mask, rle_cmp

    class_tokens = np.asarray(list(range(1, n_classes)))
    class_tokens += class_offset

    cls_end_idxs = np.nonzero(np.in1d(rle_tokens, class_tokens))[0]

    cls_start_idx = 0
    n_cls = len(cls_end_idxs)
    assert n_cls <= 255, "uint8 can only handle upto 255 classes"

    for cls_id in range(n_cls):
        cls_end_idx = cls_end_idxs[cls_id]
        cls_rle_tokens = rle_tokens[cls_start_idx:cls_end_idx]
        class_token = rle_tokens[cls_end_idx]
        class_id = class_token - class_offset

        cls_rle_cmp = rle_from_tokens(
            cls_rle_tokens,
            shape,
            allow_extra=allow_extra,
            length_as_class=False,
            starts_offset=starts_offset,
            lengths_offset=lengths_offset,
            class_offset=class_offset,
            starts_2d=starts_2d,
            multi_class=False,
            flat_order=flat_order,
            ignore_invalid=ignore_invalid,
        )
        starts, lengths = cls_rle_cmp
        class_ids = [1, ] * len(starts)

        cls_mask = rle_to_mask(
            starts, lengths, class_ids,
            shape,
        )
        cls_mask = cls_mask.astype(bool)
        mask[cls_mask] = class_id

        rle_cmp[0] += list(starts)
        rle_cmp[1] += list(lengths)
        rle_cmp[2] += list(class_ids)

        cls_start_idx = cls_end_idx + 1

    rle_cmp = [np.asarray(k) for k in rle_cmp]
    return mask, rle_cmp


def mask_from_tokens(
        rle_tokens, shape,
        allow_extra,
        length_as_class,
        max_length,
        starts_offset, lengths_offset, class_offset,
        starts_2d, multi_class, flat_order,
        ignore_invalid=False,

):
    if len(rle_tokens) == 0:
        mask = np.zeros(tuple(shape), dtype=np.uint8)
        rle_cmp = [[], []]
        if not length_as_class and multi_class:
            rle_cmp.append([])
        return mask, rle_cmp
    else:
        rle_cmp = rle_from_tokens(
            rle_tokens,
            shape,
            allow_extra=allow_extra,
            length_as_class=length_as_class,
            starts_offset=starts_offset,
            lengths_offset=lengths_offset,
            class_offset=class_offset,
            starts_2d=starts_2d,
            multi_class=multi_class,
            flat_order=flat_order,
            ignore_invalid=ignore_invalid,
        )
    if length_as_class:
        starts, lac = rle_cmp

        lengths = (lac - 1) % max_length + 1
        class_ids = (lac - 1) // max_length + 1

        rle_cmp = [starts, lengths, class_ids]

    starts, lengths = rle_cmp[:2]
    if multi_class:
        class_ids = rle_cmp[2]
    else:
        class_ids = [1, ] * len(starts)

    mask = rle_to_mask(
        starts, lengths, class_ids,
        shape,
    )
    return mask, rle_cmp


def bytes_to_str_list(image_ids):
    batch_size, vid_len = image_ids.shape
    image_ids_ = image_ids.astype(str)
    image_ids = list(image_ids_)

    assert len(image_ids) == batch_size, "batch_size mismatch"
    assert len(image_ids[0]) == vid_len, "vid_len mismatch"

    return image_ids


def vid_mask_from_tokens(
        rle_tokens,
        allow_extra,
        vid_len,
        shape,
        n_classes,
        time_as_class,
        length_as_class,
        max_length,
        starts_offset, lengths_offset, class_offset,
        starts_2d,
        multi_class,
        flat_order,
        ignore_invalid,
):
    n_rows, n_cols = shape
    has_class_tokens = (time_as_class or multi_class) and not length_as_class

    if len(rle_tokens) == 0:
        mask = np.zeros((vid_len, n_rows, n_cols), dtype=np.uint8)
        tac_mask = np.zeros((n_rows, n_cols), dtype=np.uint8) if time_as_class else None
        rle_cmp = [[], [], []] if has_class_tokens else [[], []]
        return mask, tac_mask, rle_cmp

    rle_cmp = vid_rle_from_tokens(
        rle_tokens,
        allow_extra=allow_extra,
        shape=shape,
        time_as_class=time_as_class,
        length_as_class=length_as_class,
        starts_offset=starts_offset,
        lengths_offset=lengths_offset,
        class_offset=class_offset,
        starts_2d=starts_2d,
        max_length=max_length,
        multi_class=multi_class,
        flat_order=flat_order,
        ignore_invalid=ignore_invalid,
    )
    starts, lengths = rle_cmp[:2]

    if len(rle_cmp) == 3:
        class_ids = rle_cmp[2]
    else:
        class_ids = [1, ] * len(starts)

    mask, tac_mask = rle_to_vid_mask(
        starts, lengths, class_ids,
        shape, vid_len, time_as_class,
        n_classes
    )

    return mask, tac_mask, rle_cmp


def rle_from_instance_wise_tokens(
        rle_tokens,
        shape,
        allow_extra,
        length_as_class,
        starts_offset, lengths_offset, class_offset,
        starts_2d, multi_class, flat_order,
        ignore_invalid=False
):
    has_class_tokens = multi_class and not length_as_class

    if length_as_class:
        assert lengths_offset == class_offset, "lengths_offset and class_offset must be same for length_as_class"

    if len(rle_tokens) == 0:
        rle_cmp = [[], [], []] if has_class_tokens else [[], []]
        return rle_cmp

    n_tokens_per_run = 3 if has_class_tokens else 2
    if starts_2d:
        n_tokens_per_run += 1

    seq_len = len(rle_tokens)
    n_extra_tokens = seq_len % n_tokens_per_run
    if n_extra_tokens != 0:
        if not allow_extra:
            raise AssertionError(f"rle_tokens length must be divisible by {n_tokens_per_run}")
        rle_tokens = rle_tokens[:-n_extra_tokens]

    if starts_2d:
        starts_rows = np.asarray(rle_tokens[0:][::n_tokens_per_run], dtype=int)
        starts_cols = np.asarray(rle_tokens[1:][::n_tokens_per_run], dtype=int)

        starts_rows -= (starts_offset + 1)
        starts_cols -= (starts_offset + 1)
        valid_starts_rows = starts_rows >= 0
        valid_starts_cols = starts_cols >= 0

        assert ignore_invalid or np.all(valid_starts_rows), "starts_rows must be >= 0"
        assert ignore_invalid or np.all(valid_starts_cols), "starts_rows must be >= 0"

        len_id = 2

        invalid_starts_rows = np.logical_not(valid_starts_rows)
        invalid_starts_cols = np.logical_not(valid_starts_cols)

        invalid_starts = np.logical_or(invalid_starts_rows, invalid_starts_cols)

        starts_rows[invalid_starts_rows] = 0
        starts_cols[invalid_starts_cols] = 0

        starts = np.ravel_multi_index((starts_rows, starts_cols), shape, order=flat_order)
        starts[invalid_starts] = -1
    else:
        starts = np.asarray(rle_tokens[0:][::n_tokens_per_run], dtype=int)
        starts = starts - (starts_offset + 1)

        len_id = 1

    assert ignore_invalid or np.all(starts >= 0), "starts must be >= 0"

    lengths = np.asarray(rle_tokens[len_id:][::n_tokens_per_run], dtype=int)
    lengths = lengths - lengths_offset

    assert ignore_invalid or np.all(lengths > 0), "lengths must be > 0"

    valid_bool = np.logical_and(starts >= 0, lengths > 0)

    rle_cmp = [starts, lengths]

    if has_class_tokens:
        class_ids = np.asarray(rle_tokens[len_id + 1:][::n_tokens_per_run], dtype=int)
        class_ids -= class_offset
        assert ignore_invalid or np.all(class_ids > 0), "class_ids must be > 0"

        valid_bool = np.logical_and(valid_bool, class_ids > 0)

        rle_cmp.append(class_ids)

    if ignore_invalid:
        valid_ids = np.nonzero(valid_bool)
        for rle_id, rle_arr in enumerate(rle_cmp):
            rle_cmp[rle_id] = rle_arr[valid_ids]

    return rle_cmp


def rle_from_tokens(
        rle_tokens,
        shape,
        allow_extra,
        length_as_class,
        starts_offset, lengths_offset, class_offset,
        starts_2d, multi_class, flat_order,
        ignore_invalid=False
):
    has_class_tokens = multi_class and not length_as_class

    if length_as_class:
        assert lengths_offset == class_offset, "lengths_offset and class_offset must be same for length_as_class"

    seq_len = len(rle_tokens)

    if seq_len == 0:
        rle_cmp = [[], [], []] if has_class_tokens else [[], []]
        return rle_cmp

    n_tokens_per_run = 3 if has_class_tokens else 2
    if starts_2d:
        n_tokens_per_run += 1

    n_extra_tokens = seq_len % n_tokens_per_run
    if n_extra_tokens != 0:
        if not allow_extra:
            raise AssertionError(f"rle_tokens length must be divisible by {n_tokens_per_run}")
        rle_tokens = rle_tokens[:-n_extra_tokens]

    if starts_2d:
        starts_rows = np.asarray(rle_tokens[0:][::n_tokens_per_run], dtype=int)
        starts_cols = np.asarray(rle_tokens[1:][::n_tokens_per_run], dtype=int)

        starts_rows -= (starts_offset + 1)
        starts_cols -= (starts_offset + 1)
        valid_starts_rows = starts_rows >= 0
        valid_starts_cols = starts_cols >= 0

        assert ignore_invalid or np.all(valid_starts_rows), "starts_rows must be >= 0"
        assert ignore_invalid or np.all(valid_starts_cols), "starts_rows must be >= 0"

        len_id = 2

        invalid_starts_rows = np.logical_not(valid_starts_rows)
        invalid_starts_cols = np.logical_not(valid_starts_cols)

        invalid_starts = np.logical_or(invalid_starts_rows, invalid_starts_cols)

        starts_rows[invalid_starts_rows] = 0
        starts_cols[invalid_starts_cols] = 0

        starts = np.ravel_multi_index((starts_rows, starts_cols), shape, order=flat_order)
        starts[invalid_starts] = -1
    else:
        starts = np.asarray(rle_tokens[0:][::n_tokens_per_run], dtype=int)
        starts = starts - (starts_offset + 1)

        len_id = 1

    assert ignore_invalid or np.all(starts >= 0), "starts must be >= 0"

    lengths = np.asarray(rle_tokens[len_id:][::n_tokens_per_run], dtype=int)
    lengths = lengths - lengths_offset

    assert ignore_invalid or np.all(lengths > 0), "lengths must be > 0"

    valid_bool = np.logical_and(starts >= 0, lengths > 0)

    rle_cmp = [starts, lengths]

    if has_class_tokens:
        class_ids = np.asarray(rle_tokens[len_id + 1:][::n_tokens_per_run], dtype=int)
        class_ids -= class_offset
        assert ignore_invalid or np.all(class_ids > 0), "class_ids must be > 0"

        valid_bool = np.logical_and(valid_bool, class_ids > 0)

        rle_cmp.append(class_ids)

    if ignore_invalid:
        valid_ids = np.nonzero(valid_bool)
        for rle_id, rle_arr in enumerate(rle_cmp):
            rle_cmp[rle_id] = rle_arr[valid_ids]

    return rle_cmp


def vid_rle_from_tokens(
        rle_tokens,
        allow_extra,
        shape,
        time_as_class,
        length_as_class,
        starts_offset, lengths_offset, class_offset,
        starts_2d,
        max_length,
        multi_class,
        flat_order,
        ignore_invalid,
):
    if length_as_class:
        assert lengths_offset == class_offset, "lengths_offset and class_offset must be same for length_as_class"

    has_class_tokens = (multi_class or time_as_class) and not length_as_class
    if len(rle_tokens) == 0:
        rle_cmp = [[], []]
        if has_class_tokens:
            rle_cmp.append([])
        return rle_cmp

    n_tokens_per_run = 2
    if starts_2d:
        n_tokens_per_run += 1
    if has_class_tokens:
        n_tokens_per_run += 1

    seq_len = len(rle_tokens)
    n_extra_tokens = seq_len % n_tokens_per_run
    if n_extra_tokens != 0:
        if not allow_extra:
            raise AssertionError(f"found rle with length {seq_len} that is not divisible by {n_tokens_per_run}")
        rle_tokens = rle_tokens[:-n_extra_tokens]

    if starts_2d:
        starts_rows = np.array(rle_tokens[0:][::n_tokens_per_run], dtype=np.int64)
        starts_cols = np.array(rle_tokens[1:][::n_tokens_per_run], dtype=np.int64)

        starts_rows -= (starts_offset + 1)
        starts_cols -= (starts_offset + 1)

        len_id = 2

        starts = np.ravel_multi_index((starts_rows, starts_cols), shape, order=flat_order)
    else:
        starts = np.array(rle_tokens[0:][::n_tokens_per_run], dtype=np.int64)
        starts -= (starts_offset + 1)

        len_id = 1

    valid_starts = starts >= 0
    assert ignore_invalid or np.all(valid_starts), "starts must be >= 0"

    lengths = np.array(rle_tokens[len_id:][::n_tokens_per_run], dtype=np.int64)
    lengths -= lengths_offset

    if length_as_class:
        valid_lengths_pos = lengths > 0
        invalid_ids = np.nonzero(np.logical_not(valid_lengths_pos))
        if ignore_invalid:
            lengths[invalid_ids] = 1
        else:
            assert np.all(valid_lengths_pos), "lac must be > 0"
        lengths, class_ids = rle_from_lac(lengths, max_length)
        if ignore_invalid:
            lengths[invalid_ids] = 0
            class_ids[invalid_ids] = 0
        rle_cmp = [starts, lengths, class_ids]
    else:
        rle_cmp = [starts, lengths]

    valid_lengths_pos = lengths > 0
    valid_lengths_max = lengths <= max_length

    assert ignore_invalid or np.all(valid_lengths_pos), "lengths must be > 0"
    assert ignore_invalid or np.all(valid_lengths_max), f"run length cannot be > {max_length}"

    if has_class_tokens:
        assert len(rle_cmp) == 2, "rle_cmp must have length 2 to append class IDs"

        class_ids = np.array(rle_tokens[len_id + 1:][::n_tokens_per_run], dtype=np.int64)
        class_ids -= class_offset
        rle_cmp.append(class_ids)

    valid_class = None

    if len(rle_cmp) == 3:
        class_ids = rle_cmp[2]
        valid_class = class_ids > 0

        assert ignore_invalid or np.all(valid_class), "class_ids must be > 0"

    if ignore_invalid:
        valid_lengths = np.logical_and(valid_lengths_pos, valid_lengths_max)
        valid_bool = np.logical_and(valid_starts, valid_lengths)
        if valid_class is not None:
            valid_bool = np.logical_and(valid_bool, valid_class)

        valid_ids = np.nonzero(valid_bool)
        for rle_id, rle_arr in enumerate(rle_cmp):
            rle_cmp[rle_id] = rle_arr[valid_ids]

    return rle_cmp


def get_rle_class_ids(mask, starts, lengths, class_id_to_col, order):
    n_classes = len(class_id_to_col)

    mask_flat = mask.flatten(order=order)

    class_ids = [mask_flat[k] for k in starts]

    if 0 in class_ids:
        print("class_ids must be non-zero")
        class_ids = np.asarray(class_ids)
        zero_idxs = np.nonzero(class_ids == 0)[0]
        mask_vis_rgb = mask_id_to_vis_bgr(mask, class_id_to_col)
        mask_vis = mask_id_to_vis(mask, n_classes, copy=True)
        mask_vis_res = resize_mask(mask_vis, (640, 640))
        cv2.imshow('mask_vis', mask_vis_res)

        for idx in zero_idxs:
            start, length = starts[idx], lengths[idx]
            mask_bool_flat = np.zeros_like(mask_flat, dtype=bool)
            mask_bool_flat[start:start + length] = True
            mask_bool = np.reshape(mask_bool_flat, mask.shape)

            col = (255, 255, 255)

            mask_vis_rgb[mask_bool] = col
            mask_vis_rgb_res = resize_mask(mask_vis_rgb, (640, 640))

            cv2.imshow('mask_vis_rgb', mask_vis_rgb_res)

            cv2.waitKey(0)

    assert np.all(np.asarray(class_ids) > 0), "class_ids must be > 0"
    assert np.all(np.asarray(class_ids) <= n_classes), f"class_ids must be <= {n_classes}"

    for start, length, class_id in zip(starts, lengths, class_ids):
        run_class_ids = mask_flat[start:start + length]

        assert np.all(run_class_ids == class_id), "multiple class IDs found in the same run"

        # if not np.all(run_class_ids == class_id):
        #     print("multiple class IDs found in the same run")
        #     mask_vis = mask_id_to_vis(mask, n_classes, copy=True)
        #     mask_vis = resize_mask(mask_vis, (640, 640), n_classes, is_vis=True)
        #     cv2.imshow('mask_vis', mask_vis)
        #     cv2.waitKey(0)

    return class_ids


def get_rle_class_ids_tf(mask, starts, lengths, class_id_to_col, order):
    n_classes = len(class_id_to_col)

    mask_flat = tf.experimental.numpy.flatten(mask, order=order)

    class_ids = [mask_flat[k] for k in starts]

    return class_ids


def rle_to_2d(rle, mask, flat_order):
    n_rows, n_cols = mask.shape[:2]

    starts, lengths = [np.asarray(x, dtype=int) for x in (rle[0:][::2], rle[1:][::2])]

    starts_rows, starts_cols = np.unravel_index(starts, (n_rows, n_cols), order=flat_order)

    assert np.all(starts_rows <= n_rows - 1), f"starts_rows cannot be > {n_rows - 1}"
    assert np.all(starts_cols <= n_cols - 1), f"starts_rows cannot be > {n_cols - 1}"

    rle = [int(item) for sublist in zip(starts_rows, starts_cols, lengths) for item in sublist]

    return rle


def nbits_to_decimal(nbits, base):
    assert np.all(nbits < base), f"bits must be < {base}"

    out = 0
    for bit_id, nbit in enumerate(nbits):
        out += (base ** bit_id) * nbit
    return out


def tac_to_vid_class_ids(tac_id, vid_len: int, n_classes: int):
    """convert time-as-class IDs to time-specific class IDs"""
    vid_class_ids = []
    for vid_id in range(vid_len):
        vid_class_id = tac_id % n_classes
        tac_id = tac_id // n_classes

        vid_class_ids.append(vid_class_id)

    return vid_class_ids


def vid_mask_from_tac(mask: np.ndarray, vid_len: int, n_classes: int):
    """
    convert time-as-class mask to video mask
    """
    n_rows, n_cols = mask.shape

    vid_mask = np.zeros(
        (vid_len, n_rows, n_cols),
        dtype=np.int64 if n_classes > 256 else np.uint8)

    for vid_id in range(vid_len):
        vid_mask[vid_id, ...] = mask % n_classes
        mask = mask // n_classes

    return vid_mask


def vid_mask_to_tac(
        video: np.ndarray,
        vid_mask: np.ndarray,
        n_classes: int,
        class_id_to_col,
        check):
    """
    convert video tac_mask to time-as-class tac_mask
    """
    vid_len, n_rows, n_cols = vid_mask.shape

    # n_pix = int(n_rows * n_cols)
    n_tac_classes = int(n_classes ** vid_len)

    # vid_mask_flat = vid_mask.reshape(vid_len, n_pix)
    tac_mask = np.zeros((n_rows, n_cols),
                        dtype=np.int64 if n_tac_classes > 256 else np.uint8)

    for vid_id in range(vid_len):
        tac_mask += (n_classes ** vid_id) * vid_mask[vid_id, ...].astype(tac_mask.dtype)
        # tac_mask += (n_classes ** vid_id) * vid_mask[vid_id, ...]

    # vid_mask_flat = vid_mask.flatten(order='F')
    # nbits_arr = np.split(vid_mask_flat, n_pix)
    # decimal_arr = np.asarray([nbits_to_decimal(nbits, n_classes) for nbits in nbits_arr])
    # decimal_arr = np.reshape(decimal_arr, (n_rows, n_cols), order='F')

    assert np.all(tac_mask < n_tac_classes), f"tac_mask must be < {n_tac_classes}"

    if check:
        vid_mask_rec = vid_mask_from_tac(tac_mask, vid_len, n_classes)
        if not np.array_equal(vid_mask, vid_mask_rec):
            print("vid_mask_rec mismatch")
            check_individual_vid_masks(video, vid_mask, vid_mask_rec, class_id_to_col, n_classes)

    return tac_mask


def mask_to_rle(mask, max_length, n_classes, order):
    """
    https://www.kaggle.com/stainsby/fast-tested-rle
    https://ccshenyltw.medium.com/run-length-encode-and-decode-a33383142e6b
    """
    all_starts = []
    all_lengths = []

    assert np.all(mask <= n_classes), f"mask pixels must be <= {n_classes}"

    for class_id in range(1, n_classes):
        mask_binary = (mask == class_id).astype(np.uint8)
        mask_flat = mask_binary.flatten(order=order)
        pixels = np.concatenate([[0], mask_flat, [0]])
        """the +1 in the original code was to convert indices from 0-based to 1-based"""
        runs = np.nonzero(pixels[1:] != pixels[:-1])[0]

        assert len(runs) % 2 == 0, "runs must have even length"

        if len(runs) == 0:
            starts = []
            lengths = []
        else:
            """assumes alternating 0s snd non-zeros so doesn't work with 
            non-binary masks"""
            runs[1::2] -= runs[::2]
            starts, lengths = runs[::2], runs[1::2]

            if max_length > 0:
                overlong_runs = np.nonzero(lengths > max_length)[0]
                if len(overlong_runs) > 0:
                    starts, lengths = split_runs(overlong_runs, starts, lengths, max_length)

            n_pix = mask.size
            assert np.all(starts <= n_pix - 1), f"starts cannot be > {n_pix - 1}"

            assert np.all(lengths <= max_length), f"run length cannot be > {max_length}"
            assert np.all(lengths > 0), "run length cannot be 0"

        all_starts.append(starts)
        all_lengths.append(lengths)

    all_starts = np.concatenate(all_starts, axis=0)
    all_lengths = np.concatenate(all_lengths, axis=0)
    sort_idx = np.argsort(all_starts)
    starts = all_starts[sort_idx].astype(np.int64)
    lengths = all_lengths[sort_idx].astype(np.int64)

    return starts, lengths


def mask_to_rle_tokens_tf(image, masks, config, class_id_to_col, training):
    max_length = config.dataset.train.max_length
    subsample = config.dataset.train.subsample
    batch_size = config.train.batch_size if training else config.eval.batch_size

    starts_2d = config.dataset.starts_2d
    length_as_class = config.dataset.length_as_class
    flat_order = config.dataset.flat_order

    starts_offset = config.model.coord_vocab_shift
    lengths_offset = config.model.len_vocab_shift
    class_offset = config.model.class_vocab_shift

    max_seq_len = config.model.max_seq_len
    vocab_size = config.model.vocab_size

    max_length_sub = int(max_length / subsample)

    n_cols, n_rows = image.shape[1], image.shape[2]
    n_rows_sub, n_cols_sub = int(n_rows / subsample), int(n_cols / subsample)

    n_classes = len(class_id_to_col)
    scale = 1. / subsample
    input_size = tf.cast(tf.shape(image)[1:3], tf.float32)
    scaled_size = tf.cast(tf.multiply(input_size, scale), tf.int32)

    masks_sub = tf.image.resize(masks, tf.cast(scaled_size, tf.int32),
                                method="nearest", antialias=False)

    rle_tokens_batch = []

    for batch_id in range(batch_size):

        mask_sub = tf.squeeze(masks_sub[batch_id, ...])

        starts, lengths = mask_to_rle_tf(mask_sub, max_length_sub, n_classes, flat_order)

        if n_classes > 2:
            mask_flat = tf.experimental.numpy.flatten(mask_sub, order=flat_order)
            class_ids = tf.cast(tf.gather(mask_flat, starts), tf.int64)
            if length_as_class:
                # lac = max_length * (class_ids - 1) + lengths
                lac = tf.experimental.numpy.asarray(
                    [max_length_sub * (class_id - 1) + length
                     for class_id, length in zip(class_ids, lengths)], dtype=np.int64)

                rle_cmp = [starts, lac]
            else:
                rle_cmp = [starts, lengths, class_ids]
        else:
            rle_cmp = [starts, lengths]

        rle_tokens = rle_to_tokens_tf(
            rle_cmp,
            mask_sub.shape,
            length_as_class,
            starts_offset,
            lengths_offset,
            class_offset,
            starts_2d,
            flat_order,
        )
        rle_tokens_truncated = rle_tokens[:max_seq_len]
        rle_tokens_padded = utils.pad_to_max_len(rle_tokens_truncated, max_seq_len, 0,
                                                 padding_token=vocab.PADDING_TOKEN)
        rle_tokens_batch.append(rle_tokens_padded)

    rle_tokens_batch = tf.experimental.numpy.stack(rle_tokens_batch, axis=0)

    return rle_tokens_batch


def mask_to_rle_tf(mask, max_length, n_classes, order):
    """
    https://www.kaggle.com/stainsby/fast-tested-rle
    https://ccshenyltw.medium.com/run-length-encode-and-decode-a33383142e6b
    """
    all_starts = []
    all_lengths = []
    for class_id in range(1, n_classes):
        mask_binary = tf.equal(mask, tf.cast(class_id, tf.uint8))
        mask_uint8 = tf.cast(mask_binary, tf.uint8)
        mask_flat = tf.experimental.numpy.flatten(mask_uint8, order=order)
        # dim = tf.reduce_prod(tf.shape(mask_binary)[1:])
        # mask_flat = tf.reshape(mask_binary, [-1, dim])

        pixels = tf.experimental.numpy.concatenate([tf.constant([0]), mask_flat, tf.constant([0])])
        runs = tf.experimental.numpy.nonzero(pixels[1:] != pixels[:-1])[0]

        if len(runs) == 0:
            starts = []
            lengths = []
        else:
            """assumes alternating 0s snd non-zeros so doesn't work with 
            non-binary masks"""
            # runs[1::2] -= runs[::2]
            starts, lengths = runs[::2], runs[1::2]
            lengths = lengths - starts

            if max_length > 0:
                overlong_runs = tf.experimental.numpy.nonzero(lengths > max_length)[0]
                if len(overlong_runs) > 0:
                    starts, lengths = split_runs_tf(overlong_runs, starts, lengths, max_length)

            # n_pix = mask.size
            # assert np.all(starts <= n_pix - 1), f"starts cannot be > {n_pix - 1}"
            # assert np.all(lengths <= max_length), f"run length cannot be > {max_length}"
            # assert np.all(lengths > 0), "run length cannot be 0"

        all_starts.append(starts)
        all_lengths.append(lengths)

    all_starts_tf = tf.experimental.numpy.concatenate(all_starts, axis=0)
    all_lengths_tf = tf.experimental.numpy.concatenate(all_lengths, axis=0)

    sort_idx = tf.experimental.numpy.argsort(all_starts_tf)
    starts = tf.gather(all_starts_tf, sort_idx)
    lengths = tf.gather(all_lengths_tf, sort_idx)

    return starts, lengths


def rle_to_mask(starts, lengths, class_ids, shape):
    if len(starts) == 0:
        mask = np.zeros(tuple(shape), dtype=np.uint8)
        return mask

    """ends are exclusive while starts are inclusive"""
    ends = starts + lengths
    mask_flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi, label in zip(starts, ends, class_ids):
        mask_flat[lo:hi] = label

    mask = mask_flat.reshape(shape)

    return mask


def rle_to_vid_mask(
        starts, lengths, class_ids, shape,
        vid_len, time_as_class, n_classes,
):
    n_rows, n_cols = shape
    tac_mask_rec = None

    if len(starts) == 0:
        mask = np.zeros((vid_len, n_rows, n_cols), dtype=np.uint8)
        if time_as_class:
            tac_mask_rec = np.zeros((n_rows, n_cols), dtype=np.uint8)
        return mask, tac_mask_rec

    ends = starts + lengths
    if time_as_class:
        mask_flat = np.zeros(n_rows * n_cols, dtype=np.int64)
    else:
        mask_flat = np.zeros(n_rows * n_cols * vid_len, dtype=np.uint8)

    for lo, hi, label in zip(starts, ends, class_ids):
        mask_flat[lo:hi] = label

    if time_as_class:
        tac_mask_rec = mask_flat.reshape((n_rows, n_cols))
        mask = vid_mask_from_tac(tac_mask_rec, vid_len, n_classes)
    else:
        mask = mask_flat.reshape((vid_len, n_rows, n_cols))
    return mask, tac_mask_rec


def get_class_info(category_info):
    class_id_to_name = {i: x['name'] for i, x in category_info.items()}
    class_id_to_col = {i: x['col'] for i, x in category_info.items()}

    return class_id_to_col, class_id_to_name


def read_class_info(class_names_path):
    class_info = [k.strip() for k in open(class_names_path, 'r').readlines() if k.strip()]
    class_names, class_cols = zip(*[[m.strip() for m in k.split('\t')] for k in class_info])

    if 'background' not in class_names:
        assert 'black' not in class_cols, "black should only be used for background"
        class_names = ('background',) + class_names
        class_cols = ('black',) + class_cols

    class_id_to_col = {i: x for (i, x) in enumerate(class_cols)}
    class_id_to_name = {i: x for (i, x) in enumerate(class_names)}

    # n_classes = len(class_cols)
    # palette = []
    # for class_id in range(n_classes):
    #     col = class_cols[class_id]
    #     if isinstance(col, str):
    #         col = col_bgr[col]
    #     col_rgb = col[::-1]
    #
    #     palette.append(col_rgb)
    #
    # palette_flat = [value for color in palette for value in color]
    #
    # class_name_to_id = {x: i for (i, x) in enumerate(class_names)}

    return class_id_to_col, class_id_to_name


def load_json(json_path):
    # print(f'loading json: {json_path}')
    if json_path.endswith('.json.gz'):
        import compress_json
        json_dict = compress_json.load(json_path)
    elif json_path.endswith('.json'):
        import json
        with open(json_path, 'r') as fid:
            json_dict = json.load(fid)
    else:
        raise AssertionError(f'invalid json_path: {json_path}')
    return json_dict


def get_category_names(json_path):
    if isinstance(json_path, str):
        annotations = load_json(json_path)
    else:
        annotations = json_path

    category_info = {c['id']: c for c in annotations['categories']}

    # assert 0 not in category_info.keys(), "class IDs must to be > 0"

    try:
        bkg_class = category_info[0]['name']
    except KeyError:
        category_info[0] = dict(
            name='background',
            col='black',
            supercategory='none',
            id=0,
        )
    else:
        assert bkg_class == 'background', "class id 0 must be used only for background"

    return category_info


def build_instance_prompt_seq(task_vocab_id: int, bbox, label,
                              quantization_bins, coord_vocab_shift):
    """"Build prompt seq for instance tasks like instance segmentation, keypoints.

    Args:
      task_vocab_id: Vocab id for the task.
      bbox: `float` bounding box of shape (bsz, n, 4).
      label: `int` label of shape (bsz, n).
      quantization_bins: `int`.
      coord_vocab_shift: `int`, shifting coordinates by a specified integer.

    Returns:
      discrete prompt sequence of (task_id, bbox, label) with shape (bsz, n, 6).
      tokens are zero'ed if label is padding (0).
    """
    task_id = tf.constant(task_vocab_id)
    quantized_bbox = utils.quantize(bbox, quantization_bins)
    quantized_bbox = quantized_bbox + coord_vocab_shift
    new_label = tf.expand_dims(label + vocab.BASE_VOCAB_SHIFT, -1)
    prompt_seq = tf.concat([quantized_bbox, new_label], axis=-1)
    task_id = tf.zeros_like(prompt_seq[..., :1]) + tf.cast(task_id, label.dtype)
    prompt_seq = tf.concat([task_id, prompt_seq], -1)
    is_padding = tf.expand_dims(tf.equal(label, 0), -1)
    prompt_seq = tf.where(is_padding, tf.zeros_like(prompt_seq), prompt_seq)
    return prompt_seq


def build_instance_response_seq_from_points(points, label, quantization_bins,
                                            coord_vocab_shift):
    """"Build target seq for instance tasks like instance segmentation, keypoints.

    Args:
      points: `float` points of shape (bsz, n, k).
      label: `int` label of shape (bsz, n).
      quantization_bins: `int`.
      coord_vocab_shift: `int`, shifting coordinates by a specified integer.

    Returns:
      discrete target sequence with shape (bsz, n, k). tokens are zero'ed
      if label is padding (0).
    """
    quantized_points = utils.quantize(points, quantization_bins)
    quantized_points = quantized_points + coord_vocab_shift
    response_seq = utils.replace_reserved_tokens(
        quantized_points, points, vocab.FLOAT_TO_TOKEN)
    is_padding = tf.expand_dims(tf.equal(label, 0), -1)
    response_seq = tf.where(is_padding, tf.zeros_like(response_seq), response_seq)
    return response_seq


def build_prompt_seq_from_task_id(task_vocab_id: int,
                                  response_seq=None,
                                  prompt_shape=None):
    """"Build prompt seq just using task id.

    Args:
      task_vocab_id: Vocab id for the task.
      response_seq: an (optional) discrete target sequence with shape (bsz, ..., k).
      prompt_shape: an (optional) tuple for prompt shape. One and only one of
        `response_seq` and `prompt_shape` should be specified.

    Returns:
      discrete input sequence of task id with shape (bsz, ..., 1).
    """
    task_id = tf.constant(task_vocab_id)
    prompt_seq = None
    if response_seq is not None:
        prompt_seq = tf.zeros_like(response_seq[..., :1]) + tf.cast(
            task_id, response_seq.dtype)
    if prompt_shape is not None:
        assert response_seq is None, 'double specification'
        prompt_seq = tf.zeros(prompt_shape, dtype=tf.int64) + tf.cast(task_id, dtype=tf.int64)

    assert prompt_seq is not None, "either response_seq or prompt_shape must be provided"

    return prompt_seq


def decode_instance_seq_to_points(seq, quantization_bins, coord_vocab_shift):
    """Decode points for seq from `build_instance_response_seq_from_points`."""
    assert seq.dtype in (tf.int64, tf.int32)
    points = seq - coord_vocab_shift
    points = utils.dequantize(points, quantization_bins)
    return utils.replace_reserved_tokens(points, seq, vocab.TOKEN_TO_FLOAT)


def decode_video_seq_to_bbox(
        logits,
        seq,
        vid_len,
        quantization_bins,
        coords_1d,
        coord_vocab_shift,
        seq_mask=None,
):
    _, seqlen, vocab_size = logits.shape

    if seq_mask is not None:
        seq = tf.where(seq_mask, seq, tf.cast(-1, seq.dtype))

    n_bbox_tokens = 2 if coords_1d else 4
    bbox_seq_len = vid_len * n_bbox_tokens + 1

    # truncate out the last few tokens
    if seqlen % bbox_seq_len != 0:
        truncate_len = seqlen % bbox_seq_len
        seq = seq[..., :-truncate_len]
        logits = logits[..., :-truncate_len, :]
        if seq_mask is not None:
            seq_mask = seq_mask[..., :-truncate_len]

    """
    extract probs for all classes - starting from 5th element and extracting every fifth element from there
    """
    probs = tf.nn.softmax(logits)
    class_probs = probs[:, bbox_seq_len - 1::bbox_seq_len]  # (bsz, instances, vocab_size)
    """
    mask-out non-class portions of the vocab
    """
    mask_s1 = [0.] * vocab.BASE_VOCAB_SHIFT  # reserved.
    mask_s2 = [1.] * (coord_vocab_shift - vocab.BASE_VOCAB_SHIFT)  # labels.
    mask_s3 = [0] * (vocab_size - coord_vocab_shift)  # coordinates and others.
    mask = tf.constant(mask_s1 + mask_s2 + mask_s3)
    """
    this is where the claims of automatically learning domain-specific tokens breaks down - we simply select the 
    class with the max prob even if some non-class token has higher prob than the max-prob class
    """
    class_tokens = tf.argmax(class_probs * mask[tf.newaxis, tf.newaxis, :], -1)
    """
    round-about way of selecting the class prob of each bbox as its score
    """
    # scores = tf.reduce_sum(class_probs * tf.one_hot(class_tokens, vocab_size), -1)
    scores = tf.gather_nd(class_probs, class_tokens[:, :, tf.newaxis], batch_dims=2)

    class_ids = tf.maximum(class_tokens - vocab.BASE_VOCAB_SHIFT, 0)
    bboxes, _ = seq_to_video_bbox(seq, quantization_bins, coords_1d, vid_len, coord_vocab_shift)
    return class_ids, bboxes, scores


def seq_to_video_bbox(seq, quantization_bins, coords_1d, vid_len, coord_vocab_shift):
    """Returns [0, 1] normalized yxyx bbox from token sequence."""
    # [batch, 5*num_instances]
    assert seq.shape.rank == 2, f'seq has non-rank 2 shape: {seq.shape.as_list()}'

    n_bbox_tokens = 2 if coords_1d else 4
    bbox_seq_len = vid_len * n_bbox_tokens + 1

    # [batch, num_instances, 1]

    boxes = []
    boxes_quant = []

    shape = tf.constant([quantization_bins, quantization_bins], dtype=tf.int64)
    max_coord = int(quantization_bins * quantization_bins) if coords_1d else quantization_bins

    for _id in range(vid_len):
        bbox_start_id = n_bbox_tokens * _id
        if coords_1d:
            pt1 = tf.expand_dims(seq[:, bbox_start_id::bbox_seq_len], -1)
            pt2 = tf.expand_dims(seq[:, bbox_start_id + 1::bbox_seq_len], -1)
            box_tokens = tf.concat([pt1, pt2], axis=-1)

        else:
            ymin = tf.expand_dims(seq[:, bbox_start_id::bbox_seq_len], -1)
            xmin = tf.expand_dims(seq[:, bbox_start_id + 1::bbox_seq_len], -1)
            ymax = tf.expand_dims(seq[:, bbox_start_id + 2::bbox_seq_len], -1)
            xmax = tf.expand_dims(seq[:, bbox_start_id + 3::bbox_seq_len], -1)
            box_tokens = tf.concat([ymin, xmin, ymax, xmax], axis=-1)

        box_quant = box_tokens - coord_vocab_shift

        is_no_box = tf.equal(box_tokens, vocab.NO_BOX_TOKEN)
        is_padding = tf.equal(box_tokens, vocab.PADDING_TOKEN)

        if coords_1d:
            """remove invalid tokens that cannot be unraveled"""
            is_not_coord = tf.math.logical_or(
                tf.less(box_quant, 0),
                tf.greater_equal(box_quant, max_coord))
            is_invalid = tf.math.logical_or(is_no_box, is_padding)
            is_invalid = tf.math.logical_or(is_invalid, is_not_coord)
            box_quant = tf.where(
                is_invalid,
                tf.cast(0, box_quant.dtype),
                box_quant)

            box_quant_flat = tf.reshape(box_quant, (-1,))
            box_quant_unravel = tf.unravel_index(box_quant_flat, shape)

            y_rec = tf.reshape(box_quant_unravel[0, :], tf.shape(box_quant))
            x_rec = tf.reshape(box_quant_unravel[1, :], tf.shape(box_quant))

            ymin, ymax = tf.expand_dims(y_rec[:, :, 0], -1), tf.expand_dims(y_rec[:, :, 1], -1)
            xmin, xmax = tf.expand_dims(x_rec[:, :, 0], -1), tf.expand_dims(x_rec[:, :, 1], -1)
            box_quant_rec = tf.concat([ymin, xmin, ymax, xmax], axis=-1)

            box_quant = box_quant_rec

            """if one pt in a bbox is no-box or padding, all pts in that box must be too"""
            is_no_box = tf.concat([is_no_box, is_no_box], axis=-1)
            is_padding = tf.concat([is_padding, is_padding], axis=-1)

        is_not_coord = tf.math.logical_or(
            tf.less(box_quant, 0),
            tf.greater_equal(box_quant, max_coord))

        box_dequant = utils.dequantize(box_quant, quantization_bins)

        box_clipped = tf.minimum(tf.maximum(box_dequant, 0), 1)

        is_invalid = tf.math.logical_or(is_no_box, is_not_coord)
        is_invalid = tf.math.logical_or(is_invalid, is_padding)

        box_clipped = tf.where(
            is_invalid,
            tf.cast(vocab.NO_BOX_FLOAT, box_clipped.dtype),
            box_clipped)

        box_quant = tf.where(
            is_invalid,
            tf.cast(0, box_quant.dtype),
            box_quant)

        boxes.append(box_clipped)
        boxes_quant.append(box_quant)

    boxes = tf.concat(boxes, axis=-1)
    boxes_quant = tf.concat(boxes_quant, axis=-1)

    return boxes, boxes_quant


def decode_object_seq_to_bbox(logits,
                              pred_seq,
                              quantization_bins,
                              coord_vocab_shift):
    """Decode objects (label & bbox) for seq from `build_response_seq_from_bbox`.

    Assume yxyxc format with truncation at the end for any uneven extra tokens.
      Replace class tokens with argmax instead of sampling.

    Args:
      logits: `float` output logits in shape of (bsz, max_seq_len, vocab_size).
      pred_seq: `int` pred sequence in shape of (bsz, max_seq_len).
      quantization_bins: `int` for bins.
      coord_vocab_shift: `int`, shifting coordinates by a specified integer.

    Returns:
      pred_class: `int` of shape (bsz, max_instances_per_image).
      pred_bbox: `float` of shape (bsz, max_instances_per_image, 4).
      pred_score: `float` of shape (bsz, max_instances_per_image).
    """
    _, seqlen, vocab_size = logits.shape
    if seqlen % 5 != 0:  # truncate out the last few tokens.
        pred_seq = pred_seq[..., :-(seqlen % 5)]
        logits = logits[..., :-(seqlen % 5), :]
    """
    extract probs for all classes - starting from 5th element and extracting every fifth element from there
    """
    pred_class_p = tf.nn.softmax(logits)[:, 4::5]  # (bsz, instances, vocab_size)

    """
    mask-out non-class portions of the vocab
    """
    mask_s1 = [0.] * vocab.BASE_VOCAB_SHIFT  # reserved.
    mask_s2 = [1.] * (coord_vocab_shift - vocab.BASE_VOCAB_SHIFT)  # labels.
    mask_s3 = [0] * (vocab_size - coord_vocab_shift)  # coordinates and others.
    mask = tf.constant(mask_s1 + mask_s2 + mask_s3)
    """
    this is where the claims of automatically learning domain-specific tokens breaks down - we simply select the 
    class with the max prob even if some non-class token has higher prob than the max-prob class
    """
    pred_class = tf.argmax(pred_class_p * mask[tf.newaxis, tf.newaxis, :], -1)

    """
    round-about way of selecting the class prob of each bbox as its score
    """
    pred_score = tf.reduce_sum(
        pred_class_p * tf.one_hot(pred_class, vocab_size), -1)
    pred_class = tf.maximum(pred_class - vocab.BASE_VOCAB_SHIFT, 0)
    pred_bbox = seq_to_bbox(pred_seq - coord_vocab_shift, quantization_bins)
    return pred_class, pred_bbox, pred_score


def seq_to_bbox(seq, quantization_bins, seq_format='yxyx_name'):
    """Returns [0, 1] normalized yxyx bbox from token sequence."""
    # [batch, 5*num_instances]
    assert seq.shape.rank == 2, f'seq has non-rank 2 shape: {seq.shape.as_list()}'
    # [batch, num_instances, 1]
    if seq_format.startswith('name'):
        ymin = tf.expand_dims(seq[:, 1::5], -1)
        xmin = tf.expand_dims(seq[:, 2::5], -1)
        ymax = tf.expand_dims(seq[:, 3::5], -1)
        xmax = tf.expand_dims(seq[:, 4::5], -1)
    else:
        ymin = tf.expand_dims(seq[:, 0::5], -1)
        xmin = tf.expand_dims(seq[:, 1::5], -1)
        ymax = tf.expand_dims(seq[:, 2::5], -1)
        xmax = tf.expand_dims(seq[:, 3::5], -1)
    if seq_format in ['name_cycxhw', 'cycxhw_name']:
        ycnt, xcnt, ysize, xsize = ymin, xmin, ymax, xmax
        ymin = ycnt - ysize // 2
        xmin = xcnt - xsize // 2
        ymax = ycnt + ysize // 2
        xmax = xcnt + xsize // 2
    quantized_box = tf.concat([ymin, xmin, ymax, xmax], axis=-1)
    quantized_box = utils.dequantize(quantized_box, quantization_bins)
    return tf.minimum(tf.maximum(quantized_box, 0), 1)


def compute_weighted_scores(bbox_scores, pred_seq, logits,
                            points_score_weight):
    """Computes per instance score as weighted sum of box score and mean pred_seq score."""
    probs = tf.nn.softmax(logits, axis=-1)
    # Set 0 weight for padding tokens.
    token_weight = tf.where(tf.equal(pred_seq, vocab.PADDING_TOKEN), 0.0, 1.0)
    likelihoods = tf.gather(probs, pred_seq, batch_dims=pred_seq.shape.rank)
    points_score = (
            tf.reduce_sum(likelihoods * token_weight, axis=-1) /
            tf.reduce_sum(token_weight, axis=-1))
    num_instances_in_batch = bbox_scores.shape[0]
    num_samples = points_score.shape[0] // num_instances_in_batch
    points_score = tf.reshape(points_score, [num_instances_in_batch, num_samples])
    points_score = tf.reduce_mean(points_score, axis=-1)
    return (points_score_weight * points_score +
            (1 - points_score_weight) * bbox_scores)


def join_if_not_none(args, sep):
    args = [str(arg) for arg in args if arg is not None]
    return sep.join(args)


def integer_map_to_bits(integer_map, n_bits_label, b_scale, num_channels=2):
    """Converts an integer map to analog bits.

    Args:
      integer_map: integer tensor of shape [..., num_channels].
      n_bits_label: integer. Total number of bits of the analog bits.
      b_scale: float. Scaling of the analog bits.
      num_channels: integer. Number of channels in the integer map.

    Returns:
      Analog bits of shape [..., n_bits_label].
    """
    bits = []
    for i in range(num_channels):
        bits.append(utils.int2bits(
            integer_map[..., i], n_bits_label // num_channels, tf.float32))
    bits = tf.concat(bits, -1)
    bits = (bits * 2 - 1) * b_scale
    return bits


def bits_to_panoptic_map(bits, n_bits_label, num_classes,
                         max_instances_per_image):
    """Converts analog bits to a panoptic map.

    Args:
      bits: float tensor of shape [..., n_bits_label].
      n_bits_label: integer. Number of bits of the analog bits.
      num_classes: integer. Number of semantic classes.
      max_instances_per_image: integer. Maximum number of instances in an image.

    Returns:
      The integer panoptic map of [..., 2], where the first channel is the
      semantic map and the second channel is the instance map.
    """
    s_map = utils.bits2int(bits[..., :n_bits_label // 2] > 0, tf.int32)
    s_map = tf.minimum(s_map, num_classes - 1)
    i_map = utils.bits2int(bits[..., n_bits_label // 2:] > 0, tf.int32)
    i_map = tf.minimum(i_map, max_instances_per_image - 1)
    panoptic_map = tf.stack([s_map, i_map], -1)
    return panoptic_map


def get_normalized_weight(id_map, total_num_ids, p=1.0):
    """Returns instance normalized weight given id_map (bsz, h, w)."""
    id_map_hot = tf.one_hot(id_map, total_num_ids)
    weight = 1. / (tf.reduce_sum(id_map_hot, [1, 2]) + 1)
    weight = tf.einsum('bhwk,bk->bhw', id_map_hot, weight)
    weight = tf.pow(weight, p)
    weight /= tf.reduce_sum(weight, [1, 2], keepdims=True)
    weight *= tf.cast(tf.math.reduce_prod(tf.shape(id_map)[1:]), weight.dtype)
    return weight


def polygons_from_mask(mask_img):
    # print('Getting contour pts from mask...')
    if len(mask_img.shape) == 3:
        mask_img_gs = np.squeeze(mask_img[:, :, 0]).copy()
    else:
        mask_img_gs = mask_img.copy()

    ret = cv2.findContours(mask_img_gs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_pts, _ = ret

    contour_pts = [np.squeeze(contour_pts_) for contour_pts_ in contour_pts]

    return contour_pts

    if not _contour_pts:
        return [], []
    contour_pts = list(np.squeeze(_contour_pts[0]))

    n_contours = len(_contour_pts)
    # print('n_contours: {}'.format(n_contours))
    # print('_contour_pts: {}'.format(_contour_pts))
    # print('contour_pts: {}'.format(type(contour_pts)))

    if n_contours > 1:
        max_len = len(contour_pts)
        for _pts in _contour_pts[1:]:
            # print('_pts: {}'.format(_pts))
            _pts = np.squeeze(_pts)
            _len = len(_pts)
            if max_len < _len:
                contour_pts = _pts
                max_len = _len

    # print('contour_pts len: {}'.format(len(contour_pts)))
    mask_pts = [[x, y, 1] for x, y in contour_pts]

    return contour_pts, mask_pts
