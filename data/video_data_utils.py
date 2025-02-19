from typing import Any, List, Optional, Tuple

import vocab
import tensorflow as tf

from data.data_utils import (unflatten_points, handle_out_of_frame_points, flatten_points,
                             truncation_bbox)



def flip_video_left_right(video, length):
    resized_images = [None, ] * length
    for frame_id in range(length):
        frame = video[frame_id, ...]
        resized_images[frame_id] = tf.image.flip_left_right(frame)
    resized_video = tf.stack(resized_images, axis=0)
    return resized_video


def pad_video_to_bounding_box(video, length, **kwargs):
    resized_images = [None, ] * length
    for frame_id in range(length):
        frame = video[frame_id, ...]
        resized_images[frame_id] = tf.image.pad_to_bounding_box(frame, **kwargs)
    resized_video = tf.stack(resized_images, axis=0)
    return resized_video


def resize_video(video, length, **kwargs):
    resized_images = [None, ] * length
    for frame_id in range(length):
        frame = video[frame_id, ...]
        resized_images[frame_id] = tf.image.resize(frame, **kwargs)
    resized_video = tf.stack(resized_images, axis=0)
    return resized_video


def flip_video_boxes_left_right(boxes, length):
    out_bboxes = []
    for i in range(length):
        box_idx = i * 4
        boxes_ = boxes[..., box_idx:box_idx + 4]
        ymin, xmin, ymax, xmax = tf.split(value=boxes_, num_or_size_splits=4, axis=-1)
        out_bboxes += [ymin, 1. - xmax, ymax, 1. - xmin]
    out_bboxes = tf.concat(out_bboxes, -1)
    return out_bboxes


def shift_bbox_video(bbox, length, truncation=True):
    """Shifting bbox without changing the bbox height and width."""
    n_bboxes = tf.shape(bbox)[0]
    # bbox = tf.rehape(bbox, [n_bboxes, 4, length])

    # bbox_mask = tf.math.is_nan(bbox)
    # randomly sample new bbox centers.
    shifted_bbox = []

    for i in range(length):
        idx = 4 * i
        ymin = tf.reshape(bbox[:, idx], [n_bboxes, 1])
        xmin = tf.reshape(bbox[:, idx + 1], [n_bboxes, 1])
        ymax = tf.reshape(bbox[:, idx + 2], [n_bboxes, 1])
        xmax = tf.reshape(bbox[:, idx + 3], [n_bboxes, 1])

        cy = tf.random.uniform([n_bboxes, 1], 0, 1)
        cx = tf.random.uniform([n_bboxes, 1], 0, 1)

        h = ymax - ymin
        w = xmax - xmin

        ymin_s, xmin_s, ymax_s, xmax_s = (
            cy - tf.abs(h) / 2,
            cx - tf.abs(w) / 2,
            cy + tf.abs(h) / 2,
            cx + tf.abs(w) / 2
        )

        shifted_bbox += [ymin_s, xmin_s, ymax_s, xmax_s]

    # print(f'shift_bbox_video: bbox: {tf.shape(bbox)}')
    # print(f'shift_bbox_video: ymin: {tf.shape(ymin)}')
    # print(f'shift_bbox_video: cy: {tf.shape(cy)}')
    # print(f'shift_bbox_video: h: {tf.shape(h)}')

    # ymin_d, xmin_d, ymax_d, xmax_d = (
    #     ymin_s - ymin,
    #     xmin_s - xmin,
    #     ymax_s - ymax,
    #     xmax_s - xmin
    # )

    # print(f'shift_bbox_video: ymin_s: {tf.shape(ymin_s)}')
    # print(f'shift_bbox_video: ymin_d: {tf.shape(ymin_d)}')

    # shifted_bbox = [ymin_s, xmin_s, ymax_s, xmax_s]

    shifted_bbox = tf.concat(shifted_bbox, -1)
    return truncation_bbox(shifted_bbox) if truncation else shifted_bbox


def random_bbox_video(n_bboxes, length, max_disp, max_size=1.0, truncation=True):
    """Generating random n bbox with max size specified within [0, 1]."""
    cy = tf.random.uniform([n_bboxes, 1], 0, 1)
    cx = tf.random.uniform([n_bboxes, 1], 0, 1)
    h = tf.random.truncated_normal([n_bboxes, 1], 0, max_size / 2.)
    w = tf.random.truncated_normal([n_bboxes, 1], 0, max_size / 2.)

    ymin_r, xmin_r, ymax_r, xmax_r = (
        cy - tf.abs(h) / 2,
        cx - tf.abs(w) / 2,
        cy + tf.abs(h) / 2,
        cx + tf.abs(w) / 2
    )
    # print(f'ymin_r: {tf.shape(ymin_r)}')

    rand_bbox = [ymin_r, xmin_r, ymax_r, xmax_r]
    for i in range(1, length):
        ymin_d = tf.random.truncated_normal([n_bboxes, 1], mean=0, stddev=max_disp)
        xmin_d = tf.random.truncated_normal([n_bboxes, 1], mean=0, stddev=max_disp)
        ymax_d = tf.random.truncated_normal([n_bboxes, 1], mean=0, stddev=max_disp)
        xmax_d = tf.random.truncated_normal([n_bboxes, 1], mean=0, stddev=max_disp)

        ymin_r, xmin_r, ymax_r, xmax_r = (
            ymin_r + ymin_d,
            xmin_r + xmin_d,
            ymax_r + ymax_d,
            xmax_r + xmax_d
        )
        rand_bbox += [ymin_r, xmin_r, ymax_r, xmax_r]

    rand_bbox = tf.concat(rand_bbox, -1)

    return truncation_bbox(rand_bbox) if truncation else rand_bbox


def jitter_bbox_video(bbox, length, min_range=0., max_range=0.05, truncation=True):
    n = tf.shape(bbox)[0]
    noise = []
    for i in range(length):
        idx = 4 * i
        h = bbox[:, idx + 2] - bbox[:, idx]
        w = bbox[:, idx + 3] - bbox[:, idx + 1]

        noise += [h, w, h, w]

    noise = tf.stack(noise, -1)
    if min_range == 0:
        noise_rate = tf.random.truncated_normal(
            [n, 4 * length], mean=0, stddev=max_range / 2., dtype=bbox.dtype)
    else:
        noise_rate1 = tf.random.uniform([n, 4 * length], min_range, max_range)
        noise_rate2 = tf.random.uniform([n, 4 * length], -max_range, -min_range)
        selector = tf.cast(tf.random.uniform([n, 4], 0, 1) < 0.5, tf.float32)
        noise_rate = noise_rate1 * selector + noise_rate2 * (1. - selector)
    bbox = bbox + noise * noise_rate
    return truncation_bbox(bbox) if truncation else bbox


def augment_bbox_video(bbox, class_id, class_name, length, max_disp, max_jitter, n_noise_bboxes,
                       mix_rate=0.):
    n_bboxes = tf.shape(bbox)[0]

    # print(f'augment_bbox_video: bbox: {bbox}')
    # print(f'augment_bbox_video: box shape: {tf.shape(bbox)}')
    # print(f'augment_bbox_video: n_bboxes: {n_bboxes}')

    fake_class_id = vocab.FAKE_CLASS_TOKEN - vocab.BASE_VOCAB_SHIFT
    fake_class_name = "fake"

    n_dup_bboxes = tf.random.uniform(
        [], 0, n_noise_bboxes + 1, dtype=tf.int32)
    n_dup_bboxes = 0 if n_bboxes == 0 else n_dup_bboxes
    n_bad_bboxes = n_noise_bboxes - n_dup_bboxes
    multiplier = 1 if n_bboxes == 0 else tf.math.floordiv(n_noise_bboxes, n_bboxes) + 1
    bbox_tiled = tf.tile(bbox, [multiplier, 1])

    # print(f'n_noise_bboxes: {n_noise_bboxes}')
    # print(f'n_dup_bboxes: {n_dup_bboxes}')
    # print(f'n_bad_bboxes: {n_bad_bboxes}')

    # Create bad bbox.
    """Randomly shuffle along the first dimension"""
    bbox_tiled = tf.random.shuffle(bbox_tiled)
    # print(f'augment_bbox_video: bbox_tiled shape: {tf.shape(bbox_tiled)}')
    bbox_tiled_shift = bbox_tiled[:n_bad_bboxes]
    # print(f'augment_bbox_video: bbox_tiled_shift shape: {tf.shape(bbox_tiled_shift)}')

    # tf.debugging.Assert(n_bboxes > 0, [n_bboxes, bbox])

    bad_bbox_shift = shift_bbox_video(
        bbox_tiled_shift,
        length=length,
        truncation=True)

    bad_bbox_random = random_bbox_video(
        n_bad_bboxes,
        length=length,
        max_disp=max_disp,
        max_size=1.0,
        truncation=True)

    # print(f'bad_bbox_shift shape: {tf.shape(bad_bbox_shift)}')
    # print(f'bad_bbox_random shape: {tf.shape(bad_bbox_random)}')

    """generate twice as many noise bboxes as needed and select a random subset"""
    bad_bboxes = tf.concat([bad_bbox_shift, bad_bbox_random], 0)
    bad_bboxes = tf.random.shuffle(bad_bboxes)[:n_bad_bboxes]

    bad_class_id = tf.zeros([n_bad_bboxes], dtype=class_id.dtype) + (
        fake_class_id)

    bad_class_name = tf.fill([n_bad_bboxes], fake_class_name)

    # Create dup bbox.
    bbox_tiled = tf.random.shuffle(bbox_tiled)
    dup_bbox = jitter_bbox_video(
        bbox_tiled[:n_dup_bboxes],
        length=length,
        min_range=0,
        max_range=0.1,
        truncation=True)
    dup_class_id = tf.zeros([n_dup_bboxes], dtype=class_id.dtype) + (
        fake_class_id)
    dup_class_name = tf.fill([n_dup_bboxes], fake_class_name)

    # Jitter positive bbox.
    if max_jitter > 0:
        bbox = jitter_bbox_video(
            bbox,
            length=length,
            min_range=0,
            max_range=max_jitter,
            truncation=True)

    if tf.random.uniform([]) < mix_rate:
        # Mix the bbox with bad bbox, appneded by dup bbox.
        bbox_new = tf.concat([bbox, bad_bboxes], 0)
        class_id_new = tf.concat([class_id, bad_class_id], 0)
        class_name_new = tf.concat([class_name, bad_class_name], 0)

        idx = tf.random.shuffle(tf.range(tf.shape(bbox_new)[0]))

        bbox_new = tf.gather(bbox_new, idx)
        class_id_new = tf.gather(class_id_new, idx)
        class_name_new = tf.gather(class_name_new, idx)

        bbox_new = tf.concat([bbox_new, dup_bbox], 0)
        class_id_new = tf.concat([class_id_new, dup_class_id], 0)
        class_name_new = tf.concat([class_name_new, dup_class_name], 0)
    else:
        # Merge bad bbox and dup bbox into noise bbox.
        noise_bbox = tf.concat([bad_bboxes, dup_bbox], 0)
        noise_class_id = tf.concat([bad_class_id, dup_class_id], 0)
        noise_class_name = tf.concat([bad_class_name, dup_class_name], 0)

        if n_noise_bboxes > 0:
            # print(f'augment_bbox_video: n_noise_bboxes: {n_noise_bboxes}')
            #
            # print(f'augment_bbox_video: noise_bbox shape: {tf.shape(noise_bbox)}')
            #
            # print(f'augment_bbox_video: noise_class_id: {noise_class_id}')
            # print(f'augment_bbox_video: noise_class_id shape: {tf.shape(noise_class_id)}')
            #
            # print(f'augment_bbox_video: noise_class_name: {noise_class_name}')
            # print(f'augment_bbox_video: noise_class_name shape: {tf.shape(noise_class_name)}')

            idx = tf.random.shuffle(tf.range(n_noise_bboxes))
            noise_bbox = tf.gather(noise_bbox, idx)
            noise_class_id = tf.gather(noise_class_id, idx)
            noise_class_name = tf.gather(noise_class_name, idx)

        # Append noise bbox to bbox and create mask.
        bbox_new = tf.concat([bbox, noise_bbox], 0)
        class_id_new = tf.concat([class_id, noise_class_id], 0)
        class_name_new = tf.concat([class_name, noise_class_name], 0)

    return bbox_new, class_id_new, class_name_new


def video_crop(
        example: dict[str, tf.Tensor],
        region: Tuple[Any, Any, Any, Any],
        input_keys: List[str],
        object_coordinate_keys: Optional[List[str]]):
    """Crop video to region and adjust (normalized) bbox."""
    h_offset, w_offset, h, w = region
    input_ = input_keys[0]
    if input_ == 'video':
        _, h_ori, w_ori, _ = tf.unstack(tf.shape(example[input_]))
    elif input_ == 'image':
        h_ori, w_ori, _ = tf.unstack(tf.shape(example[input_]))
    else:
        raise AssertionError(f'Invalid input: {input_}')

    for k in input_keys:
        if k == 'video':
            example[k] = example[k][:, h_offset:h_offset + h, w_offset:w_offset + w, :]
        elif k == 'image':
            example[k] = example[k][h_offset:h_offset + h, w_offset:w_offset + w, :]
        else:
            raise AssertionError(f'Invalid input: {k}')

    # Record the crop offset.
    if 'crop_offset' not in example:
        # Cropping the first time.
        example['crop_offset'] = tf.cast(region[:2], tf.int32)
    else:
        # Subsequent crops e.g. when crop_to_bbox is True and we apply random_crop.
        old_h_offset, old_w_offset = tf.unstack(example['crop_offset'])
        example['crop_offset'] = tf.cast(
            [old_h_offset + h_offset, old_w_offset + w_offset], tf.int32)

    # Crop object coordinates.
    if object_coordinate_keys:
        h_offset = tf.cast(h_offset, tf.float32)
        w_offset = tf.cast(w_offset, tf.float32)
        h, w = tf.cast(h, tf.float32), tf.cast(w, tf.float32)
        h_ori, w_ori = tf.cast(h_ori, tf.float32), tf.cast(w_ori, tf.float32)

        scale = tf.stack([h_ori / h, w_ori / w])
        offset = tf.stack([h_offset / h_ori, w_offset / w_ori])
        for key in object_coordinate_keys:
            points = example[key]
            points = unflatten_points(points)
            points = (points - offset) * scale
            points = handle_out_of_frame_points(points, key)
            example[key] = flatten_points(points)

    return example

def crop_video(frames,
               height,
               width,
               seq_len=None,
               random=False,
               seed=None,
               state=None):
    if random:
        # Random spatial crop. tf.image.random_crop is not used since the offset is
        # needed to ensure consistency between crops on different modalities.
        shape = tf.shape(input=frames)
        static_shape = frames.shape.as_list()
        if seq_len is None:
            seq_len = shape[0] if static_shape[0] is None else static_shape[0]
        channels = shape[3] if static_shape[3] is None else static_shape[3]
        size = tf.convert_to_tensor(value=(seq_len, height, width, channels))

        if state and 'crop_offset_proportion' in state:
            # Use offset set by a previous cropping: [0, offset_h, offset_w, 0].
            offset = state['crop_offset_proportion'] * tf.cast(shape, tf.float32)
            offset = tf.cast(tf.math.round(offset), tf.int32)
        else:
            # Limit of possible offset in order to fit the entire crop:
            # [1, input_h - target_h + 1, input_w - target_w + 1, 1].
            limit = shape - size + 1
            offset = tf.random.uniform(
                shape=(4,),
                dtype=tf.int32,
                maxval=tf.int32.max,
                seed=seed) % limit  # [0, offset_h, offset_w, 0]

            if state is not None:
                # Update state.
                offset_proportion = tf.cast(offset, tf.float32) / tf.cast(
                    shape, tf.float32)
                state['crop_offset_proportion'] = offset_proportion

        frames = tf.slice(frames, offset, size)
    else:
        # Central crop or pad.
        shape = tf.shape(input=frames)
        static_shape = frames.shape.as_list()
        if seq_len is None:
            seq_len = shape[0] if static_shape[0] is None else static_shape[0]
        frames = tf.image.resize_with_crop_or_pad(frames[:seq_len], height, width)
    return frames
