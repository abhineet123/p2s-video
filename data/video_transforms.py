import copy

from data import video_data_utils, data_utils
from data.transforms import TransformRegistry, Transform

import tensorflow as tf


@TransformRegistry.register('record_original_video_size')
class RecordOriginalVideoSize(Transform):
    def process_example(self, example: dict[str, tf.Tensor]):
        # video_key = self.config.get('video_key', 'video')
        # orig_video_size_key = self.config.get('original_video_size_key',
        #                                       DEFAULT_ORIG_VIDEO_SIZE_KEY)
        # example[orig_video_size_key] = tf.shape(example[video_key])[1:3]
        return example


@TransformRegistry.register('fixed_size_crop_video')
class FixedSizeVideoCrop(Transform):
    def process_example(self, example: dict[str, tf.Tensor]):
        example_out = copy.copy(example)
        target_height, target_width = self.config.target_size

        # input_size_key = self.config.input_size_key
        # input_size = example_out[input_size_key]

        input_ = self.config.inputs[0]
        if input_=='video':
            input_size = tf.shape(example[input_])[1:3]
        elif input_=='image':
            input_size = tf.shape(example[input_])[:2]
        else:
            raise AssertionError(f'Invalid input: {input_}')

        output_size = tf.stack([target_height, target_width])

        max_offset = tf.subtract(input_size, output_size)
        max_offset = tf.cast(tf.maximum(max_offset, 0), tf.float32)
        offset = tf.multiply(max_offset, tf.random.uniform([], 0.0, 1.0))
        offset = tf.cast(offset, tf.int32)

        region = (offset[0], offset[1],
                  tf.minimum(output_size[0], input_size[0] - offset[0]),
                  tf.minimum(output_size[1], input_size[1] - offset[1]))
        object_coordinate_keys = self.config.get('object_coordinate_keys', [])

        example_out = video_data_utils.video_crop(example_out, region, self.config.inputs,
                                            object_coordinate_keys)

        # example_out['fixed_size_crop_video'] = output_size

        return example_out


@TransformRegistry.register('resize_video')
class ResizeVideo(Transform):
    def process_example(self, example: dict[str, tf.Tensor]):
        example = copy.copy(example)
        num_inputs = len(self.config.inputs)
        resize_methods = self.config.get('resize_method', ['bilinear'] * num_inputs)
        antialias_list = self.config.get('antialias', [False] * num_inputs)
        preserve_ar = self.config.get('preserve_aspect_ratio', [True] * num_inputs)

        # target_size = self.config.target_size
        # if target_size is not None:
        #     frames = tf.image.resize(
        #         frames, target_size, method='bilinear',
        #         antialias=False, preserve_aspect_ratio=False)

        for k, resize_method, antialias, p_ar in zip(
                self.config.inputs, resize_methods, antialias_list, preserve_ar):
            example[k] = tf.image.resize(
                example[k],
                size=self.config.target_size, method=resize_method,
                antialias=antialias, preserve_aspect_ratio=p_ar)
        return example


@TransformRegistry.register('scale_jitter_video')
class ScaleJitterVideo(Transform):
    def process_example(self, example: dict[str, tf.Tensor]):
        example_out = copy.copy(example)
        target_height, target_width = self.config.target_size

        # input_size_key = self.config.input_size_key
        # input_size = example_out[input_size_key]

        input_ = self.config.inputs[0]
        if input_=='video':
            input_size = tf.cast(tf.shape(example_out[input_])[1:3], tf.float32)
        elif input_=='image':
            input_size = tf.cast(tf.shape(example_out[input_])[:2], tf.float32)
        else:
            raise AssertionError(f'Invalid input: {input_}')

        min_scale, max_scale = self.config.min_scale, self.config.max_scale
        output_size = tf.constant([target_height, target_width], tf.float32)
        random_scale = tf.random.uniform([], min_scale, max_scale)
        random_scale_size = tf.multiply(output_size, random_scale)
        scale = tf.minimum(
            random_scale_size[0] / input_size[0],
            random_scale_size[1] / input_size[1]
        )
        scaled_size = tf.cast(tf.multiply(input_size, scale), tf.int32)

        num_inputs = len(self.config.inputs)
        resize_methods = self.config.get('resize_method', ['bilinear'] * num_inputs)
        antialias_list = self.config.get('antialias', [False] * num_inputs)
        for k, resize_method, antialias in zip(self.config.inputs,
                                               resize_methods, antialias_list):
            video = example_out[k]
            resized_video = tf.image.resize(
                video, scaled_size,
                method=resize_method, antialias=antialias)
            example_out[k] = resized_video
        # example_out['scale_jitter_video'] = scaled_size
        return example_out


@TransformRegistry.register('random_horizontal_flip_video')
class RandomHorizontalFlipVideo(Transform):
    def process_example(self, example: dict[str, tf.Tensor]):
        out_example = copy.copy(example)
        length = self.config.length

        inputs = {k: out_example[k] for k in self.config.inputs}
        boxes = {k: out_example[k] for k in self.config.get('bbox_keys', [])}
        with tf.name_scope('RandomHorizontalFlipVideo'):
            coin_flip = tf.random.uniform([]) > 0.5
            if coin_flip:
                inputs = {k: tf.image.flip_left_right(v) for k, v in inputs.items()}
                boxes = {k: video_data_utils.flip_video_boxes_left_right(v, length) for k, v in boxes.items()}
        out_example.update(inputs)
        out_example.update(boxes)
        return out_example


@TransformRegistry.register('filter_invalid_objects_video')
class FilterInvalidObjectsVideo(Transform):
    """Filter objects with invalid bboxes.

    Required fields in config:
      inputs: names of applicable fields in the example.
      bbox_key: optional name of the bbox field. Defaults to 'bbox'.
      filter_keys: optional. Names of fields that, if True, the object will be
        filtered out. E.g. 'is_crowd', the objects with 'is_crowd=True' will be
        filtered out.
    """

    def process_example(self, example: dict[str, tf.Tensor]):
        out_example = copy.copy(example)
        bbox_key = 'bbox'

        bboxes = out_example[bbox_key]
        n_bboxes = tf.shape(bboxes)[0]
        is_no_box = tf.math.is_nan(bboxes)
        length = self.config.length

        is_no_box_count = tf.reduce_sum(tf.cast(is_no_box, tf.int32), axis=1)
        max_no_box_count = 4*(length-1)
        tf.debugging.Assert(tf.reduce_all(is_no_box_count <= max_no_box_count), [bboxes, is_no_box_count])

        # print(f'filter_invalid_objects: box shape: {tf.shape(bbox)}')
        # print(f'filter_invalid_objects: bbox: {bbox}')
        # print(f'filter_invalid_objects: n_bboxes: {n_bboxes}')

        box_valid = None
        for i in range(length):
            bbox_idx = 4 * i
            box_valid_ = tf.logical_or(
                is_no_box[:, bbox_idx],
                tf.logical_and(bboxes[:, bbox_idx + 2] > bboxes[:, bbox_idx],
                               bboxes[:, bbox_idx + 3] > bboxes[:, bbox_idx + 1])
            )
            box_valid = box_valid_ if box_valid is None else tf.logical_and(box_valid, box_valid_)

        for k in self.config.get('filter_keys', []):
            box_valid = tf.logical_and(box_valid, tf.logical_not(out_example[k]))
        valid_indices = tf.where(box_valid)[:, 0]
        for k in self.config.inputs:
            out_example[k] = tf.gather(out_example[k], valid_indices)

        # out_bbox = out_example[bbox_key]
        # n_out_bboxes = tf.shape(out_bbox)[0]
        # tf.debugging.Assert(n_out_bboxes > 0, [bboxes, out_bbox])

        return out_example


@TransformRegistry.register('inject_noise_bbox_video')
class InjectNoiseBboxVideo(Transform):
    def process_example(self, example: dict[str, tf.Tensor]):
        example = copy.copy(example)
        bbox_key = 'bbox'
        class_id_key = 'class_id'
        class_name_key = 'class_name'

        num_instances = tf.shape(example[bbox_key])[0]
        if num_instances < self.config.max_instances_per_image:
            n_noise_bbox = self.config.max_instances_per_image - num_instances
            example[bbox_key], example[class_id_key], example[class_name_key] = video_data_utils.augment_bbox_video(
                bbox=example[bbox_key],
                class_id=example[class_id_key],
                class_name=example[class_name_key],
                length=self.config.length,
                max_disp=self.config.max_disp,
                max_jitter=0.,
                n_noise_bboxes=n_noise_bbox)
        return example


@TransformRegistry.register('pad_video_to_max_size')
class PadVideoToMaxSize(Transform):
    def process_example(self, example: dict[str, tf.Tensor]):
        example_out = copy.copy(example)
        num_inputs = len(self.config.inputs)
        backgrnd_val = self.config.get('background_val', [0.3] * num_inputs)
        target_size = self.config.target_size

        # input_size_key = self.config.input_size_key
        # unpadded_video_size =example_out[input_size_key]

        input_ = self.config.inputs[0]
        if input_=='video':
            unpadded_video_size = tf.cast(tf.shape(example_out[input_])[1:3], tf.float32)
        elif input_=='image':
            unpadded_video_size = tf.cast(tf.shape(example_out[input_])[:2], tf.float32)
        else:
            raise AssertionError(f'Invalid input: {input_}')

        for k, backgrnd_val_ in zip(self.config.inputs, backgrnd_val):
            unpadded_video = example_out[k]
            if k == 'video' or k == 'image':
                example_out['unpadded_video_size'] = unpadded_video_size
                height = unpadded_video_size[0]
                width = unpadded_video_size[1]

            example_out[k] = backgrnd_val_ + tf.image.pad_to_bounding_box(
                unpadded_video - backgrnd_val_,
                offset_height=0,
                offset_width=0,
                target_height=target_size[0],
                target_width=target_size[1]
            )

        # example_out['pad_video_to_max_size'] = target_size

        # Adjust the coordinate fields.
        object_coordinate_keys = self.config.get('object_coordinate_keys', [])
        if object_coordinate_keys:

            assert 'video' in self.config.inputs or 'image' in self.config.inputs, \
                "video or image must be in inputs for object_coordinate_keys to be processed"

            hratio = tf.cast(height, tf.float32) / tf.cast(target_size[0], tf.float32)
            wratio = tf.cast(width, tf.float32) / tf.cast(target_size[1], tf.float32)
            scale = tf.stack([hratio, wratio])
            for key in object_coordinate_keys:
                example_out[key] = data_utils.flatten_points(
                    data_utils.unflatten_points(example_out[key]) * scale)
        return example_out
