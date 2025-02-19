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
"""Common transforms configs."""
from typing import Tuple
from configs.config_base import D


def get_object_detection_train_transforms(
        cfg,
        image_size: Tuple[int, int],
        max_instances_per_image: int,
):
    instance_feature_names = ['bbox', 'label', 'area', 'is_crowd']
    object_coordinate_keys = ['bbox']

    train_transforms = [
        # D(name='scale_objs',
        #   input='image',
        #   object_coordinate_keys=object_coordinate_keys,),

        # D(name='record_original_image_size'),
    ]
    if cfg.scale_jitter:
        # print('annoying scale_jitter is enabled')
        train_transforms.append(
            D(name='scale_jitter',
              inputs=['image'],
              target_size=image_size,
              min_scale=cfg.jitter_scale_min,
              max_scale=cfg.jitter_scale_max)
        )
    if cfg.fixed_crop:
        # print('equally annoying fixed_crop is enabled')
        train_transforms.append(
            D(name='fixed_size_crop',
              inputs=['image'],
              target_size=image_size,
              object_coordinate_keys=object_coordinate_keys)
        )
    else:
        train_transforms.append(
            D(name='resize_image',
              inputs=['image'],
              antialias=[True],
              target_size=image_size)
        )

    train_transforms += [
        D(name='random_horizontal_flip',
          inputs=['image'],
          bbox_keys=['bbox']),
        # Remove objects with invalid boxes (e.g. produced by cropping) as well as
        # crowded objects.
        D(name='filter_invalid_objects',
          inputs=instance_feature_names,
          filter_keys=['is_crowd']),
        D(name='reorder_object_instances',
          inputs=instance_feature_names,
          order=cfg.object_order),
        D(name='inject_noise_bbox',
          max_instances_per_image=max_instances_per_image),
        D(name='pad_image_to_max_size',
          inputs=['image'],
          target_size=image_size,
          object_coordinate_keys=object_coordinate_keys),
        D(name='truncate_or_pad_to_max_instances',
          inputs=instance_feature_names,
          max_instances=max_instances_per_image),
    ]

    return train_transforms


def get_object_detection_eval_transforms(
        cfg,
        image_size: Tuple[int, int],
        max_instances_per_image: int):
    return [
        # D(name='record_original_image_size'),
        D(name='resize_image',
          inputs=['image'],
          antialias=[True],
          target_size=image_size),
        D(name='pad_image_to_max_size',
          inputs=['image'],
          target_size=image_size,
          object_coordinate_keys=['bbox']),
        D(name='truncate_or_pad_to_max_instances',
          inputs=['bbox', 'label', 'area', 'is_crowd'],
          max_instances=max_instances_per_image),
    ]


def get_semantic_segmentation_train_transforms(
        cfg,
        image_size: Tuple[int, int],
        max_seq_len,
        rle_from_mask,
):
    train_transforms = [
        D(name='resize_image',
          inputs=['image', 'mask'],
          antialias=[True],
          target_size=image_size),
    ]
    if rle_from_mask:
        if cfg.scale_jitter:
            # print('annoying scale_jitter is enabled')
            train_transforms.append(
                D(name='scale_jitter',
                  inputs=['image', 'mask'],
                  target_size=image_size,
                  min_scale=cfg.jitter_scale_min,
                  max_scale=cfg.jitter_scale_max)
            )
        if cfg.fixed_crop:
            # print('equally annoying fixed_crop is enabled')
            train_transforms.append(
                D(name='fixed_size_crop',
                  inputs=['image', 'mask'],
                  target_size=image_size,
                  )
            )
        else:
            train_transforms.append(
                D(name='resize_image',
                  inputs=['image', 'mask'],
                  antialias=[True],
                  target_size=image_size)
            )

        train_transforms += [
            D(name='random_vertical_flip',
              inputs=['image', 'mask'],
              ),
            D(name='random_horizontal_flip',
              inputs=['image', 'mask'],
              ),
            D(name='pad_image_to_max_size',
              inputs=['image', 'mask'],
              target_size=image_size,
              ),
        ]
    else:
        train_transforms += [
            D(name='pad_image_to_max_size',
              inputs=['image', ],
              target_size=image_size,
              ),
            D(name='truncate_or_pad_to_max_instances',
              inputs=['rle', ],
              max_instances=max_seq_len),
        ]
    return train_transforms


def get_video_detection_train_transforms(
        cfg,
        target_size,
        image_size: Tuple[int, int],
        length: int,
        max_disp: int,
        max_instances_per_image: int,
):
    # return get_video_detection_eval_transforms(
    #     image_size, length, max_instances_per_image)

    instance_feature_names = ['bbox', 'class_id', 'class_name',
                              'area', 'is_crowd']
    object_coordinate_keys = ['bbox']

    train_transforms = [
        # D(name='record_original_video_size'),
    ]
    # if target_size is not None:
    #     train_transforms.append(
    #         D(name='resize_video',
    #           inputs=['video'],
    #           length=length,
    #           antialias=[True],
    #           target_size=target_size,
    #           preserve_aspect_ratio=[False],
    #           )
    #     )
    if cfg.scale_jitter:
        # print('annoying scale_jitter is enabled')
        train_transforms.append(
            D(name='scale_jitter_video',
              inputs=['video'],
              length=length,
              target_size=image_size,
              min_scale=cfg.jitter_scale_min,
              max_scale=cfg.jitter_scale_max)
        )
    if cfg.fixed_crop:
        # print('equally annoying fixed_size_crop_video is enabled')
        train_transforms.append(
            D(name='fixed_size_crop_video',
              inputs=['video'],
              target_size=image_size,
              object_coordinate_keys=object_coordinate_keys)
        )
    else:
        train_transforms.append(
            D(name='resize_video',
              inputs=['video'],
              length=length,
              antialias=[True],
              target_size=image_size)
        )
    train_transforms += [
        D(name='random_horizontal_flip_video',
          inputs=['video'],
          length=length,
          bbox_keys=['bbox']),

        D(name='filter_invalid_objects_video',
          inputs=instance_feature_names,
          length=length,
          filter_keys=['is_crowd']
          ),

        D(name='reorder_object_instances',
          inputs=instance_feature_names,
          order=cfg.object_order),

        D(name='inject_noise_bbox_video',
          length=length,
          max_disp=max_disp,
          max_instances_per_image=max_instances_per_image),

        D(name='pad_video_to_max_size',
          inputs=['video'],
          length=length,
          target_size=image_size,
          object_coordinate_keys=object_coordinate_keys),

        D(name='truncate_or_pad_to_max_instances',
          inputs=instance_feature_names,
          max_instances=max_instances_per_image),
    ]
    return train_transforms


def get_static_video_detection_train_transforms(
        cfg,
        target_size,
        image_size: Tuple[int, int],
        length: int,
        max_disp: int,
        max_instances_per_image: int,
):
    # return get_video_detection_eval_transforms(
    #     image_size, length, max_instances_per_image)

    instance_feature_names = ['bbox', 'class_id', 'class_name',
                              'area', 'is_crowd']
    object_coordinate_keys = ['bbox']

    train_transforms = [
        # D(name='record_original_video_size'),
    ]
    # if target_size is not None:
    #     train_transforms.append(
    #         D(name='resize_video',
    #           inputs=['video'],
    #           length=length,
    #           antialias=[True],
    #           target_size=target_size,
    #           preserve_aspect_ratio=[False],
    #           )
    #     )
    if cfg.scale_jitter:
        # print('annoying scale_jitter is enabled')
        train_transforms.append(
            D(name='scale_jitter_video',
              inputs=['image'],
              length=length,
              target_size=image_size,
              min_scale=cfg.jitter_scale_min,
              max_scale=cfg.jitter_scale_max)
        )
    if cfg.fixed_crop:
        # print('equally annoying fixed_size_crop_video is enabled')
        train_transforms.append(
            D(name='fixed_size_crop_video',
              inputs=['image'],
              target_size=image_size,
              object_coordinate_keys=object_coordinate_keys)
        )
    else:
        train_transforms.append(
            D(name='resize_video',
              inputs=['image'],
              length=length,
              antialias=[True],
              target_size=image_size)
        )
    train_transforms += [
        D(name='random_horizontal_flip_video',
          inputs=['image'],
          length=length,
          bbox_keys=['bbox']),

        D(name='filter_invalid_objects_video',
          inputs=instance_feature_names,
          length=length,
          filter_keys=['is_crowd']
          ),

        D(name='reorder_object_instances',
          inputs=instance_feature_names,
          order=cfg.object_order),

        D(name='inject_noise_bbox_video',
          length=length,
          max_disp=max_disp,
          max_instances_per_image=max_instances_per_image),

        D(name='pad_video_to_max_size',
          inputs=['image'],
          length=length,
          target_size=image_size,
          object_coordinate_keys=object_coordinate_keys),

        D(name='truncate_or_pad_to_max_instances',
          inputs=instance_feature_names,
          max_instances=max_instances_per_image),
    ]
    return train_transforms


def get_video_detection_eval_transforms(
        cfg,
        image_size: Tuple[int, int],
        length: int,
        max_instances_per_image: int,
):
    instance_feature_names = ['bbox', 'class_id', 'class_name',
                              'area', 'is_crowd']
    return [
        # D(name='record_original_video_size'),
        D(name='resize_video',
          inputs=['video'],
          length=length,
          antialias=[True],
          target_size=image_size),
        D(name='pad_video_to_max_size',
          length=length,
          inputs=['video'],
          target_size=image_size,
          object_coordinate_keys=['bbox']),
        D(name='truncate_or_pad_to_max_instances',
          inputs=instance_feature_names,
          max_instances=max_instances_per_image),
    ]


def get_static_video_detection_eval_transforms(
        cfg,
        image_size: Tuple[int, int],
        length: int,
        max_instances_per_image: int,
):
    instance_feature_names = ['bbox', 'class_id', 'class_name',
                              'area', 'is_crowd']
    return [
        # D(name='record_original_video_size'),
        D(name='resize_video',
          inputs=['image'],
          length=length,
          antialias=[True],
          target_size=image_size),
        D(name='pad_video_to_max_size',
          length=length,
          inputs=['image'],
          target_size=image_size,
          object_coordinate_keys=['bbox']),
        D(name='truncate_or_pad_to_max_instances',
          inputs=instance_feature_names,
          max_instances=max_instances_per_image),
    ]


def get_semantic_segmentation_eval_transforms(
        image_size: Tuple[int, int],
        max_seq_len,
        rle_from_mask,
):
    eval_transforms = [
        D(name='resize_image',
          inputs=['image', ],
          antialias=[True],
          target_size=image_size),
    ]
    if rle_from_mask:
        pass
    else:
        eval_transforms += [
            D(name='pad_image_to_max_size',
              inputs=['image', ],
              target_size=image_size,
              ),
            D(name='truncate_or_pad_to_max_instances',
              inputs=['rle', ],
              max_instances=max_seq_len),
        ]
    return eval_transforms


def get_video_segmentation_train_transforms(
        image_size: Tuple[int, int],
        max_seq_len,
        rle_from_mask,
):
    train_transforms = [
        D(name='resize_video',
          inputs=['video', ],
          antialias=[True],
          target_size=image_size),
    ]
    if rle_from_mask:
        pass
    else:
        train_transforms.append(
            D(name='truncate_or_pad_to_max_instances',
              inputs=['rle', ],
              max_instances=max_seq_len),
        )
    return train_transforms


def get_video_segmentation_eval_transforms(
        image_size: Tuple[int, int],
        max_seq_len,
        rle_from_mask,
):
    train_transforms = [
        D(name='resize_video',
          inputs=['video', ],
          antialias=[True],
          target_size=image_size),
    ]
    if rle_from_mask:
        pass
    else:
        train_transforms.append(
            D(name='truncate_or_pad_to_max_instances',
              inputs=['rle', ],
              max_instances=max_seq_len),
        )
    return train_transforms


def get_static_video_segmentation_train_transforms(
        image_size: Tuple[int, int],
        max_seq_len,
):
    train_transforms = [
        D(name='resize_video',
          inputs=['image', ],
          antialias=[True],
          target_size=image_size),
        D(name='truncate_or_pad_to_max_instances',
          inputs=['rle', ],
          max_instances=max_seq_len),
    ]
    return train_transforms


def get_static_video_segmentation_eval_transforms(
        image_size: Tuple[int, int],
        max_seq_len,
):
    return [
        D(name='resize_video',
          inputs=['image', ],
          antialias=[True],
          target_size=image_size),
        D(name='truncate_or_pad_to_max_instances',
          inputs=['rle', ],
          max_instances=max_seq_len),
    ]


def get_instance_segmentation_train_transforms(
        image_size: Tuple[int, int],
        max_instances_per_image: int,
        jitter_scale_min: float = 0.3,
        jitter_scale_max: float = 1.0):
    instance_feature_names = ['bbox', 'label', 'area', 'is_crowd', 'polygon',
                              'scores']
    object_coordinate_keys = ['bbox', 'polygon']
    return [
        # D(name='record_original_image_size'),
        D(name='convert_image_dtype_float32',
          inputs=['image']),
        D(name='filter_invalid_objects',
          inputs=instance_feature_names,
          filter_keys=['is_crowd']),
        D(name='reorder_object_instances',
          inputs=instance_feature_names,
          order='scores'),
        D(name='truncate_or_pad_to_max_instances',
          inputs=instance_feature_names,
          max_instances=max_instances_per_image),
        D(name='copy_fields', inputs=['polygon'], outputs=['polygon_orig']),
        D(name='scale_jitter',
          inputs=['image'],
          target_size=image_size,
          min_scale=jitter_scale_min,
          max_scale=jitter_scale_max),
        D(name='fixed_size_crop',
          inputs=['image'],
          target_size=image_size,
          object_coordinate_keys=object_coordinate_keys),
        D(name='random_horizontal_flip',
          inputs=['image'],
          bbox_keys=['bbox'],
          polygon_keys=['polygon']),
        D(name='pad_image_to_max_size',
          inputs=['image'],
          target_size=image_size,
          object_coordinate_keys=object_coordinate_keys),
        D(name='preserve_reserved_tokens',
          points_keys=['polygon'],
          points_orig_keys=['polygon_orig']),
    ]


def get_instance_segmentation_eval_transforms(
        image_size: Tuple[int, int],
        max_instances_per_image: int):
    instance_feature_names = ['bbox', 'label', 'area', 'is_crowd', 'polygon',
                              'scores']
    object_coordinate_keys = ['bbox', 'polygon']
    return [
        # D(name='record_original_image_size'),
        D(name='reorder_object_instances',
          inputs=instance_feature_names,
          order='scores'),
        D(name='truncate_or_pad_to_max_instances',
          inputs=instance_feature_names,
          max_instances=max_instances_per_image),
        D(name='copy_fields', inputs=['polygon'], outputs=['polygon_orig']),
        D(name='resize_image',
          inputs=['image'],
          antialias=[True],
          target_size=image_size),
        D(name='pad_image_to_max_size',
          inputs=['image'],
          target_size=image_size,
          object_coordinate_keys=object_coordinate_keys),
        D(name='preserve_reserved_tokens',
          points_keys=['polygon'],
          points_orig_keys=['polygon_orig']),
    ]


def get_keypoint_detection_train_transforms(
        image_size: Tuple[int, int],
        max_instances_per_image: int,
        jitter_scale_min: float = 0.3,
        jitter_scale_max: float = 1.0):
    instance_feature_names = ['bbox', 'label', 'area', 'is_crowd', 'keypoints',
                              'num_keypoints', 'scores']
    object_coordinate_keys = ['bbox', 'keypoints']
    return [
        # D(name='record_original_image_size'),
        D(name='filter_invalid_objects',
          inputs=instance_feature_names,
          filter_keys=['is_crowd']),
        D(name='reorder_object_instances',
          inputs=instance_feature_names,
          order='random'),
        D(name='truncate_or_pad_to_max_instances',
          inputs=instance_feature_names,
          max_instances=max_instances_per_image),
        D(name='copy_fields', inputs=['keypoints'], outputs=['keypoints_orig']),
        D(name='scale_jitter',
          inputs=['image'],
          target_size=image_size,
          min_scale=jitter_scale_min,
          max_scale=jitter_scale_max),
        D(name='fixed_size_crop',
          inputs=['image'],
          target_size=image_size,
          object_coordinate_keys=object_coordinate_keys),
        # Disabling random flip because of a bug that does not flip location of
        # keypoints (e.g. left eye vs right eye).
        # D(name='random_horizontal_flip',
        #   inputs=['image'],
        #   bbox_keys=['bbox'],
        #   keypoints_keys=['keypoints']),
        D(name='pad_image_to_max_size',
          inputs=['image'],
          target_size=image_size,
          object_coordinate_keys=object_coordinate_keys),
        D(name='preserve_reserved_tokens',
          points_keys=['keypoints'],
          points_orig_keys=['keypoints_orig']),
    ]


def get_keypoint_detection_eval_transforms(
        image_size: Tuple[int, int],
        max_instances_per_image: int):
    instance_feature_names = ['bbox', 'label', 'area', 'is_crowd', 'keypoints',
                              'num_keypoints', 'scores']
    object_coordinate_keys = ['bbox', 'keypoints']
    return [
        # D(name='record_original_image_size'),
        D(name='reorder_object_instances',
          inputs=instance_feature_names,
          order='random'),
        D(name='truncate_or_pad_to_max_instances',
          inputs=instance_feature_names,
          max_instances=max_instances_per_image),
        D(name='copy_fields', inputs=['keypoints'], outputs=['keypoints_orig']),
        D(name='resize_image',
          inputs=['image'],
          antialias=[True],
          target_size=image_size),
        D(name='pad_image_to_max_size',
          inputs=['image'],
          target_size=image_size,
          object_coordinate_keys=object_coordinate_keys),
        D(name='preserve_reserved_tokens',
          points_keys=['keypoints'],
          points_orig_keys=['keypoints_orig']),
    ]


def get_captioning_train_transforms(
        image_size: Tuple[int, int],
        color_jitter_strength: float = 0.5,
        jitter_scale_min: float = 0.3,
        jitter_scale_max: float = 2.0):
    return [
        D(name='scale_jitter',
          inputs=['image'],
          target_size=image_size,
          min_scale=jitter_scale_min,
          max_scale=jitter_scale_max),
        D(name='fixed_size_crop',
          inputs=['image'],
          target_size=image_size),
        D(name='random_horizontal_flip',
          inputs=['image']),
        D(name='random_color_jitter',
          inputs=['image'],
          color_jitter_strength=color_jitter_strength),
        D(name='pad_image_to_max_size',
          inputs=['image'],
          target_size=image_size),
    ]


def get_captioning_eval_transforms(
        image_size: Tuple[int, int]):
    return [
        D(name='resize_image',
          inputs=['image'],
          antialias=[True],
          target_size=image_size),
        D(name='pad_image_to_max_size',
          inputs=['image'],
          target_size=image_size),
    ]


def get_panoptic_segmentation_train_transforms(
        image_size: Tuple[int, int],
        mask_size: Tuple[int, int],
        jitter_scale_min: float,
        jitter_scale_max: float,
        color_jitter_strength: float):
    """Train transforms for panoptic segmentation."""
    transforms = [
        # D(name='record_original_image_size'),
        D(name='scale_jitter',
          inputs=['image', 'label_map'],
          target_size=image_size,
          min_scale=jitter_scale_min,
          max_scale=jitter_scale_max,
          resize_method=['bilinear', 'nearest'],
          antialias=[True, True]),
        D(name='fixed_size_crop',
          inputs=['image', 'label_map'],
          target_size=image_size),
        D(name='random_horizontal_flip',
          inputs=['image', 'label_map']),
        D(name='random_color_jitter',
          inputs=['image'],
          color_jitter_strength=color_jitter_strength),
        D(name='pad_image_to_max_size',
          inputs=['image', 'label_map'],
          target_size=image_size,
          background_val=[0.3, 0]),
    ]
    if image_size[0] != mask_size[0] or image_size[1] != mask_size[1]:
        transforms.append(D(
            name='resize_image',
            inputs=['label_map'],
            target_size=mask_size,
            resize_method=['nearest']))
    return transforms


def get_panoptic_segmentation_eval_transforms(image_size: Tuple[int, int]):
    return [
        # D(name='record_original_image_size'),
        # This is needed if coco_metrics.gt_annotations_path is missing, e.g.
        # for cityscapes, such that we need to read groundtruth label map from
        # the example.
        D(name='copy_fields', inputs=['label_map'], outputs=['label_map_orig']),
        D(name='resize_image',
          inputs=['image', 'label_map'],
          target_size=image_size,
          resize_method=['bilinear', 'nearest'],
          antialias=[True, True]),
        D(name='pad_image_to_max_size',
          inputs=['image', 'label_map'],
          target_size=image_size,
          background_val=[0.3, 0]),
    ]


def get_video_panoptic_segmentation_train_transforms(
        image_size: Tuple[int, int],
        mask_size: Tuple[int, int],
        jitter_scale_min: float,
        jitter_scale_max: float,
        color_jitter_strength: float):
    """Train transforms for video panoptic segmentation."""
    transforms = [
        D(name='scale_jitter',
          inputs=['image', 'cond_map', 'label_map'],
          target_size=image_size,
          min_scale=jitter_scale_min,
          max_scale=jitter_scale_max,
          resize_method=['bicubic', 'nearest', 'nearest'],
          antialias=[True, True, True]),
        D(name='fixed_size_crop',
          inputs=['image', 'cond_map', 'label_map'],
          target_size=image_size),
        D(name='random_horizontal_flip',
          inputs=['image', 'cond_map', 'label_map']),
        D(name='random_color_jitter',
          inputs=['image'],
          color_jitter_strength=color_jitter_strength),
        D(name='pad_image_to_max_size',
          inputs=['image', 'cond_map', 'label_map'],
          target_size=image_size,
          background_val=[0.3, 0, 0]),
    ]
    if image_size[0] != mask_size[0] or image_size[1] != mask_size[1]:
        transforms.append(D(
            name='resize_image',
            inputs=['cond_map', 'label_map'],
            target_size=mask_size,
            resize_method=['nearest', 'nearest']))
    return transforms


def get_video_panoptic_segmentation_eval_transforms(
        image_size: Tuple[int, int],
        mask_size: Tuple[int, int],
        max_num_frames: int):
    return [
        D(name='resize_image',
          inputs=['image'],
          target_size=image_size,
          resize_method=['bilinear'],
          antialias=[True]),
        D(name='resize_image',
          inputs=['cond_map', 'label_map'],
          target_size=mask_size,
          resize_method=['nearest', 'nearest']),
        D(name='pad_image_to_max_size',
          inputs=['image'],
          target_size=image_size,
          background_val=[0.3]),
        D(name='pad_image_to_max_size',
          inputs=['cond_map', 'label_map'],
          target_size=mask_size,
          background_val=[0, 0]),
        D(name='truncate_or_pad_to_max_frames',
          inputs=['image', 'cond_map', 'label_map'],
          max_num_frames=max_num_frames),
    ]
