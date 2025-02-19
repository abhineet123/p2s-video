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
"""Config file for object detection fine-tuning and evaluation."""

import copy

import vocab

from configs import dataset_configs
from configs import transform_configs
from configs.config_base import architecture_config_map, base_config
from configs.config_base import D


# pylint: disable=invalid-name,line-too-long,missing-docstring


def update_task_config(cfg):
    """
        hack to deal with independently defined target_size setting in tasks.eval_transforms even though
        it should match image_size
        """
    # if cfg.task.image_size == cfg.model.image_size:
    #     return

    image_size = cfg.model.image_size
    max_seq_len = cfg.model.max_seq_len
    coords_1d = cfg.model.coords_1d

    cfg.task.image_size = image_size
    cfg.task.coords_1d = coords_1d

    """"update parameters that depend on image size but inexplicably missing from pretrained config files"""
    assert cfg.task.name == 'object_detection', f"invalid task name: {cfg.task.name}"

    for task_config in cfg.tasks + [cfg.task, ]:
        n_bbox_tokens = 2 if cfg.task.coords_1d else 4
        max_instances_per_image = int(max_seq_len // (n_bbox_tokens + 1))

        task_config.max_instances_per_image = max_instances_per_image
        task_config.max_instances_per_image_test = max_instances_per_image

        task_config.image_size = image_size

        task_config.eval_transforms = transform_configs.get_object_detection_eval_transforms(
            cfg.dataset.transforms,
            image_size, task_config.max_instances_per_image_test)

        task_config.train_transforms = transform_configs.get_object_detection_train_transforms(
            cfg.dataset.transforms,
            image_size, task_config.max_instances_per_image,
        )


def get_config(config_str=None):
    """config_str is either empty or contains task,architecture variants."""

    task_variant = 'object_detection@ipsc_object_detection'

    # encoder_variant = 'vit-b'
    encoder_variant = 'resnet'

    image_size = (640, 640)
    # image_size = 640

    tasks_and_datasets = []
    for task_and_ds in task_variant.split('+'):
        tasks_and_datasets.append(task_and_ds.split('@'))

    max_instances_per_image = 100
    max_instances_per_image_test = 100

    task_config_map = {
        'object_detection': D(
            name='object_detection',
            vocab_id=vocab.TASK_OBJ_DET,
            coords_1d=0,
            image_size=image_size,
            quantization_bins=1000,
            max_instances_per_image=max_instances_per_image,
            max_instances_per_image_test=max_instances_per_image_test,

            # Train on both ground-truth and (augmented) noisy objects.
            noise_bbox_weight=1.0,
            eos_token_weight=0.1,

            # increase weight assigned to class tokens so it is equal to all the coord tokens combined
            # since no. of coord tokens is n*4 times the number of class tokens, the latter can often be
            # relatively ignored during training, thus leading to lots of misclassifications during inference
            class_equal_weight=0,

            # Train on just ground-truth objects (with an ending token).
            # noise_bbox_weight=0.0,
            # eos_token_weight=0.1,
            class_label_corruption='rand_n_fake_cls',
            top_k=0,
            top_p=0.4,
            temperature=1.0,
            weight=1.0,
            # metric=D(name='coco_object_detection', ),
        ),
    }

    task_d_list = []
    dataset_list = []
    for tv, ds_name in tasks_and_datasets:
        task_d_list.append(task_config_map[tv])
        dataset_config = copy.deepcopy(dataset_configs.dataset_configs[ds_name])
        dataset_list.append(dataset_config)

    config = D(
        dataset=dataset_list[0],
        datasets=dataset_list,

        task=task_d_list[0],
        tasks=task_d_list,

        model=D(
            name='encoder_ar_decoder',
            image_size=image_size,
            max_seq_len=512,
            coords_1d=0,
            defer_seq=0,
            defer_vocab=0,
            vocab_size=3000,  # Note: should be large enough for 100 + num_classes + quantization_bins + (optional) text
            coord_vocab_shift=1000,  # Note: make sure num_class <= coord_vocab_shift - 100
            text_vocab_shift=3000,  # Note: make sure coord_vocab_shift + quantization_bins <= text_vocab_shift
            use_cls_token=False,
            shared_decoder_embedding=True,
            decoder_output_bias=True,
            patch_size=16,
            drop_path=0.1,
            drop_units=0.1,
            drop_att=0.0,
            dec_proj_mode='mlp',
            pos_encoding='sin_cos',
            pos_encoding_dec='learned',
            pretrained_ckpt=None,
        ),

        optimization=D(
            optimizer='adamw',
            learning_rate=3e-5,
            end_lr_factor=0.01,
            warmup_epochs=2,
            warmup_steps=0,  # set to >0 to override warmup_epochs.
            weight_decay=0.05,
            global_clipnorm=-1,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            learning_rate_schedule='linear',
            learning_rate_scaling='none',
        ),
    )
    update_task_config(config)

    # Update model with architecture variant.
    for key, value in architecture_config_map[encoder_variant].items():
        config.model[key] = value

    config.update(base_config)

    return config
