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
    image_size = cfg.model.image_size
    max_seq_len = cfg.model.max_seq_len
    rle_from_mask = cfg.dataset.rle_from_mask
    cfg.task.image_size = image_size

    """"update parameters that depend on image size but inexplicably missing from pretrained config files"""
    assert cfg.task.name == 'mhd_semantic_segmentation', f"invalid task name: {cfg.task.name}"

    for task in cfg.tasks + [cfg.task, ]:
        task.image_size = image_size

        task.eval_transforms = transform_configs.get_semantic_segmentation_eval_transforms(
            image_size, max_seq_len, rle_from_mask, cfg.debug)

        task.train_transforms = transform_configs.get_semantic_segmentation_train_transforms(
            cfg.dataset.transforms,
            image_size, max_seq_len, rle_from_mask, cfg.debug)


def get_config(config_str=None):
    """config_str is either empty or contains task,architecture variants."""

    task_variant = 'mhd_semantic_segmentation@ipsc_semantic_segmentation'

    # encoder_variant = 'vit-b'
    encoder_variant = 'resnet'

    image_size = (640, 640)
    # image_size = 640

    tasks_and_datasets = []
    for task_and_ds in task_variant.split('+'):
        tasks_and_datasets.append(task_and_ds.split('@'))

    task_config_map = {
        'mhd_semantic_segmentation': D(
            name='mhd_semantic_segmentation',
            vocab_id=vocab.TASK_MHD_SEM_SEG,
            image_size=image_size,
            # starts_bins=6400,
            # lengths_bins=80,
            eos_token_weight=0.1,
            top_k=0,
            top_p=0.4,
            temperature=1.0,
            weight=1.0,

            # increase weight assigned to class tokens so it is equal to all the coord tokens combined
            # since no. of coord tokens is n*4 times the number of class tokens, the latter can often be
            # relatively ignored during training, thus leading to lots of mis classifications during inference
            class_equal_weight=0,
        ),
    }

    task_d_list = []
    dataset_list = []
    for tv, ds_name in tasks_and_datasets:
        task_d_list.append(task_config_map[tv])
        dataset_config = copy.deepcopy(dataset_configs.dataset_configs[ds_name])
        dataset_list.append(dataset_config)

    dataset_list[0]['starts_2d'] = 1
    dataset_list[0]['length_as_class'] = 0
    dataset_list[0]['shared_coord'] = 0
    dataset_list[0]['multi_class'] = 1

    config = D(
        dataset=dataset_list[0],
        datasets=dataset_list,

        task=task_d_list[0],
        tasks=task_d_list,

        model=D(
            name='mhd_encoder_ar_decoder',
            image_size=image_size,

            max_seq_len=0,
            defer_seq=0,

            shared_mha=1,
            # shared embedding matrix for x, y, l tokens
            shared_xyl=0,

            coord_vocab_size=0,
            len_vocab_size=0,
            class_vocab_size=0,
            defer_vocab=1,

            coord_vocab_shift=vocab.MHD_VOCAB_SHIFT,
            len_vocab_shift=vocab.MHD_VOCAB_SHIFT,
            class_vocab_shift=vocab.MHD_VOCAB_SHIFT,

            multi_class=0,
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
    config.debug = base_config.debug
    update_task_config(config)

    # Update model with architecture variant.
    for key, value in architecture_config_map[encoder_variant].items():
        config.model[key] = value

    config.update(base_config)

    config.train.mhd_loss_weights = '1111'
    config.train.save_suffix = ['mhd', ]
    config.model.mhd = 1

    return config
