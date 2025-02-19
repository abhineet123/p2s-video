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
    assert cfg.task.name == 'semantic_segmentation', f"invalid task name: {cfg.task.name}"

    for task in cfg.tasks + [cfg.task, ]:
        task.image_size = image_size

        task.eval_transforms = transform_configs.get_semantic_segmentation_eval_transforms(
            image_size, max_seq_len, rle_from_mask)

        task.train_transforms = transform_configs.get_semantic_segmentation_train_transforms(
            cfg.dataset.transforms,
            image_size, max_seq_len, rle_from_mask)


def get_config(config_str=None):
    """config_str is either empty or contains task,architecture variants."""

    task_variant = 'semantic_segmentation@ipsc_semantic_segmentation'

    # encoder_variant = 'vit-b'
    encoder_variant = 'resnet'

    image_size = (640, 640)
    # image_size = 640

    tasks_and_datasets = []
    for task_and_ds in task_variant.split('+'):
        tasks_and_datasets.append(task_and_ds.split('@'))

    task_config_map = {
        'semantic_segmentation': D(
            name='semantic_segmentation',
            vocab_id=vocab.TASK_SEM_SEG,
            image_size=image_size,
            # starts_bins=6400,
            # lengths_bins=80,
            eos_token_weight=0.1,
            top_k=0,
            top_p=0.4,
            temperature=1.0,
            weight=1.0,
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
            defer_seq=0,
            defer_vocab=1,
            vocab_size=8000,
            coord_vocab_shift=1000,
            len_vocab_shift=200,
            class_vocab_shift=100,
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
    update_task_config(config)

    # Update model with architecture variant.
    for key, value in architecture_config_map[encoder_variant].items():
        config.model[key] = value

    config.update(base_config)

    return config
