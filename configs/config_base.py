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
"""Common / shared settings among multiple configs."""

import ml_collections


def D(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


train_eval_config = D(
    steps=0,
    db_prefix=[],
    save_suffix=[],
    pt=0,
    batch_size=16,
)
train_config = D(
    check_ckpt=0,
    epochs=10000,
    checkpoint_epochs=1,
    checkpoint_steps=0,  # set to >0 to override checkpoint_epochs.
    keep_checkpoint_max=2,
    loss_type='xent',
    freeze_backbone=0,
    freeze_encoder=0,
    freeze_decoder=0,
    freeze_encoder_decoder=0,
    val_epochs=0,
)

eval_config = D(
    ckpt_iter=0,
    check_ckpt=0,
    add_stride_info=1,
    csv_steps=10,
    write_to_video=1,
    mask_from_gt=1,
    mask_from_logits=0,
    show_vis=0,
    save_mask=1,
    save_vis=0,
    save_csv=1,
    profile=0,
    info_file='',
    run_existing=1,
    remote='',
    proxy='',
    # time in hours to wait between evals
    sleep_eval=2.,
    # time in minutes to wait before checking for new ckpt again
    sleep_ckpt=5.,
    # only transfers checkpoints periodically while deferring the actual evaluation run for later
    defer=0.,
    tag='eval',
    checkpoint_dir='',  # checkpoint_dir will be model_dir if not set.
)

eval_config.update(train_eval_config)
train_config.update(train_eval_config)

base_config = D(
    mode="train",
    use_tpu=0,
    dist=0,
    master=None,
    eager=0,
    dyn_ram=1,
    debug=0,
    resnet_replace=[],
    gpu='',

    model_dir='',
    eval_type='',
    pretrained='',

    train=train_config,
    eval=eval_config,
)

architecture_config_map = {
    'vit-b': D(
        resnet_variant='c1',
        num_encoder_layers=12,
        dim_att=768,
        dim_mlp=3072,
        num_heads=12,
        num_decoder_layers=6,
        dim_att_dec=512,
        dim_mlp_dec=2048,
        num_heads_dec=16,
    ),
    'vit-l': D(
        resnet_variant='c1',
        num_encoder_layers=24,
        dim_att=1024,
        dim_mlp=4096,
        num_heads=16,
        num_decoder_layers=8,
        dim_att_dec=512,
        dim_mlp_dec=2048,
        num_heads_dec=16,
    ),
    'resnet': D(
        resnet_variant='standard',

        swin_variant='t',
        swin_pt=0,
        swin_patch_dim=0,

        resnet_depth=50,
        resnet_sk_ratio=0.,
        resnet_width_multiplier=1,
        num_encoder_layers=6,
        dim_att=256,
        dim_mlp=1024,
        num_heads=8,
        num_decoder_layers=6,
        dim_att_dec=256,
        dim_mlp_dec=1024,
        num_heads_dec=8,
    ),
    'resnet-c': D(
        resnet_variant='c4',
        resnet_depth=50,
        resnet_sk_ratio=0.,
        resnet_width_multiplier=1,
        num_encoder_layers=12,
        dim_att=512,
        dim_mlp=2048,
        num_heads=16,
        num_decoder_layers=8,
        dim_att_dec=512,
        dim_mlp_dec=2048,
        num_heads_dec=16,
    ),
}
