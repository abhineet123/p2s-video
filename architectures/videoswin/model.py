import os
import warnings
from functools import partial

warnings.simplefilter(action="ignore")
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from architectures.videoswin.layers import TFPatchEmbed3D
from architectures.videoswin.layers import TFPatchMerging
from architectures.videoswin.layers import TFAdaptiveAveragePooling3D
from architectures.videoswin.blocks import TFBasicLayer


class TFSwinTransformer3D(keras.layers.Layer):
    """ Swin Transformer backbone.
        A Keras impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: LayerNormalization.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(
            self,
            length,
            patch_size=(4, 4, 4),
            in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=(2, 7, 7),
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            norm_layer=layers.LayerNormalization,
            patch_norm=False,
            frozen_stages=-1,
            # num_classes=400,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.length = length
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = TFPatchEmbed3D(
            length=length,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            name='TFPatchEmbed3D'
        )
        self.pos_drop = layers.Dropout(drop_rate, name='pos_drop')

        # stochastic depth
        dpr = tf.linspace(0., drop_path_rate, sum(depths)).numpy().tolist()

        # build layers
        self.basic_layers = []
        for i_layer in range(self.num_layers):
            layer = TFBasicLayer(
                layer_id=i_layer,
                length=length,
                patch_size=patch_size,
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=TFPatchMerging if (i_layer < self.num_layers - 1) else None,
                name=f'TFBasicLayer{i_layer + 1}'
            )
            self.basic_layers.append(layer)

        self.norm = norm_layer(axis=-1, epsilon=1e-05, name='norm')
        self.avg_pool3d = TFAdaptiveAveragePooling3D((1, 1, 1), name='adt_avg_pool3d')
        # self.head = layers.Dense(num_classes, use_bias=True, name='head', dtype='float32')

    def call(self, x, return_attns=False, training=None):

        # tensor embeddings
        x = self.patch_embed(x)
        x = self.pos_drop(x, training=training)

        # video-swin block computation
        attn_scores_outputs = {}
        for layer in self.basic_layers:
            """dedicated flow path for return_attns possibly allows tensor shapes to be 
            resolved better when building the graph"""
            if return_attns:
                x, attn_scores = layer(x, return_attns=return_attns, training=training)
                attn_scores_outputs[f"{layer.name}_att"] = attn_scores
            else:
                x = layer(x, training=training)

        # head branch
        # x = self.norm(x)
        # x = self.avg_pool3d(x)

        # x = self.head(x)

        # x = tf.squeeze(x, axis=1)

        if return_attns:
            return x, attn_scores_outputs

        return x

    def build(self, input_shape):
        super().build(input_shape)
        self.build_shape = input_shape[1:]

    def build_graph(self):
        x = keras.Input(shape=self.build_shape, name='input_graph')
        return keras.Model(
            inputs=[x], outputs=self.call(x)
        )


def video_swin_tiny(length, patch_dim, pt, verify, window_size=(8, 7, 7), drop_path_rate=0.2, **kwargs):
    model = TFSwinTransformer3D(
        length=length,
        # patch_size=(2, 4, 4),
        patch_size=(patch_dim, 4, 4),
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=window_size,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=drop_path_rate,
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-05),
        patch_norm=True,
        **kwargs
    )

    if pt:
        load_pt('TFVideoSwinT_K400_IN1K_P244_W877_32x224', model, verify)
    return model


def video_swin_small(length, patch_dim, pt, verify, window_size=(8, 7, 7), drop_path_rate=0.2, **kwargs):
    model = TFSwinTransformer3D(
        length=length,
        # patch_size=(2, 4, 4),
        patch_size=(patch_dim, 4, 4),
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=window_size,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=drop_path_rate,
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-05),
        patch_norm=True,
        **kwargs
    )
    if pt:
        load_pt('TFVideoSwinS_K400_IN1K_P244_W877_32x224', model, verify)

    return model


def video_swin_base(length, patch_dim, pt, verify, window_size=(8, 7, 7), drop_path_rate=0.2, **kwargs):
    model = TFSwinTransformer3D(
        length=length,
        # patch_size=(2, 4, 4),
        patch_size=(patch_dim, 4, 4),
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=window_size,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=drop_path_rate,
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-05),
        patch_norm=True,
        **kwargs
    )
    if pt==1:
        load_pt('TFVideoSwinB_K400_IN1K_P244_W877_32x224', model, verify)
    elif pt==2:
        load_pt('TFVideoSwinB_K400_IN22K_P244_W877_32x224', model, verify)
    elif pt == 3:
        load_pt('TFVideoSwinB_K600_IN22K_P244_W877_32x224', model, verify)
    return model


def load_pt(pt_name, model, verify):
    pt_dir_path = f'pretrained/{pt_name}'
    pt_dir_path = os.path.abspath(pt_dir_path)
    assert os.path.exists(pt_dir_path), f"nonexistent pt_dir_path: {pt_dir_path}"

    pt_path = os.path.join(pt_dir_path, 'variables', 'variables')
    # assert os.path.exists(pt_path), f"nonexistent pt_path: {pt_path}"

    import utils

    checkpoint = tf.train.Checkpoint(model=model)
    ckpt_vars, ckpt_vars_dict, name_to_shape = utils.save_ckpt_vars(pt_dir_path, latest_ckpt=pt_path)
    print(f'Restoring from checkpoint: {pt_path}')

    # status = checkpoint.restore(pt_path)
    status = checkpoint.restore(pt_path).expect_partial()

    if verify:
        verify_restored = status.assert_consumed
        verify_existing = status.assert_existing_objects_matched

        try:
            verify_restored()
        except AssertionError as e:
            print('+++++++++++ restored +++++++++++')
            print(e)
            print('+++++++++++ restored +++++++++++')

        try:
            verify_existing()
        except AssertionError as e:
            print('+++++++++++ existing +++++++++++')
            print(e)
            print('+++++++++++ existing +++++++++++')

    return ckpt_vars_dict, name_to_shape
