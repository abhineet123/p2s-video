import math
import os.path
import os
import einops

import utils
from architectures import resnet
import tensorflow as tf

from architectures.transformers import (MLP, DropPath, get_shape, add_cls_token_emb,
                                        add_vis_pos_emb, suffix_id, TransformerEncoder)


class VideoTransformerEncoderLayer(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

    def __init__(self,
                 dim,
                 mlp_ratio,
                 num_heads,
                 late_fusion,
                 vid_len,
                 drop_path=0.1,
                 drop_units=0.1,
                 drop_att=0.,
                 dim_x_att=None,
                 self_attention=True,
                 cross_attention=True,
                 use_mlp=True,
                 use_ffn_ln=False,
                 ln_scale_shift=True,
                 **kwargs):
        super(VideoTransformerEncoderLayer, self).__init__(**kwargs)

        self.late_fusion = late_fusion
        self.vid_len = vid_len
        self.dim = dim

        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.use_mlp = use_mlp
        if self_attention:
            self.mha_ln = tf.keras.layers.LayerNormalization(
                epsilon=1e-6,
                center=ln_scale_shift,
                scale=ln_scale_shift,
                name='mha/ln')
            self.mha = tf.keras.layers.MultiHeadAttention(
                num_heads, dim // num_heads, dropout=drop_att, name='mha')
            if use_mlp:
                self.mlp = MLP(1, dim, mlp_ratio, drop_path, drop_units,
                               use_ffn_ln=use_ffn_ln, ln_scale_shift=ln_scale_shift,
                               name='mlp')
        if cross_attention:
            self.cross_ln = tf.keras.layers.LayerNormalization(
                epsilon=1e-6,
                center=ln_scale_shift,
                scale=ln_scale_shift,
                name='cross_mha/ln')
            dim_x_att = dim if dim_x_att is None else dim_x_att
            self.cross_mha = tf.keras.layers.MultiHeadAttention(
                num_heads, dim_x_att // num_heads,
                dropout=drop_att, name='cross_mha')
            if use_mlp:
                self.cross_mlp = MLP(
                    1, dim, mlp_ratio, drop_path, drop_units,
                    use_ffn_ln=use_ffn_ln, ln_scale_shift=ln_scale_shift,
                    name='cross_mlp')

        self.dropp = DropPath(drop_path)

    def call(self, x, mask, training):
        # x shape (bsz, vid_len, seq_len, dim_att), mask shape (bsz, seq_len, seq_len).

        if not self.late_fusion:
            bsz, vid_len, seq_len, dim_att = get_shape(x)

            assert vid_len == self.vid_len, "vid_len mismatch"

        if self.self_attention:
            if not self.late_fusion:
                x = utils.flatten_vid(x)
            x_ln = self.mha_ln(x)
            x_residual = self.mha(x_ln, x_ln, x_ln, attention_mask=mask, training=training)
            x = x + self.dropp(x_residual, training)
            if self.use_mlp:
                x = self.mlp(x, training)
            if not self.late_fusion:
                x = utils.unflatten_vid(x, self.vid_len)

        if self.cross_attention:
            assert not self.late_fusion, "cross_attention is not supported with late_fusion"
            x1 = x[:, 0, ...]
            for _id in range(1, self.vid_len):
                x2 = x[:, _id, ...]
                x1_ln = self.cross_ln(x1)
                x2_ln = self.cross_ln(x2)

                x_res = self.cross_mha(x1_ln, x2_ln, x2_ln, attention_mask=None, training=training)
                x1 = x1 + self.dropp(x_res, training)
            x = x1
            if self.use_mlp:
                x = self.cross_mlp(x, training)

        return x


class VideoTransformerEncoder(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

    def __init__(self,
                 num_layers,
                 dim,
                 mlp_ratio,
                 num_heads,
                 vid_len,
                 late_fusion,
                 drop_path=0.1,
                 drop_units=0.1,
                 drop_att=0.,
                 self_attention=True,
                 use_ffn_ln=False,
                 ln_scale_shift=True,
                 **kwargs):
        super(VideoTransformerEncoder, self).__init__(**kwargs)

        assert vid_len >= 2, "vid_len must be >= 2"

        self.vid_len = vid_len
        self.num_layers = num_layers
        self.late_fusion = late_fusion
        self.enc_layers = [
            VideoTransformerEncoderLayer(
                dim=dim,
                mlp_ratio=mlp_ratio,
                num_heads=num_heads,
                vid_len=vid_len,
                late_fusion=late_fusion,
                drop_path=drop_path,
                drop_units=drop_units,
                drop_att=drop_att,
                cross_attention=False if late_fusion or i < num_layers - 1 else True,
                self_attention=self_attention,
                use_ffn_ln=use_ffn_ln,
                ln_scale_shift=ln_scale_shift,
                name='transformer_encoder' + suffix_id(i))
            for i in range(num_layers)
        ]

    def call(self, x, mask, training, ret_list=False):
        if ret_list:
            x_list = [x]
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask, training)
            if ret_list:
                x_list.append(x)
        return (x, x_list) if ret_list else x


class VideoResNetTransformer(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

    def __init__(self,
                 image_height,
                 image_width,
                 vid_len,
                 late_fusion,
                 resnet_variant,
                 resnet_depth,
                 resnet_width_multiplier,
                 resnet_sk_ratio,
                 num_layers,
                 dim,
                 mlp_ratio,
                 num_heads,
                 drop_path=0.1,
                 drop_units=0.1,
                 drop_att=0.,
                 pos_encoding='sin_cos',
                 use_cls_token=True,
                 freeze_backbone=0,
                 **kwargs):
        super(VideoResNetTransformer, self).__init__(**kwargs)
        self.dim = dim
        self.vid_len = vid_len
        self.use_cls_token = use_cls_token
        self.late_fusion = late_fusion
        self.freeze_backbone = freeze_backbone
        self.resnet = resnet.resnet(
            resnet_depth=resnet_depth,
            width_multiplier=resnet_width_multiplier,
            sk_ratio=resnet_sk_ratio,
            variant=resnet_variant)

        if self.freeze_backbone:
            self.resnet.trainable = False

        self.dropout = tf.keras.layers.Dropout(drop_units)
        self.stem_projection = tf.keras.layers.Dense(dim, name='stem_projection')
        self.stem_ln = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name='stem_ln')
        if self.use_cls_token:
            add_cls_token_emb(self, dim)
        if resnet_variant in ['c3']:
            factor = 8.
        elif resnet_variant in ['c4', 'dc5']:
            factor = 16.
        else:
            factor = 32.
        self.n_rows = math.ceil(image_height / factor)
        self.n_cols = math.ceil(image_width / factor)

        n_images = 1
        if late_fusion:
            pos_encoding = 'learned_3d'
            n_images = self.vid_len

        self.vis_pos_emb = add_vis_pos_emb(
            self,
            pos_encoding,
            self.n_rows,
            self.n_cols,
            dim,
            n_channels=n_images,
            return_only=True,
        )
        self.transformer_encoder = VideoTransformerEncoder(
            num_layers=num_layers,
            dim=dim,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            vid_len=vid_len,
            late_fusion=late_fusion,
            drop_path=drop_path,
            drop_units=drop_units,
            drop_att=drop_att,
            name='transformer_encoder')
        self.output_ln = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name='ouput_ln')

    def call(self, videos, training):
        b, t, h, w, c = get_shape(videos)

        videos = utils.flatten_vid(videos)

        hidden_stack, _ = self.resnet(videos, training)
        """last feature layer"""
        tokens = hidden_stack[-1]

        bt, fh, fw, fc = get_shape(tokens)
        n_feat = fh * fw
        tokens = tf.reshape(tokens, [bt, n_feat, fc])
        tokens = self.stem_ln(self.stem_projection(self.dropout(tokens, training)))

        if self.late_fusion:
            tokens = utils.unflatten_vid(tokens, self.vid_len)

        tokens_vis_pos_emb = tf.expand_dims(self.vis_pos_emb, 0)
        tokens = tokens + tokens_vis_pos_emb

        if self.use_cls_token:
            cls_token = tf.tile(tf.expand_dims(self.cls_token_emb, 0), [bt, 1, 1])
            tokens = tf.concat([cls_token, tokens], 1)

        if self.late_fusion:
            b, t, n_feat, fc = get_shape(tokens)
            tokens = tf.reshape(tokens, [b, t * n_feat, fc])
        else:
            tokens = utils.unflatten_vid(tokens, self.vid_len)

        tokens = self.transformer_encoder(
            tokens, None, training=training, ret_list=False)

        # tokens = utils.flatten_vid(tokens)
        x = self.output_ln(tokens)
        # x = utils.flatten_vid(x)

        return x


class VideoSwinTransformer(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

    def __init__(self,
                 swin_variant,
                 swin_pt,
                 image_height,
                 image_width,
                 patch_dim,
                 vid_len,
                 num_layers,
                 dim,
                 mlp_ratio,
                 num_heads,
                 drop_path=0.1,
                 drop_units=0.1,
                 drop_att=0.,
                 pos_encoding='learned',
                 use_cls_token=True,
                 freeze_backbone=0,
                 **kwargs):
        super(VideoSwinTransformer, self).__init__(**kwargs)
        self.swin_variant = swin_variant
        self.swin_pt = swin_pt
        self.vid_len = vid_len
        self.use_cls_token = use_cls_token
        self.patch_dim = patch_dim
        self.pos_encoding = pos_encoding
        self.dim = dim
        self.freeze_backbone = freeze_backbone

        # ckpt_name = os.path.join('pretrained', self.ckpt_name())
        # self.backbone = tf.keras.models.load_model(ckpt_name, compile=False)

        from architectures.videoswin import video_swin_base, video_swin_small, video_swin_tiny

        # if self.patch_dim == 0:
        #     self.patch_dim = self.vid_len

        self.pos_channels = self.vid_len // self.patch_dim

        verify = self.swin_pt == 2

        if swin_variant == 'b':
            video_swin = video_swin_base
        elif swin_variant == 's':
            video_swin = video_swin_small
        elif swin_variant == 't':
            video_swin = video_swin_tiny
        else:
            raise AssertionError(f'invalid swin_variant: {swin_variant}')

        self.backbone = video_swin(
            length=self.vid_len,
            patch_dim=self.patch_dim,
            pt=self.swin_pt,
            verify=verify
        )

        if self.freeze_backbone:
            self.backbone.trainable = False

        factor = 32.

        self.n_rows = math.ceil(image_height / factor)
        self.n_cols = math.ceil(image_width / factor)

        # self.n_rows = 24
        # self.n_cols = 32

        # self.backbone.build(input_shape=())
        # self.backbone.load_weights(ckpt_name)

        self.dropout = tf.keras.layers.Dropout(drop_units)
        self.stem_proj_swin = tf.keras.layers.Dense(dim, name='stem_proj_swin')
        self.stem_ln = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name='stem_ln')
        if self.use_cls_token:
            add_cls_token_emb(self, dim)

        self.vis_pos_emb = add_vis_pos_emb(
            self,
            self.pos_encoding,
            self.n_rows,
            self.n_cols,
            self.dim,
            n_channels=self.pos_channels,
            return_only=True,
        )
        self.transformer_encoder = TransformerEncoder(
            num_layers=num_layers,
            dim=dim,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            drop_path=drop_path,
            drop_units=drop_units,
            drop_att=drop_att,
            name='transformer_encoder')
        self.output_ln = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name='ouput_ln')

        self.built = False

    def ckpt_name(self):
        dataset = 'K400'  # K400, K600,, SSV2
        pretrained_dataset = 'IN1K'  # 'IN1K', 'IN22K', 'K400'
        size = 'B'  # 'T', 'S', 'B'

        # For K400, K600, window_size=(8,7,7)
        # For SSV2, window_size=(16,7,7)
        window_size = 877
        patch_size = 244
        num_frames = 32
        input_size = 224

        checkpoint_name = (
            f'TFVideoSwin{size}_'
            f'{dataset}_'
            f'{pretrained_dataset}_'
            f'P{patch_size}_'
            f'W{window_size}_'
            f'{num_frames}x{input_size}.h5'
        )
        return checkpoint_name

    def call(self, images, training):
        # if not self.built:
        #     model = self.backbone.build_graph()
        #     self.built=True

        tokens = self.backbone(images, training=training)

        bsz, n_img, h, w, num_channels = get_shape(tokens)
        tokens = tf.reshape(tokens, [bsz, n_img * h * w, num_channels])

        # tokens = tf.reshape(tokens, [bsz, h * w, num_channels])

        tokens = self.dropout(tokens, training)
        tokens = self.stem_proj_swin(tokens)
        tokens = self.stem_ln(tokens)

        tokens_vis_pos_emb = tf.expand_dims(self.vis_pos_emb, 0)
        tokens = tokens + tokens_vis_pos_emb

        if self.use_cls_token:
            cls_token = tf.tile(tf.expand_dims(self.cls_token_emb, 0), [bsz, 1, 1])
            tokens = tf.concat([cls_token, tokens], 1)

        tokens = self.transformer_encoder(
            tokens, None, training=training, ret_list=False)

        # tokens = utils.flatten_vid(tokens)
        x = self.output_ln(tokens)
        # x = utils.flatten_vid(x)

        return x
