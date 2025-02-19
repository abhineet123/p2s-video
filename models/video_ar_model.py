import ml_collections

import utils
import vocab
from tasks import task_utils
# from tasks.visualization import vis_utils

from architectures.transformers import add_vis_pos_emb, get_shape
from architectures.transformers import AutoregressiveDecoder
from architectures.transformers import MLP
from architectures.transformers import VisionTransformer

from architectures.video_transformers import VideoResNetTransformer, VideoSwinTransformer

from models import model as model_lib
from models import model_utils

import numpy as np
import tensorflow as tf


@model_lib.ModelRegistry.register('video_encoder_ar_decoder')
class Model(tf.keras.models.Model):
    """Inputs images and returns activations."""

    def __init__(self, config: ml_collections.ConfigDict, **kwargs):
        # vocab_size and max_seq_len don't include start token, which is only used
        # inside this class.
        super().__init__(**kwargs)
        self.config_all = config
        self.config = config.model
        self.vid_len = config.dataset.length
        self.late_fusion = self.config.late_fusion
        self.pos_encoding = self.config.pos_encoding
        self.is_swin = False

        self.freeze_backbone = self.config_all.train.freeze_backbone
        self.freeze_encoder = self.config_all.train.freeze_encoder
        self.freeze_decoder = self.config_all.train.freeze_decoder
        self.freeze_encoder_decoder = self.config_all.train.freeze_encoder_decoder

        if self.freeze_encoder_decoder:
            print('freezing both encoder and decoder')
        else:
            if self.freeze_decoder:
                print('freezing decoder')
            if self.freeze_encoder:
                print('freezing encoder')
            elif self.freeze_backbone:
                print('freezing backbone')

        if self.late_fusion:
            self.pos_channels = self.vid_len
        else:
            self.pos_channels = 1

        mlp_ratio = self.config.dim_mlp // self.config.dim_att
        if self.config.resnet_variant == 'swin':
            if self.config.swin_patch_dim == 0:
                self.config.swin_patch_dim = self.vid_len

            self.is_swin = True
            assert self.config.swin_patch_dim <= self.vid_len, "swin_patch_dim must be <= vid_len"
            self.pos_channels = self.vid_len // self.config.swin_patch_dim

            self.encoder = VideoSwinTransformer(
                swin_variant=self.config.swin_variant,
                swin_pt=self.config.swin_pt,
                patch_dim=self.config.swin_patch_dim,
                image_height=self.config.image_size[0],
                image_width=self.config.image_size[1],
                vid_len=self.vid_len,
                num_layers=self.config.num_encoder_layers,
                dim=self.config.dim_att,
                mlp_ratio=mlp_ratio,
                num_heads=self.config.num_heads,
                drop_path=self.config.drop_path,
                drop_units=self.config.drop_units,
                drop_att=self.config.drop_att,
                pos_encoding=self.config.pos_encoding,
                use_cls_token=self.config.use_cls_token,
                freeze_backbone=self.freeze_backbone,
                name='video_swin')
        elif self.config.resnet_variant == 'c1':
            raise NotImplemented('Video vision transformer is not implemented yet')
        else:
            self.encoder = VideoResNetTransformer(
                image_height=self.config.image_size[0],
                image_width=self.config.image_size[1],
                vid_len=self.vid_len,
                resnet_variant=self.config.resnet_variant,
                resnet_depth=self.config.resnet_depth,
                resnet_width_multiplier=self.config.resnet_width_multiplier,
                resnet_sk_ratio=self.config.resnet_sk_ratio,
                num_layers=self.config.num_encoder_layers,
                dim=self.config.dim_att,
                mlp_ratio=mlp_ratio,
                num_heads=self.config.num_heads,
                drop_path=self.config.drop_path,
                drop_units=self.config.drop_units,
                drop_att=self.config.drop_att,
                pos_encoding=self.config.pos_encoding,
                use_cls_token=self.config.use_cls_token,
                late_fusion=self.late_fusion,
                freeze_backbone=self.freeze_backbone,
                name='rest')


        if self.freeze_encoder or self.freeze_encoder_decoder:
            self.encoder.trainable = False

        mlp_ratio_dec = self.config.dim_mlp_dec // self.config.dim_att_dec
        self.proj = tf.keras.layers.Dense(
            self.config.dim_att_dec, name='proj/linear')
        self.proj_ln = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name='proj/ln')
        if self.config.dec_proj_mode in ['linear_p', 'mlp']:
            """
            add visual positional embedding
            """
            self.vis_pos_emb = add_vis_pos_emb(
                self,
                pos_encoding=self.pos_encoding,
                n_rows=self.encoder.n_rows,
                n_cols=self.encoder.n_cols,
                dim=self.config.dim_att_dec,
                n_channels=self.pos_channels,
                name_prefix='proj',
                return_only=True,
            )
            if self.late_fusion and self.pos_encoding == 'learned_3d':
                t, n_feat, fc = get_shape(self.vis_pos_emb)
                self.vis_pos_emb = tf.reshape(self.vis_pos_emb, [t * n_feat, fc])

            if self.config.dec_proj_mode == 'mlp':
                self.proj_mlp = MLP(1, self.config.dim_att_dec, mlp_ratio, self.config.drop_path,
                                    self.config.drop_units, name='proj/mlp')

        self.decoder = AutoregressiveDecoder(
            defer_vocab=self.config.defer_vocab,
            defer_seq=self.config.defer_seq,
            vocab_size=self.config.vocab_size,
            max_seq_len=self.config.max_seq_len,
            num_layers=self.config.num_decoder_layers,
            dim=self.config.dim_att_dec,
            mlp_ratio=mlp_ratio_dec,
            num_heads=self.config.num_heads_dec,
            drop_path=self.config.drop_path,
            drop_units=self.config.drop_units,
            drop_att=self.config.drop_att,
            pos_encoding=self.config.pos_encoding_dec,
            shared_embedding=self.config.shared_decoder_embedding,
            output_bias=self.config.decoder_output_bias,
            name='ar_decoder')

        if self.freeze_decoder or self.freeze_encoder_decoder:
            self.decoder.trainable = False

        self.is_inited = False
        self.trainable_modules = ['encoder', 'decoder', 'proj', 'proj_mlp']

    def _encode_videos(self, videos, training):
        config = self.config
        encoded = self.encoder(videos, training)
        # encoded = utils.flatten_vid(encoded)

        encoded = self.proj_ln(self.proj(encoded))
        # Add (optional) positional embedding to encoded visual units.
        if config.dec_proj_mode != 'linear':
            vis_pos_emb = tf.expand_dims(self.vis_pos_emb, 0)
            # vis_pos_emb = tf.expand_dims(vis_pos_emb, 0)
            # if self.pos_encoding == 'learned_3d':
            #     encoded = utils.unflatten_vid(encoded, self.vid_len)
            #     if config.use_cls_token:
            #         raise AssertionError('use_cls_token is not supported with 3-D positional encoding')
            #     else:
            #         encoded = encoded + vis_pos_emb
            #     encoded = utils.flatten_vid(encoded)
            # else:
            if config.use_cls_token:
                encoded = encoded + tf.concat(
                    [tf.zeros_like(vis_pos_emb[:, :1]), vis_pos_emb], 1)
            else:
                encoded = encoded + vis_pos_emb
            if config.dec_proj_mode == 'mlp':
                encoded = self.proj_mlp(encoded, training)
            else:
                assert config.dec_proj_mode == 'linear_p'
        # encoded = utils.unflatten_vid(encoded, self.vid_len)
        return encoded

    def call(self, videos, seq, training=True):
        with tf.name_scope(''):  # for other functions to have the same name scope.
            encoded = self._encode_videos(videos, training)

            """_tile_vis_output is only needed if seq is 3D or above"""
            # encoded, seq = self._tile_vis_output(encoded, seq)

            logits = self.decoder(seq, encoded, training)

            if not self.is_inited:
                model_utils.get_params_counts(self)
                self.is_inited = True

            return logits, encoded

    def infer(self, videos, prompt_seq, encoded=None, max_seq_len=None,
              temperature=1, top_k=1, top_p=1., num_samples=1,
              sampling_callback=None, training=False):
        if encoded is None:
            encoded = self._encode_videos(videos, training=training)

        """only needed if prompt_seq is 3D or above"""
        # encoded, prompt_seq = self._tile_vis_output(encoded, prompt_seq)

        """only needed if num_samples > 1"""
        # encoded = utils.tile_along_batch(encoded, num_samples)
        # prompt_seq = utils.tile_along_batch(prompt_seq, num_samples)

        """decoder forward pass for debugging"""
        # logits = self.decoder(prompt_seq, encoded, training)
        # pred_seq = tf.argmax(logits, axis=2)

        pred_seq, logits = self.decoder.infer(
            prompt_seq, encoded, max_seq_len,
            temperature, top_k, top_p, sampling_callback, training=training)

        return pred_seq, logits, encoded


@model_lib.TrainerRegistry.register('video_encoder_ar_decoder')
class VideoARTrainer(model_lib.Trainer):
    """A trainer for Video AR model."""

    def __init__(self, config: ml_collections.ConfigDict, **kwargs):
        """Init and setup basic training elements under strategy scope.

        Note: the trainer needs to be created under `strategy.scope()`.

        Args:
          config: object for holding hyperparameters and other configurations.
          **kwargs: other neccesary configurations to pass for training setup.
        """
        super().__init__(config, **kwargs)
        self.sample = None
        self.step = 0

        self.vid_len = config.dataset.length
        self._category_names = task_utils.get_category_names(
            config.dataset.get('category_names_path'))

        self._metrics.update({
            'loss_notpad': tf.keras.metrics.Mean('loss_notpad'),
            'accuracy_notpad': tf.keras.metrics.SparseCategoricalAccuracy(
                'accuracy_notpad'),
        })
        self._val_metrics.update({
            'loss_notpad': tf.keras.metrics.Mean('loss_notpad'),
            'correct_pc': tf.keras.metrics.Mean('correct_pc'),
            'accuracy_notpad': tf.keras.metrics.SparseCategoricalAccuracy(
                'accuracy_notpad'),
        })

    def sample_to_tb(self):
        self.step += 1
        for video_id, video in enumerate(self.sample):
            tf.summary.image(f'video {video_id}', video, self.step)

    def compute_loss(self, preprocess_outputs, validation):
        batched_examples, input_seq, target_seq, token_weights = preprocess_outputs

        videos = batched_examples['video']

        self.sample = videos

        model = self.model  # type:Model

        target_seq = utils.flatten_batch_dims(target_seq, out_rank=2)
        token_weights = utils.flatten_batch_dims(token_weights, out_rank=2)
        token_weights = utils.tf_float32(token_weights)
        is_padding = tf.equal(target_seq, vocab.PADDING_TOKEN)  # padding tokens.
        token_weights_notpad = tf.where(
            is_padding, tf.zeros_like(token_weights), token_weights)

        logits, pred_encoded = model(videos, input_seq)
        losses = model_utils.get_loss(
            logits, target_seq, self.config.train.loss_type)
        loss = tf.reduce_sum(losses * token_weights) / (
                tf.reduce_sum(token_weights) + 1e-9)
        loss_notpad = tf.reduce_sum(losses * token_weights_notpad) / (
                tf.reduce_sum(token_weights_notpad) + 1e-9)

        y_mask = tf.greater(token_weights_notpad, 0)

        y_true = tf.boolean_mask(target_seq, y_mask)
        y_pred_logits = tf.boolean_mask(logits, y_mask)

        # update metrics
        if validation:
            self._val_metrics['loss_notpad'].update_state(loss_notpad)
            self._val_metrics['accuracy_notpad'].update_state(y_true, y_pred_logits)
            y_mask = tf.greater(token_weights_notpad, 0)
            y_correct = model_utils.get_val_metrics(
                target_seq, logits, y_mask)
            self._val_metrics['correct_pc'].update_state(y_correct)
        else:
            self._metrics['loss_notpad'].update_state(loss_notpad)
            self._metrics['accuracy_notpad'].update_state(y_true, y_pred_logits)

            # if self.config.debug:
            #     bsz = tf.shape(videos)[0]
            #     prompt_seq = task_utils.build_prompt_seq_from_task_id(
            #         self.config.task.vocab_id,
            #         prompt_shape=(bsz, 1))
            #
            #     infer_seq, infer_logits, infer_encoded = model.infer(
            #         videos, prompt_seq, encoded=pred_encoded,
            #         max_seq_len=self.config.task.max_seq_len_test,
            #         temperature=self.config.task.temperature,
            #         top_k=self.config.task.top_k,
            #         top_p=self.config.task.top_p,
            #         training=True)
            #
            #     pred_encoded_np = pred_encoded.numpy()
            #     infer_encoded_np = infer_encoded.numpy()
            #
            #     model_utils.debug_loss(
            #         self.config, self._category_names, batched_examples, target_seq,
            #         logits, y_mask, y_pred=None, run_type='train',
            #         y_infer=infer_seq, y_infer_logits=infer_logits)

        return loss
