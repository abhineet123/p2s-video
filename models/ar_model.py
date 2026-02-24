import cv2
import ml_collections

import utils
from architectures.transformers import add_vis_pos_emb
from architectures.transformers import AutoregressiveDecoder
from architectures.transformers import FIT
from architectures.transformers import MLP
from architectures.transformers import ResNetTransformer
from architectures.transformers import VisionTransformer
from models import model as model_lib
from models import model_utils
import tensorflow as tf
import numpy as np

from tasks import task_utils
import vocab

@model_lib.ModelRegistry.register('encoder_ar_decoder')
class Model(tf.keras.models.Model):
    """Inputs images and returns activations."""

    def __init__(self, config: ml_collections.ConfigDict, **kwargs):
        # vocab_size and max_seq_len don't include start token, which is only used
        # inside this class.
        super().__init__(**kwargs)
        self.config_all = config
        config = config.model
        self.config = config

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

        mlp_ratio = config.dim_mlp // config.dim_att
        if config.resnet_variant == 'c1':
            self.encoder = VisionTransformer(
                config.image_size[0], config.image_size[1], config.patch_size,
                config.num_encoder_layers, config.dim_att, mlp_ratio,
                config.num_heads, config.drop_path, config.drop_units,
                config.drop_att, config.pos_encoding, config.use_cls_token,
                freeze_backbone=self.freeze_backbone,
                name='vit')
        else:
            self.encoder = ResNetTransformer(
                image_height=config.image_size[0],
                image_width=config.image_size[1],
                resnet_variant=config.resnet_variant,
                resnet_depth=config.resnet_depth,
                resnet_width_multiplier=config.resnet_width_multiplier,
                resnet_sk_ratio=config.resnet_sk_ratio,
                num_layers=config.num_encoder_layers,
                dim=config.dim_att,
                mlp_ratio=mlp_ratio,
                num_heads=config.num_heads,
                drop_path=config.drop_path,
                drop_units=config.drop_units,
                drop_att=config.drop_att,
                pos_encoding=config.pos_encoding,
                use_cls_token=config.use_cls_token,
                freeze_backbone=self.freeze_backbone,
                name='rest')

        if self.freeze_encoder or self.freeze_encoder_decoder:
            self.encoder.trainable = False

        mlp_ratio_dec = config.dim_mlp_dec // config.dim_att_dec
        self.proj = tf.keras.layers.Dense(
            config.dim_att_dec, name='proj/linear')
        self.proj_ln = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name='proj/ln')
        if config.dec_proj_mode in ['linear_p', 'mlp']:
            """
            add visual positional embedding
            """
            add_vis_pos_emb(
                self, config.pos_encoding,
                self.encoder.n_rows,
                self.encoder.n_cols,
                config.dim_att_dec,
                name_prefix='proj')

            if config.dec_proj_mode == 'mlp':
                self.proj_mlp = MLP(1, config.dim_att_dec, mlp_ratio, config.drop_path,
                                    config.drop_units, name='proj/mlp')

        self.decoder = AutoregressiveDecoder(
            defer_vocab=config.defer_vocab,
            defer_seq=self.config.defer_seq,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            num_layers=config.num_decoder_layers,
            dim=config.dim_att_dec,
            mlp_ratio=mlp_ratio_dec,
            num_heads=config.num_heads_dec,
            drop_path=config.drop_path,
            drop_units=config.drop_units,
            drop_att=config.drop_att,
            pos_encoding=config.pos_encoding_dec,
            shared_embedding=config.shared_decoder_embedding,
            output_bias=config.decoder_output_bias,
            name='ar_decoder')

        if self.freeze_decoder or self.freeze_encoder_decoder:
            self.decoder.trainable = False

        self.is_inited = False
        self.trainable_modules = ['encoder', 'decoder', 'proj', 'proj_mlp']

    def _tile_vis_output(self, vis_output, seq):
        """Tile vis_output per seq.

        Args:
          vis_output: `float` tensor of encoded images in shape of (bsz, ....).
          seq: `int` sequence in shape of (bsz, seqlen),
            or (bsz, instances, seqlen) if there are multiple sequences per image.

        Returns:
          vis_output of (bsz*instances, ...).
          seq of (bsz*instances, ...).
        """
        if seq.shape.rank > 2:
            tile_factor = seq.shape.as_list()[-2]
            vis_output = utils.tile_along_batch(vis_output, tile_factor)
            seq = utils.flatten_batch_dims(seq, out_rank=2)
        return vis_output, seq

    def _encode_images(self, images, training):
        """Encode images into latents for decoder to condition on."""
        config = self.config
        encoded = self.encoder(images, training)
        encoded = self.proj_ln(self.proj(encoded))
        # Add (optional) positional embedding to encoded visual units.
        if config.dec_proj_mode != 'linear':
            vis_pos_emb = tf.expand_dims(self.vis_pos_emb, 0)
            if config.use_cls_token:
                encoded = encoded + tf.concat(
                    [tf.zeros_like(vis_pos_emb[:, :1]), vis_pos_emb], 1)
            else:
                encoded = encoded + vis_pos_emb
            if config.dec_proj_mode == 'mlp':
                encoded = self.proj_mlp(encoded, training)
            else:
                assert config.dec_proj_mode == 'linear_p'
        return encoded

    def call(self, images, seq,
             training=True):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
        """Model function call for *training*.

        Args:
          images: `float` tensor of (bsz, h, w, c).
          seq: `int` sequence visible to the model of shape (bsz, seqlen),
            or (bsz, instances, seqlen) if there are multiple sequences per image.
          training: `bool` indicator.

        Returns:
          logits for each predicted tokens of (bsz * instances, seqlen, vocab_size).
        """
        with tf.name_scope(''):  # for other functions to have the same name scope.
            encoded = self._encode_images(images, training)
            encoded, seq = self._tile_vis_output(encoded, seq)
            logits = self.decoder(seq, encoded, training)

            if not self.is_inited:
                model_utils.get_params_counts(self)
                self.is_inited = True

            return logits

    def infer(self, images, prompt_seq, encoded=None, max_seq_len=None,
              temperature=1, top_k=1, top_p=1., num_samples=1,
              sampling_callback=None):
        """Model function call for inference.

        Args:
          images: `float` tensor of (bsz, h, w, c).
          prompt_seq: `int` sequence visible to the model of shape (bsz, seqlen),
            or (bsz, instances, seqlen) if there are multiple sequences per image.
          encoded: cache for encoded images for decoder. Skip image encoding if this
            is given.
          max_seq_len: `int` of max generated sequence length (including prompt).
          temperature: `float` scalar for scaling the logits before sampling.
          top_k: `int` scalar for truncating top-k tokens according to logits before
            token sampling.
          top_p: `float` scalar specifying the threshold of cumulative probablity
            for truncating tokens before token sampling.
          num_samples: `int` number of samples to be generated for each instance.
          sampling_callback: a callbak `function` that take `next_logits`, and
            return `next_token`. This is used when users need a specific logic
            for sampling. Default to `None` with standard free-form sampling.

        Returns:
          pred_seq: `int` prediction sequence of shape
              (bsz * instances * num_samples, seqlen)
          logits: `float` of shape
              (bsz * instances * num_samples, seqlen, vocab_size)
          encoded: `float` tensor of encoded images.
        """
        if encoded is None:
            encoded = self._encode_images(images, training=False)
        encoded, prompt_seq = self._tile_vis_output(encoded, prompt_seq)

        # Tile by num_samples too.
        encoded = utils.tile_along_batch(encoded, num_samples)
        prompt_seq = utils.tile_along_batch(prompt_seq, num_samples)

        pred_seq, logits = self.decoder.infer(
            prompt_seq, encoded, max_seq_len,
            temperature, top_k, top_p, sampling_callback)

        return pred_seq, logits, encoded


@model_lib.TrainerRegistry.register('encoder_ar_decoder')
class ARTrainer(model_lib.Trainer):
    """A trainer for AR model."""

    def __init__(self, config: ml_collections.ConfigDict, **kwargs):
        """Init and setup basic training elements under strategy scope.

        Note: the trainer needs to be created under `strategy.scope()`.

        Args:
          config: object for holding hyperparameters and other configurations.
          **kwargs: other neccesary configurations to pass for training setup.
        """
        super().__init__(config, **kwargs)
        self.examples = None
        self.y_pred = None
        self.y_true = None
        self.loss = None
        self.loss_notpad = None
        self.y_correct = None
        self.step = 0

        self._category_names = task_utils.get_category_names(
            config.dataset.get('category_names_path'))

        self.class_id_to_col, self.class_id_to_name = task_utils.get_class_info(self._category_names)

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
        images = self.examples['image']
        tf.summary.image(f'images', images, self.step)

        n_pred_non_zeros = tf.math.count_nonzero(tf.not_equal(self.y_pred, 0))
        n_gt_non_zeros = tf.math.count_nonzero(tf.not_equal(self.y_true, 0))

        tf.summary.scalar(f'iter/loss', self.loss, self.step)
        tf.summary.scalar(f'iter/loss_notpad', self.loss_notpad, self.step)
        tf.summary.scalar(f'iter/y_correct', self.y_correct, self.step)
        tf.summary.scalar(f'iter/n_pred_non_zeros', n_pred_non_zeros, self.step)
        tf.summary.scalar(f'iter/n_gt_non_zeros', n_gt_non_zeros, self.step)

        try:
            masks = self.examples['mask']
        except KeyError:
            pass
        else:
            # return
            batch_size, n_rows, n_cols = images.shape[:3]
            img_h, img_w = n_rows, n_cols

            mode_cfg = self.config.dataset.train

            max_length = mode_cfg.max_length
            subsample = mode_cfg.subsample

            assert max_length > 0, "max_length must be > 0"
            # assert subsample >= 1, "subsample must be >= 1"

            rle_from_mask = self.config.dataset.rle_from_mask
            target_size = self.config.dataset.target_size
            instance_wise = self.config.dataset.instance_wise
            class_wise = self.config.dataset.class_wise
            multi_class = self.config.dataset.multi_class

            flat_order = self.config.dataset.flat_order
            length_as_class = self.config.dataset.length_as_class
            diff_mask = self.config.dataset.diff_mask
            starts_2d = self.config.dataset.starts_2d

            starts_offset = self.config.model.coord_vocab_shift
            lengths_offset = self.config.model.len_vocab_shift
            class_offset = self.config.model.class_vocab_shift

            n_classes = len(self.class_id_to_name)

            if subsample > 1:
                max_length = int(max_length / subsample)
                n_rows, n_cols = int(n_rows / subsample), int(n_cols / subsample)

            masks_vis = []
            masks_rec_vis = []
            pred_masks_rec_vis = []
            for batch_id in range(batch_size):
                mask = masks[batch_id].numpy().squeeze()
                image = images[batch_id].numpy()

                pred_rle_tokens = self.y_pred[batch_id].numpy()

                pred_rle_tokens = pred_rle_tokens[pred_rle_tokens != vocab.PADDING_TOKEN]
                pred_mask_rec, _ = task_utils.mask_from_tokens(
                    pred_rle_tokens,
                    (n_rows, n_cols),
                    allow_extra=True,
                    length_as_class=length_as_class,
                    max_length=max_length,
                    starts_offset=starts_offset,
                    lengths_offset=lengths_offset,
                    class_offset=class_offset,
                    starts_2d=starts_2d,
                    multi_class=multi_class,
                    flat_order=flat_order,
                    ignore_invalid=True,
                    diff_mask=diff_mask,
                    max_seq_len=None,
                    n_classes=n_classes,
                )
                pred_mask_rec_vis = task_utils.mask_id_to_vis_bgr(pred_mask_rec, self.class_id_to_col)
                pred_mask_rec_vis = task_utils.resize_mask(pred_mask_rec_vis, (img_h, img_w))
                pred_masks_rec_vis.append(pred_mask_rec_vis)

                true_rle_tokens = self.y_true[batch_id].numpy()
                true_rle_tokens = true_rle_tokens[true_rle_tokens != vocab.PADDING_TOKEN]
                mask_rec, rle_rec_cmp = task_utils.mask_from_tokens(
                    true_rle_tokens,
                    (n_rows, n_cols),
                    allow_extra=False,
                    length_as_class=length_as_class,
                    max_length=max_length,
                    starts_offset=starts_offset,
                    lengths_offset=lengths_offset,
                    class_offset=class_offset,
                    starts_2d=starts_2d,
                    multi_class=multi_class,
                    flat_order=flat_order,
                    ignore_invalid=False,
                    max_seq_len=None,
                    diff_mask=diff_mask,
                    n_classes=n_classes,
                )
                mask_vis = task_utils.mask_id_to_vis_bgr(mask, self.class_id_to_col)
                masks_vis.append(mask_vis)

                mask_rec_vis = task_utils.mask_id_to_vis_bgr(mask_rec, self.class_id_to_col)
                mask_rec_vis = task_utils.resize_mask(mask_rec_vis, (img_h, img_w))
                masks_rec_vis.append(mask_rec_vis)

                image = (image*255.).astype(np.uint8)
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # print('\npress any key to continue\n')

                image_vis = np.concatenate((image_bgr, mask_vis, mask_rec_vis, pred_mask_rec_vis), axis=1)
                image_vis = task_utils.resize_ar(image_vis, width=1800)
                cv2.imshow('image_vis', image_vis)
                cv2.waitKey(1000)

            pred_masks_rec_vis = np.stack(pred_masks_rec_vis, axis=0)
            pred_masks_rec_vis = tf.convert_to_tensor(pred_masks_rec_vis)

            masks_rec_vis = np.stack(masks_rec_vis, axis=0)
            masks_rec_vis = tf.convert_to_tensor(masks_rec_vis)

            masks_vis = np.stack(masks_vis, axis=0)
            masks_vis = tf.convert_to_tensor(masks_vis)

            tf.summary.image(f'masks', masks_vis, self.step)
            tf.summary.image(f'masks_rec', masks_rec_vis, self.step)
            tf.summary.image(f'pred_masks_rec', pred_masks_rec_vis, self.step)

    def compute_loss(self, preprocess_outputs, validation):
        """Compute loss based on model outputs and targets."""
        examples, input_seq, target_seq, token_weights = preprocess_outputs

        target_seq = utils.flatten_batch_dims(target_seq, out_rank=2)
        token_weights = utils.flatten_batch_dims(token_weights, out_rank=2)
        token_weights = utils.tf_float32(token_weights)

        is_padding = tf.equal(target_seq, 0)  # padding tokens.
        token_weights_notpad = tf.where(
            is_padding, tf.zeros_like(token_weights), token_weights)

        image = examples["image"]
        logits = self.model(image, input_seq)
        losses = model_utils.get_loss(
            logits, target_seq, self.config.train.loss_type)
        loss = tf.reduce_sum(losses * token_weights) / (
                tf.reduce_sum(token_weights) + 1e-9)
        loss_notpad = tf.reduce_sum(losses * token_weights_notpad) / (
                tf.reduce_sum(token_weights_notpad) + 1e-9)

        # update metrics
        y_mask = tf.greater(token_weights_notpad, 0)
        """
        flatten tokens from all batch samples into a single sequence
        each sequence might have different length so maintaining a 
        batch-wise shape is not possible
        """
        y_true_unbatched = tf.boolean_mask(target_seq, y_mask)
        y_pred_logits_unbatched = tf.boolean_mask(logits, y_mask)
        y_mask = tf.greater(token_weights_notpad, 0)
        y_correct = model_utils.get_val_metrics(
            target_seq, logits, y_mask)


        if self.config.debug:
            self.examples = examples
            self.loss = loss
            self.loss_notpad = loss_notpad

            self.y_true = target_seq
            self.y_correct = y_correct
            self.y_pred = tf.argmax(logits, axis=2)

        if validation:
            self._val_metrics['loss_notpad'].update_state(loss_notpad)
            self._val_metrics['accuracy_notpad'].update_state(
                y_true_unbatched,
                y_pred_logits_unbatched,
            )
            self._val_metrics['correct_pc'].update_state(y_correct)
        else:
            self._metrics['loss_notpad'].update_state(loss_notpad)
            self._metrics['accuracy_notpad'].update_state(
                y_true_unbatched,
                y_pred_logits_unbatched,
            )
            # if self.config.debug:
            #     from tasks.visualization import vis_utils
            #     vis_utils.debug_loss(
            #         self.config, self._category_names, examples, target_seq,
            #         logits, y_mask, y_pred=None, run_type='train',is_video=False)

            # # type error
            # if self.config.debug:
            #     from .model_utils import debug_loss
            #     debug_loss(
            #         self.config, self._category_names, examples, target_seq,
            #         logits, y_mask, y_pred=None, run_type='train',is_video=False)

        return loss


@model_lib.ModelRegistry.register('fit_encoder_ar_decoder')
class ModelT(Model):
    """Inputs images and returns activations."""

    def __init__(self, config: ml_collections.ConfigDict, **kwargs):
        super(Model, self).__init__(**kwargs)
        config = config.model
        self.config = config
        self.drate = config.glimpse_div_rate
        self.mini_x = config.glimpse_mini_x
        num_groups = self.drate ** 2 + self.mini_x
        x_per_group = (config.image_size[0] // self.drate // config.patch_size) ** 2
        self.encoder = FIT(
            layers=config.layers,
            x_size=x_per_group * num_groups,
            num_groups=num_groups,
            latents_per_group=config.latents_per_group,
            x_dim=config.dim_att,
            latent_dim=config.dim_latent,
            x_num_heads=config.num_heads,
            latent_num_heads=config.num_heads,
            mlp_ratio=config.dim_mlp // config.dim_att,
            drop_path=config.drop_path,
            drop_units=config.drop_units,
            drop_att=config.drop_att,
            x_pos_encoding=config.pos_encoding,
            latent_pos_encoding=config.latent_pos_encoding,
            mask=config.mask)

        mlp_ratio_dec = config.dim_mlp_dec // config.dim_att_dec
        self.decoder = AutoregressiveDecoder(
            defer_vocab=config.defer_vocab,
            defer_seq=self.config.defer_seq,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            num_layers=config.num_decoder_layers,
            dim=config.dim_att_dec,
            mlp_ratio=mlp_ratio_dec,
            num_heads=config.num_heads_dec,
            drop_path=config.drop_path,
            drop_units=config.drop_units,
            drop_att=config.drop_att,
            pos_encoding=config.pos_encoding_dec,
            shared_embedding=config.shared_decoder_embedding,
            output_bias=config.decoder_output_bias,
            name='ar_decoder')

    def _encode_images(self, images, training):
        """Encode images into latents for decoder to condition on."""
        config = self.config
        sub_isize = [config.image_size[0] // self.drate,
                     config.image_size[1] // self.drate]
        patch_size = [config.patch_size, config.patch_size]
        if config.shuffle_glimpses:
            images, idx = utils.images2glimpses2tokens(
                images, sub_isize, patch_size, mini_x=self.mini_x, shuffle=True)
        else:
            images, idx = utils.images2glimpses2tokens(
                images, sub_isize, patch_size, mini_x=self.mini_x, shuffle=False)
            idx = None
        x_tokens, l_tokens = self.encoder(images, idx, training)
        if self.encoder.layer_configs[-1][-1] > 0:
            # using latent tokens only when the network ends with global layer(s).
            encoded = l_tokens
        else:
            encoded = x_tokens
        bsz, seqlen, slots, dim = utils.shape_as_list(encoded)
        encoded = tf.reshape(encoded, [bsz, seqlen * slots, dim])
        return encoded


"""
fancy way of saying TrainerRegistry['fit_encoder_ar_decoder'] = ARTrainer
Separate line here since both fit_encoder_ar_decoder And encoder_ar_decoder 
have the same trainer class
"""
model_lib.TrainerRegistry.register('fit_encoder_ar_decoder')(ARTrainer)
