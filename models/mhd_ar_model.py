import cv2
import ml_collections

import utils
from architectures.transformers import add_vis_pos_emb
from architectures.transformers import AutoregressiveDecoder, AutoregressiveMHD
from architectures.transformers import MLP
from architectures.transformers import ResNetTransformer
from architectures.transformers import VisionTransformer
from models import model as model_lib
from models import model_utils
import tensorflow as tf
import numpy as np

from tasks import task_utils
import vocab


@model_lib.ModelRegistry.register('mhd_encoder_ar_decoder')
class MHDModel(tf.keras.models.Model):
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

        shared_decoder_params = dict(
            defer_vocab=config.defer_vocab,
            defer_seq=config.defer_seq,

            shared_xyl=config.shared_xyl,

            dim=config.dim_att_dec,
            mlp_ratio=mlp_ratio_dec,
            num_heads=config.num_heads_dec,
            drop_path=config.drop_path,
            drop_units=config.drop_units,
            drop_att=config.drop_att,
            pos_encoding=config.pos_encoding_dec,
            shared_embedding=config.shared_decoder_embedding,
            output_bias=config.decoder_output_bias,
            max_seq_len=config.max_seq_len,
            num_layers=config.num_decoder_layers,
        )

        assert config.coord_vocab_size > 0, "coord_vocab_size must be > 0"
        assert config.len_vocab_size > 0, "len_vocab_size must be > 0"
        assert config.class_vocab_size > 0, "class_vocab_size must be > 0"

        if config.shared_xyl:
            assert config.coord_vocab_size==config.len_vocab_size,\
                "coord_vocab_size and len_vocab_size must be same for shared_xyl"
            assert config.shared_mha, "shared_mha must be enabled for shared_xyl"

        self.trainable_modules = ['encoder', 'proj', 'proj_mlp']

        if config.shared_mha:
            self.decoder = AutoregressiveMHD(
                coord_vocab_size=config.coord_vocab_size,
                len_vocab_size=config.len_vocab_size,
                class_vocab_size=config.class_vocab_size,
                name=f'ar_decoder',
                **shared_decoder_params)
            self.trainable_modules.append('decoder')

            if self.freeze_decoder or self.freeze_encoder_decoder:
                self.decoder.trainable = False
        else:

            print('\n\nusing separate AutoregressiveDecoder for each token type\n\n')

            self.decoder_x = AutoregressiveDecoder(
                vocab_size=config.coord_vocab_size,
                name=f'ar_decoder_x',
                **shared_decoder_params)

            self.decoder_y = AutoregressiveDecoder(
                vocab_size=config.coord_vocab_size,
                name=f'ar_decoder_y',
                **shared_decoder_params)

            self.decoder_l = AutoregressiveDecoder(
                vocab_size=config.len_vocab_size,
                name=f'ar_decoder_l',
                **shared_decoder_params)

            self.decoder_c = AutoregressiveDecoder(
                vocab_size=config.class_vocab_size,
                name=f'ar_decoder_c',
                **shared_decoder_params)

            self.trainable_modules += [
                'decoder_x', 'decoder_y',
                'decoder_l', 'decoder_c',
            ]

            if self.freeze_decoder or self.freeze_encoder_decoder:
                self.decoder_x.trainable = False
                self.decoder_y.trainable = False
                self.decoder_l.trainable = False
                self.decoder_c.trainable = False

        self.is_inited = False

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
            if vis_output is not None:
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

    def call(self, images, seq_x, seq_y, seq_l, seq_c,
             training=True):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
        with tf.name_scope(''):  # for other functions to have the same name scope.
            encoded = self._encode_images(images, training)

            if self.config.shared_mha:
                encoded, seq_x = self._tile_vis_output(encoded, seq_x)
                _, seq_y = self._tile_vis_output(None, seq_y)
                _, seq_l = self._tile_vis_output(None, seq_l)
                _, seq_c = self._tile_vis_output(None, seq_c)

                logits_x, logits_y, logits_l, logits_c = self.decoder(seq_x, seq_y, seq_l, seq_c, encoded, training)
            else:
                encoded_x, seq_x = self._tile_vis_output(encoded, seq_x)
                logits_x = self.decoder_x(seq_x, encoded_x, training)

                encoded_y, seq_y = self._tile_vis_output(encoded, seq_y)
                logits_y = self.decoder_y(seq_y, encoded_y, training)

                encoded_l, seq_l = self._tile_vis_output(encoded, seq_l)
                logits_l = self.decoder_y(seq_l, encoded_l, training)

                encoded_c, seq_c = self._tile_vis_output(encoded, seq_c)
                logits_c = self.decoder_y(seq_c, encoded_c, training)

            if not self.is_inited:
                model_utils.get_params_counts(self)
                self.is_inited = True

            return logits_x, logits_y, logits_l, logits_c

    def infer(self, images, prompt_seqs, encoded=None, max_seq_len=None,
              temperature=1, top_k=1, top_p=1., num_samples=1,
              sampling_callback=None):

        prompt_seq_x, prompt_seq_y, prompt_seq_l, prompt_seq_c = prompt_seqs

        if encoded is None:
            encoded = self._encode_images(images, training=False)

        if prompt_seq_x.shape.rank > 2:
            tile_factor = prompt_seq_x.shape.as_list()[-2]
            encoded = utils.tile_along_batch(encoded, tile_factor)

            prompt_seq_x = utils.flatten_batch_dims(prompt_seq_x, out_rank=2)
            prompt_seq_y = utils.flatten_batch_dims(prompt_seq_y, out_rank=2)
            prompt_seq_l = utils.flatten_batch_dims(prompt_seq_l, out_rank=2)
            prompt_seq_c = utils.flatten_batch_dims(prompt_seq_c, out_rank=2)

        encoded = utils.tile_along_batch(encoded, num_samples)
        prompt_seq_x = utils.tile_along_batch(prompt_seq_x, num_samples)
        prompt_seq_y = utils.tile_along_batch(prompt_seq_y, num_samples)
        prompt_seq_l = utils.tile_along_batch(prompt_seq_l, num_samples)
        prompt_seq_c = utils.tile_along_batch(prompt_seq_c, num_samples)

        if self.config.shared_mha:
            pred_seqs, logits = self.decoder.infer(
                [prompt_seq_x, prompt_seq_y, prompt_seq_l, prompt_seq_c],
                encoded, max_seq_len,
                temperature, top_k, top_p, sampling_callback)
            pred_seq_x, pred_seq_y, pred_seq_l, pred_seq_c = pred_seqs
            logits_x, logits_y, logits_l, logits_c = logits
        else:

            pred_seq_x, logits_x = self.decoder_x.infer(
                prompt_seq_x, encoded, max_seq_len,
                temperature, top_k, top_p, sampling_callback)

            pred_seq_y, logits_y = self.decoder_y.infer(
                prompt_seq_y, encoded, max_seq_len,
                temperature, top_k, top_p, sampling_callback)

            pred_seq_l, logits_l = self.decoder_l.infer(
                prompt_seq_l, encoded, max_seq_len,
                temperature, top_k, top_p, sampling_callback)

            pred_seq_c, logits_c = self.decoder_c.infer(
                prompt_seq_c, encoded, max_seq_len,
                temperature, top_k, top_p, sampling_callback)

        return ([pred_seq_x, pred_seq_y, pred_seq_l, pred_seq_c],
                [logits_x, logits_y, logits_l, logits_c],
                encoded)


@model_lib.TrainerRegistry.register('mhd_encoder_ar_decoder')
class MHDARTrainer(model_lib.Trainer):
    """A trainer for MHD AR model."""

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

        self.mhd_loss_weights = config.train.mhd_loss_weights
        self.mhd_loss_weights = list(map(int, list(self.mhd_loss_weights)))

        self.mhd_loss_weights_sum = sum(self.mhd_loss_weights)

        self._category_names = task_utils.get_category_names(
            config.dataset.get('category_names_path'))

        self.class_id_to_col, self.class_id_to_name = task_utils.get_class_info(self._category_names)

        for metrics in (self._metrics, self._val_metrics):
            metrics.update({
                'correct_pc_x': tf.keras.metrics.Mean('correct_pc_x'),
                'correct_pc_y': tf.keras.metrics.Mean('correct_pc_y'),
                'correct_pc_l': tf.keras.metrics.Mean('correct_pc_l'),
                'correct_pc_c': tf.keras.metrics.Mean('correct_pc_c'),

                'loss_notpad': tf.keras.metrics.Mean('loss_notpad'),
                'loss_notpad_x': tf.keras.metrics.Mean('loss_notpad_x'),
                'loss_notpad_y': tf.keras.metrics.Mean('loss_notpad_y'),
                'loss_notpad_l': tf.keras.metrics.Mean('loss_notpad_l'),
                'loss_notpad_c': tf.keras.metrics.Mean('loss_notpad_c'),

                'accuracy_notpad': tf.keras.metrics.SparseCategoricalAccuracy('accuracy_notpad'),
                'accuracy_notpad_x': tf.keras.metrics.SparseCategoricalAccuracy('accuracy_notpad_x'),
                'accuracy_notpad_y': tf.keras.metrics.SparseCategoricalAccuracy('accuracy_notpad_y'),
                'accuracy_notpad_l': tf.keras.metrics.SparseCategoricalAccuracy('accuracy_notpad_l'),
                'accuracy_notpad_c': tf.keras.metrics.SparseCategoricalAccuracy('accuracy_notpad_c'),
            })

    def mhd_to_rle(self, rle_x, rle_y, rle_l, rle_c):

        rle_x = rle_x[rle_x != vocab.PADDING_TOKEN]
        rle_y = rle_y[rle_y != vocab.PADDING_TOKEN]
        rle_l = rle_l[rle_l != vocab.PADDING_TOKEN]
        rle_c = rle_c[rle_c != vocab.PADDING_TOKEN]

        min_rle_seq_len = min(rle_x.size, rle_y.size, rle_l.size, rle_c.size)

        rle_x = rle_x[:min_rle_seq_len]
        rle_y = rle_y[:min_rle_seq_len]
        rle_l = rle_l[:min_rle_seq_len]
        rle_c = rle_c[:min_rle_seq_len]

        rle_ = np.stack((rle_x, rle_y, rle_l, rle_c), axis=1)
        rle_ = rle_.flatten()

        return rle_

    def sample_to_tb(self):
        self.step += 1
        images = self.examples['image']
        tf.summary.image(f'images', images, self.step)

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
            assert max_length > 0, "max_length must be > 0"

            subsample = mode_cfg.subsample
            # assert subsample >= 1, "subsample must be >= 1"
            multi_class = self.config.dataset.multi_class
            assert multi_class, "mhd is only implemented for multi_class case"

            flat_order = self.config.dataset.flat_order

            length_as_class = self.config.dataset.length_as_class
            assert not length_as_class, "mhd is not implemented for length_as_class"

            starts_2d = self.config.dataset.starts_2d
            assert starts_2d, "mhd is only implemented for starts_2d"

            starts_offset = self.config.model.coord_vocab_shift
            lengths_offset = self.config.model.len_vocab_shift
            class_offset = self.config.model.class_vocab_shift

            if subsample > 1:
                max_length = int(max_length / subsample)
                n_rows, n_cols = int(n_rows / subsample), int(n_cols / subsample)

            n_classes = len(self.class_id_to_name)

            masks_vis = []
            masks_rec_vis = []
            pred_masks_rec_vis = []
            for batch_id in range(batch_size):
                mask = masks[batch_id].numpy().squeeze()
                image = images[batch_id].numpy()

                pred_rle_tokens = self.mhd_to_rle(*(k[batch_id].numpy() for k in self.y_pred))
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
                    max_seq_len=None,
                    diff_mask=False,
                    n_classes=n_classes,
                )
                pred_mask_rec_vis = task_utils.mask_id_to_vis_bgr(pred_mask_rec, self.class_id_to_col)
                pred_mask_rec_vis = task_utils.resize_mask(pred_mask_rec_vis, (img_h, img_w))
                pred_masks_rec_vis.append(pred_mask_rec_vis)

                true_rle_tokens = self.mhd_to_rle(*(k[batch_id].numpy() for k in self.y_true))
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
                    diff_mask=False,
                    n_classes=n_classes,
                )
                mask_vis = task_utils.mask_id_to_vis_bgr(mask, self.class_id_to_col)
                masks_vis.append(mask_vis)

                mask_rec_vis = task_utils.mask_id_to_vis_bgr(mask_rec, self.class_id_to_col)
                mask_rec_vis = task_utils.resize_mask(mask_rec_vis, (img_h, img_w))
                masks_rec_vis.append(mask_rec_vis)

                image = (image * 255.).astype(np.uint8)
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

    def get_loss(self, logits, target_seq, token_weights):
        target_seq = utils.flatten_batch_dims(target_seq, out_rank=2)
        token_weights = utils.flatten_batch_dims(token_weights, out_rank=2)
        token_weights = utils.tf_float32(token_weights)

        losses = model_utils.get_loss(
            logits, target_seq, self.config.train.loss_type)

        loss = tf.reduce_sum(losses * token_weights) / (
                tf.reduce_sum(token_weights) + 1e-9)

        is_padding = tf.equal(target_seq, 0)  # padding tokens.
        token_weights_notpad = tf.where(
            is_padding, tf.zeros_like(token_weights), token_weights)
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

        return loss, loss_notpad, y_correct, y_true_unbatched, y_pred_logits_unbatched

    def compute_loss(self, preprocess_outputs, validation):
        """Compute loss based on model outputs and targets."""
        examples, mhd_input_seq, mhd_target_seq, mhd_token_weight = preprocess_outputs

        input_seq_x, input_seq_y, input_seq_l, input_seq_c = mhd_input_seq
        target_seq_x, target_seq_y, target_seq_l, target_seq_c = mhd_target_seq
        token_weights_x, token_weights_y, token_weights_l, token_weights_c = mhd_token_weight

        image = examples["image"]
        logits_x, logits_y, logits_l, logits_c = self.model(image, input_seq_x, input_seq_y, input_seq_l, input_seq_c)

        loss_x, loss_notpad_x, y_correct_x, y_true_unbatched_x, y_pred_logits_unbatched_x = self.get_loss(
            logits_x, target_seq_x, token_weights_x)
        loss_y, loss_notpad_y, y_correct_y, y_true_unbatched_y, y_pred_logits_unbatched_y = self.get_loss(
            logits_y, target_seq_y, token_weights_y)
        loss_l, loss_notpad_l, y_correct_l, y_true_unbatched_l, y_pred_logits_unbatched_l = self.get_loss(
            logits_l, target_seq_l, token_weights_l)
        loss_c, loss_notpad_c, y_correct_c, y_true_unbatched_c, y_pred_logits_unbatched_c = self.get_loss(
            logits_c, target_seq_c, token_weights_c)

        loss_weight_x, loss_weight_y, loss_weight_l, loss_weight_c = self.mhd_loss_weights

        loss = ((loss_x * loss_weight_x +
                 loss_y * loss_weight_y +
                 loss_l * loss_weight_l +
                 loss_c * loss_weight_c) /
                self.mhd_loss_weights_sum)

        loss_notpad = ((loss_notpad_x * loss_weight_x +
                        loss_notpad_y * loss_weight_y +
                        loss_notpad_l * loss_weight_l +
                        loss_notpad_c * loss_weight_c) /
                       self.mhd_loss_weights_sum)

        if self.config.debug:
            self.examples = examples
            self.loss = loss
            self.loss_notpad = loss_notpad

            self.y_true = (target_seq_x, target_seq_y, target_seq_l, target_seq_c)
            self.y_correct = (y_correct_x, y_correct_y, y_correct_l, y_correct_c)
            self.y_pred = tuple(tf.argmax(logits, axis=2) for logits in (logits_x, logits_y, logits_l, logits_c))

        metrics = self._val_metrics if validation else self._metrics
        metrics['loss_notpad'].update_state(loss_notpad)

        metrics['loss_notpad_x'].update_state(loss_notpad_x)
        metrics['loss_notpad_y'].update_state(loss_notpad_y)
        metrics['loss_notpad_l'].update_state(loss_notpad_l)
        metrics['loss_notpad_c'].update_state(loss_notpad_c)

        metrics['accuracy_notpad_x'].update_state(
            y_true_unbatched_x,
            y_pred_logits_unbatched_x,
        )
        metrics['accuracy_notpad_y'].update_state(
            y_true_unbatched_y,
            y_pred_logits_unbatched_y,
        )
        metrics['accuracy_notpad_l'].update_state(
            y_true_unbatched_l,
            y_pred_logits_unbatched_l,
        )
        metrics['accuracy_notpad_c'].update_state(
            y_true_unbatched_c,
            y_pred_logits_unbatched_c,
        )
        metrics['correct_pc_x'].update_state(y_correct_x)
        metrics['correct_pc_y'].update_state(y_correct_y)
        metrics['correct_pc_l'].update_state(y_correct_l)
        metrics['correct_pc_c'].update_state(y_correct_c)

        return loss
