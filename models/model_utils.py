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
"""Modeling related utils."""

import math
import re
import os
import tensorflow as tf
import tensorflow_addons as tfa

from utils import linux_path


def check_ckpt_match(model_dir, model, ckpt_vars_p):
    import shutil
    import pandas as pd

    from operator import mul
    from functools import reduce

    temp_model_dir = linux_path(model_dir, "temp")
    os.makedirs(temp_model_dir, exist_ok=True)
    checkpoint = tf.train.Checkpoint(model=model)
    temp_checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, temp_model_dir, 1)
    temp_checkpoint_manager.save(0)
    latest_ckpt = tf.train.latest_checkpoint(temp_model_dir)
    curr_ckpt_vars = tf.train.list_variables(latest_ckpt)
    curr_ckpt_dict = {
        ckpt_var[0]: ckpt_var[1] for ckpt_var in curr_ckpt_vars
    }
    pt_ckpt_vars_all = ckpt_vars_p

    pt_ckpt_vars = [k for k in pt_ckpt_vars_all if 'optimizer' not in k[0]]
    pt_ckpt_dict = {
        ckpt_var[0]: ckpt_var[1] for ckpt_var in pt_ckpt_vars
    }
    pt_ckpt_names = set(pt_ckpt_dict.keys())
    curr_ckpt_names = set(curr_ckpt_dict.keys())

    not_in_model = list(pt_ckpt_names - curr_ckpt_names)
    not_in_ckpt = list(curr_ckpt_names - pt_ckpt_names)

    not_in_model = [(reduce(mul, pt_ckpt_dict[k], 1), k, pt_ckpt_dict[k],)
                    for k in not_in_model]
    not_in_ckpt = [(reduce(mul, curr_ckpt_dict[k], 1), k, curr_ckpt_dict[k])
                   for k in not_in_ckpt]

    not_in_model_sum = sum(k[0] for k in not_in_model)
    not_in_ckpt_sum = sum(k[0] for k in not_in_ckpt)

    print(f'not_in_model_sum: {num_to_words(not_in_model_sum)}')
    print(f'not_in_ckpt_sum: {num_to_words(not_in_ckpt_sum)}')

    not_in_model.insert(0, (not_in_model_sum, 'all', None))
    not_in_ckpt.insert(0, (not_in_ckpt_sum, 'all', None))

    not_in_model_dict = dict(
        n_params=[k[0] for k in not_in_model],
        name=[k[1] for k in not_in_model],
        shape=[k[2] for k in not_in_model],
    )
    not_in_ckpt_dict = dict(
        n_params=[k[0] for k in not_in_ckpt],
        name=[k[1] for k in not_in_ckpt],
        shape=[k[2] for k in not_in_ckpt],
    )
    not_in_model_df = pd.DataFrame.from_dict(not_in_model_dict)
    not_in_model_csv = linux_path(model_dir, "not_in_model.csv")
    not_in_model_df.to_csv(
        not_in_model_csv,
        index=False,
    )
    not_in_ckpt_df = pd.DataFrame.from_dict(not_in_ckpt_dict)
    not_in_ckpt_csv = linux_path(model_dir, "not_in_ckpt.csv")
    not_in_ckpt_df.to_csv(
        not_in_ckpt_csv,
        index=False,
    )

    matching_names = list(pt_ckpt_names.intersection(curr_ckpt_names))
    mismatching_shapes = [(k, pt_ckpt_dict[k], curr_ckpt_dict[k])
                          for k in matching_names if pt_ckpt_dict[k] != curr_ckpt_dict[k]]
    if mismatching_shapes:
        mismatching_shapes_dict = dict(
            name=[k[0] for k in mismatching_shapes],
            pt=[k[1] for k in mismatching_shapes],
            model=[k[2] for k in mismatching_shapes],
        )
        mismatching_shapes_df = pd.DataFrame.from_dict(mismatching_shapes_dict)
        mismatching_shapes_csv = linux_path(model_dir, "mismatching_shapes.csv")
        mismatching_shapes_df.to_csv(
            mismatching_shapes_csv,
            index=False,
        )

    shutil.rmtree(temp_model_dir)


class WarmUpAndDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applies a warmup schedule on a given learning rate decay schedule."""

    def __init__(self, base_learning_rate, learning_rate_scaling, batch_size,
                 learning_rate_schedule, warmup_steps, total_steps, tail_steps=0,
                 end_lr_factor=0.):
        super(WarmUpAndDecay, self).__init__()
        self.schedule = learning_rate_schedule
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.tail_steps = tail_steps  # final part of training with a small fixed lr
        self.end_lr_factor = end_lr_factor
        if learning_rate_scaling == 'linear':
            self.base_lr = base_learning_rate * batch_size / 256.
        elif learning_rate_scaling == 'sqrt':
            self.base_lr = base_learning_rate * math.sqrt(batch_size)
        elif learning_rate_scaling == 'none' or learning_rate_scaling is None:
            self.base_lr = base_learning_rate
        else:
            raise ValueError('Unknown learning rate scaling {}'.format(
                learning_rate_scaling))

    def __call__(self, step):
        base_lr = self.base_lr
        schedule = self.schedule
        total_steps = self.total_steps
        warmup_steps = self.warmup_steps
        tail_steps = self.tail_steps
        end_lr_factor = self.end_lr_factor

        if schedule == 'linear':
            linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
                base_lr, total_steps - warmup_steps - tail_steps,
                end_learning_rate=base_lr * end_lr_factor, power=1.0)
            decayed_lr = linear_decay(step - warmup_steps)
        elif schedule == 'cosine':
            cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
                base_lr, total_steps - warmup_steps - tail_steps, alpha=end_lr_factor)
            decayed_lr = cosine_decay(step - warmup_steps)
        elif schedule.startswith('cosine@'):  # Take a part of the cosine curve.
            rate = float(schedule.split('@')[1])
            cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
                base_lr,
                (total_steps - warmup_steps - tail_steps) / rate,
                alpha=end_lr_factor)
            decayed_lr = cosine_decay(step - warmup_steps)
        elif schedule.startswith('exp@'):
            if tail_steps > 0:
                raise ValueError(f'tail_steps={tail_steps} is not effective for exp schedule.')
            rate = float(schedule.split('@')[1])
            exp_decay = tf.keras.optimizers.schedules.ExponentialDecay(
                base_lr, total_steps - warmup_steps, rate)
            decayed_lr = exp_decay(step - warmup_steps)
        elif schedule == 'none':
            decayed_lr = base_lr
        else:
            raise ValueError('Unknown learnig rate schedule {}'.format(
                schedule))

        learning_rate_warmup = (
            tf.cast(step, tf.float32) / float(warmup_steps) * base_lr
            if warmup_steps else base_lr)
        learning_rate = tf.where(step < warmup_steps, learning_rate_warmup,
                                 decayed_lr)
        return learning_rate


class AdamWeightDecay(tf.keras.optimizers.legacy.Adam):
    """Adam enables L2 weight decay and clip_by_global_norm on gradients.

    Just adding the square of the weights to the loss function is *not* the
    correct way of using L2 regularization/weight decay with Adam, since that will
    interact with the m and v parameters in strange ways.

    Instead we want ot decay the weights in a manner that doesn't interact with
    the m/v parameters. This is equivalent to adding the square of the weights to
    the loss with plain (non-momentum) SGD.
    """

    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 amsgrad=False,
                 weight_decay_rate=0.0,
                 include_in_weight_decay=None,
                 exclude_from_weight_decay=None,
                 name='AdamWeightDecay',
                 **kwargs):
        super(AdamWeightDecay, self).__init__(learning_rate, beta_1, beta_2,
                                              epsilon, amsgrad, name=name, **kwargs)
        self.weight_decay_rate = weight_decay_rate
        self._include_in_weight_decay = include_in_weight_decay
        self._exclude_from_weight_decay = exclude_from_weight_decay
        if include_in_weight_decay and exclude_from_weight_decay:
            raise ValueError('Sepcify wd vars using only one of include and exclude.')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdamWeightDecay, self)._prepare_local(var_device, var_dtype,
                                                    apply_state)
        apply_state[(var_device, var_dtype)]['weight_decay_rate'] = tf.constant(
            self.weight_decay_rate, name='adam_weight_decay_rate')

    def _decay_weights_op(self, var, learning_rate, apply_state):
        do_decay = self._do_use_weight_decay(var.name)
        if do_decay:
            return var.assign_sub(
                learning_rate * var *
                apply_state[(var.device, var.dtype.base_dtype)]['weight_decay_rate'],
                use_locking=self._use_locking)
        return tf.no_op()

    def _get_lr(self, var_device, var_dtype, apply_state):
        """Retrieves the learning rate with the given state."""
        if apply_state is None:
            return self._decayed_lr_t[var_dtype], {}

        apply_state = apply_state or {}
        coefficients = apply_state.get((var_device, var_dtype))
        if coefficients is None:
            coefficients = self._fallback_apply_state(var_device, var_dtype)
            apply_state[(var_device, var_dtype)] = coefficients

        return coefficients['lr_t'], dict(apply_state=apply_state)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay,
                         self)._resource_apply_dense(grad, var, **kwargs)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay,
                         self)._resource_apply_sparse(grad, var, indices, **kwargs)

    def get_config(self):
        config = super(AdamWeightDecay, self).get_config()
        config.update({
            'weight_decay_rate': self.weight_decay_rate,
        })
        return config

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if self.weight_decay_rate == 0:
            return False

        if self._include_in_weight_decay:
            for r in self._include_in_weight_decay:
                if re.search(r, param_name) is not None:
                    return True
            return False

        if self._exclude_from_weight_decay:
            for r in self._exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
            return True

        return True


def build_optimizer(config, learning_rate):
    """Returns the optimizer."""
    if config.optimizer == 'momentum':
        return tf.keras.optimizers.SGD(
            learning_rate, config.momentum, nesterov=True)
    elif config.optimizer == 'adam':
        return tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=config.beta1,
            beta_2=config.beta2,
            epsilon=config.eps)
    elif config.optimizer == 'adamw_legacy':
        clipnorm = None if config.global_clipnorm <= 0 else config.global_clipnorm
        include_str = config.get('include_from_weight_decay', 'kernel')
        exclude_str = config.get('exclude_from_weight_decay', '')
        return AdamWeightDecay(
            learning_rate=learning_rate,
            weight_decay_rate=config.weight_decay,
            beta_1=config.beta1,
            beta_2=config.beta2,
            epsilon=config.eps,
            global_clipnorm=clipnorm,
            include_in_weight_decay=include_str.split(',') if include_str else [],
            exclude_from_weight_decay=exclude_str.split(',') if exclude_str else [])
    elif config.optimizer == 'adamw':
        clipnorm = None if config.global_clipnorm <= 0 else config.global_clipnorm
        exclude_str = config.get('exclude_from_weight_decay', '')
        optimizer = tf.keras.optimizers.AdamW(
            weight_decay=config.weight_decay,
            learning_rate=learning_rate,
            beta_1=config.beta1,
            beta_2=config.beta2,
            epsilon=config.eps,
            global_clipnorm=clipnorm,
        )
        optimizer.exclude_from_weight_decay(
            var_names=exclude_str.split(',') if exclude_str else [])
        return optimizer
    elif config.optimizer == 'lamb':
        exclude_str = config.get('exclude_from_weight_decay', 'bias,beta,gamma,emb')
        return tfa.optimizers.LAMB(
            learning_rate=learning_rate,
            weight_decay_rate=config.weight_decay,
            beta_1=config.beta1,
            beta_2=config.beta2,
            epsilon=config.eps,
            exclude_from_weight_decay=exclude_str.split(',') if exclude_str else [])
    else:
        raise ValueError('Unknown optimizer {}'.format(config.optimizer))


def get_loss(logits, label_seq, loss_type):
    """Returns loss.

    Args:
      logits: tensor of shape (bsz, seqlen, vocab_size).
      label_seq: tensor of shape (bsz, seqlen).
      loss_type: string of loss type.

    Returns:
      per token loss tensor of shape (bsz, seqlen).
    """

    def _extract_loss_param(loss_type, default='0'):
        # loss_type is in `loss|loss@param` format where param is loss param.
        if '@' in loss_type:
            return loss_type.split('@')[1]
        return default

    label_hot = tf.cast(tf.one_hot(label_seq, tf.shape(logits)[-1]), logits.dtype)
    if 'xent' in loss_type:
        label_smoothing = float(_extract_loss_param(loss_type))
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=label_smoothing,
            reduction=tf.keras.losses.Reduction.NONE)(label_hot, logits)
    elif 'logistic' in loss_type:
        label_smoothing = float(_extract_loss_param(loss_type))
        logits -= tf.math.log(tf.cast(logits.shape[1], tf.float32))
        if label_smoothing > 0:
            label_hot = label_smoothing + label_hot * (1. - 2. * label_smoothing)
        log_p = tf.math.log_sigmoid(logits)
        log_p_not = tf.math.log_sigmoid(-logits)
        loss = -tf.reduce_sum(
            label_hot * log_p + (1. - label_hot) * log_p_not, axis=-1)
    elif 'focal' in loss_type:
        gamma = float(_extract_loss_param(loss_type))
        p = tf.nn.softmax(logits)
        logp = tf.math.log(p + 1e-8)
        focal_weight = tf.pow(1. - p, gamma) if gamma > 0 else 1.
        loss = - tf.reduce_sum(focal_weight * label_hot * logp, -1)
    else:
        raise ValueError('Unknown loss type {}'.format(loss_type))
    return loss


def get_val_metrics(y_true, y_pred_logits, y_mask):
    y_true_m = tf.boolean_mask(y_true, y_mask)

    y_pred = tf.argmax(y_pred_logits, axis=2)
    y_pred_m = tf.boolean_mask(y_pred, y_mask)

    """Don't care about output tokens corresponding to GT tokens marked as padding"""
    y_total_m = tf.cast(tf.size(y_true_m), tf.int64)

    y_correct_m = tf.math.equal(y_true_m, y_pred_m)
    y_correct_count_m = tf.reduce_sum(tf.cast(y_correct_m, tf.int64))
    y_correct_pc_m = (y_correct_count_m / y_total_m) * 100

    return y_correct_pc_m


def debug_loss(config, class_names, examples, y_true, y_pred_logits, y_mask=None, y_pred=None,
               pred_name='pred', gt_name='gt', run_type='train', is_video=True,
               y_infer=None, y_infer_logits=None, infer_name='infer'):
    vocab_size = config.model.vocab_size

    if y_pred is None:
        y_pred = tf.argmax(y_pred_logits, axis=2)
    else:
        y_pred_from_logits = tf.argmax(y_pred_logits, axis=2)
        y_pred_match = tf.math.equal(y_pred, y_pred_from_logits)
        y_pred_total = tf.cast(tf.size(y_pred), tf.int64)
        y_pred_match_pc = (tf.reduce_sum(tf.cast(y_pred_match, tf.int64)) / y_pred_total) * 100

    pred_metrics = get_metrics_info(y_true, y_pred, y_pred_logits, None)

    if y_infer_logits is not None:
        infer_metrics = get_metrics_info(y_true, y_infer, y_infer_logits, None)

    y_true_logits = tf.one_hot(y_true, depth=vocab_size)

    from datetime import datetime
    import os

    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")

    from tasks.visualization import vis_utils

    vis_out_dir = os.path.join(config.model_dir, 'debug_loss', f'{run_type}_{time_stamp}')

    # print(f'vis_out_dir: {vis_out_dir}')
    os.makedirs(vis_out_dir, exist_ok=True)

    if is_video:
        vis_fn = vis_utils.visualize_video
    else:
        vis_fn = vis_utils.visualize_image

    bbox_info_gt = vis_fn(
        config, examples, y_true_logits, y_true, f'{gt_name}', class_names, None, vis_out_dir)
    bbox_info_pred = vis_fn(
        config, examples, y_pred_logits, y_pred, f'{pred_name}', class_names, None, vis_out_dir)

    if y_infer_logits is not None:
        bbox_info_infer = vis_fn(
            config, examples, y_infer_logits, y_infer, f'{infer_name}', class_names, None, vis_out_dir)

    if y_mask is None:
        return bbox_info_gt, bbox_info_pred

    pred_metrics_m = get_metrics_info(y_true, None, y_pred_logits, y_mask)

    bbox_info_gt_m = vis_fn(config, examples, y_true_logits, y_true, f'{gt_name} masked',
                            class_names, y_mask, vis_out_dir)
    bbox_info_pred_m = vis_fn(config, examples, y_pred_logits, y_pred, f'{pred_name} masked',
                              class_names, y_mask, vis_out_dir)

    if y_infer_logits is not None:
        bbox_info_infer_m = vis_fn(config, examples, y_infer_logits, y_infer, f'{infer_name} masked',
                                   class_names, y_mask, vis_out_dir)
        infer_metrics_m = get_metrics_info(y_true, None, y_infer_logits, y_mask)

    return bbox_info_gt, bbox_info_pred, bbox_info_gt_m, bbox_info_pred_m, pred_metrics_m


def get_metrics_info(y_true, y_pred, y_pred_logits, y_mask):
    if y_mask is not None:
        y_true = tf.boolean_mask(y_true, y_mask)
        y_pred_logits = tf.boolean_mask(y_pred_logits, y_mask)

    # y_mask_int = tf.cast(y_mask, tf.int64)
    # y_mask_count = tf.reduce_sum(y_mask_int, axis=1)

    # y_true_logits_masked = tf.one_hot(y_true_m, depth=vocab_size)

    """Don't care about output tokens corresponding to GT tokens marked as padding"""
    y_total_m = tf.cast(tf.size(y_true), tf.int64)
    if y_pred is None:
        y_pred = tf.argmax(y_pred_logits, axis=1)
    y_correct_m = tf.math.equal(y_true, y_pred)
    y_correct_count_m = tf.reduce_sum(tf.cast(y_correct_m, tf.int64))
    y_correct_pc_m = (y_correct_count_m / y_total_m) * 100

    # y_cmb = tf.stack((y_true_m, y_pred_m, y_correct_int), axis=0)

    m = tf.keras.metrics.SparseCategoricalAccuracy()
    m.update_state(y_true, y_pred_logits)
    accuracy_notpad_m = m.result().numpy()

    metrics_info = [y_correct_pc_m, accuracy_notpad_m]

    return metrics_info


def num_to_words(num):
    if num >= 1e12:
        num_tril = num / 1e12
        words = f'{num_tril:.2f} T'
    elif num >= 1e9:
        num_bil = num / 1e9
        words = f'{num_bil:.2f} B'
    elif num >= 1e6:
        num_mil = num / 1e6
        words = f'{num_mil:.2f} M'
    elif num >= 1e3:
        num_th = num / 1e3
        words = f'{num_th:.2f} K'
    else:
        words = f'{num}'
    return words


def get_params_counts(model, level=0):
    import numpy as np
    total_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])

    if level == 0:
        print(f'\ntotal ({type(model).__name__}): {num_to_words(total_params)}')

    level += 1

    try:
        trainable_modules = model.trainable_modules
    except AttributeError:
        # print(f'\nno trainable_modules in {type(model).__name__}\n')
        trainable_modules = []
        model_attrs = list(model.__dict__.keys())
        for k in model_attrs:
            attr_obj = getattr(model, k)
            if hasattr(attr_obj, 'trainable_weights'):
                trainable_modules.append(k)

    for module_name in trainable_modules:
        module = getattr(model, module_name)
        try:
            trainable_weights = module.trainable_weights
        except AttributeError as e:
            # print(f'\nno trainable_weights in {module}\n')
            # continue
            raise e
        else:
            module_params = np.sum([np.prod(v.get_shape())
                                    for v in trainable_weights])

        assert module_params <= total_params, "module_params cannot exceed total_params"
        if total_params > 0:
            module_params_pc = (module_params / total_params) * 100
        else:
            module_params_pc = 0

        txt = f'{module_name} ({type(module).__name__}): {num_to_words(module_params)} ({module_params_pc:.2f}%)'
        if level > 0:
            txt = '\t' * level + txt

        print(txt)

        get_params_counts(module, level=level + 1)
