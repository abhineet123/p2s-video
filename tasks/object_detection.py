import json
import os
import pickle
from typing import Any, Dict, List

from absl import logging
import ml_collections
import numpy as np
import utils
import vocab
from metrics import metric_utils
from tasks import task as task_lib
from tasks import task_utils
from tasks.visualization import vis_utils
import tensorflow as tf


@task_lib.TaskRegistry.register('object_detection')
class TaskObjectDetection(task_lib.Task):
    """
    Object detection task with coco metric evaluation.
    """

    def __init__(self,
                 config: ml_collections.ConfigDict):
        super().__init__(config)

        if config.task.get('max_seq_len', 'auto') == 'auto':
            self.config.task.max_seq_len = config.task.max_instances_per_image * 5
        self._category_names = task_utils.get_category_names(
            config.dataset.get('category_names_path'))
        # metric_config = config.task.get('metric')
        # if metric_config and metric_config.get('name'):
        #     self._coco_metrics = metric_registry.MetricRegistry.lookup(
        #         metric_config.name)(config)
        # else:
        #     self._coco_metrics = None
        # if self.config.task.get('eval_outputs_json_path', None):
        #     self.eval_output_annotations = []

    def preprocess_single(self, dataset, batch_duplicates, training, validation):
        """Task-specific preprocessing of individual example in the dataset.

        Typical operations in this preprocessing step for detection task:
          - Image augmentation such random resize & cropping, color jittering, flip.
          - Label augmentation such as sampling noisy & duplicated boxes.

        Args:
          dataset: A tf.data.Dataset.
          batch_duplicates: `int`, enlarge a batch by augmenting it multiple times
            (as specified) and concating the augmented examples.
          training: bool.

        Returns:
          A dataset.
        """
        if self.config.debug != 2:
            if training:
                dataset = dataset.filter(  # Filter out images with no annotations.
                    lambda example: tf.shape(example['label'])[0] > 0)

            dataset = dataset.map(
                lambda x: self.preprocess_single_example(
                    x, training, validation, batch_duplicates),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return dataset

    def preprocess_batched(self, batched_examples, training):
        config = self.config.task
        mconfig = self.config.model

        if self.config.debug == 2:
            batched_examples = vis_utils.debug_image_pipeline(
                self.dataset,
                self.train_transforms, batched_examples,
                vis=1, model_dir=self.config.model_dir, training=training)

        # Create input/target seq.
        """coord_vocab_shift needed to accommodate class tokens before the coord tokens"""
        ret = build_response_seq_from_bbox(
            batched_examples['bbox'], batched_examples['label'],
            config.quantization_bins, config.noise_bbox_weight,
            mconfig.coord_vocab_shift,
            class_label_corruption=config.class_label_corruption,
            class_equal_weight=config.class_equal_weight,
        )

        """response_seq_cm has random and noise class labels by default"""
        response_seq, response_seq_cm, token_weights = ret

        """
        vocab_id=10 for object_detection
        prompt_seq is apparently just a [bsz, 1] vector containing vocab_id and serves as the start token        
        """
        prompt_seq = task_utils.build_prompt_seq_from_task_id(
            task_vocab_id=self.task_vocab_id,
            response_seq=response_seq)  # (bsz, 1)
        input_seq = tf.concat([prompt_seq, response_seq_cm], -1)
        target_seq = tf.concat([prompt_seq, response_seq], -1)

        # Pad sequence to a unified maximum length.
        """
        max_seq_len=512 for object_detection
        """
        # assert input_seq.shape[-1] <= config.max_seq_len + 1, \
        #     f"input_seq length {input_seq.shape[-1]} exceeds max_seq_len {config.max_seq_len + 1}"

        input_seq = utils.pad_to_max_len(input_seq, config.max_seq_len + 1,
                                         dim=-1, padding_token=vocab.PADDING_TOKEN)
        target_seq = utils.pad_to_max_len(target_seq, config.max_seq_len + 1,
                                          dim=-1, padding_token=vocab.PADDING_TOKEN)
        """
        right shift the target_seq and left-shift the input_seq
        """
        input_seq, target_seq = input_seq[..., :-1], target_seq[..., 1:]
        token_weights = utils.pad_to_max_len(token_weights, config.max_seq_len,
                                             dim=-1, padding_token=vocab.PADDING_TOKEN)

        # Assign lower weights for ending/padding tokens.
        """
        eos_token_weight = 0.1 for object_detection
        """
        token_weights = tf.where(
            target_seq == vocab.PADDING_TOKEN,
            # stupid way of assigning eos_token_weight to padding tokens,
            # just eos_token_weight should work here too since tf.where is supposed to broadcast
            tf.zeros_like(token_weights) + config.eos_token_weight,
            token_weights)

        return batched_examples, input_seq, target_seq, token_weights

        # if training:
        #     return batched_examples, input_seq, target_seq, token_weights
        # else:
        #     return batched_examples, input_seq, response_seq, target_seq, token_weights

    def infer(self, model, preprocessed_outputs):
        """Perform inference given the model and preprocessed outputs."""
        config = self.config.task
        examples, input_seq, target_seq, token_weights = preprocessed_outputs  # response_seq unused by
        # default
        image = examples["image"]
        bsz = tf.shape(image)[0]
        prompt_seq = task_utils.build_prompt_seq_from_task_id(
            self.task_vocab_id, prompt_shape=(bsz, 1))
        pred_seq, logits, _ = model.infer(
            image, prompt_seq, encoded=None,
            max_seq_len=(config.max_instances_per_image_test * 5 + 1),
            temperature=config.temperature, top_k=config.top_k, top_p=config.top_p)

        # if self.config.validation:
        #     from models import model_utils
        #
        #     target_seq = utils.flatten_batch_dims(target_seq, out_rank=2)
        #     token_weights = utils.flatten_batch_dims(token_weights, out_rank=2)
        #     token_weights = utils.tf_float32(token_weights)
        #
        #     is_padding = tf.equal(target_seq, vocab.PADDING_TOKEN)  # padding tokens.
        #     token_weights_notpad = tf.where(
        #         is_padding, tf.zeros_like(token_weights), token_weights)
        #     losses = model_utils.get_loss(
        #         logits, target_seq, self.config.train.loss_type)
        #     loss = tf.reduce_sum(losses * token_weights) / (
        #             tf.reduce_sum(token_weights) + 1e-9)
        #     loss_notpad = tf.reduce_sum(losses * token_weights_notpad) / (
        #             tf.reduce_sum(token_weights_notpad) + 1e-9)
        #
        #     y_mask = tf.greater(token_weights_notpad, 0)
        #     y_correct_pc_m, accuracy_notpad_m = model_utils.get_val_metrics(
        #         target_seq, pred_seq, logits, y_mask, self.val_m)
        #     return loss, loss_notpad, y_correct_pc_m, accuracy_notpad_m

        # if True:  # Sanity check by using gt response_seq as pred_seq.
        #   pred_seq = preprocessed_outputs[1]
        #   logits = tf.one_hot(pred_seq, mconfig.vocab_size)
        return examples, pred_seq, logits

    def postprocess_tpu(self, batched_examples, pred_seq, logits,
                        training=False):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
        """Organizing results after fitting the batched examples in graph.

        Such as updating metrics, putting together results for computing metrics in
          CPU/numpy mode.

        Note: current implementation only support eval mode where gt are given in
          metrics as they are not constructed here from input_seq/target_seq.

        Args:
          batched_examples: a tuple of features (`dict`) and labels (`dict`),
            containing images and labels.
          pred_seq: `int` sequence of shape (bsz, seqlen').
          logits: `float` sequence of shape (bsz, seqlen', vocab_size).
          training: `bool` indicating training or inference mode.

        Returns:
          results for passing to `postprocess_cpu` which runs in CPU mode.
        """
        config = self.config.task
        mconfig = self.config.model
        example = batched_examples
        images, image_ids = example['image'], example['image/id']
        orig_image_size = example['orig_image_size']
        unpadded_image_size = example['unpadded_image_size']

        # Decode sequence output.
        pred_classes, pred_bboxes, scores = task_utils.decode_object_seq_to_bbox(
            logits, pred_seq, config.quantization_bins, mconfig.coord_vocab_shift)

        # Compute coordinate scaling from [0., 1.] to actual pixels in orig image.
        image_size = images.shape[1:3].as_list()
        if training:
            # scale points to whole image size during train.
            scale = utils.tf_float32(image_size)
        else:
            # scale points to original image size during eval.
            scale = (
                    utils.tf_float32(image_size)[tf.newaxis, :] /
                    utils.tf_float32(unpadded_image_size))
            scale = scale * utils.tf_float32(orig_image_size)
            scale = tf.expand_dims(scale, 1)
        pred_bboxes_rescaled = utils.scale_points(pred_bboxes, scale)

        gt_classes, gt_bboxes = example['label'], example['bbox']
        gt_bboxes_rescaled = utils.scale_points(gt_bboxes, scale)
        area, is_crowd = example['area'], example['is_crowd']

        return (images, image_ids, pred_bboxes, pred_bboxes_rescaled, pred_classes,
                scores, gt_classes, gt_bboxes, gt_bboxes_rescaled, area,
                orig_image_size, unpadded_image_size,
                # is_crowd
                )

    def postprocess_cpu(
            self,
            outputs,
            train_step,
            # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
            out_vis_dir=None,
            det_vid_writers=None,
            csv_data=None,
            eval_step=None,
            training=False,
            summary_tag='eval',
            ret_results=False,
            **kwargs
    ):
        # Copy outputs to cpu.
        new_outputs = []
        for i in range(len(outputs)):
            # logging.info('Copying output at index %d to cpu for cpu post-process', i)
            new_outputs.append(tf.identity(outputs[i]))
        (images, image_ids, pred_bboxes, pred_bboxes_rescaled, pred_classes,
         scores, gt_classes, gt_bboxes, gt_bboxes_rescaled, area,
         orig_image_size, unpadded_image_size,
         # is_crowd
         ) = new_outputs

        unpadded_image_size = unpadded_image_size.numpy()
        orig_image_size = orig_image_size.numpy()

        # Image summary.
        image_ids_ = image_ids.numpy().flatten().astype(str)
        image_ids__ = list(image_ids_)
        ret_images = []
        bboxes_, bboxes_rescaled_, classes_, scores_ = (
            pred_bboxes.numpy(), pred_bboxes_rescaled.numpy(), pred_classes.numpy(), scores.numpy())
        images_ = np.copy(tf.image.convert_image_dtype(images, tf.uint8))

        ret_images += vis_utils.add_image_summary_with_bbox(
            images_, bboxes_, bboxes_rescaled_, classes_, scores_,
            self._category_names,
            image_ids__,
            # train_step, tag,
            # max_images_shown=(-1 if ret_results else 3)
            out_vis_dir=out_vis_dir,
            vid_writers=det_vid_writers,
            csv_data=csv_data,
            unpadded_size=unpadded_image_size,
            orig_size=orig_image_size,
        )

        if ret_results:
            return ret_images

    def evaluate(self, summary_writer, step, eval_tag):
        raise AssertionError('not implemented')

    def compute_scalar_metrics(self, step):
        raise AssertionError('not implemented')

    def reset_metrics(self):
        raise AssertionError('not implemented')


def build_response_seq_from_bbox(
        bbox,
        label,
        quantization_bins,
        noise_bbox_weight,
        coord_vocab_shift,
        class_label_corruption,
        class_equal_weight,
):
    """"Build target seq from bounding bboxes for object detection.

    Objects are serialized using the format of yxyxc.

    Args:
      bbox: `float` bounding box of shape (bsz, n, 4).
      label: `int` label of shape (bsz, n).
      quantization_bins: `int`.
      noise_bbox_weight: `float` on the token weights for noise bboxes.
      coord_vocab_shift: `int`, shifting coordinates by a specified integer.
      class_label_corruption: `string` specifying how labels are corrupted for the
        input_seq.

    Returns:
      discrete sequences with shape (bsz, seqlen).
    """
    # Bbox and label quantization.
    is_padding = tf.expand_dims(tf.equal(label, 0), -1)
    quantized_bbox = utils.quantize(bbox, quantization_bins)
    quantized_bbox = quantized_bbox + coord_vocab_shift

    """set 0-labeled bbox tokens to zero"""
    """(bsz, n, 4)"""
    quantized_bbox = tf.where(is_padding,
                              tf.zeros_like(quantized_bbox), quantized_bbox)
    """(bsz, n, 1)"""
    new_label = tf.expand_dims(label + vocab.BASE_VOCAB_SHIFT, -1)
    new_label = tf.where(is_padding, tf.zeros_like(new_label), new_label)
    lb_shape = tf.shape(new_label)

    # Bbox and label serialization.
    """(bsz, n, 5)"""
    response_seq = tf.concat([quantized_bbox, new_label], axis=-1)

    """Merge last few dims to have rank-2 shape [bsz, n_tokens] where
    n_tokens = n_bboxes*5    
    """
    """(bsz, n*5)"""
    response_seq = utils.flatten_non_batch_dims(response_seq, 2)

    """
    different Combinations of random, fake and real class labels apparently 
    created just in case something other than the real labels is required 
    according to class_label_corruption parameter (rand_n_fake_cls by default)
    """
    rand_cls = vocab.BASE_VOCAB_SHIFT + tf.random.uniform(
        lb_shape,
        0,
        coord_vocab_shift - vocab.BASE_VOCAB_SHIFT,
        dtype=new_label.dtype)
    fake_cls = vocab.FAKE_CLASS_TOKEN + tf.zeros_like(new_label)
    rand_n_fake_cls = tf.where(
        tf.random.uniform(lb_shape) > 0.5, rand_cls, fake_cls)
    real_n_fake_cls = tf.where(
        tf.random.uniform(lb_shape) > 0.5, new_label, fake_cls)
    real_n_rand_n_fake_cls = tf.where(
        tf.random.uniform(lb_shape) > 0.5, new_label, rand_n_fake_cls)
    label_mapping = {'none': new_label,
                     'rand_cls': rand_cls,
                     'real_n_fake_cls': real_n_fake_cls,
                     'rand_n_fake_cls': rand_n_fake_cls,
                     'real_n_rand_n_fake_cls': real_n_rand_n_fake_cls}
    new_label_m = label_mapping[class_label_corruption]
    new_label_m = tf.where(is_padding, tf.zeros_like(new_label_m), new_label_m)

    """response_seq_class_m is apparently same as response_seq if no corruptions are needed,
    i.e. if class_label_corruption=none"""
    response_seq_class_m = tf.concat([quantized_bbox, new_label_m], axis=-1)
    response_seq_class_m = utils.flatten_non_batch_dims(response_seq_class_m, 2)

    # Get token weights.
    is_real = tf.cast(tf.not_equal(new_label, vocab.FAKE_CLASS_TOKEN), tf.float32)

    """noise and real bbox coord tokens have weights 1 and 0 respectively"""
    bbox_weight = tf.tile(is_real, [1, 1, 4])
    if class_equal_weight:
        norm_factor = 1. / 4.
        bbox_weight = bbox_weight * norm_factor
    """
    real bbox class tokens have weight 1
    noise bbox class tokens have weight noise_bbox_weight    
    """
    label_weight = is_real + (1. - is_real) * noise_bbox_weight

    token_weights = tf.concat([bbox_weight, label_weight], -1)
    token_weights = utils.flatten_non_batch_dims(token_weights, 2)

    return response_seq, response_seq_class_m, token_weights


def build_annotations(image_ids, category_ids, boxes, scores,
                      counter) -> List[Dict[str, Any]]:
    """Builds annotations."""
    annotations = []
    for image_id, category_id_list, box_list, score_list in zip(
            image_ids, category_ids, boxes, scores):
        for category_id, box, score in zip(category_id_list, box_list, score_list):
            category_id = int(category_id)
            if category_id:
                annotations.append({
                    'id': counter,
                    'image_id': int(image_id),
                    'category_id': category_id,
                    'bbox': metric_utils.yxyx_to_xywh(box.tolist()),
                    'iscrowd': False,
                    'score': float(score)
                })
                counter += 1
    return annotations
