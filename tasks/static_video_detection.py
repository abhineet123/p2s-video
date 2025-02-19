import ml_collections
import numpy as np
import utils
import vocab

from tasks import task as task_lib
from tasks import task_utils
from tasks.visualization import vis_utils

import tensorflow as tf


@task_lib.TaskRegistry.register('static_video_detection')
class TaskStaticVideoDetection(task_lib.Task):
    def __init__(self,
                 config: ml_collections.ConfigDict):
        super().__init__(config)

        self.max_seq_len = config.task.get('max_seq_len', 'auto')
        self.max_seq_len_test = config.task.get('max_seq_len_test', 'auto')
        self.vid_len = self.config.dataset.length
        self.inst_len = self.vid_len * 2 + 1 if config.task.coords_1d else self.vid_len * 4 + 1
        self.max_inst_per_image = self.config.task.max_instances_per_image
        self.max_inst_per_image_test = self.config.task.max_instances_per_image_test

        if self.max_seq_len == 'auto':
            self.max_seq_len = self.max_inst_per_image * self.inst_len

        if self.max_seq_len_test == 'auto':
            self.max_seq_len_test = self.max_inst_per_image_test * self.inst_len + 1

        self.config.task.max_seq_len = self.max_seq_len
        self.config.task.max_seq_len_test = self.max_seq_len_test

        self._category_names = task_utils.get_category_names(
            config.dataset.get('category_names_path'))
        self._coco_metrics = None

        if self.config.task.get('eval_outputs_json_path', None):
            self.eval_output_annotations = []

    def preprocess_single(self, dataset, batch_duplicates, training, validation):
        def _convert_video_to_image_features(example):
            new_example = dict(
                # orig_video_size=video_shape_2[1:3],
                orig_video_size=example['video/size'],
                video_id=example['video/id'],
                num_frames=example['video/num_frames'],
                image=example['video/image'],
                file_names=example['video/file_names'],
                file_ids=example['video/file_ids'],
                bbox=example['bbox'],
                class_name=example['class_name'],
                class_id=example['class_id'],
                area=example['area'],
                is_crowd=example['is_crowd'],
            )
            return new_example

        dataset = dataset.map(
            _convert_video_to_image_features,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.map(
            lambda x: self.preprocess_single_example(
                x, training, validation, batch_duplicates),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        return dataset

    def preprocess_batched(self, batched_examples, training):
        config = self.config.task
        mconfig = self.config.model
        dconfig = self.config.dataset
        batch_size = self.config.train.batch_size if training else self.config.eval.batch_size
        # batched_examples = vis_utils.debug_video_transforms(
        #     self.train_transforms if training else self.eval_transforms,
        #     batched_examples,
        #     vis=1, model_dir=self.config.model_dir)

        bboxes = batched_examples['bbox']
        """coord_vocab_shift needed to accomodate class tokens before the coord tokens"""
        ret = build_response_seq_from_video_bboxes(
            bboxes=bboxes,
            label=batched_examples['class_id'],
            quantization_bins=config.quantization_bins,
            coords_1d=config.coords_1d,
            noise_bbox_weight=config.noise_bbox_weight,
            coord_vocab_shift=mconfig.coord_vocab_shift,
            vid_len=dconfig.length,
            max_instances_per_image=config.max_instances_per_image,
            # batch_size=batch_size,
            debug=self.config.debug,
            class_label_corruption=config.class_label_corruption,
            class_equal_weight=config.class_equal_weight,
        )

        """
        response_seq_cm has random and noise labels by default
        target_seq = prompt_seq + response_seq
        input_seq = prompt_seq + response_seq_cm
        """
        response_seq, response_seq_cm, token_weights, boxes_quant = ret

        if self.config.debug:
            bboxes_rec, boxes_quant_rec = task_utils.seq_to_video_bbox(
                seq=response_seq,
                quantization_bins=config.quantization_bins,
                coords_1d=config.coords_1d,
                vid_len=dconfig.length,
                coord_vocab_shift=mconfig.coord_vocab_shift,
            )
            is_eq = tf.math.reduce_all(tf.equal(boxes_quant, boxes_quant_rec))
            assert is_eq, "unravel_idx boxes_quant_rec mismatch"

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

        # input_shape = utils.shape_as_list(input_seq)
        # assert input_shape[-1] <= config.max_seq_len + 1, \
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
            tf.cast(config.eos_token_weight, token_weights.dtype),
            token_weights)

        return batched_examples, input_seq, target_seq, token_weights

    def infer(self, model, preprocessed_outputs):
        """Perform inference given the model and preprocessed outputs."""
        config = self.config.task
        examples, input_seq, target_seq, token_weights = preprocessed_outputs  # response_seq unused by default
        images = examples['image']

        bsz = tf.shape(images)[0]

        prompt_seq = task_utils.build_prompt_seq_from_task_id(
            self.task_vocab_id,
            prompt_shape=(bsz, 1))

        pred_seq, logits, encoded = model.infer(
            images, prompt_seq, encoded=None,
            max_seq_len=config.max_seq_len_test,
            temperature=config.temperature, top_k=config.top_k, top_p=config.top_p)

        return examples, pred_seq, logits

    def postprocess_tpu(self, batched_examples, pred_seq, logits, training=False):
        config = self.config.task
        mconfig = self.config.model
        example = batched_examples

        gt_classes, gt_bboxes = example['class_id'], example['bbox']
        area, is_crowd = example['area'], example['is_crowd']

        images, video_ids = example['image'], example['video_id']
        file_names = example['file_names']
        file_ids = example['file_ids']
        orig_video_size = example['orig_video_size']
        unpadded_video_size = example['unpadded_video_size']

        # Decode sequence output.
        pred_classes, pred_bboxes, scores = task_utils.decode_video_seq_to_bbox(
            logits, pred_seq, self.vid_len, config.quantization_bins, config.coords_1d, mconfig.coord_vocab_shift)

        # Compute coordinate scaling from [0., 1.] to actual pixels in orig image.
        image_size = images.shape[1:3].as_list()
        if training:
            # scale points to whole image size during train.
            scale = utils.tf_float32(image_size)
        else:
            # scale points to original image size during eval.
            scale = (
                    utils.tf_float32(image_size)[tf.newaxis, :] /
                    utils.tf_float32(unpadded_video_size))
            scale = scale * utils.tf_float32(orig_video_size)
            scale = tf.expand_dims(scale, 1)

        pred_bboxes_rescaled = utils.scale_points(pred_bboxes, scale)
        gt_bboxes_rescaled = utils.scale_points(gt_bboxes, scale)

        return (
            images, video_ids, pred_bboxes, pred_bboxes_rescaled, pred_classes,
            scores, gt_classes, gt_bboxes, gt_bboxes_rescaled, area,
            orig_video_size, unpadded_video_size,
            file_names, file_ids
        )

    def postprocess_cpu(
            self,
            outputs,
            train_step,
            out_vis_dir=None,
            det_vid_writers=None,
            csv_data=None,
            eval_step=None,
            summary_tag='eval',
            training=False,
            ret_results=False,
            **kwargs
    ):

        assert out_vis_dir is None, "visualization is not currently supported in static video detection"

        """move to cpu"""
        new_outputs = []
        for i in range(len(outputs)):
            new_outputs.append(tf.identity(outputs[i]))
        (frames, video_ids,
         pred_bboxes, pred_bboxes_rescaled,
         pred_classes, scores, gt_classes,
         gt_bboxes, gt_bboxes_rescaled, area,
         orig_video_size, unpadded_video_size,
         file_names, file_ids
         ) = new_outputs

        unpadded_video_size = unpadded_video_size.numpy()
        orig_video_size = orig_video_size.numpy()

        bboxes_, bboxes_rescaled_, classes_, scores_ = (
            pred_bboxes.numpy(), pred_bboxes_rescaled.numpy(), pred_classes.numpy(), scores.numpy())

        frames_ = np.copy(tf.image.convert_image_dtype(frames, tf.uint8))

        vis_utils.add_video_summary_with_bbox(
            frames_, bboxes_, bboxes_rescaled_,
            classes_, scores_,
            category_names=self._category_names,
            video_ids=video_ids.numpy(),
            filenames=file_names.numpy(),
            file_ids=file_ids.numpy(),
            vid_len=self.vid_len,
            out_vis_dir=out_vis_dir,
            vid_writers=det_vid_writers,
            csv_data=csv_data,
            unpadded_size=unpadded_video_size,
            orig_size=orig_video_size,
        )

    def compute_scalar_metrics(self, step):
        """Returns a dict containing scalar metrics to log."""
        if self._coco_metrics:
            return self._coco_metrics.result(step)
        else:
            return {}

    def reset_metrics(self):
        """Reset states of metrics accumulators."""
        if self._coco_metrics:
            self._coco_metrics.reset_states()


def tf_ravel_multi_index(bboxes, dims, vid_len, max_instances_per_image, check):
    strides = tf.math.cumprod(dims, exclusive=True, reverse=True)

    # strides_tiled = tf.tile(strides, [length*2])
    # strides_exp_1 = tf.expand_dims(strides, 0)
    # strides_exp_2 = tf.expand_dims(strides_exp_1, 0)
    # strides_exp_3 = tf.expand_dims(strides_exp_2,
    # 0)

    # B, N, K = tf.shape(bboxes)

    strides_exp_3 = tf.reshape(strides, (1, 2))
    bboxes_res = tf.reshape(bboxes, (-1, 2))

    bboxes_tmp = bboxes_res * strides_exp_3
    ravel_idx = tf.reduce_sum(bboxes_tmp, axis=-1)
    ravel_idx = tf.reshape(ravel_idx, (-1, max_instances_per_image, vid_len * 2))

    if check:
        ravel_idx_flat = tf.reshape(ravel_idx, (-1,))
        unravel_idx = tf.unravel_index(ravel_idx_flat, dims)

        bboxes_rec = tf.reshape(tf.transpose(unravel_idx), tf.shape(bboxes))

        is_eq = tf.math.reduce_all(tf.equal(bboxes, bboxes_rec))

        assert is_eq, "unravel_idx bboxes_rec mismatch"

    # pt1 = bboxes[:, :2]
    # pt2 = bboxes[:, 2:]
    # pt1_temp = pt1 * strides_exp_2
    # pt2_temp = pt2 * strides_exp_2
    # pt1_idx = tf.reduce_sum(pt1_temp, axis=0)
    # pt2_idx = tf.reduce_sum(pt2_temp, axis=0)
    # ravel_idx = tf.concat([pt1_idx, pt2_idx], axis=-1)

    return ravel_idx


def build_response_seq_from_video_bboxes(
        bboxes,
        label,
        quantization_bins,
        noise_bbox_weight,
        class_equal_weight,
        coord_vocab_shift,
        vid_len,
        max_instances_per_image,
        # batch_size,
        coords_1d,
        debug,
        class_label_corruption='rand_cls',
):
    # assert bboxes.shape[-1] % 4 == 0, f"invalid bboxes shape: {bboxes.shape}"
    # n_bboxes_per_vid = int(bboxes.shape[-1] / 4)
    # assert vid_len == n_bboxes_per_vid, f"Mismatch between vid_len: {vid_len} and n_bboxes_per_vid: {
    # n_bboxes_per_vid}"

    is_no_box = tf.math.is_nan(bboxes)
    # rand_tensor = tf.random.uniform(is_no_box.shape)

    if debug:
        """There should be at least one valid box per object"""
        max_no_boxes_per_obj = vid_len - 1
        n_no_boxes_per_obj = tf.reduce_sum(tf.cast(is_no_box, dtype=tf.int32), axis=-1) / 4
        tf.debugging.assert_less_equal(
            n_no_boxes_per_obj,
            tf.cast(max_no_boxes_per_obj, n_no_boxes_per_obj.dtype),
            message="There should be at least one valid box per object")

    quantized_bboxes_2d = utils.quantize(bboxes, quantization_bins)

    if coords_1d:
        shape = tf.constant([quantization_bins, quantization_bins], dtype=tf.int64)
        quantized_bboxes_1d = tf_ravel_multi_index(
            quantized_bboxes_2d, shape,
            max_instances_per_image=max_instances_per_image,
            vid_len=vid_len, check=debug)
        quantized_bboxes = quantized_bboxes_1d
        # is_no_box_quant = tf.math.is_nan(quantized_bboxes)
        is_no_box_1d = is_no_box[:, :, ::2]
        is_no_box = is_no_box_1d
        # rand_tensor_quant = rand_tensor[:, :, ::2]
    else:
        quantized_bboxes = quantized_bboxes_2d

    quantized_bboxes = quantized_bboxes + coord_vocab_shift

    # np_dict = utils.to_numpy(locals())

    # quantized_bboxes_np = quantized_bboxes.numpy()
    # add_name(vars())

    quantized_bboxes = tf.where(
        is_no_box,
        # this is needed rather than simply vocab.NO_BOX_TOKEN to make sure that data types match otherwise
        # annoying errors like
        # "TypeError: Input 'e' of 'SelectV2' Op has type int64 that does not match type int32 of argument 't'."
        # will occur and only in non-eager mode for some reason
        tf.cast(vocab.NO_BOX_TOKEN, dtype=quantized_bboxes.dtype),
        # tf.zeros_like(quantized_bboxes) + vocab.NO_BOX_TOKEN,
        quantized_bboxes)

    """set 0-labeled (padding) bboxes to zero"""
    is_padding = tf.expand_dims(tf.equal(label, 0), -1)
    quantized_bboxes = tf.where(
        is_padding,
        tf.zeros_like(quantized_bboxes),
        quantized_bboxes)
    new_label = tf.expand_dims(label + vocab.BASE_VOCAB_SHIFT, -1)
    new_label = tf.where(is_padding, tf.zeros_like(new_label), new_label)
    lb_shape = tf.shape(new_label)

    # Bbox and label serialization.
    response_seq = tf.concat([quantized_bboxes, new_label], axis=-1)

    """Merge last few dims to have rank-2 shape [bsz, n_tokens] where
    n_tokens = n_bboxes*5    
    """
    response_seq = utils.flatten_non_batch_dims(response_seq, 2)

    """
    different combinations of random, fake and real class labels apparently 
    created in case something other than the real labels is required 
    according to the class_label_corruption Parameter

    class_label_corruption=rand_n_fake_cls by default  
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

    """response_seq_class_m is same as response_seq if no corruptions are needed,
    i.e. if class_label_corruption=none
    otherwise, some or all the real labels are randomly replaced by noise label to 
    generate corrupted labels that are used as input sequence

    The rationale for corrupting the input sequences might be that we want the network to produce 
    the right class labels in the subsequent outputs even if the previous one was incorrect 
    So we do not want to condition the Generation of class output In the next tokens Lee to is strongly on 
    the class label being correct in the previously generated tokens
    """
    response_seq_class_m = tf.concat([quantized_bboxes, new_label_m], axis=-1)
    response_seq_class_m = utils.flatten_non_batch_dims(response_seq_class_m, 2)

    # Get token weights.
    is_real = tf.cast(tf.not_equal(new_label, vocab.FAKE_CLASS_TOKEN), tf.float32)

    """    
    noise and real bbox coord tokens have weights 0 and 1 respectively

    real bbox class tokens have weight 1
    noise bbox class tokens have weight noise_bbox_weight 

    noise_bbox_weight = 1.0 when training with fake objects

    We don't care about the coordinates of fake boxes but we do care about their class   
    """
    n_coord_tokens = int(quantized_bboxes.shape[-1])
    bbox_weight = tf.tile(is_real, [1, 1, n_coord_tokens])

    if class_equal_weight:
        norm_factor = 1.0 / n_coord_tokens
        bbox_weight = bbox_weight * norm_factor

    label_weight = is_real + (1. - is_real) * noise_bbox_weight

    token_weights = tf.concat([bbox_weight, label_weight], -1)
    token_weights = utils.flatten_non_batch_dims(token_weights, 2)

    return response_seq, response_seq_class_m, token_weights, quantized_bboxes_2d
