import ml_collections
import numpy as np
import utils
import vocab
from tasks import task as task_lib
from tasks import task_utils
from tasks.visualization import vis_utils
import tensorflow as tf


@task_lib.TaskRegistry.register('semantic_segmentation')
class TaskSemanticSegmentation(task_lib.Task):
    def __init__(self,
                 config: ml_collections.ConfigDict):
        super().__init__(config)

        json_dict = task_utils.load_json(self.config.dataset.category_names_path)
        self._category_names = task_utils.get_category_names(json_dict)

        self.frame_id_to_img_info = {
            img['frame_id']: img for img in json_dict['images']
        }

        class_id_to_col, class_id_to_name = task_utils.get_class_info(self._category_names)

        # class_info_path = config.dataset.get('class_info_path')
        # assert class_info_path, "class_info_path must be provided"
        # class_id_to_col, class_id_to_name = task_utils.read_class_info(
        #     class_info_path)

        # n_classes = len(self._category_names)
        # class_names_from_json = tuple(self._category_names[i]['name'] for i in range(n_classes))
        # assert class_names_from_json == class_names, "class_names mismatch"

        self.palette_flat = vis_utils.get_palette(class_id_to_col)
        self.class_id_to_col = class_id_to_col
        self.class_id_to_name = class_id_to_name

    def preprocess_single(self, dataset, batch_duplicates, training, validation):
        if self.config.debug != 2:
            """apply transforms"""
            dataset = dataset.map(
                lambda x: self.preprocess_single_example(
                    x, training, validation, batch_duplicates),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return dataset

    def preprocess_batched(self, batched_examples, training):
        config = self.config.task
        if self.config.debug == 2:
            batched_examples = vis_utils.debug_image_pipeline(
                self.dataset,
                self.train_transforms,
                batched_examples,
                vis=0,
                model_dir=self.config.model_dir,
                training=training)

        image = batched_examples['image']
        # height = batched_examples['image/height']
        # width = batched_examples['image/width']

        if self.config.dataset.rle_from_mask:
            mask = batched_examples['mask']
            response_seq = task_utils.mask_to_rle_tokens_tf(
                image, mask, self.config, self.class_id_to_col, training)
        else:
            response_seq = batched_examples['rle']
        token_weights = tf.ones_like(response_seq, dtype=tf.float32)

        if self.config.debug:
            mask_vid_paths = batched_examples['mask_vid_path'].numpy()
            images = batched_examples['image'].numpy()
            img_ids = batched_examples['image/id'].numpy()
            frame_ids = batched_examples['frame_id'].numpy()
            rles = response_seq.numpy()

            task_utils.check_rle(self.config, mask_vid_paths, images, img_ids, frame_ids, rles,
                                 training=True, class_id_to_col=self.class_id_to_col)

        # response_seq, token_weights = build_response_seq_from_rle(
        #     batched_examples['rle'],
        #     config.starts_bins,
        #     config.lengths_bins,
        #     mconfig.coord_vocab_shift,
        # )

        prompt_seq = task_utils.build_prompt_seq_from_task_id(
            task_vocab_id=self.task_vocab_id,
            response_seq=response_seq)

        input_seq = tf.concat([prompt_seq, response_seq], -1)
        target_seq = tf.concat([prompt_seq, response_seq], -1)

        """rle seq is already padded"""
        # input_seq = utils.pad_to_max_len(input_seq, config.max_seq_len + 1,
        #                                  dim=-1, padding_token=vocab.PADDING_TOKEN)
        # target_seq = utils.pad_to_max_len(target_seq, config.max_seq_len + 1,
        #                                   dim=-1, padding_token=vocab.PADDING_TOKEN)

        """
        right shift the target_seq and left-shift the input_seq
        """
        input_seq, target_seq = input_seq[..., :-1], target_seq[..., 1:]

        """
        token_weights should already be config.max_seq_len since it is created from response_seq
        """
        # token_weights = utils.pad_to_max_len(token_weights, config.max_seq_len,
        #                                      dim=-1, padding_token=vocab.PADDING_TOKEN)

        """
        Assign lower weights for ending/padding tokens.
        eos_token_weight = 0.1
        """
        token_weights = tf.where(
            target_seq == vocab.PADDING_TOKEN,
            tf.zeros_like(token_weights) + config.eos_token_weight,
            token_weights)

        return batched_examples, input_seq, target_seq, token_weights

    def infer(self, model, preprocessed_outputs):
        """Perform inference given the model and preprocessed outputs."""
        config = self.config.task
        mconfig = self.config.model
        examples, input_seq, target_seq, token_weights = preprocessed_outputs
        image = examples["image"]
        bsz = tf.shape(image)[0]
        prompt_seq = task_utils.build_prompt_seq_from_task_id(
            self.task_vocab_id, prompt_shape=(bsz, 1))
        pred_seq, logits, _ = model.infer(
            image, prompt_seq, encoded=None,
            max_seq_len=mconfig.max_seq_len + 1,
            temperature=config.temperature, top_k=config.top_k, top_p=config.top_p)

        return examples, pred_seq, logits

    def postprocess_tpu(self, batched_examples, pred_rle, logits, training=False):
        example = batched_examples
        images, image_id = example['image'], example['image/id']
        orig_image_size = example['orig_image_size']
        unpadded_image_size = example['unpadded_image_size']
        frame_id = example['frame_id']

        gt_rle = example['rle']
        vid_path = example['vid_path']
        mask_vid_path = example['mask_vid_path']

        return (images, image_id, frame_id,
                pred_rle, logits, gt_rle,
                orig_image_size, unpadded_image_size,
                vid_path, mask_vid_path)

    def postprocess_cpu(self,
                        outputs,
                        train_step,
                        out_vis_dir,
                        out_mask_dir,
                        out_instance_dir,
                        out_mask_logits_dir,
                        json_vid_info=None,
                        det_vid_writers=None,
                        seg_vid_writers=None,
                        eval_step=None,
                        training=False,
                        show=False,
                        summary_tag='eval',
                        ret_results=False,
                        csv_data=None,
                        **kwargs
                        ):

        # Copy outputs to cpu.
        new_outputs = []
        for i in range(len(outputs)):
            new_outputs.append(
                tf.identity(outputs[i]).numpy()
            )

        (images, image_ids, frame_ids,
         rles, logits, gt_rles,
         orig_sizes, unpadded_sizes,
         vid_paths, mask_vid_paths) = new_outputs

        image_ids = image_ids.flatten().astype(str)
        vid_paths = vid_paths.flatten().astype(str)
        mask_vid_paths = mask_vid_paths.flatten().astype(str)

        images = np.copy(tf.image.convert_image_dtype(images, tf.uint8))

        max_length = self.config.dataset.eval.max_length
        subsample = self.config.dataset.eval.subsample

        assert self.config.dataset.train.max_length == max_length, "max_length mismatch"
        assert self.config.dataset.train.subsample == subsample, "subsample mismatch"

        allow_overlap = self.config.dataset.eval.allow_overlap

        multi_class = self.config.dataset.multi_class
        n_classes = len(self.class_id_to_col)

        assert max_length > 0, "max_length must be > 0"
        # assert subsample >= 1, "subsample must be >= 1"
        if not multi_class:
            assert n_classes == 2, "n_classes must be 2 for no multi_class"
        else:
            assert n_classes > 2, "n_classes must be > 2 for multi_class"

        instance_wise = self.config.dataset.instance_wise

        mask_from_logits_ = self.config.eval.mask_from_logits
        starts_2d = self.config.dataset.starts_2d
        length_as_class = self.config.dataset.length_as_class
        flat_order = self.config.dataset.flat_order

        starts_offset = self.config.model.coord_vocab_shift
        lengths_offset = self.config.model.len_vocab_shift
        class_offset = self.config.model.class_vocab_shift

        max_seq_len = self.config.model.max_seq_len
        vocab_size = self.config.model.vocab_size

        if subsample > 1:
            max_length = int(max_length / subsample)

        obj_masks_batch = []
        obj_classes_batch = []
        obj_scores_batch = []
        for (image_id_, image_, frame_id_, rle_, logits_,
             orig_size_, gt_rle_, vid_path, mask_vid_path) in zip(
            image_ids, images, frame_ids, rles, logits,
            orig_sizes, gt_rles, vid_paths, mask_vid_paths, strict=True):

            assert '/' in image_id_, f"invalid image_id_: {image_id_}"

            seq = image_id_.split('/')[0]

            orig_size_ = tuple(orig_size_)
            n_rows, n_cols = orig_size_

            if subsample > 1:
                n_rows, n_cols = int(n_rows / subsample), int(n_cols / subsample)

            rle_ = rle_[rle_ != vocab.PADDING_TOKEN]
            gt_rle_ = gt_rle_[gt_rle_ != vocab.PADDING_TOKEN]

            mask_from_file = mask_from_file_sub = None
            mask_instance = mask_logits = rle_logits_cmp = None

            # if self.config.debug:
            #     vid_masks, vid_masks_sub = task_utils.check_rle(
            #         mask_vid_path, image_, image_id_, frame_id_, gt_rle_, training=False)
            #     mask_from_file = vid_masks[0]
            #     mask_from_file_sub = vid_masks_sub[0]

            if instance_wise:
                mask_rec, instance_info_rec, rle_rec_cmp = task_utils.mask_from_instance_wise_tokens(
                    rle_tokens=rle_,
                    rle_logits=logits_,
                    shape=(n_rows, n_cols),
                    n_classes=n_classes,
                    starts_offset=starts_offset,
                    lengths_offset=lengths_offset,
                    class_offset=class_offset,
                    starts_2d=starts_2d,
                    flat_order=flat_order,
                    # out-of-place tokens and such (especially early on in a training and)
                    allow_extra=True,
                    ignore_invalid=True,
                )
                obj_mask_rec, obj_class, obj_score, mask_instance = instance_info_rec
                obj_masks_batch.append(np.asarray(obj_mask_rec, dtype=bool))
                obj_classes_batch.append(np.asarray(obj_class, dtype=np.int32))
                obj_scores_batch.append(np.asarray(obj_score, dtype=np.float32))

                mask_gt, instance_info_gt, rle_gt_cmp = task_utils.mask_from_instance_wise_tokens(
                    gt_rle_,
                    (n_rows, n_cols),
                    n_classes,
                    starts_offset=starts_offset,
                    lengths_offset=lengths_offset,
                    class_offset=class_offset,
                    starts_2d=starts_2d,
                    flat_order=flat_order,
                    # There should be no invalid or out of place tokens in the GT
                    allow_extra=False,
                    ignore_invalid=False,
                )
            else:
                mask_rec, rle_rec_cmp = task_utils.mask_from_tokens(
                    rle_,
                    (n_rows, n_cols),
                    allow_extra=True,
                    starts_offset=starts_offset,
                    lengths_offset=lengths_offset,
                    class_offset=class_offset,
                    starts_2d=starts_2d,
                    multi_class=multi_class,
                    max_length=max_length,
                    length_as_class=length_as_class,
                    flat_order=flat_order,
                    ignore_invalid=True,
                )

                mask_gt, rle_gt_cmp = task_utils.mask_from_tokens(
                    gt_rle_,
                    (n_rows, n_cols),
                    allow_extra=False,
                    starts_offset=starts_offset,
                    lengths_offset=lengths_offset,
                    class_offset=class_offset,
                    starts_2d=starts_2d,
                    multi_class=multi_class,
                    max_length=max_length,
                    length_as_class=length_as_class,
                    flat_order=flat_order,
                )

                if mask_from_logits_:
                    mask_logits, rle_logits_cmp = task_utils.mask_from_logits(
                        rle_logits=logits_,
                        shape=(n_rows, n_cols),
                        max_length=max_length,
                        n_classes=n_classes,
                        starts_offset=starts_offset,
                        lengths_offset=lengths_offset,
                        class_offset=class_offset,
                        length_as_class=length_as_class,
                        starts_2d=starts_2d,
                        flat_order=flat_order,
                        multi_class=multi_class,
                        max_seq_len=max_seq_len,
                        vocab_size=vocab_size,
                        allow_overlap=allow_overlap,
                    )

            if self.config.debug:
                # mask_from_file = task_utils.mask_vis_to_id(mask_from_file, n_classes, copy=True)
                mask_from_file_sub = task_utils.mask_vis_to_id(
                    mask_from_file_sub, n_classes, copy=True)
                if not np.array_equal(mask_from_file_sub, mask_gt):
                    raise AssertionError("mask_from_file_sub - mask_gt mismatch")

            n_classes = len(self.class_id_to_col)

            seq_img_infos = json_vid_info[seq]
            if seq_img_infos:
                out_frame_id = seq_img_infos[-1]['out_frame_id']
            else:
                out_frame_id = 0

            out_frame_id += 1
            img_info = dict(
                seq=str(seq),
                vid_id=-1,
                image_id=str(image_id_),
                src_frame_id=int(frame_id_),
                out_frame_id=int(out_frame_id),
                vid_path=str(vid_path),
                mask_vid_path=str(mask_vid_path),
            )
            vis_utils.visualize_mask(
                image_id_,
                image_,
                mask=mask_rec,
                mask_logits=mask_logits,
                mask_instance=mask_instance,
                mask_gt=mask_gt,
                class_to_col=self.class_id_to_col,
                seq_id=seq,
                img_info=img_info,
                out_mask_dir=out_mask_dir,
                out_mask_logits_dir=out_mask_logits_dir,
                out_instance_dir=out_instance_dir,
                out_vis_dir=out_vis_dir,
                vid_writers=seg_vid_writers,
                orig_size=orig_size_,
                show=show,
                palette_flat=self.palette_flat,
            )

            seq_img_infos.append(
                img_info
            )

        if instance_wise:
            vis_utils.add_image_summary_with_bbox(
                images,
                masks=obj_masks_batch,
                bboxes=None,
                csv_normalized=True,
                bboxes_rescaled=None,
                scores=obj_scores_batch,
                classes=obj_classes_batch,
                category_names=self._category_names,
                image_ids=image_ids,
                out_vis_dir=out_vis_dir if self.config.eval.save_vis else None,
                vid_writers=det_vid_writers,
                csv_data=csv_data,
                unpadded_size=unpadded_sizes,
                orig_size=orig_sizes,
                class_id_to_col=self.class_id_to_col,
            )

    def compute_scalar_metrics(self, step):
        raise AssertionError('not implemented')

    def reset_metrics(self):
        raise AssertionError('not implemented')

#
# def build_response_seq_from_rle(
#         rle_norm,
#         starts_bins,
#         lengths_bins,
#         coord_vocab_shift
# ):
#     batch_size, seq_len = rle_norm.shape
#     n_elem = batch_size * seq_len
#     is_padding = tf.equal(rle_norm, 0)
#
#     rle_norm_flat = tf.reshape(rle_norm, [-1])
#     starts = rle_norm_flat[::2]
#     lengths = rle_norm_flat[1::2]
#     quantized_starts = utils.quantize(starts, starts_bins)
#     quantized_lengths = utils.quantize(lengths, lengths_bins)
#
#     quantized_starts = quantized_starts + coord_vocab_shift
#     quantized_lengths = quantized_lengths + vocab.BASE_VOCAB_SHIFT
#
#     even_indices = [[k, ] for k in range(0, n_elem, 2)]
#     odd_indices = [[k, ] for k in range(1, n_elem, 2)]
#
#     even_indices_tf = tf.constant(even_indices)
#     odd_indices_tf = tf.constant(odd_indices)
#
#     # even_indices_tf = tf.reshape(even_indices_tf, [1, -1])
#     # odd_indices_tf = tf.reshape(odd_indices_tf, [1, -1])
#
#     quantized_rle_flat = tf.zeros_like(rle_norm_flat, dtype=tf.int64)
#     quantized_rle_flat = tf.tensor_scatter_nd_update(quantized_rle_flat, even_indices_tf, quantized_starts)
#     quantized_rle_flat = tf.tensor_scatter_nd_update(quantized_rle_flat, odd_indices_tf, quantized_lengths)
#
#     # is_even = np.zeros((n_elem,), dtype=bool)
#     # is_even[even_indices] = True
#     # is_even_tf = tf.constant(is_even)
#     # quantized_rle_flat = tf.where(is_even_tf, quantized_starts, quantized_starts)
#
#     quantized_rle = tf.reshape(quantized_rle_flat, rle_norm.shape)
#
#     quantized_rle = tf.where(is_padding,
#                              tf.zeros_like(quantized_rle), quantized_rle)
#
#     token_weights = tf.ones_like(quantized_rle)
#
#     return quantized_rle, token_weights
