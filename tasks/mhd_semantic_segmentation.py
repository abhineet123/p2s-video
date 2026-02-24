import ml_collections
import numpy as np
import utils
import vocab
from tasks import task as task_lib
from tasks import task_utils
from tasks.visualization import vis_utils
import tensorflow as tf


@task_lib.TaskRegistry.register('mhd_semantic_segmentation')
class TaskMHDSemanticSegmentation(task_lib.Task):
    def __init__(self,
                 config: ml_collections.ConfigDict):
        super().__init__(config)

        json_dict = task_utils.load_json(self.config.dataset.category_names_path)
        self._category_names = task_utils.get_category_names(json_dict)

        self.frame_id_to_img_info = {
            img['frame_id']: img for img in json_dict['images']
        }

        class_id_to_col, class_id_to_name = task_utils.get_class_info(self._category_names)

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

    def postprocess_response_seq(self, response_seq, token_type_id):
        config = self.config.task
        token_weights = tf.ones_like(response_seq, dtype=tf.float32)
        prompt_seq = task_utils.build_prompt_seq_from_task_id(
            task_vocab_id=self.task_vocab_id + token_type_id,
            response_seq=response_seq)
        input_seq = tf.concat([prompt_seq, response_seq], -1)
        target_seq = tf.concat([prompt_seq, response_seq], -1)
        input_seq, target_seq = input_seq[..., :-1], target_seq[..., 1:]
        token_weights = tf.where(
            target_seq == vocab.PADDING_TOKEN,
            tf.zeros_like(token_weights) + config.eos_token_weight,
            token_weights)

        return input_seq, target_seq, token_weights

    def preprocess_batched(self, batched_examples, training):
        rle_from_mask = self.config.dataset.rle_from_mask
        if self.config.debug == 2:
            batched_examples = vis_utils.debug_image_pipeline(
                self.dataset,
                self.train_transforms,
                batched_examples,
                vis=0,
                model_dir=self.config.model_dir,
                training=training)

        try:
            cls_eq = self.config.task.class_equal_weight
        except AttributeError:
            cls_eq = 0

        image = batched_examples['image']
        # batch_size = self.config.train.batch_size if training else self.config.eval.batch_size
        batch_size = tf.shape(image)[0]

        # height = batched_examples['image/height']
        # width = batched_examples['image/width']
        if rle_from_mask:
            mask = batched_examples['mask']
            with tf.device("CPU:0"):
                mask = tf.identity(mask)
                mask = tf.stop_gradient(mask)
            response_seqs = task_utils.mask_to_rle_tokens_tf(
                image, mask, self.config, self.class_id_to_col, mhd=1, training=training)
        else:
            response_seqs = batched_examples['rle']
        response_seq_x, response_seq_y, response_seq_l, response_seq_c = response_seqs

        input_seq_x, target_seq_x, token_weights_x = self.postprocess_response_seq(response_seq_x, 0)
        input_seq_y, target_seq_y, token_weights_y = self.postprocess_response_seq(response_seq_y, 1)

        input_seq_l, target_seq_l, token_weights_l = self.postprocess_response_seq(response_seq_l, 2)
        input_seq_c, target_seq_c, token_weights_c = self.postprocess_response_seq(response_seq_c, 3)

        """output goes to mhd_ar_model.compute_loss"""
        return (batched_examples,
                [input_seq_x, input_seq_y, input_seq_l, input_seq_c],
                [target_seq_x, target_seq_y, target_seq_l, target_seq_c],
                [token_weights_x, token_weights_y, token_weights_l, token_weights_c],
                )

    def infer(self, model, preprocessed_outputs):
        config = self.config.task
        mconfig = self.config.model
        examples = preprocessed_outputs[0]

        image = examples["image"]
        bsz = tf.shape(image)[0]

        prompt_seq_x = task_utils.build_prompt_seq_from_task_id(self.task_vocab_id, prompt_shape=(bsz, 1))
        prompt_seq_y = task_utils.build_prompt_seq_from_task_id(self.task_vocab_id + 1, prompt_shape=(bsz, 1))
        prompt_seq_l = task_utils.build_prompt_seq_from_task_id(self.task_vocab_id + 2, prompt_shape=(bsz, 1))
        prompt_seq_c = task_utils.build_prompt_seq_from_task_id(self.task_vocab_id + 3, prompt_shape=(bsz, 1))

        mhd_pred_seq, mhd_logits = model.infer(
            image, [prompt_seq_x, prompt_seq_y, prompt_seq_l, prompt_seq_c], encoded=None,
            max_seq_len=mconfig.max_seq_len + 1,
            temperature=config.temperature, top_k=config.top_k, top_p=config.top_p)

        return examples, mhd_pred_seq, mhd_logits

    def postprocess_tpu(self, batched_examples, mhd_pred_seq, mhd_logits, training=False):
        example = batched_examples
        images, image_id = example['image'], example['image/id']
        orig_image_size = example['orig_image_size']
        unpadded_image_size = example['unpadded_image_size']
        frame_id = example['frame_id']
        rle_from_mask = self.config.dataset.rle_from_mask

        if rle_from_mask:
            masks = batched_examples['mask']
            gt_rle, _ = task_utils.mask_to_rle_tokens_tf(
                images, masks, self.config, self.class_id_to_col, mhd=0, training=training)
        else:
            gt_rle = example['rle']

        vid_path = example['vid_path']
        mask_vid_path = example['mask_vid_path']

        return (images, image_id, frame_id,
                mhd_pred_seq, mhd_logits, gt_rle,
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
                        save_as_zip=False,
                        **kwargs
                        ):

        (images, image_ids, frame_ids,
         mhd_rles, mhd_logits,
         gt_rles,
         orig_sizes, unpadded_sizes,
         vid_paths, mask_vid_paths) = outputs

        rles_x, rles_y, rles_l, rles_c = mhd_rles
        logits_x, logits_y, logits_l, logits_c = mhd_logits

        rles_x = tf.identity(rles_x).numpy()
        rles_y = tf.identity(rles_y).numpy()
        rles_l = tf.identity(rles_l).numpy()
        rles_c = tf.identity(rles_c).numpy()

        logits_x = tf.identity(logits_x).numpy()
        logits_y = tf.identity(logits_y).numpy()
        logits_l = tf.identity(logits_l).numpy()
        logits_c = tf.identity(logits_c).numpy()

        images = tf.identity(images).numpy()
        image_ids = tf.identity(image_ids).numpy()
        frame_ids = tf.identity(frame_ids).numpy()
        gt_rles = tf.identity(gt_rles).numpy()
        orig_sizes = tf.identity(orig_sizes).numpy()
        unpadded_sizes = tf.identity(unpadded_sizes).numpy()
        vid_paths = tf.identity(vid_paths).numpy()
        mask_vid_paths = tf.identity(mask_vid_paths).numpy()

        image_ids = image_ids.flatten().astype(str)
        vid_paths = vid_paths.flatten().astype(str)
        mask_vid_paths = mask_vid_paths.flatten().astype(str)

        images = np.copy(tf.image.convert_image_dtype(images, tf.uint8))

        max_length = self.config.dataset.eval.max_length
        assert max_length > 0, "max_length must be > 0"
        assert self.config.dataset.train.max_length == max_length, "max_length mismatch"

        subsample = self.config.dataset.eval.subsample
        assert self.config.dataset.train.subsample == subsample, "subsample mismatch"

        allow_overlap = self.config.dataset.eval.allow_overlap

        multi_class = self.config.dataset.multi_class
        assert multi_class, "mhd is only implemented for multi_class case"

        n_classes = len(self.class_id_to_col)

        # assert subsample >= 1, "subsample must be >= 1"
        if not multi_class:
            assert n_classes == 2, "n_classes must be 2 for no multi_class"
        else:
            assert n_classes > 2, "n_classes must be > 2 for multi_class"

        starts_2d = self.config.dataset.starts_2d
        assert starts_2d, "mhd is only implemented for starts_2d"

        length_as_class = self.config.dataset.length_as_class
        assert not length_as_class, "mhd is not implemented for length_as_class"

        flat_order = self.config.dataset.flat_order

        starts_offset = self.config.model.coord_vocab_shift
        lengths_offset = self.config.model.len_vocab_shift
        class_offset = self.config.model.class_vocab_shift

        coord_vocab_size = self.config.model.coord_vocab_size
        len_vocab_size = self.config.model.len_vocab_size
        class_vocab_size = self.config.model.class_vocab_size

        max_seq_len = self.config.model.max_seq_len
        max_runs = self.config.model.max_runs

        vocab_ids_x = (starts_offset, coord_vocab_size - 1)
        vocab_ids_y = (starts_offset, coord_vocab_size - 1)
        vocab_ids_l = (lengths_offset, len_vocab_size - 1)
        vocab_ids_c = (class_offset, class_vocab_size - 1)

        if subsample > 1:
            max_length = int(max_length / subsample)

        for (image_id_, image_, frame_id_,
             rle_x_, rle_y_, rle_l_, rle_c_,
             logits_x_, logits_y_, logits_l_, logits_c_,
             orig_size_, gt_rle_, vid_path, mask_vid_path) in zip(
            image_ids, images, frame_ids,
            rles_x, rles_y, rles_l, rles_c,
            logits_x, logits_y, logits_l, logits_c,
            orig_sizes, gt_rles, vid_paths, mask_vid_paths, strict=True):

            assert '/' in image_id_, f"invalid image_id_: {image_id_}"

            """last element of image_id is the name of the image, everything else is the seq name"""
            seq = '/'.join(image_id_.split('/')[:-1])

            orig_size_ = tuple(orig_size_)
            n_rows, n_cols = orig_size_

            if subsample > 1:
                n_rows, n_cols = int(n_rows / subsample), int(n_cols / subsample)

            rle_x_, rle_y_, rle_l_, rle_c_ = task_utils.denoise_with_selective_argmax(
                (rle_x_, rle_y_, rle_l_, rle_c_),
                (logits_x_, logits_y_, logits_l_, logits_c_),
                (vocab_ids_x, vocab_ids_y, vocab_ids_l, vocab_ids_c),
                exclude_eos_padding=True
            )
            rle_x_, rle_y_, rle_l_, rle_c_ = task_utils.expand_to_max(
                (rle_x_, rle_y_, rle_l_, rle_c_),
                (logits_x_, logits_y_, logits_l_, logits_c_),
                (vocab_ids_x, vocab_ids_y, vocab_ids_l, vocab_ids_c),
            )

            # rle_x_, rle_y_, rle_l_, rle_c_ = task_utils.naive_denoise(rle_x_, rle_y_, rle_l_, rle_c_)

            rle_ = np.stack((rle_x_, rle_y_, rle_l_, rle_c_), axis=1)
            rle_ = rle_.flatten()

            gt_rle_ = gt_rle_[gt_rle_ != vocab.PADDING_TOKEN]

            mask_from_file = mask_from_file_sub = None
            mask_instance = mask_logits = rle_logits_cmp = None

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
                max_seq_len=None,
                diff_mask=False,
                n_classes=n_classes,
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
                ignore_invalid=False,
                max_seq_len=max_seq_len,
                diff_mask=False,
                n_classes=n_classes,
            )

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
                save_as_zip=save_as_zip,
            )

            seq_img_infos.append(
                img_info
            )

    def compute_scalar_metrics(self, step):
        raise AssertionError('not implemented')

    def reset_metrics(self):
        raise AssertionError('not implemented')
