import cv2
import ml_collections
import numpy as np
import utils
import vocab
from tasks import task as task_lib
from tasks import task_utils
from tasks.visualization import vis_utils
import tensorflow as tf


@task_lib.TaskRegistry.register('video_segmentation')
class TaskVideoSegmentation(task_lib.Task):
    def __init__(self,
                 config: ml_collections.ConfigDict):
        super().__init__(config)

        json_dict = task_utils.load_json(self.config.dataset.category_names_path)

        self._category_names = task_utils.get_category_names(json_dict)

        self.vid_id_to_info = {
            vid['id']: vid for vid in json_dict['videos']
        }
        class_id_to_col, class_id_to_name = task_utils.get_class_info(self._category_names)

        # class_info_path = config.dataset.get('class_info_path')
        # assert class_info_path, "class_info_path must be provided"
        # class_id_to_col, class_id_to_name = task_utils.read_class_info(class_info_path)

        # n_classes = len(self._category_names)
        # class_names_from_json = tuple(self._category_names[i]['name'] for i in range(n_classes))
        # assert class_names_from_json == class_names, "class_names mismatch"

        self.class_id_to_col = class_id_to_col
        self.class_id_to_name = class_id_to_name
        self.palette_flat = vis_utils.get_palette(class_id_to_col)

    def preprocess_single(self, dataset, batch_duplicates, training, validation):
        if self.config.debug != 2:
            """apply transforms"""
            dataset = dataset.map(
                lambda x: self.preprocess_single_example(
                    x, training, validation, batch_duplicates),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return dataset

    def check_video_rle(self, mask_vid_paths, videos, vid_ids, img_ids_all, frame_ids_all,
                        rles, rle_lens, n_runs, training):

        if isinstance(mask_vid_paths, str):
            videos = np.expand_dims(videos, axis=0)
            img_ids_all = np.expand_dims(img_ids_all, axis=0)
            frame_ids_all = np.expand_dims(frame_ids_all, axis=0)
            rles = np.expand_dims(rles, axis=0)

            mask_vid_paths = [mask_vid_paths, ]
            rle_lens = [rle_lens, ]
            vid_ids = [vid_ids, ]
            n_runs = [n_runs, ]

        batch_size = frame_ids_all.shape[0]

        if training:
            mode_cfg = self.config.dataset.train
        else:
            mode_cfg = self.config.dataset.eval

        max_length = mode_cfg.max_length
        subsample = mode_cfg.subsample

        multi_class = self.config.dataset.multi_class
        time_as_class = self.config.dataset.time_as_class
        length_as_class = self.config.dataset.length_as_class
        flat_order = self.config.dataset.flat_order

        starts_offset = self.config.model.coord_vocab_shift
        lengths_offset = self.config.model.len_vocab_shift
        class_offset = self.config.model.class_vocab_shift

        n_classes = len(self.class_id_to_col)
        max_seq_len = self.config.model.max_seq_len
        vocab_size = self.config.model.vocab_size

        class_id_to_col = self.class_id_to_col
        class_id_to_name = self.class_id_to_name

        allow_overlap = self.config.dataset.eval.allow_overlap

        vid_masks = []
        vid_masks_sub = []

        for batch_id in range(batch_size):
            n_runs_ = n_runs[batch_id]
            rle_len = rle_lens[batch_id]
            rle = rles[batch_id]
            rle_tokens = rle[rle != vocab.PADDING_TOKEN]
            if rle_len > max_seq_len:
                assert rle_tokens.size == max_seq_len, "curtailed RLE length mismatch"
                print(f'skipping curtailed rle with original length {rle_len}')
                continue

            mask_vid_path = mask_vid_paths[batch_id]
            if not isinstance(mask_vid_path, str):
                mask_vid_path = mask_vid_path.decode('utf-8')

            vid_id = vid_ids[batch_id]
            video = videos[batch_id]

            vid_info = self.vid_id_to_info[vid_id]
            seq = vid_info['seq']
            file_ids_from_json = vid_info['file_ids']
            img_ids_from_json = [f'{seq}/{file_id}' for file_id in file_ids_from_json]
            img_ids = list(img_ids_all[batch_id])
            assert img_ids == img_ids_from_json, "img_ids_from_json mismatch"

            frame_ids_from_json = list(map(int, vid_info['frame_ids']))
            frame_ids = list(frame_ids_all[batch_id])
            assert frame_ids == frame_ids_from_json, "frame_ids_from_json mismatch"

            vid_reader, vid_width, vid_height, num_frames = task_utils.load_video(mask_vid_path)
            n_rows, n_cols = vid_height, vid_width

            vid_mask = []
            vid_mask_sub = []
            for frame_id in frame_ids:
                mask = task_utils.read_frame(vid_reader, frame_id - 1, mask_vid_path)
                if not multi_class:
                    mask = task_utils.mask_to_binary(mask)
                mask = task_utils.mask_to_gs(mask)

                if subsample > 1:
                    mask_sub = task_utils.subsample_mask(mask, subsample, n_classes, is_vis=1)
                else:
                    mask_sub = np.copy(mask)
                vid_mask.append(mask)
                vid_mask_sub.append(mask_sub)
            vid_mask = np.stack(vid_mask, axis=0)
            vid_mask_sub = np.stack(vid_mask_sub, axis=0)

            assert rle_tokens.size == rle_len, "rle_len mismatch"

            if rle_len:
                task_utils.check_video_rle_ranges(
                    rle_tokens, n_runs_, multi_class, time_as_class, length_as_class,
                    vocab_size, starts_offset, lengths_offset, class_offset)

            task_utils.check_video_rle_tokens(
                video, vid_mask, vid_mask_sub,
                rle_tokens,
                n_classes=n_classes,
                length_as_class=length_as_class,
                starts_offset=starts_offset,
                time_as_class=time_as_class,
                lengths_offset=lengths_offset,
                class_offset=class_offset,
                max_length=max_length,
                subsample=subsample,
                class_id_to_name=class_id_to_name,
                class_id_to_col=class_id_to_col,
                multi_class=multi_class,
                flat_order=flat_order,
                is_vis=1,
                tac_mask_sub=None,
                tac_id_to_col=None,
                allow_overlap=allow_overlap,
                allow_extra=False,
            )
            vid_masks.append(vid_mask)
            vid_masks_sub.append(vid_mask_sub)

        return vid_masks, vid_masks_sub

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

        response_seq = batched_examples['rle']
        token_weights = tf.ones_like(response_seq, dtype=tf.float32)

        if self.config.debug:
            mask_vid_paths = batched_examples['mask_vid_path'].numpy().astype(str)
            videos = batched_examples['video'].numpy()

            img_ids_all = batched_examples['image_ids'].numpy().astype(str)
            frame_ids_all = batched_examples['frame_ids'].numpy()
            rles = batched_examples['rle'].numpy()
            rle_lens = batched_examples['rle_len'].numpy()
            n_runs = batched_examples['n_runs'].numpy()

            vid_ids = batched_examples['vid_id'].numpy()

            self.check_video_rle(mask_vid_paths, videos, vid_ids, img_ids_all, frame_ids_all,
                                 rles, rle_lens, n_runs, training=True)

        prompt_seq = task_utils.build_prompt_seq_from_task_id(
            task_vocab_id=self.task_vocab_id,
            response_seq=response_seq)
        input_seq = tf.concat([prompt_seq, response_seq], -1)
        target_seq = tf.concat([prompt_seq, response_seq], -1)

        """
        right shift the target_seq and left-shift the input_seq
        """
        input_seq, target_seq = input_seq[..., :-1], target_seq[..., 1:]

        """
        Assign lower weights for ending/padding tokens.
        eos_token_weight = 0.1
        """
        token_weights = tf.where(
            target_seq == vocab.PADDING_TOKEN,
            tf.zeros_like(token_weights) + config.eos_token_weight,
            token_weights)

        """goes into video_ar_model.compute_loss"""
        return batched_examples, input_seq, target_seq, token_weights

    def infer(self, model, preprocessed_outputs):
        config = self.config.task
        mconfig = self.config.model
        examples, input_seq, target_seq, token_weights = preprocessed_outputs
        video = examples["video"]
        bsz = tf.shape(video)[0]
        prompt_seq = task_utils.build_prompt_seq_from_task_id(
            self.task_vocab_id, prompt_shape=(bsz, 1))
        pred_seq, logits, _ = model.infer(
            video, prompt_seq, encoded=None,
            max_seq_len=mconfig.max_seq_len + 1,
            temperature=config.temperature, top_k=config.top_k, top_p=config.top_p)

        return examples, pred_seq, logits

    def postprocess_tpu(self, batched_examples, pred_rle, logits, training=False):
        example = batched_examples
        videos, image_ids, frame_ids = example['video'], example['image_ids'], example['frame_ids']
        vid_ids = example['vid_id']
        vid_paths = example['vid_path']
        seqs = example['seq']
        mask_vid_paths = example['mask_vid_path']
        orig_image_size = example['orig_image_size']

        gt_rle = example['rle']
        rle_len = example['rle_len']
        n_runs = example['n_runs']

        """goes to postprocess_cpu"""
        return (
            videos, vid_ids, image_ids, frame_ids, pred_rle, logits,
            gt_rle, rle_len, n_runs, orig_image_size, seqs,
            vid_paths, mask_vid_paths,
        )

    def postprocess_cpu(self,
                        outputs,
                        train_step,
                        out_vis_dir,
                        out_mask_dir,
                        out_mask_logits_dir,
                        det_vid_writers=None,
                        seg_vid_writers=None,
                        json_vid_info=None,
                        eval_step=None,
                        training=False,
                        show=False,
                        summary_tag='eval',
                        ret_results=False,
                        **kwargs
                        ):
        outputs_np = []
        for i in range(len(outputs)):
            outputs_np.append(tf.identity(outputs[i]).numpy())

        (videos, vid_ids, image_ids, frame_ids, rles, logits,
         gt_rles, gt_rle_lens, n_runs, orig_sizes, seqs,
         vid_paths, mask_vid_paths) = outputs_np

        # orig_sizes = orig_sizes.numpy()
        # gt_rles = gt_rles.numpy()
        # rles = rles.numpy()
        # logits = logits.numpy()

        image_ids = image_ids.astype(str)
        seqs = seqs.astype(str)
        vid_paths = vid_paths.astype(str)
        mask_vid_paths = mask_vid_paths.astype(str)

        # image_ids = task_utils.bytes_to_str_list(image_ids)
        # seqs = task_utils.bytes_to_str_list(seqs)
        # vid_paths = task_utils.bytes_to_str_list(vid_paths)
        # mask_vid_paths = task_utils.bytes_to_str_list(mask_vid_paths)

        videos = np.copy(tf.image.convert_image_dtype(videos, tf.uint8))

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

        time_as_class = self.config.dataset.time_as_class
        length_as_class = self.config.dataset.length_as_class
        starts_2d = self.config.dataset.starts_2d
        flat_order = self.config.dataset.flat_order

        starts_offset = self.config.model.coord_vocab_shift
        lengths_offset = self.config.model.len_vocab_shift
        class_offset = self.config.model.class_vocab_shift

        max_seq_len = self.config.model.max_seq_len
        vocab_size = self.config.model.vocab_size

        if subsample > 1:
            max_length = int(max_length / subsample)

        for (image_ids_, frame_ids_, video, vid_id, rle, logits_,
             orig_size, gt_rle, gt_rle_len, n_runs_, seq,
             vid_path, mask_vid_path) in (
                zip(image_ids, frame_ids, videos, vid_ids, rles, logits,
                    orig_sizes, gt_rles, gt_rle_lens, n_runs, seqs,
                    vid_paths, mask_vid_paths, strict=True)):

            orig_size = tuple(orig_size)
            n_rows, n_cols = orig_size
            if subsample > 1:
                n_rows, n_cols = int(n_rows / subsample), int(n_cols / subsample)

            gt_rle_tokens = gt_rle[gt_rle != vocab.PADDING_TOKEN]

            assert gt_rle_len > max_seq_len or len(gt_rle_tokens) == gt_rle_len, "gt_rle_len mismatch"

            # if gt_rle_len == 0:
            #     print('skipping empty mask')
            #     continue

            mask_from_file = mask_from_file_sub = None
            if self.config.debug:
                vid_masks, vid_masks_sub = self.check_video_rle(
                    mask_vid_path, video, vid_id, image_ids_, frame_ids_, gt_rle_tokens,
                    gt_rle_len, n_runs_, training=False)
                mask_from_file = vid_masks[0]
                mask_from_file_sub = vid_masks_sub[0]

            vid_len = video.shape[0]

            rle_tokens = rle[rle != vocab.PADDING_TOKEN]

            vid_mask_logits = [None, ] * vid_len
            rle_logits_len = 0

            if self.config.eval.mask_from_logits:
                vid_mask_logits, tac_mask_logits, rle_cmp_logits = task_utils.vid_mask_from_logits(
                    logits_,
                    (n_rows, n_cols),
                    max_length,
                    n_classes,
                    starts_offset, lengths_offset, class_offset,
                    time_as_class,
                    length_as_class,
                    starts_2d,
                    flat_order,
                    multi_class,
                    vid_len,
                    max_seq_len,
                    vocab_size,
                    allow_overlap,
                )
                rle_logits_len = len(rle_cmp_logits[0])

            vid_mask_rec, tac_mask_rec, rle_rec_cmp = task_utils.vid_mask_from_tokens(
                rle_tokens,
                allow_extra=True,
                vid_len=vid_len,
                shape=(n_rows, n_cols),
                length_as_class=length_as_class,
                max_length=max_length,
                starts_offset=starts_offset,
                lengths_offset=lengths_offset,
                class_offset=class_offset,
                starts_2d=False,
                multi_class=multi_class,
                flat_order=flat_order,
                time_as_class=time_as_class,
                n_classes=n_classes,
                ignore_invalid=True,
            )
            rle_rec_len = len(rle_rec_cmp[0])

            if show and gt_rle_len == 0 and rle_rec_len == 0 and rle_logits_len == 0:
                continue

            if gt_rle_len > 0 and rle_rec_len == 0:
                # print('empty pred rle')
                pass

            vid_mask_gt = [None, ] * vid_len

            if self.config.eval.mask_from_gt:
                is_curtailed = len(gt_rle_tokens) < gt_rle_len
                vid_mask_gt, tac_mask_gt, rle_gt_cmp = task_utils.vid_mask_from_tokens(
                    gt_rle_tokens,
                    allow_extra=is_curtailed,
                    vid_len=vid_len,
                    shape=(n_rows, n_cols),
                    length_as_class=length_as_class,
                    max_length=max_length,
                    starts_offset=starts_offset,
                    lengths_offset=lengths_offset,
                    class_offset=class_offset,
                    starts_2d=False,
                    multi_class=multi_class,
                    flat_order=flat_order,
                    time_as_class=time_as_class,
                    n_classes=n_classes,
                    ignore_invalid=False,
                )

                if self.config.debug:
                    # mask_from_file = task_utils.mask_vis_to_id(mask_from_file, n_classes, copy=True)
                    mask_from_file_sub = task_utils.mask_vis_to_id(mask_from_file_sub, n_classes, copy=True)
                    if not np.array_equal(mask_from_file_sub, vid_mask_gt):
                        print("vid_mask_gt mismatch")
                        task_utils.check_individual_vid_masks(
                            video, mask_from_file_sub, vid_mask_gt, self.class_id_to_col, n_classes)
                    # if show:
                    #     vid_mask_vis = task_utils.mask_id_to_vis_bgr(vid_mask_, self.class_id_to_col)
                    #     vid_mask_sub_vis = task_utils.mask_id_to_vis_bgr(vid_mask_sub_, self.class_id_to_col)
                    #     vid_mask_sub_vis = task_utils.resize_mask(vid_mask_sub_vis, vid_mask_.shape)
                    #     vid_mask_all = np.concatenate((vid_mask_vis, vid_mask_sub_vis), axis=1)
                    #     cv2.imshow('vid_mask_all', vid_mask_all)

            seq_img_infos = json_vid_info[seq]
            if seq_img_infos:
                out_frame_id = seq_img_infos[-1]['out_frame_id']
            else:
                out_frame_id = 0

            for image_id_, frame_id, image_, mask_rec, mask_logits, mask_gt in zip(
                    image_ids_, frame_ids_, video, vid_mask_rec, vid_mask_logits, vid_mask_gt, strict=True):
                out_frame_id += 1
                img_info = dict(
                    seq=str(seq),
                    vid_id=int(vid_id),
                    image_id=str(image_id_),
                    src_frame_id=int(frame_id),
                    out_frame_id=int(out_frame_id),
                    vid_path=str(vid_path),
                    mask_vid_path=str(mask_vid_path),
                )
                vis_utils.visualize_mask(
                    image_id_,
                    image_,
                    mask_rec,
                    mask_logits,
                    mask_gt,
                    self.class_id_to_col,
                    video_id=vid_id,
                    seq_id=seq,
                    img_info=img_info,
                    out_mask_dir=out_mask_dir,
                    out_mask_logits_dir=out_mask_logits_dir,
                    out_vis_dir=out_vis_dir,
                    vid_writers=det_vid_writers,
                    orig_size=orig_size,
                    show=show,
                    palette_flat=self.palette_flat,
                )
                seq_img_infos.append(
                    img_info
                )

    def compute_scalar_metrics(self, step):
        raise AssertionError('not implemented')

    def reset_metrics(self):
        raise AssertionError('not implemented')
