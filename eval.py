# from absl import logging
import os
import collections
import time
import json
import pickle
import pandas as pd
from tqdm import tqdm
from datetime import datetime

import utils

from tasks import task_utils
from tasks.visualization import vis_utils

from eval_utils import profile, print_with_time, linux_path


def run(cfg, dataset, task, eval_steps, ckpt, strategy, model, checkpoint, tf):
    """Perform evaluation."""
    eval_tag = cfg.eval.tag
    # summary_writer = None
    # summary_writer = tf.summary.create_file_writer(FLAGS.model_dir)

    is_video = 'video' in cfg.task.name
    is_seg = 'segmentation' in cfg.task.name

    with strategy.scope():
        # Restore model checkpoint.
        print_with_time(f'Restoring from {ckpt:s}')
        status = checkpoint.restore(ckpt).expect_partial()  # Not restore optimizer.
        verify_restored = status.assert_consumed
        verify_existing = status.assert_existing_objects_matched
        global_step = checkpoint.global_step
        print_with_time(f'Performing inference at step {global_step.numpy():d}')

    ckpt_name = os.path.splitext(os.path.basename(ckpt))[0]
    json_name = cfg.dataset.eval_filename_for_metrics

    assert json_name, "eval_filename_for_metrics must be provided for evaluation"
    json_name = os.path.basename(json_name).split(os.extsep)[0]

    out_dir = os.path.join(cfg.model_dir, f'{ckpt_name}-{json_name}')

    os.makedirs(out_dir, exist_ok=True)
    if is_seg:
        rle_lens = cfg.dataset.eval.rle_lens
        rle_lens_str = '\n'.join(rle_lens)
        rle_lens_path = os.path.join(out_dir, "rle_lens.txt")
        with open(rle_lens_path, 'w') as fid:
            fid.write(rle_lens_str)

    save_suffix = ''
    if cfg.eval.save_suffix:
        save_suffix = '-'.join(cfg.eval.save_suffix)

    csv_dir_name = f'csv'
    vis_dir_name = f'vis'
    mask_dir_name = 'masks'
    instance_dir_name = 'instances'
    mask_logits_dir_name = 'masks_logits'
    out_vis_dir = None

    if save_suffix:
        csv_dir_name = f'{csv_dir_name:s}-{save_suffix:s}'
        vis_dir_name = f'{vis_dir_name:s}-{save_suffix:s}'
        mask_dir_name = f'{mask_dir_name:s}-{save_suffix:s}'
        instance_dir_name = f'{instance_dir_name:s}-{save_suffix:s}'
        mask_logits_dir_name = f'{mask_logits_dir_name:s}-{save_suffix:s}'

    out_mask_dir = os.path.join(out_dir, mask_dir_name)
    out_instance_dir = os.path.join(out_dir, instance_dir_name)
    out_mask_logits_dir = os.path.join(out_dir, mask_logits_dir_name)
    out_csv_dir = os.path.join(out_dir, csv_dir_name)

    if is_seg:
        inference_dirs = [out_mask_dir,]
        if cfg.dataset.instance_wise:
            inference_dirs.append(out_csv_dir)
            inference_dirs.append(out_instance_dir)
    else:
        inference_dirs = [out_csv_dir,]

    inference_flags = [linux_path(inference_dir, '__inference') for inference_dir in inference_dirs]

    if all(os.path.exists(inference_flag) for inference_flag in inference_flags):
        timestamp = open(inference_flags[0], 'r').read()
        print(f'\n\nskipping inference completed previously at {timestamp}\n\n')
        return None

    out_json_name = f"vid_info.json.gz"
    out_json_path = os.path.join(out_dir, out_json_name)

    csv_columns = csv_exists = det_vid_writers = seg_vid_writers = seq_to_csv_rows = None

    if is_seg:
        if cfg.dataset.instance_wise:
            cfg.eval.mask_from_logits = 0

        if cfg.eval.save_mask:
            print(f'\nwriting masks to: {out_mask_dir}\n')
            os.makedirs(out_mask_dir, exist_ok=True)
            if cfg.eval.mask_from_logits:
                print(f'\nwriting logits masks to: {out_mask_logits_dir}\n')
                os.makedirs(out_mask_logits_dir, exist_ok=True)

            print(f'\nwriting vid info json to: {out_json_path}\n')

        if cfg.eval.write_to_video:
            seg_vid_writers = collections.defaultdict(lambda: None)

        if cfg.dataset.instance_wise:
            print(f'\nwriting instance masks to: {out_instance_dir}\n')
            os.makedirs(out_instance_dir, exist_ok=True)

            if cfg.eval.save_csv:
                print(f'\nwriting csv files to: {out_csv_dir}\n')
                os.makedirs(out_csv_dir, exist_ok=True)

                csv_columns = [
                    "ImageID", "LabelName",
                    "XMin", "XMax", "YMin", "YMax",
                    "mask_counts", "mask_h", "mask_w",
                    "Confidence",
                ]
                if is_video:
                    csv_columns.insert(1, 'VideoID')

                seq_to_csv_rows = dict()
                csv_exists = []

            if cfg.eval.write_to_video:
                det_vid_writers = collections.defaultdict(lambda: None)

    else:
        if cfg.eval.save_csv:
            print(f'\nwriting csv files to: {out_csv_dir}\n')
            os.makedirs(out_csv_dir, exist_ok=True)

            csv_columns = [
                "ImageID", "LabelName",
                "XMin", "XMax", "YMin", "YMax", "Confidence",
            ]
            if is_video:
                csv_columns.insert(1, 'VideoID')

            seq_to_csv_rows = dict()
            csv_exists = []

        if cfg.eval.write_to_video:
            det_vid_writers = collections.defaultdict(lambda: None)

    if cfg.eval.save_vis:
        out_vis_dir = os.path.join(out_dir, vis_dir_name)
        print(f'\nwriting vis images to: {out_vis_dir}\n')
        os.makedirs(out_vis_dir, exist_ok=True)

    json_vid_info = collections.defaultdict(list)
    if is_video and cfg.eval.add_stride_info:
        print(f'loading stride_to_video_ids from {cfg.dataset.category_names_path}')
        json_dict = task_utils.load_json(cfg.dataset.category_names_path)
        stride_to_video_ids = json_dict['stride_to_video_ids']
        json_vid_info['stride_to_video_ids'] = stride_to_video_ids

        try:
            stride_to_file_names = json_dict['stride_to_file_names']
        except KeyError:
            print('skipping stride_to_file_names')
        else:
            json_vid_info['stride_to_file_names'] = stride_to_file_names

    def single_step(examples):
        preprocessed_outputs = task.preprocess_batched(examples, training=False)
        infer_outputs = task.infer(model, preprocessed_outputs)
        postprocessed_outputs = task.postprocess_tpu(*infer_outputs)
        return postprocessed_outputs

    with strategy.scope():
        @tf.function
        def run_single_step(dataset_iter):
            examples = next(dataset_iter)
            # outputs = single_step(examples)
            outputs = strategy.run(single_step, (examples,))
            if outputs is not None:
                outputs = [strategy.gather(t, axis=0) for t in outputs]
            return outputs

        iterator = iter(dataset)
        start_time = timestamp = time.time()
        cur_step = 0
        img_id = 0

        # cfg.eager = 1

        # print(f'min_score_thresh: {cfg.eval.min_score_thresh}')
        pbar = tqdm(total=eval_steps, ncols=120, position=0, leave=True)

        while True:
            if eval_steps and cur_step >= eval_steps:
                break

            per_step_outputs = None

            # if cur_step == 0 and os.path.isfile('per_step_outputs.pkl'):
            #     print('loading per_step_outputs')
            #     with open('per_step_outputs.pkl', 'rb') as fid:
            #         per_step_outputs = pickle.load(fid)

            if per_step_outputs is None:
                if cfg.eager:
                    enable_profiling = cfg.eval.profile
                    _times = collections.OrderedDict()
                    _rel_times = collections.OrderedDict()
                    with profile('iterator', _times, _rel_times, enable_profiling, show=True):
                        examples = next(iterator)
                    with profile('preprocess_batched', _times, _rel_times, enable_profiling, show=True):
                        preprocessed_outputs = task.preprocess_batched(examples, training=False)
                    with profile('infer', _times, _rel_times, enable_profiling, show=True):
                        infer_outputs = task.infer(model, preprocessed_outputs)
                    with profile('postprocess_tpu', _times, _rel_times, enable_profiling, show=True):
                        per_step_outputs = task.postprocess_tpu(*infer_outputs)

                    if enable_profiling:
                        print(f'times: {_times}')
                        print(f'rel_times: {_rel_times}')
                else:
                    per_step_outputs = run_single_step(iterator)

            # with open('per_step_outputs.pkl', 'wb') as fid:
            #     pickle.dump(per_step_outputs, fid)

            if cfg.eval.check_ckpt and cur_step == 0:
                utils.check_checkpoint_restored(
                    strict_verifiers=(),
                    loose_verifiers=[verify_restored, verify_existing],
                )

            cur_step += 1
            time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")

            pbar.set_description(time_stamp)
            pbar.update(1)

            # if eval_steps:
            #     steps_per_sec = 1. / (time.time() - timestamp)
            #     timestamp = time.time()
            #     progress = cur_step / float(eval_steps) * 100
            #     eta = (eval_steps - cur_step) / steps_per_sec / 60.
            #     print_with_time(f'Completed: {cur_step} / {eval_steps} steps ({progress:.2f}%), ETA {eta:.2f} mins')
            # else:
            #     print_with_time(f'Completed: {cur_step:d} steps')

            task.postprocess_cpu(
                outputs=per_step_outputs,
                train_step=global_step.numpy(),
                out_mask_dir=out_mask_dir,
                out_instance_dir=out_instance_dir,
                out_mask_logits_dir=out_mask_logits_dir,
                out_vis_dir=out_vis_dir,
                show=cfg.eval.show_vis,
                json_vid_info=json_vid_info,
                det_vid_writers=det_vid_writers,
                seg_vid_writers=seg_vid_writers,
                csv_data=seq_to_csv_rows,
                eval_step=cur_step,
                summary_tag=eval_tag,
                ret_results=False,
            )

            if seq_to_csv_rows is not None and (
                    (eval_steps and cur_step >= eval_steps) or (
                    cfg.eval.csv_steps > 0 and cur_step % cfg.eval.csv_steps == 0)):
                for seq_id, csv_rows in seq_to_csv_rows.items():
                    out_csv_path = os.path.join(out_csv_dir, f"{seq_id}.csv")

                    if seq_id not in csv_exists:
                        pd.DataFrame([], columns=csv_columns).to_csv(out_csv_path, index=False)
                        csv_exists.append(seq_id)

                    if not csv_rows:
                        continue

                    # print(f'{csv_seq_name} :: saving csv to {out_csv_path}')
                    df = pd.DataFrame(csv_rows, columns=csv_columns)
                    df.to_csv(out_csv_path, index=False, mode='a', header=False)

                    seq_to_csv_rows[seq_id] = []

        print_with_time(f'Finished eval in {(time.time() - start_time) / 60.:.2f} mins')

    if det_vid_writers is not None:
        print(f'closing det_vid_writers')
        for seq_name, vid_writers in det_vid_writers.items():
            vis_utils.close_video_writers(vid_writers)

    if seg_vid_writers is not None:
        print(f'closing seg_vid_writers')
        for seq_name, vid_writers in seg_vid_writers.items():
            vis_utils.close_video_writers(vid_writers)

    if is_seg or is_video:
        json_kwargs = dict(
            indent=4
        )
        # print(f'saving vid info json to {out_json_path}')
        import compress_json
        compress_json.dump(json_vid_info, out_json_path, json_kwargs=json_kwargs)

    if seq_to_csv_rows is not None:
        for csv_seq_name, csv_rows in seq_to_csv_rows.items():
            assert not csv_rows, "unexplained non-empty csv_rows found"

    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    for inference_flag in inference_flags:
        with open(inference_flag, 'w') as f:
            f.write(time_stamp + '\n')

    return out_dir
