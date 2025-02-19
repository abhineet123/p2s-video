import copy
import os
import sys
import cv2
from tqdm import tqdm


def linux_path(*args, **kwargs):
    return os.path.join(*args, **kwargs).replace(os.sep, '/')


dproc_path = linux_path(os.path.expanduser("~"), "ipsc/ipsc_data_processing")
seg_path = linux_path(os.path.expanduser("~"), "617")

sys.path.append(os.getcwd())
sys.path.append(dproc_path)
sys.path.append(seg_path)

import numpy as np
import tensorflow as tf
import paramparse

from data.scripts import tfrecord_lib
from tasks import task_utils


class Params(paramparse.CFG):
    """
    :ivar subsample_method:
    1: create RLE of full-res mask and sample the starts and lengths thus generated
    2: decrease mask resolution by resizing and create RLE of the low-res mask
    """

    def __init__(self):
        paramparse.CFG.__init__(self, cfg_prefix='tf_vid_seg')
        self.class_names_path = ''
        self.db_path = ''
        self.db_suffix = ''
        self.vis = 0
        self.stats_only = 0
        self.rle_to_json = 1
        self.json_only = 0

        self.add_stride_info = 0

        self.check = 0
        self.load = 0
        self.save_json = 1

        self.excluded_src_ids = []

        self.pad_tokens = 0

        self.n_proc = 0
        self.ann_ext = 'json.gz'
        self.num_shards = 32
        self.output_dir = ''

        self.seq_id = -1
        self.seq_start_id = 0
        self.seq_end_id = -1

        self.patch_start_id = 0
        self.patch_end_id = -1

        self.n_rot = 0
        self.max_rot = 0
        self.min_rot = 0

        self.resize = 0
        self.start_id = 0
        self.end_id = -1

        self.patch_height = 0
        self.patch_width = 0

        self.max_stride = 0
        self.min_stride = 0

        self.enable_flip = 0

        self.flat_order = 'C'
        self.time_as_class = 0
        self.length_as_class = 0
        self.max_length = 0
        self.starts_2d = 0
        self.starts_offset = 1000
        self.lengths_offset = 200
        self.class_offset = 100
        self.subsample = 1
        self.subsample_method = 2

        self.show = 0
        self.vid = Params.Video()

    def process(self, n_classes):
        if self.patch_width <= 0:
            self.patch_width = self.patch_height

        if self.min_stride <= 0:
            self.min_stride = self.patch_height

        if self.max_stride <= self.min_stride:
            self.max_stride = self.min_stride

        if self.max_length <= 0:
            if self.flat_order == 'C':
                self.max_length = self.patch_width
            else:
                self.max_length = self.patch_height
                if not self.time_as_class:
                    self.max_length *= self.vid.length

        if self.seq_id >= 0:
            self.seq_start_id = self.seq_end_id = self.seq_id

        get_db_suffix(self)

        self.db_path = f'{self.db_path}-{self.db_suffix}'

        max_length = self.max_length
        if self.subsample > 1:
            max_length /= self.subsample

        n_classes_ = n_classes
        if self.time_as_class:
            n_classes_ = n_classes_ ** self.vid.length
        if self.length_as_class:
            n_total_classes = max_length * (n_classes_ - 1)
        else:
            n_total_classes = n_classes_

        if self.length_as_class:
            self.lengths_offset = self.class_offset
            min_starts_offset = n_total_classes + self.class_offset

            if self.starts_offset < min_starts_offset:
                import math
                min_starts_offset = int(math.ceil(min_starts_offset / 100) * 100)
                print(f'setting starts_offset to {min_starts_offset}')
                self.starts_offset = min_starts_offset
        else:
            min_lengths_offset = self.class_offset + n_total_classes
            if self.lengths_offset < min_lengths_offset:
                import math
                min_lengths_offset = int(math.ceil(min_lengths_offset / 100) * 100)
                print(f'setting lengths_offset to {min_lengths_offset}')
                self.lengths_offset = min_lengths_offset
            min_starts_offset = self.lengths_offset + max_length
            if self.starts_offset < min_starts_offset:
                import math
                min_starts_offset = int(math.ceil(min_starts_offset / 100) * 100)
                print(f'setting starts_offset to {min_starts_offset}')
                self.starts_offset = min_starts_offset

    class Video:
        def __init__(self):
            self.frame_gap = 1
            self.length = 2
            self.stride = 1
            self.sample = 0


def append_metrics(metrics, out):
    for metric, val in metrics.items():
        try:
            out[metric].append(val)
        except KeyError:
            out[metric] = [val, ]

    # vis_txt = ' '.join(f'{metric}: {val:.2f}' if isinstance(val, float)
    #                    else f'{metric}: {val}'
    #                    for metric, val in metrics.items())
    # return vis_txt


def eval_mask(pred_mask, gt_mask, rle_len):
    import densenet.evaluation.eval_segm as eval_segm
    class_ids = [0, 1]
    pix_acc = eval_segm.pixel_accuracy(pred_mask, gt_mask, class_ids)
    _acc, mean_acc = eval_segm.mean_accuracy(pred_mask, gt_mask, class_ids, return_acc=1)
    _IU, mean_IU = eval_segm.mean_IU(pred_mask, gt_mask, class_ids, return_iu=1)
    fw_IU = eval_segm.frequency_weighted_IU(pred_mask, gt_mask, class_ids)
    return dict(
        rle_len=rle_len,
        pix_acc=pix_acc,
        mean_acc=mean_acc,
        mean_IU=mean_IU,
        fw_IU=fw_IU,
    )


def save_vid_info_to_json(
        params: Params,
        videos: list,
        class_id_to_name: dict,
        class_id_to_col: dict,
        out_path: str,
        stride_to_video_ids=None):
    from datetime import datetime
    annotations = []
    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
    params_dict = paramparse.to_dict(params)
    info = {
        "version": "1.0",
        "year": datetime.now().strftime("%y"),
        "contributor": "asingh1",
        "date_created": time_stamp,
        "counts": dict(
            videos=len(videos),
            annotations=len(annotations),
        ),
        "params": params_dict,
    }
    licenses = [
        {
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "id": 1,
            "name": "Creative Commons Attribution 4.0 License"
        }
    ]
    categories = []
    for label_id, label in class_id_to_name.items():
        if label_id == 0:
            continue
        col = class_id_to_col[label_id]
        category_info = {
            'supercategory': 'object',
            'id': label_id,
            'name': label,
            'col': col,
        }
        categories.append(category_info)
    annotations = []
    json_dict = {
        "info": info,
        "licenses": licenses,
        "videos": videos,
        "categories": categories,
        "annotations": annotations,
    }
    if stride_to_video_ids is not None:
        json_dict['stride_to_video_ids'] = stride_to_video_ids

    n_vids = len(json_dict['videos'])
    print(f'saving json for {n_vids} videos to: {out_path}')
    json_kwargs = dict(
        indent=4
    )
    if out_path.endswith('.json'):
        import json
        output_json = json.dumps(json_dict, **json_kwargs)
        with open(out_path, 'w') as f:
            f.write(output_json)
    elif out_path.endswith('.json.gz'):
        import compress_json
        compress_json.dump(json_dict, out_path, json_kwargs=json_kwargs)
    else:
        raise AssertionError(f'Invalid out_path: {out_path}')


def load_img_info_from_json(annotation_path):
    print(f'Reading annotations from {annotation_path}')
    if annotation_path.endswith('.json'):
        import json
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
    elif annotation_path.endswith('.json.gz'):
        import compress_json
        annotations = compress_json.load(annotation_path)
    else:
        raise AssertionError(f'Invalid annotation_path: {annotation_path}')

    image_info = annotations['images']

    class_id_to_name = dict(
        (element['id'], element['name']) for element in annotations['categories'])

    try:
        bkg_class = class_id_to_name[0]
    except KeyError:
        pass
    else:
        assert bkg_class == 'background', "class id 0 can be used only for background"

    return image_info


def generate_patch_vid_infos(
        image_infos: list[dict],
        patch_start_id,
        patch_end_id,

        # vid_infos: dict,
):
    from collections import defaultdict
    patch_vids = defaultdict(list)
    # seq_names = list(vid_infos.keys()).sort()
    for image_info in image_infos:
        img_id = image_info['img_id']
        src_id, patch_id = img_id.split('_')
        seq_id = image_info['seq']

        patch_id = int(patch_id)

        image_info['src_id'] = src_id
        image_info['patch_id'] = patch_id

        if patch_id < patch_start_id > 0:
            continue

        if patch_id > patch_end_id > 0:
            continue

        patch_seq_id = f'{seq_id}_{patch_id}'
        patch_vids[patch_seq_id].append(image_info)
    return patch_vids


def generate_subseq_infos(
        patch_vids,
        length, stride, frame_gap,
        excluded_src_ids):
    all_subseq_img_infos = []
    videos = []
    vid_id = 0
    skipped = 0
    n_patch_vids = len(patch_vids)

    for _id, (patch_seq_id, patch_infos) in tqdm(enumerate(patch_vids.items()),
                                                 desc='generate_subseq_infos',
                                                 total=n_patch_vids):
        sorted(patch_infos, key=lambda x: int(x['frame_id']))

        n_all_files = len(patch_infos)
        # subseq_start_ids = list(range(0, n_all_files - params.vid.stride, params.vid.stride))
        subseq_end_ids = list(range(length - 1, n_all_files, stride))

        if subseq_end_ids[-1] != n_all_files - 1:
            subseq_end_ids.append(n_all_files - 1)

        subseq_end_ids = np.asarray(subseq_end_ids)
        subseq_start_ids = subseq_end_ids - (length - 1)

        n_subseq = len(subseq_end_ids)

        if _id == 0:
            print()

        for subseq_id, (subseq_start_id, subseq_end_id) in enumerate(
                zip(subseq_start_ids, subseq_end_ids, strict=True)):
            subseq_end_id_ = min(subseq_start_id + (length - 1) * frame_gap, n_all_files - 1)

            assert subseq_end_id == subseq_end_id_, "subseq_end_id_ mismatch"

            if subseq_start_id > subseq_end_id:
                break

            subseq_img_infos = patch_infos[subseq_start_id:subseq_end_id + 1:frame_gap]

            src_ids = tuple(image_info['src_id'] for image_info in subseq_img_infos)
            from itertools import chain, combinations
            src_ids_subsets = list(chain.from_iterable(combinations(src_ids, r)
                                                       for r in range(2, len(src_ids) + 1)))

            if any(src_ids_subset in excluded_src_ids for src_ids_subset in src_ids_subsets):
                # print(f'Skipping excluded src_ids: {src_ids}')
                skipped += 1
                continue

            n_subseq_files = len(subseq_img_infos)

            if n_subseq_files < length:
                # print(f'skipping subseq {subseq_id + 1} - with length {n_subseq_files}')
                skipped += 1
                continue

            img_info = subseq_img_infos[0]

            vid_w, vid_h, seq = img_info['width'], img_info['height'], img_info['seq']

            mask_file_names = [img_info['mask_file_name'] for img_info in subseq_img_infos]
            file_names = [img_info['file_name'] for img_info in subseq_img_infos]
            file_ids = [img_info['img_id'] for img_info in subseq_img_infos]
            frame_ids = [img_info['frame_id'] for img_info in subseq_img_infos]

            video_dict = {
                "width": vid_w,
                "height": vid_h,
                "length": length,
                "seq": seq,
                "date_captured": "",
                "license": 1,
                "flickr_url": "",
                "file_names": file_names,
                "mask_file_names": mask_file_names,
                "file_ids": file_ids,
                "frame_ids": frame_ids,
                "id": vid_id,
                "coco_url": "",
            }

            videos.append(video_dict)
            all_subseq_img_infos.append(subseq_img_infos)

            vid_id += 1

    if skipped > 0:
        print(f'skipped {skipped} videos')

    return all_subseq_img_infos, videos


def generate_annotations(
        params,
        skip_tfrecord,
        class_id_to_col,
        class_id_to_name,
        tac_id_to_col,
        tac_id_to_name,
        metrics,
        all_subseq_img_infos,
        videos,
        vid_infos,
):
    assert len(videos) == len(all_subseq_img_infos), "videos and all_subseq_img_infos must have the same length"
    for video, subseq_img_infos in zip(videos, all_subseq_img_infos):
        seq = subseq_img_infos[0]['seq']
        vid_id = video['id']

        yield (
            params,
            skip_tfrecord,
            class_id_to_col,
            class_id_to_name,
            tac_id_to_col,
            tac_id_to_name,
            metrics,
            vid_id,
            subseq_img_infos,
            video,
            seq,
            vid_infos[seq]
        )


def create_tf_example(
        params: Params,
        skip_tfrecord,
        class_id_to_col,
        class_id_to_name,
        tac_id_to_col,
        tac_id_to_name,
        metrics,
        vid_id,
        subseq_img_infos,
        video,
        seq,
        video_file_info):
    n_classes = len(class_id_to_col)
    vid_len = len(subseq_img_infos)
    frame_ids = []
    image_ids = []
    subseq_imgs = []
    subseq_masks = []
    subseq_masks_sub = []

    vid_reader, mask_vid_reader, vid_path, mask_vid_path, num_frames, vid_width, vid_height = video_file_info
    video_feature_dict = None

    if not skip_tfrecord:
        video_feature_dict = tfrecord_lib.video_seg_info_to_feature_dict(
            vid_id, vid_height, vid_width, vid_path, mask_vid_path,
            vid_len, seq)

    subsample_method = params.subsample_method
    max_length = params.max_length
    n_rows, n_cols = vid_height, vid_width

    vid = None

    multi_class = n_classes > 2

    start_filename = subseq_img_infos[0]['file_name']

    for _id, image_info in enumerate(subseq_img_infos):

        image_height = image_info['height']
        image_width = image_info['width']

        assert image_height == vid_height, "image_info height mismatch"
        assert image_width == vid_width, "image_info width mismatch"

        filename = image_info['file_name']
        image_id = image_info['img_id']
        seq_ = image_info['seq']

        assert seq == seq_, "seq mismatch"

        frame_id = int(image_info['frame_id'])
        mask_filename = image_info['mask_file_name']

        frame_ids.append(frame_id)
        image_ids.append(image_id)

        if params.subsample <= 1:
            subsample_method = 0

        # image_path = linux_path(params.db_path, filename)
        # mask_image_path = linux_path(params.db_path, mask_filename)

        if not image_id.startswith('seq'):
            image_id = f'{seq}/{image_id}'

        if not skip_tfrecord:
            image = task_utils.read_frame(vid_reader, frame_id - 1, vid_path)

            img_h, img_w = image.shape[:2]
            assert img_h == vid_height, "img_h mismatch"
            assert img_w == vid_width, "img_w mismatch"

            subseq_imgs.append(image)

            encoded_jpg = cv2.imencode('.jpg', image)[1].tobytes()
            video_frame_feature_dict = tfrecord_lib.video_seg_frame_info_to_feature_dict(
                _id, image_id, frame_id, filename, encoded_jpg, 'jpg')
            video_feature_dict.update(video_frame_feature_dict)

        mask = task_utils.read_frame(mask_vid_reader, frame_id - 1, mask_vid_path)

        if not multi_class:
            mask = task_utils.mask_to_binary(mask)
        mask = task_utils.mask_to_gs(mask)

        mask_h, mask_w = mask.shape
        assert mask_h == vid_height, "mask_h mismatch"
        assert mask_w == vid_width, "mask_w mismatch"

        if subsample_method == 2:
            mask_sub = task_utils.subsample_mask(mask, params.subsample, n_classes, is_vis=1)
        else:
            mask_sub = np.copy(mask)

        # mask_sub_vis = task_utils.resize_mask(mask_sub, mask.shape, n_classes)
        # mask_sub_vis = np.stack((mask_sub_vis,) * 3, axis=2)
        # mask_vis = np.stack((mask,) * 3, axis=2)
        # file_txt = f'{image_id}'
        # frg_col = (255, 255, 255)
        # concat_img = np.concatenate((image, mask_vis, mask_sub_vis), axis=1)
        # concat_img, _, _ = vis_utils.write_text(concat_img, file_txt, 5, 5, frg_col, font_size=24)
        # cv2.imshow('concat_img', concat_img)
        # cv2.waitKey(0)

        task_utils.mask_vis_to_id(mask, n_classes=n_classes)
        task_utils.mask_vis_to_id(mask_sub, n_classes=n_classes)

        subseq_masks.append(mask)
        subseq_masks_sub.append(mask_sub)
    # return
    if not skip_tfrecord:
        vid = np.stack(subseq_imgs, axis=0)

    vid_mask_orig = np.stack(subseq_masks, axis=0)
    vid_mask_sub_orig = np.stack(subseq_masks_sub, axis=0)
    tac_mask = tac_mask_sub = None
    if params.time_as_class:
        rle_id_to_col, rle_id_to_name = tac_id_to_col, tac_id_to_name
    else:
        rle_id_to_col, rle_id_to_name = class_id_to_col, class_id_to_name

    if params.load:
        rle_tokens = video["rle"]
        n_runs = video["n_runs"]
    else:
        if params.time_as_class:
            tac_mask = task_utils.vid_mask_to_tac(
                vid, vid_mask_orig, n_classes, class_id_to_col, params.check)
            tac_mask_sub = task_utils.vid_mask_to_tac(
                vid, vid_mask_sub_orig, n_classes, class_id_to_col, params.check)

            # vid_mask_rec = task_utils.vid_mask_from_tac(tac_mask, vid_len, n_classes)
            # vid_mask_sub_rec = task_utils.vid_mask_from_tac(tac_mask_sub, vid_len, n_classes)
            # assert np.array_equal(vid_mask, vid_mask_rec), "vid_mask_rec mismatch"
            # assert np.array_equal(vid_mask_sub, vid_mask_sub_rec), "vid_mask_rec mismatch"
            vid_mask = tac_mask
            vid_mask_sub = tac_mask_sub
            # return
        else:
            vid_mask = vid_mask_orig
            vid_mask_sub = vid_mask_sub_orig

        n_rle_classes = len(rle_id_to_col)

        if subsample_method == 2:
            max_length_sub = int(max_length / params.subsample)
            n_rows_sub, n_cols_sub = int(n_rows / params.subsample), int(n_cols / params.subsample)
        else:
            n_rows_sub, n_cols_sub = n_rows, n_cols
            max_length_sub = max_length

        starts, lengths = task_utils.mask_to_rle(
            mask=vid_mask_sub,
            max_length=max_length_sub,
            n_classes=n_rle_classes,
            order=params.flat_order,
        )
        rle_cmp = [starts, lengths]

        n_runs = len(starts)

        multi_class = False
        if n_rle_classes > 2:
            multi_class = True

            assert params.class_offset > 0, "class_offset must be > 0"

            class_ids = task_utils.get_rle_class_ids(
                vid_mask_sub, starts, lengths, rle_id_to_col,
                order=params.flat_order)

            rle_cmp.append(class_ids)
            if params.length_as_class:
                rle_cmp = task_utils.rle_to_lac(rle_cmp, max_length_sub)

        if params.vis:
            task_utils.vis_video_rle(
                rle_cmp,
                class_id_to_col, class_id_to_name,
                image_ids,
                vid, vid_mask, vid_mask_sub,
                params.time_as_class,
                params.length_as_class,
                max_length_sub,
                params.flat_order,
                rle_id_to_name,
                rle_id_to_col,
                params.pad_tokens,
            )

        rle_tokens = task_utils.rle_to_tokens(
            rle_cmp,
            vid_mask_sub.shape,
            params.length_as_class,
            params.starts_offset,
            params.lengths_offset,
            params.class_offset,
            params.starts_2d,
            params.flat_order,
        )

    if params.check:
        task_utils.check_video_rle_tokens(
            vid, vid_mask_orig, vid_mask_sub_orig,
            rle_tokens,
            n_classes=n_classes,
            length_as_class=params.length_as_class,
            starts_offset=params.starts_offset,
            time_as_class=params.time_as_class,
            lengths_offset=params.lengths_offset,
            class_offset=params.class_offset,
            max_length=max_length,
            subsample=params.subsample,
            class_id_to_name=class_id_to_name,
            class_id_to_col=class_id_to_col,
            multi_class=multi_class,
            flat_order=params.flat_order,
            tac_mask_sub=tac_mask_sub,
            tac_id_to_col=rle_id_to_col,
            is_vis=0,
            allow_overlap=False,
            allow_extra=False,
        )

    rle_len = len(rle_tokens)

    if multi_class:
        if params.length_as_class:
            tokens_per_run = 2
        else:
            tokens_per_run = 3
    else:
        if params.time_as_class:
            if params.length_as_class:
                tokens_per_run = 2
            else:
                tokens_per_run = 3
        else:
            tokens_per_run = 2

    assert rle_len % tokens_per_run == 0, f"rle_len must be divisible by {tokens_per_run}"

    assert rle_len == n_runs * tokens_per_run, "n_runs mismatch"

    if not params.load and params.rle_to_json:
        video['rle'] = rle_tokens
        video['rle_len'] = rle_len
        video['n_runs'] = n_runs

    elif not skip_tfrecord:
        seg_feature_dict = {
            # 'video/frame_ids': tfrecord_lib.convert_to_feature(frame_ids, value_type='int64_list'),
            # 'video/image_ids': tfrecord_lib.convert_to_feature(image_ids, value_type='bytes_list'),
            'video/rle': tfrecord_lib.convert_to_feature(rle_tokens, value_type='int64_list'),
            'video/rle_len': tfrecord_lib.convert_to_feature(rle_len),
        }
        video_feature_dict.update(seg_feature_dict)

    if rle_len > 0:
        append_metrics(
            dict(rle_len=f'{rle_len}'), metrics[f'db'])
        append_metrics(
            dict(vid_to_rle_len=f'{vid_id}\t{start_filename}\t{rle_len}'), metrics[f'db'])
    else:
        append_metrics(
            dict(empty_vid=f'{vid_id}\t{start_filename}'), metrics[f'db'])

    if not skip_tfrecord:
        video_feature_dict['video/n_runs'] = tfrecord_lib.convert_to_feature(n_runs, value_type='int64')
        example = tf.train.Example(features=tf.train.Features(feature=video_feature_dict))
        return example, 0


def get_vid_infos(image_infos, db_path):
    vid_infos = {}
    for image_info in tqdm(image_infos, desc="get_vid_infos"):
        seq = image_info['seq']
        mask_filename = image_info['mask_file_name']
        vid_path = linux_path(db_path, f'{seq}.mp4')
        mask_dir = os.path.dirname(mask_filename)
        mask_vid_path = linux_path(db_path, f'{mask_dir}.mp4')

        try:
            vid_info = vid_infos[seq]
        except KeyError:
            vid_reader, vid_width, vid_height, num_frames = task_utils.load_video(vid_path)
            mask_reader, mask_width, mask_height, mask_num_frames = task_utils.load_video(mask_vid_path)

            assert num_frames == mask_num_frames, "num_frames mismatch"
            assert vid_width == mask_width, "vid_width mismatch"
            assert vid_height == mask_height, "vid_height mismatch"

            vid_infos[seq] = vid_reader, mask_reader, vid_path, mask_vid_path, num_frames, vid_width, vid_height
        else:
            vid_reader, mask_reader, vid_path, mask_vid_path, num_frames, vid_width, vid_height = vid_info

        frame_id = int(image_info['frame_id'])
        img_id = image_info['img_id']

        assert frame_id <= num_frames, (f"frame_id {frame_id} for image {img_id} exceeds num_frames {num_frames} for "
                                        f"seq {seq}")
    return vid_infos


def get_db_suffix(params: Params):
    if not params.db_suffix:
        db_suffixes = []
        if params.resize:
            db_suffixes.append(f'resize_{params.resize}')

        db_suffixes += [f'{params.start_id:d}_{params.end_id:d}',
                        f'{params.patch_height:d}_{params.patch_width:d}',
                        f'{params.min_stride:d}_{params.max_stride:d}',
                        ]
        if params.n_rot > 0:
            db_suffixes.append(f'rot_{params.min_rot:d}_{params.max_rot:d}_{params.n_rot:d}')

        if params.enable_flip:
            db_suffixes.append('flip')

        params.db_suffix = '-'.join(db_suffixes)


def get_vid_suffix(vid_params: Params.Video):
    vid_suffixes = []
    assert vid_params.length > 1, "video length must be > 1"

    vid_suffixes.append(f'length-{vid_params.length}')

    if vid_params.stride:
        vid_suffixes.append(f'stride-{vid_params.stride}')

    if vid_params.sample:
        vid_suffixes.append(f'sample-{vid_params.sample}')

    if vid_params.frame_gap > 1:
        vid_suffixes.append(f'fg-{vid_params.frame_gap}')

    vid_suffix = '-'.join(vid_suffixes)
    return vid_suffix


def get_rle_suffix(params: Params, multi_class):
    rle_suffixes = []

    if params.subsample > 1:
        rle_suffixes.append(f'sub_{params.subsample}')

    if params.time_as_class:
        if params.length_as_class:
            rle_suffixes.append('ltac')
        else:
            rle_suffixes.append('tac')
    elif params.length_as_class:
        rle_suffixes.append('lac')

    if multi_class:
        rle_suffixes.append('mc')

    if params.flat_order != 'C':
        rle_suffixes.append(f'flat_{params.flat_order}')

    rle_suffix = '-'.join(rle_suffixes)
    return rle_suffix


def main():
    params: Params = paramparse.process(Params)

    assert params.db_path, "db_path must be provided"

    assert params.end_id >= params.start_id, f"invalid end_id: {params.end_id}"

    if params.stats_only == 2 or params.vis == 2:
        params.check = 0

    if params.stats_only:
        print('running in stats only mode')
        # params.vis = params.show = False

    if not params.check:
        print('RLE reconstruction check is disabled')

    class_id_to_col, class_id_to_name = task_utils.read_class_info(params.class_names_path)
    vid_len = params.vid.length
    # n_tac_classes = int(n_classes ** vid_len)
    tac_id_to_col, tac_id_to_name = task_utils.get_tac_info(vid_len, class_id_to_name)

    n_classes = len(class_id_to_col)
    multi_class = False
    if n_classes > 2:
        assert params.class_offset > 0, "class_offset must be > 0 for multi_class mode"
        multi_class = True

    params.process(n_classes)

    img_json_name = params.db_suffix
    if params.seq_start_id > 0 or params.seq_end_id >= 0:
        assert params.seq_end_id >= params.seq_start_id, "end_seq_id must to be >= start_seq_id"
        seq_suffix = f'seq_{params.seq_start_id}_{params.seq_end_id}'
        img_json_name = f'{img_json_name}-{seq_suffix}'

    img_json_fname = f'{img_json_name}.{params.ann_ext}'
    img_json_path = linux_path(params.db_path, img_json_fname)

    print("loading img_info_from_json")
    image_infos = load_img_info_from_json(img_json_path)

    vid_out_name = img_json_name
    """video-specific stuff"""
    vid_suffix = get_vid_suffix(params.vid)
    vid_out_name = f'{img_json_name}-{vid_suffix}'
    """RLE-specific stuff that doesn't go into output json since that doesn't contain RLE"""
    rle_suffix = get_rle_suffix(params, multi_class)

    rle_out_name = vid_out_name
    if rle_suffix:
        rle_out_name = f'{rle_out_name}-{rle_suffix}'

    if not params.output_dir:
        params.output_dir = linux_path(params.db_path, 'tfrecord')
    os.makedirs(params.output_dir, exist_ok=True)

    vid_json_path = os.path.join(params.db_path, f'{rle_out_name}.{params.ann_ext}')

    vid_infos = get_vid_infos(image_infos, params.db_path)

    patch_vids = generate_patch_vid_infos(
        image_infos,
        params.patch_start_id,
        params.patch_end_id
    )

    length = params.vid.length
    stride = params.vid.stride
    frame_gap = params.vid.frame_gap

    all_subseq_img_infos, videos = generate_subseq_infos(
        patch_vids,
        length, stride, frame_gap,
        params.excluded_src_ids
    )
    file_names = [tuple(video_['file_names']) for video_ in videos]
    file_names_unique = list(dict.fromkeys(file_names))
    assert file_names == file_names_unique, "file_names_unique mismatch"

    stride_to_video_ids = None
    if params.add_stride_info:
        stride_to_video_ids = {}

        video_ids = [video_['id'] for video_ in videos]
        stride_to_video_ids[stride] = video_ids

        file_names_to_vid_id = dict(
            (tuple(video_['file_names']), video_['id']) for video_ in videos
        )
        # vid_id_to_file_names = dict(
        #     (video_['id'], video_['file_names']) for video_ in videos
        # )
        for _stride in range(stride + 1, length + 1):
            _, _stride_videos = generate_subseq_infos(
                patch_vids,
                length, _stride, frame_gap,
                params.excluded_src_ids
            )
            _stride_file_names = [tuple(video_['file_names']) for video_ in _stride_videos]
            _stride_file_names_unique = list(dict.fromkeys(_stride_file_names))
            assert _stride_file_names == _stride_file_names_unique, "_stride_file_names_unique mismatch"

            _stride_video_ids = [file_names_to_vid_id[_stride_file_name]
                                 for _stride_file_name in _stride_file_names]
            _stride_video_ids_unique = list(dict.fromkeys(_stride_video_ids))
            assert _stride_video_ids == _stride_video_ids_unique, "_stride_video_ids_unique mismatch"

            stride_to_video_ids[_stride] = _stride_video_ids


        stride_to_video_ids = dict((_stride, ','.join(str(x) for x in _video_ids))
                                   for _stride, _video_ids in stride_to_video_ids.items())

    if params.load:
        print(f'loading vid json: {vid_json_path}')
        if params.ann_ext == 'json.gz':
            import compress_json
            vid_json_dict = compress_json.load(vid_json_path)
        else:
            import json
            with open(vid_json_path, 'r') as fid:
                vid_json_dict = json.load(fid)
        videos = vid_json_dict['videos']

        if params.check:
            n_tokens_per_run_gt = None
            for video in videos:
                n_runs = video['n_runs']
                rle_len = video['rle_len']
                rle = video['rle']
                assert len(rle) == rle_len, "rle_len mismatch"
                if n_runs == 0 or rle_len == 0:
                    assert n_runs == 0 and rle_len == 0, "n_runs and rle_len must both be zero or non-zero"
                else:
                    assert rle_len % n_runs == 0, "rle_len must be divisible by n_runs"
                    n_tokens_per_run = rle_len // n_runs
                    if n_tokens_per_run_gt is None:
                        n_tokens_per_run_gt = n_tokens_per_run
                    else:
                        assert n_tokens_per_run == n_tokens_per_run_gt, "n_tokens_per_run mismatch"
            # print()

    assert len(all_subseq_img_infos) == len(videos), "all_subseq_img_infos length mismatch"

    # if params.json_only:
    #     return

    metrics = dict(
        db={},
        method_0={},
        method_1={},
        method_2={},

    )
    if params.rle_to_json:
        print(f'writing RLE to json: {vid_json_path}')

    skip_tfrecord = params.stats_only or (params.rle_to_json and params.json_only) or params.add_stride_info == 2

    annotations_iter = generate_annotations(
        params=params,
        skip_tfrecord=skip_tfrecord,
        class_id_to_col=class_id_to_col,
        class_id_to_name=class_id_to_name,
        tac_id_to_col=tac_id_to_col,
        tac_id_to_name=tac_id_to_name,
        metrics=metrics,
        all_subseq_img_infos=all_subseq_img_infos,
        videos=videos,
        vid_infos=vid_infos,
    )

    tfrecord_path = linux_path(params.output_dir, vid_out_name if params.rle_to_json else rle_out_name)
    os.makedirs(tfrecord_path, exist_ok=True)

    if not params.load or params.check or not skip_tfrecord:
        if skip_tfrecord:
            print('skipping tfrecord creation')
            for idx, annotations_iter_ in tqdm(enumerate(annotations_iter),
                                               total=len(all_subseq_img_infos)):
                create_tf_example(*annotations_iter_)
        else:
            print(f'tfrecord_path: {tfrecord_path}')
            tfrecord_pattern = linux_path(tfrecord_path, 'shard')
            tfrecord_lib.write_tf_record_dataset(
                output_path=tfrecord_pattern,
                annotation_iterator=annotations_iter,
                process_func=create_tf_example,
                num_shards=params.num_shards,
                multiple_processes=params.n_proc,
                iter_len=len(all_subseq_img_infos),
            )

    if params.save_json:
        save_vid_info_to_json(params, videos, class_id_to_name, class_id_to_col, vid_json_path, stride_to_video_ids)

    metrics_dir = linux_path(params.db_path, '_metrics_')

    print(f'metrics_dir: {metrics_dir}')
    os.makedirs(metrics_dir, exist_ok=True)

    for method, metrics_ in metrics.items():

        for metric_, val in metrics_.items():
            metrics_path = linux_path(metrics_dir, f'{rle_out_name}-{method}-{metric_}.txt')
            with open(metrics_path, 'w') as f:
                f.write('\n'.join(map(str, val)))


if __name__ == '__main__':
    main()
