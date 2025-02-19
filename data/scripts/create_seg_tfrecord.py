import collections
import json
import os
import shutil
import sys
import math
import cv2
from tqdm import tqdm

dproc_path = os.path.join(os.path.expanduser("~"), "ipsc/ipsc_data_processing")
seg_path = os.path.join(os.path.expanduser("~"), "617")
sys.path.append(os.getcwd())
sys.path.append(dproc_path)
sys.path.append(seg_path)

import numpy as np
import tensorflow as tf
import paramparse

import vocab
from data.scripts import tfrecord_lib
from tasks.visualization import vis_utils
from tasks import task_utils
from tasks import quadtree

from eval_utils import add_suffix, mask_rle_to_img, linux_path


class Params(paramparse.CFG):
    """
    :ivar subsample:
    factor by which to reduce mask resolution

    :ivar subsample_method:
    1: create RLE of full-res mask and sample the starts and lengths thus generated
    2: decrease mask resolution by resizing and create RLE of the low-res mask
    """

    def __init__(self):
        paramparse.CFG.__init__(self, cfg_prefix='tf_seg')
        self.class_names_path = ''
        self.db_path = ''
        self.db_suffix = ''
        self.vis = 0
        self.stats_only = 0
        self.rle_to_json = 1
        self.json_only = 0
        self.check = 0

        self.load_vid = 1

        self.flat_order = 'C'

        self.seg_metrics = 0
        self.poly_len = 0

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

        self.length_as_class = 0

        self.max_length = 0
        self.starts_2d = 0
        self.starts_offset = 1000
        self.lengths_offset = 100
        self.class_offset = 0
        self.shared_coord = 0
        self.subsample = 1
        self.subsample_method = 2

        self.rle_from_mask = 0

        self.class_wise = 0

        self.instance_wise = 0
        self.instance_coco_rle = 0

        self.show = 0


def get_rle_suffix(params: Params, n_classes, multi_class):
    rle_suffixes = []

    if params.subsample > 1:
        rle_suffixes.append(f'sub_{params.subsample}')

    if params.starts_2d:
        rle_suffixes.append(f'2d')

    if params.shared_coord:
        assert  params.starts_2d, "shared_coord should only be used with 2D starts"
        rle_suffixes.append(f'sco')

    max_length = params.max_length
    if params.subsample > 1:
        max_length /= params.subsample

    n_classes_ = n_classes

    if params.length_as_class:
        assert multi_class, "length_as_class can be enabled only in multi_class mode"
        assert not params.shared_coord, "shared coord tokens cannot be used with LAC"

        params.lengths_offset = params.class_offset
        rle_suffixes.append(f'lac')
        n_total_classes = max_length * (n_classes_ - 1)
        min_starts_offset = n_total_classes + params.class_offset

        if params.starts_offset < min_starts_offset:
            min_starts_offset = int(math.ceil(min_starts_offset / 100) * 100)
            print(f'setting starts_offset to {min_starts_offset}')
            params.starts_offset = min_starts_offset
    else:
        n_total_classes = n_classes_
        min_lengths_offset = params.class_offset + n_total_classes
        if params.lengths_offset < min_lengths_offset:
            min_lengths_offset = int(math.ceil(min_lengths_offset / 100) * 100)
            print(f'setting lengths_offset to {min_lengths_offset}')
            params.lengths_offset = min_lengths_offset

        if params.shared_coord:
            print(f'setting starts_offset to { params.lengths_offset} for shared coord tokens')
            params.starts_offset = params.lengths_offset
        else:
            min_starts_offset = params.lengths_offset + max_length
            if params.starts_offset < min_starts_offset:
                min_starts_offset = int(math.ceil(min_starts_offset / 100) * 100)
                print(f'setting starts_offset to {min_starts_offset}')
                params.starts_offset = min_starts_offset

        if multi_class:
            rle_suffixes.append(f'mc')

    if params.flat_order != 'C':
        rle_suffixes.append(f'flat_{params.flat_order}')

    if params.class_wise:
        rle_suffixes.append('cw')

    rle_suffix = '-'.join(rle_suffixes) if rle_suffixes else ''
    return rle_suffix


def append_metrics(metrics, out):
    for metric, val in metrics.items():
        try:
            if isinstance(val, (list, tuple)):
                out[metric] += val
            else:
                out[metric].append(val)
        except KeyError:
            if isinstance(val, (list, tuple)):
                out[metric] = list(val)
            else:
                out[metric] = [val, ]

    # vis_txt = ' '.join(f'{metric}: {val:.2f}' if isinstance(val, float)
    #                    else f'{metric}: {val}'
    #                    for metric, val in metrics.items())
    return


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


def get_vid_infos(db_path, image_infos, instance_wise, instance_coco_rle):
    vid_infos = {}

    # frame_ids = set([int(image_info['frame_id']) for image_info in image_infos])

    for image_info in image_infos:
        # assert 'patch_info' in image_info, "patch_info not found in image_info"

        seq = image_info['seq']
        try:
            vid_info = vid_infos[seq]
        except KeyError:

            vid_path = linux_path(db_path, f'{seq}.mp4')

            mask_filename = image_info['mask_file_name']
            mask_dir = os.path.dirname(mask_filename)
            mask_vid_path = linux_path(db_path, f'{mask_dir}.mp4')

            vid_reader, vid_width, vid_height, num_frames = task_utils.load_video(vid_path)
            mask_reader, mask_width, mask_height, mask_num_frames = task_utils.load_video(mask_vid_path)

            assert num_frames == mask_num_frames, "mask_num_frames mismatch"
            assert vid_width == mask_width, "mask_width mismatch"
            assert vid_height == mask_height, "mask_height mismatch"

            vid_infos[seq] = {
                'vid': (vid_reader, vid_path, num_frames, vid_width, vid_height),
                'mask': (mask_reader, mask_vid_path),
            }
            if instance_wise and not instance_coco_rle:
                instance_filename = image_info['instance_file_name']
                instance_dir = os.path.dirname(instance_filename)
                instance_vid_path = linux_path(db_path, f'{instance_dir}.mp4')

                instance_reader, instance_width, instance_height, instance_num_frames = task_utils.load_video(
                    instance_vid_path)
                assert num_frames == instance_num_frames, "instance_num_frames mismatch"
                assert vid_width == instance_width, "instance_width mismatch"
                assert vid_height == instance_height, "instance_height mismatch"

                vid_infos[seq]['instance'] = (instance_reader, instance_vid_path)
        else:
            vid_reader, vid_path, num_frames, vid_width, vid_height = vid_info['vid']

        frame_id = int(image_info['frame_id'])
        img_id = image_info['img_id']

        assert frame_id <= num_frames, (f"frame_id {frame_id} for image {img_id} exceeds num_frames {num_frames} for "
                                        f"seq {seq}")

    return vid_infos


def save_seg_annotations(
        params: Params,
        img_json_dict: dict,
        class_id_to_name: dict,
        class_id_to_col: dict,
        out_path: str):
    from datetime import datetime

    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
    params_dict = paramparse.to_dict(params)
    n_imgs = len(img_json_dict['images'])
    info = {
        "version": "1.0",
        "year": datetime.now().strftime("%y"),
        "contributor": "asingh1",
        "date_created": time_stamp,
        "counts": dict(
            images=n_imgs,
            annotations=0,
        ),
        "params": params_dict,
    }
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

    img_json_dict["info"] = info
    img_json_dict["categories"] = categories

    print(f'saving json for {n_imgs} images to: {out_path}')
    json_kwargs = dict(
        indent=4
    )
    if out_path.endswith('.json'):
        import json
        output_json = json.dumps(img_json_dict, **json_kwargs)
        with open(out_path, 'w') as f:
            f.write(output_json)
    elif out_path.endswith('.json.gz'):
        import compress_json
        compress_json.dump(img_json_dict, out_path, json_kwargs=json_kwargs)
    else:
        raise AssertionError(f'Invalid out_path: {out_path}')


def load_seg_annotations(annotation_path):
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

    class_id_to_name = dict(
        (element['id'], element['name']) for element in annotations['categories'])

    try:
        bkg_class = class_id_to_name[0]
    except KeyError:
        pass
    else:
        assert bkg_class == 'background', "class id 0 must be used only for background"

    return annotations


def generate_annotations(
        params,
        class_id_to_col,
        class_id_to_name,
        metrics,
        image_infos,
        vid_infos,
        skip_tfrecord,
):
    for image_info in image_infos:
        seq = image_info['seq']
        seq_id = image_info['seq']
        img_id = image_info['img_id']

        src_id, patch_id = img_id.split('_')[:2]

        patch_id = int(patch_id)

        image_info['src_id'] = src_id
        image_info['patch_id'] = patch_id

        if patch_id < params.patch_start_id > 0:
            continue

        if patch_id > params.patch_end_id > 0:
            continue

        vid_info = vid_infos[seq] if vid_infos is not None else None
        yield (
            params,
            class_id_to_col,
            class_id_to_name,
            metrics,
            image_info,
            vid_info,
            skip_tfrecord,
        )


def get_rle_tokens(
        params, image, mask, mask_sub,
        max_length, max_length_sub, n_classes, subsample_method,
        n_rows, n_cols,
        class_id_to_col, class_id_to_name,
):
    starts, lengths = task_utils.mask_to_rle(
        mask=mask_sub,
        max_length=max_length_sub,
        n_classes=n_classes,
        order=params.flat_order,
    )
    if subsample_method == 1:
        """subsample RLE of high-res mask"""
        starts, lengths = task_utils.subsample_rle(
            starts, lengths,
            subsample=params.subsample,
            shape=(n_rows, n_cols),
            max_length=max_length,
            flat_order=params.flat_order,
        )

    rle_cmp = [starts, lengths]

    n_runs = len(starts)

    multi_class = n_classes > 2

    if multi_class:
        assert params.class_offset > 0, "class_offset must be > 0"
        class_ids = task_utils.get_rle_class_ids(mask_sub, starts, lengths, class_id_to_col,
                                                 order=params.flat_order)
        rle_cmp.append(class_ids)

        if params.length_as_class:
            rle_cmp = task_utils.rle_to_lac(rle_cmp, max_length_sub)

    if params.vis and n_runs > 0:
        task_utils.vis_rle(
            rle_cmp,
            params.length_as_class,
            max_length_sub,
            class_id_to_col, class_id_to_name,
            image, mask, mask_sub,
            flat_order=params.flat_order)

    rle_tokens = task_utils.rle_to_tokens(
        rle_cmp, mask_sub.shape,
        params.length_as_class,
        params.starts_offset,
        params.lengths_offset,
        params.class_offset,
        params.starts_2d,
        params.flat_order,
    )
    rle_len = len(rle_tokens)

    n_tokens_per_run = 2
    if params.starts_2d:
        n_tokens_per_run += 1
    if multi_class and not params.length_as_class:
        n_tokens_per_run += 1

    assert rle_len % n_tokens_per_run == 0, f"rle_len must be divisible by {n_tokens_per_run}"
    assert n_runs * n_tokens_per_run == rle_len, f"mismatch between n_runs and rle_len"

    if params.check:
        task_utils.check_rle_tokens(
            image, mask, mask_sub, rle_tokens, n_classes,
            params.length_as_class,
            params.starts_2d,
            params.starts_offset,
            params.lengths_offset,
            params.class_offset,
            max_length,
            params.subsample,
            multi_class,
            params.flat_order,
            class_id_to_col,
            is_vis=False)
    return rle_tokens, n_runs, rle_len


def create_tf_example(
        params: Params,
        class_id_to_col: dict,
        class_id_to_name: dict,
        metrics: dict,
        image_info: dict,
        vid_info: dict,
        skip_tfrecord: dict,
):
    n_classes = len(class_id_to_col)
    multi_class = n_classes > 2

    image_height = image_info['height']
    image_width = image_info['width']
    filename = image_info['file_name']
    image_id = image_info['img_id']
    seq = image_info['seq']
    frame_id = int(image_info['frame_id'])

    subsample_method = params.subsample_method
    if params.subsample <= 1:
        subsample_method = 0

    image_path = linux_path(params.db_path, filename)
    mask_filename = image_info['mask_file_name']
    mask_image_path = linux_path(params.db_path, mask_filename)

    if not image_id.startswith('seq'):
        image_id = f'{seq}/{image_id}'

    image = encoded_jpg = feature_dict = None

    read_image = params.show or params.check or not skip_tfrecord

    if not skip_tfrecord:
        feature_dict = {}

    if params.instance_wise:
        instance_to_target_id: dict = image_info['instance_to_target_id']
        instance_to_target_id = {
            int(instance_id_): (int(target_id_), int(class_id_))
            for instance_id_, (target_id_, class_id_) in
            instance_to_target_id.items()
        }
        instance_ids_1 = sorted(list(instance_to_target_id.keys()))
        n_instances = len(instance_to_target_id)

        if params.instance_coco_rle:
            instance_to_rle_coco = image_info['instance_to_rle_coco']
            instance_to_rle_coco = {
                int(instance_id_): rle_coco
                for instance_id_, rle_coco in
                instance_to_rle_coco.items()
            }
        else:
            instance_filename = image_info['instance_file_name']
            instance_image_path = linux_path(params.db_path, instance_filename)
            instance_id_to_col: dict = image_info['instance_id_to_col']

            instance_id_to_col = {
                int(instance_id_): int(col)
                for instance_id_, col in
                instance_id_to_col.items()
            }

            instance_ids_2 = sorted(list(instance_id_to_col.keys()))
            instance_cols = sorted(list(instance_id_to_col.values()))

            assert instance_ids_1 == instance_ids_2, "instance_ids_2 mismatch"

            instance_col_to_id = {v: k for k, v in instance_id_to_col.items()}

    if vid_info is not None:
        vid_reader, vid_path, num_frames, vid_width, vid_height = vid_info['vid']
        mask_vid_reader, mask_vid_path = vid_info['mask']
        if read_image:
            image = task_utils.read_frame(vid_reader, frame_id - 1, vid_path)

        if not skip_tfrecord:
            vid_feature_dict = {
                'image/seq': tfrecord_lib.convert_to_feature(seq.encode('utf8')),
                'image/vid_path': tfrecord_lib.convert_to_feature(vid_path.encode('utf8')),
                'image/mask_vid_path': tfrecord_lib.convert_to_feature(mask_vid_path.encode('utf8')),
            }
            feature_dict.update(vid_feature_dict)
            # from PIL import Image
            # from io import BytesIO
            # buffer = BytesIO()
            # Image.fromarray(image).save(buffer, format="JPEG")
            # encoded_jpg = buffer.getvalue()
            encoded_jpg = cv2.imencode('.jpg', image)[1].tobytes()
            # encoded_png = cv2.imencode('.png', mask)[1].tobytes()

        mask = task_utils.read_frame(mask_vid_reader, frame_id - 1, mask_vid_path)
    else:
        if not skip_tfrecord:
            with tf.io.gfile.GFile(image_path, 'rb') as fid:
                encoded_jpg = fid.read()

            if read_image:
                image = cv2.imread(image_path)

        mask = cv2.imread(mask_image_path)

    # mask_orig = np.copy(mask)
    # mask_orig = task_utils.mask_to_gs(mask_orig)

    if not multi_class:
        mask = task_utils.mask_to_binary(mask)
    mask = task_utils.mask_to_gs(mask)

    # mask_unique = np.unique(mask)

    n_rows, n_cols = mask.shape
    max_length = params.max_length

    if subsample_method == 2:
        """        
        decrease mask resolution by resizing and create RLE of the low-res mask
        """
        max_length_sub = int(max_length / params.subsample)
        n_rows_sub, n_cols_sub = int(n_rows / params.subsample), int(n_cols / params.subsample)

        mask_sub = task_utils.resize_mask(mask, (n_rows_sub, n_cols_sub))
        # mask_sub = task_utils.subsample_mask(mask, params.subsample, n_mask_classes, is_vis=1)
    else:
        mask_sub = np.copy(mask)
        n_rows_sub, n_cols_sub = n_rows, n_cols
        max_length_sub = max_length

    mask = task_utils.mask_vis_to_id(mask, n_classes=n_classes)
    mask_sub = task_utils.mask_vis_to_id(mask_sub, n_classes=n_classes)

    if not skip_tfrecord:
        image_feature_dict = tfrecord_lib.image_info_to_feature_dict(
            image_height, image_width, filename, image_id, encoded_jpg, 'jpg')
        if params.rle_from_mask:
            encoded_png = cv2.imencode('.png', mask)[1].tobytes()
            image_feature_dict.update({
                'mask/encoded': tfrecord_lib.convert_to_feature(encoded_png),
                'mask/format': tfrecord_lib.convert_to_feature('png'.encode('utf8')),
            })
        feature_dict.update(image_feature_dict)

    class_token_idxs = []
    if params.instance_wise:
        if params.instance_coco_rle:
            # instance_mask = np.zeros_like(mask)
            instance_ids = sorted(list(instance_to_rle_coco.keys()))
            # for instance_id, rle_coco in instance_to_rle_coco.items():
            #     instance_binary_mask = mask_rle_to_img(rle_coco)
            #     instance_mask[instance_binary_mask] = instance_id
            # if subsample_method == 2:
            #     instance_mask_sub = task_utils.resize_mask(instance_mask, (n_rows_sub, n_cols_sub))
            # else:
            #     instance_mask_sub = np.copy(instance_mask)

        else:
            if vid_info is not None:
                instance_vid_reader, instance_vid_path = vid_info['instance']
                instance_mask = task_utils.read_frame(instance_vid_reader, frame_id - 1, instance_vid_path)
                feature_dict.update({
                    'image/instance_vid_path': tfrecord_lib.convert_to_feature(instance_vid_path.encode('utf8')),
                })
            else:
                instance_mask = cv2.imread(instance_image_path)

            instance_cols_from_mask = np.unique(instance_mask).tolist()
            assert instance_cols == instance_cols_from_mask, "instance_cols_from_mask mismatch"
            instance_mask = task_utils.mask_to_gs(instance_mask)
            if subsample_method == 2:
                instance_mask_sub = task_utils.resize_mask(instance_mask, (n_rows_sub, n_cols_sub))
            else:
                instance_mask_sub = np.copy(instance_mask)

            instance_mask = task_utils.mask_vis_to_id(instance_mask, col_to_id=instance_col_to_id)
            instance_mask_sub = task_utils.mask_vis_to_id(instance_mask_sub, col_to_id=instance_col_to_id)
            instance_ids = np.unique(instance_mask_sub).tolist()

            if params.check:
                unique_cols = np.unique(instance_mask)
                unique_cols_sub = np.unique(instance_mask_sub)

                n_unique_cols = len(unique_cols)
                n_unique_cols_sub = len(unique_cols_sub)

                assert n_unique_cols == n_unique_cols_sub, "n_unique_cols_sub mismatch"
                assert n_unique_cols == n_instances, "n_unique_cols mismatch"
                assert instance_ids == instance_ids_2, "instance_ids mismatch"

        rle_tokens = []
        n_runs = rle_len = 0
        inst_rle_lens = []
        for instance_id in instance_ids:
            if instance_id == 0:
                continue
            target_id, class_id = instance_to_target_id[instance_id]
            if params.instance_coco_rle:
                rle_coco = instance_to_rle_coco[instance_id]
                instance_binary_mask = mask_rle_to_img(rle_coco)
                obj_mask = instance_binary_mask.astype(np.uint8)
                if subsample_method == 2:
                    obj_mask_sub = task_utils.resize_mask(obj_mask, (n_rows_sub, n_cols_sub))
                else:
                    obj_mask_sub = np.copy(obj_mask)
            else:
                obj_mask = np.zeros_like(instance_mask)
                obj_mask_sub = np.zeros_like(instance_mask_sub)
                obj_mask[instance_mask == instance_id] = 1
                obj_mask_sub[instance_mask_sub == instance_id] = 1

            obj_rle_tokens, obj_n_runs, obj_rle_len = get_rle_tokens(
                params, image, obj_mask, obj_mask_sub,
                max_length, max_length_sub, 2,
                subsample_method,
                n_rows, n_cols,
                class_id_to_col, class_id_to_name,
            )
            obj_class_token = [class_id + params.class_offset, ]
            rle_tokens += obj_rle_tokens + obj_class_token
            class_token_idx = len(rle_tokens) - 1
            class_token_idxs.append(class_token_idx)

            n_runs += obj_n_runs
            rle_len += obj_rle_len + 1
            inst_rle_lens.append(obj_rle_len)

        if params.check:
            task_utils.check_instance_wise_rle_tokens(
                image, mask, mask_sub,
                rle_tokens, n_classes,
                params.starts_2d,
                params.starts_offset,
                params.lengths_offset,
                params.class_offset,
                max_length,
                params.subsample,
                params.flat_order,
                class_id_to_col,
                is_vis=False)

    elif params.class_wise:
        class_ids = np.unique(mask_sub).tolist()

        rle_tokens = []
        n_runs = rle_len = 0
        cls_rle_lens = []

        for class_id in class_ids:
            if class_id == 0:
                continue
            class_mask = np.zeros_like(mask)
            class_mask_sub = np.zeros_like(mask_sub)

            class_mask[mask == class_id] = 1
            class_mask_sub[mask_sub == class_id] = 1

            class_rle_tokens, class_n_runs, class_rle_len = get_rle_tokens(
                params, image, class_mask, class_mask_sub,
                max_length, max_length_sub, 2,
                subsample_method,
                n_rows, n_cols,
                class_id_to_col, class_id_to_name,
            )
            class_token = [class_id + params.class_offset, ]
            rle_tokens += class_rle_tokens + class_token

            class_token_idx = len(rle_tokens) - 1
            class_token_idxs.append(class_token_idx)

            n_runs += class_n_runs
            rle_len += class_rle_len + 1
            cls_rle_lens.append(class_rle_len)

        if params.check:
            task_utils.check_class_wise_rle_tokens(
                image, mask, mask_sub,
                rle_tokens, n_classes,
                params.starts_2d,
                params.starts_offset,
                params.lengths_offset,
                params.class_offset,
                max_length,
                params.subsample,
                params.flat_order,
                class_id_to_col,
                is_vis=False)

    else:
        rle_tokens, n_runs, rle_len = get_rle_tokens(
            params, image, mask, mask_sub,
            max_length, max_length_sub, n_classes,
            subsample_method,
            n_rows, n_cols,
            class_id_to_col, class_id_to_name,
        )

    if rle_len > 0:
        append_metrics(
            dict(rle_len=f'{rle_len}'), metrics['db'])
        append_metrics(
            dict(vid_to_rle_len=f'{image_id}\t{rle_len}'), metrics['db'])

        if params.instance_wise:
            append_metrics(
                dict(inst_rle_lens=inst_rle_lens), metrics['db'])
        elif params.class_wise:
            append_metrics(
                dict(cls_rle_lens=cls_rle_lens), metrics['db'])

        # quad = quadtree.dense2quad(mask_sub, num_levels=6, return_255=False)

        if params.poly_len:
            # polygons = task_utils.polygons_from_mask(mask)
            polygons_sub = task_utils.polygons_from_mask(mask_sub)
            polygon_len = sum(polygon.size for polygon in polygons_sub)
            append_metrics(dict(len=polygon_len), metrics[f'polygons'])

        if not params.instance_wise and params.seg_metrics and params.subsample > 1:
            from densenet.evaluation.eval_segm import Metrics

            mask_sub_rec = task_utils.supersample_mask(mask_sub, params.subsample, n_classes, is_vis=0)

            seg_metrics = Metrics(class_id_to_name)
            seg_metrics.update(mask_sub_rec, mask)

            seg_metrics_dict = dict()
            for key, dice in seg_metrics.dice.items():
                if key in ('background', 'mean'):
                    continue
                metric_key = f'dice_iu_acc-{key}'
                iu = seg_metrics.iu[key]
                acc = seg_metrics.acc[key]
                seg_metrics_dict[metric_key] = f'{image_id}\t{dice}\t{iu}\t{acc}'
            append_metrics(
                seg_metrics_dict, metrics[f'method_{subsample_method}'])
    else:
        append_metrics(
            dict(empty_vid=f'{image_id}'), metrics[f'db'])

    if params.show and n_runs > 0:
        vis_imgs = []

        vis_imgs.append(image)

        rle_rec_cmp = task_utils.rle_from_tokens(
            rle_tokens,
            False,
            mask_sub.shape,
            params.length_as_class,
            params.starts_offset,
            params.lengths_offset,
            params.class_offset,
            params.starts_2d,
            multi_class,
            params.flat_order,
        )
        if params.length_as_class:
            starts_rec, lac = rle_rec_cmp
            lengths_rec, class_ids_rec = task_utils.rle_from_lac(lac, max_length_sub)
            rle_rec_cmp = [starts_rec, lengths_rec, class_ids_rec]

        starts_rec, lengths_rec = rle_rec_cmp[:2]
        if multi_class:
            class_ids_rec = rle_rec_cmp[2]
        else:
            class_ids_rec = [1, ] * len(starts_rec)

        if subsample_method == 1:
            """reconstruct full-res mask by super sampling / scaling up the starts and lengths"""
            starts_rec, lengths_rec = task_utils.supersample_rle(
                starts_rec, lengths_rec,
                subsample=params.subsample,
                shape=(n_rows, n_cols),
                max_length=max_length,
                flat_order=params.flat_order,
            )

        mask_rec = task_utils.rle_to_mask(
            starts_rec, lengths_rec, class_ids_rec,
            (n_rows_sub, n_cols_sub),
        )

        mask_vis = task_utils.mask_id_to_vis_bgr(mask, class_id_to_col)
        mask_sub_vis = task_utils.mask_id_to_vis_bgr(mask_sub, class_id_to_col)
        mask_rec_vis = task_utils.mask_id_to_vis_bgr(mask_rec, class_id_to_col)

        if subsample_method == 2:
            """reconstruct low-res mask and resize to scale it up"""
            mask_sub_vis = cv2.resize(mask_sub_vis, (n_cols, n_rows))
            mask_rec_vis = cv2.resize(mask_rec_vis, (n_cols, n_rows))
            # metrics_ = eval_mask(mask_rec, mask, rle_len)
            # vis_txt.append(append_metrics(metrics_, metrics['method_1']))

        vis_imgs.append(mask_vis)
        vis_imgs.append(mask_sub_vis)
        vis_imgs.append(mask_rec_vis)

        import eval_utils

        vis_imgs_1 = np.concatenate((image, mask_vis), axis=1)
        vis_imgs_2 = np.concatenate((mask_sub_vis, mask_rec_vis), axis=1)
        vis_imgs = np.concatenate((vis_imgs_1, vis_imgs_2), axis=0)
        vis_imgs = cv2.resize(vis_imgs, (960, 960))
        # vis_txt = ' '.join(vis_txt)
        vis_imgs = eval_utils.annotate(vis_imgs, f'{image_id}')
        # cv2.imshow('mask_vis', mask_vis)
        # cv2.imshow('mask_rec_vis', mask_rec_vis)
        cv2.imshow('vis_imgs', vis_imgs)
        k = cv2.waitKey(0)
        if k == 27:
            exit()

    if params.rle_to_json:
        image_info['rle_len'] = rle_len
        image_info['n_runs'] = n_runs
        image_info['rle'] = rle_tokens
        if params.instance_wise or params.class_wise:
            image_info['class_token_idxs'] = class_token_idxs

    if not skip_tfrecord:
        seg_feature_dict = {
            'image/mask_file_name': tfrecord_lib.convert_to_feature(mask_filename.encode('utf8')),
            'image/frame_id': tfrecord_lib.convert_to_feature(frame_id),
            'image/n_runs': tfrecord_lib.convert_to_feature(n_runs),
        }
        if not params.rle_to_json:
            seg_feature_dict['image/rle'] = tfrecord_lib.convert_to_feature(rle_tokens, value_type='int64_list')

        feature_dict.update(seg_feature_dict)
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return example, 0


def main():
    params: Params = paramparse.process(Params)

    assert params.db_path, "db_path must be provided"

    assert params.end_id >= params.start_id, f"invalid end_id: {params.end_id}"

    class_id_to_col, class_id_to_name = task_utils.read_class_info(params.class_names_path)

    if params.stats_only == 2 or params.vis == 2:
        params.check = 0

    if params.stats_only:
        print('running in stats only mode')
        params.vis = params.show = False

    if params.instance_wise or params.class_wise:
        assert not params.length_as_class, \
            'length_as_class does not make sense with instance_wise or class_wise'

    n_classes = len(class_id_to_col)
    multi_class = False
    if n_classes > 2:
        assert params.class_offset > 0, "class_offset must be > 0 for multi_class mode"
        multi_class = True

    if params.patch_width <= 0:
        params.patch_width = params.patch_height

    if params.min_stride <= 0:
        params.min_stride = params.patch_height

    if params.max_length <= 0:
        params.max_length = params.patch_width

    if params.max_stride <= params.min_stride:
        params.max_stride = params.min_stride

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

        if params.instance_wise:
            db_suffixes.append('inst')

        params.db_suffix = '-'.join(db_suffixes)

    params.db_path = f'{params.db_path}-{params.db_suffix}'

    in_json_name = params.db_suffix

    if params.seq_id >= 0:
        params.seq_start_id = params.seq_end_id = params.seq_id

    if params.seq_start_id > 0 or params.seq_end_id >= 0:
        assert params.seq_end_id >= params.seq_start_id, "end_seq_id must to be >= start_seq_id"
        seq_suffix = f'seq_{params.seq_start_id}_{params.seq_end_id}'
        in_json_name = f'{in_json_name}-{seq_suffix}'

    in_json_fname = f'{in_json_name}.{params.ann_ext}'
    in_json_path = linux_path(params.db_path, in_json_fname)

    img_json_dict = load_seg_annotations(in_json_path)
    image_infos = img_json_dict['images']

    out_json_name = in_json_name

    rle_suffix = get_rle_suffix(params, n_classes, multi_class)
    if rle_suffix:
        out_json_name = f'{out_json_name}-{rle_suffix}'

    out_json_fname = f'{out_json_name}.{params.ann_ext}'
    out_json_path = linux_path(params.db_path, out_json_fname)

    # shutil.copy(in_json_path, out_json_path)

    if not params.output_dir:
        params.output_dir = linux_path(params.db_path, 'tfrecord')

    os.makedirs(params.output_dir, exist_ok=True)

    vid_infos = None
    if params.load_vid:
        vid_infos = get_vid_infos(params.db_path, image_infos, params.instance_wise, params.instance_coco_rle)

    metrics = dict(
        db={},
        method_0={},
        method_1={},
        method_2={},
        polygons={},
    )

    if params.rle_to_json:
        print(f'writing RLE to json: {out_json_path}')

    if not params.check:
        print(f'RLE check is disabled')
    else:
        print(f'RLE check is enabled')

    skip_tfrecord = params.stats_only or params.vis or params.rle_to_json and params.json_only

    annotations_iter = generate_annotations(
        params=params,
        class_id_to_col=class_id_to_col,
        class_id_to_name=class_id_to_name,
        metrics=metrics,
        image_infos=image_infos,
        vid_infos=vid_infos,
        skip_tfrecord=skip_tfrecord,
    )

    # for idx, annotations_iter_ in tqdm(enumerate(annotations_iter), total=len(image_infos)):
    #     create_tf_example(*annotations_iter_)

    tfrecord_path = linux_path(params.output_dir, in_json_name if params.rle_to_json else out_json_name)
    os.makedirs(tfrecord_path, exist_ok=True)

    if skip_tfrecord:
        print('skipping tfrecord creation')
        for idx, annotations_iter_ in tqdm(enumerate(annotations_iter), total=len(image_infos)):
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
            iter_len=len(image_infos),
        )

    save_seg_annotations(params, img_json_dict, class_id_to_name, class_id_to_col, out_json_path)

    metrics_dir = linux_path(params.db_path, '_metrics_')

    print(f'metrics_dir: {metrics_dir}')
    os.makedirs(metrics_dir, exist_ok=True)

    for method, metrics_ in metrics.items():
        for metric_, val in metrics_.items():
            metrics_path = linux_path(metrics_dir, f'{out_json_name}-{method}-{metric_}.txt')
            print(f'metrics_path: {metrics_path}')

            with open(metrics_path, 'w') as f:
                f.write('\n'.join(map(str, val)))


if __name__ == '__main__':
    main()
