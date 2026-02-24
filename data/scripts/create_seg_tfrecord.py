import collections
import json
import os
import shutil
import sys
import math
import cv2
from PIL import Image
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

from eval_utils import col_bgr, mask_rle_to_img, linux_path


class Params(paramparse.CFG):
    """
    :ivar ade20k:
    ade20k has different mask storage system and needs RGB images to store the masks because of the excessive
    number of classes (>3600) that cannot be represented by a single general 8 bit image

    :ivar coco:
    coco classes need to be read from json
    also requires image attributes like img_id, frame_id, mask_file_name, seq to constructed differently
    """

    def __init__(self):
        paramparse.CFG.__init__(self, cfg_prefix='tf_seg')
        self.class_names_path = ''
        self.db_path = ''
        self.db_suffix = ''
        self.split_suffix = ''
        self.json_path = ''
        self.masks_path = ''
        self.seq_name = ''
        self.vis = 0

        self.rle_from_mask = 0
        self.stats_only = 0
        self.rle_to_json = 1
        self.json_only = 0
        self.check = 0

        self.load_vid = 1

        self.ade20k = 0
        self.coco = 0

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

        self.shuffle = 0
        self.sample = 0

        self.patch_mode = 0
        self.subseq_mode = 0

        self.n_rot = 0
        self.max_rot = 0
        self.min_rot = 0

        self.resize = 0
        self.resize_x = 0
        self.resize_y = 0

        self.start_id = 0
        self.end_id = -1

        self.patch_height = 0
        self.patch_width = 0

        self.max_stride = 0
        self.min_stride = 0

        self.enable_flip = 0

        self.rle = Params.RLE()
        self.show = 0

    class RLE:
        """
        :ivar starts_2d:
        represent each start with 2D coordinates (x, y) instead of 1D coordinate corresponding to the flattened mask

        :ivar no_starts:
        represent runs with only (length, class) pairs, thus removing starts

        :ivar flat_order: order in which to flatten the mask before computing the RLE
            flat_order='C': row-wise flattening
            flat_order='F': column-wise flattening
        
        :ivar diff_mask: use differential mask
            diff_mask=1: perform mask differencing without flattening by taking the difference between consecutive columns
            diff_mask=2: perform mask differencing after flattening the mask in the order specified by flat_order

        :ivar subsample:
        factor by which to reduce mask resolution

        :ivar subsample_method:
        1: create RLE of full-res mask and sample the starts and lengths thus generated
        2: decrease mask resolution by resizing and create RLE of the low-res mask
        """

        def __init__(self):
            self.length_as_class = 0

            self.diff_mask = 0
            self.flat_order = 'C'
            self.max_length = 0
            self.max_length_factor = 0
            self.no_starts = 0
            self.starts_2d = 0
            self.starts_offset = 1000
            self.lengths_offset = 100
            self.class_offset = 0
            self.shared_coord = 0
            self.subsample = 1
            self.subsample_method = 2

            self.class_wise = 0
            self.instance_wise = 0
            self.instance_coco_rle = 0


def get_rle_suffix(params: Params.RLE, multi_class):
    rle_suffixes = []

    if params.subsample > 1:
        rle_suffixes.append(f'sub_{params.subsample}')

    if params.diff_mask == 1:
        rle_suffixes.append(f'dm')
    elif params.diff_mask == 2:
        rle_suffixes.append(f'dm2')

    if params.no_starts:
        assert not params.starts_2d, "no_starts and starts_2d cannot both be enabled"
        rle_suffixes.append(f'nos')

    if params.starts_2d:
        assert not params.no_starts, "no_starts and starts_2d cannot both be enabled"
        rle_suffixes.append(f'2d')

    if params.shared_coord:
        assert params.starts_2d, "shared_coord should only be used with 2D starts"
        rle_suffixes.append(f'sco')

    if params.length_as_class:
        rle_suffixes.append(f'lac')
    else:
        if multi_class:
            rle_suffixes.append(f'mc')

    if params.flat_order != 'C':
        rle_suffixes.append(f'flat_{params.flat_order}')

    if params.class_wise:
        rle_suffixes.append('cw')

    if params.max_length_factor > 0:
        rle_suffixes.append(f'mlf_{params.max_length_factor:d}')

    rle_suffix = '-'.join(rle_suffixes) if rle_suffixes else ''
    return rle_suffix


def append_metrics(metrics: dict, out: dict):
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
    }
    if params.rle_to_json:
        params_dict = paramparse.to_dict(params)
        info["params"] = params_dict

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
        params: Params,
        class_id_to_col,
        class_id_to_name,
        metrics,
        image_infos,
        vid_infos,
        skip_tfrecord,
):
    for image_info_id, image_info in enumerate(image_infos):
        # seq_id = image_info['seq']
        file_name = image_info['file_name']

        if params.coco and params.rle_from_mask:
            img_id = src_id = os.path.splitext(file_name)[0]
            patch_id = 0
            image_info['img_id'] = img_id
            image_info['frame_id'] = image_info_id
            image_info['mask_file_name'] = f'{img_id}.png'
            image_info['seq'] = params.seq_name
        else:
            img_id = image_info['img_id']
            img_id_cmps = img_id.split('_')
            src_id = img_id_cmps[0]
            try:
                patch_id = int(img_id_cmps[1])
            except ValueError:
                patch_id = 1
            except IndexError:
                patch_id = 1

        image_info['src_id'] = src_id
        image_info['patch_id'] = patch_id

        if patch_id < params.patch_start_id > 0:
            continue

        if patch_id > params.patch_end_id > 0:
            continue

        vid_info = vid_infos[image_info['seq']] if vid_infos is not None else None
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
        params: Params,
        image,
        mask,
        mask_sub,
        max_length, max_length_sub,
        n_classes, subsample_method,
        n_rows, n_cols,
        class_id_to_col, class_id_to_name,
        show_eos=True,
        prev_run_info=None,
        vis_images=None,
        txt_col=None,
):
    multi_class = n_classes > 2

    if params.rle.no_starts:
        assert not params.rle.diff_mask, "diff_mask is incompatible with no_starts"
        assert subsample_method != 1, "rle subsampling is currently not supported with no_starts"
        lengths, class_ids, lengths_unsplit, class_ids_unsplit = task_utils.mask_to_rle_no_starts(
            mask=mask_sub,
            max_length=max_length_sub,
            n_classes=n_classes,
            order=params.rle.flat_order,
            return_unsplit=True
        )
        rle_cmp = [np.copy(lengths), np.copy(class_ids)]
        rle_cmp_unsplit = [lengths_unsplit, class_ids_unsplit]

    else:
        if params.rle.diff_mask:
            mask_sub = task_utils.mask_to_diff(mask_sub, n_classes - 1,
                                               flatten=params.rle.diff_mask == 2,
                                               flat_order=params.rle.flat_order)
            """
            diff_mask=1 will not give entirely accurate RLEs since this assumes that all the runs have length 1
            which is not true when differential mask is created without flattening because the same class 
            might extend across multiple rows/columns
            """
            starts,  class_ids = task_utils.diff_mask_to_rle(
                diff_mask=mask_sub,
                n_classes=n_classes,
                order=params.rle.flat_order,
            )
            # if params.check:
                # starts_old, lengths, starts_unsplit, lengths_unsplit = task_utils.mask_to_rle(
                #     mask=mask_sub,
                #     max_length=max_length_sub,
                #     n_classes=n_classes * 2 - 1,
                #     order=params.rle.flat_order,
                #     return_unsplit=True
                # )
                # assert np.array_equal(starts, starts_old), "starts_old mismatch"

                # class_ids_arr = np.asarray(class_ids)
                # class_ids_diff = class_ids_arr[:-1] - class_ids_arr[1:]
                # class_ids_diff_zeros = np.argwhere(class_ids_diff == 0)
                # mask_sub_flat = mask_sub.flatten(order=params.rle.flat_order)
                # assert class_ids_diff_zeros.size == 0, "class_ids_diff_zeros must be empty"

                # non_unit_length_ids = np.asarray(lengths != 1).nonzero()[0]
                # non_unit_lengths = lengths[non_unit_length_ids]
                # non_unit_starts = starts[non_unit_length_ids]
                # assert non_unit_length_ids.size == 0, "lengths must be all ones for diff_mask"

            rle_cmp = [np.copy(starts), np.copy(class_ids)]
            rle_cmp_unsplit = []
        else:
            starts, lengths, starts_unsplit, lengths_unsplit = task_utils.mask_to_rle(
                mask=mask_sub,
                max_length=max_length_sub,
                n_classes=n_classes,
                order=params.rle.flat_order,
                return_unsplit=True
            )
            rle_cmp_unsplit = [starts_unsplit, lengths_unsplit]

            if subsample_method == 1:
                assert not params.rle.diff_mask, "diff_mask is incompatible with subsample_method=1"
                """subsample RLE of high-res mask instead of taking RLE of subsampled mask"""
                starts, lengths = task_utils.subsample_rle(
                    starts, lengths,
                    subsample=params.rle.subsample,
                    shape=(n_rows, n_cols),
                    max_length=max_length,
                    flat_order=params.rle.flat_order,
                )
            rle_cmp = [np.copy(starts), np.copy(lengths)]

            if multi_class:
                assert params.rle.class_offset > 0, "class_offset must be > 0"
                if not params.rle.no_starts:
                    class_ids = task_utils.get_rle_class_ids(
                        mask_sub,
                        starts,
                        n_classes=n_classes,
                        order=params.rle.flat_order)
                    rle_cmp.append(class_ids)
                if params.rle.length_as_class:
                    rle_cmp = task_utils.rle_to_lac(rle_cmp, max_length_sub)

    n_runs = len(rle_cmp[0])
    run_txts = None

    if params.vis:
        assert not params.rle.diff_mask, "diff_mask rle vis is currently not supported"
        assert not params.rle.no_starts, "no_starts rle vis is currently not supported"
        run_txts = task_utils.vis_rle(
            rle_cmp,
            params.rle.length_as_class,
            max_length_sub,
            class_id_to_col, class_id_to_name,
            image, mask, mask_sub,
            flat_order=params.rle.flat_order,
            show_eos=show_eos,
            prev_run_info=prev_run_info,
            vis_images=vis_images,
            txt_col=txt_col,
        )

    rle_tokens = task_utils.rle_to_tokens(
        rle_cmp, mask_sub.shape,
        length_as_class=params.rle.length_as_class,
        starts_offset=params.rle.starts_offset,
        lengths_offset=params.rle.lengths_offset,
        class_offset=params.rle.class_offset,
        starts_2d=params.rle.starts_2d,
        flat_order=params.rle.flat_order,
        no_starts=params.rle.no_starts,
        diff_mask=params.rle.diff_mask,
    )
    rle_len = len(rle_tokens)
    if params.rle.no_starts:
        if params.rle.length_as_class:
            n_tokens_per_run = 1
        else:
            n_tokens_per_run = 2
    else:
        n_tokens_per_run = 2
        if params.rle.starts_2d:
            n_tokens_per_run += 1
        if not params.rle.diff_mask and multi_class and not params.rle.length_as_class:
            n_tokens_per_run += 1

    assert rle_len % n_tokens_per_run == 0, f"rle_len must be divisible by {n_tokens_per_run}"
    assert n_runs * n_tokens_per_run == rle_len, f"mismatch between n_runs and rle_len"

    if params.check:
        # assert not params.rle.no_starts, "no_starts rle check is currently not supported"
        task_utils.check_rle_tokens(
            image, mask, mask_sub, rle_tokens,
            n_classes,
            params.rle.length_as_class,
            params.rle.starts_2d,
            params.rle.starts_offset,
            params.rle.lengths_offset,
            params.rle.class_offset,
            max_length,
            params.rle.subsample,
            multi_class,
            params.rle.flat_order,
            class_id_to_col,
            is_vis=False,
            no_starts=params.rle.no_starts,
            diff_mask=params.rle.diff_mask,
        )

    return rle_tokens, n_runs, rle_len, run_txts, rle_cmp_unsplit


def create_tf_example(
        params: Params,
        class_id_to_col: dict,
        class_id_to_name: dict,
        metrics: dict,
        image_info: dict,
        vid_info: dict,
        skip_tfrecord: dict,
):
    n_classes_with_bkg = len(class_id_to_col)
    multi_class = n_classes_with_bkg > 2

    image_height = image_info['height']
    image_width = image_info['width']
    filename = image_info['file_name']
    image_id = image_info['img_id']
    frame_id = int(image_info['frame_id'])
    mask_filename = image_info['mask_file_name']

    if not params.rle_from_mask:
        subsample_method = params.rle.subsample_method
        if params.rle.subsample <= 1:
            subsample_method = 0

    image_root_dir = masks_root_dir = params.db_path
    if params.seq_name:
        image_root_dir = linux_path(image_root_dir, params.seq_name)
    if params.masks_path:
        masks_root_dir = linux_path(masks_root_dir, params.masks_path)

    image_path = linux_path(image_root_dir, filename)
    mask_image_path = linux_path(masks_root_dir, mask_filename)

    if not image_id.startswith('seq'):
        seq = image_info['seq']
        image_id = f'{seq}/{image_id}'

    image = encoded_jpg = feature_dict = None

    read_image = params.vis or params.show or params.check or not skip_tfrecord

    if not skip_tfrecord:
        feature_dict = {}

    if not params.rle_from_mask:
        if params.rle.instance_wise:
            instance_to_target_id: dict = image_info['instance_to_target_id']
            instance_to_target_id = {
                int(instance_id_): (int(target_id_), int(class_id_))
                for instance_id_, (target_id_, class_id_) in
                instance_to_target_id.items()
            }
            instance_ids_1 = sorted(list(instance_to_target_id.keys()))
            n_instances = len(instance_to_target_id)

            if params.rle.instance_coco_rle:
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

    # encoded_png = None

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

        # mask = cv2.imread(mask_image_path)
        mask_pil = Image.open(mask_image_path)
        mask = np.array(mask_pil)
        # if params.coco:
        #     with tf.io.gfile.GFile(mask_image_path, 'rb') as fid:
        #         encoded_png = fid.read()

    # mask_orig = np.copy(mask)
    # mask_orig = task_utils.mask_to_gs(mask_orig)

    # cv2.imshow('mask', mask)

    if not multi_class:
        mask = task_utils.mask_to_binary(mask)

    mask = task_utils.mask_to_gs(mask)
    # cv2.imshow('mask_gs', mask)
    # cv2.waitKey(0)

    # mask_unique = np.unique(mask)

    n_rows, n_cols = mask.shape

    if not params.rle_from_mask:
        max_length = params.rle.max_length
        if subsample_method == 2:
            """        
            decrease mask resolution by resizing and create RLE of the low-res mask
            """
            max_length_sub = int(max_length / params.rle.subsample)
            n_rows_sub, n_cols_sub = int(n_rows / params.rle.subsample), int(n_cols / params.rle.subsample)

            mask_sub = task_utils.resize_mask(mask, (n_rows_sub, n_cols_sub))
            # mask_sub = task_utils.subsample_mask(mask, params.subsample, n_mask_classes, is_vis=1)
        else:
            mask_sub = np.copy(mask)
            n_rows_sub, n_cols_sub = n_rows, n_cols
            max_length_sub = max_length

    if vid_info is not None:
        mask = task_utils.mask_vis_to_id(mask, n_classes=n_classes_with_bkg)
        if not params.rle_from_mask:
            mask_sub = task_utils.mask_vis_to_id(mask_sub, n_classes=n_classes_with_bkg)

    cls_mask = np.copy(mask)

    if not skip_tfrecord:
        image_feature_dict = tfrecord_lib.image_info_to_feature_dict(
            image_height, image_width, filename, image_id, encoded_jpg, 'jpg')
        if params.rle_from_mask:
            # cv2.imshow('image', image)
            # cv2.imshow('mask', mask)
            # cv2.waitKey(0)

            encoded_png = cv2.imencode('.png', mask)[1].tobytes()

            image_feature_dict.update({
                'mask/encoded': tfrecord_lib.convert_to_feature(encoded_png),
                'mask/format': tfrecord_lib.convert_to_feature('png'.encode('utf8')),
                'mask/channels': tfrecord_lib.convert_to_feature(3 if params.ade20k else 1),
            })
        feature_dict.update(image_feature_dict)

    if not params.rle_from_mask:
        class_token_idxs = []
        if params.rle.instance_wise:
            if params.rle.instance_coco_rle:
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
            n_runs = rle_len = coco_rle_len = 0
            inst_rle_lens = []
            inst_coco_rle_lens = []
            obj_coco_rle_len = 0
            if params.rle.instance_coco_rle:
                instance_binary_masks = {}
                instance_mask = np.zeros_like(mask)
                instance_mask_vis = np.zeros_like(image)

                instance_vis_cols = task_utils.get_cols_rgb(n_instances, min_col=100, max_col=200)
                instance_id_to_vis_col = {
                    instance_id: vis_col for instance_id, vis_col in zip(instance_ids, instance_vis_cols)
                }
                binary_class_id_to_col = {
                    1: (255, 255, 255)
                }
                for instance_id in instance_ids:
                    rle_coco = instance_to_rle_coco[instance_id]
                    instance_binary_mask = mask_rle_to_img(rle_coco)
                    instance_binary_mask_bool = instance_binary_mask.astype(bool)
                    # instance_binary_mask[instance_binary_mask == 1] = 255
                    instance_mask[instance_binary_mask_bool] = instance_id
                    instance_binary_masks[instance_id] = instance_binary_mask

                    # instance_binary_mask_vis = instance_binary_mask * 255
                    # instance_mask_vis[instance_binary_mask_bool] = instance_id_to_vis_col[instance_id]
                    # cv2.imshow('instance_binary_mask_vis', instance_binary_mask_vis)
                    # cv2.imshow('instance_mask', instance_mask)
                    # cv2.imshow('instance_mask_vis', instance_mask_vis)
                    # cv2.waitKey(0)
                instance_mask_sub = task_utils.resize_mask(instance_mask, (n_rows_sub, n_cols_sub))

            prev_run_info = None
            for instance_id in instance_ids:
                if instance_id == 0:
                    continue
                target_id, class_id = instance_to_target_id[instance_id]

                instance_col = instance_id_to_vis_col[instance_id]

                class_col = class_id_to_col[class_id]
                class_col_bgr = vis_utils.col_bgr[class_col]

                class_name = class_id_to_name[class_id]
                if params.rle.instance_coco_rle:
                    obj_binary_mask = instance_binary_masks[instance_id]
                    if subsample_method == 2:
                        obj_binary_mask_sub = task_utils.resize_mask(obj_binary_mask, (n_rows_sub, n_cols_sub))
                    else:
                        obj_binary_mask_sub = np.copy(obj_binary_mask)
                else:
                    obj_binary_mask = np.zeros_like(instance_mask)
                    obj_binary_mask_sub = np.zeros_like(instance_mask_sub)
                    obj_binary_mask[instance_mask == instance_id] = 1
                    obj_binary_mask_sub[instance_mask_sub == instance_id] = 1

                if params.vis:
                    obj_instance_mask = np.zeros_like(instance_mask)
                    obj_instance_mask[instance_mask == instance_id] = instance_id

                    vis_image = task_utils.blend_mask(instance_mask, image, instance_id_to_vis_col,
                                                      alpha=0.25, class_ids=instance_ids)
                    cls_vis_image = task_utils.blend_mask(cls_mask, image, class_id_to_col,
                                                          alpha=0.25, class_ids=None)

                    # image_darkened = np.copy(image)
                    # alpha = 0.25
                    # bkg_col = (0, 0, 0)
                    # obj_instance_mask_inv = instance_mask != instance_id
                    # image_darkened[obj_instance_mask_inv] = image_darkened[obj_instance_mask_inv] * (1 - alpha) +
                    # np.asarray(
                    #     bkg_col) * alpha

                    obj_vis_image = task_utils.blend_mask(obj_instance_mask, image, instance_id_to_vis_col,
                                                          alpha=0.25, class_ids=instance_ids)

                    vis_images = (cls_vis_image, vis_image, obj_vis_image)

                    # cmb_vis_image = np.concatenate(vis_images, axis=1)
                    # cv2.imshow('cmb_vis_image', cmb_vis_image)
                    # cv2.waitKey(0)

                obj_rle_tokens, obj_n_runs, obj_rle_len, obj_run_txts, _ = get_rle_tokens(
                    params, image, obj_binary_mask, obj_binary_mask_sub,
                    max_length, max_length_sub, 2,
                    subsample_method,
                    n_rows, n_cols,
                    binary_class_id_to_col, class_id_to_name,
                    show_eos=False,
                    prev_run_info=prev_run_info,
                    vis_images=vis_images,
                    txt_col=instance_col,
                )
                if not obj_rle_tokens:
                    """object had some pixels in the original mask but none in the subsampled mask"""
                    continue

                obj_class_token = [class_id + params.rle.class_offset, ]

                rle_tokens += obj_rle_tokens + obj_class_token
                # obj_run_txts[-1] = f'{obj_run_txts[-1]}{class_name}, '
                if prev_run_info is None:
                    prev_run_info = []
                prev_run_info += [(obj_run_txt, instance_col) for obj_run_txt in obj_run_txts]
                prev_run_info.append((f'{class_name}, ', class_col_bgr))

                class_token_idx = len(rle_tokens) - 1
                class_token_idxs.append(class_token_idx)

                n_runs += obj_n_runs
                if params.rle.instance_coco_rle:
                    coco_rle_len += obj_coco_rle_len + 1
                    inst_coco_rle_lens.append(obj_coco_rle_len)
                rle_len += obj_rle_len + 1
                inst_rle_lens.append(obj_rle_len)

            if params.vis:
                """show class token of last object followed by EOS"""
                obj_binary_mask = np.zeros_like(mask)
                obj_binary_mask_sub = np.zeros_like(mask_sub)
                get_rle_tokens(
                    params, image, obj_binary_mask, obj_binary_mask_sub,
                    max_length, max_length_sub, 2,
                    subsample_method,
                    n_rows, n_cols,
                    binary_class_id_to_col, class_id_to_name,
                    show_eos=True,
                    prev_run_info=prev_run_info,
                    vis_images=vis_images,
                )
            if params.check:
                task_utils.check_instance_wise_rle_tokens(
                    image, mask, mask_sub,
                    rle_tokens, n_classes_with_bkg,
                    params.rle.starts_2d,
                    params.rle.starts_offset,
                    params.rle.lengths_offset,
                    params.rle.class_offset,
                    max_length,
                    params.rle.subsample,
                    params.rle.flat_order,
                    class_id_to_col,
                    is_vis=False)

        elif params.rle.class_wise:
            class_ids = np.unique(mask_sub).tolist()

            rle_tokens = []
            n_runs = rle_len = 0
            cls_rle_lens = []
            cls_to_rle_lens = {
                class_name: [] for class_name in class_id_to_name.values()
            }

            for class_id in class_ids:
                if class_id == 0:
                    continue
                class_mask = np.zeros_like(mask)
                class_mask_sub = np.zeros_like(mask_sub)

                class_mask[mask == class_id] = 1
                class_mask_sub[mask_sub == class_id] = 1

                class_rle_tokens, class_n_runs, class_rle_len, _, _ = get_rle_tokens(
                    params, image, class_mask, class_mask_sub,
                    max_length, max_length_sub, 2,
                    subsample_method,
                    n_rows, n_cols,
                    class_id_to_col, class_id_to_name,
                )
                if not class_rle_tokens:
                    """object had some pixels in the original mask but none in the subsampled mask"""
                    continue
                class_token = [class_id + params.rle.class_offset, ]
                rle_tokens += class_rle_tokens + class_token

                class_token_idx = len(rle_tokens) - 1
                class_token_idxs.append(class_token_idx)
                class_name = class_id_to_name[class_id]

                n_runs += class_n_runs
                rle_len += class_rle_len + 1

                cls_to_rle_lens[class_name].append(class_rle_len)
                cls_rle_lens.append(class_rle_len)

            if params.check:
                task_utils.check_class_wise_rle_tokens(
                    image, mask, mask_sub,
                    rle_tokens,
                    n_classes_with_bkg,
                    params.rle.starts_2d,
                    params.rle.starts_offset,
                    params.rle.lengths_offset,
                    params.rle.class_offset,
                    max_length,
                    params.rle.subsample,
                    params.rle.flat_order,
                    class_id_to_col,
                    is_vis=False)

        else:
            rle_tokens, n_runs, rle_len, _, rle_cmp_unsplit = get_rle_tokens(
                params, image,
                mask,
                mask_sub,
                max_length=max_length,
                max_length_sub=1 if params.rle.diff_mask else max_length_sub,
                n_classes=n_classes_with_bkg,
                subsample_method=subsample_method,
                n_rows=n_rows,
                n_cols=n_cols,
                class_id_to_col=class_id_to_col,
                class_id_to_name=class_id_to_name,
            )
            if rle_len > 0:
                if not params.rle.diff_mask:
                    if params.rle.no_starts:
                        lengths_unsplit, class_ids_unsplit = rle_cmp_unsplit
                    else:
                        starts_unsplit, lengths_unsplit = rle_cmp_unsplit
                        # if not params.rle.length_as_class:
                        #     starts_tokens = np.asarray(rle_tokens[::3]) - params.rle.starts_offset
                        #     lengths_tokens = np.asarray(rle_tokens[1::3]) - params.rle.lengths_offset
                        #     class_tokens = np.asarray(rle_tokens[2::3]) - params.rle.class_offset
                    lengths_unsplit = list(lengths_unsplit)
                    append_metrics(
                        dict(lengths=list(lengths_unsplit)), metrics['db'])

        if rle_len > 0:
            append_metrics(
                dict(rle_len=f'{rle_len}'), metrics['db'])
            append_metrics(
                dict(vid_to_rle_len=f'{image_id}\t{rle_len}'), metrics['db'])
            append_metrics(
                dict(vid_to_rle_len=f'{image_id}\t{rle_len}'), metrics['db'])

            if params.rle.instance_wise:
                if params.rle.instance_coco_rle:
                    append_metrics(
                        dict(inst_coco_rle_lens=inst_coco_rle_lens), metrics['db'])
                append_metrics(
                    dict(inst_rle_lens=inst_rle_lens), metrics['db'])
            elif params.rle.class_wise:
                for cls_, rle_len_ in cls_to_rle_lens.items():
                    append_metrics(
                        {f'cls_rle_lens/{cls_}': rle_len_}, metrics['db']
                    )
                append_metrics(
                    {f'cls_rle_lens/all': cls_rle_lens}, metrics['db']
                )
            if params.poly_len:
                # quad = quadtree.dense2quad(mask_sub, num_levels=6, return_255=False)
                # polygons = task_utils.polygons_from_mask(mask)
                polygons_sub = task_utils.polygons_from_mask(mask_sub)
                polygon_len = sum(polygon.size for polygon in polygons_sub)
                append_metrics(dict(len=polygon_len), metrics[f'polygons'])

            if not params.rle.instance_wise and params.seg_metrics and params.rle.subsample > 1:
                from densenet.evaluation.eval_segm import Metrics

                mask_sub_rec = task_utils.supersample_mask(mask_sub, params.rle.subsample, n_classes_with_bkg, is_vis=0)

                seg_metrics = Metrics(class_id_to_name, class_id_to_col)
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
                params.rle.length_as_class,
                params.rle.starts_offset,
                params.rle.lengths_offset,
                params.rle.class_offset,
                params.rle.starts_2d,
                multi_class,
                params.rle.flat_order,
            )
            if params.rle.length_as_class:
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
                    subsample=params.rle.subsample,
                    shape=(n_rows, n_cols),
                    max_length=max_length,
                    flat_order=params.rle.flat_order,
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
            if params.rle.instance_wise or params.rle.class_wise:
                image_info['class_token_idxs'] = class_token_idxs

    if not skip_tfrecord:
        seg_feature_dict = {
            'image/mask_image_path': tfrecord_lib.convert_to_feature(mask_image_path.encode('utf8')),
            'image/frame_id': tfrecord_lib.convert_to_feature(frame_id),
        }

        if not params.rle_from_mask:
            seg_feature_dict['image/n_runs'] = tfrecord_lib.convert_to_feature(n_runs)
            if not params.rle_to_json:
                seg_feature_dict['image/rle'] = tfrecord_lib.convert_to_feature(rle_tokens, value_type='int64_list')

        feature_dict.update(seg_feature_dict)
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return example, 0


def main():
    params: Params = paramparse.process(Params)

    assert params.db_path, "db_path must be provided"

    assert params.end_id < 0 or params.end_id >= params.start_id, f"invalid end_id: {params.end_id}"

    if params.rle_from_mask:
        print('writing masks to tfrecord')
        params.rle_to_json = 0

    if params.rle_to_json:
        print('writing RLE to JSON')
    else:
        print('not writing RLE to JSON')

    if params.coco and params.rle_from_mask:
        assert params.json_path, "json_path must be provided for coco rle_from_mask"
        in_json_path = linux_path(params.db_path, params.json_path)
        json_dict = load_seg_annotations(in_json_path)
        categories = json_dict["categories"]
        class_cols = []
        n_categories = len(categories)
        class_cols_rgb = task_utils.get_cols_rgb(n_categories)
        for class_col_rgb in class_cols_rgb:
            r, g, b = class_col_rgb
            col_name = f'{b}_{g}_{r}'
            col_bgr[col_name] = (b, g, r)
            class_cols.append(col_name)
        class_id_to_col = {0: 'background'}
        class_id_to_name = {0: 'black'}
        # class_txt = ''
        for class_id, (category, col) in enumerate(zip(categories, class_cols, strict=True)):
            category_id, class_name = category['id'], category['name']
            assert category_id == class_id + 1, "class_id mismatch"
            class_id_to_name[category_id] = class_name
            class_id_to_col[category_id] = col

            # class_txt += f'{class_name}\t{col}\n'

        # with open('coco.txt', 'w') as fid:
        #     fid.write(class_txt)
    else:
        class_id_to_col, class_id_to_name = task_utils.read_class_info(params.class_names_path)

    if params.stats_only == 2 or params.vis == 2:
        params.check = 0

    if params.stats_only:
        print('running in stats only mode')
        params.vis = params.show = False

    if not params.rle_to_json:
        if params.rle.instance_wise or params.rle.class_wise:
            assert not params.rle.length_as_class, \
                'length_as_class does not make sense with instance_wise or class_wise'

    n_classes = len(class_id_to_col)
    multi_class = False
    if n_classes > 2:
        assert not params.rle_to_json or params.rle.class_offset > 0, "class_offset must be > 0 for multi_class mode"
        multi_class = True

    patch_mode = params.patch_mode
    subseq_mode = params.subseq_mode or params.start_id > 0

    if params.resize:
        params.resize_x = params.resize_y = params.resize

    enable_resize = 0
    if params.resize_x or params.resize_y:
        assert params.resize_x and params.resize_y, "either both or none of resize_x and resize_y must be provided"
        enable_resize = 1

    if params.end_id >= params.start_id:
        subseq_mode = 1

    if params.patch_height <= 0:
        if params.resize_y:
            params.patch_height = params.resize_y
    else:
        patch_mode = 1

    if params.patch_width <= 0:
        if params.resize_x:
            params.patch_width = params.resize_x
        else:
            params.patch_width = params.patch_height
    else:
        patch_mode = 1

    if params.min_stride <= 0:
        params.min_stride = params.patch_height
    else:
        patch_mode = 1

    if params.max_stride <= params.min_stride:
        params.max_stride = params.min_stride
    else:
        patch_mode = 1

    if params.rle_to_json:
        if params.rle.max_length <= 0:
            if params.rle.flat_order == 'C':
                params.rle.max_length = params.patch_width if params.patch_width > 0 else params.resize_x
            else:
                params.rle.max_length = params.patch_height if params.patch_width > 0 else params.resize_y
            if params.rle.max_length_factor > 0:
                params.rle.max_length = int(params.rle.max_length * params.rle.max_length_factor)

    if params.json_path:
        in_json_path = linux_path(params.db_path, params.json_path)
        in_json_name = os.path.splitext(os.path.basename(params.json_path))[0]
    else:
        if not params.db_suffix:
            db_suffixes = []

            if params.split_suffix:
                db_suffixes.append(params.split_suffix)

            if enable_resize:
                if params.resize:
                    db_suffixes.append(f'resize_{params.resize}')
                else:
                    db_suffixes.append(f'resize_{params.resize_x}x{params.resize_y}')

            if subseq_mode:
                db_suffixes += [
                    f'{params.start_id:d}_{params.end_id:d}',
                ]

            if patch_mode:
                db_suffixes += [
                    f'{params.patch_height:d}_{params.patch_width:d}',
                    f'{params.min_stride:d}_{params.max_stride:d}',
                ]
            if params.shuffle:
                db_suffixes.append('rnd')

            if params.sample:
                db_suffixes.append('smp_{}'.format(params.sample))

            if params.n_rot > 0:
                db_suffixes.append(f'rot_{params.min_rot:d}_{params.max_rot:d}_{params.n_rot:d}')

            if params.enable_flip:
                db_suffixes.append('flip')

            if params.rle.instance_wise:
                db_suffixes.append('inst')

            params.db_suffix = '-'.join(db_suffixes)

        params.db_path = f'{params.db_path}-{params.db_suffix}'

        in_json_name = params.db_suffix

        if params.seq_id >= 0:
            params.seq_start_id = params.seq_end_id = params.seq_id

        if params.seq_start_id > 0 or params.seq_end_id >= 0:
            assert params.seq_end_id >= params.seq_start_id, "seq_end_id must to be >= seq_start_id"
            seq_suffix = f'seq_{params.seq_start_id}_{params.seq_end_id}'
            in_json_name = f'{in_json_name}-{seq_suffix}'

        in_json_fname = f'{in_json_name}.{params.ann_ext}'
        in_json_path = linux_path(params.db_path, in_json_fname)

    img_json_dict = load_seg_annotations(in_json_path)
    image_infos = img_json_dict['images']

    # shutil.copy(in_json_path, out_json_path)

    if not params.output_dir:
        params.output_dir = linux_path(params.db_path, 'tfrecord')

    os.makedirs(params.output_dir, exist_ok=True)

    vid_infos = None
    if params.load_vid:
        vid_infos = get_vid_infos(params.db_path, image_infos, params.rle.instance_wise, params.rle.instance_coco_rle)

    metrics = dict(
        db={},
        method_0={},
        method_1={},
        method_2={},
        polygons={},
    )

    out_json_name = in_json_name

    if params.rle_to_json:
        params.rle.starts_offset, params.rle.lengths_offset, params.rle.class_offset = task_utils.get_rle_offsets(
            params.rle.starts_offset, params.rle.lengths_offset, params.rle.class_offset,
            params.rle.length_as_class, params.rle.max_length, params.rle.subsample,
            params.rle.shared_coord,
            params.rle.diff_mask,
            n_classes)

        rle_suffix = get_rle_suffix(params.rle, multi_class)
        if rle_suffix:
            out_json_name = f'{out_json_name}-{rle_suffix}'

    out_json_fname = f'{out_json_name}.{params.ann_ext}'
    out_json_path = linux_path(params.db_path, out_json_fname)

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
            metrics_path_dir = os.path.dirname(metrics_path)
            os.makedirs(metrics_path_dir, exist_ok=True)

            with open(metrics_path, 'w') as f:
                f.write('\n'.join(map(str, val)))


if __name__ == '__main__':
    main()
