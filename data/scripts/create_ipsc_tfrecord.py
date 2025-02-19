import collections
import json
import os
import sys

import cv2

dproc_path = os.path.join(os.path.expanduser("~"), "ipsc/ipsc_data_processing")

sys.path.append(os.getcwd())
sys.path.append(dproc_path)

import numpy as np
import tensorflow as tf
import paramparse

import vocab
from data.scripts import tfrecord_lib
from tasks.visualization import vis_utils
from tasks import task_utils

from eval_utils import add_suffix


class Params(paramparse.CFG):

    def __init__(self):
        paramparse.CFG.__init__(self, cfg_prefix='tf')
        self.ann_file = ''
        self.ann_suffix = []
        self.image_dir = ''
        self.masks_dir = ''
        self.enable_masks = 1
        self.vis = 0

        self.n_proc = 0
        self.ann_ext = 'json.gz'
        self.num_shards = 32
        self.output_dir = ''
        self.xml_output_file = ''

        self.start_seq_id = 0
        self.end_seq_id = -1

        self.start_frame_id = 0
        self.end_frame_id = -1
        self.frame_stride = 1


def load_instance_annotations(annotation_path):
    print(f'Reading coco annotations from {annotation_path}')
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
    category_id_to_name_map = dict(
        (element['id'], element['name']) for element in annotations['categories'])

    assert 0 not in category_id_to_name_map.keys(), "class IDs must to be > 0"

    img_to_ann = collections.defaultdict(list)
    for ann in annotations['annotations']:
        image_id = ann['image_id']
        img_to_ann[image_id].append(ann)

    return image_info, category_id_to_name_map, img_to_ann


def coco_annotations_to_lists(obj_annotations, id_to_name_map):
    data = dict((k, list()) for k in [
        'xmin', 'xmax', 'ymin', 'ymax', 'is_crowd', 'category_id',
        'category_names', 'area'])

    for ann in obj_annotations:
        (x, y, width, height) = tuple(ann['bbox'])
        xmin, xmax, ymin, ymax = float(x), float(x + width), float(y), float(y + height)

        assert xmax > xmin and ymax > ymin, f"invalid bbox: {[xmin, xmax, ymin, ymax]}"

        data['xmin'].append(xmin)
        data['xmax'].append(xmax)
        data['ymin'].append(ymin)
        data['ymax'].append(ymax)
        data['is_crowd'].append(ann['iscrowd'])
        category_id = int(ann['category_id'])
        data['category_id'].append(category_id)
        data['category_names'].append(id_to_name_map[category_id].encode('utf8'))
        data['area'].append(ann['area'])

    return data


def obj_annotations_to_feature_dict(obj_annotations, id_to_name_map):
    data = coco_annotations_to_lists(obj_annotations, id_to_name_map)

    feature_dict = {
        'image/object/bbox/xmin':
            tfrecord_lib.convert_to_feature(data['xmin']),
        'image/object/bbox/xmax':
            tfrecord_lib.convert_to_feature(data['xmax']),
        'image/object/bbox/ymin':
            tfrecord_lib.convert_to_feature(data['ymin']),
        'image/object/bbox/ymax':
            tfrecord_lib.convert_to_feature(data['ymax']),
        'image/object/class/text':
            tfrecord_lib.convert_to_feature(data['category_names']),
        'image/object/class/label':
            tfrecord_lib.convert_to_feature(data['category_id']),
        'image/object/is_crowd':
            tfrecord_lib.convert_to_feature(data['is_crowd']),
        'image/object/area':
            tfrecord_lib.convert_to_feature(data['area'], value_type='float_list'),
    }
    return feature_dict


def flatten_segmentation(seg):
    flat_seg = []
    for i, s in enumerate(seg):
        if i > 0:
            flat_seg.extend([vocab.SEPARATOR_FLOAT, vocab.SEPARATOR_FLOAT])
        flat_seg.extend(s)
    return flat_seg


def obj_annotations_to_seg_dict(obj_annotations):
    segs = []
    seg_lens = []
    for ann in obj_annotations:
        if ann['iscrowd']:
            seg_lens.append(0)
        else:
            seg = flatten_segmentation(ann['segmentation'])
            segs.extend(seg)
            seg_lens.append(len(seg))
    seg_sep = [0] + list(np.cumsum(seg_lens))
    return {
        'image/object/segmentation_v': tfrecord_lib.convert_to_feature(segs),
        'image/object/segmentation_sep': tfrecord_lib.convert_to_feature(seg_sep),
    }


def generate_annotations(images,
                         image_dir,
                         masks_dir,
                         category_id_to_name_map,
                         img_to_obj_ann,
                         enable_masks,
                         vis,
                         # img_to_cap_ann,
                         # img_to_key_ann,
                         # img_to_pan_ann,
                         # is_category_thing,
                         ):
    """Generator for COCO annotations."""
    for image in images:
        object_ann = img_to_obj_ann.get(image['id'], {})

        # caption_ann = img_to_cap_ann.get(image['id'], {})
        #
        # keypoint_ann = img_to_key_ann.get(image['id'], {})
        #
        # panoptic_ann = img_to_pan_ann.get(image['id'], {})

        if vis:
            vis_utils.vis_json_ann(image, object_ann, category_id_to_name_map,
                                   image_dir, is_video=False)

        yield (
            image,
            image_dir,
            masks_dir,
            category_id_to_name_map,
            object_ann,
            enable_masks,
            # caption_ann,
            # keypoint_ann,
            # panoptic_ann,
            # is_category_thing,
        )


def create_tf_example(
        image,
        image_dir,
        masks_dir,
        category_id_to_name_map,
        object_ann,
        enable_masks,
        # caption_ann,
        # keypoint_ann, panoptic_ann,
        # is_category_thing
):
    """Converts image and annotations to a tf.Example proto."""
    # Add image features.
    image_height = image['height']
    image_width = image['width']
    filename = image['file_name']
    image_id = image['id']

    image_path = os.path.join(image_dir, filename)
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()

    feature_dict = tfrecord_lib.image_info_to_feature_dict(
        image_height, image_width, filename, image_id, encoded_jpg, 'jpg')

    # Add annotation features.
    if object_ann:
        obj_feature_dict = obj_annotations_to_feature_dict(
            object_ann,
            category_id_to_name_map)
        feature_dict.update(obj_feature_dict)

        if enable_masks:
            assert masks_dir, "masks_dir must be provided"
            # seg_feature_dict = obj_annotations_to_seg_dict(object_ann)
            image_dir_, image_name_ = (os.path.dirname(image_path),
                                       os.path.splitext(os.path.basename(image_path))[0])
            masks_path = os.path.join(image_dir_, masks_dir, f'{image_name_}.png')

            # mask = cv2.imread(masks_path)
            # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # rle_lens = []
            # for size_ in (640, 320, 160, 80, 40):
            #     mask_ = cv2.resize(mask, (size_, size_))
            #     rle, rle_norm = task_utils.mask_to_rle(mask_)
            #     rle_len = len(rle_norm)
            #     rle_lens.append(str(rle_len))
            #
            # rle_lens_str = '\t'.join(rle_lens)
            # with open('rle_len.txt', 'a') as fid:
            #     fid.write(f'{filename}\t{rle_lens_str}\n')

            # rle_str = rle_str.encode('utf-8')
            with tf.io.gfile.GFile(masks_path, 'rb') as fid:
                encoded_png = fid.read()
            seg_feature_dict = {
                'image/mask': tfrecord_lib.convert_to_feature(encoded_png),
                # 'image/rle': tfrecord_lib.convert_to_feature(rle_str),
            }
            feature_dict.update(seg_feature_dict)

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example, 0


def main():
    params: Params = paramparse.process(Params)

    assert params.ann_file, "ann_file must be provided"
    assert params.image_dir, "image_dir must be provided"

    if params.enable_masks:
        print('masks are enabled')
        if not params.masks_dir:
            params.masks_dir = 'masks'
    ann_suffix = params.ann_suffix
    if ann_suffix:
        ann_suffix = '-'.join(ann_suffix)
        params.ann_file = f'{params.ann_file}-{ann_suffix}'

    if params.start_seq_id > 0 or params.end_seq_id >= 0:
        assert params.end_seq_id >= params.start_seq_id, "end_seq_id must to be >= start_seq_id"
        seq_sufix = f'seq-{params.start_seq_id}_{params.end_seq_id}'
        params.ann_file = add_suffix(params.ann_file, seq_sufix, sep='-')

    if params.start_frame_id > 0 or params.end_frame_id >= 0 or params.frame_stride > 1:
        frame_suffix = f'{params.start_frame_id}_{params.end_frame_id}'
        if params.frame_stride > 1:
            frame_suffix = f'{frame_suffix}_{params.frame_stride}'

        params.ann_file = add_suffix(params.ann_file, frame_suffix, sep='-')

    params.ann_file = os.path.join(params.image_dir, f'{params.ann_file}.{params.ann_ext}')

    image_info, category_id_to_name_map, img_to_obj_ann = load_instance_annotations(params.ann_file)

    if not params.output_dir:
        params.output_dir = os.path.join(params.image_dir, 'tfrecord')

    # img_to_key_ann = load_keypoint_annotations(params.key_ann_file)
    # img_to_cap_ann = load_caption_annotations(params.cap_ann_file)
    # img_to_pan_ann, is_category_thing = load_panoptic_annotations(
    #     params.pan_ann_file)

    os.makedirs(params.output_dir, exist_ok=True)

    out_name = os.path.basename(params.ann_file).split(os.extsep)[0]

    annotations_iter = generate_annotations(
        images=image_info,
        image_dir=params.image_dir,
        masks_dir=params.masks_dir,
        category_id_to_name_map=category_id_to_name_map,
        img_to_obj_ann=img_to_obj_ann,
        enable_masks=params.enable_masks,
        vis=params.vis,
        # img_to_cap_ann=img_to_cap_ann,
        # img_to_key_ann=img_to_key_ann,
        # img_to_pan_ann=img_to_pan_ann,
        # is_category_thing=is_category_thing
    )
    output_path = os.path.join(params.output_dir, out_name)
    os.makedirs(output_path, exist_ok=True)

    tfrecord_pattern = os.path.join(output_path, 'shard')

    tfrecord_lib.write_tf_record_dataset(
        output_path=tfrecord_pattern,
        annotation_iterator=annotations_iter,
        process_func=create_tf_example,
        num_shards=params.num_shards,
        multiple_processes=params.n_proc,
        iter_len=len(image_info),
    )

    print(f'out_name: {out_name}')
    print(f'output_path: {output_path}')


if __name__ == '__main__':
    main()
