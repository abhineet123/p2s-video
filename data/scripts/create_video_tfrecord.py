import collections
import os
import sys
import pickle

import cv2
import math
import numpy as np

dproc_path = os.path.join(os.path.expanduser("~"), "ipsc/ipsc_data_processing")

sys.path.append(os.getcwd())
sys.path.append(dproc_path)

# import numpy as np
from PIL import Image
import imageio

import paramparse
from data.scripts import tfrecord_lib
from tasks.visualization import vis_utils
from tasks import task_utils

from eval_utils import col_bgr, show_labels, resize_ar, draw_box


class Params(paramparse.CFG):
    def __init__(self):
        paramparse.CFG.__init__(self, cfg_prefix='tf_vid')
        self.ann_file = ''
        self.ann_suffix = []
        self.ann_ext = 'json.gz'
        self.class_names_path = ''

        self.frame_gaps = []
        self.length = 0
        self.stride = 1
        self.strides = []
        self.sample = 0

        self.add_stride_info = 1

        self.image_dir = ''
        self.n_proc = 0
        self.save_json = 0
        self.num_shards = 32
        self.output_dir = ''

        self.start_seq_id = 0
        self.end_seq_id = -1

        self.start_frame_id = 0
        self.end_frame_id = -1
        self.frame_stride = -1

        self.vis = 0
        self.show_mask = 0
        self.arrow = 1


def save_dict_to_json(json_dict, json_path, label='ytvis annotations'):
    print(f'saving {label} to {json_path}')
    json_kwargs = dict(
        indent=4
    )
    if json_path.endswith('.json.gz'):
        import compress_json
        compress_json.dump(json_dict, json_path, json_kwargs=json_kwargs)
    elif json_path.endswith('.json'):
        import json
        output_json = json.dumps(json_dict, **json_kwargs)
        with open(json_path, 'w') as f:
            f.write(output_json)
    else:
        raise AssertionError(f'Invalid json_path: {json_path}')


def load_ytvis_annotations(annotation_path, vid_id_offset):
    assert os.path.exists(annotation_path), f"ytvis json does not exist: {annotation_path}"

    print(f'Reading ytvis annotations from {annotation_path}')
    if annotation_path.endswith('.json'):
        import json
        with open(annotation_path, 'r') as f:
            json_data = json.load(f)
    elif annotation_path.endswith('.json.gz'):
        import compress_json
        json_data = compress_json.load(annotation_path)
    else:
        raise AssertionError(f'Invalid annotation_path: {annotation_path}')

    video_info = json_data['videos']
    annotations = json_data['annotations']
    categories = json_data['categories']

    if vid_id_offset > 0:
        for vid in video_info:
            vid["id"] += vid_id_offset

    category_id_to_name_map = dict(
        (category['id'], category['name']) for category in categories)

    # assert 0 not in category_id_to_name_map.keys(), "class IDs must to be > 0"

    try:
        bkg_class = category_id_to_name_map[0]
    except KeyError:
        pass
    else:
        assert bkg_class == 'background', "class id 0 can be used only for background"

    vid_to_ann = collections.defaultdict(list)
    for ann in annotations:
        ann['video_id'] += vid_id_offset
        vid_id = ann['video_id']
        vid_to_ann[vid_id].append(ann)

    for video_ in video_info:
        file_ids = video_['file_ids']
        assert all(i < j for i, j in zip(file_ids, file_ids[1:])), \
            "file_ids should be strictly increasing"

    return video_info, category_id_to_name_map, vid_to_ann, json_data


def show_img_rgb(frame, cu_z, mlab):
    from tvtk.api import tvtk

    h, w = frame.shape[:2]

    frame_gs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gs = np.transpose(frame_gs)
    mlab_im = mlab.imshow(frame_gs, extent=[0, w, 0, h, cu_z, cu_z], opacity=0.75)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    colors = tvtk.UnsignedCharArray()
    frame_ = frame
    # frame_ = frame_.transpose((1, 0, 2))
    colors.from_array(frame_.reshape(-1, 3))
    # mlab_im = mlab.imshow(np.ones(frame.shape[:2]),  extent=[0, w, 0, h, cu_z, cu_z])
    mlab_im.actor.input.point_data.scalars = colors


def to_rgb(col, norm=1):
    if isinstance(col, str):
        col = col_bgr[col]

    col_rgb = col[::-1]
    if norm:
        col_rgb = [k / 255. for k in col_rgb]
    return tuple(col_rgb)


def get_camera_view(mlab):
    azimuth, elevation, distance, focalpoint = mlab.view()
    roll = mlab.roll()

    return dict(
        azimuth=azimuth,
        elevation=elevation,
        distance=distance,
        focalpoint=focalpoint,
        roll=roll,
    )


def set_camera_view(mlab, params_dict):
    mlab.view(**params_dict)
    # roll = mlab.roll()


def show_vid_objs(video, image_dir, obj_annotations, id_to_name_map, class_id_to_col, fig,
                  show_mask, arrow):
    from mayavi import mlab

    file_ids = video['file_ids']
    file_names = video['file_names']

    file_paths = [os.path.join(image_dir, filename) for filename in file_names]
    base_file_names = [os.path.splitext(os.path.basename(file_path))[0] for file_path in file_paths]

    # z_gap = 500
    # title_font_size=48

    txt_above_img = 1
    single_row = 0
    opacity = 0.25
    token_win_name = 'Token Sequence'
    init_pause = 1

    z_gap = 1000
    title_font_size = 64
    margin = 10

    if txt_above_img:
        """text above img"""
        txt_font_size = 32
        txt_line_gap = 8
        allow_linebreak = 1
        cmb_concat_axis = 0
        if single_row:
            """single row"""
            text_img_h, text_img_w = 100, 5100
        else:
            """two rows"""
            text_img_h, text_img_w = 100, 1920
    else:
        """text to the right of img"""
        txt_font_size = 24
        txt_line_gap = 5
        allow_linebreak = 1

        text_img_h, text_img_w = 1030, 600
        cmb_concat_axis = 1

    only_border = 0

    vis_img_h, vis_img_w = 1030, 1320
    show_header = 1
    white_bkg = 0
    pause = 1

    if white_bkg:
        cols = (
            'purple',
            'dark_orange', 'peach_puff_4', 'deep_sky_blue',
            'magenta', 'blue', 'orange',
            'maroon', 'slate_gray', 'dark_orange',
            'forest_green', 'cyan', 'green_yellow',
            'green', 'red',
        )
    else:
        cols = (
            'dark_khaki',
            'yellow', 'peach_puff', 'cyan',
            'magenta', 'dark_orange',
            'maroon', 'slate_gray',
            'medium_purple',
            'deep_sky_blue',
            'forest_green', 'pale_turquoise', 'green_yellow',
            'green', 'red',

        )

    if show_mask:
        cols = [col for col in cols if col not in class_id_to_col.values()]

    bkg_col = 'white' if white_bkg else 'black'
    frg_col = 'white' if bkg_col == 'black' else 'black'

    bkg_col = col_bgr[bkg_col]
    frg_col = col_bgr[frg_col]

    bkg_col_rgb = to_rgb(bkg_col)
    frg_col_rgb = to_rgb(frg_col)

    frames = [cv2.imread(file_path)
              for file_path in file_paths]
    masks = None
    if show_mask:
        mask_file_paths = [os.path.join(os.path.dirname(file_path), 'masks',
                                        f'{os.path.splitext(os.path.basename(file_path))[0]}.png')
                           for file_path in file_paths]
        masks = [np.asarray(Image.open(file_path)) for file_path in mask_file_paths]

    img_h, img_w = frames[0].shape[:2]

    if text_img_w <= 0:
        text_img_w = img_w

    if text_img_h <= 0:
        text_img_h = img_h

    if fig is None:
        fig = mlab.figure(
            'Video Clip',
            size=(vis_img_w, vis_img_h),
            bgcolor=bkg_col_rgb
        )
        first_fig = True
        try:
            with open('view_params.pkl', 'rb') as f:
                view_params = pickle.load(f)
        except FileNotFoundError:
            view_params = None
        else:
            set_camera_view(mlab, view_params)
    else:
        mlab.clf(figure=fig)
        first_fig = False
        view_params = get_camera_view(mlab)

    vid_len = len(file_paths)

    # z = (vid_len - 1) * z_gap
    # containing_box = (
    #     (0, 0, 0), (0, h, 0), (w, h, 0), (w, 0, 0), (0, 0, 0),
    #     (0, 0, z), (0, h, z), (w, h, z), (w, 0, z), (0, 0, z),
    #     (0, h, z), (0, h, 0), (w, h, 0), (w, h, z), (w, 0, z), (w, 0, 0),
    # )
    #
    # box_x = [k[0] for k in containing_box]
    # box_y = [k[1] for k in containing_box]
    # box_z = [k[2] for k in containing_box]

    # mlab.plot3d(box_x, box_y, box_z, color=(1, 1, 1), line_width=2.0, tube_radius=1.5)
    text_img = np.full((text_img_h, text_img_w, 3), bkg_col, dtype=np.uint8)

    n_objs = len(obj_annotations)

    # curr_obj_ids = [ann['id'] for ann in obj_annotations]
    curr_obj_ids = list(range(n_objs))

    # max_obj_id = max(curr_obj_ids)
    n_col_levels = int(n_objs ** (1. / 3) + 1)

    if n_objs <= len(cols):
        rgb_cols = cols[:n_objs]
    else:
        col_levels = [int(x) for x in np.linspace(
            # exclude too light and too dark colours to avoid confusion with white and black
            50, 200,
            n_col_levels, dtype=int)]
        import itertools
        import random
        rgb_cols = list(itertools.product(col_levels, repeat=3))
        random.shuffle(rgb_cols)

    assert len(rgb_cols) >= n_objs, "insufficient number of colours created"

    curr_obj_cols = [rgb_cols[k] for k in curr_obj_ids]

    for frame_id, frame in enumerate(frames):
        cu_z = (frame_id + 1) * z_gap

        if show_mask:
            seg_mask = masks[frame_id]
            for class_id, class_name in id_to_name_map.items():
                class_col = class_id_to_col[class_id]
                class_col_bgr = col_bgr[class_col]
                seg_mask_binary = seg_mask == class_id
                frame[seg_mask_binary] = (frame[seg_mask_binary] * (1. - opacity) +
                                          np.asarray(class_col_bgr) * opacity)
        file_id = int(file_ids[frame_id])
        file_name = base_file_names[frame_id]

        if show_header:
            # mlab.text(w/2, 0, file_name, z=cu_z, figure=fig, color=frg_col_rgb, line_width=1.0)
            if show_header == 2:
                obj_txt = "object" if n_objs == 1 else "objects"
                file_txt = f'frame {file_id + 1}: {file_name} :: {n_objs} {obj_txt}'
            else:
                file_txt = f'frame {file_id + 1}'

            frame, _, _ = vis_utils.write_text(frame, file_txt, 5, 5, frg_col, font_size=title_font_size)
            # show_labels(frame, curr_obj_ids, curr_obj_cols)

        show_img_rgb(frame, cu_z, mlab)

        frames[frame_id] = frame

    # mlab.axes(figure=fig, color=frg_col_rgb,
    #           x_axis_visibility=True,
    #           y_axis_visibility=True,
    #           z_axis_visibility=True,
    #           extent=[0, w, 0, h, 0, cu_z],
    #           # ranges=[0, w, 0, h, cu_z, cu_z]
    #           )

    # mlab.draw()
    # mlab.show()

    if first_fig:
        cv2.imshow(token_win_name, text_img)
        k = cv2.waitKey(1 - init_pause)
        if k == 32:
            init_pause = 1 - init_pause
            pause = 1 - pause
        elif k == 27:
            exit()
        view_params = get_camera_view(mlab)
        with open('view_params.pkl', 'wb') as f:
            pickle.dump(view_params, f, pickle.HIGHEST_PROTOCOL)
    else:
        if view_params is not None:
            set_camera_view(mlab, view_params)

    text_x = text_y = margin

    frame = np.copy(frames[0])
    bbs = []
    # gif_images = []

    for ann_id, ann in enumerate(obj_annotations):
        bboxes = ann['bboxes']
        col_id = ann_id % len(cols)
        col_str = cols[col_id]
        col = col_bgr[col_str]

        prev_bbox = None

        col_rgb = to_rgb(col)

        for bbox_id, bbox in enumerate(bboxes):
            if vid_len == 1:
                assert bbox is not None, "bbox cannot be None for static frames"

            if bbox is not None:

                draw_box(frame, box=bbox, color=col_str, thickness=3, xywh=True)

                (x, y, width, height) = tuple(bbox)
                xmin, ymin, xmax, ymax = int(x), int(y), int(x + width), int(y + height)

                assert ymax > ymin and xmax > xmin, f"invalid bbox: {bbox}"

                cu_z = (bbox_id + 1) * z_gap

                xs = [xmin, xmin, xmax, xmax, xmin]
                ys = [ymin, ymax, ymax, ymin, ymin]
                zs = [cu_z, ] * len(xs)

                mlab.plot3d(xs, ys, zs, color=col_rgb, line_width=3.0, tube_radius=3.0)
                set_camera_view(mlab, view_params)

                if prev_bbox is not None:
                    xmin_, ymin_, xmax_, ymax_, pre_z = prev_bbox
                    xs_ = [xmin, xmin_, xmin_, xmin, xmax, xmax_, xmax_, xmax]
                    ys_ = [ymin, ymin_, ymax_, ymax, ymax, ymax_, ymin_, ymin]
                    zs_ = [cu_z, pre_z, pre_z, cu_z, cu_z, pre_z, pre_z, cu_z]
                    mlab.plot3d(xs_, ys_, zs_, color=col_rgb, line_width=2.0, tube_radius=2.0)
                    set_camera_view(mlab, view_params)

                # bbox_txts.append(bbox_txt)
                prev_bbox = [xmin, ymin, xmax, ymax, cu_z]

                bbox_txt = f'{int(xmin)}, {int(ymin)}, {int(xmax)}, {int(ymax)}, '

            else:
                prev_bbox = None
                bbox_txt = f'NA, NA, NA, NA, '

            text_img, text_x, text_y, text_bbs = vis_utils.write_text(
                text_img, bbox_txt, text_x, text_y, col, show=1, win_name=token_win_name,
                bb=2, line_gap=txt_line_gap, font_size=txt_font_size,
                allow_linebreak=allow_linebreak, margin=margin)

            if vid_len == 1:
                bb = (text_bbs, bbox, col)
                bbs.append(bb)

            k = cv2.waitKey(100)
            if k == 27:
                sys.exit()

        class_id = int(ann['category_id'])
        class_name = id_to_name_map[class_id]

        text_img, text_x, text_y, text_bbs = vis_utils.write_text(
            text_img, f'{class_name}, ', text_x, text_y, col,
            show=1, win_name=token_win_name, bb=2, line_gap=txt_line_gap,
            font_size=txt_font_size, allow_linebreak=allow_linebreak)

        if vid_len == 1:
            text_bbs_, bbox_, col_ = bbs[-1]
            text_bbs_ += text_bbs
            bbs[-1] = (text_bbs_, bbox_, col_)
            cmb_frames = show_cmb(frame, text_img, 2, [bbs[-1], ], cmb_concat_axis,
                                  white_bkg, only_border)
            for cmb_frame_id, cmb_frame in enumerate(cmb_frames):
                cv2.imwrite(f'log/{base_file_names[0]}_obj_{ann_id:03d}_{cmb_frame_id}.png', cmb_frame)
                # gif_images.append(cmb_frame)

        k = cv2.waitKey(1 - pause)
        if k == 32:
            pause = 1 - pause
        elif k == 27:
            sys.exit()

    text_img, text_x, text_y, text_bbs = vis_utils.write_text(
        text_img, 'EOS', text_x, text_y, frg_col, show=1, win_name=token_win_name,
        bb=2, line_gap=txt_line_gap, font_size=txt_font_size, allow_linebreak=allow_linebreak)
    if vid_len == 1:
        cmb_frames = show_cmb(frame, text_img, 1, bbs, cmb_concat_axis, white_bkg, only_border)
        cv2.imwrite(f'log/static_det_token_vis.png', cmb_frames[-1])
        # gif_images.append(cmb_frames[-1])
        # imageio.mimsave('log/static_det_token_vis.gif', gif_images, fps=1)

    k = cv2.waitKey(0)
    if k == 27:
        sys.exit()

    return fig


def show_cmb(frame, text_img, arrow, bbs, axis, white_bkg, only_border):
    if axis == 1:
        vis_frame, resize_factor, start_row, start_col = resize_ar(
            frame, height=text_img.shape[0], return_factors=1, only_border=only_border,
            white_bkg=white_bkg)
        cmb_frame = np.concatenate((vis_frame, text_img), axis=1)

    else:
        vis_frame, resize_factor, start_row, start_col = resize_ar(
            frame, width=text_img.shape[1], return_factors=1, only_border=only_border,
            white_bkg=white_bkg)
        cmb_frame = np.concatenate((text_img, vis_frame), axis=0)

    bb_offset = 5
    arrow_offset = 10

    arrow_tip_norm = 700

    cmb_frames = []

    for text_bbs, bb, col in bbs:
        n_words = len(text_bbs)
        (x, y, width, height) = tuple(bb)
        """vis_frame has been resized"""
        x = int(x * resize_factor + start_col)
        y = int(y * resize_factor + start_row)
        width = int(width * resize_factor)
        height = int(height * resize_factor)

        if axis == 1:
            """text_img is to the right of vis_frame"""
            for text_bb in text_bbs:
                text_bb[0] += int(vis_frame.shape[1])
                text_bb[2] += int(vis_frame.shape[1])
        else:
            """text_img is above vis_frame"""
            y += int(text_img.shape[0])

        multi_line_ids = [_id for _id, text_bb in enumerate(text_bbs) if text_bb[-1]]

        if arrow == 2:
            assert n_words >= 4, "n_words must be >= 4"
            if multi_line_ids:
                assert 1 not in multi_line_ids, "second word cannot be on a new line"
                assert 3 not in multi_line_ids, "fourth word cannot be on a new line"

            left1, top1, right1, bottom1, _ = text_bbs[0]
            left2, top2, right2, bottom2, _ = text_bbs[1]
            text_bb1 = [left1 - bb_offset, top1 - bb_offset, right2 + bb_offset, bottom2 + bb_offset]

            left3, top3, right3, bottom3, _ = text_bbs[2]
            left4, top4, right4, bottom4, _ = text_bbs[3]
            text_bb2 = [left3 - bb_offset, top3 - bb_offset, right4 + bb_offset, bottom4 + bb_offset]

            cmb_frames.append(np.copy(cmb_frame))

            pt1 = [int(x), int(y)]
            pt2 = [int((text_bb1[0] + text_bb1[2]) / 2), int(text_bb1[3]) + arrow_offset]
            pt_dist1 = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
            tip_length = 0.02 * arrow_tip_norm / pt_dist1
            print(f'pt_dist1: {pt_dist1}')
            print(f'tip_length: {tip_length}')

            draw_box(cmb_frame, box=text_bb1, color=col, thickness=1, xywh=False, is_dotted=1)
            cmb_frame = cv2.arrowedLine(
                cmb_frame, pt1, pt2, col, 2, tipLength=tip_length)
            cmb_frames.append(np.copy(cmb_frame))
            cv2.imshow('cmb_frame', cmb_frame)
            cv2.waitKey(100)

            pt1 = [int(x + width), int(y + height)]
            pt2 = [int((text_bb2[0] + text_bb2[2]) / 2), int(text_bb2[3]) + arrow_offset]
            pt_dist2 = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
            tip_length = 0.02 * arrow_tip_norm / pt_dist2
            print(f'pt_dist2: {pt_dist2}')
            print(f'tip_length: {tip_length}')

            draw_box(cmb_frame, box=text_bb2, color=col, thickness=1, xywh=False, is_dotted=1)
            cmb_frame = cv2.arrowedLine(
                cmb_frame, pt1, pt2, col, 2, tipLength=tip_length)
            cmb_frames.append(np.copy(cmb_frame))
            cv2.imshow('cmb_frame', cmb_frame)
            cv2.waitKey(100)

        elif arrow == 1:
            id1, id2 = 0, -1
            if multi_line_ids:
                if multi_line_ids[0] >= n_words // 2:
                    id1 = multi_line_ids[0]
                else:
                    id2 = multi_line_ids[0] - 1

            left1, top1, right1, bottom1, _ = text_bbs[id1]
            left2, top2, right2, bottom2, _ = text_bbs[id2]
            text_bb_x, text_bb_y = int((left1 + right2) / 2), int(bottom1)
            text_bb_y += arrow_offset

            # """find corner nearest to text"""
            # bb_corners = [
            #     (x, y),
            #     (x + width, y),
            #     (x + width, y + height),
            #     (x, y + height),
            #     # (x + width / 2, y),
            # ]
            # bb_dists = np.asarray([
            #     math.sqrt((bb_x - text_bb_x) ** 2 + (bb_y - text_bb_y) ** 2)
            #     for (bb_x, bb_y) in bb_corners])
            # nearest_pt_id = np.argmin(bb_dists)
            # bb_x, bb_y = bb_corners[nearest_pt_id]

            if axis == 1:
                """text_img is to the right of vis_frame"""
                # center of right edge
                bb_x, bb_y = int(x + width), int(y + height / 2)
            else:
                """text_img is above vis_frame"""
                # center of top edge
                bb_x, bb_y = int(x + width / 2), int(y)

            pt1 = (bb_x, bb_y)
            pt2 = (text_bb_x, text_bb_y)
            pt_dist = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
            tip_length = 0.02 * arrow_tip_norm / pt_dist
            print(f'pt_dist: {pt_dist}')
            print(f'tip_length: {tip_length}')

            cmb_frame = cv2.arrowedLine(
                cmb_frame, pt1, pt2, col, 2, tipLength=0.02 * arrow_tip_norm / pt_dist)
            cmb_frames.append(np.copy(cmb_frame))

            # cmb_frame = vis_utils.arrowed_line(
            #     cmb_frame,
            #     (bb_x, bb_y),
            #     (text_bb_x, text_bb_y),
            #     width=2,
            #     color=col)
            cv2.imshow('cmb_frame', cmb_frame)
            cv2.waitKey(1)

    return cmb_frames


def ytvis_annotations_to_lists(obj_annotations: dict, id_to_name_map: dict, vid_len: int):
    """
    Converts YTVIS annotations to feature lists.
    """

    data = dict((k, list()) for k in [
        'is_crowd', 'category_id',
        'category_names', 'target_id'])

    for _id in range(vid_len):
        frame_data = dict((k, list()) for k in [
            f'xmin-{_id}', f'xmax-{_id}', f'ymin-{_id}', f'ymax-{_id}', f'area-{_id}'
        ])
        data.update(frame_data)

    for ann_id, ann in enumerate(obj_annotations):
        bboxes = ann['bboxes']
        areas = ann['areas']
        assert len(bboxes) == vid_len, \
            f"number of boxes: {len(bboxes)} does not match the video length: {vid_len}"
        assert len(areas) == vid_len, \
            f"number of areas: {len(areas)} does not match the video length: {vid_len}"

        data['is_crowd'].append(ann['iscrowd'])
        data['target_id'].append(int(ann['id']))
        category_id = int(ann['category_id'])
        data['category_id'].append(category_id)
        data['category_names'].append(id_to_name_map[category_id].encode('utf8'))

        valid_box_exists = False

        for bbox_id, bbox in enumerate(bboxes):
            area = areas[bbox_id]

            if bbox is None:
                assert area is None, "null bbox for non-null area"
                # xmin, ymin, xmax, ymax = -1, -1, -1, -1
                # area = -1
                # xmin = ymin = xmax = ymax = area = -1
                xmin = ymin = xmax = ymax = area = None
            else:
                assert area is not None, "null area for non-null bbox"
                (x, y, width, height) = tuple(bbox)
                xmin, ymin, xmax, ymax = float(x), float(y), float(x + width), float(y + height)

                assert ymax > ymin and xmax > xmin, f"invalid bbox: {bbox}"

                valid_box_exists = True

            # if bbox_id == 0:
            #     xmin = ymin = xmax = ymax = area = None

            data[f'xmin-{bbox_id}'].append(xmin)
            data[f'xmax-{bbox_id}'].append(xmax)
            data[f'ymin-{bbox_id}'].append(ymin)
            data[f'ymax-{bbox_id}'].append(ymax)
            data[f'area-{bbox_id}'].append(area)

        if not valid_box_exists:
            raise AssertionError('all boxes for an object cannot be None')

    return data


def obj_annotations_to_feature_dict(obj_annotations, id_to_name_map, vid_len):
    """Convert COCO annotations to an encoded feature dict.

    Args:
      obj_annotations: a list of object annotations.
      id_to_name_map: category id to category name map.

    Returns:
      a dict of tf features.
    """

    data = ytvis_annotations_to_lists(obj_annotations, id_to_name_map, vid_len)
    feature_dict = {
        'video/object/class/text':
            tfrecord_lib.convert_to_feature(data['category_names']),
        'video/object/class/label':
            tfrecord_lib.convert_to_feature(data['category_id']),
        'video/object/is_crowd':
            tfrecord_lib.convert_to_feature(data['is_crowd']),
        'video/object/target_id':
            tfrecord_lib.convert_to_feature(data['target_id']),
    }

    for _id in range(vid_len):
        frame_feature_dict = {
            f'video/object/bbox/xmin-{_id}':
                tfrecord_lib.convert_to_feature(data[f'xmin-{_id}']),
            f'video/object/bbox/xmax-{_id}':
                tfrecord_lib.convert_to_feature(data[f'xmax-{_id}']),
            f'video/object/bbox/ymin-{_id}':
                tfrecord_lib.convert_to_feature(data[f'ymin-{_id}']),
            f'video/object/bbox/ymax-{_id}':
                tfrecord_lib.convert_to_feature(data[f'ymax-{_id}']),
            f'video/object/area-{_id}':
                tfrecord_lib.convert_to_feature(data[f'area-{_id}'], value_type='float_list'),
        }
        feature_dict.update(frame_feature_dict)

    return feature_dict


def generate_video_annotations(
        params: Params,
        videos,
        category_id_to_name_map,
        class_id_to_col,
        vid_to_obj_ann,
        image_dir,

):
    fig = None
    for video in videos:
        object_anns = vid_to_obj_ann.get(video['id'], {})
        if params.vis == 1:
            from tasks.visualization import vis_utils
            vis_utils.vis_json_ann(video, object_anns, category_id_to_name_map, image_dir)

        if params.vis == 2:
            fig = show_vid_objs(
                video, image_dir, object_anns, category_id_to_name_map,
                class_id_to_col, fig, params.show_mask, params.arrow)
        else:
            yield (
                video,
                category_id_to_name_map,
                object_anns,
                image_dir,
            )


def create_video_tf_example(
        video,
        category_id_to_name_map,
        object_ann,
        image_dir,
):
    file_ids = video['file_ids']

    assert all(i < j for i, j in zip(file_ids, file_ids[1:])), \
        "file_ids should be strictly increasing"

    video_height = video['height']
    video_width = video['width']
    file_names = video['file_names']
    video_id = video['id']

    vid_len = len(file_names)

    file_paths = [os.path.join(image_dir, filename) for filename in file_names]
    # file_paths = [os.path.realpath(file_path) for file_path in file_paths]

    feature_dict = tfrecord_lib.video_info_to_feature_dict(
        video_height, video_width, file_ids, video_id, file_paths)

    if object_ann:
        # Bbox, area, etc.
        obj_feature_dict = obj_annotations_to_feature_dict(
            object_ann,
            category_id_to_name_map,
            vid_len)
        feature_dict.update(obj_feature_dict)

    import tensorflow as tf
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example, 0


def get_seq_info(video_info_, filenames_to_vid_id, length):
    vid_file_names = [video_['file_names'] for video_ in video_info_]

    if filenames_to_vid_id is not None:
        vid_ids = [filenames_to_vid_id[tuple(file_names)]
                   for file_names in vid_file_names]
    else:
        vid_ids = [str(video_['id']) for video_ in video_info_]

    vid_id_to_filenames = dict(
        (vid_id, file_names_) for vid_id, file_names_ in zip(vid_ids, vid_file_names, strict=True)
    )

    vid_id_to_seq_name = dict((vid_id_, file_names_[0].split('/')[0])
                              for vid_id_, file_names_ in zip(vid_ids, vid_file_names, strict=True))
    seq_to_vid_ids = collections.defaultdict(list)
    seq_to_file_names = collections.defaultdict(list)
    for vid_id, seq_name in vid_id_to_seq_name.items():
        seq_to_vid_ids[seq_name].append(vid_id)
        file_names_ = vid_id_to_filenames[vid_id]
        assert all(file_name.startswith(f'{seq_name}/') for file_name in file_names_), \
            f"invalid file name for {seq_name}"

        assert len(file_names_) == length, 'invalid subseq length'

        file_names_ = [file_name.replace(f'{seq_name}/', '') for file_name in file_names_]
        seq_to_file_names[seq_name].append(','.join(file_names_))

    seq_to_vid_ids = dict((seq, ','.join(vid_ids_)) for seq, vid_ids_ in seq_to_vid_ids.items())

    return dict(seq_to_file_names), dict(seq_to_vid_ids)


def main():
    params = Params()
    paramparse.process(params)

    assert params.image_dir, "image_dir must be provided"
    assert params.ann_file, "ann_file must be provided"
    assert params.class_names_path, "class_names_path must be provided"

    assert os.path.exists(params.image_dir), f"image_dir does not exist: {params.image_dir}"

    class_id_to_col, class_id_to_name = task_utils.read_class_info(params.class_names_path)

    if params.start_frame_id > 0 or params.end_frame_id >= 0 or params.frame_stride > 1:
        frame_suffix = f'{params.start_frame_id}_{params.end_frame_id}'
        if params.frame_stride > 1:
            frame_suffix = f'{frame_suffix}_{params.frame_stride}'
        params.ann_file = f'{params.ann_file}-{frame_suffix}'

    if params.length:
        params.ann_file = f'{params.ann_file}-length-{params.length}'

    if params.stride:
        params.ann_file = f'{params.ann_file}-stride-{params.stride}'

    if params.sample:
        params.ann_file = f'{params.ann_file}-sample-{params.sample}'

    if params.frame_gaps:
        ann_files = [f'{params.ann_file}-frame_gap-{frame_gap}' if frame_gap > 1 else params.ann_file
                     for frame_gap in params.frame_gaps]
    else:
        ann_files = [params.ann_file, ]

    ann_suffix = params.ann_suffix
    if ann_suffix:
        ann_suffix = '-'.join(ann_suffix)
        ann_files = [f'{ann_file}-{ann_suffix}' for ann_file in ann_files]

    if params.start_seq_id > 0 or params.end_seq_id >= 0:
        assert params.end_seq_id >= params.start_seq_id, "end_seq_id must to be >= start_seq_id"
        ann_files = [f'{ann_file}-seq-{params.start_seq_id}_{params.end_seq_id}'
                     for ann_file in ann_files]

    # params.ann_file = None

    ann_files = [os.path.join(params.image_dir, 'ytvis19', f'{ann_file}.{params.ann_ext}')
                 for ann_file in ann_files]
    vid_id_offset = 0
    n_all_vid = 0
    n_all_ann = 0
    video_info = []
    annotations_all = {}
    category_id_to_name_map = {}
    vid_to_obj_ann = collections.defaultdict(list)

    stride_to_file_names = {}
    stride_to_video_ids = {}

    out_name = os.path.basename(ann_files[0]).split(os.extsep)[0]

    for ann_file in ann_files:
        video_info_, category_id_to_name_map_, vid_to_obj_ann_, annotations_ = load_ytvis_annotations(
            ann_file, vid_id_offset)

        if params.add_stride_info:

            # assert all(len(video_['file_names'])==params.length for video_ in video_info_), \
            #     f"invalid vid length found"

            filenames_to_vid_id = dict(
                (tuple(video_['file_names']), str(video_['id'])) for video_ in video_info_
            )

            seq_name_to_file_names, seq_name_to_vid_ids = get_seq_info(video_info_, None, params.length)

            stride_to_video_ids[params.stride] = seq_name_to_vid_ids
            stride_to_file_names[params.stride] = seq_name_to_file_names

            if not params.strides:
                params.strides = range(params.stride + 1, params.length + 1)

            for _stride in params.strides:
                _stride_ann_file = ann_file.replace(f'stride-{params.stride}', f'stride-{_stride}')
                _stride_video_info_, _, _, _ = load_ytvis_annotations(_stride_ann_file, vid_id_offset=0)
                _stride_file_names = [str(video_['file_names']) for video_ in _stride_video_info_]
                _stride_video_ids = [filenames_to_vid_id[tuple(video_['file_names'])]
                                     for video_ in _stride_video_info_]

                seq_name_to_file_names, seq_name_to_vid_ids = get_seq_info(_stride_video_info_, filenames_to_vid_id,
                                                                           params.length)

                stride_to_video_ids[_stride] = seq_name_to_vid_ids
                stride_to_file_names[_stride] = seq_name_to_file_names

                print()
            annotations_['stride_to_video_ids'] = stride_to_video_ids
            annotations_['stride_to_file_names'] = stride_to_file_names

            vid_info_dict = dict(
                stride_to_video_ids=stride_to_video_ids,
                stride_to_file_names=stride_to_file_names,
            )
            vid_info_path = os.path.join(params.image_dir, 'ytvis19', f'{out_name}-vid_info.{params.ann_ext}')
            save_dict_to_json(vid_info_dict, vid_info_path, 'vid_info')

        new_vid_ids = set(vid_to_obj_ann_.keys())
        n_new_vid_ids = len(new_vid_ids)
        counts_ = annotations_['info']['counts'][0]
        n_new_vids = len(annotations_['videos'])
        n_new_anns = len(annotations_['annotations'])

        assert n_new_vids == counts_['videos'], "videos counts mismatch"
        assert n_new_vids == n_new_vid_ids, "n_new_vid_ids mismatch"
        assert n_new_anns == counts_['annotations'], "annotations counts mismatch"

        existing_vid_ids = set(vid_to_obj_ann.keys())
        overlapped_vid_ids = existing_vid_ids.intersection(new_vid_ids)

        assert not overlapped_vid_ids, f"overlapped_vid_ids found: {overlapped_vid_ids}"

        video_info += video_info_
        vid_to_obj_ann.update(vid_to_obj_ann_)
        category_id_to_name_map.update(category_id_to_name_map_)

        vid_id_offset = max(vid_to_obj_ann.keys())

        if not params.save_json:
            continue

        if not annotations_all:
            annotations_all = annotations_
        else:
            annotations_all['videos'] += annotations_['videos']
            annotations_all['annotations'] += annotations_['annotations']
            # annotations_all['categories'] += annotations_['categories']
            if params.add_stride_info:
                annotations_all['stride_to_video_ids'].update(annotations_['stride_to_video_ids'])

            counts = annotations_all['info']['counts'][0]

            counts['videos'] += n_new_vids
            counts['annotations'] += n_new_anns

        n_all_vid += n_new_vids
        n_all_ann += n_new_anns

        counts = annotations_all['info']['counts'][0]

        assert n_all_vid == counts['videos'], "n_all_vid mismatch"
        assert n_all_ann == counts['annotations'], "n_all_ann mismatch"

        assert n_all_vid == len(annotations_all['videos']), "annotations_all videos mismatch"
        assert n_all_ann == len(annotations_all['annotations']), "annotations_all annotations mismatch"

    # categories_all = annotations_all['categories']
    # categories_unique = [dict(t) for t in {tuple(d.items()) for d in categories_all}]
    # annotations_all['categories'] = categories_unique

    if not params.output_dir:
        output_dir = os.path.join(params.image_dir, 'ytvis19', 'tfrecord')
        print(f'setting output_dir to {output_dir}')
        params.output_dir = output_dir

    os.makedirs(params.output_dir, exist_ok=True)

    if len(params.frame_gaps) > 1:
        frame_gaps_suffix = 'fg_' + '_'.join(map(str, params.frame_gaps))
        if frame_gaps_suffix not in out_name:
            out_name = f'{out_name}-{frame_gaps_suffix}'

    if params.save_json:
        out_json_path = os.path.join(params.image_dir, 'ytvis19', f'{out_name}.{params.ann_ext}')
        annotations_all['info']['description'] = out_name
        save_dict_to_json(annotations_all, out_json_path)

    if params.save_json != 2:
        annotations_iter = generate_video_annotations(
            params=params,
            videos=video_info,
            category_id_to_name_map=category_id_to_name_map,
            class_id_to_col=class_id_to_col,
            vid_to_obj_ann=vid_to_obj_ann,
            image_dir=params.image_dir,
        )
        output_path = os.path.join(params.output_dir, out_name)
        os.makedirs(output_path, exist_ok=True)

        tfrecord_pattern = os.path.join(output_path, 'shard')

        tfrecord_lib.write_tf_record_dataset(
            output_path=tfrecord_pattern,
            annotation_iterator=annotations_iter,
            process_func=create_video_tf_example,
            num_shards=params.num_shards,
            multiple_processes=params.n_proc,
            iter_len=len(video_info),
        )
        print(f'output_path: {output_path}')

    print(f'out_name: {out_name}')


if __name__ == '__main__':
    main()
