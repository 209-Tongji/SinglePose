import os
import copy
import numpy as np
import torch.utils.data as data
from pycocotools.coco import COCO
from PIL import Image, ImageDraw, ImageFont
from utils import bbox_clip_xyxy, bbox_xywh_to_xyxy
from transform import SimpleTransform

import cv2

class MSCOCO(data.Dataset):
    CLASSES = ['person']
    EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    num_joints = 17
    joint_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                   [9, 10], [11, 12], [13, 14], [15, 16]]
    joints_name = ('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',    # 4
                   'left_shoulder', 'right_shoulder',                           # 6
                   'left_elbow', 'right_elbow',                                 # 8
                   'left_wrist', 'right_wrist',                                 # 10
                   'left_hip', 'right_hip',                                     # 12
                   'left_knee', 'right_knee',                                   # 14
                   'left_ankle', 'right_ankle')                                 # 16
    
    def __init__(self,
                 root,
                 ann_file,
                 images_dir,
                 cfg,
                 train=True,
                 skip_empty=True):
        
        self._root = root
        self._ann_file = os.path.join(root, ann_file)
        self._images_dir = images_dir
        self._train = train
        self._skip_empty = skip_empty
        self._check_centers = False
        self.num_class = len(self.CLASSES)

        self._input_size = cfg.DATA_PRESET.IMAGE_SIZE
        self._output_size = cfg.DATA_PRESET.HEATMAP_SIZE

        self._scale_factor = cfg.DATASET.TRAIN.AUG.SCALE_FACTOR
        self._rot = cfg.DATASET.TRAIN.AUG.ROT_FACTOR
        self.num_joints_half_body = cfg.DATASET.TRAIN.AUG.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.TRAIN.AUG.PROB_HALF_BODY
        self._sigma = 2

        self._need_heatmap = cfg.DATA_PRESET.NEED_HEATMAP
        self._need_coord = cfg.DATA_PRESET.NEED_COORD
        self._need_centermap = cfg.DATA_PRESET.NEED_CENTERMAP

        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.lower_body_ids = (11, 12, 13, 14, 15, 16)

        
        self.transformation = SimpleTransform(
            self, scale_factor = self._scale_factor,
            input_size = self._input_size,
            output_size = self._output_size,
            rot = self._rot, sigma = self._sigma,
            train = self._train,
            need_heatmap=self._need_heatmap, need_coord=self._need_coord, need_centermap=self._need_centermap
        )

        self._items, self._labels = self._load_json()
        

    def __getitem__(self, idx):
        img_path = self._items[idx]
        img_id = int(os.path.splitext(os.path.basename(img_path))[0])

        label = copy.deepcopy(self._labels[idx])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        target = self.transformation(img, label)

        img = target.pop('image')
        bbox = target.pop('bbox')

        return img, target, img_id, bbox


    def __len__(self):
        return len(self._items)

    def _load_json(self):
        items = []
        labels = []
        
        _coco = COCO(self._ann_file)
        classes = [c['name'] for c in _coco.loadCats(_coco.getCatIds())]
        assert classes == self.CLASSES, "Incompatible category names with COCO. "

        self.json_id_to_contiguous = {
            v: k for k, v in enumerate(_coco.getCatIds())}

            # iterate through the annotations
        image_ids = sorted(_coco.getImgIds())
        for entry in _coco.loadImgs(image_ids):
            dirname, filename = entry['coco_url'].split('/')[-2:]
            abs_path = os.path.join(self._root, self._images_dir, dirname, filename)
            label = self._check_load_keypoints(_coco, entry)
            if not label:
                continue

            # num of items are relative to person, not image
            for obj in label:
                items.append(abs_path)
                labels.append(obj)

        return items, labels
    
    def _check_load_keypoints(self, coco, entry):
        """Check and load ground-truth keypoints"""
        ann_ids = coco.getAnnIds(imgIds=entry['id'], iscrowd=False)
        objs = coco.loadAnns(ann_ids)
        # check valid bboxes
        valid_objs = []
        width = entry['width']
        height = entry['height']

        for obj in objs:
            contiguous_cid = self.json_id_to_contiguous[obj['category_id']]
            if contiguous_cid >= self.num_class:
                # not class of interest
                continue
            if max(obj['keypoints']) == 0:
                continue
            # convert from (x, y, w, h) to (xmin, ymin, xmax, ymax) and clip bound
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(obj['bbox']), width, height)
            # require non-zero box area
            if obj['area'] <= 0 or xmax <= xmin or ymax <= ymin:
                continue
            if obj['num_keypoints'] == 0:
                continue
            # joints 3d: (num_joints, 3, 2); 3 is for x, y, z; 2 is for position, visibility
            joints_3d = np.zeros((self.num_joints, 3, 2), dtype=np.float32)
            for i in range(self.num_joints):
                joints_3d[i, 0, 0] = obj['keypoints'][i * 3 + 0]
                joints_3d[i, 1, 0] = obj['keypoints'][i * 3 + 1]
                # joints_3d[i, 2, 0] = 0
                visible = min(1, obj['keypoints'][i * 3 + 2])
                joints_3d[i, :2, 1] = visible
                # joints_3d[i, 2, 1] = 0

            if np.sum(joints_3d[:, 0, 1]) < 1:
                # no visible keypoint
                continue

            if self._check_centers and self._train:
                bbox_center, bbox_area = self._get_box_center_area((xmin, ymin, xmax, ymax))
                kp_center, num_vis = self._get_keypoints_center_count(joints_3d)
                ks = np.exp(-2 * np.sum(np.square(bbox_center - kp_center)) / bbox_area)
                if (num_vis / 80.0 + 47 / 80.0) > ks:
                    continue

            valid_objs.append({
                'bbox': (xmin, ymin, xmax, ymax),
                'width': width,
                'height': height,
                'joints_3d': joints_3d
            })

        if not valid_objs:
            if not self._skip_empty:
                # dummy invalid labels if no valid objects are found
                valid_objs.append({
                    'bbox': np.array([-1, -1, 0, 0]),
                    'width': width,
                    'height': height,
                    'joints_3d': np.zeros((self.num_joints, 2, 2), dtype=np.float32)
                })
        return valid_objs

    def _get_box_center_area(self, bbox):
        """Get bbox center"""
        c = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        return c, area

    def _get_keypoints_center_count(self, keypoints):
        """Get geometric center of all keypoints"""
        keypoint_x = np.sum(keypoints[:, 0, 0] * (keypoints[:, 0, 1] > 0))
        keypoint_y = np.sum(keypoints[:, 1, 0] * (keypoints[:, 1, 1] > 0))
        num = float(np.sum(keypoints[:, 0, 1]))
        return np.array([keypoint_x / num, keypoint_y / num]), num

if __name__ == '__main__':
    dataset =  MSCOCO(root='/home/xyh/dataset/coco/', ann_file='annotations/person_keypoints_train2017.json')
    dataset.__getitem__(123)