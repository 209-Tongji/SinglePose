import copy
import os, sys
import json 
import numpy as np
import torch.utils.data as data
import cv2
import scipy.misc
sys.path.append("..") 
from utils import (bbox_clip_xyxy, bbox_xywh_to_xyxy, draw_origin_joints, 
                cam2pixel, draw_origin_joints_index)

from transform import SimpleTransform3D


class MirrorDataset(data.Dataset):
    #               [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17]
    HUMAN36_INDEX = [8,  9, 10, 11, 12, 13, 14,  0,  0,  0,  0,  5,  6,  7,  2,  3,  4,  1]
    VIS_MASK      = [1,  1,  1,  1,  1,  1,  1,  0,  0,  1,  0,  1,  1,  1,  1,  1,  1,  1]
    bbox_3d_shape = (2000, 2000, 2000)
    joints_name = ('Pelvis',  # 0
                   'R_Hip', 'R_Knee', 'R_Ankle',  # 3
                   'L_Hip', 'L_Knee', 'L_Ankle',  # 6
                   'Torso', 'Neck',  # 8
                   'Nose', 'Head',  # 10
                   'L_Shoulder', 'L_Elbow', 'L_Wrist',  # 13
                   'R_Shoulder', 'R_Elbow', 'R_Wrist',  # 16
                   'Thorax')  # 17
    skeleton = ((1, 0), (2, 1), (3, 2),  # 2
                (4, 0), (5, 4), (6, 5),  # 5
                (7, 0), (8, 7),  # 7
                (9, 8), (10, 9),  # 9
                (11, 7), (12, 11), (13, 12),  # 12
                (14, 7), (15, 14), (16, 15),  # 15
                (17, 7))  # 16
    def __init__(self,
                root,
                cfg,
                train=True,
                skip_empty=True,
                dpg=False,
                **kwargs):
        self._root = root
        self._preset_cfg = cfg.DATA_PRESET
        self.protocol = self._preset_cfg.PROTOCOL
        self._train = train
        self._dpg = False
        self._loss_type = self._preset_cfg.HEATMAP2COORD

        self._ann_file_dir = os.path.join(self._root, "annots")
        self._img_file_dir = os.path.join(self._root, "images")
        self._k3d_file_dir = os.path.join(self._root, "keypoints3d")
        self._smpl_file_dir = os.path.join(self._root, "smpl")

        self._scale_factor = self._preset_cfg.SCALE_FACTOR
        self._color_factor = self._preset_cfg.COLOR_FACTOR
        self._rot = self._preset_cfg.ROT_FACTOR
        self._input_size = self._preset_cfg.IMAGE_SIZE
        self._output_size = self._preset_cfg.HEATMAP_SIZE
        self._occlusion = self._preset_cfg.OCCLUSION
        self._sigma = self._preset_cfg.SIGMA
        self._check_centers = False
        self.num_joints = self._preset_cfg.NUM_JOINTS

        self.num_joints_half_body = self._preset_cfg.NUM_JOINTS_HALF_BODY
        self.prob_half_body = self._preset_cfg.PROB_HALF_BODY

        self._loss_type = self._preset_cfg.HEATMAP2COORD

        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)

        self.root_idx = self.joints_name.index('Pelvis')
        self.lshoulder_idx = self.joints_name.index('L_Shoulder')
        self.rshoulder_idx = self.joints_name.index('R_Shoulder')

        self._items = self._load_jsons()

        if self._preset_cfg.TYPE == 'simple_3d':
            self.transformation = SimpleTransform3D(
                self, scale_factor=self._scale_factor,
                color_factor=self._color_factor,
                occlusion=self._occlusion,
                input_size=self._input_size,
                output_size=self._output_size,
                bbox_3d_shape=self.bbox_3d_shape,
                rot=self._rot, sigma=self._sigma,
                train=self._train, add_dpg=self._dpg,
                loss_type=self._loss_type, scale_mult=1)

        pass                                                                                           

    def _load_jsons(self):
        items = []

        img_dirs = os.listdir(self._img_file_dir)
        print(len(img_dirs))

        count = 0
        for _dir in img_dirs:
            img_dir = os.path.join(self._img_file_dir, _dir)
            ann_dir = os.path.join(self._ann_file_dir, _dir)
            k3d_dir = os.path.join(self._k3d_file_dir, _dir)
            smpl_dir = os.path.join(self._smpl_file_dir, _dir)
        
            img_list = os.listdir(img_dir)

            for _img_file in img_list:
                _id = _img_file.split('.')[0]
                _json = _id + '.json'
                _img_path = os.path.join(img_dir, _img_file)
                _ann_path = os.path.join(ann_dir, _json)
                _k3d_path = os.path.join(k3d_dir, _json)
                _smpl_path = os.path.join(smpl_dir, _json)

                with open(_ann_path, 'r') as f:
                    _ann_data = json.load(f)

                with open(_k3d_path, 'r') as f:
                    _k3d_data = json.load(f)

                with open(_smpl_path, 'r') as f:
                    _smpl_data = json.load(f)
                
                for i in range(len(_ann_data['annots'])):
                    #print(_img_path)

                    #keypoints2d = np.array(_ann_data['annots'][i]['keypoints'])
                    keypoints3d = np.array(_k3d_data[i]['keypoints3d'])
                    cameraK = np.array(_ann_data['K'])
                    f = [cameraK[0,0],cameraK[1,1]]
                    c = [cameraK[0,2],cameraK[1,2]]
                    joint_cam = keypoints3d[self.HUMAN36_INDEX]
                    joint_img = cam2pixel(joint_cam, f, c)
                    joint_img[:, 2] = joint_img[:, 2] - joint_cam[self.root_idx, 2]
                    joint_vis = np.ones((self.num_joints, 3))  #(18,3)
                    for j in range(joint_vis.shape[0]):
                        if self.VIS_MASK[j] == 0:
                            joint_vis[j, :] = 0

                    width = _ann_data['width']
                    height = _ann_data['height']

                    item = {
                        "img_id": count,
                        "img_path": _img_path,
                        "width": width,
                        "height": height,
                        "bbox": _ann_data['annots'][i]['bbox'][:4],
                        "joint_img": joint_img,
                        "joint_vis": joint_vis,
                        "joint_cam": joint_cam,
                        #"keypoints2d": np.array(_ann_data['annots'][i]['keypoints']),
                        #"keypoints3d": np.array(_k3d_data[i]['keypoints3d']),
                        "smpl": _smpl_data[i],
                        "cameraK": cameraK,
                        #"vanish_line": _ann_data['vanish_line'],
                        #"vanish_point": _ann_data['vanish_point']
                    }

                    count += 1
                
                    items.append(item)
        print(len(items))
        return items
    
    def __getitem__(self, idx):
        # get image id
        item = copy.deepcopy(self._items[idx])
        img_path = item["img_path"]
        img_id = item["img_id"]

        img = scipy.misc.imread(img_path, mode='RGB')
        # img = load_image(img_path)
        # img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        # transform ground truth into training label and apply data augmentation
        target = self.transformation(img, item)
        img = target.pop('image')
        bbox = target.pop('bbox')
        return img, target, img_id, bbox

    def __len__(self):
        return len(self._items)
            
    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))

    @property
    def bone_pairs(self):
        """Bone pairs which defines the pairs of bone to be swapped
        when the image is flipped horizontally."""
        return ((0, 3), (1, 4), (2, 5), (10, 13), (11, 14), (12, 15))

            




if __name__ == '__main__':
    dataset = MirrorDataset("/home/xyh/dataset/mirrored-human-base")
    #dataset._load_jsons()
    item = dataset._items[501]
    print(item["joint_vis"])
    img_path = item['img_path']
    print(img_path)
    img = scipy.misc.imread(img_path, mode='RGB')
    #keypoints2d = item['keypoints2d']
    keypoints3d = item['joint_cam']
    K = item['cameraK']
    f = [K[0,0],K[1,1]]
    c = [K[0,2],K[1,2]]

    coord = cam2pixel(keypoints3d, f, c)
    
    #print(coord)

    #draw_origin_joints(img, keypoints2d, output="keypoint2d.png")
    #draw_origin_joints(img, coord, output="cam2pixel.png")

    print(coord.shape)

    for i in range(coord.shape[0]):
        save_img = "k3d_%d.png" % i
        draw_origin_joints_index(img, coord, index=i, output=save_img)