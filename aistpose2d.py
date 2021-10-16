import os
import copy
import json
import numpy as np
import torch.utils.data as data
from pycocotools.coco import COCO
from PIL import Image, ImageDraw, ImageFont
from utils import bbox_clip_xyxy, bbox_xywh_to_xyxy, torch_to_im
from transform import SimpleTransform

import cv2
from matplotlib import pyplot as plt

class AISTPose2D(data.Dataset):
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
    def __init__(self, root, ann_file, images_dir, train=True):
        self._root = root
        self._ann_file = os.path.join(root, ann_file)
        self._images_dir = os.path.join(root, images_dir)

        self._train = train
        self._scale_factor = 0.25
        self._rot = 45
        self._sigma = 2

        self.num_joints_half_body = 3
        self.prob_half_body = 0.3
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.lower_body_ids = (11, 12, 13, 14, 15, 16)

        self.transformation = SimpleTransform(
            self, scale_factor = self._scale_factor,
            input_size = (256, 192),
            output_size = (64, 48),
            rot = self._rot, sigma = self._sigma,
            train = self._train, loss_type = 'coord'
        )

        with open(self._ann_file, "r") as f:
            self._annotations = json.load(f)

        

    def __getitem__(self, idx):
        img_id = self._annotations[idx]['image_id']
        img_path = os.path.join(self._images_dir, img_id + '.jpg')

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        joints = np.zeros([17,3,2],dtype=float)
        for _i, _joint in enumerate(self._annotations[idx]['joints']):
            joints[_i,0,0] = _joint[0]
            joints[_i,1,0] = _joint[1]
            joints[_i,0,1] = 1.0
            joints[_i,1,1] = 1.0

        label = {
            'joints_3d': joints,
            'bbox': self._annotations[idx]['bbox'],
            'width': img.shape[1],
            'height': img.shape[0]
        }
        
        print(label)
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(img)
        ax.scatter(label['joints_3d'][:, 0, 0], label['joints_3d'][:, 1, 0])
        fig.savefig("picture_before_trans_{:0>4d}.png".format(idx))
        ax.clear()
        
        target = self.transformation(img, label)

        img = target.pop('image')
        bbox = target.pop('bbox')
        joints_visible = target.pop('target_uv_weight')
        
        label = target.pop('joint_3d')
        image = torch_to_im(img.clone())
        ax.imshow(image)
        ax.scatter(label[:, 0, 0], label[:, 1, 0])
        fig.savefig("picture_after_trans_{:0>4d}.png".format(idx))

        hm_img = target.pop('target_hm')
        hm_img = hm_img.numpy()[0]
        pmin = np.min(hm_img)
        pmax = np.max(hm_img)
        hm_img = ((hm_img - pmin) / (pmax - pmin + 0.000001))*255
        hm_img = hm_img.astype(np.uint8)
        hm_img = cv2.applyColorMap(hm_img, cv2.COLORMAP_JET)
        cv2.imwrite("heatmap.png", hm_img)
        plt.close()

        
        return img, target, img_id, bbox   

    def __len__(self):
        #print(len(self._annotations))
        return len(self._annotations)

    def __validate__(self, idx):
        img_id = self._annotations[idx]['image_id']
        joints = np.zeros([17,3,2],dtype=float)
        for _i, _joint in enumerate(self._annotations[idx]['joints']):
            joints[_i,0,0] = _joint[0]
            joints[_i,1,0] = _joint[1]
            joints[_i,0,1] = 1.0
            joints[_i,1,1] = 1.0

        label = {
            'joints_3d': joints,
            'bbox': self._annotations[idx]['bbox'],
            'width': 1920,
            'height': 1080
        }
        try:
            self.transformation.validate(label)
            self.new_annotations.append(self._annotations[idx])
        except Exception as e:
            print(e)

        if idx % 10000:
            print(idx)

    def clean_data(self):
        self.new_annotations = []
        for idx in range(self.__len__()):
            self.__validate__(idx)
        _annotations_file = os.path.join(self._root, "annotations/test_annotations_clean.json")
        with open(_annotations_file, "w") as f:
            json.dump(self.new_annotations, f)





if __name__ == '__main__':
    dataset = AISTPose2D(root="/dataset/PoseEstimation/AISTPose2D/",ann_file='annotations/train_annotations.json',
        images_dir="images", train=True)
    dataset.__getitem__(11410)
    #dataset.clean_data()
    #for idx in range(dataset.__len__()):
    #    dataset.__validate__(idx)