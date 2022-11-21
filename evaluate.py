from cgi import print_directory
import os,sys
import torch
import logging
import time
import argparse
from collections import defaultdict
from collections import OrderedDict
import numpy as np
import json
from tqdm import tqdm
import pickle as pk

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.append("./models/")
from models.HRNet import HRNet
from models.LiteHRNet import LiteHRNet
from models.Hourglass import Hourglass
from models.RegressFlow import RegressFlow, RegressFlow3D
from models.DeepPose import DeepPose
from models.CPM import CPM
from models.DSHourglass import DSHourglass
from models.DSPose import DSPose
from models.DSPosev2 import DSPosev2
from models.DSPosev3 import DSPosev3
from models.DSPosev4 import DSPosev4
from models.DSPosev5 import DSPosev5
from models.DSPosev6 import DSPosev6

from loss import MSELoss, RLELoss, RLELoss3D

from datasets.mscoco import MSCOCO
from datasets.aistpose2d import AISTPose2D
from datasets.h36m import H36m
from datasets.mixed_dataset import MixedDataset
from datasets.maskcoco import MaskCOCO
from datasets.maskcoco2 import MaskCOCO2

from metrics import DataLogger, calc_accuracy, calc_coord_accuracy, evaluate_mAP
from config import update_config, opt

from utils import oks_nms, soft_oks_nms, get_final_preds, get_coord

cfg = update_config(opt.cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pretrained(model, pretrained_path, device):
    if device == torch.device('cpu'):
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    else:
        pretrained_dict = torch.load(pretrained_path)
    model_dict = model.state_dict()
    for k, v in model_dict.items():
        if v.shape == pretrained_dict[k].shape:
            model_dict[k] = pretrained_dict[k]
        else:
            print('size mismatch in %s. Expect %s but get %s' %(k, v.shape, pretrained_dict[k].shape))
    model.load_state_dict(model_dict)
    print("Successfully load pre-trained model from %s " %pretrained_path)
    return model

def main_worker():
    if cfg.MODEL.TYPE == 'Hourglass':
        model = Hourglass(cfg=cfg)
    elif cfg.MODEL.TYPE == 'HRNet':
        model = HRNet(32, cfg.DATA_PRESET.NUM_JOINTS, 0.1)
    elif cfg.MODEL.TYPE == 'LiteHRNet':
        model = LiteHRNet()
    elif cfg.MODEL.TYPE == 'RegressFlow':
        model = RegressFlow(cfg=cfg)
    elif cfg.MODEL.TYPE == 'RegressFlow3d':
        model = RegressFlow3D(cfg=cfg)
    elif cfg.MODEL.TYPE == 'DeepPose':
        model = DeepPose(cfg=cfg)
    elif cfg.MODEL.TYPE == 'CPM':
        model = CPM(cfg.DATA_PRESET.NUM_JOINTS)
    elif cfg.MODEL.TYPE == 'DSHourglass':
        model = DSHourglass(cfg=cfg)
    elif cfg.MODEL.TYPE == 'DSPose':
        model = DSPose(cfg=cfg)
    elif cfg.MODEL.TYPE == 'DSPosev2':
        model = DSPosev2(cfg=cfg)
    elif cfg.MODEL.TYPE == 'DSPosev3':
        model = DSPosev3(cfg=cfg)
    elif cfg.MODEL.TYPE == 'DSPosev4':
        model = DSPosev4(cfg=cfg)
    elif cfg.MODEL.TYPE == 'DSPosev5':
        model = DSPosev5(cfg=cfg)
    elif cfg.MODEL.TYPE == 'DSPosev6':
        model = DSPosev6(cfg=cfg)


    else:
        print("Error : unkown model name.")
        exit(0)
        
    if cfg.MODEL.PRETRAINED:
        model = load_pretrained(model, cfg.MODEL.PRETRAINED, device)
    model = model.cuda()

    if cfg.DATASET.TRAIN.TYPE == 'COCO':
        val_dataset =  MSCOCO(root=cfg.DATASET.VAL.ROOT, ann_file=cfg.DATASET.VAL.ANN, images_dir=cfg.DATASET.VAL.IMG_PREFIX, cfg=cfg, train=False)
    
    elif cfg.DATASET.TRAIN.TYPE == 'AISTPose':
        val_dataset =  AISTPose2D(root=cfg.DATASET.VAL.ROOT, ann_file=cfg.DATASET.VAL.ANN, images_dir=cfg.DATASET.VAL.IMG_PREFIX, cfg=cfg, train=False)

    elif cfg.DATASET.TRAIN.TYPE == 'H36M':
        val_dataset =  H36m(root=cfg.DATASET.VAL.ROOT, ann_file=cfg.DATASET.VAL.ANN, images_dir=cfg.DATASET.VAL.IMG_PREFIX, cfg=cfg, train=False)

    elif cfg.DATASET.TRAIN.TYPE == 'Mixed':
        val_dataset =  H36m(root=cfg.DATASET.VAL.ROOT, ann_file=cfg.DATASET.VAL.ANN, images_dir=cfg.DATASET.VAL.IMG_PREFIX, cfg=cfg, train=False)
    
    elif cfg.DATASET.TRAIN.TYPE == 'MaskCOCO':
        val_dataset =  MaskCOCO(root=cfg.DATASET.VAL.ROOT, ann_file=cfg.DATASET.VAL.ANN, images_dir=cfg.DATASET.VAL.IMG_PREFIX, cfg=cfg, train=False)
    
    elif cfg.DATASET.TRAIN.TYPE == 'MaskCOCO2':
        val_dataset =  MaskCOCO2(root=cfg.DATASET.VAL.ROOT, ann_file=cfg.DATASET.VAL.ANN, images_dir=cfg.DATASET.VAL.IMG_PREFIX, cfg=cfg, train=False)

    output_3d = cfg.DATA_PRESET.get('OUT_3D', False)
    heatmap_to_coord = get_coord(cfg, cfg.DATA_PRESET.HEATMAP_SIZE, output_3d)
    gt_AP = validate_gt(model, cfg, heatmap_to_coord, val_dataset)
    print('##### gt box: {} mAP #####'.format(gt_AP))
    

def validate_gt(m, cfg, heatmap_to_coord, gt_val_dataset, batch_size=20, work_dir="./results"):
    #gt_val_dataset =  MSCOCO(root=cfg.DATASET.VAL.ROOT, ann_file=cfg.DATASET.VAL.ANN, images_dir=cfg.DATASET.VAL.IMG_PREFIX, cfg=cfg, train=False)
    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_dataset, batch_size=batch_size, shuffle=False, num_workers=20, drop_last=False)
    kpt_json = []
    m.eval()

    '''
    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE
    flip_shift = cfg.TEST.get('FLIP_SHIFT', True)
    '''

    gt_val_loader = tqdm(gt_val_loader, dynamic_ncols=True)

    for inps, labels, img_ids, bboxes in gt_val_loader:
        inps = inps.cuda()
        output = m(inps)

        '''
        if opt.flip_test:
            inps_flipped = flip(inps).cuda()
            output_flipped = flip_output(
                m(inps_flipped), gt_val_dataset.joint_pairs,
                width_dim=hm_size[1], shift=flip_shift)
            for k in output.keys():
                if isinstance(output[k], list):
                    continue
                if output[k] is not None:
                    output[k] = (output[k] + output_flipped[k]) / 2
        '''

        for i in range(inps.shape[0]):
            bbox = bboxes[i].tolist()
            pose_coords, pose_scores = heatmap_to_coord(
                output, bbox, idx=i)

            keypoints = np.concatenate((pose_coords[0], pose_scores[0]), axis=1)
            keypoints = keypoints.reshape(-1).tolist()

            data = dict()
            data['bbox'] = bboxes[i].tolist()
            data['image_id'] = int(img_ids[i])
            data['score'] = float(np.mean(pose_scores) + np.max(pose_scores))
            data['category_id'] = 1
            data['keypoints'] = keypoints

            kpt_json.append(data)

    with open(os.path.join(work_dir, 'test_gt_kpt.json'), 'w') as fid:
        json.dump(kpt_json, fid)
    res = evaluate_mAP(os.path.join(work_dir, 'test_gt_kpt.json'), ann_type='keypoints')
    print(res)
    return res['AP']


def main():
    main_worker()
    

if __name__ == '__main__':
    main()
