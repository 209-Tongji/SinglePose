
from datasets.maskcoco2 import MaskCOCO2
from models.MobileNetv3 import MobileNetV3_Small_v1

from easydict import EasyDict 
cfg = EasyDict(
    MODEL=EasyDict(
        TYPE="Hourglass",
        BACKBONE=EasyDict(
            TYPE="ResNet",
            NUM_LAYERS=50
        )
    ),
    DATASET=EasyDict(
        TRAIN=EasyDict(
            TYPE='COCO',
            ROOT='/home/xyh/dataset/coco/',
            IMG_PREFIX='images',
            ANN='annotations/person_keypoints_train2017.json',
            AUG=EasyDict(
                FLIP=True,
                ROT_FACTOR=45,
                SCALE_FACTOR=0.25,
                NUM_JOINTS_HALF_BODY=3,
                PROB_HALF_BODY=0.3
            )
        ),
        VAL=EasyDict(
            TYPE='COCO',
            ROOT='/home/xyh/dataset/coco/',
            IMG_PREFIX='images',
            ANN='annotations/person_keypoints_val2017.json'
        )
    ),
    DATA_PRESET=EasyDict(
        TYPE='simple',
        SIGMA=2,
        NUM_JOINTS=17,
        IMAGE_SIZE=[256,192],
        HEATMAP_SIZE=[64,48],
        NEED_HEATMAP=True,
        NEED_COORD=False,
        NEED_CENTERMAP=False,
    )

)

import torch

if __name__ == '__main__':
    torch.set_printoptions(profile="full")
    train_dataset = MaskCOCO2(root=cfg.DATASET.TRAIN.ROOT, ann_file=cfg.DATASET.TRAIN.ANN, images_dir=cfg.DATASET.TRAIN.IMG_PREFIX, cfg=cfg, train=True)
    img, target, img_id, bbox = train_dataset.__getitem__(1280)

    print(img[3])
    img  = torch.stack([img, img], dim=0)
    print(img.shape)
    model = MobileNetV3_Small_v1()
    output = model(img)
    print(output.shape)