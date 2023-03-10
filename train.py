import os,sys
import torch
import logging
import time
import argparse


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
from models.HourglassPose import PoseNet

from loss import MSELoss, RLELoss, RLELoss3D, HourglassHeatmapLoss

from datasets.mscoco import MSCOCO
from datasets.aistpose2d import AISTPose2D
from datasets.h36m import H36m
from datasets.mixed_dataset import MixedDataset
from datasets.maskcoco import MaskCOCO
from datasets.maskcoco2 import MaskCOCO2

from metrics import DataLogger, calc_accuracy, calc_coord_accuracy, evaluate_mAP
from config import update_config, opt

cfg = update_config(opt.cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger('')


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

def setup_logger(log_file_name):
    filehandler = logging.FileHandler(log_file_name)
    streamhandler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    filehandler.setFormatter(formatter)
    streamhandler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    logger.info('******************************')
    logger.info(cfg)
    logger.info('******************************')


def train(train_loader, model, optimizer, criterion, epoch, cfg):
    loss_logger = DataLogger()
    #acc_logger = DataLogger()
    logger.info(optimizer.state_dict()['param_groups'][0]['lr'])
    model.train()
    total = len(train_loader)
    for i, (inputs, labels, _, bboxes) in enumerate(train_loader):
        inputs = inputs.cuda()
        for k, _ in labels.items():
            if k == 'type':
                continue
            labels[k] = labels[k].cuda()
        if cfg.MODEL.TYPE == 'RegressFlow':
            outputs = model(inputs, labels)
        elif cfg.MODEL.TYPE == 'RegressFlow3d':
            outputs = model(inputs, labels)
        elif cfg.MODEL.TYPE == 'CPM':
            outputs = model(inputs, labels['center_map'])
        else:
            outputs = model(inputs)
        loss = criterion(outputs, labels)
        #acc = calc_coord_accuracy(outputs, labels, (256,192))
        loss_logger.update(loss.item(), inputs.size(0))
        #acc_logger.update(acc, inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            logger.info( 'EPOCH {epoch} [{num:04d}:{total:04d}] loss: {loss:.8f} '.format(epoch=epoch, num=i, total=total, loss=loss_logger.avg))

    return loss_logger.avg

def validate(val_loader, model, criterion, cfg):
    acc_logger = DataLogger()
    model.eval()

    for i, (inputs, labels, _, bboxes) in enumerate(val_loader):
        inputs = inputs.cuda()
        for k, _ in labels.items():
            if k == 'type':
                continue
            labels[k] = labels[k].cuda()
        if cfg.MODEL.TYPE == 'RegressFlow':
            outputs = model(inputs, labels)
            acc = calc_coord_accuracy(outputs, labels, (cfg.DATA_PRESET.IMAGE_SIZE[0],cfg.DATA_PRESET.IMAGE_SIZE[1]))
        elif cfg.MODEL.TYPE == 'RegressFlow3d':
            outputs = model(inputs, labels)
            acc = calc_coord_accuracy(outputs, labels, (256,256,64), output_3d=True)
        elif cfg.MODEL.TYPE == 'DeepPose':
            outputs = model(inputs)   
            acc = calc_coord_accuracy(outputs, labels, (cfg.DATA_PRESET.IMAGE_SIZE[0],cfg.DATA_PRESET.IMAGE_SIZE[1]))
        elif cfg.MODEL.TYPE == 'CPM':
            outputs = model(inputs, labels['center_map'])
            acc = calc_accuracy(outputs[:,5,:,:], labels)
        elif cfg.MODEL.TYPE == 'HourglassPose':
            outputs = model(inputs)
            outputs = outputs[:,-1,:,:,:]   
            acc = calc_accuracy(outputs, labels)
        else:
            outputs = model(inputs)   
            acc = calc_accuracy(outputs, labels)
        acc_logger.update(acc, inputs.size(0))
    

    logger.info("--- *Val  Acc : {loss:.4f}".format(loss=acc_logger.avg))
    return acc_logger.avg


def main_worker():
    setup_logger(cfg.WORK.LOG)
    if cfg.MODEL.TYPE == 'HourglassPose':
        model = PoseNet(8, 256, 17)
    elif cfg.MODEL.TYPE == 'Hourglass':
        model = Hourglass(cfg=cfg)
    elif cfg.MODEL.TYPE == 'HRNet':
        model = HRNet(cfg.MODEL.W, cfg.DATA_PRESET.NUM_JOINTS, 0.1)
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

    if cfg.MODEL.TYPE == 'HourglassPose':
        criterion = HourglassHeatmapLoss().cuda()
    elif cfg.MODEL.TYPE == 'RegressFlow':
        criterion = RLELoss().cuda()
    elif cfg.MODEL.TYPE == 'RegressFlow3d':
        criterion = RLELoss3D().cuda()
    else:
        criterion = MSELoss(cfg.LOSS.HEATMAP2COORD).cuda()

    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': cfg.TRAIN.LR}], lr=cfg.TRAIN.LR)
    elif cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters(), 'initial_lr': cfg.TRAIN.LR}], lr=cfg.TRAIN.LR, momentum=0.9, weight_decay=0.0001)
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_FACTOR, last_epoch=cfg.TRAIN.BEGIN_EPOCH)
    
    if cfg.DATASET.TRAIN.TYPE == 'COCO':
        train_dataset = MSCOCO(root=cfg.DATASET.TRAIN.ROOT, ann_file=cfg.DATASET.TRAIN.ANN, images_dir=cfg.DATASET.TRAIN.IMG_PREFIX, cfg=cfg, train=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4)

        val_dataset =  MSCOCO(root=cfg.DATASET.VAL.ROOT, ann_file=cfg.DATASET.VAL.ANN, images_dir=cfg.DATASET.VAL.IMG_PREFIX, cfg=cfg, train=False)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE // 2, shuffle=True, num_workers=4)
    
    elif cfg.DATASET.TRAIN.TYPE == 'AISTPose':
        train_dataset = AISTPose2D(root=cfg.DATASET.TRAIN.ROOT, ann_file=cfg.DATASET.TRAIN.ANN, images_dir=cfg.DATASET.TRAIN.IMG_PREFIX, cfg=cfg, train=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4)

        val_dataset =  AISTPose2D(root=cfg.DATASET.VAL.ROOT, ann_file=cfg.DATASET.VAL.ANN, images_dir=cfg.DATASET.VAL.IMG_PREFIX, cfg=cfg, train=False)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE // 4, shuffle=True, num_workers=4)

    elif cfg.DATASET.TRAIN.TYPE == 'H36M':
        train_dataset = H36m(root=cfg.DATASET.TRAIN.ROOT, ann_file=cfg.DATASET.TRAIN.ANN, images_dir=cfg.DATASET.TRAIN.IMG_PREFIX, cfg=cfg, train=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4)

        val_dataset =  H36m(root=cfg.DATASET.VAL.ROOT, ann_file=cfg.DATASET.VAL.ANN, images_dir=cfg.DATASET.VAL.IMG_PREFIX, cfg=cfg, train=False)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE // 4, shuffle=True, num_workers=4)

    elif cfg.DATASET.TRAIN.TYPE == 'Mixed':
        train_dataset = MixedDataset(cfg=cfg)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4)

        val_dataset =  H36m(root=cfg.DATASET.VAL.ROOT, ann_file=cfg.DATASET.VAL.ANN, images_dir=cfg.DATASET.VAL.IMG_PREFIX, cfg=cfg, train=False)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE // 4, shuffle=True, num_workers=4)
    
    elif cfg.DATASET.TRAIN.TYPE == 'MaskCOCO':
        train_dataset = MaskCOCO(root=cfg.DATASET.TRAIN.ROOT, ann_file=cfg.DATASET.TRAIN.ANN, images_dir=cfg.DATASET.TRAIN.IMG_PREFIX, cfg=cfg, train=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4)

        val_dataset =  MaskCOCO(root=cfg.DATASET.VAL.ROOT, ann_file=cfg.DATASET.VAL.ANN, images_dir=cfg.DATASET.VAL.IMG_PREFIX, cfg=cfg, train=False)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE // 2, shuffle=True, num_workers=4)
    
    elif cfg.DATASET.TRAIN.TYPE == 'MaskCOCO2':
        train_dataset = MaskCOCO2(root=cfg.DATASET.TRAIN.ROOT, ann_file=cfg.DATASET.TRAIN.ANN, images_dir=cfg.DATASET.TRAIN.IMG_PREFIX, cfg=cfg, train=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4)

        val_dataset =  MaskCOCO2(root=cfg.DATASET.VAL.ROOT, ann_file=cfg.DATASET.VAL.ANN, images_dir=cfg.DATASET.VAL.IMG_PREFIX, cfg=cfg, train=False)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE // 2, shuffle=True, num_workers=4)
    
    is_best = False
    best_acc = -99999.9
    best_save = cfg.WORK.BEST_MODEL
    final_save = cfg.WORK.FINAL_MODEL
    epochs_since_improvement = 0
    total = len(train_loader)

    for i in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        train(train_loader, model, optimizer, criterion, i, cfg=cfg)
        val_acc = validate(val_loader, model, criterion, cfg=cfg)

        lr_scheduler.step()

        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)

        if not is_best:
            epochs_since_improvement += 1
            print('\nEpochs since last improvment: %d\n' %(epochs_since_improvement))
            logger.debug('\nEpochs since last improvment: %d\n' %(epochs_since_improvement))
            pass
        else:
            epochs_since_improvement = 0
            torch.save(model.state_dict(), best_save)
        
    torch.save(model.state_dict(), final_save)
    logger.info("----* End of training. *----")

def main():
    main_worker()
    

if __name__ == '__main__':
    main()
