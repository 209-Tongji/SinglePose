import os
import torch
import logging
import time
import argparse
from HRNet import HRNet
from ResPose import PoseResNet
from RegressFlow import RegressFlow

from loss import MSELoss
from loss import RLELoss

from mscoco import MSCOCO
from aistpose2d import AISTPose2D

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
    res_logger = DataLogger()
    model.eval()

    for i, (inputs, labels, _, bboxes) in enumerate(val_loader):
        inputs = inputs.cuda()
        for k, _ in labels.items():
            if k == 'type':
                continue
            labels[k] = labels[k].cuda()
        if cfg.MODEL.TYPE == 'RegressFlow':
            outputs = model(inputs, labels)
            acc = calc_coord_accuracy(outputs, labels, (256,192))
            res_logger.update(acc, inputs.size(0))
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            res_logger.update(loss.item(), inputs.size(0))
    
    if cfg.MODEL.TYPE == 'RegressFlow':
        logger.info("--- *Val  Acc : {loss:.4f}".format(loss=res_logger.avg))
        return - res_logger.avg
    else:
        logger.info("--- *Val  Loss : {loss:.4f}".format(loss=res_logger.avg))
        return res_logger.avg


def main_worker():
    setup_logger(cfg.WORK.LOG)
    if cfg.MODEL.TYPE == 'ResPose':
        model = PoseResNet(50, cfg.DATA_PRESET.NUM_JOINTS, 0.1)
    elif cfg.MODEL.TYPE == 'HRNet':
        model = HRNet(32, cfg.DATA_PRESET.NUM_JOINTS, 0.1)
    elif cfg.MODEL.TYPE == 'RegressFlow':
        model = RegressFlow(cfg=cfg)
    else:
        print("Error : unkown model name.")
        exit(0)
        
    if cfg.MODEL.PRETRAINED:
        model = load_pretrained(model, cfg.MODEL.PRETRAINED, device)
    model = model.cuda()

    if cfg.MODEL.TYPE == 'RegressFlow':
        criterion = RLELoss().cuda()
    else:
        criterion = MSELoss().cuda()

    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    elif cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=0.9, weight_decay=0.0001)
    
    if cfg.DATASET.TRAIN.TYPE == 'COCO':
        train_dataset = MSCOCO(root=cfg.DATASET.TRAIN.ROOT, ann_file=cfg.DATASET.TRAIN.ANN, images_dir=cfg.DATASET.TRAIN.IMG_PREFIX, cfg=cfg, train=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4)

        val_dataset =  MSCOCO(root=cfg.DATASET.VAL.ROOT, ann_file=cfg.DATASET.VAL.ANN, images_dir=cfg.DATASET.VAL.IMG_PREFIX, cfg=cfg, train=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE // 2, shuffle=True, num_workers=4)
    
    if cfg.DATASET.TRAIN.TYPE == 'AISTPose':
        train_dataset = AISTPose2D(root=cfg.DATASET.TRAIN.ROOT, ann_file=cfg.DATASET.TRAIN.ANN, images_dir=cfg.DATASET.TRAIN.IMG_PREFIX, cfg=cfg, train=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4)

        val_dataset =  AISTPose2D(root=cfg.DATASET.VAL.ROOT, ann_file=cfg.DATASET.VAL.ANN, images_dir=cfg.DATASET.VAL.IMG_PREFIX, cfg=cfg, train=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE // 4, shuffle=True, num_workers=4)
    
    is_best = False
    best_loss = 99999.9
    best_save = cfg.WORK.BEST_MODEL
    final_save = cfg.WORK.FINAL_MODEL
    epochs_since_improvement = 0
    total = len(train_loader)

    for i in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        train(train_loader, model, optimizer, criterion, i, cfg=cfg)
        val_loss = validate(val_loader, model, criterion, cfg=cfg)

        is_best = val_loss < best_loss
        best_loss = min(best_loss, val_loss)

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
