import os
import torch
import logging
import time

from ResPose import PoseResNet
from loss import MSELoss
from mscoco import MSCOCO
from aistpose2d import AISTPose2D

from metrics import DataLogger, calc_accuracy, calc_coord_accuracy, evaluate_mAP
from config import update_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger('')
cfg = update_config('respose_coco.yaml')


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
    filehandler = logging.FileHandler(
        './log/{}'.format(log_file_name))
    streamhandler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    filehandler.setFormatter(formatter)
    streamhandler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

def train(train_loader, model, optimizer, criterion, epoch):
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
        outputs = model(inputs, labels)
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

def validate(val_loader, model, criterion):
    loss_logger = DataLogger()
    model.eval()

    for i, (inputs, labels, _, bboxes) in enumerate(val_loader):
        inputs = inputs.cuda()
        for k, _ in labels.items():
            if k == 'type':
                continue
            labels[k] = labels[k].cuda()
        outputs = model(inputs, labels)
        loss = criterion(outputs, labels)
        loss_logger.update(loss.item(), inputs.size(0))
        #acc = calc_coord_accuracy(outputs, labels, (256,192))
        #acc_logger.update(acc, inputs.size(0))
    
        logger.info("--- *Val  Loss : {loss:.4f}".format(loss=loss_logger.avg))

    return loss_logger.avg


def main_worker():
    setup_logger("ResPose_coco.log")
    model = PoseResNet(50, cfg.DATA_PRESET.NUM_JOINTS, 0.1)
    if cfg.MODEL.PRETRAINED:
        model = load_pretrained(model, cfg.MODEL.PRETRAINED, device)
    model = model.cuda()

    criterion = MSELoss().cuda()

    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    elif cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=0.9, weight_decay=0.0001)
    '''
    train_dataset = MSCOCO(root='/home/xyh/dataset/coco/', ann_file='annotations/person_keypoints_train2017.json')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=opt.nThreads)

    val_dataset = MSCOCO(root='/home/xyh/dataset/coco/', ann_file='annotations/person_keypoints_val2017.json')
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=20, shuffle=True, num_workers=opt.nThreads)
    '''

    train_dataset = AISTPose2D(root="/home/xyh/dataset/AISTPose2D/", ann_file='annotations/train_annotations.json', images_dir="images", train=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=8)

    val_dataset =  AISTPose2D(root="/home/xyh/dataset/AISTPose2D/", ann_file='annotations/val_annotations.json', images_dir="images", train=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=20, shuffle=True, num_workers=8)

    is_best = False
    best_loss = 99999.9
    best_save = "./models/respose_aistpose2d_best.pth"
    final_save = "./models/respose_aistpose2d_final.pth"
    epochs_since_improvement = 0
    total = len(train_loader)

    for i in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        train(train_loader, model, optimizer, criterion, i)
        val_loss = validate(val_loader, model, criterion)

        is_best = val_loss < best_acc
        best_acc = min(best_loss, val_loss)

        if not is_best:
            epochs_since_improvement += 1
            print('\nEpochs since last improvment: %d\n' %(epochs_since_improvement))
            logger.debug('\nEpochs since last improvment: %d\n' %(epochs_since_improvement))
            pass
        else:
            epochs_since_improvement = 0
            torch.save(model.module.state_dict(), best_save)
        
        torch.distributed.barrier() 
        
    torch.save(model.module.state_dict(), final_save)
    logger.info("----* End of training. *----")

def main():
    main_worker()
    

if __name__ == '__main__':
    main()
