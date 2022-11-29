import torch
import torch.nn as nn
import math

class MSELoss(nn.Module):
    ''' MSE Loss
    '''
    def __init__(self, heatmap2coord):
        super(MSELoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.heatmap2coord = heatmap2coord

    def forward(self, output, labels):
        pred = output
        if self.heatmap2coord == 'heatmap':
            gt_hm = labels['target_hm']
            gt_hm_weight = labels['target_hm_weight']
            loss = 0.5 * self.criterion(pred.mul(gt_hm_weight), gt_hm.mul(gt_hm_weight))
        elif self.heatmap2coord == 'coord':
            gt_uv = labels['target_uv'].reshape(pred.shape)
            gt_uv_weight = labels['target_uv_weight'].reshape(pred.shape)
            loss = 0.5 * self.criterion(pred.mul(gt_uv_weight), gt_uv.mul(gt_uv_weight))
        elif self.heatmap2coord == 'cpm':
            gt_hm = torch.stack([labels['target_hm']]*6,dim=1)
            gt_hm_weight = torch.stack([labels['target_hm_weight']]*6,dim=1)
            loss = 0.5 * self.criterion(pred.mul(gt_hm_weight), gt_hm.mul(gt_hm_weight))

        return loss

class HourglassHeatmapLoss(nn.Module):
    """
    loss for detection heatmap
    """
    def __init__(self):
        super(HourglassHeatmapLoss, self).__init__()
        self.nstack = 8

    def heatmapLoss(self, pred, labels):
        gt_hm = labels['target_hm']
        gt_hm_weight = labels['target_hm_weight']
        pred = pred.mul(gt_hm_weight)
        gt = gt_hm.mul(gt_hm_weight)
        l = ((pred - gt)**2)
        l = l.mean(dim=3).mean(dim=2).mean(dim=1)
        return l ## l of dim bsize

    def forward(self, combined_hm_preds, labels):
        combined_loss = []
        for i in range(self.nstack):
            combined_loss.append(self.heatmapLoss(combined_hm_preds[:,i], labels))
        combined_loss = torch.stack(combined_loss, dim=1)
        return torch.mean(combined_loss)
        #return combined_loss

class RLELoss(nn.Module):
    ''' RLE Regression Loss
    '''

    def __init__(self, OUTPUT_3D=False, size_average=True):
        super(RLELoss, self).__init__()
        self.size_average = size_average
        self.amp = 1 / math.sqrt(2 * math.pi)

    def logQ(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)

    def forward(self, output, labels):
        nf_loss = output.nf_loss
        pred_jts = output.pred_jts
        sigma = output.sigma
        gt_uv = labels['target_uv'].reshape(pred_jts.shape)
        gt_uv_weight = labels['target_uv_weight'].reshape(pred_jts.shape)

        nf_loss = nf_loss * gt_uv_weight[:, :, :1]

        residual = True
        if residual:
            Q_logprob = self.logQ(gt_uv, pred_jts, sigma) * gt_uv_weight
            loss = nf_loss + Q_logprob

        if self.size_average and gt_uv_weight.sum() > 0:
            return loss.sum() / len(loss)
        else:
            return loss.sum()

class RLELoss3D(nn.Module):
    ''' RLE Regression Loss 3D
    '''

    def __init__(self, OUTPUT_3D=False, size_average=True):
        super(RLELoss3D, self).__init__()
        self.size_average = size_average
        self.amp = 1 / math.sqrt(2 * math.pi)

    def logQ(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)

    def forward(self, output, labels):
        nf_loss = output.nf_loss
        pred_jts = output.pred_jts
        sigma = output.sigma
        gt_uv = labels['target_uvd'].reshape(pred_jts.shape)
        gt_uv_weight = labels['target_uvd_weight'].reshape(pred_jts.shape)
        nf_loss = nf_loss * gt_uv_weight

        residual = True
        if residual:
            Q_logprob = self.logQ(gt_uv, pred_jts, sigma) * gt_uv_weight
            loss = nf_loss + Q_logprob

        if self.size_average and gt_uv_weight.sum() > 0:
            return loss.sum() / len(loss)
        else:
            return loss.sum()