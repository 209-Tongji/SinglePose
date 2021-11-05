import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributions as distributions
from easydict import EasyDict
import torchvision

from .MobileNet import MobileNetV2


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=0.1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)

        self.bn2 = norm_layer(planes, momentum=0.1)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4, momentum=0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    """ ResNet """

    def __init__(self, architecture, norm_layer=nn.BatchNorm2d):
        super(ResNet, self).__init__()
        self._norm_layer = norm_layer
        assert architecture in ["resnet18", "resnet34", "resnet50", "resnet101", 'resnet152']
        layers = {
            'resnet18': [2, 2, 2, 2],
            'resnet34': [3, 4, 6, 3],
            'resnet50': [3, 4, 6, 3],
            'resnet101': [3, 4, 23, 3],
            'resnet152': [3, 8, 36, 3],
        }
        self.inplanes = 64
        if architecture == "resnet18" or architecture == 'resnet34':
            self.block = BasicBlock
        else:
            self.block = Bottleneck
        self.layers = layers[architecture]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64, eps=1e-5, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(
            self.block, 64, self.layers[0])
        self.layer2 = self.make_layer(
            self.block, 128, self.layers[1], stride=2)
        self.layer3 = self.make_layer(
            self.block, 256, self.layers[2], stride=2)

        self.layer4 = self.make_layer(
            self.block, 512, self.layers[3], stride=2)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # 64 * h/4 * w/4
        x = self.layer1(x)  # 256 * h/4 * w/4
        x = self.layer2(x)  # 512 * h/8 * w/8
        x = self.layer3(x)  # 1024 * h/16 * w/16
        x = self.layer4(x)  # 2048 * h/32 * w/32
        return x

    def forward_feat(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # 64 * h/4 * w/4
        x1 = self.layer1(x)  # 256 * h/4 * w/4
        x2 = self.layer2(x1)  # 512 * h/8 * w/8
        x3 = self.layer3(x2)  # 1024 * h/16 * w/16
        x4 = self.layer4(x3)  # 2048 * h/32 * w/32
        return x1, x2, x3, x4

    def stages(self):
        return [self.layer1, self.layer2, self.layer3, self.layer4]

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self._norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            norm_layer=self._norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                norm_layer=self._norm_layer))

        return nn.Sequential(*layers)

class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()

        self.prior = prior
        self.register_buffer('mask', mask)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))])

    def _init(self):
        for m in self.t:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)
        for m in self.s:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)

    def forward_p(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def backward_p(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self, x):
        DEVICE = x.device
        if self.prior.loc.device != DEVICE:
            self.prior.loc = self.prior.loc.to(DEVICE)
            self.prior.scale_tril = self.prior.scale_tril.to(DEVICE)
            self.prior._unbroadcasted_scale_tril = self.prior._unbroadcasted_scale_tril.to(DEVICE)
            self.prior.covariance_matrix = self.prior.covariance_matrix.to(DEVICE)
            self.prior.precision_matrix = self.prior.precision_matrix.to(DEVICE)

        z, logp = self.backward_p(x)
        return self.prior.log_prob(z) + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        x = self.forward_p(z)
        return x

    def forward(self, x):
        return self.log_prob(x)


def nets():
    return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2), nn.Tanh())


def nett():
    return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2))


class Linear(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, norm=True):
        super(Linear, self).__init__()
        self.bias = bias
        self.norm = norm
        self.linear = nn.Linear(in_channel, out_channel, bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)

    def forward(self, x):
        y = x.matmul(self.linear.weight.t())

        if self.norm:
            x_norm = torch.norm(x, dim=1, keepdim=True)
            y = y / x_norm

        if self.bias:
            y = y + self.linear.bias
        return y


class RegressFlow(nn.Module):
    def __init__(self, cfg, norm_layer=nn.BatchNorm2d):
        super(RegressFlow, self).__init__()
        print(cfg)
        self._preset_cfg = cfg.DATA_PRESET
        self.fc_dim = cfg.MODEL.NUM_FC_FILTERS
        self._norm_layer = norm_layer
        self.num_joints = self._preset_cfg.NUM_JOINTS
        self.height_dim = self._preset_cfg.IMAGE_SIZE[0]
        self.width_dim = self._preset_cfg.IMAGE_SIZE[1]

        if cfg.MODEL.BACKBONE.TYPE == 'ResNet':

            self.preact = ResNet(f"resnet{cfg.MODEL.BACKBONE.NUM_LAYERS}")
            # Imagenet pretrain model
            import torchvision.models as tm  # noqa: F401,F403
            assert cfg.MODEL.BACKBONE.NUM_LAYERS in [18, 34, 50, 101, 152]
            x = eval(f"tm.resnet{cfg.MODEL.BACKBONE.NUM_LAYERS}(pretrained=True)")

            self.feature_channel = {
                18: 512,
                34: 512,
                50: 2048,
                101: 2048,
                152: 2048
            }[cfg.MODEL.BACKBONE.NUM_LAYERS]
        
        
        elif cfg.MODEL.BACKBONE.TYPE == 'MobileNet':
            self.preact = MobileNetV2()
            import torchvision.models as tm  # noqa: F401,F403
            x = eval(f"tm.mobilenet_v2(pretrained=True)")

            self.feature_channel = 1280

        self.hidden_list = cfg.MODEL.HIDDEN_LIST
        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                    if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fcs, out_channel = self._make_fc_layer()

        self.fc_coord = Linear(out_channel, self.num_joints * 2)
        self.fc_sigma = Linear(out_channel, self.num_joints * 2, norm=False)

        self.fc_layers = [self.fc_coord, self.fc_sigma]

        prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
        masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))

        self.flow = RealNVP(nets, nett, masks, prior)

    def _make_fc_layer(self):
        fc_layers = []
        num_deconv = len(self.fc_dim)
        input_channel = self.feature_channel
        for i in range(num_deconv):
            if self.fc_dim[i] > 0:
                fc = nn.Linear(input_channel, self.fc_dim[i])
                bn = nn.BatchNorm1d(self.fc_dim[i])
                fc_layers.append(fc)
                fc_layers.append(bn)
                fc_layers.append(nn.ReLU(inplace=True))
                input_channel = self.fc_dim[i]
            else:
                fc_layers.append(nn.Identity())

        return nn.Sequential(*fc_layers), input_channel

    def _initialize(self):
        for m in self.fcs:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def forward(self, x, labels=None):
        BATCH_SIZE = x.shape[0]

        feat = self.preact(x)

        _, _, f_h, f_w = feat.shape
        feat = self.avg_pool(feat).reshape(BATCH_SIZE, -1)

        out_coord = self.fc_coord(feat).reshape(BATCH_SIZE, self.num_joints, 2)
        assert out_coord.shape[2] == 2

        out_sigma = self.fc_sigma(feat).reshape(BATCH_SIZE, self.num_joints, -1)

        # (B, N, 2)
        pred_jts = out_coord.reshape(BATCH_SIZE, self.num_joints, 2)
        sigma = out_sigma.reshape(BATCH_SIZE, self.num_joints, -1).sigmoid()
        scores = 1 - sigma

        scores = torch.mean(scores, dim=2, keepdim=True)

        if self.training and labels is not None:
            gt_uv = labels['target_uv'].reshape(pred_jts.shape)
            bar_mu = (pred_jts - gt_uv) / sigma
            # (B, K, 2)
            log_phi = self.flow.log_prob(bar_mu.reshape(-1, 2)).reshape(BATCH_SIZE, self.num_joints, 1)

            nf_loss = torch.log(sigma) - log_phi
        else:
            nf_loss = None

        output = EasyDict(
            pred_jts=pred_jts,
            sigma=sigma,
            maxvals=scores.float(),
            nf_loss=nf_loss
        )
        return output

    
def nets3d():
    return nn.Sequential(nn.Linear(3, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 3), nn.Tanh())
    # return nn.Sequential(nn.Linear(3, 256), nn.LeakyReLU(), nn.Linear(256, 2), nn.Tanh())


def nett3d():
    return nn.Sequential(nn.Linear(3, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 3))
    # return nn.Sequential(nn.Linear(3, 256), nn.LeakyReLU(), nn.Linear(256, 2))

class RegressFlow3D(nn.Module):
    def __init__(self, cfg, norm_layer=nn.BatchNorm2d):
        super(RegressFlow3D, self).__init__()
        print(cfg)
        self._preset_cfg = cfg.DATA_PRESET
        self.fc_dim = cfg.MODEL.NUM_FC_FILTERS
        self._norm_layer = norm_layer
        self.num_joints = self._preset_cfg.NUM_JOINTS
        self.height_dim = self._preset_cfg.IMAGE_SIZE[0]
        self.width_dim = self._preset_cfg.IMAGE_SIZE[1]

        self.root_idx = 0

        if cfg.MODEL.BACKBONE.TYPE == 'ResNet':

            self.preact = ResNet(f"resnet{cfg.MODEL.BACKBONE.NUM_LAYERS}")
            # Imagenet pretrain model
            import torchvision.models as tm  # noqa: F401,F403
            assert cfg.MODEL.BACKBONE.NUM_LAYERS in [18, 34, 50, 101, 152]
            x = eval(f"tm.resnet{cfg.MODEL.BACKBONE.NUM_LAYERS}(pretrained=True)")

            self.feature_channel = {
                18: 512,
                34: 512,
                50: 2048,
                101: 2048,
                152: 2048
            }[cfg.MODEL.BACKBONE.NUM_LAYERS]
        
        
        elif cfg.MODEL.BACKBONE.TYPE == 'MobileNet':
            self.preact = MobileNetV2()
            import torchvision.models as tm  # noqa: F401,F403
            x = eval(f"tm.mobilenet_v2(pretrained=True)")

            self.feature_channel = 1280

        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                 if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fcs, out_channel = self._make_fc_layer()

        self.fc_coord = Linear(out_channel, self.num_joints * 3)
        self.fc_sigma = nn.Linear(out_channel, self.num_joints * 3)

        self.fc_layers = [self.fc_coord, self.fc_sigma]

        self.share_flow = True

        prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
        masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))

        prior3d = distributions.MultivariateNormal(torch.zeros(3), torch.eye(3))
        masks3d = torch.from_numpy(np.array([[0, 0, 1], [1, 1, 0]] * 3).astype(np.float32))
        # masks3d = torch.from_numpy(np.array([[1, 1, 0], [0, 0, 1]] * 3).astype(np.float32))

        self.flow2d = RealNVP(nets, nett, masks, prior)
        self.flow3d = RealNVP(nets3d, nett3d, masks3d, prior3d)

    def _make_fc_layer(self):
        fc_layers = []
        num_deconv = len(self.fc_dim)
        input_channel = self.feature_channel
        for i in range(num_deconv):
            if self.fc_dim[i] > 0:
                fc = nn.Linear(input_channel, self.fc_dim[i])
                bn = nn.BatchNorm1d(self.fc_dim[i])
                fc_layers.append(fc)
                fc_layers.append(bn)
                fc_layers.append(nn.ReLU(inplace=True))
                input_channel = self.fc_dim[i]
            else:
                fc_layers.append(nn.Identity())

        return nn.Sequential(*fc_layers), input_channel

    def _initialize(self):
        for m in self.fcs:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def forward(self, x, labels=None):
        BATCH_SIZE = x.shape[0]

        feat = self.preact(x)

        # Positional Pooling
        _, _, f_h, f_w = feat.shape
        feat = self.avg_pool(feat).reshape(BATCH_SIZE, -1)

        out_coord = self.fc_coord(feat).reshape(BATCH_SIZE, self.num_joints, 3)
        assert out_coord.shape[2] == 3

        out_sigma = self.fc_sigma(feat).reshape(BATCH_SIZE, self.num_joints, -1)

        # (B, N, 3)
        pred_jts = out_coord.reshape(BATCH_SIZE, self.num_joints, 3)
        if not self.training:
            pred_jts[:, :, 2] = pred_jts[:, :, 2] - pred_jts[:, self.root_idx:self.root_idx + 1, 2]

        sigma = out_sigma.reshape(BATCH_SIZE, self.num_joints, -1).sigmoid() + 1e-9
        scores = 1 - sigma

        scores = torch.mean(scores, dim=2, keepdim=True)

        if labels is not None:
            gt_uvd = labels['target_uvd'].reshape(pred_jts.shape)
            gt_uvd_weight = labels['target_uvd_weight'].reshape(pred_jts.shape)
            gt_3d_mask = gt_uvd_weight[:, :, 2].reshape(-1)

            assert pred_jts.shape == sigma.shape, (pred_jts.shape, sigma.shape)
            bar_mu = (pred_jts - gt_uvd) / sigma
            bar_mu = bar_mu.reshape(-1, 3)
            bar_mu_3d = bar_mu[gt_3d_mask > 0]
            bar_mu_2d = bar_mu[gt_3d_mask < 1][:, :2]
            # (B, K, 3)
            log_phi_3d = self.flow3d.log_prob(bar_mu_3d)
            log_phi_2d = self.flow2d.log_prob(bar_mu_2d)
            log_phi = torch.zeros_like(bar_mu[:, 0])
            log_phi[gt_3d_mask > 0] = log_phi_3d
            log_phi[gt_3d_mask < 1] = log_phi_2d
            log_phi = log_phi.reshape(BATCH_SIZE, self.num_joints, 1)

            nf_loss = torch.log(sigma) - log_phi
        else:
            nf_loss = None

        output = EasyDict(
            pred_jts=pred_jts,
            sigma=sigma,
            maxvals=scores.float(),
            nf_loss=nf_loss
        )
        return output