import torch
from torch import nn
import torch.nn.functional as F

from MobileNet import MobileNetV2

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


resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3])
}

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


# derived from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
class Hourglass(nn.Module):
    def __init__(self, cfg):
        super(Hourglass, self).__init__()
        self.num_joints=cfg.DATA_PRESET.NUM_JOINTS
        if cfg.MODEL.BACKBONE.TYPE == 'ResNet':
            self.num_layers=cfg.MODEL.BACKBONE.NUM_LAYERS

            self.preact = ResNet(f"resnet{self.num_layers}")

            self.feature_channel = {
                    18: 512,
                    34: 512,
                    50: 2048,
                    101: 2048,
                    152: 2048
            }[self.num_layers]
        
            import torchvision.models as tm
            x = eval(f"tm.resnet{self.num_layers}(pretrained=True)")

        elif cfg.MODEL.BACKBONE.TYPE == 'MobileNet':
            self.preact = MobileNetV2()
            import torchvision.models as tm  # noqa: F401,F403
            x = eval(f"tm.mobilenet_v2(pretrained=True)")

            self.feature_channel = 1280
        
        '''
        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                    if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)
        '''

        self.inplanes = self.feature_channel

        # used for deconv layers
        self.deconv_with_bias = False
        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 256, 256],
            [4, 4, 4],
            bn_momentum=0.1
        )

        self.final_layer = nn.Conv2d(
            in_channels=256,
            out_channels= self.num_joints,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels, bn_momentum=0.1):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.preact(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x


from torchstat import stat
from thop import profile, clever_format
from ptflops import get_model_complexity_info

from easydict import EasyDict 
  
cfg = EasyDict(
    MODEL=EasyDict(
        TYPE="Hourglass",
        BACKBONE=EasyDict(
            TYPE="MobileNet",
            NUM_LAYERS=50
        )
    ),
    DATA_PRESET=EasyDict(
        NUM_JOINTS=17
    )

)

if __name__ == '__main__':
    model = Hourglass(cfg)

    flops, params = get_model_complexity_info(model, (3,256,192), as_strings=True, print_per_layer_stat=True)  #(3,512,512)输入图片的尺寸
    print("Flops: {}".format(flops))
    print("Params: " + params)

    image = torch.randn(1, 3, 256, 192)
    flops, params = profile(model, inputs=(image,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

    stat(model, (3, 256,  192))

    '''
    print(model)

    model.load_state_dict(
        torch.load('./weights/pose_resnet_50_256x192.pth')
    )
    print('ok!!')

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print(device)

    model = model.to(device)

    y = model(torch.ones(1, 3, 256, 192).to(device))
    print(y.shape)
    print(torch.min(y).item(), torch.mean(y).item(), torch.max(y).item())
    '''

