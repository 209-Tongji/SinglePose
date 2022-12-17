import torch
from torch import nn
import torch.nn.functional as F


# derived from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
class Hourglass(nn.Module):
    def __init__(self, cfg):
        super(Hourglass, self).__init__()
        self.num_joints=cfg.DATA_PRESET.NUM_JOINTS
        self.feature_channel = 1280
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
        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x

class DSHourglass(nn.Module):
    def __init__(self, cfg):
        super(DSHourglass, self).__init__()
        self.num_joints=cfg.DATA_PRESET.NUM_JOINTS
        self.feature_channel = 1280
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
                    out_channels=self.inplanes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    groups=self.inplanes,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(self.inplanes, momentum=bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            # Add depth-wise 1x1 conv layer
            layers.append(nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(planes, momentum=bn_momentum))
            layers.append(nn.ReLU6(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
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
            TYPE="ResNet",
            NUM_LAYERS=50
        )
    ),
    DATA_PRESET=EasyDict(
        NUM_JOINTS=17
    )

)

if __name__ == '__main__':
    model = DSHourglass(cfg)
    #model = Hourglass(cfg)
    input = torch.randn(2, 1280, 12, 9)
    output = model(input)
    print(output.shape)

    flops, params = get_model_complexity_info(model, (1280,12,9), as_strings=True, print_per_layer_stat=True)  #(3,512,512)输入图片的尺寸
    print("Flops: {}".format(flops))
    print("Params: " + params)

    image = torch.randn(1, 1280, 12, 9)
    flops, params = profile(model, inputs=(image,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

    #stat(model, (2048, 12,  9))

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

