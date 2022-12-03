from turtle import goto
import torch
from torch import nn
import torch.nn.functional as F

from MobileNet import MobileNetV2
from MobileNetv3 import MobileNetV3_Small, MobileNetV3_Large, MobileNetV3_Small_v1
from MobileNet_Mask import MobileNet_Mask
from Hourglass import ResNet

# derived from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
class DSHourglass(nn.Module):
    def __init__(self, cfg):
        super(DSHourglass, self).__init__()
        self.num_joints=cfg.DATA_PRESET.NUM_JOINTS
        backbone_pretrained = False
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
            # x = eval(f"tm.mobilenet_v2(pretrained=True)")
            #x = eval(f"tm.mobilenet_v2(pretrained=False)")
            #backbone_pretrained = True
            self.feature_channel = 1280
        
        elif cfg.MODEL.BACKBONE.TYPE == 'MobileNetV3_Small_v1':
            self.preact = MobileNetV3_Small_v1()
            self.feature_channel = 512
        
        elif cfg.MODEL.BACKBONE.TYPE == 'MobileNetV3_Small_i4':
            self.preact = MobileNetV3_Small(img_channel=4)
            self.feature_channel = 512
        
        elif cfg.MODEL.BACKBONE.TYPE == 'MobileNetV3_Small':
            self.preact = MobileNetV3_Small()
            self.feature_channel = 512
        
        elif cfg.MODEL.BACKBONE.TYPE == 'MobileNetV3_Large':
            self.preact = MobileNetV3_Large()
            self.feature_channel = 1280
        
        elif cfg.MODEL.BACKBONE.TYPE == 'MobileNet_i4':
            self.preact = MobileNetV2(img_channel=4)
            self.feature_channel = 1280
        
        elif cfg.MODEL.BACKBONE.TYPE == 'MobileNet_Mask':
            self.preact = MobileNet_Mask(img_channel=4, mask_layer=cfg.MODEL.BACKBONE.MASK_LIST)
            self.feature_channel = 1280
        
        elif cfg.MODEL.BACKBONE.TYPE == 'MobileNet_Mask_Attention':
            self.preact = MobileNet_Mask(img_channel=4, mask_layer=cfg.MODEL.BACKBONE.MASK_LIST)
            self.feature_channel = 1280


        if backbone_pretrained:
            model_state = self.preact.state_dict()
            state = {k: v for k, v in x.state_dict().items()
                        if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
            model_state.update(state)
            self.preact.load_state_dict(model_state)

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
        TYPE="DHourglass",
        BACKBONE=EasyDict(
            TYPE="ResNet",
            NUM_LAYERS=50 
        )
    ),
    DATA_PRESET=EasyDict(
        NUM_JOINTS=17,
    )

)

'''
Total params: 9,569,809
----------------------------------------------------------------------------------------------------------------------------------------------------------------------
Total memory: 84.74MB
Total MAdd: 3.16GMAdd
Total Flops: 328.97MFlops
Total MemR+W: 173.24MB
'''
if __name__ == '__main__':
    model = DSHourglass(cfg)

    flops, params = get_model_complexity_info(model, (3,256,192), as_strings=True, print_per_layer_stat=True)  #(3,512,512)输入图片的尺寸
    print("Flops: {}".format(flops))
    print("Params: " + params)

    image = torch.randn(1, 3, 256, 192)
    flops, params = profile(model, inputs=(image,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

    stat(model, (3, 256, 192))

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

