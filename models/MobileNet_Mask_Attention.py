import torch
from torch import nn

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(3, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out, mask], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNet_Mask_Attention(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None,
                 img_channel=3,
                 mask_layer=[]):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNet_Mask_Attention, self).__init__()

        if block is None:
            self.block = InvertedResidual

        if norm_layer is None:
            self.norm_layer = nn.BatchNorm2d

        self.width_mult = width_mult
        self.round_nearest = round_nearest

        self.input_channel = 32
        self.last_channel = 1280
        self.mask_layer = mask_layer

        if inverted_residual_setting is None:
            self.inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(self.inverted_residual_setting) == 0 or len(self.inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(self.inverted_residual_setting))

        # building layers
        self.input_channel = _make_divisible(self.input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(self.last_channel * max(1.0, self.width_mult), self.round_nearest)
        self.layer0 = ConvBNReLU(img_channel, self.input_channel, stride=2, norm_layer=self.norm_layer)
        self.layer1 = self._make_layer(1)
        self.layer2 = self._make_layer(2)
        self.layer3 = self._make_layer(3)
        self.layer4 = self._make_layer(4)
        self.layer5 = self._make_layer(5)
        self.layer6 = self._make_layer(6)
        self.layer7 = self._make_layer(7)
        self.layer8 = ConvBNReLU(self.input_channel, self.last_channel, kernel_size=1, norm_layer=self.norm_layer)

        if 0 in self.mask_layer:
            self.sa0 = SpatialAttention()
        if 1 in self.mask_layer:
            self.sa1 = SpatialAttention()
        if 2 in self.mask_layer:
            self.sa2 = SpatialAttention()
        if 3 in self.mask_layer:
            self.sa3 = SpatialAttention()    
    

        # building classifier
        #self.classifier = nn.Sequential(
        #    nn.Dropout(0.2),
        #    nn.Linear(self.last_channel, num_classes),
        #)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _make_layer(self, index):
        t, c, n, s = self.inverted_residual_setting[index-1]
        self.output_channel = _make_divisible(c * self.width_mult, self.round_nearest)
        Blocks = []
        for i in range(n):
            stride = s if i == 0 else 1
            Blocks.append(self.block(self.input_channel, self.output_channel, stride, expand_ratio=t, norm_layer=self.norm_layer))
            self.input_channel = self.output_channel
        return nn.Sequential(*Blocks)


    def forward(self, x):
        mask = x[:,3:4,:,:]

        x = self.layer0(x)

        mask = mask[:, :, 0::2, 0::2]
        if 0 in self.mask_layer:
            x = self.sa0(x, mask) * x

        x = self.layer1(x)

        if 1 in self.mask_layer:
            x = self.sa1(x, mask) * x

        x = self.layer2(x)

        mask = mask[:, :, 0::2, 0::2]
        if 2 in self.mask_layer:
            x = self.sa2(x, mask) * x

        x = self.layer3(x)
        mask = mask[:, :, 0::2, 0::2]
        if 3 in self.mask_layer:
            x = self.sa3(x, mask) * x

        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x

from ptflops import get_model_complexity_info

if __name__ == '__main__':
    model = MobileNet_Mask(img_channel=4, mask_layer=[0,1,2,3])
    #print(model)
    img = torch.randn(1, 4, 256, 192)
    out = model(img)
    print(out.shape)

    flops, params = get_model_complexity_info(model, (4,256,192), as_strings=True, print_per_layer_stat=True)  #(3,512,512)输入图片的尺寸
    print("Flops: {}".format(flops))
    print("Params: " + params)