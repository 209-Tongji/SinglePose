from turtle import goto
import torch
from torch import nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride
        planes = expansion * in_planes
        #1X1卷积用于提升通道数
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        #分组卷积，组数为通道数，大大减少参数量和计算量
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu6(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu6(out)
        out = self.conv3(out)
        out = self.bn3(out)
        return out

class DeconvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DeconvBlock, self).__init__()

        self.convtrans = nn.ConvTranspose2d(in_channels=in_planes, out_channels=in_planes, kernel_size=4, stride=2, 
                padding=1, output_padding=0, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
    
    def forward(self, x):
        out = self.convtrans(x)
        out = self.bn1(out)
        out = F.relu6(out)
        out = self.conv(out)
        out = self.bn2(out)
        out = F.relu6(out)
        return out

class ShortCut(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ShortCut, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out

class DSPosev4(nn.Module):
    def __init__(self, cfg):
        super(DSPosev4, self).__init__()
        self.num_joints=cfg.DATA_PRESET.NUM_JOINTS
        self.in_channels = cfg.MODEL.IMG_CHANNEL
        self.layer1 = nn.Sequential(*[
            nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=3, stride=2, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(32)
        ])
        # self.layer2 = Block(in_planes=32, out_planes=24, expansion=1, stride=2)
        self.layer2 = nn.Sequential(*[
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(32),
        ])
        self.layer3 = Block(in_planes=32, out_planes=24, expansion=6, stride=2)
        self.layer4 = Block(in_planes=24, out_planes=48, expansion=6, stride=2)
        self.layer5 = Block(in_planes=48, out_planes=64, expansion=6, stride=2)
        
        self.layer6 = DeconvBlock(in_planes=64, out_planes=64)
        self.layer7 = DeconvBlock(in_planes=64, out_planes=64)
        self.layer8 = DeconvBlock(in_planes=64, out_planes=64)
        self.layer9 = DeconvBlock(in_planes=64, out_planes=64)

        self.layer10 = Block(in_planes=64, out_planes=24, expansion=1, stride=2)
        self.layer11 = Block(in_planes=24, out_planes=32, expansion=6, stride=2)
        self.layer12 = Block(in_planes=32, out_planes=48, expansion=6, stride=2)
        self.layer13 = Block(in_planes=48, out_planes=256, expansion=6, stride=2)

        self.layer14 = DeconvBlock(in_planes=256, out_planes=256)
        self.layer15 = DeconvBlock(in_planes=256, out_planes=256)
        self.layer16 = DeconvBlock(in_planes=256, out_planes=256)

        self.layer17 = nn.Conv2d(
            in_channels=256,
            out_channels= self.num_joints,
            kernel_size=1,
            stride=1,
            padding=0
        )

        #self.shortcut1 = ShortCut(24, 256)
        #self.shortcut2 = ShortCut(32, 256)
        #self.shortcut3 = ShortCut(48, 256)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)

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
        ),
        IMG_CHANNEL=4
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
    model = DSPosev4(cfg)

    flops, params = get_model_complexity_info(model, (4,256,192), as_strings=True, print_per_layer_stat=True)  #(3,512,512)输入图片的尺寸
    print("Flops: {}".format(flops))
    print("Params: " + params)

    image = torch.randn(1, 4, 256, 192)
    flops, params = profile(model, inputs=(image,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

    stat(model, (4, 256, 192))

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

