from turtle import goto
import torch
from torch import nn
import torch.nn.functional as F

class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret

class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret

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

class DSPosev6(nn.Module):
    def __init__(self, cfg):
        super(DSPosev6, self).__init__()
        self.num_joints=cfg.DATA_PRESET.NUM_JOINTS
        self.in_channels = cfg.MODEL.IMG_CHANNEL
        self.layer1 = nn.Sequential(*[
            #nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=3, stride=2, padding=1, groups=1, bias=False),
            CoordConv(in_channels=self.in_channels, out_channels=32, kernel_size=3, stride=2, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(32)
        ])
        self.sa1 = SpatialAttention()
        # self.layer2 = Block(in_planes=32, out_planes=24, expansion=1, stride=2)
        self.layer2 = nn.Sequential(*[
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(32),
        ])
        self.sa2 = SpatialAttention()
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
    
    def forward(self, x):
        mask = x[:,3:4,:,:]
        x = self.layer1(x)
        mask = mask[:, :, 0::2, 0::2]
        x = self.sa1(x, mask) * x
        x = self.layer2(x)
        mask = mask[:, :, 0::2, 0::2]
        x = self.sa2(x, mask) * x
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
    model = DSPosev6(cfg)

    input = torch.randn(2, 4, 256, 192)
    output = model(input)
    print(output.shape)
    '''
    flops, params = get_model_complexity_info(model, (4,256,192), as_strings=True, print_per_layer_stat=True)  #(3,512,512)输入图片的尺寸
    print("Flops: {}".format(flops))
    print("Params: " + params)

    image = torch.randn(1, 4, 256, 192)
    flops, params = profile(model, inputs=(image,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

    stat(model, (4, 256, 192))
    '''

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

