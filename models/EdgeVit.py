import torch
import torch.nn as nn

# edgevits的配置信息
edgevit_configs = {
    'XXS': {
        'channels': (36, 72, 144, 288),
        'blocks': (1, 1, 3, 2),
        'heads': (1, 2, 4, 8)
    }
    ,
    'XS': {
        'channels': (48, 96, 240, 384),
        'blocks': (1, 1, 2, 2),
        'heads': (1, 2, 4, 8)
    }
    ,
    'S': {
        'channels': (48, 96, 240, 384),
        'blocks': (1, 2, 3, 2),
        'heads': (1, 2, 4, 8)
    }
}

HYPERPARAMETERS = {
    'r': (4, 2, 2, 1)
}


class Residual(nn.Module):
    """
    残差网络
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class ConditionalPositionalEncoding(nn.Module):
    """
    
    """

    def __init__(self, channels):
        super(ConditionalPositionalEncoding, self).__init__()
        self.conditional_positional_encoding = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels,
                                                         bias=False)

    def forward(self, x):
        x = self.conditional_positional_encoding(x)
        return x


class MLP(nn.Module):
    """
    FFN 模块
    """

    def __init__(self, channels):
        super(MLP, self).__init__()
        expansion = 4
        self.mlp_layer_0 = nn.Conv2d(channels, channels * expansion, kernel_size=1, bias=False)
        self.mlp_act = nn.GELU()
        self.mlp_layer_1 = nn.Conv2d(channels * expansion, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.mlp_layer_0(x)
        x = self.mlp_act(x)
        x = self.mlp_layer_1(x)
        return x


class LocalAgg(nn.Module):
    """
    局部模块，LocalAgg
    卷积操作能够有效的提取局部特征
    为了能够降低计算量，使用 逐点卷积+深度可分离卷积实现
    """

    def __init__(self, channels):
        super(LocalAgg, self).__init__()
        self.bn = nn.BatchNorm2d(channels)
        # 逐点卷积，相当于全连接层。增加非线性，提高特征提取能力
        self.pointwise_conv_0 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        # 深度可分离卷积
        self.depthwise_conv = nn.Conv2d(channels, channels, padding=1, kernel_size=3, groups=channels, bias=False)
        # 归一化
        self.pointwise_prenorm_1 = nn.BatchNorm2d(channels)
        # 逐点卷积，相当于全连接层，增加非线性，提高特征提取能力
        self.pointwise_conv_1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.bn(x)
        x = self.pointwise_conv_0(x)
        x = self.depthwise_conv(x)
        x = self.pointwise_prenorm_1(x)
        x = self.pointwise_conv_1(x)
        return x


class GlobalSparseAttention(nn.Module):
    """
    全局模块，选取特定的tokens,进行全局作用
    """

    def __init__(self, channels, r, heads):
        """

        Args:
            channels: 通道数
            r: 下采样倍率
            heads: 注意力头的数目
                   这里使用的是多头注意力机制，MHSA,multi-head self-attention
        """
        super(GlobalSparseAttention, self).__init__()
        #
        self.head_dim = channels // heads
        # 扩张的
        self.scale = self.head_dim ** -0.5

        self.num_heads = heads

        # 使用平均池化,来进行特征提取
        self.sparse_sampler = nn.AvgPool2d(kernel_size=1, stride=r)

        # 计算qkv
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.sparse_sampler(x)
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).view(B, self.num_heads, -1, H * W).split([self.head_dim, self.head_dim, self.head_dim],
                                                                       dim=2)
        # 计算特征图 attention map
        attn = (q.transpose(-2, -1) @ k).softmax(-1)
        # value和特征图进行计算，得出全局注意力的结果
        x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W)
        # print(x.shape)
        return x


class LocalPropagation(nn.Module):
    def __init__(self, channels, r):
        super(LocalPropagation, self).__init__()

        # 组归一化
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        # 使用转置卷积 恢复 GlobalSparseAttention模块 r倍的下采样率
        self.local_prop = nn.ConvTranspose2d(channels,
                                             channels,
                                             kernel_size=r,
                                             stride=r,
                                             groups=channels)
        # 使用逐点卷积
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.local_prop(x)
        x = self.norm(x)
        x = self.proj(x)
        return x


class LGL(nn.Module):
    def __init__(self, channels, r, heads):
        super(LGL, self).__init__()

        self.cpe1 = ConditionalPositionalEncoding(channels)
        self.LocalAgg = LocalAgg(channels)
        self.mlp1 = MLP(channels)
        self.cpe2 = ConditionalPositionalEncoding(channels)
        self.GlobalSparseAttention = GlobalSparseAttention(channels, r, heads)
        self.LocalPropagation = LocalPropagation(channels, r)
        self.mlp2 = MLP(channels)

    def forward(self, x):
        # 1. 经过 位置编码操作
        x = self.cpe1(x) + x
        # 2. 经过第一步的 局部操作
        x = self.LocalAgg(x) + x
        # 3. 经过一个前馈网络
        x = self.mlp1(x) + x
        # 4. 经过一个位置编码操作
        x = self.cpe2(x) + x
        # 5. 经过一个全局捕捉的操作。长和宽缩小 r倍。然后通过一个
        # 6. 经过一个 局部操作部
        x = self.LocalPropagation(self.GlobalSparseAttention(x)) + x
        # 7. 经过一个前馈网络
        x = self.mlp2(x) + x

        return x


class DownSampleLayer(nn.Module):
    def __init__(self, dim_in, dim_out, r):
        super(DownSampleLayer, self).__init__()
        self.downsample = nn.Conv2d(dim_in,
                                    dim_out,
                                    kernel_size=r,
                                    stride=r)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)

    def forward(self, x):
        x = self.downsample(x)
        x = self.norm(x)
        return x

    # if __name__ == '__main__':


#     # 64通道，图片大小为32*32
#     x = torch.randn(size=(1, 64, 32, 32))
#     # 64通道，下采样2倍，8个头的注意力
#     model = LGL(64, 2, 8)
#     out = model(x)
#     print(out.shape)


class EdgeViT(nn.Module):

    def __init__(self, channels, blocks, heads, r=[4, 2, 2, 1], num_classes=1000, distillation=False):
        super(EdgeViT, self).__init__()
        self.distillation = distillation
        l = []
        in_channels = 3
        # 主体部分
        for stage_id, (num_channels, num_blocks, num_heads, sample_ratio) in enumerate(zip(channels, blocks, heads, r)):
            # print(num_channels,num_blocks,num_heads,sample_ratio)
            # print(in_channels)
            l.append(DownSampleLayer(dim_in=in_channels, dim_out=num_channels, r=4 if stage_id == 0 else 2))
            for _ in range(num_blocks):
                l.append(LGL(channels=num_channels, r=sample_ratio, heads=num_heads))

            in_channels = num_channels

        self.main_body = nn.Sequential(*l)
        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(in_channels, num_classes, bias=True)

        if self.distillation:
            self.dist_classifier = nn.Linear(in_channels, num_classes, bias=True)
        # print(self.main_body)

    def forward(self, x):
        print(x.shape)
        x = self.main_body(x)
        print(x.shape)
        x = self.pooling(x).flatten(1)
        print(x.shape)

        if self.distillation:
            x = self.classifier(x), self.dist_classifier(x)

            if not self.training:
                x = 1 / 2 * (x[0] + x[1])
        else:
            x = self.classifier(x)

        return x


def EdgeViT_XXS(pretrained=False):
    model = EdgeViT(**edgevit_configs['XXS'])

    if pretrained:
        raise NotImplementedError

    return model


def EdgeViT_XS(pretrained=False):
    model = EdgeViT(**edgevit_configs['XS'])

    if pretrained:
        raise NotImplementedError

    return model


def EdgeViT_S(pretrained=False):
    model = EdgeViT(**edgevit_configs['S'])

    if pretrained:
        raise NotImplementedError

    return model


if __name__ == '__main__':
    x = torch.randn(size=(1, 3, 256, 192))
    model = EdgeViT_XXS(False)
    y = model(x)
    print(y.shape)

    from thop import profile

    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input,))
    print("flops:{:.3f}G".format(flops /1e9))
    print("params:{:.3f}M".format(params /1e6))


