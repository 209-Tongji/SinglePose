from easydict import EasyDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexPose(nn.Module):
    def __init__(self, nof_joints=17):
        super(AlexPose, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4)         #(224,224) -> (55,55)
        self.conv2 = nn.Conv2d(96, 256, 5, padding=2)       
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)      
        self.conv4 = nn.Conv2d(384, 384, 3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, 3, padding=1)
        self.fc6 = nn.Linear(256*6*6, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, nof_joints*2)
        self.nof_joints = nof_joints

    def forward(self, x):
        # layer1
        h = F.relu(self.conv1(x))               #(227,227) -> (55,55)
        h = F.max_pool2d(h, 3, stride=2)        #(55,55) -> (27,27)
        # layer2
        h = F.relu(self.conv2(h))
        h = F.max_pool2d(h, 3, stride=2)        #(27,27) -> (12,12)
        # layer3-5
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.max_pool2d(h, 3, stride=2)        #(12,12) -> (6,6)

        h = h.view(-1, 256*6*6)
        # layer6-8
        h = F.dropout(F.relu(self.fc6(h)), training=self.training)
        h = F.dropout(F.relu(self.fc7(h)), training=self.training)
        h = self.fc8(h)

        return h.view(-1, self.nof_joints, 2)


from RegressFlow import ResNet
from MobileNet import MobileNetV2

class DeepPose(nn.Module):
    def __init__(self, cfg):
        super(DeepPose, self).__init__()
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

        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                    if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc_coord = nn.Linear(self.feature_channel, self.num_joints * 2)
        #self.fc_sigma = nn.Linear(self.feature_channel, self.num_joints * 2, norm=False)
    
    def forward(self, x):
        BATCH_SIZE = x.shape[0]

        feat = self.preact(x)

        feat = self.avg_pool(feat).reshape(BATCH_SIZE, -1)

        out_coord = self.fc_coord(feat).reshape(BATCH_SIZE, self.num_joints, 2)
        assert out_coord.shape[2] == 2

        #out_sigma = self.fc_sigma(feat).reshape(BATCH_SIZE, self.num_joints, -1)

        return out_coord

if __name__ == '__main__':
    model = DeepPose()
    image = torch.randn(1, 3, 227, 227)
    output = model(image)
    print(output.shape)




