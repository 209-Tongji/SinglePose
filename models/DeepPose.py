from easydict import EasyDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepPose(nn.Module):
    def __init__(self, nof_joints=17):
        super(DeepPose, self).__init__()
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


if __name__ == '__main__':
    model = DeepPose()
    image = torch.randn(1, 3, 227, 227)
    output = model(image)
    print(output.shape)


