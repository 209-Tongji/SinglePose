
import torch
import numpy as np

from .mirror import MirrorDataset
from .h36m import H36m

class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, **kwargs):
        self.datasets = [
            H36m(root=cfg.DATASET.TRAIN.LIST[0].ROOT, ann_file=cfg.DATASET.TRAIN.LIST[0].ANN, images_dir=cfg.DATASET.TRAIN.LIST[0].IMG_PREFIX, cfg=cfg, train=True),
            MirrorDataset(root=cfg.DATASET.TRAIN.LIST[1].ROOT, cfg=cfg, train=True)
        ]
        self.length = sum([len(ds) for ds in self.datasets])

        self.partition = [.8, .2]
        self.partition = np.array(self.partition).cumsum()

        #print(self.partition)

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(2):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]
    
    def __len__(self):
        return self.length