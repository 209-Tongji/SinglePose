import os
import sys
import torch
import torch.nn

sys.path.append("./models/")
from models.HRNet import HRNet
from models.LiteHRNet import LiteHRNet
from models.Hourglass import Hourglass
from models.RegressFlow import RegressFlow, RegressFlow3D
from models.DeepPose import DeepPose
from models.CPM import CPM
from models.DSHourglass import DSHourglass
from models.DSPose import DSPose
from models.DSPosev2 import DSPosev2
from models.DSPosev3 import DSPosev3
from models.DSPosev4 import DSPosev4
from models.DSPosev5 import DSPosev5
from models.DSPosev6 import DSPosev6


from config import update_config, opt
cfg = update_config(opt.cfg)

def load_pretrained(model, pretrained_path, device):
    if device == torch.device('cpu'):
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    else:
        pretrained_dict = torch.load(pretrained_path)
    model_dict = model.state_dict()
    for k, v in model_dict.items():
        if v.shape == pretrained_dict[k].shape:
            model_dict[k] = pretrained_dict[k]
        else:
            print('size mismatch in %s. Expect %s but get %s' %(k, v.shape, pretrained_dict[k].shape))
    model.load_state_dict(model_dict)
    print("Successfully load pre-trained model from %s " %pretrained_path)
    return model

if cfg.MODEL.TYPE == 'Hourglass':
    model = Hourglass(cfg=cfg)
elif cfg.MODEL.TYPE == 'HRNet':
    model = HRNet(32, cfg.DATA_PRESET.NUM_JOINTS, 0.1)
elif cfg.MODEL.TYPE == 'LiteHRNet':
    model = LiteHRNet()
elif cfg.MODEL.TYPE == 'RegressFlow':
    model = RegressFlow(cfg=cfg)
elif cfg.MODEL.TYPE == 'RegressFlow3d':
    model = RegressFlow3D(cfg=cfg)
elif cfg.MODEL.TYPE == 'DeepPose':
    model = DeepPose(cfg=cfg)
elif cfg.MODEL.TYPE == 'CPM':
    model = CPM(cfg.DATA_PRESET.NUM_JOINTS)
elif cfg.MODEL.TYPE == 'DSHourglass':
    model = DSHourglass(cfg=cfg)
elif cfg.MODEL.TYPE == 'DSPose':
    model = DSPose(cfg=cfg)
elif cfg.MODEL.TYPE == 'DSPosev2':
    model = DSPosev2(cfg=cfg)
elif cfg.MODEL.TYPE == 'DSPosev3':
    model = DSPosev3(cfg=cfg)
elif cfg.MODEL.TYPE == 'DSPosev4':
    model = DSPosev4(cfg=cfg)
elif cfg.MODEL.TYPE == 'DSPosev5':
    model = DSPosev5(cfg=cfg)
elif cfg.MODEL.TYPE == 'DSPosev6':
    model = DSPosev6(cfg=cfg)

model = load_pretrained(model, cfg.MODEL.PRETRAINED, torch.device("cpu"))
model.eval()

input_names = ['input']
output_names = ['output']

x = torch.randn(1,3,256,192,requires_grad=True)

onnx_model = os.path.basename(cfg.MODEL.PRETRAINED).split('.')[0] + '.onnx'

torch.onnx.export(model, x, onnx_model, input_names=input_names, output_names=output_names, verbose="True", opset_version=11)

import onnx
from onnxsim import simplify
origin_model = onnx.load(onnx_model)  # load onnx model
model_simp, check = simplify(origin_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, onnx_model)
print('finished exporting onnx')