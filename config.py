import yaml
from easydict import EasyDict as edict
import argparse

parser = argparse.ArgumentParser(description='Residual Log-Likehood')
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    required=True,
                    type=str)

opt = parser.parse_args()

def update_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config
