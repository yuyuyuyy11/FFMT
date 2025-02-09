import torch
from torch import nn as nn
from models.block import UnaryBlock
from utils.transformations import transform_points_Rt
from models.alignment import align
from models.backbones import *
from models.correspondence import get_correspondences
from models.model_util import get_grid, grid_to_pointcloud, points_to_ndc
from models.renderer import PointsRenderer
from monai.networks.nets import UNet
from pytorch3d.ops.knn import knn_points
from models.correspondence import calculate_ratio_test, get_topk_matches
import warnings
warnings.filterwarnings("ignore")

from utils.time import time_it


class PCReg_ffmt(nn.Module):
    def __init__(self,cfg):
        super(PCReg_ffmt, self).__init__()
        chan_in = 3
        self.cfg = cfg
        feat_dim = cfg.feat_dim
        
        self.cnn_pre_stages = 
        
        
class CnnPreStages(nn.Module):
    def __init__(self,cfg):
        super(CnnPreStages,self).__init__()
        self.
        
        
    def forward(x):
        