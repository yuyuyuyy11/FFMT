import argparse
import os
import shutil
import torch
import sys

# from datasets.builder import build_loader, get_dataloader
from models.build_model import build_model

from models.kpfcn_config import *
from tqdm import tqdm
from nnutils.fusion_trainer import Fusion_Trainer

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


DATASET = add_argument_group('Dataset')
DATASET.add_argument('--name', type=str, default='RGBD_3DMatch', help=['ScanNet', 'RGBD_3DMatch'])
DATASET.add_argument('--RGBD_3D_ROOT', type=str, default='/data/datasets/3dmatch')     
DATASET.add_argument('--SCANNET_ROOT', type=str, default='/data/datasets/scansNet')
DATASET.add_argument('--batch_size', type=int, default= 2)
DATASET.add_argument('--num_views', type=int, default= 2)
DATASET.add_argument('--view_spacing', type=int, default= 2)
DATASET.add_argument('--img_dim', type=int, default= 128)
DATASET.add_argument('--overfit', type=bool, default= False)
DATASET.add_argument('--num_workers', type=int, default= 12)
DATASET.add_argument('--input_type', type=str, default='hybrid')  #geo or hybrid
DATASET.add_argument('--filter', type=bool, default=True)
DATASET.add_argument('--pooling', type=str, default='mean', help=['max', 'mean'])
DATASET.add_argument('--complete', type=bool, default=True)
DATASET.add_argument('--voxelize', type=bool, default=False)
DATASET.add_argument('--voxel_size', type=float, default=0.025)
DATASET.add_argument('--processed', type=bool, default=False)


MODEL = add_argument_group('Model')
MODEL.add_argument('--model', type=str, default='FFMT')
MODEL.add_argument('--feat_dim', type=int, default=32)
MODEL.add_argument('--use_gt_vp', type=bool, default=False)

RENDER = add_argument_group('Render')
RENDER.add_argument('--render_size', type=int, default=128)
RENDER.add_argument('--points_per_pixel', type=int, default=16)
RENDER.add_argument('--radius', type=float, default=0.02 )
RENDER.add_argument('--weight_calculation', type=str, default="exponential")
RENDER.add_argument('--compositor', type=str, default="norm_weighted_sum")
RENDER.add_argument('--pointcloud_source', type=str, default="other")

ALIGN = add_argument_group('Alignment')
ALIGN.add_argument('--algorithm', type=str, default="weighted_procrustes")
ALIGN.add_argument('--base_weight', type=str, default="nn_ratio")
ALIGN.add_argument('--num_correspodances', type=int, default=200)
ALIGN.add_argument('--point_ratio', type=int, default=0.2)
ALIGN.add_argument('--num_seeds', type=int, default=10)

SYSTEM = add_argument_group('System')
SYSTEM.add_argument('--RANDOM_SEED', type=int, default=8)
SYSTEM.add_argument('--NUM_WORKERS', type=int, default=6)
SYSTEM.add_argument('--TQDM', type=bool, default=True)

TRAIN = add_argument_group('Traning')
TRAIN.add_argument('--eval_step', type=int, default=5000)
TRAIN.add_argument('--num_epochs', type=int, default=4)
TRAIN.add_argument('--vis_step', type=int, default=500)
TRAIN.add_argument('--optimizer', type=str, default="Adam")
TRAIN.add_argument('--lr', type=float, default=1e-4)
TRAIN.add_argument('--momentum', type=float, default=0.9)
TRAIN.add_argument('--weight_decay', type=float, default=1e-6)
TRAIN.add_argument('--scheduler', type=str, default="constant")
TRAIN.add_argument('--rgb_render_loss_weight', type=float, default=1.0)
TRAIN.add_argument('--rgb_decode_loss_weight', type=float, default=0.0)
TRAIN.add_argument('--depth_loss_weight', type=float, default=1.0)
TRAIN.add_argument('--correspondance_loss_weight', type=float, default=0.1)
TRAIN.add_argument('--resume', type=str, default="")
TRAIN.add_argument('--valid_num',type=int,default=100,help='fast valid')

EXPERIMENT = add_argument_group('Experiment')
EXPERIMENT.add_argument('--checkpoint', type=str, default="")
EXPERIMENT.add_argument('--EXPname', type=str, default="URR3dmatch")
EXPERIMENT.add_argument('--rationale', type=str, default="")
EXPERIMENT.add_argument('--just_evaluate', type=bool, default=False)

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
REPO_PATH = os.path.dirname(PROJECT_PATH)
PATHS = add_argument_group('Paths')
PATHS.add_argument('--project_root', type=str, default=PROJECT_PATH)
PATHS.add_argument('--html_visual_dir', type=str, default="")
PATHS.add_argument('--tensorboard_dir', type=str, default=os.path.join(REPO_PATH, "logs", "tensor_logs"))
PATHS.add_argument('--experiments_dir', type=str, default=os.path.join(REPO_PATH, "logs", "experiments"))

KPFCN = add_argument_group('KPFCN')
num_layer = 3
if num_layer == 3:
    KPFCN.add_argument('--architectures', type=list, default=kpfcn_backbone3)
else:
    KPFCN.add_argument('--architectures', type=list, default=kpfcn_backbone4)
KPFCN.add_argument('--num_layers', type=int, default=num_layer)
KPFCN.add_argument('--deform_radius', type=float, default=5.0)
KPFCN.add_argument('--first_subsampling_dl', type=float, default=0.025)
KPFCN.add_argument('--in_feats_dim', type=int, default=1)
KPFCN.add_argument('--conv_radius', type=float, default=2.5)
KPFCN.add_argument('--num_kernel_points', type=int, default=15)
KPFCN.add_argument('--KP_extent', type=float, default=2.0)
KPFCN.add_argument('--KP_influence', type=str, default='linear')
KPFCN.add_argument('--aggregation_mode', type=str, default='sum')
KPFCN.add_argument('--fixed_kernel_points', type=str, default='center')
KPFCN.add_argument('--use_batch_norm', type=bool, default=True)
KPFCN.add_argument('--deformable', type=bool, default=False)
KPFCN.add_argument('--batch_norm_momentum', type=float, default=0.02)
KPFCN.add_argument('--use_padding', type=bool, default=True)
KPFCN.add_argument('--first_feats_dim', type=int, default=128)
KPFCN.add_argument('--in_points_dim', type=int, default=3)
KPFCN.add_argument('--modulated', type=bool, default=False)


FUSION = add_argument_group('')
FUSION.add_argument('--num_i2p', type=int, default=16)
FUSION.add_argument('--num_p2i', type=int, default=1)



TS = add_argument_group('Using teacher student to initial KPFCN')
TS.add_argument('--use_teacher', type=bool, default=False)
TS.add_argument('--rot_augment', type=bool, default=False)
TS.add_argument('--rot_factor', type=float, default=1.0)
TS.add_argument('--teacher_checkpoint', type=str, default='')


UNET = add_argument_group('UNET')
UNET.add_argument('--num_downsample', type=int, default=num_layer)

TRANS = add_argument_group('Transformer')
TRANS.add_argument('--use_patch_emb', type=bool, default=True)
TRANS.add_argument('--use_res', type=bool, default=True)
TRANS.add_argument('--depth', type=int, default=4)
TRANS.add_argument('--num_heads', type=int, default=4)





if __name__ == "__main__":
    cfg = parser.parse_args()
    assert cfg.num_layers == cfg.num_downsample

    # to save memory
    if cfg.name == 'ScanNet':
        cfg.voxelize = True

    trainer = Fusion_Trainer(cfg)
    trainer.train()

