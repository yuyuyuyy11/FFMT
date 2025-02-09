import torch
from torch import nn as nn
from models.block import UnaryBlock
from utils.transformations import transform_points_Rt
from .alignment import align
from .backbones import *
from .correspondence import get_correspondences
from .model_util import get_grid, grid_to_pointcloud, points_to_ndc
from .renderer import PointsRenderer
from monai.networks.nets import UNet
from pytorch3d.ops.knn import knn_points
from models.correspondence import calculate_ratio_test, get_topk_matches
import warnings
warnings.filterwarnings("ignore")

from utils.time import time_it
from models.FFMT.loftr.position_encoding import PositionEncodingSine

def project_rgb(pc_0in1_X, rgb_src, renderer):
    # create rgb_features
    B, _, H, W = rgb_src.shape
    rgb_src = rgb_src.view(B, 3, H * W)
    rgb_src = rgb_src.permute(0, 2, 1).contiguous()

    # Rasterize and Blend
    project_0in1 = renderer(pc_0in1_X, rgb_src)

    return project_0in1["feats"]


#baseline
class PCReg(nn.Module):
    def __init__(self, cfg):
        super(PCReg, self).__init__()
        # set encoder decoder
        chan_in = 3
        self.cfg = cfg
        feat_dim = cfg.feat_dim

        # No imagenet pretraining
        pretrained = False
        self.encode = ResNetEncoder(chan_in, feat_dim, pretrained)
        self.decode = ResNetDecoder(feat_dim, 3, nn.Tanh(), pretrained)

        self.renderer = PointsRenderer(cfg)
        self.num_corres = cfg.num_correspodances
        self.pointcloud_source = cfg.pointcloud_source
        self.align_cfg = cfg

    def forward(self, rgbs, K, deps, vps=None):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)
        output = {}

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert feats[0].shape[-1] == deps[0].shape[-1], "Same size"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]

        if vps is not None:
            # Drop first viewpoint -- assumed to be identity transformation
            vps = vps[1:]
        elif self.align_cfg.algorithm == "weighted_procrustes":
            vps = []
            cor_loss = []
            for i in range(1, n_views):
                corr_i = get_correspondences(
                    P1=pcs_F[0],
                    P2=pcs_F[i],
                    P1_X=pcs_X[0],
                    P2_X=pcs_X[i],
                    num_corres=self.num_corres,
                    ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
                )
                Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

                vps.append(Rt_i)
                cor_loss.append(cor_loss_i)

                # add for visualization
                output[f"corres_0{i}"] = corr_i
                output[f"vp_{i}"] = Rt_i
        else:
            raise ValueError(f"How to align using {self.align_cfg.algorithm}?")

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        # Get RGB pointcloud as well for direct rendering
        pcs_rgb = [rgb.view(B, 3, -1).permute(0, 2, 1).contiguous() for rgb in rgbs]

        projs = []
        # get joint for all values
        if self.pointcloud_source == "joint":
            pcs_X_joint = torch.cat(pcs_X, dim=1)
            pcs_F_joint = torch.cat(pcs_F, dim=1)
            pcs_RGB_joint = torch.cat(pcs_rgb, dim=1)
            pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

        # Rasterize and Blend
        for i in range(n_views):
            if self.pointcloud_source == "other":
                # get joint for all values except the one
                pcs_X_joint = torch.cat(pcs_X[0:i] + pcs_X[i + 1 : n_views], dim=1)
                pcs_F_joint = torch.cat(pcs_F[0:i] + pcs_F[i + 1 : n_views], dim=1)
                pcs_RGB_joint = torch.cat(
                    pcs_rgb[0:i] + pcs_rgb[i + 1 : n_views], dim=1
                )
                pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

            if i > 0:
                rot_joint_X = transform_points_Rt(pcs_X_joint, vps[i - 1])
                rot_joint_X = points_to_ndc(rot_joint_X, K, (H, W))
            else:
                rot_joint_X = points_to_ndc(pcs_X_joint, K, (H, W))
            projs.append(self.renderer(rot_joint_X, pcs_FRGB_joint))

        # Decode
        for i in range(n_views):
            proj_FRGB_i = projs[i]["feats"]
            proj_RGB_i = proj_FRGB_i[:, -3:]
            proj_F_i = proj_FRGB_i[:, :-3]

            output[f"rgb_decode_{i}"] = self.decode(proj_F_i)
            output[f"rgb_render_{i}"] = proj_RGB_i
            output[f"ras_depth_{i}"] = projs[i]["depth"]
            output[f"cover_{i}"] = projs[i]["mask"].unsqueeze(1)  # useless

        return output

    def forward_pcreg(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)
        output = {}

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert feats[0].shape[-1] == deps[0].shape[-1], "Same size"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]

        vps = []
        cor_loss = []
        for i in range(1, n_views):
            corr_i = get_correspondences(
                P1=pcs_F[0],
                P2=pcs_F[i],
                P1_X=pcs_X[0],
                P2_X=pcs_X[i],
                num_corres=self.num_corres,
                ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
            )
            Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

            vps.append(Rt_i)
            cor_loss.append(cor_loss_i)

            # add for visualization
            output[f"corres_0{i}"] = corr_i
            output[f"vp_{i}"] = Rt_i

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        return output

    def generate_pointclouds(self, K, deps, vps=None):
        n_views = len(deps)
        # generate pointclouds - generate grid once for efficiency
        B, _, H, W = deps[0].shape
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pcs_X = [
            grid_to_pointcloud(K_inv, deps[i], None, grid)[0] for i in range(n_views)
        ]

        if vps is not None:
            pcs_X_rot = [
                transform_points_Rt(pcs_X[i + 1], vps[i + 1], inverse=True,)
                for i in range(n_views - 1)
            ]
            pcs_X = pcs_X[0:1] + pcs_X_rot
            pcs_X = torch.cat(pcs_X, dim=1).detach().cpu()

        return pcs_X

    def get_feature_pcs(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert (
            feats[0].shape[-1] == deps[0].shape[-1]
        ), f"Same size {feats[0].shape} - {deps[0].shape}"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]
        return pcs_X, pcs_F, None


#replace recurrent resblock in baseline with 2 resnet blocks
from .backbones import ResNetEncoder_modified


'''
pointMBF原版网络结构
'''
class PCReg_KPURes18_MSF(nn.Module):
    def __init__(self, cfg):
        super(PCReg_KPURes18_MSF, self).__init__()
        # set encoder decoder
        chan_in = 3
        self.cfg = cfg
        feat_dim = cfg.feat_dim

        # No imagenet pretraining
        pretrained = False

        encode_I = URes18Encoder1(chan_in, feat_dim, pretrained)
        encode_P = KPFCN(cfg)
        self.cnn_pre_stages = nn.Sequential(
            encode_I.inconv,
            encode_I.layer1
        )
        self.pcd_pre_stages = nn.ModuleList()
        for i in range(0,2):
            self.pcd_pre_stages.append(encode_P.encoder_blocks[i])

        self.cnn_ds_0 = encode_I.layer2
        self.pcd_ds_0 = nn.ModuleList()
        for i in range(2,5):
            self.pcd_ds_0.append(encode_P.encoder_blocks[i])

        self.cnn_ds_1 = encode_I.layer3
        self.pcd_ds_1 = nn.ModuleList()
        for i in range(5,8):
            self.pcd_ds_1.append(encode_P.encoder_blocks[i])

        self.cnn_up_0 = encode_I.up1
        self.pcd_up_0 = nn.ModuleList()
        for i in range(0, 2):
            self.pcd_up_0.append(encode_P.decoder_blocks[i])

        self.cnn_up_1 = nn.Sequential(
            encode_I.up2,
            encode_I.outconv
        )
        self.pcd_up_1 = nn.ModuleList()
        for i in range(2, 4):
            self.pcd_up_1.append(encode_P.decoder_blocks[i])

        # 0
        self.fuse_p2i_ds_0 = nn.ModuleList()
        self.fuse_p2i_ds_0.append(
            UnaryBlock(128, 64, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_0.append(
            nn.Sequential(
                nn.Conv2d(64*2, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_0 = nn.ModuleList()
        self.fuse_i2p_ds_0.append(
            UnaryBlock(64, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_0.append(
            UnaryBlock(128 * 2, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 1
        self.fuse_p2i_ds_1 = nn.ModuleList()
        self.fuse_p2i_ds_1.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_1.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_1 = nn.ModuleList()
        self.fuse_i2p_ds_1.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_1.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 2
        self.fuse_p2i_ds_2 = nn.ModuleList()
        self.fuse_p2i_ds_2.append(
            UnaryBlock(512, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_2.append(
            nn.Sequential(
                nn.Conv2d(256 * 2, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_2 = nn.ModuleList()
        self.fuse_i2p_ds_2.append(
            UnaryBlock(256, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_2.append(
            UnaryBlock(512 * 2, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        self.fuse_p2i_up_0 = nn.ModuleList()
        self.fuse_p2i_up_0.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_up_0.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
        ))
        self.fuse_i2p_up_0 = nn.ModuleList()
        self.fuse_i2p_up_0.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_up_0.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # self.fuse_p2i_up_1 = nn.ModuleList()
        # self.fuse_i2p_up_1 = nn.ModuleList()
        self.align_cfg = cfg
        self.renderer = PointsRenderer(cfg)
        self.num_corres = cfg.num_correspodances
        self.pointcloud_source = cfg.pointcloud_source
        self.map = Fusion_CATL(feat_dim)

    def forward(self, batch, rgbs, K, deps, vps=None):
        # Estimate Depth -- now for 1 and 2
        n_views = 2
        output = {}
        B, _, H, W = rgbs[0].shape

        # Encode features
        feat_p_encode = []
        feat_i_encode = []

        #pre stage
        feat_p = batch['features'].clone().detach() #全为1 (130982,1)
        for block_op in self.pcd_pre_stages:
            feat_p = block_op(feat_p, batch)# feat_p (130982,128)
            
        
        feat_i = [self.cnn_pre_stages(rgbs[i]) for i in range(n_views)] # feat_i [2](4,64,128,128)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][0].squeeze())#(131072,128)
        feat_p2i = self.fuse_p2i_ds_0[0](feat_p2i)#(131072,64)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_0[1])# [2](4,64,128,128)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][0])#(130982,32,64)
        feat_i2p = self.fuse_i2p_ds_0[0](feat_i2p.max(1)[0])#(130982,128)类似于pointnet的最大池化
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_0[1])#(130982,128)

        feat_p = feat_p + feat_i2p#(130982,128)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,64,128,128)

        feat_p_encode.append(feat_p)# [1](130982,128)
        feat_i_encode.append(feat_i)# [1][2](4,64,128,128)

        #downsample 0
        for block_op in self.pcd_ds_0:
            feat_p = block_op(feat_p, batch)# (17420,256)
        feat_i = [self.cnn_ds_0(feat_i[i]) for i in range(n_views)]# [2](4,128,64,64)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())#(32768,256)
        feat_p2i = self.fuse_p2i_ds_1[0](feat_p2i)#(32768,128)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_1[1])# [2](4,128,64,64)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])# (17420,32,128)
        feat_i2p = self.fuse_i2p_ds_1[0](feat_i2p.max(1)[0])# (17420,256)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_1[1])# (17420,256)

        feat_p = feat_p + feat_i2p# (17420,256)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,128,64,64)

        feat_p_encode.append(feat_p)# [(130982,128)(17420,256)]
        feat_i_encode.append(feat_i)# [[2](4,64,128,128),[2](4,128,64,64)]

        #downsample 1
        for block_op in self.pcd_ds_1:
            feat_p = block_op(feat_p, batch)# (5895,512)
        feat_i = [self.cnn_ds_1(feat_i[i]) for i in range(n_views)]# [2](4,256,32,32)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][2].squeeze())#(8192,512)
        feat_p2i = self.fuse_p2i_ds_2[0](feat_p2i)#(8192,256)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_2[1])# [2](4,256,32,32)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][2])#(5895,32,256)
        feat_i2p = self.fuse_i2p_ds_2[0](feat_i2p.max(1)[0])#(5895,512)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_2[1])#(5895,512)

        feat_p = feat_p + feat_i2p#(5895,512)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,256,32,32)


        # upsample0
        for block_i, block_op in enumerate(self.pcd_up_0):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)#(17420,256)

        feat_i = [
            self.cnn_up_0(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-1][i]), dim=1)) for i in range(n_views)]#[2](4,128,64,64)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())#(32768,256)
        feat_p2i = self.fuse_p2i_up_0[0](feat_p2i)#(32768,128)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_up_0[1])#[2](4,128,64,64)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])# (17420,32,128)
        feat_i2p = self.fuse_i2p_up_0[0](feat_i2p.max(1)[0])# (17420,256)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_up_0[1])# (17420,256)

        feat_p = feat_p + feat_i2p # (17420,256)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]#[2](4,128,64,64)

        # upsample1
        for block_i, block_op in enumerate(self.pcd_up_1):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)#(130982,32)

        feat_i = [
            self.cnn_up_1(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-2][i]), dim=1)) for i in range(n_views)]#[2](4,32,128,128)

        pcs_X = batch['points_img']     #[2](4,16384,3)
        pcs_F = self.map(feat_i, feat_p, batch) #[2](4,16384,32)

        if vps is not None:
            # Drop first viewpoint -- assumed to be identity transformation
            vps = vps[1:]
        elif self.align_cfg.algorithm == "weighted_procrustes":
            vps = []
            cor_loss = []
            for i in range(1, n_views):
                corr_i = get_correspondences(
                    P1=pcs_F[0],
                    P2=pcs_F[i],
                    P1_X=pcs_X[0],
                    P2_X=pcs_X[i],
                    num_corres=self.num_corres,
                    ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
                )
                Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

                vps.append(Rt_i)
                cor_loss.append(cor_loss_i)

                # add for visualization
                output[f"corres_0{i}"] = corr_i
                output[f"vp_{i}"] = Rt_i
        else:
            raise ValueError(f"How to align using {self.align_cfg.algorithm}?")

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        # Get RGB pointcloud as well for direct rendering
        pcs_rgb = [rgb.view(B, 3, -1).permute(0, 2, 1).contiguous() for rgb in rgbs]

        projs = []
        # get joint for all values
        if self.pointcloud_source == "joint":
            pcs_X_joint = torch.cat(pcs_X, dim=1)
            pcs_F_joint = torch.cat(pcs_F, dim=1)
            pcs_RGB_joint = torch.cat(pcs_rgb, dim=1)
            pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

        # Rasterize and Blend
        for i in range(n_views):
            if self.pointcloud_source == "other":
                # get joint for all values except the one
                pcs_X_joint = torch.cat(pcs_X[0:i] + pcs_X[i + 1: n_views], dim=1)
                pcs_F_joint = torch.cat(pcs_F[0:i] + pcs_F[i + 1: n_views], dim=1)
                pcs_RGB_joint = torch.cat(
                    pcs_rgb[0:i] + pcs_rgb[i + 1: n_views], dim=1
                )
                pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

            if i > 0:
                rot_joint_X = transform_points_Rt(pcs_X_joint, vps[i - 1])
                rot_joint_X = points_to_ndc(rot_joint_X, K, (H, W))
            else:
                rot_joint_X = points_to_ndc(pcs_X_joint, K, (H, W))
            projs.append(self.renderer(rot_joint_X, pcs_FRGB_joint))

        # Decode
        for i in range(n_views):
            proj_FRGB_i = projs[i]["feats"]
            proj_RGB_i = proj_FRGB_i[:, -3:]
            proj_F_i = proj_FRGB_i[:, :-3]

            output[f"rgb_render_{i}"] = proj_RGB_i
            output[f"ras_depth_{i}"] = projs[i]["depth"]
            output[f"cover_{i}"] = projs[i]["mask"].unsqueeze(1)  # useless

        return output

    def gather_p2i(self, feat_p, idx):
        feat_p = torch.cat((feat_p, torch.zeros_like(feat_p[:1, :])), 0)
        return feat_p[idx]

    def gather_i2p(self, feat_i, idx):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        feat_i2p = []
        for i in range(B):
            feat_i2p.append(src_feat_i[i].reshape(C,H*W).permute(1,0))
            feat_i2p.append(tgt_feat_i[i].reshape(C,H*W).permute(1,0))

        feat_i2p = torch.cat(feat_i2p, 0)
        feat_i2p = torch.cat((feat_i2p, torch.zeros_like(feat_i2p[:1, :])), 0)
        return feat_i2p[idx]  #N,16,C

    def fusep2i(self, feat_i, feat_p2i, layer):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        # src_feat_i = src_feat_i.reshape(B, C, H * W).permute(0, 2, 1)
        # tgt_feat_i = tgt_feat_i.reshape(B, C, H * W).permute(0, 2, 1)

        src_feat_p2i = []
        tgt_feat_p2i = []

        for i in range(2*B):
            if i % 2 == 0:
                src_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))#[4](1,16384,64)
            else:
                tgt_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))#[4](1,16384,64)

        src_feat_p2i = torch.vstack(src_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)#(4,64,128,128)
        tgt_feat_p2i = torch.vstack(tgt_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)#(4,64,128,128)

        src_feat_p2i = torch.cat([src_feat_i, src_feat_p2i], 1)#(4,128,128,128)
        tgt_feat_p2i = torch.cat([tgt_feat_i, tgt_feat_p2i], 1)#(4,128,128,128)

        src_feat_p2i = layer(src_feat_p2i)#(4,64,128,128)
        tgt_feat_p2i = layer(tgt_feat_p2i)#(4,64,128,128)

        return [src_feat_p2i, tgt_feat_p2i]

    def fusei2p(self, feat_p, feat_i2p, layer):
        feat_i2p = torch.cat([feat_p, feat_i2p], -1)
        feat_i2p = layer(feat_i2p)
        return feat_i2p


    def generate_pointclouds(self, K, deps, vps=None):
        n_views = len(deps)
        # generate pointclouds - generate grid once for efficiency
        B, _, H, W = deps[0].shape
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pcs_X = [
            grid_to_pointcloud(K_inv, deps[i], None, grid)[0] for i in range(n_views)
        ]

        if vps is not None:
            pcs_X_rot = [
                transform_points_Rt(pcs_X[i + 1], vps[i + 1], inverse=True, )
                for i in range(n_views - 1)
            ]
            pcs_X = pcs_X[0:1] + pcs_X_rot
            pcs_X = torch.cat(pcs_X, dim=1).detach().cpu()

        return pcs_X

    def get_feature_pcs(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert (
                feats[0].shape[-1] == deps[0].shape[-1]
        ), f"Same size {feats[0].shape} - {deps[0].shape}"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]
        return pcs_X, pcs_F, None



'''
pointMBF 添加自注意力模块
'''
from models.FFMT.loftr.transfromer import LocalFeatureTransformer,LoFTREncoderLayer

class PCReg_KPURes18_MSF_loftr(nn.Module):
    def __init__(self, cfg):
        super(PCReg_KPURes18_MSF_loftr, self).__init__()
        # set encoder decoder
        chan_in = 3
        self.cfg = cfg
        feat_dim = cfg.feat_dim

        # No imagenet pretraining
        pretrained = False

        encode_I = URes18Encoder1(chan_in, feat_dim, pretrained)
        encode_P = KPFCN(cfg)
        self.cnn_pre_stages = nn.Sequential(
            encode_I.inconv,
            encode_I.layer1
        )
        self.pcd_pre_stages = nn.ModuleList()
        for i in range(0,2):
            self.pcd_pre_stages.append(encode_P.encoder_blocks[i])

        self.cnn_ds_0 = encode_I.layer2
        self.pcd_ds_0 = nn.ModuleList()
        for i in range(2,5):
            self.pcd_ds_0.append(encode_P.encoder_blocks[i])

        self.cnn_ds_1 = encode_I.layer3
        self.pcd_ds_1 = nn.ModuleList()
        for i in range(5,8):
            self.pcd_ds_1.append(encode_P.encoder_blocks[i])

        self.cnn_up_0 = encode_I.up1
        self.pcd_up_0 = nn.ModuleList()
        for i in range(0, 2):
            self.pcd_up_0.append(encode_P.decoder_blocks[i])

        self.cnn_up_1 = nn.Sequential(
            encode_I.up2,
            encode_I.outconv
        )
        self.pcd_up_1 = nn.ModuleList()
        for i in range(2, 4):
            self.pcd_up_1.append(encode_P.decoder_blocks[i])

        # 0
        self.fuse_p2i_ds_0 = nn.ModuleList()
        self.fuse_p2i_ds_0.append(
            UnaryBlock(128, 64, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_0.append(
            nn.Sequential(
                nn.Conv2d(64*2, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_0 = nn.ModuleList()
        self.fuse_i2p_ds_0.append(
            UnaryBlock(64, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_0.append(
            UnaryBlock(128 * 2, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 1
        self.fuse_p2i_ds_1 = nn.ModuleList()
        self.fuse_p2i_ds_1.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_1.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_1 = nn.ModuleList()
        self.fuse_i2p_ds_1.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_1.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 2
        self.fuse_p2i_ds_2 = nn.ModuleList()
        self.fuse_p2i_ds_2.append(
            UnaryBlock(512, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_2.append(
            nn.Sequential(
                nn.Conv2d(256 * 2, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_2 = nn.ModuleList()
        self.fuse_i2p_ds_2.append(
            UnaryBlock(256, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_2.append(
            UnaryBlock(512 * 2, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        self.fuse_p2i_up_0 = nn.ModuleList()
        self.fuse_p2i_up_0.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_up_0.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
        ))
        self.fuse_i2p_up_0 = nn.ModuleList()
        self.fuse_i2p_up_0.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_up_0.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # self.fuse_p2i_up_1 = nn.ModuleList()
        # self.fuse_i2p_up_1 = nn.ModuleList()
        self.align_cfg = cfg
        self.renderer = PointsRenderer(cfg)
        self.num_corres = cfg.num_correspodances
        self.pointcloud_source = cfg.pointcloud_source
        self.map = Fusion_CATL(feat_dim)
        

        self.loftr_coarse = LocalFeatureTransformer({'d_model':32,
                'nhead':8,
                'layer_names':['self', 'cross'] * 2,
                'attention':'linear'})
        

    def forward(self, batch, rgbs, K, deps, vps=None):
        # Estimate Depth -- now for 1 and 2
        n_views = 2
        output = {}
        B, _, H, W = rgbs[0].shape

        # Encode features
        feat_p_encode = []
        feat_i_encode = []

        #pre stage
        feat_p = batch['features'].clone().detach()
        for block_op in self.pcd_pre_stages:
            feat_p = block_op(feat_p, batch)# feat_p (130982,128)
            
        
        feat_i = [self.cnn_pre_stages(rgbs[i]) for i in range(n_views)] # feat_i [2](4,64,128,128)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][0].squeeze())
        feat_p2i = self.fuse_p2i_ds_0[0](feat_p2i)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_0[1])

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][0])
        feat_i2p = self.fuse_i2p_ds_0[0](feat_i2p.max(1)[0])
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_0[1])

        feat_p = feat_p + feat_i2p
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]

        feat_p_encode.append(feat_p)
        feat_i_encode.append(feat_i)

        #downsample 0
        for block_op in self.pcd_ds_0:
            feat_p = block_op(feat_p, batch)
        feat_i = [self.cnn_ds_0(feat_i[i]) for i in range(n_views)]

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())
        feat_p2i = self.fuse_p2i_ds_1[0](feat_p2i)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_1[1])

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])
        feat_i2p = self.fuse_i2p_ds_1[0](feat_i2p.max(1)[0])
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_1[1])

        feat_p = feat_p + feat_i2p
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]

        feat_p_encode.append(feat_p)
        feat_i_encode.append(feat_i)

        #downsample 1
        for block_op in self.pcd_ds_1:
            feat_p = block_op(feat_p, batch)
        feat_i = [self.cnn_ds_1(feat_i[i]) for i in range(n_views)]

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][2].squeeze())
        feat_p2i = self.fuse_p2i_ds_2[0](feat_p2i)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_2[1])

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][2])
        feat_i2p = self.fuse_i2p_ds_2[0](feat_i2p.max(1)[0])
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_2[1])

        feat_p = feat_p + feat_i2p
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]


        # upsample0
        for block_i, block_op in enumerate(self.pcd_up_0):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)

        feat_i = [
            self.cnn_up_0(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-1][i]), dim=1)) for i in range(n_views)]

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())
        feat_p2i = self.fuse_p2i_up_0[0](feat_p2i)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_up_0[1])

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])
        feat_i2p = self.fuse_i2p_up_0[0](feat_i2p.max(1)[0])
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_up_0[1])

        feat_p = feat_p + feat_i2p
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]

        # upsample1
        for block_i, block_op in enumerate(self.pcd_up_1):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)

        feat_i = [
            self.cnn_up_1(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-2][i]), dim=1)) for i in range(n_views)]

        pcs_X = batch['points_img']     #[2](4,16384,3)
        pcs_F = self.map(feat_i, feat_p, batch) #[2](4,16384,32)
        
        #-----------------
        #修改此处

        pcs_F = list(self.loftr_coarse(pcs_F[0],pcs_F[1]))
        # pcs_F = pcs_F+pcs_F_sc
        
        if vps is not None:
            # Drop first viewpoint -- assumed to be identity transformation
            vps = vps[1:]
        elif self.align_cfg.algorithm == "weighted_procrustes":
            vps = []
            cor_loss = []
            for i in range(1, n_views):
                corr_i = get_correspondences(
                    P1=pcs_F[0],
                    P2=pcs_F[i],
                    P1_X=pcs_X[0],
                    P2_X=pcs_X[i],
                    num_corres=self.num_corres,
                    ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
                )
                Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

                vps.append(Rt_i)
                cor_loss.append(cor_loss_i)

                # add for visualization
                output[f"corres_0{i}"] = corr_i
                output[f"vp_{i}"] = Rt_i
        else:
            raise ValueError(f"How to align using {self.align_cfg.algorithm}?")

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        # Get RGB pointcloud as well for direct rendering
        pcs_rgb = [rgb.view(B, 3, -1).permute(0, 2, 1).contiguous() for rgb in rgbs]

        projs = []
        # get joint for all values
        if self.pointcloud_source == "joint":
            pcs_X_joint = torch.cat(pcs_X, dim=1)
            pcs_F_joint = torch.cat(pcs_F, dim=1)
            pcs_RGB_joint = torch.cat(pcs_rgb, dim=1)
            pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

        # Rasterize and Blend
        for i in range(n_views):
            if self.pointcloud_source == "other":
                # get joint for all values except the one
                pcs_X_joint = torch.cat(pcs_X[0:i] + pcs_X[i + 1: n_views], dim=1)
                pcs_F_joint = torch.cat(pcs_F[0:i] + pcs_F[i + 1: n_views], dim=1)
                pcs_RGB_joint = torch.cat(
                    pcs_rgb[0:i] + pcs_rgb[i + 1: n_views], dim=1
                )
                pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

            if i > 0:
                rot_joint_X = transform_points_Rt(pcs_X_joint, vps[i - 1])
                rot_joint_X = points_to_ndc(rot_joint_X, K, (H, W))
            else:
                rot_joint_X = points_to_ndc(pcs_X_joint, K, (H, W))
            projs.append(self.renderer(rot_joint_X, pcs_FRGB_joint))

        # Decode
        for i in range(n_views):
            proj_FRGB_i = projs[i]["feats"]
            proj_RGB_i = proj_FRGB_i[:, -3:]
            proj_F_i = proj_FRGB_i[:, :-3]

            output[f"rgb_render_{i}"] = proj_RGB_i
            output[f"ras_depth_{i}"] = projs[i]["depth"]
            output[f"cover_{i}"] = projs[i]["mask"].unsqueeze(1)  # useless

        return output

    def gather_p2i(self, feat_p, idx):
        feat_p = torch.cat((feat_p, torch.zeros_like(feat_p[:1, :])), 0)
        return feat_p[idx]

    def gather_i2p(self, feat_i, idx):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        feat_i2p = []
        for i in range(B):
            feat_i2p.append(src_feat_i[i].reshape(C,H*W).permute(1,0))
            feat_i2p.append(tgt_feat_i[i].reshape(C,H*W).permute(1,0))

        feat_i2p = torch.cat(feat_i2p, 0)
        feat_i2p = torch.cat((feat_i2p, torch.zeros_like(feat_i2p[:1, :])), 0)
        return feat_i2p[idx]  #N,16,C

    def fusep2i(self, feat_i, feat_p2i, layer):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        # src_feat_i = src_feat_i.reshape(B, C, H * W).permute(0, 2, 1)
        # tgt_feat_i = tgt_feat_i.reshape(B, C, H * W).permute(0, 2, 1)

        src_feat_p2i = []
        tgt_feat_p2i = []

        for i in range(2*B):
            if i % 2 == 0:
                src_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))
            else:
                tgt_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))

        src_feat_p2i = torch.vstack(src_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)
        tgt_feat_p2i = torch.vstack(tgt_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)

        src_feat_p2i = torch.cat([src_feat_i, src_feat_p2i], 1)
        tgt_feat_p2i = torch.cat([tgt_feat_i, tgt_feat_p2i], 1)

        src_feat_p2i = layer(src_feat_p2i)
        tgt_feat_p2i = layer(tgt_feat_p2i)

        return [src_feat_p2i, tgt_feat_p2i]

    def fusei2p(self, feat_p, feat_i2p, layer):
        feat_i2p = torch.cat([feat_p, feat_i2p], -1)
        feat_i2p = layer(feat_i2p)
        return feat_i2p


    def generate_pointclouds(self, K, deps, vps=None):
        n_views = len(deps)
        # generate pointclouds - generate grid once for efficiency
        B, _, H, W = deps[0].shape
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pcs_X = [
            grid_to_pointcloud(K_inv, deps[i], None, grid)[0] for i in range(n_views)
        ]

        if vps is not None:
            pcs_X_rot = [
                transform_points_Rt(pcs_X[i + 1], vps[i + 1], inverse=True, )
                for i in range(n_views - 1)
            ]
            pcs_X = pcs_X[0:1] + pcs_X_rot
            pcs_X = torch.cat(pcs_X, dim=1).detach().cpu()

        return pcs_X

    def get_feature_pcs(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert (
                feats[0].shape[-1] == deps[0].shape[-1]
        ), f"Same size {feats[0].shape} - {deps[0].shape}"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]
        return pcs_X, pcs_F, None

class PCReg_KPURes18_MSF_loftr1(nn.Module):
    def __init__(self, cfg):
        super(PCReg_KPURes18_MSF_loftr1, self).__init__()
        # set encoder decoder
        chan_in = 3
        self.cfg = cfg
        feat_dim = cfg.feat_dim

        # No imagenet pretraining
        pretrained = False

        encode_I = URes18Encoder1(chan_in, feat_dim, pretrained)
        encode_P = KPFCN(cfg)
        self.cnn_pre_stages = nn.Sequential(
            encode_I.inconv,
            encode_I.layer1
        )
        self.pcd_pre_stages = nn.ModuleList()
        for i in range(0,2):
            self.pcd_pre_stages.append(encode_P.encoder_blocks[i])

        self.cnn_ds_0 = encode_I.layer2
        self.pcd_ds_0 = nn.ModuleList()
        for i in range(2,5):
            self.pcd_ds_0.append(encode_P.encoder_blocks[i])

        self.cnn_ds_1 = encode_I.layer3
        self.pcd_ds_1 = nn.ModuleList()
        for i in range(5,8):
            self.pcd_ds_1.append(encode_P.encoder_blocks[i])

        self.cnn_up_0 = encode_I.up1
        self.pcd_up_0 = nn.ModuleList()
        for i in range(0, 2):
            self.pcd_up_0.append(encode_P.decoder_blocks[i])

        self.cnn_up_1 = nn.Sequential(
            encode_I.up2,
            encode_I.outconv
        )
        self.pcd_up_1 = nn.ModuleList()
        for i in range(2, 4):
            self.pcd_up_1.append(encode_P.decoder_blocks[i])

        # 0
        self.fuse_p2i_ds_0 = nn.ModuleList()
        self.fuse_p2i_ds_0.append(
            UnaryBlock(128, 64, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_0.append(
            nn.Sequential(
                nn.Conv2d(64*2, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_0 = nn.ModuleList()
        self.fuse_i2p_ds_0.append(
            UnaryBlock(64, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_0.append(
            UnaryBlock(128 * 2, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 1
        self.fuse_p2i_ds_1 = nn.ModuleList()
        self.fuse_p2i_ds_1.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_1.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_1 = nn.ModuleList()
        self.fuse_i2p_ds_1.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_1.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 2
        self.fuse_p2i_ds_2 = nn.ModuleList()
        self.fuse_p2i_ds_2.append(
            UnaryBlock(512, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_2.append(
            nn.Sequential(
                nn.Conv2d(256 * 2, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_2 = nn.ModuleList()
        self.fuse_i2p_ds_2.append(
            UnaryBlock(256, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_2.append(
            UnaryBlock(512 * 2, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        self.fuse_p2i_up_0 = nn.ModuleList()
        self.fuse_p2i_up_0.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_up_0.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
        ))
        self.fuse_i2p_up_0 = nn.ModuleList()
        self.fuse_i2p_up_0.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_up_0.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # self.fuse_p2i_up_1 = nn.ModuleList()
        # self.fuse_i2p_up_1 = nn.ModuleList()
        self.align_cfg = cfg
        self.renderer = PointsRenderer(cfg)
        self.num_corres = cfg.num_correspodances
        self.pointcloud_source = cfg.pointcloud_source
        self.map = Fusion_CATL(feat_dim)
        
        self.position_encoding = PositionEncodingSine(d_model=32)
        self.loftr_coarse = LocalFeatureTransformer({'d_model':32,
                'nhead':8,
                'layer_names':['self', 'cross'] * 2,
                'attention':'linear'})
        

    def forward(self, batch, rgbs, K, deps, vps=None):
        # Estimate Depth -- now for 1 and 2
        n_views = 2
        output = {}
        B, _, H, W = rgbs[0].shape

        # Encode features
        feat_p_encode = []
        feat_i_encode = []

        #pre stage
        feat_p = batch['features'].clone().detach()
        for block_op in self.pcd_pre_stages:
            feat_p = block_op(feat_p, batch)# feat_p (130982,128)
            
        
        feat_i = [self.cnn_pre_stages(rgbs[i]) for i in range(n_views)] # feat_i [2](4,64,128,128)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][0].squeeze())
        feat_p2i = self.fuse_p2i_ds_0[0](feat_p2i)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_0[1])

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][0])
        feat_i2p = self.fuse_i2p_ds_0[0](feat_i2p.max(1)[0])
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_0[1])

        feat_p = feat_p + feat_i2p
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]

        feat_p_encode.append(feat_p)
        feat_i_encode.append(feat_i)

        #downsample 0
        for block_op in self.pcd_ds_0:
            feat_p = block_op(feat_p, batch)
        feat_i = [self.cnn_ds_0(feat_i[i]) for i in range(n_views)]

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())
        feat_p2i = self.fuse_p2i_ds_1[0](feat_p2i)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_1[1])

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])
        feat_i2p = self.fuse_i2p_ds_1[0](feat_i2p.max(1)[0])
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_1[1])

        feat_p = feat_p + feat_i2p
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]

        feat_p_encode.append(feat_p)
        feat_i_encode.append(feat_i)

        #downsample 1
        for block_op in self.pcd_ds_1:
            feat_p = block_op(feat_p, batch)
        feat_i = [self.cnn_ds_1(feat_i[i]) for i in range(n_views)]

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][2].squeeze())
        feat_p2i = self.fuse_p2i_ds_2[0](feat_p2i)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_2[1])

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][2])
        feat_i2p = self.fuse_i2p_ds_2[0](feat_i2p.max(1)[0])
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_2[1])

        feat_p = feat_p + feat_i2p
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]


        # upsample0
        for block_i, block_op in enumerate(self.pcd_up_0):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)

        feat_i = [
            self.cnn_up_0(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-1][i]), dim=1)) for i in range(n_views)]

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())
        feat_p2i = self.fuse_p2i_up_0[0](feat_p2i)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_up_0[1])

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])
        feat_i2p = self.fuse_i2p_up_0[0](feat_i2p.max(1)[0])
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_up_0[1])

        feat_p = feat_p + feat_i2p
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]

        # upsample1
        for block_i, block_op in enumerate(self.pcd_up_1):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)

        feat_i = [
            self.cnn_up_1(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-2][i]), dim=1)) for i in range(n_views)]

        pcs_X = batch['points_img']     #[2](4,16384,3)
        pcs_F = self.map(feat_i, feat_p, batch) #[2](4,16384,32)
        
        #-----------------
        #修改此处
        pcs_F = [
            self.position_encoding(pcs_F[0].permute(0,2,1).reshape(B,32,128,128)).reshape(B,32,16384).permute(0,2,1),
            self.position_encoding(pcs_F[1].permute(0,2,1).reshape(B,32,128,128)).reshape(B,32,16384).permute(0,2,1)
        ]
        pcs_F = pcs_F+list(self.loftr_coarse(pcs_F[0],pcs_F[1]))
        
        
        if vps is not None:
            # Drop first viewpoint -- assumed to be identity transformation
            vps = vps[1:]
        elif self.align_cfg.algorithm == "weighted_procrustes":
            vps = []
            cor_loss = []
            for i in range(1, n_views):
                corr_i = get_correspondences(
                    P1=pcs_F[0],
                    P2=pcs_F[i],
                    P1_X=pcs_X[0],
                    P2_X=pcs_X[i],
                    num_corres=self.num_corres,
                    ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
                )
                Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

                vps.append(Rt_i)
                cor_loss.append(cor_loss_i)

                # add for visualization
                output[f"corres_0{i}"] = corr_i
                output[f"vp_{i}"] = Rt_i
        else:
            raise ValueError(f"How to align using {self.align_cfg.algorithm}?")

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        # Get RGB pointcloud as well for direct rendering
        pcs_rgb = [rgb.view(B, 3, -1).permute(0, 2, 1).contiguous() for rgb in rgbs]

        projs = []
        # get joint for all values
        if self.pointcloud_source == "joint":
            pcs_X_joint = torch.cat(pcs_X, dim=1)
            pcs_F_joint = torch.cat(pcs_F, dim=1)
            pcs_RGB_joint = torch.cat(pcs_rgb, dim=1)
            pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

        # Rasterize and Blend
        for i in range(n_views):
            if self.pointcloud_source == "other":
                # get joint for all values except the one
                pcs_X_joint = torch.cat(pcs_X[0:i] + pcs_X[i + 1: n_views], dim=1)
                pcs_F_joint = torch.cat(pcs_F[0:i] + pcs_F[i + 1: n_views], dim=1)
                pcs_RGB_joint = torch.cat(
                    pcs_rgb[0:i] + pcs_rgb[i + 1: n_views], dim=1
                )
                pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

            if i > 0:
                rot_joint_X = transform_points_Rt(pcs_X_joint, vps[i - 1])
                rot_joint_X = points_to_ndc(rot_joint_X, K, (H, W))
            else:
                rot_joint_X = points_to_ndc(pcs_X_joint, K, (H, W))
            projs.append(self.renderer(rot_joint_X, pcs_FRGB_joint))

        # Decode
        for i in range(n_views):
            proj_FRGB_i = projs[i]["feats"]
            proj_RGB_i = proj_FRGB_i[:, -3:]
            proj_F_i = proj_FRGB_i[:, :-3]

            output[f"rgb_render_{i}"] = proj_RGB_i
            output[f"ras_depth_{i}"] = projs[i]["depth"]
            output[f"cover_{i}"] = projs[i]["mask"].unsqueeze(1)  # useless

        return output

    def gather_p2i(self, feat_p, idx):
        feat_p = torch.cat((feat_p, torch.zeros_like(feat_p[:1, :])), 0)
        return feat_p[idx]

    def gather_i2p(self, feat_i, idx):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        feat_i2p = []
        for i in range(B):
            feat_i2p.append(src_feat_i[i].reshape(C,H*W).permute(1,0))
            feat_i2p.append(tgt_feat_i[i].reshape(C,H*W).permute(1,0))

        feat_i2p = torch.cat(feat_i2p, 0)
        feat_i2p = torch.cat((feat_i2p, torch.zeros_like(feat_i2p[:1, :])), 0)
        return feat_i2p[idx]  #N,16,C

    def fusep2i(self, feat_i, feat_p2i, layer):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        # src_feat_i = src_feat_i.reshape(B, C, H * W).permute(0, 2, 1)
        # tgt_feat_i = tgt_feat_i.reshape(B, C, H * W).permute(0, 2, 1)

        src_feat_p2i = []
        tgt_feat_p2i = []

        for i in range(2*B):
            if i % 2 == 0:
                src_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))
            else:
                tgt_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))

        src_feat_p2i = torch.vstack(src_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)
        tgt_feat_p2i = torch.vstack(tgt_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)

        src_feat_p2i = torch.cat([src_feat_i, src_feat_p2i], 1)
        tgt_feat_p2i = torch.cat([tgt_feat_i, tgt_feat_p2i], 1)

        src_feat_p2i = layer(src_feat_p2i)
        tgt_feat_p2i = layer(tgt_feat_p2i)

        return [src_feat_p2i, tgt_feat_p2i]

    def fusei2p(self, feat_p, feat_i2p, layer):
        feat_i2p = torch.cat([feat_p, feat_i2p], -1)
        feat_i2p = layer(feat_i2p)
        return feat_i2p


    def generate_pointclouds(self, K, deps, vps=None):
        n_views = len(deps)
        # generate pointclouds - generate grid once for efficiency
        B, _, H, W = deps[0].shape
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pcs_X = [
            grid_to_pointcloud(K_inv, deps[i], None, grid)[0] for i in range(n_views)
        ]

        if vps is not None:
            pcs_X_rot = [
                transform_points_Rt(pcs_X[i + 1], vps[i + 1], inverse=True, )
                for i in range(n_views - 1)
            ]
            pcs_X = pcs_X[0:1] + pcs_X_rot
            pcs_X = torch.cat(pcs_X, dim=1).detach().cpu()

        return pcs_X

    def get_feature_pcs(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert (
                feats[0].shape[-1] == deps[0].shape[-1]
        ), f"Same size {feats[0].shape} - {deps[0].shape}"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]
        return pcs_X, pcs_F, None


class PCReg_KPURes18_MSF_loftr2(nn.Module):
    def __init__(self, cfg):
        super(PCReg_KPURes18_MSF_loftr2, self).__init__()
        # set encoder decoder
        chan_in = 3
        self.cfg = cfg
        feat_dim = cfg.feat_dim

        # No imagenet pretraining
        pretrained = False

        encode_I = URes18Encoder1(chan_in, feat_dim, pretrained)
        encode_P = KPFCN(cfg)
        self.cnn_pre_stages = nn.Sequential(
            encode_I.inconv,
            encode_I.layer1
        )
        self.pcd_pre_stages = nn.ModuleList()
        for i in range(0,2):
            self.pcd_pre_stages.append(encode_P.encoder_blocks[i])

        self.cnn_ds_0 = encode_I.layer2
        self.pcd_ds_0 = nn.ModuleList()
        for i in range(2,5):
            self.pcd_ds_0.append(encode_P.encoder_blocks[i])

        self.cnn_ds_1 = encode_I.layer3
        self.pcd_ds_1 = nn.ModuleList()
        for i in range(5,8):
            self.pcd_ds_1.append(encode_P.encoder_blocks[i])

        self.cnn_up_0 = encode_I.up1
        self.pcd_up_0 = nn.ModuleList()
        for i in range(0, 2):
            self.pcd_up_0.append(encode_P.decoder_blocks[i])

        self.cnn_up_1 = nn.Sequential(
            encode_I.up2,
            encode_I.outconv
        )
        self.pcd_up_1 = nn.ModuleList()
        for i in range(2, 4):
            self.pcd_up_1.append(encode_P.decoder_blocks[i])

        # 0
        self.fuse_p2i_ds_0 = nn.ModuleList()
        self.fuse_p2i_ds_0.append(
            UnaryBlock(128, 64, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_0.append(
            nn.Sequential(
                nn.Conv2d(64*2, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_0 = nn.ModuleList()
        self.fuse_i2p_ds_0.append(
            UnaryBlock(64, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_0.append(
            UnaryBlock(128 * 2, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 1
        self.fuse_p2i_ds_1 = nn.ModuleList()
        self.fuse_p2i_ds_1.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_1.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_1 = nn.ModuleList()
        self.fuse_i2p_ds_1.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_1.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 2
        self.fuse_p2i_ds_2 = nn.ModuleList()
        self.fuse_p2i_ds_2.append(
            UnaryBlock(512, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_2.append(
            nn.Sequential(
                nn.Conv2d(256 * 2, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_2 = nn.ModuleList()
        self.fuse_i2p_ds_2.append(
            UnaryBlock(256, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_2.append(
            UnaryBlock(512 * 2, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        self.fuse_p2i_up_0 = nn.ModuleList()
        self.fuse_p2i_up_0.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_up_0.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
        ))
        self.fuse_i2p_up_0 = nn.ModuleList()
        self.fuse_i2p_up_0.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_up_0.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # self.fuse_p2i_up_1 = nn.ModuleList()
        # self.fuse_i2p_up_1 = nn.ModuleList()
        self.align_cfg = cfg
        self.renderer = PointsRenderer(cfg)
        self.num_corres = cfg.num_correspodances
        self.pointcloud_source = cfg.pointcloud_source
        self.map = Fusion_CATL(feat_dim)
        
        self.position_encoding = PositionEncodingSine(d_model=32)
        self.loftr_coarse = LocalFeatureTransformer({'d_model':32,
                'nhead':8,
                'layer_names':['self', 'cross'] * 2,
                'attention':'linear'})
        

    def forward(self, batch, rgbs, K, deps, vps=None):
        # Estimate Depth -- now for 1 and 2
        n_views = 2
        output = {}
        B, _, H, W = rgbs[0].shape

        # Encode features
        feat_p_encode = []
        feat_i_encode = []

        #pre stage
        feat_p = batch['features'].clone().detach()
        for block_op in self.pcd_pre_stages:
            feat_p = block_op(feat_p, batch)# feat_p (130982,128)
            
        
        feat_i = [self.cnn_pre_stages(rgbs[i]) for i in range(n_views)] # feat_i [2](4,64,128,128)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][0].squeeze())
        feat_p2i = self.fuse_p2i_ds_0[0](feat_p2i)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_0[1])

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][0])
        feat_i2p = self.fuse_i2p_ds_0[0](feat_i2p.max(1)[0])
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_0[1])

        feat_p = feat_p + feat_i2p
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]

        feat_p_encode.append(feat_p)
        feat_i_encode.append(feat_i)

        #downsample 0
        for block_op in self.pcd_ds_0:
            feat_p = block_op(feat_p, batch)
        feat_i = [self.cnn_ds_0(feat_i[i]) for i in range(n_views)]

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())
        feat_p2i = self.fuse_p2i_ds_1[0](feat_p2i)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_1[1])

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])
        feat_i2p = self.fuse_i2p_ds_1[0](feat_i2p.max(1)[0])
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_1[1])

        feat_p = feat_p + feat_i2p
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]

        feat_p_encode.append(feat_p)
        feat_i_encode.append(feat_i)

        #downsample 1
        for block_op in self.pcd_ds_1:
            feat_p = block_op(feat_p, batch)
        feat_i = [self.cnn_ds_1(feat_i[i]) for i in range(n_views)]

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][2].squeeze())
        feat_p2i = self.fuse_p2i_ds_2[0](feat_p2i)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_2[1])

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][2])
        feat_i2p = self.fuse_i2p_ds_2[0](feat_i2p.max(1)[0])
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_2[1])

        feat_p = feat_p + feat_i2p
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]


        # upsample0
        for block_i, block_op in enumerate(self.pcd_up_0):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)

        feat_i = [
            self.cnn_up_0(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-1][i]), dim=1)) for i in range(n_views)]

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())
        feat_p2i = self.fuse_p2i_up_0[0](feat_p2i)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_up_0[1])

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])
        feat_i2p = self.fuse_i2p_up_0[0](feat_i2p.max(1)[0])
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_up_0[1])

        feat_p = feat_p + feat_i2p
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]

        # upsample1
        for block_i, block_op in enumerate(self.pcd_up_1):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)

        feat_i = [
            self.cnn_up_1(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-2][i]), dim=1)) for i in range(n_views)]

        pcs_X = batch['points_img']     #[2](4,16384,3)
        pcs_F = self.map(feat_i, feat_p, batch) #[2](4,16384,32)
        
        #-----------------
        #修改此处
        pcs_F_sc = list(self.loftr_coarse(pcs_F[0],pcs_F[1]))
        pcs_F = pcs_F+pcs_F_sc
        
        if vps is not None:
            # Drop first viewpoint -- assumed to be identity transformation
            vps = vps[1:]
        elif self.align_cfg.algorithm == "weighted_procrustes":
            vps = []
            cor_loss = []
            for i in range(1, n_views):
                corr_i = get_correspondences(
                    P1=pcs_F[0],
                    P2=pcs_F[i],
                    P1_X=pcs_X[0],
                    P2_X=pcs_X[i],
                    num_corres=self.num_corres,
                    ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
                )
                Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

                vps.append(Rt_i)
                cor_loss.append(cor_loss_i)

                # add for visualization
                output[f"corres_0{i}"] = corr_i
                output[f"vp_{i}"] = Rt_i
        else:
            raise ValueError(f"How to align using {self.align_cfg.algorithm}?")

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        # Get RGB pointcloud as well for direct rendering
        pcs_rgb = [rgb.view(B, 3, -1).permute(0, 2, 1).contiguous() for rgb in rgbs]

        projs = []
        # get joint for all values
        if self.pointcloud_source == "joint":
            pcs_X_joint = torch.cat(pcs_X, dim=1)
            pcs_F_joint = torch.cat(pcs_F, dim=1)
            pcs_RGB_joint = torch.cat(pcs_rgb, dim=1)
            pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

        # Rasterize and Blend
        for i in range(n_views):
            if self.pointcloud_source == "other":
                # get joint for all values except the one
                pcs_X_joint = torch.cat(pcs_X[0:i] + pcs_X[i + 1: n_views], dim=1)
                pcs_F_joint = torch.cat(pcs_F[0:i] + pcs_F[i + 1: n_views], dim=1)
                pcs_RGB_joint = torch.cat(
                    pcs_rgb[0:i] + pcs_rgb[i + 1: n_views], dim=1
                )
                pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

            if i > 0:
                rot_joint_X = transform_points_Rt(pcs_X_joint, vps[i - 1])
                rot_joint_X = points_to_ndc(rot_joint_X, K, (H, W))
            else:
                rot_joint_X = points_to_ndc(pcs_X_joint, K, (H, W))
            projs.append(self.renderer(rot_joint_X, pcs_FRGB_joint))

        # Decode
        for i in range(n_views):
            proj_FRGB_i = projs[i]["feats"]
            proj_RGB_i = proj_FRGB_i[:, -3:]
            proj_F_i = proj_FRGB_i[:, :-3]

            output[f"rgb_render_{i}"] = proj_RGB_i
            output[f"ras_depth_{i}"] = projs[i]["depth"]
            output[f"cover_{i}"] = projs[i]["mask"].unsqueeze(1)  # useless

        return output

    def gather_p2i(self, feat_p, idx):
        feat_p = torch.cat((feat_p, torch.zeros_like(feat_p[:1, :])), 0)
        return feat_p[idx]

    def gather_i2p(self, feat_i, idx):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        feat_i2p = []
        for i in range(B):
            feat_i2p.append(src_feat_i[i].reshape(C,H*W).permute(1,0))
            feat_i2p.append(tgt_feat_i[i].reshape(C,H*W).permute(1,0))

        feat_i2p = torch.cat(feat_i2p, 0)
        feat_i2p = torch.cat((feat_i2p, torch.zeros_like(feat_i2p[:1, :])), 0)
        return feat_i2p[idx]  #N,16,C

    def fusep2i(self, feat_i, feat_p2i, layer):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        # src_feat_i = src_feat_i.reshape(B, C, H * W).permute(0, 2, 1)
        # tgt_feat_i = tgt_feat_i.reshape(B, C, H * W).permute(0, 2, 1)

        src_feat_p2i = []
        tgt_feat_p2i = []

        for i in range(2*B):
            if i % 2 == 0:
                src_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))
            else:
                tgt_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))

        src_feat_p2i = torch.vstack(src_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)
        tgt_feat_p2i = torch.vstack(tgt_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)

        src_feat_p2i = torch.cat([src_feat_i, src_feat_p2i], 1)
        tgt_feat_p2i = torch.cat([tgt_feat_i, tgt_feat_p2i], 1)

        src_feat_p2i = layer(src_feat_p2i)
        tgt_feat_p2i = layer(tgt_feat_p2i)

        return [src_feat_p2i, tgt_feat_p2i]

    def fusei2p(self, feat_p, feat_i2p, layer):
        feat_i2p = torch.cat([feat_p, feat_i2p], -1)
        feat_i2p = layer(feat_i2p)
        return feat_i2p


    def generate_pointclouds(self, K, deps, vps=None):
        n_views = len(deps)
        # generate pointclouds - generate grid once for efficiency
        B, _, H, W = deps[0].shape
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pcs_X = [
            grid_to_pointcloud(K_inv, deps[i], None, grid)[0] for i in range(n_views)
        ]

        if vps is not None:
            pcs_X_rot = [
                transform_points_Rt(pcs_X[i + 1], vps[i + 1], inverse=True, )
                for i in range(n_views - 1)
            ]
            pcs_X = pcs_X[0:1] + pcs_X_rot
            pcs_X = torch.cat(pcs_X, dim=1).detach().cpu()

        return pcs_X

    def get_feature_pcs(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert (
                feats[0].shape[-1] == deps[0].shape[-1]
        ), f"Same size {feats[0].shape} - {deps[0].shape}"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]
        return pcs_X, pcs_F, None


class PCReg_KPURes18_MSF_loftr3(nn.Module):
    def __init__(self, cfg):
        super(PCReg_KPURes18_MSF_loftr3, self).__init__()
        # set encoder decoder
        chan_in = 3
        self.cfg = cfg
        feat_dim = cfg.feat_dim

        # No imagenet pretraining
        pretrained = False

        encode_I = URes18Encoder1(chan_in, feat_dim, pretrained)
        encode_P = KPFCN(cfg)
        self.cnn_pre_stages = nn.Sequential(
            encode_I.inconv,
            encode_I.layer1
        )
        self.pcd_pre_stages = nn.ModuleList()
        for i in range(0,2):
            self.pcd_pre_stages.append(encode_P.encoder_blocks[i])

        self.cnn_ds_0 = encode_I.layer2
        self.pcd_ds_0 = nn.ModuleList()
        for i in range(2,5):
            self.pcd_ds_0.append(encode_P.encoder_blocks[i])

        self.cnn_ds_1 = encode_I.layer3
        self.pcd_ds_1 = nn.ModuleList()
        for i in range(5,8):
            self.pcd_ds_1.append(encode_P.encoder_blocks[i])

        self.cnn_up_0 = encode_I.up1
        self.pcd_up_0 = nn.ModuleList()
        for i in range(0, 2):
            self.pcd_up_0.append(encode_P.decoder_blocks[i])

        self.cnn_up_1 = nn.Sequential(
            encode_I.up2,
            encode_I.outconv
        )
        self.pcd_up_1 = nn.ModuleList()
        for i in range(2, 4):
            self.pcd_up_1.append(encode_P.decoder_blocks[i])

        # 0
        self.fuse_p2i_ds_0 = nn.ModuleList()
        self.fuse_p2i_ds_0.append(
            UnaryBlock(128, 64, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_0.append(
            nn.Sequential(
                nn.Conv2d(64*2, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_0 = nn.ModuleList()
        self.fuse_i2p_ds_0.append(
            UnaryBlock(64, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_0.append(
            UnaryBlock(128 * 2, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 1
        self.fuse_p2i_ds_1 = nn.ModuleList()
        self.fuse_p2i_ds_1.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_1.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_1 = nn.ModuleList()
        self.fuse_i2p_ds_1.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_1.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 2
        self.fuse_p2i_ds_2 = nn.ModuleList()
        self.fuse_p2i_ds_2.append(
            UnaryBlock(512, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_2.append(
            nn.Sequential(
                nn.Conv2d(256 * 2, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_2 = nn.ModuleList()
        self.fuse_i2p_ds_2.append(
            UnaryBlock(256, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_2.append(
            UnaryBlock(512 * 2, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        self.fuse_p2i_up_0 = nn.ModuleList()
        self.fuse_p2i_up_0.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_up_0.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
        ))
        self.fuse_i2p_up_0 = nn.ModuleList()
        self.fuse_i2p_up_0.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_up_0.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # self.fuse_p2i_up_1 = nn.ModuleList()
        # self.fuse_i2p_up_1 = nn.ModuleList()
        self.align_cfg = cfg
        self.renderer = PointsRenderer(cfg)
        self.num_corres = cfg.num_correspodances
        self.pointcloud_source = cfg.pointcloud_source
        self.map = Fusion_CATL(feat_dim)
        
        self.position_encoding = PositionEncodingSine(d_model=64)
        self.loftr_coarse = LocalFeatureTransformer({'d_model':32,
                'nhead':8,
                'layer_names':['self', 'cross'] * 2,
                'attention':'linear'})
        

    def forward(self, batch, rgbs, K, deps, vps=None):
        # Estimate Depth -- now for 1 and 2
        n_views = 2
        output = {}
        B, _, H, W = rgbs[0].shape

        # Encode features
        feat_p_encode = []
        feat_i_encode = []

        #pre stage
        feat_p = batch['features'].clone().detach()
        for block_op in self.pcd_pre_stages:
            feat_p = block_op(feat_p, batch)# feat_p (130982,128)
            
        
        feat_i = [self.cnn_pre_stages(rgbs[i]) for i in range(n_views)] # feat_i [2](4,64,128,128)
        feat_i = [self.position_encoding(feat_i[i]) for i in range(n_views)]

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][0].squeeze())
        feat_p2i = self.fuse_p2i_ds_0[0](feat_p2i)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_0[1])

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][0])
        feat_i2p = self.fuse_i2p_ds_0[0](feat_i2p.max(1)[0])
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_0[1])

        feat_p = feat_p + feat_i2p
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]

        feat_p_encode.append(feat_p)
        feat_i_encode.append(feat_i)

        #downsample 0
        for block_op in self.pcd_ds_0:
            feat_p = block_op(feat_p, batch)
        feat_i = [self.cnn_ds_0(feat_i[i]) for i in range(n_views)]

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())
        feat_p2i = self.fuse_p2i_ds_1[0](feat_p2i)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_1[1])

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])
        feat_i2p = self.fuse_i2p_ds_1[0](feat_i2p.max(1)[0])
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_1[1])

        feat_p = feat_p + feat_i2p
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]

        feat_p_encode.append(feat_p)
        feat_i_encode.append(feat_i)

        #downsample 1
        for block_op in self.pcd_ds_1:
            feat_p = block_op(feat_p, batch)
        feat_i = [self.cnn_ds_1(feat_i[i]) for i in range(n_views)]

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][2].squeeze())
        feat_p2i = self.fuse_p2i_ds_2[0](feat_p2i)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_2[1])

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][2])
        feat_i2p = self.fuse_i2p_ds_2[0](feat_i2p.max(1)[0])
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_2[1])

        feat_p = feat_p + feat_i2p
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]


        # upsample0
        for block_i, block_op in enumerate(self.pcd_up_0):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)

        feat_i = [
            self.cnn_up_0(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-1][i]), dim=1)) for i in range(n_views)]

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())
        feat_p2i = self.fuse_p2i_up_0[0](feat_p2i)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_up_0[1])

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])
        feat_i2p = self.fuse_i2p_up_0[0](feat_i2p.max(1)[0])
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_up_0[1])

        feat_p = feat_p + feat_i2p
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]

        # upsample1
        for block_i, block_op in enumerate(self.pcd_up_1):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)

        feat_i = [
            self.cnn_up_1(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-2][i]), dim=1)) for i in range(n_views)]

        pcs_X = batch['points_img']     #[2](4,16384,3)
        pcs_F = self.map(feat_i, feat_p, batch) #[2](4,16384,32)
        
        #-----------------
        #修改此处
        pcs_F_sc = list(self.loftr_coarse(pcs_F[0],pcs_F[1]))
        pcs_F = pcs_F+pcs_F_sc
        
        if vps is not None:
            # Drop first viewpoint -- assumed to be identity transformation
            vps = vps[1:]
        elif self.align_cfg.algorithm == "weighted_procrustes":
            vps = []
            cor_loss = []
            for i in range(1, n_views):
                corr_i = get_correspondences(
                    P1=pcs_F[0],
                    P2=pcs_F[i],
                    P1_X=pcs_X[0],
                    P2_X=pcs_X[i],
                    num_corres=self.num_corres,
                    ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
                )
                Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

                vps.append(Rt_i)
                cor_loss.append(cor_loss_i)

                # add for visualization
                output[f"corres_0{i}"] = corr_i
                output[f"vp_{i}"] = Rt_i
        else:
            raise ValueError(f"How to align using {self.align_cfg.algorithm}?")

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        # Get RGB pointcloud as well for direct rendering
        pcs_rgb = [rgb.view(B, 3, -1).permute(0, 2, 1).contiguous() for rgb in rgbs]

        projs = []
        # get joint for all values
        if self.pointcloud_source == "joint":
            pcs_X_joint = torch.cat(pcs_X, dim=1)
            pcs_F_joint = torch.cat(pcs_F, dim=1)
            pcs_RGB_joint = torch.cat(pcs_rgb, dim=1)
            pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

        # Rasterize and Blend
        for i in range(n_views):
            if self.pointcloud_source == "other":
                # get joint for all values except the one
                pcs_X_joint = torch.cat(pcs_X[0:i] + pcs_X[i + 1: n_views], dim=1)
                pcs_F_joint = torch.cat(pcs_F[0:i] + pcs_F[i + 1: n_views], dim=1)
                pcs_RGB_joint = torch.cat(
                    pcs_rgb[0:i] + pcs_rgb[i + 1: n_views], dim=1
                )
                pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

            if i > 0:
                rot_joint_X = transform_points_Rt(pcs_X_joint, vps[i - 1])
                rot_joint_X = points_to_ndc(rot_joint_X, K, (H, W))
            else:
                rot_joint_X = points_to_ndc(pcs_X_joint, K, (H, W))
            projs.append(self.renderer(rot_joint_X, pcs_FRGB_joint))

        # Decode
        for i in range(n_views):
            proj_FRGB_i = projs[i]["feats"]
            proj_RGB_i = proj_FRGB_i[:, -3:]
            proj_F_i = proj_FRGB_i[:, :-3]

            output[f"rgb_render_{i}"] = proj_RGB_i
            output[f"ras_depth_{i}"] = projs[i]["depth"]
            output[f"cover_{i}"] = projs[i]["mask"].unsqueeze(1)  # useless

        return output

    def gather_p2i(self, feat_p, idx):
        feat_p = torch.cat((feat_p, torch.zeros_like(feat_p[:1, :])), 0)
        return feat_p[idx]

    def gather_i2p(self, feat_i, idx):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        feat_i2p = []
        for i in range(B):
            feat_i2p.append(src_feat_i[i].reshape(C,H*W).permute(1,0))
            feat_i2p.append(tgt_feat_i[i].reshape(C,H*W).permute(1,0))

        feat_i2p = torch.cat(feat_i2p, 0)
        feat_i2p = torch.cat((feat_i2p, torch.zeros_like(feat_i2p[:1, :])), 0)
        return feat_i2p[idx]  #N,16,C

    def fusep2i(self, feat_i, feat_p2i, layer):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        # src_feat_i = src_feat_i.reshape(B, C, H * W).permute(0, 2, 1)
        # tgt_feat_i = tgt_feat_i.reshape(B, C, H * W).permute(0, 2, 1)

        src_feat_p2i = []
        tgt_feat_p2i = []

        for i in range(2*B):
            if i % 2 == 0:
                src_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))
            else:
                tgt_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))

        src_feat_p2i = torch.vstack(src_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)
        tgt_feat_p2i = torch.vstack(tgt_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)

        src_feat_p2i = torch.cat([src_feat_i, src_feat_p2i], 1)
        tgt_feat_p2i = torch.cat([tgt_feat_i, tgt_feat_p2i], 1)

        src_feat_p2i = layer(src_feat_p2i)
        tgt_feat_p2i = layer(tgt_feat_p2i)

        return [src_feat_p2i, tgt_feat_p2i]

    def fusei2p(self, feat_p, feat_i2p, layer):
        feat_i2p = torch.cat([feat_p, feat_i2p], -1)
        feat_i2p = layer(feat_i2p)
        return feat_i2p


    def generate_pointclouds(self, K, deps, vps=None):
        n_views = len(deps)
        # generate pointclouds - generate grid once for efficiency
        B, _, H, W = deps[0].shape
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pcs_X = [
            grid_to_pointcloud(K_inv, deps[i], None, grid)[0] for i in range(n_views)
        ]

        if vps is not None:
            pcs_X_rot = [
                transform_points_Rt(pcs_X[i + 1], vps[i + 1], inverse=True, )
                for i in range(n_views - 1)
            ]
            pcs_X = pcs_X[0:1] + pcs_X_rot
            pcs_X = torch.cat(pcs_X, dim=1).detach().cpu()

        return pcs_X

    def get_feature_pcs(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert (
                feats[0].shape[-1] == deps[0].shape[-1]
        ), f"Same size {feats[0].shape} - {deps[0].shape}"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]
        return pcs_X, pcs_F, None


class PCReg_KPURes18_MSF_loftr4(nn.Module):
    def __init__(self, cfg):
        super(PCReg_KPURes18_MSF_loftr4, self).__init__()
        # set encoder decoder
        chan_in = 3
        self.cfg = cfg
        feat_dim = cfg.feat_dim

        # No imagenet pretraining
        pretrained = False

        encode_I = URes18Encoder1(chan_in, feat_dim, pretrained)
        encode_P = KPFCN(cfg)
        self.cnn_pre_stages = nn.Sequential(
            encode_I.inconv,
            encode_I.layer1
        )
        self.pcd_pre_stages = nn.ModuleList()
        for i in range(0,2):
            self.pcd_pre_stages.append(encode_P.encoder_blocks[i])

        self.cnn_ds_0 = encode_I.layer2
        self.pcd_ds_0 = nn.ModuleList()
        for i in range(2,5):
            self.pcd_ds_0.append(encode_P.encoder_blocks[i])

        self.cnn_ds_1 = encode_I.layer3
        self.pcd_ds_1 = nn.ModuleList()
        for i in range(5,8):
            self.pcd_ds_1.append(encode_P.encoder_blocks[i])

        self.cnn_up_0 = encode_I.up1
        self.pcd_up_0 = nn.ModuleList()
        for i in range(0, 2):
            self.pcd_up_0.append(encode_P.decoder_blocks[i])

        self.cnn_up_1 = nn.Sequential(
            encode_I.up2,
            encode_I.outconv
        )
        self.pcd_up_1 = nn.ModuleList()
        for i in range(2, 4):
            self.pcd_up_1.append(encode_P.decoder_blocks[i])

        # 0
        self.fuse_p2i_ds_0 = nn.ModuleList()
        self.fuse_p2i_ds_0.append(
            UnaryBlock(128, 64, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_0.append(
            nn.Sequential(
                nn.Conv2d(64*2, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_0 = nn.ModuleList()
        self.fuse_i2p_ds_0.append(
            UnaryBlock(64, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_0.append(
            UnaryBlock(128 * 2, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 1
        self.fuse_p2i_ds_1 = nn.ModuleList()
        self.fuse_p2i_ds_1.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_1.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_1 = nn.ModuleList()
        self.fuse_i2p_ds_1.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_1.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 2
        self.fuse_p2i_ds_2 = nn.ModuleList()
        self.fuse_p2i_ds_2.append(
            UnaryBlock(512, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_2.append(
            nn.Sequential(
                nn.Conv2d(256 * 2, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_2 = nn.ModuleList()
        self.fuse_i2p_ds_2.append(
            UnaryBlock(256, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_2.append(
            UnaryBlock(512 * 2, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        self.fuse_p2i_up_0 = nn.ModuleList()
        self.fuse_p2i_up_0.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_up_0.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
        ))
        self.fuse_i2p_up_0 = nn.ModuleList()
        self.fuse_i2p_up_0.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_up_0.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # self.fuse_p2i_up_1 = nn.ModuleList()
        # self.fuse_i2p_up_1 = nn.ModuleList()
        self.align_cfg = cfg
        self.renderer = PointsRenderer(cfg)
        self.num_corres = cfg.num_correspodances
        self.pointcloud_source = cfg.pointcloud_source
        self.map = Fusion_CATL(feat_dim)
        
        self.loftr_coarse1 = LocalFeatureTransformer({'d_model':32,
                'nhead':8,
                'layer_names':['self', 'cross'] * 1,
                'attention':'linear'})
        self.loftr_coarse2 = LocalFeatureTransformer({'d_model':256,
                'nhead':8,
                'layer_names':['self', 'cross'] * 1,
                'attention':'linear'})
        self.loftr_coarse3 = LocalFeatureTransformer({'d_model':32,
                'nhead':8,
                'layer_names':['self', 'cross'] * 1,
                'attention':'linear'})
        self.loftr_coarse4 = LocalFeatureTransformer({'d_model':32,
                'nhead':8,
                'layer_names':['self', 'cross'] * 4,
                'attention':'linear'})

    def forward(self, batch, rgbs, K, deps, vps=None):
        # Estimate Depth -- now for 1 and 2
        n_views = 2
        output = {}
        B, _, H, W = rgbs[0].shape

        # Encode features
        feat_p_encode = []
        feat_i_encode = []

        #pre stage
        feat_p = batch['features'].clone().detach() #全为1 (130982,1)
        for block_op in self.pcd_pre_stages:
            feat_p = block_op(feat_p, batch)# feat_p (130982,128)
            
        
        feat_i = [self.cnn_pre_stages(rgbs[i]) for i in range(n_views)] # feat_i [2](4,64,128,128)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][0].squeeze())#(131072,128)
        feat_p2i = self.fuse_p2i_ds_0[0](feat_p2i)#(131072,64)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_0[1])# [2](4,64,128,128)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][0])#(130982,32,64)
        feat_i2p = self.fuse_i2p_ds_0[0](feat_i2p.max(1)[0])#(130982,128)类似于pointnet的最大池化
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_0[1])#(130982,128)

        feat_p = feat_p + feat_i2p#(130982,128)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,64,128,128)
        
        

        feat_p_encode.append(feat_p)# [1](130982,128)
        feat_i_encode.append(feat_i)# [1][2](4,64,128,128)

        #downsample 0
        for block_op in self.pcd_ds_0:
            feat_p = block_op(feat_p, batch)# (17420,256)
        feat_i = [self.cnn_ds_0(feat_i[i]) for i in range(n_views)]# [2](4,128,64,64)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())#(32768,256)
        feat_p2i = self.fuse_p2i_ds_1[0](feat_p2i)#(32768,128)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_1[1])# [2](4,128,64,64)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])# (17420,32,128)
        feat_i2p = self.fuse_i2p_ds_1[0](feat_i2p.max(1)[0])# (17420,256)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_1[1])# (17420,256)

        feat_p = feat_p + feat_i2p# (17420,256)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,128,64,64)

        feat_p_encode.append(feat_p)# [(130982,128)(17420,256)]
        feat_i_encode.append(feat_i)# [[2](4,64,128,128),[2](4,128,64,64)]

        #downsample 1
        for block_op in self.pcd_ds_1:
            feat_p = block_op(feat_p, batch)# (5895,512)
        feat_i = [self.cnn_ds_1(feat_i[i]) for i in range(n_views)]# [2](4,256,32,32)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][2].squeeze())#(8192,512)
        feat_p2i = self.fuse_p2i_ds_2[0](feat_p2i)#(8192,256)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_2[1])# [2](4,256,32,32)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][2])#(5895,32,256)
        feat_i2p = self.fuse_i2p_ds_2[0](feat_i2p.max(1)[0])#(5895,512)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_2[1])#(5895,512)

        feat_p = feat_p + feat_i2p#(5895,512)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,256,32,32)

        
        feat_i[0] = feat_i[0].reshape(B,256,1024).permute(0,2,1)
        feat_i[1] = feat_i[1].reshape(B,256,1024).permute(0,2,1)
        feat_i[0], feat_i[1]= self.loftr_coarse2(feat_i[0],feat_i[1])
        feat_i[0] = feat_i[0].permute(0,2,1).reshape(B,256,32,32)
        feat_i[1] = feat_i[1].permute(0,2,1).reshape(B,256,32,32)

        # upsample0
        for block_i, block_op in enumerate(self.pcd_up_0):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)#(17420,256)

        feat_i = [
            self.cnn_up_0(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-1][i]), dim=1)) for i in range(n_views)]#[2](4,128,64,64)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())#(32768,256)
        feat_p2i = self.fuse_p2i_up_0[0](feat_p2i)#(32768,128)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_up_0[1])#[2](4,128,64,64)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])# (17420,32,128)
        feat_i2p = self.fuse_i2p_up_0[0](feat_i2p.max(1)[0])# (17420,256)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_up_0[1])# (17420,256)

        feat_p = feat_p + feat_i2p # (17420,256)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]#[2](4,128,64,64)

        # upsample1
        for block_i, block_op in enumerate(self.pcd_up_1):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)#(130982,32)

        feat_i = [
            self.cnn_up_1(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-2][i]), dim=1)) for i in range(n_views)]#[2](4,32,128,128)

        pcs_X = batch['points_img']     #[2](4,16384,3)
        pcs_F = self.map(feat_i, feat_p, batch) #[2](4,16384,32)

        pcs_F_sc4 = list(self.loftr_coarse4(pcs_F[0],pcs_F[1]))
        pcs_F = pcs_F+pcs_F_sc4
        
        if vps is not None:
            # Drop first viewpoint -- assumed to be identity transformation
            vps = vps[1:]
        elif self.align_cfg.algorithm == "weighted_procrustes":
            vps = []
            cor_loss = []
            for i in range(1, n_views):
                corr_i = get_correspondences(
                    P1=pcs_F[0],
                    P2=pcs_F[i],
                    P1_X=pcs_X[0],
                    P2_X=pcs_X[i],
                    num_corres=self.num_corres,
                    ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
                )
                Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

                vps.append(Rt_i)
                cor_loss.append(cor_loss_i)

                # add for visualization
                output[f"corres_0{i}"] = corr_i
                output[f"vp_{i}"] = Rt_i
        else:
            raise ValueError(f"How to align using {self.align_cfg.algorithm}?")

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        # Get RGB pointcloud as well for direct rendering
        pcs_rgb = [rgb.view(B, 3, -1).permute(0, 2, 1).contiguous() for rgb in rgbs]

        projs = []
        # get joint for all values
        if self.pointcloud_source == "joint":
            pcs_X_joint = torch.cat(pcs_X, dim=1)
            pcs_F_joint = torch.cat(pcs_F, dim=1)
            pcs_RGB_joint = torch.cat(pcs_rgb, dim=1)
            pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

        # Rasterize and Blend
        for i in range(n_views):
            if self.pointcloud_source == "other":
                # get joint for all values except the one
                pcs_X_joint = torch.cat(pcs_X[0:i] + pcs_X[i + 1: n_views], dim=1)
                pcs_F_joint = torch.cat(pcs_F[0:i] + pcs_F[i + 1: n_views], dim=1)
                pcs_RGB_joint = torch.cat(
                    pcs_rgb[0:i] + pcs_rgb[i + 1: n_views], dim=1
                )
                pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

            if i > 0:
                rot_joint_X = transform_points_Rt(pcs_X_joint, vps[i - 1])
                rot_joint_X = points_to_ndc(rot_joint_X, K, (H, W))
            else:
                rot_joint_X = points_to_ndc(pcs_X_joint, K, (H, W))
            projs.append(self.renderer(rot_joint_X, pcs_FRGB_joint))

        # Decode
        for i in range(n_views):
            proj_FRGB_i = projs[i]["feats"]
            proj_RGB_i = proj_FRGB_i[:, -3:]
            proj_F_i = proj_FRGB_i[:, :-3]

            output[f"rgb_render_{i}"] = proj_RGB_i
            output[f"ras_depth_{i}"] = projs[i]["depth"]
            output[f"cover_{i}"] = projs[i]["mask"].unsqueeze(1)  # useless

        return output

    def gather_p2i(self, feat_p, idx):
        feat_p = torch.cat((feat_p, torch.zeros_like(feat_p[:1, :])), 0)
        return feat_p[idx]

    def gather_i2p(self, feat_i, idx):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        feat_i2p = []
        for i in range(B):
            feat_i2p.append(src_feat_i[i].reshape(C,H*W).permute(1,0))
            feat_i2p.append(tgt_feat_i[i].reshape(C,H*W).permute(1,0))

        feat_i2p = torch.cat(feat_i2p, 0)
        feat_i2p = torch.cat((feat_i2p, torch.zeros_like(feat_i2p[:1, :])), 0)
        return feat_i2p[idx]  #N,16,C

    def fusep2i(self, feat_i, feat_p2i, layer):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        # src_feat_i = src_feat_i.reshape(B, C, H * W).permute(0, 2, 1)
        # tgt_feat_i = tgt_feat_i.reshape(B, C, H * W).permute(0, 2, 1)

        src_feat_p2i = []
        tgt_feat_p2i = []

        for i in range(2*B):
            if i % 2 == 0:
                src_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))#[4](1,16384,64)
            else:
                tgt_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))#[4](1,16384,64)

        src_feat_p2i = torch.vstack(src_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)#(4,64,128,128)
        tgt_feat_p2i = torch.vstack(tgt_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)#(4,64,128,128)

        src_feat_p2i = torch.cat([src_feat_i, src_feat_p2i], 1)#(4,128,128,128)
        tgt_feat_p2i = torch.cat([tgt_feat_i, tgt_feat_p2i], 1)#(4,128,128,128)

        src_feat_p2i = layer(src_feat_p2i)#(4,64,128,128)
        tgt_feat_p2i = layer(tgt_feat_p2i)#(4,64,128,128)

        return [src_feat_p2i, tgt_feat_p2i]

    def fusei2p(self, feat_p, feat_i2p, layer):
        feat_i2p = torch.cat([feat_p, feat_i2p], -1)
        feat_i2p = layer(feat_i2p)
        return feat_i2p


    def generate_pointclouds(self, K, deps, vps=None):
        n_views = len(deps)
        # generate pointclouds - generate grid once for efficiency
        B, _, H, W = deps[0].shape
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pcs_X = [
            grid_to_pointcloud(K_inv, deps[i], None, grid)[0] for i in range(n_views)
        ]

        if vps is not None:
            pcs_X_rot = [
                transform_points_Rt(pcs_X[i + 1], vps[i + 1], inverse=True, )
                for i in range(n_views - 1)
            ]
            pcs_X = pcs_X[0:1] + pcs_X_rot
            pcs_X = torch.cat(pcs_X, dim=1).detach().cpu()

        return pcs_X

    def get_feature_pcs(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert (
                feats[0].shape[-1] == deps[0].shape[-1]
        ), f"Same size {feats[0].shape} - {deps[0].shape}"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]
        return pcs_X, pcs_F, None


class PCReg_KPURes18_MSF_loftr5(nn.Module):
    def __init__(self, cfg):
        super(PCReg_KPURes18_MSF_loftr5, self).__init__()
        # set encoder decoder
        chan_in = 3
        self.cfg = cfg
        feat_dim = cfg.feat_dim

        # No imagenet pretraining
        pretrained = False

        encode_I = URes18Encoder1(chan_in, feat_dim, pretrained)
        encode_P = KPFCN(cfg)
        self.cnn_pre_stages = nn.Sequential(
            encode_I.inconv,
            encode_I.layer1
        )
        self.pcd_pre_stages = nn.ModuleList()
        for i in range(0,2):
            self.pcd_pre_stages.append(encode_P.encoder_blocks[i])

        self.cnn_ds_0 = encode_I.layer2
        self.pcd_ds_0 = nn.ModuleList()
        for i in range(2,5):
            self.pcd_ds_0.append(encode_P.encoder_blocks[i])

        self.cnn_ds_1 = encode_I.layer3
        self.pcd_ds_1 = nn.ModuleList()
        for i in range(5,8):
            self.pcd_ds_1.append(encode_P.encoder_blocks[i])

        self.cnn_up_0 = encode_I.up1
        self.pcd_up_0 = nn.ModuleList()
        for i in range(0, 2):
            self.pcd_up_0.append(encode_P.decoder_blocks[i])

        self.cnn_up_1 = nn.Sequential(
            encode_I.up2,
            encode_I.outconv
        )
        self.pcd_up_1 = nn.ModuleList()
        for i in range(2, 4):
            self.pcd_up_1.append(encode_P.decoder_blocks[i])

        # 0
        self.fuse_p2i_ds_0 = nn.ModuleList()
        self.fuse_p2i_ds_0.append(
            UnaryBlock(128, 64, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_0.append(
            nn.Sequential(
                nn.Conv2d(64*2, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_0 = nn.ModuleList()
        self.fuse_i2p_ds_0.append(
            UnaryBlock(64, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_0.append(
            UnaryBlock(128 * 2, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 1
        self.fuse_p2i_ds_1 = nn.ModuleList()
        self.fuse_p2i_ds_1.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_1.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_1 = nn.ModuleList()
        self.fuse_i2p_ds_1.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_1.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 2
        self.fuse_p2i_ds_2 = nn.ModuleList()
        self.fuse_p2i_ds_2.append(
            UnaryBlock(512, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_2.append(
            nn.Sequential(
                nn.Conv2d(256 * 2, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_2 = nn.ModuleList()
        self.fuse_i2p_ds_2.append(
            UnaryBlock(256, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_2.append(
            UnaryBlock(512 * 2, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        self.fuse_p2i_up_0 = nn.ModuleList()
        self.fuse_p2i_up_0.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_up_0.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
        ))
        self.fuse_i2p_up_0 = nn.ModuleList()
        self.fuse_i2p_up_0.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_up_0.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # self.fuse_p2i_up_1 = nn.ModuleList()
        # self.fuse_i2p_up_1 = nn.ModuleList()
        self.align_cfg = cfg
        self.renderer = PointsRenderer(cfg)
        self.num_corres = cfg.num_correspodances
        self.pointcloud_source = cfg.pointcloud_source
        self.map = Fusion_CATL(feat_dim)
        
        self.loftr_coarse1 = LocalFeatureTransformer({'d_model':32,
                'nhead':8,
                'layer_names':['self', 'cross'] * 1,
                'attention':'linear'})
        self.loftr_coarse2 = LocalFeatureTransformer({'d_model':256,
                'nhead':8,
                'layer_names':['self'] * 4,
                'attention':'linear'})
        self.loftr_coarse3 = LocalFeatureTransformer({'d_model':32,
                'nhead':8,
                'layer_names':['self', 'cross'] * 1,
                'attention':'linear'})
        self.loftr_coarse4 = LocalFeatureTransformer({'d_model':32,
                'nhead':8,
                'layer_names':['self', 'cross'] * 4,
                'attention':'linear'})

    def forward(self, batch, rgbs, K, deps, vps=None):
        # Estimate Depth -- now for 1 and 2
        n_views = 2
        output = {}
        B, _, H, W = rgbs[0].shape

        # Encode features
        feat_p_encode = []
        feat_i_encode = []

        #pre stage
        feat_p = batch['features'].clone().detach() #全为1 (130982,1)
        for block_op in self.pcd_pre_stages:
            feat_p = block_op(feat_p, batch)# feat_p (130982,128)
            
        
        feat_i = [self.cnn_pre_stages(rgbs[i]) for i in range(n_views)] # feat_i [2](4,64,128,128)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][0].squeeze())#(131072,128)
        feat_p2i = self.fuse_p2i_ds_0[0](feat_p2i)#(131072,64)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_0[1])# [2](4,64,128,128)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][0])#(130982,32,64)
        feat_i2p = self.fuse_i2p_ds_0[0](feat_i2p.max(1)[0])#(130982,128)类似于pointnet的最大池化
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_0[1])#(130982,128)

        feat_p = feat_p + feat_i2p#(130982,128)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,64,128,128)
        
        

        feat_p_encode.append(feat_p)# [1](130982,128)
        feat_i_encode.append(feat_i)# [1][2](4,64,128,128)

        #downsample 0
        for block_op in self.pcd_ds_0:
            feat_p = block_op(feat_p, batch)# (17420,256)
        feat_i = [self.cnn_ds_0(feat_i[i]) for i in range(n_views)]# [2](4,128,64,64)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())#(32768,256)
        feat_p2i = self.fuse_p2i_ds_1[0](feat_p2i)#(32768,128)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_1[1])# [2](4,128,64,64)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])# (17420,32,128)
        feat_i2p = self.fuse_i2p_ds_1[0](feat_i2p.max(1)[0])# (17420,256)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_1[1])# (17420,256)

        feat_p = feat_p + feat_i2p# (17420,256)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,128,64,64)

        feat_p_encode.append(feat_p)# [(130982,128)(17420,256)]
        feat_i_encode.append(feat_i)# [[2](4,64,128,128),[2](4,128,64,64)]

        #downsample 1
        for block_op in self.pcd_ds_1:
            feat_p = block_op(feat_p, batch)# (5895,512)
        feat_i = [self.cnn_ds_1(feat_i[i]) for i in range(n_views)]# [2](4,256,32,32)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][2].squeeze())#(8192,512)
        feat_p2i = self.fuse_p2i_ds_2[0](feat_p2i)#(8192,256)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_2[1])# [2](4,256,32,32)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][2])#(5895,32,256)
        feat_i2p = self.fuse_i2p_ds_2[0](feat_i2p.max(1)[0])#(5895,512)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_2[1])#(5895,512)

        feat_p = feat_p + feat_i2p#(5895,512)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,256,32,32)

        feat_i_32 = feat_i
        feat_i[0] = feat_i[0].reshape(B,256,1024).permute(0,2,1)
        feat_i[1] = feat_i[1].reshape(B,256,1024).permute(0,2,1)
        feat_i[0], feat_i[1]= self.loftr_coarse2(feat_i[0],feat_i[1])
        feat_i[0] = feat_i[0].permute(0,2,1).reshape(B,256,32,32)
        feat_i[1] = feat_i[1].permute(0,2,1).reshape(B,256,32,32)
        feat_i += feat_i_32
        # upsample0
        for block_i, block_op in enumerate(self.pcd_up_0):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)#(17420,256)

        feat_i = [
            self.cnn_up_0(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-1][i]), dim=1)) for i in range(n_views)]#[2](4,128,64,64)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())#(32768,256)
        feat_p2i = self.fuse_p2i_up_0[0](feat_p2i)#(32768,128)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_up_0[1])#[2](4,128,64,64)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])# (17420,32,128)
        feat_i2p = self.fuse_i2p_up_0[0](feat_i2p.max(1)[0])# (17420,256)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_up_0[1])# (17420,256)

        feat_p = feat_p + feat_i2p # (17420,256)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]#[2](4,128,64,64)

        # upsample1
        for block_i, block_op in enumerate(self.pcd_up_1):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)#(130982,32)

        feat_i = [
            self.cnn_up_1(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-2][i]), dim=1)) for i in range(n_views)]#[2](4,32,128,128)

        pcs_X = batch['points_img']     #[2](4,16384,3)
        pcs_F = self.map(feat_i, feat_p, batch) #[2](4,16384,32)

        pcs_F_sc4 = list(self.loftr_coarse4(pcs_F[0],pcs_F[1]))
        pcs_F = pcs_F+pcs_F_sc4
        
        if vps is not None:
            # Drop first viewpoint -- assumed to be identity transformation
            vps = vps[1:]
        elif self.align_cfg.algorithm == "weighted_procrustes":
            vps = []
            cor_loss = []
            for i in range(1, n_views):
                corr_i = get_correspondences(
                    P1=pcs_F[0],
                    P2=pcs_F[i],
                    P1_X=pcs_X[0],
                    P2_X=pcs_X[i],
                    num_corres=self.num_corres,
                    ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
                )
                Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

                vps.append(Rt_i)
                cor_loss.append(cor_loss_i)

                # add for visualization
                output[f"corres_0{i}"] = corr_i
                output[f"vp_{i}"] = Rt_i
        else:
            raise ValueError(f"How to align using {self.align_cfg.algorithm}?")

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        # Get RGB pointcloud as well for direct rendering
        pcs_rgb = [rgb.view(B, 3, -1).permute(0, 2, 1).contiguous() for rgb in rgbs]

        projs = []
        # get joint for all values
        if self.pointcloud_source == "joint":
            pcs_X_joint = torch.cat(pcs_X, dim=1)
            pcs_F_joint = torch.cat(pcs_F, dim=1)
            pcs_RGB_joint = torch.cat(pcs_rgb, dim=1)
            pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

        # Rasterize and Blend
        for i in range(n_views):
            if self.pointcloud_source == "other":
                # get joint for all values except the one
                pcs_X_joint = torch.cat(pcs_X[0:i] + pcs_X[i + 1: n_views], dim=1)
                pcs_F_joint = torch.cat(pcs_F[0:i] + pcs_F[i + 1: n_views], dim=1)
                pcs_RGB_joint = torch.cat(
                    pcs_rgb[0:i] + pcs_rgb[i + 1: n_views], dim=1
                )
                pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

            if i > 0:
                rot_joint_X = transform_points_Rt(pcs_X_joint, vps[i - 1])
                rot_joint_X = points_to_ndc(rot_joint_X, K, (H, W))
            else:
                rot_joint_X = points_to_ndc(pcs_X_joint, K, (H, W))
            projs.append(self.renderer(rot_joint_X, pcs_FRGB_joint))

        # Decode
        for i in range(n_views):
            proj_FRGB_i = projs[i]["feats"]
            proj_RGB_i = proj_FRGB_i[:, -3:]
            proj_F_i = proj_FRGB_i[:, :-3]

            output[f"rgb_render_{i}"] = proj_RGB_i
            output[f"ras_depth_{i}"] = projs[i]["depth"]
            output[f"cover_{i}"] = projs[i]["mask"].unsqueeze(1)  # useless

        return output

    def gather_p2i(self, feat_p, idx):
        feat_p = torch.cat((feat_p, torch.zeros_like(feat_p[:1, :])), 0)
        return feat_p[idx]

    def gather_i2p(self, feat_i, idx):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        feat_i2p = []
        for i in range(B):
            feat_i2p.append(src_feat_i[i].reshape(C,H*W).permute(1,0))
            feat_i2p.append(tgt_feat_i[i].reshape(C,H*W).permute(1,0))

        feat_i2p = torch.cat(feat_i2p, 0)
        feat_i2p = torch.cat((feat_i2p, torch.zeros_like(feat_i2p[:1, :])), 0)
        return feat_i2p[idx]  #N,16,C

    def fusep2i(self, feat_i, feat_p2i, layer):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        # src_feat_i = src_feat_i.reshape(B, C, H * W).permute(0, 2, 1)
        # tgt_feat_i = tgt_feat_i.reshape(B, C, H * W).permute(0, 2, 1)

        src_feat_p2i = []
        tgt_feat_p2i = []

        for i in range(2*B):
            if i % 2 == 0:
                src_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))#[4](1,16384,64)
            else:
                tgt_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))#[4](1,16384,64)

        src_feat_p2i = torch.vstack(src_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)#(4,64,128,128)
        tgt_feat_p2i = torch.vstack(tgt_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)#(4,64,128,128)

        src_feat_p2i = torch.cat([src_feat_i, src_feat_p2i], 1)#(4,128,128,128)
        tgt_feat_p2i = torch.cat([tgt_feat_i, tgt_feat_p2i], 1)#(4,128,128,128)

        src_feat_p2i = layer(src_feat_p2i)#(4,64,128,128)
        tgt_feat_p2i = layer(tgt_feat_p2i)#(4,64,128,128)

        return [src_feat_p2i, tgt_feat_p2i]

    def fusei2p(self, feat_p, feat_i2p, layer):
        feat_i2p = torch.cat([feat_p, feat_i2p], -1)
        feat_i2p = layer(feat_i2p)
        return feat_i2p


    def generate_pointclouds(self, K, deps, vps=None):
        n_views = len(deps)
        # generate pointclouds - generate grid once for efficiency
        B, _, H, W = deps[0].shape
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pcs_X = [
            grid_to_pointcloud(K_inv, deps[i], None, grid)[0] for i in range(n_views)
        ]

        if vps is not None:
            pcs_X_rot = [
                transform_points_Rt(pcs_X[i + 1], vps[i + 1], inverse=True, )
                for i in range(n_views - 1)
            ]
            pcs_X = pcs_X[0:1] + pcs_X_rot
            pcs_X = torch.cat(pcs_X, dim=1).detach().cpu()

        return pcs_X

    def get_feature_pcs(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert (
                feats[0].shape[-1] == deps[0].shape[-1]
        ), f"Same size {feats[0].shape} - {deps[0].shape}"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]
        return pcs_X, pcs_F, None


'''
pointMBF 添加mamba模块
'''
from models.FFMT.mamba.mamba import Mamba,ModelArgs
class PCReg_KPURes18_MSF_mamba(nn.Module):
    def __init__(self, cfg):
        super(PCReg_KPURes18_MSF_mamba, self).__init__()
        # set encoder decoder
        chan_in = 3
        self.cfg = cfg
        feat_dim = cfg.feat_dim

        # No imagenet pretraining
        pretrained = False

        encode_I = URes18Encoder1(chan_in, feat_dim, pretrained)
        encode_P = KPFCN(cfg)
        self.cnn_pre_stages = nn.Sequential(
            encode_I.inconv,
            encode_I.layer1
        )
        self.pcd_pre_stages = nn.ModuleList()
        for i in range(0,2):
            self.pcd_pre_stages.append(encode_P.encoder_blocks[i])

        self.cnn_ds_0 = encode_I.layer2
        self.pcd_ds_0 = nn.ModuleList()
        for i in range(2,5):
            self.pcd_ds_0.append(encode_P.encoder_blocks[i])

        self.cnn_ds_1 = encode_I.layer3
        self.pcd_ds_1 = nn.ModuleList()
        for i in range(5,8):
            self.pcd_ds_1.append(encode_P.encoder_blocks[i])

        self.cnn_up_0 = encode_I.up1
        self.pcd_up_0 = nn.ModuleList()
        for i in range(0, 2):
            self.pcd_up_0.append(encode_P.decoder_blocks[i])

        self.cnn_up_1 = nn.Sequential(
            encode_I.up2,
            encode_I.outconv
        )
        self.pcd_up_1 = nn.ModuleList()
        for i in range(2, 4):
            self.pcd_up_1.append(encode_P.decoder_blocks[i])

        # 0
        self.fuse_p2i_ds_0 = nn.ModuleList()
        self.fuse_p2i_ds_0.append(
            UnaryBlock(128, 64, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_0.append(
            nn.Sequential(
                nn.Conv2d(64*2, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_0 = nn.ModuleList()
        self.fuse_i2p_ds_0.append(
            UnaryBlock(64, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_0.append(
            UnaryBlock(128 * 2, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 1
        self.fuse_p2i_ds_1 = nn.ModuleList()
        self.fuse_p2i_ds_1.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_1.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_1 = nn.ModuleList()
        self.fuse_i2p_ds_1.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_1.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 2
        self.fuse_p2i_ds_2 = nn.ModuleList()
        self.fuse_p2i_ds_2.append(
            UnaryBlock(512, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_2.append(
            nn.Sequential(
                nn.Conv2d(256 * 2, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_2 = nn.ModuleList()
        self.fuse_i2p_ds_2.append(
            UnaryBlock(256, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_2.append(
            UnaryBlock(512 * 2, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        self.fuse_p2i_up_0 = nn.ModuleList()
        self.fuse_p2i_up_0.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_up_0.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
        ))
        self.fuse_i2p_up_0 = nn.ModuleList()
        self.fuse_i2p_up_0.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_up_0.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # self.fuse_p2i_up_1 = nn.ModuleList()
        # self.fuse_i2p_up_1 = nn.ModuleList()
        self.align_cfg = cfg
        self.renderer = PointsRenderer(cfg)
        self.num_corres = cfg.num_correspodances
        self.pointcloud_source = cfg.pointcloud_source
        self.map = Fusion_CATL(feat_dim)
        # self.loftr_coarse = LocalFeatureTransformer({'d_model':32,
        #         'nhead':8,
        #         'layer_names':['self', 'cross'] * 2,
        #         'attention':'linear'})
        mamba_args = ModelArgs()
        mamba_args.d_state = 1
        self.mamba = Mamba(mamba_args)
        # self.mamba2 = Mamba(mamba_args)
        
    # @time_it
    def forward(self, batch, rgbs, K, deps, vps=None):
        # Estimate Depth -- now for 1 and 2
        n_views = 2
        output = {}
        B, _, H, W = rgbs[0].shape

        # Encode features
        feat_p_encode = []
        feat_i_encode = []

        #pre stage
        feat_p = batch['features'].clone().detach()
        for block_op in self.pcd_pre_stages:
            feat_p = block_op(feat_p, batch)# feat_p (130982,128)
            
        
        feat_i = [self.cnn_pre_stages(rgbs[i]) for i in range(n_views)] # feat_i [2](4,64,128,128)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][0].squeeze())
        feat_p2i = self.fuse_p2i_ds_0[0](feat_p2i)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_0[1])

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][0])
        feat_i2p = self.fuse_i2p_ds_0[0](feat_i2p.max(1)[0])
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_0[1])

        feat_p = feat_p + feat_i2p
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]

        feat_p_encode.append(feat_p)
        feat_i_encode.append(feat_i)

        #downsample 0
        for block_op in self.pcd_ds_0:
            feat_p = block_op(feat_p, batch)
        feat_i = [self.cnn_ds_0(feat_i[i]) for i in range(n_views)]

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())
        feat_p2i = self.fuse_p2i_ds_1[0](feat_p2i)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_1[1])

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])
        feat_i2p = self.fuse_i2p_ds_1[0](feat_i2p.max(1)[0])
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_1[1])

        feat_p = feat_p + feat_i2p
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]

        feat_p_encode.append(feat_p)
        feat_i_encode.append(feat_i)

        #downsample 1
        for block_op in self.pcd_ds_1:
            feat_p = block_op(feat_p, batch)
        feat_i = [self.cnn_ds_1(feat_i[i]) for i in range(n_views)]

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][2].squeeze())
        feat_p2i = self.fuse_p2i_ds_2[0](feat_p2i)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_2[1])

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][2])
        feat_i2p = self.fuse_i2p_ds_2[0](feat_i2p.max(1)[0])
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_2[1])

        feat_p = feat_p + feat_i2p
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]


        # upsample0
        for block_i, block_op in enumerate(self.pcd_up_0):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)

        feat_i = [
            self.cnn_up_0(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-1][i]), dim=1)) for i in range(n_views)]

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())
        feat_p2i = self.fuse_p2i_up_0[0](feat_p2i)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_up_0[1])

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])
        feat_i2p = self.fuse_i2p_up_0[0](feat_i2p.max(1)[0])
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_up_0[1])

        feat_p = feat_p + feat_i2p
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]

        # upsample1
        for block_i, block_op in enumerate(self.pcd_up_1):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)

        feat_i = [
            self.cnn_up_1(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-2][i]), dim=1)) for i in range(n_views)]

        pcs_X = batch['points_img']     #[2](4,16384,3)
        pcs_F = self.map(feat_i, feat_p, batch) #[2](4,16384,32)
        
        #-----------------
        #修改此处
        pcs_F = [self.mamba(pcs_F[0]),self.mamba(pcs_F[1])]
        
        if vps is not None:
            # Drop first viewpoint -- assumed to be identity transformation
            vps = vps[1:]
        elif self.align_cfg.algorithm == "weighted_procrustes":
            vps = []
            cor_loss = []
            for i in range(1, n_views):
                corr_i = get_correspondences(
                    P1=pcs_F[0],
                    P2=pcs_F[i],
                    P1_X=pcs_X[0],
                    P2_X=pcs_X[i],
                    num_corres=self.num_corres,
                    ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
                )
                Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

                vps.append(Rt_i)
                cor_loss.append(cor_loss_i)

                # add for visualization
                output[f"corres_0{i}"] = corr_i
                output[f"vp_{i}"] = Rt_i
        else:
            raise ValueError(f"How to align using {self.align_cfg.algorithm}?")

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        # Get RGB pointcloud as well for direct rendering
        pcs_rgb = [rgb.view(B, 3, -1).permute(0, 2, 1).contiguous() for rgb in rgbs]

        projs = []
        # get joint for all values
        if self.pointcloud_source == "joint":
            pcs_X_joint = torch.cat(pcs_X, dim=1)
            pcs_F_joint = torch.cat(pcs_F, dim=1)
            pcs_RGB_joint = torch.cat(pcs_rgb, dim=1)
            pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

        # Rasterize and Blend
        for i in range(n_views):
            if self.pointcloud_source == "other":
                # get joint for all values except the one
                pcs_X_joint = torch.cat(pcs_X[0:i] + pcs_X[i + 1: n_views], dim=1)
                pcs_F_joint = torch.cat(pcs_F[0:i] + pcs_F[i + 1: n_views], dim=1)
                pcs_RGB_joint = torch.cat(
                    pcs_rgb[0:i] + pcs_rgb[i + 1: n_views], dim=1
                )
                pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

            if i > 0:
                rot_joint_X = transform_points_Rt(pcs_X_joint, vps[i - 1])
                rot_joint_X = points_to_ndc(rot_joint_X, K, (H, W))
            else:
                rot_joint_X = points_to_ndc(pcs_X_joint, K, (H, W))
            projs.append(self.renderer(rot_joint_X, pcs_FRGB_joint))

        # Decode
        for i in range(n_views):
            proj_FRGB_i = projs[i]["feats"]
            proj_RGB_i = proj_FRGB_i[:, -3:]
            proj_F_i = proj_FRGB_i[:, :-3]

            output[f"rgb_render_{i}"] = proj_RGB_i
            output[f"ras_depth_{i}"] = projs[i]["depth"]
            output[f"cover_{i}"] = projs[i]["mask"].unsqueeze(1)  # useless

        return output

    def gather_p2i(self, feat_p, idx):
        feat_p = torch.cat((feat_p, torch.zeros_like(feat_p[:1, :])), 0)
        return feat_p[idx]

    def gather_i2p(self, feat_i, idx):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        feat_i2p = []
        for i in range(B):
            feat_i2p.append(src_feat_i[i].reshape(C,H*W).permute(1,0))
            feat_i2p.append(tgt_feat_i[i].reshape(C,H*W).permute(1,0))

        feat_i2p = torch.cat(feat_i2p, 0)
        feat_i2p = torch.cat((feat_i2p, torch.zeros_like(feat_i2p[:1, :])), 0)
        return feat_i2p[idx]  #N,16,C

    def fusep2i(self, feat_i, feat_p2i, layer):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        # src_feat_i = src_feat_i.reshape(B, C, H * W).permute(0, 2, 1)
        # tgt_feat_i = tgt_feat_i.reshape(B, C, H * W).permute(0, 2, 1)

        src_feat_p2i = []
        tgt_feat_p2i = []

        for i in range(2*B):
            if i % 2 == 0:
                src_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))
            else:
                tgt_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))

        src_feat_p2i = torch.vstack(src_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)
        tgt_feat_p2i = torch.vstack(tgt_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)

        src_feat_p2i = torch.cat([src_feat_i, src_feat_p2i], 1)
        tgt_feat_p2i = torch.cat([tgt_feat_i, tgt_feat_p2i], 1)

        src_feat_p2i = layer(src_feat_p2i)
        tgt_feat_p2i = layer(tgt_feat_p2i)

        return [src_feat_p2i, tgt_feat_p2i]

    def fusei2p(self, feat_p, feat_i2p, layer):
        feat_i2p = torch.cat([feat_p, feat_i2p], -1)
        feat_i2p = layer(feat_i2p)
        return feat_i2p


    def generate_pointclouds(self, K, deps, vps=None):
        n_views = len(deps)
        # generate pointclouds - generate grid once for efficiency
        B, _, H, W = deps[0].shape
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pcs_X = [
            grid_to_pointcloud(K_inv, deps[i], None, grid)[0] for i in range(n_views)
        ]

        if vps is not None:
            pcs_X_rot = [
                transform_points_Rt(pcs_X[i + 1], vps[i + 1], inverse=True, )
                for i in range(n_views - 1)
            ]
            pcs_X = pcs_X[0:1] + pcs_X_rot
            pcs_X = torch.cat(pcs_X, dim=1).detach().cpu()

        return pcs_X

    def get_feature_pcs(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert (
                feats[0].shape[-1] == deps[0].shape[-1]
        ), f"Same size {feats[0].shape} - {deps[0].shape}"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]
        return pcs_X, pcs_F, None


'''
pointMBF原版缩减通道数
'''
class PCReg_KPURes18_MSF_redc(nn.Module):
    def __init__(self, cfg):
        super(PCReg_KPURes18_MSF_redc, self).__init__()
        # set encoder decoder
        chan_in = 3
        self.cfg = cfg
        feat_dim = cfg.feat_dim

        # No imagenet pretraining
        pretrained = False

        encode_I = URes18Encoder1(chan_in, feat_dim, pretrained)
        encode_P = KPFCN(cfg)
        self.cnn_pre_stages = nn.Sequential(
            encode_I.inconv,
            encode_I.layer1
        )
        self.pcd_pre_stages = nn.ModuleList()
        for i in range(0,2):
            self.pcd_pre_stages.append(encode_P.encoder_blocks[i])

        self.cnn_ds_0 = encode_I.layer2
        self.pcd_ds_0 = nn.ModuleList()
        for i in range(2,5):
            self.pcd_ds_0.append(encode_P.encoder_blocks[i])

        self.cnn_ds_1 = encode_I.layer3
        self.pcd_ds_1 = nn.ModuleList()
        for i in range(5,8):
            self.pcd_ds_1.append(encode_P.encoder_blocks[i])

        self.cnn_up_0 = encode_I.up1
        self.pcd_up_0 = nn.ModuleList()
        for i in range(0, 2):
            self.pcd_up_0.append(encode_P.decoder_blocks[i])

        self.cnn_up_1 = nn.Sequential(
            encode_I.up2,
            encode_I.outconv
        )
        self.pcd_up_1 = nn.ModuleList()
        for i in range(2, 4):
            self.pcd_up_1.append(encode_P.decoder_blocks[i])

        # 0
        self.fuse_p2i_ds_0 = nn.ModuleList()
        self.fuse_p2i_ds_0.append(
            UnaryBlock(32, 32, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_0.append(
            nn.Sequential(
                nn.Conv2d(32, 32, 1),
                nn.BatchNorm2d(32),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_0 = nn.ModuleList()
        self.fuse_i2p_ds_0.append(
            UnaryBlock(64, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_0.append(
            UnaryBlock(128 * 2, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 1
        self.fuse_p2i_ds_1 = nn.ModuleList()
        self.fuse_p2i_ds_1.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_1.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_1 = nn.ModuleList()
        self.fuse_i2p_ds_1.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_1.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 2
        self.fuse_p2i_ds_2 = nn.ModuleList()
        self.fuse_p2i_ds_2.append(
            UnaryBlock(512, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_2.append(
            nn.Sequential(
                nn.Conv2d(256 * 2, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_2 = nn.ModuleList()
        self.fuse_i2p_ds_2.append(
            UnaryBlock(256, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_2.append(
            UnaryBlock(512 * 2, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        self.fuse_p2i_up_0 = nn.ModuleList()
        self.fuse_p2i_up_0.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_up_0.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
        ))
        self.fuse_i2p_up_0 = nn.ModuleList()
        self.fuse_i2p_up_0.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_up_0.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # self.fuse_p2i_up_1 = nn.ModuleList()
        # self.fuse_i2p_up_1 = nn.ModuleList()
        self.align_cfg = cfg
        self.renderer = PointsRenderer(cfg)
        self.num_corres = cfg.num_correspodances
        self.pointcloud_source = cfg.pointcloud_source
        self.map = Fusion_CATL(feat_dim)

    def forward(self, batch, rgbs, K, deps, vps=None):
        # Estimate Depth -- now for 1 and 2
        n_views = 2
        output = {}
        B, _, H, W = rgbs[0].shape

        # Encode features
        feat_p_encode = []
        feat_i_encode = []

        #pre stage
        feat_p = batch['features'].clone().detach()
        for block_op in self.pcd_pre_stages:
            feat_p = block_op(feat_p, batch)# feat_p (130982,128)
            
        
        feat_i = [self.cnn_pre_stages(rgbs[i]) for i in range(n_views)] # feat_i [2](4,64,128,128)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][0].squeeze())#(131072,128)
        feat_p2i = self.fuse_p2i_ds_0[0](feat_p2i)#(131072,64)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_0[1])# [2](4,64,128,128)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][0])#(130982,32,64)
        feat_i2p = self.fuse_i2p_ds_0[0](feat_i2p.max(1)[0])#(130982,128)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_0[1])#(130982,128)

        feat_p = feat_p + feat_i2p#(130982,128)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,64,128,128)

        feat_p_encode.append(feat_p)# [1](130982,128)
        feat_i_encode.append(feat_i)# [1][2](4,64,128,128)

        #downsample 0
        for block_op in self.pcd_ds_0:
            feat_p = block_op(feat_p, batch)# (17420,256)
        feat_i = [self.cnn_ds_0(feat_i[i]) for i in range(n_views)]# [2](4,128,64,64)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())#(32768,256)
        feat_p2i = self.fuse_p2i_ds_1[0](feat_p2i)#(32768,128)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_1[1])# [2](4,128,64,64)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])# (17420,32,128)
        feat_i2p = self.fuse_i2p_ds_1[0](feat_i2p.max(1)[0])# (17420,256)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_1[1])# (17420,256)

        feat_p = feat_p + feat_i2p# (17420,256)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,128,64,64)

        feat_p_encode.append(feat_p)# [(130982,128)(17420,256)]
        feat_i_encode.append(feat_i)# [[2](4,64,128,128),[2](4,128,64,64)]

        #downsample 1
        for block_op in self.pcd_ds_1:
            feat_p = block_op(feat_p, batch)# (5895,512)
        feat_i = [self.cnn_ds_1(feat_i[i]) for i in range(n_views)]# [2](4,256,32,32)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][2].squeeze())#(8192,512)
        feat_p2i = self.fuse_p2i_ds_2[0](feat_p2i)#(8192,256)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_2[1])# [2](4,256,32,32)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][2])#(5895,32,256)
        feat_i2p = self.fuse_i2p_ds_2[0](feat_i2p.max(1)[0])#(5895,512)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_2[1])#(5895,512)

        feat_p = feat_p + feat_i2p#(5895,512)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,256,32,32)


        # upsample0
        for block_i, block_op in enumerate(self.pcd_up_0):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)#(17420,256)

        feat_i = [
            self.cnn_up_0(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-1][i]), dim=1)) for i in range(n_views)]#[2](4,128,64,64)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())#(32768,256)
        feat_p2i = self.fuse_p2i_up_0[0](feat_p2i)#(32768,128)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_up_0[1])#[2](4,128,64,64)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])# (17420,32,128)
        feat_i2p = self.fuse_i2p_up_0[0](feat_i2p.max(1)[0])# (17420,256)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_up_0[1])# (17420,256)

        feat_p = feat_p + feat_i2p # (17420,256)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]#[2](4,128,64,64)

        # upsample1
        for block_i, block_op in enumerate(self.pcd_up_1):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)#(130982,32)

        feat_i = [
            self.cnn_up_1(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-2][i]), dim=1)) for i in range(n_views)]#[2](4,32,128,128)

        pcs_X = batch['points_img']     #[2](4,16384,3)
        pcs_F = self.map(feat_i, feat_p, batch) #[2](4,16384,32)

        if vps is not None:
            # Drop first viewpoint -- assumed to be identity transformation
            vps = vps[1:]
        elif self.align_cfg.algorithm == "weighted_procrustes":
            vps = []
            cor_loss = []
            for i in range(1, n_views):
                corr_i = get_correspondences(
                    P1=pcs_F[0],
                    P2=pcs_F[i],
                    P1_X=pcs_X[0],
                    P2_X=pcs_X[i],
                    num_corres=self.num_corres,
                    ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
                )
                Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

                vps.append(Rt_i)
                cor_loss.append(cor_loss_i)

                # add for visualization
                output[f"corres_0{i}"] = corr_i
                output[f"vp_{i}"] = Rt_i
        else:
            raise ValueError(f"How to align using {self.align_cfg.algorithm}?")

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        # Get RGB pointcloud as well for direct rendering
        pcs_rgb = [rgb.view(B, 3, -1).permute(0, 2, 1).contiguous() for rgb in rgbs]

        projs = []
        # get joint for all values
        if self.pointcloud_source == "joint":
            pcs_X_joint = torch.cat(pcs_X, dim=1)
            pcs_F_joint = torch.cat(pcs_F, dim=1)
            pcs_RGB_joint = torch.cat(pcs_rgb, dim=1)
            pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

        # Rasterize and Blend
        for i in range(n_views):
            if self.pointcloud_source == "other":
                # get joint for all values except the one
                pcs_X_joint = torch.cat(pcs_X[0:i] + pcs_X[i + 1: n_views], dim=1)
                pcs_F_joint = torch.cat(pcs_F[0:i] + pcs_F[i + 1: n_views], dim=1)
                pcs_RGB_joint = torch.cat(
                    pcs_rgb[0:i] + pcs_rgb[i + 1: n_views], dim=1
                )
                pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

            if i > 0:
                rot_joint_X = transform_points_Rt(pcs_X_joint, vps[i - 1])
                rot_joint_X = points_to_ndc(rot_joint_X, K, (H, W))
            else:
                rot_joint_X = points_to_ndc(pcs_X_joint, K, (H, W))
            projs.append(self.renderer(rot_joint_X, pcs_FRGB_joint))

        # Decode
        for i in range(n_views):
            proj_FRGB_i = projs[i]["feats"]
            proj_RGB_i = proj_FRGB_i[:, -3:]
            proj_F_i = proj_FRGB_i[:, :-3]

            output[f"rgb_render_{i}"] = proj_RGB_i
            output[f"ras_depth_{i}"] = projs[i]["depth"]
            output[f"cover_{i}"] = projs[i]["mask"].unsqueeze(1)  # useless

        return output

    def gather_p2i(self, feat_p, idx):
        feat_p = torch.cat((feat_p, torch.zeros_like(feat_p[:1, :])), 0)
        return feat_p[idx]

    def gather_i2p(self, feat_i, idx):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        feat_i2p = []
        for i in range(B):
            feat_i2p.append(src_feat_i[i].reshape(C,H*W).permute(1,0))
            feat_i2p.append(tgt_feat_i[i].reshape(C,H*W).permute(1,0))

        feat_i2p = torch.cat(feat_i2p, 0)
        feat_i2p = torch.cat((feat_i2p, torch.zeros_like(feat_i2p[:1, :])), 0)
        return feat_i2p[idx]  #N,16,C

    def fusep2i(self, feat_i, feat_p2i, layer):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        # src_feat_i = src_feat_i.reshape(B, C, H * W).permute(0, 2, 1)
        # tgt_feat_i = tgt_feat_i.reshape(B, C, H * W).permute(0, 2, 1)

        src_feat_p2i = []
        tgt_feat_p2i = []

        for i in range(2*B):
            if i % 2 == 0:
                src_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))
            else:
                tgt_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))

        src_feat_p2i = torch.vstack(src_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)
        tgt_feat_p2i = torch.vstack(tgt_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)

        src_feat_p2i = torch.cat([src_feat_i, src_feat_p2i], 1)
        tgt_feat_p2i = torch.cat([tgt_feat_i, tgt_feat_p2i], 1)

        src_feat_p2i = layer(src_feat_p2i)
        tgt_feat_p2i = layer(tgt_feat_p2i)

        return [src_feat_p2i, tgt_feat_p2i]

    def fusei2p(self, feat_p, feat_i2p, layer):
        feat_i2p = torch.cat([feat_p, feat_i2p], -1)
        feat_i2p = layer(feat_i2p)
        return feat_i2p


    def generate_pointclouds(self, K, deps, vps=None):
        n_views = len(deps)
        # generate pointclouds - generate grid once for efficiency
        B, _, H, W = deps[0].shape
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pcs_X = [
            grid_to_pointcloud(K_inv, deps[i], None, grid)[0] for i in range(n_views)
        ]

        if vps is not None:
            pcs_X_rot = [
                transform_points_Rt(pcs_X[i + 1], vps[i + 1], inverse=True, )
                for i in range(n_views - 1)
            ]
            pcs_X = pcs_X[0:1] + pcs_X_rot
            pcs_X = torch.cat(pcs_X, dim=1).detach().cpu()

        return pcs_X

    def get_feature_pcs(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert (
                feats[0].shape[-1] == deps[0].shape[-1]
        ), f"Same size {feats[0].shape} - {deps[0].shape}"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]
        return pcs_X, pcs_F, None

from models.FFMT.GAU.gau_attiontion import GAU_attn
class PCReg_KPURes18_MSF_gau0(nn.Module):
    def __init__(self, cfg):
        super(PCReg_KPURes18_MSF_gau0, self).__init__()
        # set encoder decoder
        chan_in = 3
        self.cfg = cfg
        feat_dim = cfg.feat_dim

        # No imagenet pretraining
        pretrained = False

        encode_I = URes18Encoder1(chan_in, feat_dim, pretrained)
        encode_P = KPFCN(cfg)
        self.cnn_pre_stages = nn.Sequential(
            encode_I.inconv,
            encode_I.layer1
        )
        self.pcd_pre_stages = nn.ModuleList()
        for i in range(0,2):
            self.pcd_pre_stages.append(encode_P.encoder_blocks[i])

        self.cnn_ds_0 = encode_I.layer2
        self.pcd_ds_0 = nn.ModuleList()
        for i in range(2,5):
            self.pcd_ds_0.append(encode_P.encoder_blocks[i])

        self.cnn_ds_1 = encode_I.layer3
        self.pcd_ds_1 = nn.ModuleList()
        for i in range(5,8):
            self.pcd_ds_1.append(encode_P.encoder_blocks[i])

        self.cnn_up_0 = encode_I.up1
        self.pcd_up_0 = nn.ModuleList()
        for i in range(0, 2):
            self.pcd_up_0.append(encode_P.decoder_blocks[i])

        self.cnn_up_1 = nn.Sequential(
            encode_I.up2,
            encode_I.outconv
        )
        self.pcd_up_1 = nn.ModuleList()
        for i in range(2, 4):
            self.pcd_up_1.append(encode_P.decoder_blocks[i])

        # 0
        self.fuse_p2i_ds_0 = nn.ModuleList()
        self.fuse_p2i_ds_0.append(
            UnaryBlock(128, 64, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_0.append(
            nn.Sequential(
                nn.Conv2d(64*2, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_0 = nn.ModuleList()
        self.fuse_i2p_ds_0.append(
            UnaryBlock(64, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_0.append(
            UnaryBlock(128 * 2, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 1
        self.fuse_p2i_ds_1 = nn.ModuleList()
        self.fuse_p2i_ds_1.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_1.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_1 = nn.ModuleList()
        self.fuse_i2p_ds_1.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_1.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 2
        self.fuse_p2i_ds_2 = nn.ModuleList()
        self.fuse_p2i_ds_2.append(
            UnaryBlock(512, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_2.append(
            nn.Sequential(
                nn.Conv2d(256 * 2, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_2 = nn.ModuleList()
        self.fuse_i2p_ds_2.append(
            UnaryBlock(256, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_2.append(
            UnaryBlock(512 * 2, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        self.fuse_p2i_up_0 = nn.ModuleList()
        self.fuse_p2i_up_0.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_up_0.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
        ))
        self.fuse_i2p_up_0 = nn.ModuleList()
        self.fuse_i2p_up_0.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_up_0.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # self.fuse_p2i_up_1 = nn.ModuleList()
        # self.fuse_i2p_up_1 = nn.ModuleList()
        self.align_cfg = cfg
        self.renderer = PointsRenderer(cfg)
        self.num_corres = cfg.num_correspodances
        self.pointcloud_source = cfg.pointcloud_source
        
        self.map = Fusion_CATL(feat_dim)
        
        self.gau  = GAU_attn({
        'd_model':32,
        'layer_names':['self','cross']*4,
        'query_key_dim':128,
        'laplace_attn_fn':False
        })

    def forward(self, batch, rgbs, K, deps, vps=None):
        # Estimate Depth -- now for 1 and 2
        n_views = 2
        output = {}
        B, _, H, W = rgbs[0].shape

        # Encode features
        feat_p_encode = []
        feat_i_encode = []

        #pre stage
        feat_p = batch['features'].clone().detach() #全为1 (130982,1)
        for block_op in self.pcd_pre_stages:
            feat_p = block_op(feat_p, batch)# feat_p (130982,128)
            
        
        feat_i = [self.cnn_pre_stages(rgbs[i]) for i in range(n_views)] # feat_i [2](4,64,128,128)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][0].squeeze())#(131072,128)
        feat_p2i = self.fuse_p2i_ds_0[0](feat_p2i)#(131072,64)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_0[1])# [2](4,64,128,128)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][0])#(130982,32,64)
        feat_i2p = self.fuse_i2p_ds_0[0](feat_i2p.max(1)[0])#(130982,128)类似于pointnet的最大池化
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_0[1])#(130982,128)

        feat_p = feat_p + feat_i2p#(130982,128)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,64,128,128)

        feat_p_encode.append(feat_p)# [1](130982,128)
        feat_i_encode.append(feat_i)# [1][2](4,64,128,128)

        #downsample 0
        for block_op in self.pcd_ds_0:
            feat_p = block_op(feat_p, batch)# (17420,256)
        feat_i = [self.cnn_ds_0(feat_i[i]) for i in range(n_views)]# [2](4,128,64,64)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())#(32768,256)
        feat_p2i = self.fuse_p2i_ds_1[0](feat_p2i)#(32768,128)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_1[1])# [2](4,128,64,64)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])# (17420,32,128)
        feat_i2p = self.fuse_i2p_ds_1[0](feat_i2p.max(1)[0])# (17420,256)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_1[1])# (17420,256)

        feat_p = feat_p + feat_i2p# (17420,256)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,128,64,64)

        feat_p_encode.append(feat_p)# [(130982,128)(17420,256)]
        feat_i_encode.append(feat_i)# [[2](4,64,128,128),[2](4,128,64,64)]

        #downsample 1
        for block_op in self.pcd_ds_1:
            feat_p = block_op(feat_p, batch)# (5895,512)
        feat_i = [self.cnn_ds_1(feat_i[i]) for i in range(n_views)]# [2](4,256,32,32)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][2].squeeze())#(8192,512)
        feat_p2i = self.fuse_p2i_ds_2[0](feat_p2i)#(8192,256)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_2[1])# [2](4,256,32,32)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][2])#(5895,32,256)
        feat_i2p = self.fuse_i2p_ds_2[0](feat_i2p.max(1)[0])#(5895,512)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_2[1])#(5895,512)

        feat_p = feat_p + feat_i2p#(5895,512)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,256,32,32)


        # upsample0
        for block_i, block_op in enumerate(self.pcd_up_0):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)#(17420,256)

        feat_i = [
            self.cnn_up_0(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-1][i]), dim=1)) for i in range(n_views)]#[2](4,128,64,64)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())#(32768,256)
        feat_p2i = self.fuse_p2i_up_0[0](feat_p2i)#(32768,128)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_up_0[1])#[2](4,128,64,64)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])# (17420,32,128)
        feat_i2p = self.fuse_i2p_up_0[0](feat_i2p.max(1)[0])# (17420,256)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_up_0[1])# (17420,256)

        feat_p = feat_p + feat_i2p # (17420,256)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]#[2](4,128,64,64)

        # upsample1
        for block_i, block_op in enumerate(self.pcd_up_1):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)#(130982,32)

        feat_i = [
            self.cnn_up_1(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-2][i]), dim=1)) for i in range(n_views)]#[2](4,32,128,128)

        pcs_X = batch['points_img']     #[2](4,16384,3)
        pcs_F = self.map(feat_i, feat_p, batch) #[2](4,16384,32)

        pcs_F = pcs_F+list(self.gau(pcs_F[0],pcs_F[1]))
        
        
        if vps is not None:
            # Drop first viewpoint -- assumed to be identity transformation
            vps = vps[1:]
        elif self.align_cfg.algorithm == "weighted_procrustes":
            vps = []
            cor_loss = []
            for i in range(1, n_views):
                corr_i = get_correspondences(
                    P1=pcs_F[0],
                    P2=pcs_F[i],
                    P1_X=pcs_X[0],
                    P2_X=pcs_X[i],
                    num_corres=self.num_corres,
                    ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
                )
                Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

                vps.append(Rt_i)
                cor_loss.append(cor_loss_i)

                # add for visualization
                output[f"corres_0{i}"] = corr_i
                output[f"vp_{i}"] = Rt_i
        else:
            raise ValueError(f"How to align using {self.align_cfg.algorithm}?")

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        # Get RGB pointcloud as well for direct rendering
        pcs_rgb = [rgb.view(B, 3, -1).permute(0, 2, 1).contiguous() for rgb in rgbs]

        projs = []
        # get joint for all values
        if self.pointcloud_source == "joint":
            pcs_X_joint = torch.cat(pcs_X, dim=1)
            pcs_F_joint = torch.cat(pcs_F, dim=1)
            pcs_RGB_joint = torch.cat(pcs_rgb, dim=1)
            pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

        # Rasterize and Blend
        for i in range(n_views):
            if self.pointcloud_source == "other":
                # get joint for all values except the one
                pcs_X_joint = torch.cat(pcs_X[0:i] + pcs_X[i + 1: n_views], dim=1)
                pcs_F_joint = torch.cat(pcs_F[0:i] + pcs_F[i + 1: n_views], dim=1)
                pcs_RGB_joint = torch.cat(
                    pcs_rgb[0:i] + pcs_rgb[i + 1: n_views], dim=1
                )
                pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

            if i > 0:
                rot_joint_X = transform_points_Rt(pcs_X_joint, vps[i - 1])
                rot_joint_X = points_to_ndc(rot_joint_X, K, (H, W))
            else:
                rot_joint_X = points_to_ndc(pcs_X_joint, K, (H, W))
            projs.append(self.renderer(rot_joint_X, pcs_FRGB_joint))

        # Decode
        for i in range(n_views):
            proj_FRGB_i = projs[i]["feats"]
            proj_RGB_i = proj_FRGB_i[:, -3:]
            proj_F_i = proj_FRGB_i[:, :-3]

            output[f"rgb_render_{i}"] = proj_RGB_i
            output[f"ras_depth_{i}"] = projs[i]["depth"]
            output[f"cover_{i}"] = projs[i]["mask"].unsqueeze(1)  # useless

        return output

    def gather_p2i(self, feat_p, idx):
        feat_p = torch.cat((feat_p, torch.zeros_like(feat_p[:1, :])), 0)
        return feat_p[idx]

    def gather_i2p(self, feat_i, idx):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        feat_i2p = []
        for i in range(B):
            feat_i2p.append(src_feat_i[i].reshape(C,H*W).permute(1,0))
            feat_i2p.append(tgt_feat_i[i].reshape(C,H*W).permute(1,0))

        feat_i2p = torch.cat(feat_i2p, 0)
        feat_i2p = torch.cat((feat_i2p, torch.zeros_like(feat_i2p[:1, :])), 0)
        return feat_i2p[idx]  #N,16,C

    def fusep2i(self, feat_i, feat_p2i, layer):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        # src_feat_i = src_feat_i.reshape(B, C, H * W).permute(0, 2, 1)
        # tgt_feat_i = tgt_feat_i.reshape(B, C, H * W).permute(0, 2, 1)

        src_feat_p2i = []
        tgt_feat_p2i = []

        for i in range(2*B):
            if i % 2 == 0:
                src_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))#[4](1,16384,64)
            else:
                tgt_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))#[4](1,16384,64)

        src_feat_p2i = torch.vstack(src_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)#(4,64,128,128)
        tgt_feat_p2i = torch.vstack(tgt_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)#(4,64,128,128)

        src_feat_p2i = torch.cat([src_feat_i, src_feat_p2i], 1)#(4,128,128,128)
        tgt_feat_p2i = torch.cat([tgt_feat_i, tgt_feat_p2i], 1)#(4,128,128,128)

        src_feat_p2i = layer(src_feat_p2i)#(4,64,128,128)
        tgt_feat_p2i = layer(tgt_feat_p2i)#(4,64,128,128)

        return [src_feat_p2i, tgt_feat_p2i]

    def fusei2p(self, feat_p, feat_i2p, layer):
        feat_i2p = torch.cat([feat_p, feat_i2p], -1)
        feat_i2p = layer(feat_i2p)
        return feat_i2p


    def generate_pointclouds(self, K, deps, vps=None):
        n_views = len(deps)
        # generate pointclouds - generate grid once for efficiency
        B, _, H, W = deps[0].shape
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pcs_X = [
            grid_to_pointcloud(K_inv, deps[i], None, grid)[0] for i in range(n_views)
        ]

        if vps is not None:
            pcs_X_rot = [
                transform_points_Rt(pcs_X[i + 1], vps[i + 1], inverse=True, )
                for i in range(n_views - 1)
            ]
            pcs_X = pcs_X[0:1] + pcs_X_rot
            pcs_X = torch.cat(pcs_X, dim=1).detach().cpu()

        return pcs_X

    def get_feature_pcs(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert (
                feats[0].shape[-1] == deps[0].shape[-1]
        ), f"Same size {feats[0].shape} - {deps[0].shape}"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]
        return pcs_X, pcs_F, None

class PCReg_KPURes18_MSF_gau1(nn.Module):
    def __init__(self, cfg):
        super(PCReg_KPURes18_MSF_gau1, self).__init__()
        # set encoder decoder
        chan_in = 3
        self.cfg = cfg
        feat_dim = cfg.feat_dim

        # No imagenet pretraining
        pretrained = False

        encode_I = URes18Encoder1(chan_in, feat_dim, pretrained)
        encode_P = KPFCN(cfg)
        self.cnn_pre_stages = nn.Sequential(
            encode_I.inconv,
            encode_I.layer1
        )
        self.pcd_pre_stages = nn.ModuleList()
        for i in range(0,2):
            self.pcd_pre_stages.append(encode_P.encoder_blocks[i])

        self.cnn_ds_0 = encode_I.layer2
        self.pcd_ds_0 = nn.ModuleList()
        for i in range(2,5):
            self.pcd_ds_0.append(encode_P.encoder_blocks[i])

        self.cnn_ds_1 = encode_I.layer3
        self.pcd_ds_1 = nn.ModuleList()
        for i in range(5,8):
            self.pcd_ds_1.append(encode_P.encoder_blocks[i])

        self.cnn_up_0 = encode_I.up1
        self.pcd_up_0 = nn.ModuleList()
        for i in range(0, 2):
            self.pcd_up_0.append(encode_P.decoder_blocks[i])

        self.cnn_up_1 = nn.Sequential(
            encode_I.up2,
            encode_I.outconv
        )
        self.pcd_up_1 = nn.ModuleList()
        for i in range(2, 4):
            self.pcd_up_1.append(encode_P.decoder_blocks[i])

        # 0
        self.fuse_p2i_ds_0 = nn.ModuleList()
        self.fuse_p2i_ds_0.append(
            UnaryBlock(128, 64, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_0.append(
            nn.Sequential(
                nn.Conv2d(64*2, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_0 = nn.ModuleList()
        self.fuse_i2p_ds_0.append(
            UnaryBlock(64, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_0.append(
            UnaryBlock(128 * 2, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 1
        self.fuse_p2i_ds_1 = nn.ModuleList()
        self.fuse_p2i_ds_1.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_1.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_1 = nn.ModuleList()
        self.fuse_i2p_ds_1.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_1.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 2
        self.fuse_p2i_ds_2 = nn.ModuleList()
        self.fuse_p2i_ds_2.append(
            UnaryBlock(512, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_2.append(
            nn.Sequential(
                nn.Conv2d(256 * 2, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_2 = nn.ModuleList()
        self.fuse_i2p_ds_2.append(
            UnaryBlock(256, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_2.append(
            UnaryBlock(512 * 2, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        self.fuse_p2i_up_0 = nn.ModuleList()
        self.fuse_p2i_up_0.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_up_0.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
        ))
        self.fuse_i2p_up_0 = nn.ModuleList()
        self.fuse_i2p_up_0.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_up_0.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # self.fuse_p2i_up_1 = nn.ModuleList()
        # self.fuse_i2p_up_1 = nn.ModuleList()
        self.align_cfg = cfg
        self.renderer = PointsRenderer(cfg)
        self.num_corres = cfg.num_correspodances
        self.pointcloud_source = cfg.pointcloud_source
        
        self.map = Fusion_CATL(feat_dim)
        
        self.gau  = GAU_attn({
        'd_model':32,
        'layer_names':['self','cross']*1,
        'query_key_dim':64,
        'laplace_attn_fn':False
        })

    def forward(self, batch, rgbs, K, deps, vps=None):
        # Estimate Depth -- now for 1 and 2
        n_views = 2
        output = {}
        B, _, H, W = rgbs[0].shape

        # Encode features
        feat_p_encode = []
        feat_i_encode = []

        #pre stage
        feat_p = batch['features'].clone().detach() #全为1 (130982,1)
        for block_op in self.pcd_pre_stages:
            feat_p = block_op(feat_p, batch)# feat_p (130982,128)
            
        
        feat_i = [self.cnn_pre_stages(rgbs[i]) for i in range(n_views)] # feat_i [2](4,64,128,128)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][0].squeeze())#(131072,128)
        feat_p2i = self.fuse_p2i_ds_0[0](feat_p2i)#(131072,64)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_0[1])# [2](4,64,128,128)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][0])#(130982,32,64)
        feat_i2p = self.fuse_i2p_ds_0[0](feat_i2p.max(1)[0])#(130982,128)类似于pointnet的最大池化
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_0[1])#(130982,128)

        feat_p = feat_p + feat_i2p#(130982,128)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,64,128,128)

        feat_p_encode.append(feat_p)# [1](130982,128)
        feat_i_encode.append(feat_i)# [1][2](4,64,128,128)

        #downsample 0
        for block_op in self.pcd_ds_0:
            feat_p = block_op(feat_p, batch)# (17420,256)
        feat_i = [self.cnn_ds_0(feat_i[i]) for i in range(n_views)]# [2](4,128,64,64)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())#(32768,256)
        feat_p2i = self.fuse_p2i_ds_1[0](feat_p2i)#(32768,128)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_1[1])# [2](4,128,64,64)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])# (17420,32,128)
        feat_i2p = self.fuse_i2p_ds_1[0](feat_i2p.max(1)[0])# (17420,256)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_1[1])# (17420,256)

        feat_p = feat_p + feat_i2p# (17420,256)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,128,64,64)

        feat_p_encode.append(feat_p)# [(130982,128)(17420,256)]
        feat_i_encode.append(feat_i)# [[2](4,64,128,128),[2](4,128,64,64)]

        #downsample 1
        for block_op in self.pcd_ds_1:
            feat_p = block_op(feat_p, batch)# (5895,512)
        feat_i = [self.cnn_ds_1(feat_i[i]) for i in range(n_views)]# [2](4,256,32,32)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][2].squeeze())#(8192,512)
        feat_p2i = self.fuse_p2i_ds_2[0](feat_p2i)#(8192,256)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_2[1])# [2](4,256,32,32)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][2])#(5895,32,256)
        feat_i2p = self.fuse_i2p_ds_2[0](feat_i2p.max(1)[0])#(5895,512)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_2[1])#(5895,512)

        feat_p = feat_p + feat_i2p#(5895,512)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,256,32,32)


        # upsample0
        for block_i, block_op in enumerate(self.pcd_up_0):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)#(17420,256)

        feat_i = [
            self.cnn_up_0(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-1][i]), dim=1)) for i in range(n_views)]#[2](4,128,64,64)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())#(32768,256)
        feat_p2i = self.fuse_p2i_up_0[0](feat_p2i)#(32768,128)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_up_0[1])#[2](4,128,64,64)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])# (17420,32,128)
        feat_i2p = self.fuse_i2p_up_0[0](feat_i2p.max(1)[0])# (17420,256)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_up_0[1])# (17420,256)

        feat_p = feat_p + feat_i2p # (17420,256)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]#[2](4,128,64,64)

        # upsample1
        for block_i, block_op in enumerate(self.pcd_up_1):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)#(130982,32)

        feat_i = [
            self.cnn_up_1(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-2][i]), dim=1)) for i in range(n_views)]#[2](4,32,128,128)

        pcs_X = batch['points_img']     #[2](4,16384,3)
        pcs_F = self.map(feat_i, feat_p, batch) #[2](4,16384,32)

        
        pcs_F = list(self.gau(pcs_F[0],pcs_F[1]))
        
        
        if vps is not None:
            # Drop first viewpoint -- assumed to be identity transformation
            vps = vps[1:]
        elif self.align_cfg.algorithm == "weighted_procrustes":
            vps = []
            cor_loss = []
            for i in range(1, n_views):
                corr_i = get_correspondences(
                    P1=pcs_F[0],
                    P2=pcs_F[i],
                    P1_X=pcs_X[0],
                    P2_X=pcs_X[i],
                    num_corres=self.num_corres,
                    ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
                )
                Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

                vps.append(Rt_i)
                cor_loss.append(cor_loss_i)

                # add for visualization
                output[f"corres_0{i}"] = corr_i
                output[f"vp_{i}"] = Rt_i
        else:
            raise ValueError(f"How to align using {self.align_cfg.algorithm}?")

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        # Get RGB pointcloud as well for direct rendering
        pcs_rgb = [rgb.view(B, 3, -1).permute(0, 2, 1).contiguous() for rgb in rgbs]

        projs = []
        # get joint for all values
        if self.pointcloud_source == "joint":
            pcs_X_joint = torch.cat(pcs_X, dim=1)
            pcs_F_joint = torch.cat(pcs_F, dim=1)
            pcs_RGB_joint = torch.cat(pcs_rgb, dim=1)
            pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

        # Rasterize and Blend
        for i in range(n_views):
            if self.pointcloud_source == "other":
                # get joint for all values except the one
                pcs_X_joint = torch.cat(pcs_X[0:i] + pcs_X[i + 1: n_views], dim=1)
                pcs_F_joint = torch.cat(pcs_F[0:i] + pcs_F[i + 1: n_views], dim=1)
                pcs_RGB_joint = torch.cat(
                    pcs_rgb[0:i] + pcs_rgb[i + 1: n_views], dim=1
                )
                pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

            if i > 0:
                rot_joint_X = transform_points_Rt(pcs_X_joint, vps[i - 1])
                rot_joint_X = points_to_ndc(rot_joint_X, K, (H, W))
            else:
                rot_joint_X = points_to_ndc(pcs_X_joint, K, (H, W))
            projs.append(self.renderer(rot_joint_X, pcs_FRGB_joint))

        # Decode
        for i in range(n_views):
            proj_FRGB_i = projs[i]["feats"]
            proj_RGB_i = proj_FRGB_i[:, -3:]
            proj_F_i = proj_FRGB_i[:, :-3]

            output[f"rgb_render_{i}"] = proj_RGB_i
            output[f"ras_depth_{i}"] = projs[i]["depth"]
            output[f"cover_{i}"] = projs[i]["mask"].unsqueeze(1)  # useless

        return output

    def gather_p2i(self, feat_p, idx):
        feat_p = torch.cat((feat_p, torch.zeros_like(feat_p[:1, :])), 0)
        return feat_p[idx]

    def gather_i2p(self, feat_i, idx):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        feat_i2p = []
        for i in range(B):
            feat_i2p.append(src_feat_i[i].reshape(C,H*W).permute(1,0))
            feat_i2p.append(tgt_feat_i[i].reshape(C,H*W).permute(1,0))

        feat_i2p = torch.cat(feat_i2p, 0)
        feat_i2p = torch.cat((feat_i2p, torch.zeros_like(feat_i2p[:1, :])), 0)
        return feat_i2p[idx]  #N,16,C

    def fusep2i(self, feat_i, feat_p2i, layer):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        # src_feat_i = src_feat_i.reshape(B, C, H * W).permute(0, 2, 1)
        # tgt_feat_i = tgt_feat_i.reshape(B, C, H * W).permute(0, 2, 1)

        src_feat_p2i = []
        tgt_feat_p2i = []

        for i in range(2*B):
            if i % 2 == 0:
                src_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))#[4](1,16384,64)
            else:
                tgt_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))#[4](1,16384,64)

        src_feat_p2i = torch.vstack(src_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)#(4,64,128,128)
        tgt_feat_p2i = torch.vstack(tgt_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)#(4,64,128,128)

        src_feat_p2i = torch.cat([src_feat_i, src_feat_p2i], 1)#(4,128,128,128)
        tgt_feat_p2i = torch.cat([tgt_feat_i, tgt_feat_p2i], 1)#(4,128,128,128)

        src_feat_p2i = layer(src_feat_p2i)#(4,64,128,128)
        tgt_feat_p2i = layer(tgt_feat_p2i)#(4,64,128,128)

        return [src_feat_p2i, tgt_feat_p2i]

    def fusei2p(self, feat_p, feat_i2p, layer):
        feat_i2p = torch.cat([feat_p, feat_i2p], -1)
        feat_i2p = layer(feat_i2p)
        return feat_i2p


    def generate_pointclouds(self, K, deps, vps=None):
        n_views = len(deps)
        # generate pointclouds - generate grid once for efficiency
        B, _, H, W = deps[0].shape
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pcs_X = [
            grid_to_pointcloud(K_inv, deps[i], None, grid)[0] for i in range(n_views)
        ]

        if vps is not None:
            pcs_X_rot = [
                transform_points_Rt(pcs_X[i + 1], vps[i + 1], inverse=True, )
                for i in range(n_views - 1)
            ]
            pcs_X = pcs_X[0:1] + pcs_X_rot
            pcs_X = torch.cat(pcs_X, dim=1).detach().cpu()

        return pcs_X

    def get_feature_pcs(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert (
                feats[0].shape[-1] == deps[0].shape[-1]
        ), f"Same size {feats[0].shape} - {deps[0].shape}"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]
        return pcs_X, pcs_F, None


from models.FFMT.mca.mca import MutualCrossAttention
class PCReg_KPURes18_MSF_mca0(nn.Module):
    def __init__(self, cfg):
        super(PCReg_KPURes18_MSF_mca0, self).__init__()
        # set encoder decoder
        chan_in = 3
        self.cfg = cfg
        feat_dim = cfg.feat_dim

        # No imagenet pretraining
        pretrained = False

        encode_I = URes18Encoder1(chan_in, feat_dim, pretrained)
        encode_P = KPFCN(cfg)
        self.cnn_pre_stages = nn.Sequential(
            encode_I.inconv,
            encode_I.layer1
        )
        self.pcd_pre_stages = nn.ModuleList()
        for i in range(0,2):
            self.pcd_pre_stages.append(encode_P.encoder_blocks[i])

        self.cnn_ds_0 = encode_I.layer2
        self.pcd_ds_0 = nn.ModuleList()
        for i in range(2,5):
            self.pcd_ds_0.append(encode_P.encoder_blocks[i])

        self.cnn_ds_1 = encode_I.layer3
        self.pcd_ds_1 = nn.ModuleList()
        for i in range(5,8):
            self.pcd_ds_1.append(encode_P.encoder_blocks[i])

        self.cnn_up_0 = encode_I.up1
        self.pcd_up_0 = nn.ModuleList()
        for i in range(0, 2):
            self.pcd_up_0.append(encode_P.decoder_blocks[i])

        self.cnn_up_1 = nn.Sequential(
            encode_I.up2,
            encode_I.outconv
        )
        self.pcd_up_1 = nn.ModuleList()
        for i in range(2, 4):
            self.pcd_up_1.append(encode_P.decoder_blocks[i])

        # 0
        self.fuse_p2i_ds_0 = nn.ModuleList()
        self.fuse_p2i_ds_0.append(
            UnaryBlock(128, 64, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_0.append(
            nn.Sequential(
                nn.Conv2d(64*2, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_0 = nn.ModuleList()
        self.fuse_i2p_ds_0.append(
            UnaryBlock(64, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_0.append(
            UnaryBlock(128 * 2, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 1
        self.fuse_p2i_ds_1 = nn.ModuleList()
        self.fuse_p2i_ds_1.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_1.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_1 = nn.ModuleList()
        self.fuse_i2p_ds_1.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_1.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 2
        self.fuse_p2i_ds_2 = nn.ModuleList()
        self.fuse_p2i_ds_2.append(
            UnaryBlock(512, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_2.append(
            nn.Sequential(
                nn.Conv2d(256 * 2, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_2 = nn.ModuleList()
        self.fuse_i2p_ds_2.append(
            UnaryBlock(256, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_2.append(
            UnaryBlock(512 * 2, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        self.fuse_p2i_up_0 = nn.ModuleList()
        self.fuse_p2i_up_0.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_up_0.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
        ))
        self.fuse_i2p_up_0 = nn.ModuleList()
        self.fuse_i2p_up_0.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_up_0.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # self.fuse_p2i_up_1 = nn.ModuleList()
        # self.fuse_i2p_up_1 = nn.ModuleList()
        self.align_cfg = cfg
        self.renderer = PointsRenderer(cfg)
        self.num_corres = cfg.num_correspodances
        self.pointcloud_source = cfg.pointcloud_source
        
        self.map = Fusion_CATL(feat_dim)
        
        
        # self.lo1 = LocalFeatureTransformer({'d_model':32,
        #         'nhead':8,
        #         'layer_names':['self'] * 1,
        #         'attention':'linear'})
        # self.lo2 = LocalFeatureTransformer({'d_model':32,
        #         'nhead':8,
        #         'layer_names':['self'] * 1,
        #         'attention':'linear'})
        # self.lo3 = LocalFeatureTransformer({'d_model':32,
        #         'nhead':8,
        #         'layer_names':['self'] * 1,
        #         'attention':'linear'})
        # self.lo4 = LocalFeatureTransformer({'d_model':32,
        #         'nhead':8,
        #         'layer_names':['self'] * 1,
        #         'attention':'linear'})
        self.mca1 = MutualCrossAttention(0.3)
        # self.mca2 = MutualCrossAttention(0.3)
        # self.mca3 = MutualCrossAttention(0.3)
        # self.mca4 = MutualCrossAttention(0.3)
        
        

    def forward(self, batch, rgbs, K, deps, vps=None):
        # Estimate Depth -- now for 1 and 2
        n_views = 2
        output = {}
        B, _, H, W = rgbs[0].shape

        # Encode features
        feat_p_encode = []
        feat_i_encode = []

        #pre stage
        feat_p = batch['features'].clone().detach() #全为1 (130982,1)
        for block_op in self.pcd_pre_stages:
            feat_p = block_op(feat_p, batch)# feat_p (130982,128)
            
        
        feat_i = [self.cnn_pre_stages(rgbs[i]) for i in range(n_views)] # feat_i [2](4,64,128,128)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][0].squeeze())#(131072,128)
        feat_p2i = self.fuse_p2i_ds_0[0](feat_p2i)#(131072,64)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_0[1])# [2](4,64,128,128)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][0])#(130982,32,64)
        feat_i2p = self.fuse_i2p_ds_0[0](feat_i2p.max(1)[0])#(130982,128)类似于pointnet的最大池化
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_0[1],B)#(130982,128)

        feat_p = feat_p + feat_i2p#(130982,128)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,64,128,128)

        feat_p_encode.append(feat_p)# [1](130982,128)
        feat_i_encode.append(feat_i)# [1][2](4,64,128,128)

        #downsample 0
        for block_op in self.pcd_ds_0:
            feat_p = block_op(feat_p, batch)# (17420,256)
        feat_i = [self.cnn_ds_0(feat_i[i]) for i in range(n_views)]# [2](4,128,64,64)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())#(32768,256)
        feat_p2i = self.fuse_p2i_ds_1[0](feat_p2i)#(32768,128)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_1[1])# [2](4,128,64,64)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])# (17420,32,128)
        feat_i2p = self.fuse_i2p_ds_1[0](feat_i2p.max(1)[0])# (17420,256)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_1[1],B)# (17420,256)

        feat_p = feat_p + feat_i2p# (17420,256)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,128,64,64)

        feat_p_encode.append(feat_p)# [(130982,128)(17420,256)]
        feat_i_encode.append(feat_i)# [[2](4,64,128,128),[2](4,128,64,64)]

        #downsample 1
        for block_op in self.pcd_ds_1:
            feat_p = block_op(feat_p, batch)# (5895,512)
        feat_i = [self.cnn_ds_1(feat_i[i]) for i in range(n_views)]# [2](4,256,32,32)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][2].squeeze())#(8192,512)
        feat_p2i = self.fuse_p2i_ds_2[0](feat_p2i)#(8192,256)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_2[1])# [2](4,256,32,32)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][2])#(5895,32,256)
        feat_i2p = self.fuse_i2p_ds_2[0](feat_i2p.max(1)[0])#(5895,512)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_2[1],B)#(5895,512)

        feat_p = feat_p + feat_i2p#(5895,512)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,256,32,32)


        # upsample0
        for block_i, block_op in enumerate(self.pcd_up_0):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)#(17420,256)

        feat_i = [
            self.cnn_up_0(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-1][i]), dim=1)) for i in range(n_views)]#[2](4,128,64,64)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())#(32768,256)
        feat_p2i = self.fuse_p2i_up_0[0](feat_p2i)#(32768,128)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_up_0[1])#[2](4,128,64,64)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])# (17420,32,128)
        feat_i2p = self.fuse_i2p_up_0[0](feat_i2p.max(1)[0])# (17420,256)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_up_0[1],B)# (17420,256)

        feat_p = feat_p + feat_i2p # (17420,256)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]#[2](4,128,64,64)

        # upsample1
        for block_i, block_op in enumerate(self.pcd_up_1):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)#(130982,32)

        feat_i = [
            self.cnn_up_1(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-2][i]), dim=1)) for i in range(n_views)]#[2](4,32,128,128)

        pcs_X = batch['points_img']     #[2](4,16384,3)
        pcs_F = self.map(feat_i, feat_p, batch) #[2](4,16384,32)

        # pcs_F_sc = [self.lo1(pcs_F[0],pcs_F[1])]
        # pcs_F_sc = [self.mca1(pcs_F_sc[0],pcs_F_sc[1])]
        # pcs_F_sc = [self.lo2(pcs_F_sc[0],pcs_F_sc[1])]
        # pcs_F_sc = [self.mca2(pcs_F_sc[0],pcs_F_sc[1])]
        # pcs_F_sc = [self.lo3(pcs_F_sc[0],pcs_F_sc[1])]
        # pcs_F_sc = [self.mca3(pcs_F_sc[0],pcs_F_sc[1])]
        # pcs_F_sc = [self.lo4(pcs_F_sc[0],pcs_F_sc[1])]
        # pcs_F_sc = [self.mca4(pcs_F_sc[0],pcs_F_sc[1])]
        # # pcs_F_sc = [self.lo1(pcs_F[0],pcs_F[1])]
        # # pcs_F_sc = [self.mca1(pcs_F_sc)]
        # # pcs_F_sc = [self.lo2(pcs_F_sc)]
        # # pcs_F_sc = [self.mca2(pcs_F_sc)]
        # # pcs_F_sc = [self.lo3(pcs_F_sc)]
        # # pcs_F_sc = [self.mca3(pcs_F_sc)]
        # # pcs_F_sc = [self.lo4(pcs_F_sc)]
        # # pcs_F_sc = [self.mca4(pcs_F_sc)]
        # pcs_F += pcs_F_sc
        
        if vps is not None:
            # Drop first viewpoint -- assumed to be identity transformation
            vps = vps[1:]
        elif self.align_cfg.algorithm == "weighted_procrustes":
            vps = []
            cor_loss = []
            for i in range(1, n_views):
                corr_i = get_correspondences(
                    P1=pcs_F[0],
                    P2=pcs_F[i],
                    P1_X=pcs_X[0],
                    P2_X=pcs_X[i],
                    num_corres=self.num_corres,
                    ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
                )
                Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

                vps.append(Rt_i)
                cor_loss.append(cor_loss_i)

                # add for visualization
                output[f"corres_0{i}"] = corr_i
                output[f"vp_{i}"] = Rt_i
        else:
            raise ValueError(f"How to align using {self.align_cfg.algorithm}?")

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        # Get RGB pointcloud as well for direct rendering
        pcs_rgb = [rgb.view(B, 3, -1).permute(0, 2, 1).contiguous() for rgb in rgbs]

        projs = []
        # get joint for all values
        if self.pointcloud_source == "joint":
            pcs_X_joint = torch.cat(pcs_X, dim=1)
            pcs_F_joint = torch.cat(pcs_F, dim=1)
            pcs_RGB_joint = torch.cat(pcs_rgb, dim=1)
            pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

        # Rasterize and Blend
        for i in range(n_views):
            if self.pointcloud_source == "other":
                # get joint for all values except the one
                pcs_X_joint = torch.cat(pcs_X[0:i] + pcs_X[i + 1: n_views], dim=1)
                pcs_F_joint = torch.cat(pcs_F[0:i] + pcs_F[i + 1: n_views], dim=1)
                pcs_RGB_joint = torch.cat(
                    pcs_rgb[0:i] + pcs_rgb[i + 1: n_views], dim=1
                )
                pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

            if i > 0:
                rot_joint_X = transform_points_Rt(pcs_X_joint, vps[i - 1])
                rot_joint_X = points_to_ndc(rot_joint_X, K, (H, W))
            else:
                rot_joint_X = points_to_ndc(pcs_X_joint, K, (H, W))
            projs.append(self.renderer(rot_joint_X, pcs_FRGB_joint))

        # Decode
        for i in range(n_views):
            proj_FRGB_i = projs[i]["feats"]
            proj_RGB_i = proj_FRGB_i[:, -3:]
            proj_F_i = proj_FRGB_i[:, :-3]

            output[f"rgb_render_{i}"] = proj_RGB_i
            output[f"ras_depth_{i}"] = projs[i]["depth"]
            output[f"cover_{i}"] = projs[i]["mask"].unsqueeze(1)  # useless

        return output

    def gather_p2i(self, feat_p, idx):
        feat_p = torch.cat((feat_p, torch.zeros_like(feat_p[:1, :])), 0)
        return feat_p[idx]

    def gather_i2p(self, feat_i, idx):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        feat_i2p = []
        for i in range(B):
            feat_i2p.append(src_feat_i[i].reshape(C,H*W).permute(1,0))
            feat_i2p.append(tgt_feat_i[i].reshape(C,H*W).permute(1,0))

        feat_i2p = torch.cat(feat_i2p, 0)
        feat_i2p = torch.cat((feat_i2p, torch.zeros_like(feat_i2p[:1, :])), 0)
        return feat_i2p[idx]  #N,16,C

    def fusep2i(self, feat_i, feat_p2i, layer):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        # src_feat_i = src_feat_i.reshape(B, C, H * W).permute(0, 2, 1)
        # tgt_feat_i = tgt_feat_i.reshape(B, C, H * W).permute(0, 2, 1)

        src_feat_p2i = []
        tgt_feat_p2i = []

        for i in range(2*B):
            if i % 2 == 0:
                src_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))#[4](1,16384,64)
            else:
                tgt_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))#[4](1,16384,64)

        src_feat_p2i = torch.vstack(src_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)#(4,64,128,128)
        tgt_feat_p2i = torch.vstack(tgt_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)#(4,64,128,128)

        src_feat_p2i = torch.cat([src_feat_i, src_feat_p2i], 1)#(4,128,128,128)
        tgt_feat_p2i = torch.cat([tgt_feat_i, tgt_feat_p2i], 1)#(4,128,128,128)

        src_feat_p2i = layer(src_feat_p2i)#(4,64,128,128)
        tgt_feat_p2i = layer(tgt_feat_p2i)#(4,64,128,128)

        return [src_feat_p2i, tgt_feat_p2i]

    def fusei2p(self, feat_p, feat_i2p, layer,B):
        # feat_i2p = torch.cat([feat_p, feat_i2p], -1)
        # feat_i2p = layer(feat_i2p)
        # return feat_i2p
        # print(feat_i2p.size(),feat_p.size())
        feat_p = torch.chunk(feat_p,B,dim = 0)
        feat_p = torch.stack([feat_p[i] for i in range(B)],dim=0)
        feat_i2p = torch.chunk(feat_i2p,B,dim = 0)
        feat_i2p = torch.stack([feat_i2p[i] for i in range(B)],dim=0)
        
        res = self.mca1(feat_p,feat_i2p)
        res = torch.cat([res[i] for i in range(res.size()[0])],dim=0)
        return res


    def generate_pointclouds(self, K, deps, vps=None):
        n_views = len(deps)
        # generate pointclouds - generate grid once for efficiency
        B, _, H, W = deps[0].shape
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pcs_X = [
            grid_to_pointcloud(K_inv, deps[i], None, grid)[0] for i in range(n_views)
        ]

        if vps is not None:
            pcs_X_rot = [
                transform_points_Rt(pcs_X[i + 1], vps[i + 1], inverse=True, )
                for i in range(n_views - 1)
            ]
            pcs_X = pcs_X[0:1] + pcs_X_rot
            pcs_X = torch.cat(pcs_X, dim=1).detach().cpu()

        return pcs_X

    def get_feature_pcs(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert (
                feats[0].shape[-1] == deps[0].shape[-1]
        ), f"Same size {feats[0].shape} - {deps[0].shape}"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]
        return pcs_X, pcs_F, None
    
    
class PCReg_KPURes18_MSF_loftr6(nn.Module):
    def __init__(self, cfg):
        super(PCReg_KPURes18_MSF_loftr6, self).__init__()
        # set encoder decoder
        chan_in = 3
        self.cfg = cfg
        feat_dim = cfg.feat_dim

        # No imagenet pretraining
        pretrained = False

        encode_I = URes18Encoder1(chan_in, feat_dim, pretrained)
        encode_P = KPFCN(cfg)
        self.cnn_pre_stages = nn.Sequential(
            encode_I.inconv,
            encode_I.layer1
        )
        self.pcd_pre_stages = nn.ModuleList()
        for i in range(0,2):
            self.pcd_pre_stages.append(encode_P.encoder_blocks[i])

        self.cnn_ds_0 = encode_I.layer2
        self.pcd_ds_0 = nn.ModuleList()
        for i in range(2,5):
            self.pcd_ds_0.append(encode_P.encoder_blocks[i])

        self.cnn_ds_1 = encode_I.layer3
        self.pcd_ds_1 = nn.ModuleList()
        for i in range(5,8):
            self.pcd_ds_1.append(encode_P.encoder_blocks[i])

        self.cnn_up_0 = encode_I.up1
        self.pcd_up_0 = nn.ModuleList()
        for i in range(0, 2):
            self.pcd_up_0.append(encode_P.decoder_blocks[i])

        self.cnn_up_1 = nn.Sequential(
            encode_I.up2,
            encode_I.outconv
        )
        self.pcd_up_1 = nn.ModuleList()
        for i in range(2, 4):
            self.pcd_up_1.append(encode_P.decoder_blocks[i])

        # 0
        self.fuse_p2i_ds_0 = nn.ModuleList()
        self.fuse_p2i_ds_0.append(
            UnaryBlock(128, 64, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_0.append(
            nn.Sequential(
                nn.Conv2d(64*2, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_0 = nn.ModuleList()
        self.fuse_i2p_ds_0.append(
            UnaryBlock(64, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_0.append(
            UnaryBlock(128 * 2, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 1
        self.fuse_p2i_ds_1 = nn.ModuleList()
        self.fuse_p2i_ds_1.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_1.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_1 = nn.ModuleList()
        self.fuse_i2p_ds_1.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_1.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 2
        self.fuse_p2i_ds_2 = nn.ModuleList()
        self.fuse_p2i_ds_2.append(
            UnaryBlock(512, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_2.append(
            nn.Sequential(
                nn.Conv2d(256 * 2, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_2 = nn.ModuleList()
        self.fuse_i2p_ds_2.append(
            UnaryBlock(256, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_2.append(
            UnaryBlock(512 * 2, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        self.fuse_p2i_up_0 = nn.ModuleList()
        self.fuse_p2i_up_0.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_up_0.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
        ))
        self.fuse_i2p_up_0 = nn.ModuleList()
        self.fuse_i2p_up_0.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_up_0.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # self.fuse_p2i_up_1 = nn.ModuleList()
        # self.fuse_i2p_up_1 = nn.ModuleList()
        self.align_cfg = cfg
        self.renderer = PointsRenderer(cfg)
        self.num_corres = cfg.num_correspodances
        self.pointcloud_source = cfg.pointcloud_source
        self.map = Fusion_CATL(feat_dim)
        
        
        self.loftr_coarse4 = LocalFeatureTransformer({'d_model':32,
                'nhead':8,
                'layer_names':['self', 'cross'] * 4,
                'attention':'linear'})
        
        self.self_attn_p1 = LoFTREncoderLayer(d_model=128,nhead=8)
        self.self_attn_i2p1 = LoFTREncoderLayer(d_model=128,nhead=8)
        self.self_attn_i1 = LoFTREncoderLayer(d_model=64,nhead=8)
        self.self_attn_p2i1 = LoFTREncoderLayer(d_model=64,nhead=8)
        
        self.self_attn_p2 = LoFTREncoderLayer(d_model=256,nhead=8)
        self.self_attn_i2p2 = LoFTREncoderLayer(d_model=256,nhead=8)
        self.self_attn_i2 = LoFTREncoderLayer(d_model=128,nhead=8)
        self.self_attn_p2i2 = LoFTREncoderLayer(d_model=128,nhead=8)
        
        self.self_attn_p3 = LoFTREncoderLayer(d_model=512,nhead=8)
        self.self_attn_i2p3 = LoFTREncoderLayer(d_model=512,nhead=8)
        self.self_attn_i3 = LoFTREncoderLayer(d_model=256,nhead=8)
        self.self_attn_p2i3 = LoFTREncoderLayer(d_model=256,nhead=8)
        
        self.self_attn_p4 = LoFTREncoderLayer(d_model=256,nhead=8)
        self.self_attn_i2p4 = LoFTREncoderLayer(d_model=256,nhead=8)
        self.self_attn_i4 = LoFTREncoderLayer(d_model=128,nhead=8)
        self.self_attn_p2i4 = LoFTREncoderLayer(d_model=128,nhead=8)
        
        # self.self_attn_p5 = LoFTREncoderLayer(d_model=128,nhead=8)
        # self.self_attn_i2p5 = LoFTREncoderLayer(d_model=128,nhead=8)
        # self.self_attn_i5 = LoFTREncoderLayer(d_model=64,nhead=8)
        # self.self_attn_p2i5 = LoFTREncoderLayer(d_model=64,nhead=8)
    def selfp(self,p,layer,num_view):
        p1 = torch.chunk(p,num_view,dim=0)
        p1 = [torch.reshape(p1[i],(1,p1[i].size()[0],p1[i].size()[1])) for i in range(num_view)]
        p1 = [layer(p1[i],p1[i]).squeeze() for i in range(num_view)]
        p2 = torch.cat(p1,dim=0)
        return p2
    
    def selfi(self,feat_i,layer,num_view):
        B,C,H,W = feat_i[0].size()
        
        for i in range(num_view):
            feat_i[i] = feat_i[i].reshape(B,C,H*W).permute(0,2,1)
            feat_i[i] = layer(feat_i[i],feat_i[i])
            feat_i[i] = feat_i[i].permute(0,2,1).reshape(B,C,H,W)
        return feat_i
    
    def crossp(self,p,layer):
        p1 = torch.chunk(p,2,dim=0)
        p1 = [torch.reshape(p1[i],(1,p1[i].size()[0],p1[i].size()[1])) for i in range(2)]
        p1 = [layer(p1[0],p1[1]).squeeze(),layer(p1[1],p1[0]).squeeze()]
        p2 = torch.cat(p1,dim=0)
        return p2
    
    def crossi(self,feat_i,layer):
        B,C,H,W = feat_i[0].size()
        for i in range(2):
            feat_i[i] = feat_i[i].reshape(B,C,H*W).permute(0,2,1)
            feat_i[i] = layer(feat_i[i],feat_i[i+1%2])
            feat_i[i] = feat_i[i].permute(0,2,1).reshape(B,C,H,W)
        return feat_i
    
    def forward(self, batch, rgbs, K, deps, vps=None):
        # Estimate Depth -- now for 1 and 2
        n_views = 2
        output = {}
        B, _, H, W = rgbs[0].shape

        # Encode features
        feat_p_encode = []
        feat_i_encode = []

        #pre stage
        feat_p = batch['features'].clone().detach() #全为1 (130982,1)
        for block_op in self.pcd_pre_stages:
            feat_p = block_op(feat_p, batch)# feat_p (130982,128)
        
        
        feat_i = [self.cnn_pre_stages(rgbs[i]) for i in range(n_views)] # feat_i [2](4,64,128,128)
        

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][0].squeeze())#(131072,128)
        feat_p2i = self.fuse_p2i_ds_0[0](feat_p2i)#(131072,64)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_0[1])# [2](4,64,128,128)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][0])#(130982,32,64)
        feat_i2p = self.fuse_i2p_ds_0[0](feat_i2p.max(1)[0])#(130982,128)类似于pointnet的最大池化
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_0[1])#(130982,128)

        feat_p = self.selfp(feat_p,self.self_attn_p1,n_views)
        feat_i = self.selfi(feat_i,self.self_attn_i1,n_views)
        feat_i2p = self.selfp(feat_i2p,self.self_attn_i2p1,n_views)
        feat_p2i = self.selfi(feat_p2i,self.self_attn_p2i1,n_views)
        
        
        feat_p = feat_p + feat_i2p#(130982,128)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,64,128,128)
        
        

        feat_p_encode.append(feat_p)# [1](130982,128)
        feat_i_encode.append(feat_i)# [1][2](4,64,128,128)

        #downsample 0
        for block_op in self.pcd_ds_0:
            feat_p = block_op(feat_p, batch)# (17420,256)
        feat_i = [self.cnn_ds_0(feat_i[i]) for i in range(n_views)]# [2](4,128,64,64)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())#(32768,256)
        feat_p2i = self.fuse_p2i_ds_1[0](feat_p2i)#(32768,128)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_1[1])# [2](4,128,64,64)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])# (17420,32,128)
        feat_i2p = self.fuse_i2p_ds_1[0](feat_i2p.max(1)[0])# (17420,256)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_1[1])# (17420,256)

        feat_p = self.selfp(feat_p,self.self_attn_p2,n_views)
        feat_i = self.selfi(feat_i,self.self_attn_i2,n_views)
        feat_i2p = self.selfp(feat_i2p,self.self_attn_i2p2,n_views)
        feat_p2i = self.selfi(feat_p2i,self.self_attn_p2i2,n_views)
        
        feat_p = feat_p + feat_i2p# (17420,256)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,128,64,64)

        feat_p_encode.append(feat_p)# [(130982,128)(17420,256)]
        feat_i_encode.append(feat_i)# [[2](4,64,128,128),[2](4,128,64,64)]

        #downsample 1
        for block_op in self.pcd_ds_1:
            feat_p = block_op(feat_p, batch)# (5895,512)
        feat_i = [self.cnn_ds_1(feat_i[i]) for i in range(n_views)]# [2](4,256,32,32)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][2].squeeze())#(8192,512)
        feat_p2i = self.fuse_p2i_ds_2[0](feat_p2i)#(8192,256)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_2[1])# [2](4,256,32,32)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][2])#(5895,32,256)
        feat_i2p = self.fuse_i2p_ds_2[0](feat_i2p.max(1)[0])#(5895,512)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_2[1])#(5895,512)
        
        feat_p = self.selfp(feat_p,self.self_attn_p3,n_views)
        feat_i = self.selfi(feat_i,self.self_attn_i3,n_views)
        feat_i2p = self.selfp(feat_i2p,self.self_attn_i2p3,n_views)
        feat_p2i = self.selfi(feat_p2i,self.self_attn_p2i3,n_views)

        feat_p = feat_p + feat_i2p#(5895,512)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,256,32,32)

        
        # upsample0
        for block_i, block_op in enumerate(self.pcd_up_0):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)#(17420,256)

        feat_i = [
            self.cnn_up_0(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-1][i]), dim=1)) for i in range(n_views)]#[2](4,128,64,64)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())#(32768,256)
        feat_p2i = self.fuse_p2i_up_0[0](feat_p2i)#(32768,128)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_up_0[1])#[2](4,128,64,64)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])# (17420,32,128)
        feat_i2p = self.fuse_i2p_up_0[0](feat_i2p.max(1)[0])# (17420,256)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_up_0[1])# (17420,256)

        feat_p = self.selfp(feat_p,self.self_attn_p4,n_views)
        feat_i = self.selfi(feat_i,self.self_attn_i4,n_views)
        feat_i2p = self.selfp(feat_i2p,self.self_attn_i2p4,n_views)
        feat_p2i = self.selfi(feat_p2i,self.self_attn_p2i4,n_views)
        
        feat_p = feat_p + feat_i2p # (17420,256)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]#[2](4,128,64,64)

        # upsample1
        for block_i, block_op in enumerate(self.pcd_up_1):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)#(130982,32)

        feat_i = [
            self.cnn_up_1(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-2][i]), dim=1)) for i in range(n_views)]#[2](4,32,128,128)

        pcs_X = batch['points_img']     #[2](4,16384,3)
        pcs_F = self.map(feat_i, feat_p, batch) #[2](4,16384,32)

        pcs_F_sc4 = list(self.loftr_coarse4(pcs_F[0],pcs_F[1]))
        pcs_F = pcs_F+pcs_F_sc4
        
        if vps is not None:
            # Drop first viewpoint -- assumed to be identity transformation
            vps = vps[1:]
        elif self.align_cfg.algorithm == "weighted_procrustes":
            vps = []
            cor_loss = []
            for i in range(1, n_views):
                corr_i = get_correspondences(
                    P1=pcs_F[0],
                    P2=pcs_F[i],
                    P1_X=pcs_X[0],
                    P2_X=pcs_X[i],
                    num_corres=self.num_corres,
                    ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
                )
                Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

                vps.append(Rt_i)
                cor_loss.append(cor_loss_i)

                # add for visualization
                output[f"corres_0{i}"] = corr_i
                output[f"vp_{i}"] = Rt_i
        else:
            raise ValueError(f"How to align using {self.align_cfg.algorithm}?")

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        # Get RGB pointcloud as well for direct rendering
        pcs_rgb = [rgb.view(B, 3, -1).permute(0, 2, 1).contiguous() for rgb in rgbs]

        projs = []
        # get joint for all values
        if self.pointcloud_source == "joint":
            pcs_X_joint = torch.cat(pcs_X, dim=1)
            pcs_F_joint = torch.cat(pcs_F, dim=1)
            pcs_RGB_joint = torch.cat(pcs_rgb, dim=1)
            pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

        # Rasterize and Blend
        for i in range(n_views):
            if self.pointcloud_source == "other":
                # get joint for all values except the one
                pcs_X_joint = torch.cat(pcs_X[0:i] + pcs_X[i + 1: n_views], dim=1)
                pcs_F_joint = torch.cat(pcs_F[0:i] + pcs_F[i + 1: n_views], dim=1)
                pcs_RGB_joint = torch.cat(
                    pcs_rgb[0:i] + pcs_rgb[i + 1: n_views], dim=1
                )
                pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

            if i > 0:
                rot_joint_X = transform_points_Rt(pcs_X_joint, vps[i - 1])
                rot_joint_X = points_to_ndc(rot_joint_X, K, (H, W))
            else:
                rot_joint_X = points_to_ndc(pcs_X_joint, K, (H, W))
            projs.append(self.renderer(rot_joint_X, pcs_FRGB_joint))

        # Decode
        for i in range(n_views):
            proj_FRGB_i = projs[i]["feats"]
            proj_RGB_i = proj_FRGB_i[:, -3:]
            proj_F_i = proj_FRGB_i[:, :-3]

            output[f"rgb_render_{i}"] = proj_RGB_i
            output[f"ras_depth_{i}"] = projs[i]["depth"]
            output[f"cover_{i}"] = projs[i]["mask"].unsqueeze(1)  # useless

        return output

    def gather_p2i(self, feat_p, idx):
        feat_p = torch.cat((feat_p, torch.zeros_like(feat_p[:1, :])), 0)
        return feat_p[idx]

    def gather_i2p(self, feat_i, idx):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        feat_i2p = []
        for i in range(B):
            feat_i2p.append(src_feat_i[i].reshape(C,H*W).permute(1,0))
            feat_i2p.append(tgt_feat_i[i].reshape(C,H*W).permute(1,0))

        feat_i2p = torch.cat(feat_i2p, 0)
        feat_i2p = torch.cat((feat_i2p, torch.zeros_like(feat_i2p[:1, :])), 0)
        return feat_i2p[idx]  #N,16,C

    def fusep2i(self, feat_i, feat_p2i, layer):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        # src_feat_i = src_feat_i.reshape(B, C, H * W).permute(0, 2, 1)
        # tgt_feat_i = tgt_feat_i.reshape(B, C, H * W).permute(0, 2, 1)

        src_feat_p2i = []
        tgt_feat_p2i = []

        for i in range(2*B):
            if i % 2 == 0:
                src_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))#[4](1,16384,64)
            else:
                tgt_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))#[4](1,16384,64)

        src_feat_p2i = torch.vstack(src_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)#(4,64,128,128)
        tgt_feat_p2i = torch.vstack(tgt_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)#(4,64,128,128)

        src_feat_p2i = torch.cat([src_feat_i, src_feat_p2i], 1)#(4,128,128,128)
        tgt_feat_p2i = torch.cat([tgt_feat_i, tgt_feat_p2i], 1)#(4,128,128,128)

        src_feat_p2i = layer(src_feat_p2i)#(4,64,128,128)
        tgt_feat_p2i = layer(tgt_feat_p2i)#(4,64,128,128)

        return [src_feat_p2i, tgt_feat_p2i]

    def fusei2p(self, feat_p, feat_i2p, layer):
        feat_i2p = torch.cat([feat_p, feat_i2p], -1)
        feat_i2p = layer(feat_i2p)
        return feat_i2p


    def generate_pointclouds(self, K, deps, vps=None):
        n_views = len(deps)
        # generate pointclouds - generate grid once for efficiency
        B, _, H, W = deps[0].shape
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pcs_X = [
            grid_to_pointcloud(K_inv, deps[i], None, grid)[0] for i in range(n_views)
        ]

        if vps is not None:
            pcs_X_rot = [
                transform_points_Rt(pcs_X[i + 1], vps[i + 1], inverse=True, )
                for i in range(n_views - 1)
            ]
            pcs_X = pcs_X[0:1] + pcs_X_rot
            pcs_X = torch.cat(pcs_X, dim=1).detach().cpu()

        return pcs_X

    def get_feature_pcs(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert (
                feats[0].shape[-1] == deps[0].shape[-1]
        ), f"Same size {feats[0].shape} - {deps[0].shape}"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]
        return pcs_X, pcs_F, None
    
class PCReg_KPURes18_MSF_loftr7(nn.Module):
    def __init__(self, cfg):
        super(PCReg_KPURes18_MSF_loftr7, self).__init__()
        # set encoder decoder
        chan_in = 3
        self.cfg = cfg
        feat_dim = cfg.feat_dim

        # No imagenet pretraining
        pretrained = False

        encode_I = URes18Encoder1(chan_in, feat_dim, pretrained)
        encode_P = KPFCN(cfg)
        self.cnn_pre_stages = nn.Sequential(
            encode_I.inconv,
            encode_I.layer1
        )
        self.pcd_pre_stages = nn.ModuleList()
        for i in range(0,2):
            self.pcd_pre_stages.append(encode_P.encoder_blocks[i])

        self.cnn_ds_0 = encode_I.layer2
        self.pcd_ds_0 = nn.ModuleList()
        for i in range(2,5):
            self.pcd_ds_0.append(encode_P.encoder_blocks[i])

        self.cnn_ds_1 = encode_I.layer3
        self.pcd_ds_1 = nn.ModuleList()
        for i in range(5,8):
            self.pcd_ds_1.append(encode_P.encoder_blocks[i])

        self.cnn_up_0 = encode_I.up1
        self.pcd_up_0 = nn.ModuleList()
        for i in range(0, 2):
            self.pcd_up_0.append(encode_P.decoder_blocks[i])

        self.cnn_up_1 = nn.Sequential(
            encode_I.up2,
            encode_I.outconv
        )
        self.pcd_up_1 = nn.ModuleList()
        for i in range(2, 4):
            self.pcd_up_1.append(encode_P.decoder_blocks[i])

        # 0
        self.fuse_p2i_ds_0 = nn.ModuleList()
        self.fuse_p2i_ds_0.append(
            UnaryBlock(128, 64, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_0.append(
            nn.Sequential(
                nn.Conv2d(64*2, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_0 = nn.ModuleList()
        self.fuse_i2p_ds_0.append(
            UnaryBlock(64, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_0.append(
            UnaryBlock(128 * 2, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 1
        self.fuse_p2i_ds_1 = nn.ModuleList()
        self.fuse_p2i_ds_1.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_1.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_1 = nn.ModuleList()
        self.fuse_i2p_ds_1.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_1.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 2
        self.fuse_p2i_ds_2 = nn.ModuleList()
        self.fuse_p2i_ds_2.append(
            UnaryBlock(512, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_2.append(
            nn.Sequential(
                nn.Conv2d(256 * 2, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_2 = nn.ModuleList()
        self.fuse_i2p_ds_2.append(
            UnaryBlock(256, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_2.append(
            UnaryBlock(512 * 2, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        self.fuse_p2i_up_0 = nn.ModuleList()
        self.fuse_p2i_up_0.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_up_0.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
        ))
        self.fuse_i2p_up_0 = nn.ModuleList()
        self.fuse_i2p_up_0.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_up_0.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # self.fuse_p2i_up_1 = nn.ModuleList()
        # self.fuse_i2p_up_1 = nn.ModuleList()
        self.align_cfg = cfg
        self.renderer = PointsRenderer(cfg)
        self.num_corres = cfg.num_correspodances
        self.pointcloud_source = cfg.pointcloud_source
        self.map = Fusion_CATL(feat_dim)
        
        
        self.loftr_coarse4 = LocalFeatureTransformer({'d_model':32,
                'nhead':8,
                'layer_names':['self', 'cross'] * 4,
                'attention':'linear'})
        
        self.self_attn_p1 = LoFTREncoderLayer(d_model=128,nhead=8)
        self.self_attn_i2p1 = LoFTREncoderLayer(d_model=128,nhead=8)
        self.self_attn_i1 = LoFTREncoderLayer(d_model=64,nhead=8)
        self.self_attn_p2i1 = LoFTREncoderLayer(d_model=64,nhead=8)
        
        self.self_attn_p2 = LoFTREncoderLayer(d_model=256,nhead=8)
        self.self_attn_i2p2 = LoFTREncoderLayer(d_model=256,nhead=8)
        self.self_attn_i2 = LoFTREncoderLayer(d_model=128,nhead=8)
        self.self_attn_p2i2 = LoFTREncoderLayer(d_model=128,nhead=8)
        
        self.self_attn_p3 = LoFTREncoderLayer(d_model=512,nhead=8)
        self.self_attn_i2p3 = LoFTREncoderLayer(d_model=512,nhead=8)
        self.self_attn_i3 = LoFTREncoderLayer(d_model=256,nhead=8)
        self.self_attn_p2i3 = LoFTREncoderLayer(d_model=256,nhead=8)
        
        self.self_attn_p4 = LoFTREncoderLayer(d_model=256,nhead=8)
        self.self_attn_i2p4 = LoFTREncoderLayer(d_model=256,nhead=8)
        self.self_attn_i4 = LoFTREncoderLayer(d_model=128,nhead=8)
        self.self_attn_p2i4 = LoFTREncoderLayer(d_model=128,nhead=8)
        
        # self.self_attn_p5 = LoFTREncoderLayer(d_model=128,nhead=8)
        # self.self_attn_i2p5 = LoFTREncoderLayer(d_model=128,nhead=8)
        # self.self_attn_i5 = LoFTREncoderLayer(d_model=64,nhead=8)
        # self.self_attn_p2i5 = LoFTREncoderLayer(d_model=64,nhead=8)
    def selfp(self,p,layer,num_view):
        p1 = torch.chunk(p,num_view,dim=0)
        p1 = [torch.reshape(p1[i],(1,p1[i].size()[0],p1[i].size()[1])) for i in range(num_view)]
        p1 = [layer(p1[i],p1[i]).squeeze() for i in range(num_view)]
        p2 = torch.cat(p1,dim=0)
        return p2
    
    def selfi(self,feat_i,layer,num_view):
        B,C,H,W = feat_i[0].size()
        
        for i in range(num_view):
            feat_i[i] = feat_i[i].reshape(B,C,H*W).permute(0,2,1)
            feat_i[i] = layer(feat_i[i],feat_i[i])
            feat_i[i] = feat_i[i].permute(0,2,1).reshape(B,C,H,W)
        return feat_i
    
    def crossp(self,p,layer):
        p1 = torch.chunk(p,2,dim=0)
        p1 = [torch.reshape(p1[i],(1,p1[i].size()[0],p1[i].size()[1])) for i in range(2)]
        p1 = [layer(p1[0],p1[1]).squeeze(),layer(p1[1],p1[0]).squeeze()]
        p2 = torch.cat(p1,dim=0)
        return p2
    
    def crossi(self,feat_i,layer):
        B,C,H,W = feat_i[0].size()
        for i in range(2):
            feat_i[i] = feat_i[i].reshape(B,C,H*W).permute(0,2,1)
            feat_i[i] = layer(feat_i[i],feat_i[i+1%2])
            feat_i[i] = feat_i[i].permute(0,2,1).reshape(B,C,H,W)
        return feat_i
    
    def forward(self, batch, rgbs, K, deps, vps=None):
        # Estimate Depth -- now for 1 and 2
        n_views = 2
        output = {}
        B, _, H, W = rgbs[0].shape

        # Encode features
        feat_p_encode = []
        feat_i_encode = []

        #pre stage
        feat_p = batch['features'].clone().detach() #全为1 (130982,1)
        for block_op in self.pcd_pre_stages:
            feat_p = block_op(feat_p, batch)# feat_p (130982,128)
        
        
        feat_i = [self.cnn_pre_stages(rgbs[i]) for i in range(n_views)] # feat_i [2](4,64,128,128)
        

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][0].squeeze())#(131072,128)
        feat_p2i = self.fuse_p2i_ds_0[0](feat_p2i)#(131072,64)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_0[1])# [2](4,64,128,128)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][0])#(130982,32,64)
        feat_i2p = self.fuse_i2p_ds_0[0](feat_i2p.max(1)[0])#(130982,128)类似于pointnet的最大池化
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_0[1])#(130982,128)

        feat_p = feat_p + self.selfp(feat_p,self.self_attn_p1,n_views)
        feat_i = feat_i + self.selfi(feat_i,self.self_attn_i1,n_views)
        feat_i2p = feat_i2p + self.selfp(feat_i2p,self.self_attn_i2p1,n_views)
        feat_p2i = feat_p2i + self.selfi(feat_p2i,self.self_attn_p2i1,n_views)
        
        
        feat_p = feat_p + feat_i2p#(130982,128)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,64,128,128)
        
        

        feat_p_encode.append(feat_p)# [1](130982,128)
        feat_i_encode.append(feat_i)# [1][2](4,64,128,128)

        #downsample 0
        for block_op in self.pcd_ds_0:
            feat_p = block_op(feat_p, batch)# (17420,256)
        feat_i = [self.cnn_ds_0(feat_i[i]) for i in range(n_views)]# [2](4,128,64,64)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())#(32768,256)
        feat_p2i = self.fuse_p2i_ds_1[0](feat_p2i)#(32768,128)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_1[1])# [2](4,128,64,64)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])# (17420,32,128)
        feat_i2p = self.fuse_i2p_ds_1[0](feat_i2p.max(1)[0])# (17420,256)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_1[1])# (17420,256)

        feat_p = feat_p + self.selfp(feat_p,self.self_attn_p2,n_views)
        feat_i = feat_i + self.selfi(feat_i,self.self_attn_i2,n_views)
        feat_i2p = feat_i2p + self.selfp(feat_i2p,self.self_attn_i2p2,n_views)
        feat_p2i = feat_p2i + self.selfi(feat_p2i,self.self_attn_p2i2,n_views)
        
        feat_p = feat_p + feat_i2p# (17420,256)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,128,64,64)

        feat_p_encode.append(feat_p)# [(130982,128)(17420,256)]
        feat_i_encode.append(feat_i)# [[2](4,64,128,128),[2](4,128,64,64)]

        #downsample 1
        for block_op in self.pcd_ds_1:
            feat_p = block_op(feat_p, batch)# (5895,512)
        feat_i = [self.cnn_ds_1(feat_i[i]) for i in range(n_views)]# [2](4,256,32,32)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][2].squeeze())#(8192,512)
        feat_p2i = self.fuse_p2i_ds_2[0](feat_p2i)#(8192,256)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_2[1])# [2](4,256,32,32)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][2])#(5895,32,256)
        feat_i2p = self.fuse_i2p_ds_2[0](feat_i2p.max(1)[0])#(5895,512)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_2[1])#(5895,512)
        
        feat_p = feat_p + self.selfp(feat_p,self.self_attn_p3,n_views)
        feat_i = feat_i + self.selfi(feat_i,self.self_attn_i3,n_views)
        feat_i2p = feat_i2p + self.selfp(feat_i2p,self.self_attn_i2p3,n_views)
        feat_p2i = feat_p2i + self.selfi(feat_p2i,self.self_attn_p2i3,n_views)

        feat_p = feat_p + feat_i2p#(5895,512)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,256,32,32)

        
        # upsample0
        for block_i, block_op in enumerate(self.pcd_up_0):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)#(17420,256)

        feat_i = [
            self.cnn_up_0(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-1][i]), dim=1)) for i in range(n_views)]#[2](4,128,64,64)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())#(32768,256)
        feat_p2i = self.fuse_p2i_up_0[0](feat_p2i)#(32768,128)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_up_0[1])#[2](4,128,64,64)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])# (17420,32,128)
        feat_i2p = self.fuse_i2p_up_0[0](feat_i2p.max(1)[0])# (17420,256)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_up_0[1])# (17420,256)

        feat_p = feat_p + self.selfp(feat_p,self.self_attn_p4,n_views)
        feat_i = feat_i + self.selfi(feat_i,self.self_attn_i4,n_views)
        feat_i2p = feat_i2p + self.selfp(feat_i2p,self.self_attn_i2p4,n_views)
        feat_p2i = feat_p2i + self.selfi(feat_p2i,self.self_attn_p2i4,n_views)
        
        feat_p = feat_p + feat_i2p # (17420,256)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]#[2](4,128,64,64)

        # upsample1
        for block_i, block_op in enumerate(self.pcd_up_1):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)#(130982,32)

        feat_i = [
            self.cnn_up_1(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-2][i]), dim=1)) for i in range(n_views)]#[2](4,32,128,128)

        pcs_X = batch['points_img']     #[2](4,16384,3)
        pcs_F = self.map(feat_i, feat_p, batch) #[2](4,16384,32)

        pcs_F_sc4 = list(self.loftr_coarse4(pcs_F[0],pcs_F[1]))
        pcs_F = pcs_F+pcs_F_sc4
        
        if vps is not None:
            # Drop first viewpoint -- assumed to be identity transformation
            vps = vps[1:]
        elif self.align_cfg.algorithm == "weighted_procrustes":
            vps = []
            cor_loss = []
            for i in range(1, n_views):
                corr_i = get_correspondences(
                    P1=pcs_F[0],
                    P2=pcs_F[i],
                    P1_X=pcs_X[0],
                    P2_X=pcs_X[i],
                    num_corres=self.num_corres,
                    ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
                )
                Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

                vps.append(Rt_i)
                cor_loss.append(cor_loss_i)

                # add for visualization
                output[f"corres_0{i}"] = corr_i
                output[f"vp_{i}"] = Rt_i
        else:
            raise ValueError(f"How to align using {self.align_cfg.algorithm}?")

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        # Get RGB pointcloud as well for direct rendering
        pcs_rgb = [rgb.view(B, 3, -1).permute(0, 2, 1).contiguous() for rgb in rgbs]

        projs = []
        # get joint for all values
        if self.pointcloud_source == "joint":
            pcs_X_joint = torch.cat(pcs_X, dim=1)
            pcs_F_joint = torch.cat(pcs_F, dim=1)
            pcs_RGB_joint = torch.cat(pcs_rgb, dim=1)
            pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

        # Rasterize and Blend
        for i in range(n_views):
            if self.pointcloud_source == "other":
                # get joint for all values except the one
                pcs_X_joint = torch.cat(pcs_X[0:i] + pcs_X[i + 1: n_views], dim=1)
                pcs_F_joint = torch.cat(pcs_F[0:i] + pcs_F[i + 1: n_views], dim=1)
                pcs_RGB_joint = torch.cat(
                    pcs_rgb[0:i] + pcs_rgb[i + 1: n_views], dim=1
                )
                pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

            if i > 0:
                rot_joint_X = transform_points_Rt(pcs_X_joint, vps[i - 1])
                rot_joint_X = points_to_ndc(rot_joint_X, K, (H, W))
            else:
                rot_joint_X = points_to_ndc(pcs_X_joint, K, (H, W))
            projs.append(self.renderer(rot_joint_X, pcs_FRGB_joint))

        # Decode
        for i in range(n_views):
            proj_FRGB_i = projs[i]["feats"]
            proj_RGB_i = proj_FRGB_i[:, -3:]
            proj_F_i = proj_FRGB_i[:, :-3]

            output[f"rgb_render_{i}"] = proj_RGB_i
            output[f"ras_depth_{i}"] = projs[i]["depth"]
            output[f"cover_{i}"] = projs[i]["mask"].unsqueeze(1)  # useless
        
        return output

    def gather_p2i(self, feat_p, idx):
        feat_p = torch.cat((feat_p, torch.zeros_like(feat_p[:1, :])), 0)
        return feat_p[idx]

    def gather_i2p(self, feat_i, idx):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        feat_i2p = []
        for i in range(B):
            feat_i2p.append(src_feat_i[i].reshape(C,H*W).permute(1,0))
            feat_i2p.append(tgt_feat_i[i].reshape(C,H*W).permute(1,0))

        feat_i2p = torch.cat(feat_i2p, 0)
        feat_i2p = torch.cat((feat_i2p, torch.zeros_like(feat_i2p[:1, :])), 0)
        return feat_i2p[idx]  #N,16,C

    def fusep2i(self, feat_i, feat_p2i, layer):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        # src_feat_i = src_feat_i.reshape(B, C, H * W).permute(0, 2, 1)
        # tgt_feat_i = tgt_feat_i.reshape(B, C, H * W).permute(0, 2, 1)

        src_feat_p2i = []
        tgt_feat_p2i = []

        for i in range(2*B):
            if i % 2 == 0:
                src_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))#[4](1,16384,64)
            else:
                tgt_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))#[4](1,16384,64)

        src_feat_p2i = torch.vstack(src_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)#(4,64,128,128)
        tgt_feat_p2i = torch.vstack(tgt_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)#(4,64,128,128)

        src_feat_p2i = torch.cat([src_feat_i, src_feat_p2i], 1)#(4,128,128,128)
        tgt_feat_p2i = torch.cat([tgt_feat_i, tgt_feat_p2i], 1)#(4,128,128,128)

        src_feat_p2i = layer(src_feat_p2i)#(4,64,128,128)
        tgt_feat_p2i = layer(tgt_feat_p2i)#(4,64,128,128)

        return [src_feat_p2i, tgt_feat_p2i]

    def fusei2p(self, feat_p, feat_i2p, layer):
        feat_i2p = torch.cat([feat_p, feat_i2p], -1)
        feat_i2p = layer(feat_i2p)
        return feat_i2p


    def generate_pointclouds(self, K, deps, vps=None):
        n_views = len(deps)
        # generate pointclouds - generate grid once for efficiency
        B, _, H, W = deps[0].shape
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pcs_X = [
            grid_to_pointcloud(K_inv, deps[i], None, grid)[0] for i in range(n_views)
        ]

        if vps is not None:
            pcs_X_rot = [
                transform_points_Rt(pcs_X[i + 1], vps[i + 1], inverse=True, )
                for i in range(n_views - 1)
            ]
            pcs_X = pcs_X[0:1] + pcs_X_rot
            pcs_X = torch.cat(pcs_X, dim=1).detach().cpu()

        return pcs_X

    def get_feature_pcs(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert (
                feats[0].shape[-1] == deps[0].shape[-1]
        ), f"Same size {feats[0].shape} - {deps[0].shape}"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]
        return pcs_X, pcs_F, None
    
    
class PCReg_KPURes18_MSF_loftr8(nn.Module):
    def __init__(self, cfg):
        super(PCReg_KPURes18_MSF_loftr8, self).__init__()
        # set encoder decoder
        chan_in = 3
        self.cfg = cfg
        feat_dim = cfg.feat_dim

        # No imagenet pretraining
        pretrained = False

        encode_I = URes18Encoder1(chan_in, feat_dim, pretrained)
        encode_P = KPFCN(cfg)
        self.cnn_pre_stages = nn.Sequential(
            encode_I.inconv,
            encode_I.layer1
        )
        self.pcd_pre_stages = nn.ModuleList()
        for i in range(0,2):
            self.pcd_pre_stages.append(encode_P.encoder_blocks[i])

        self.cnn_ds_0 = encode_I.layer2
        self.pcd_ds_0 = nn.ModuleList()
        for i in range(2,5):
            self.pcd_ds_0.append(encode_P.encoder_blocks[i])

        self.cnn_ds_1 = encode_I.layer3
        self.pcd_ds_1 = nn.ModuleList()
        for i in range(5,8):
            self.pcd_ds_1.append(encode_P.encoder_blocks[i])

        self.cnn_up_0 = encode_I.up1
        self.pcd_up_0 = nn.ModuleList()
        for i in range(0, 2):
            self.pcd_up_0.append(encode_P.decoder_blocks[i])

        self.cnn_up_1 = nn.Sequential(
            encode_I.up2,
            encode_I.outconv
        )
        self.pcd_up_1 = nn.ModuleList()
        for i in range(2, 4):
            self.pcd_up_1.append(encode_P.decoder_blocks[i])

        # 0
        self.fuse_p2i_ds_0 = nn.ModuleList()
        self.fuse_p2i_ds_0.append(
            UnaryBlock(128, 64, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_0.append(
            nn.Sequential(
                nn.Conv2d(64*2, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_0 = nn.ModuleList()
        self.fuse_i2p_ds_0.append(
            UnaryBlock(64, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_0.append(
            UnaryBlock(128 * 2, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 1
        self.fuse_p2i_ds_1 = nn.ModuleList()
        self.fuse_p2i_ds_1.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_1.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_1 = nn.ModuleList()
        self.fuse_i2p_ds_1.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_1.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 2
        self.fuse_p2i_ds_2 = nn.ModuleList()
        self.fuse_p2i_ds_2.append(
            UnaryBlock(512, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_2.append(
            nn.Sequential(
                nn.Conv2d(256 * 2, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_2 = nn.ModuleList()
        self.fuse_i2p_ds_2.append(
            UnaryBlock(256, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_2.append(
            UnaryBlock(512 * 2, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        self.fuse_p2i_up_0 = nn.ModuleList()
        self.fuse_p2i_up_0.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_up_0.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
        ))
        self.fuse_i2p_up_0 = nn.ModuleList()
        self.fuse_i2p_up_0.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_up_0.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # self.fuse_p2i_up_1 = nn.ModuleList()
        # self.fuse_i2p_up_1 = nn.ModuleList()
        self.align_cfg = cfg
        self.renderer = PointsRenderer(cfg)
        self.num_corres = cfg.num_correspodances
        self.pointcloud_source = cfg.pointcloud_source
        self.map = Fusion_CATL(feat_dim)
        
        
        self.loftr_coarse4 = LocalFeatureTransformer({'d_model':32,
                'nhead':8,
                'layer_names':['self', 'cross'] * 4,
                'attention':'linear'})
        
        self.self_attn_p1 = LoFTREncoderLayer(d_model=128,nhead=8)
        self.self_attn_i2p1 = LoFTREncoderLayer(d_model=128,nhead=8)
        self.self_attn_i1 = LoFTREncoderLayer(d_model=64,nhead=8)
        self.self_attn_p2i1 = LoFTREncoderLayer(d_model=64,nhead=8)
        
        self.self_attn_p2 = LoFTREncoderLayer(d_model=256,nhead=8)
        self.self_attn_i2p2 = LoFTREncoderLayer(d_model=256,nhead=8)
        self.self_attn_i2 = LoFTREncoderLayer(d_model=128,nhead=8)
        self.self_attn_p2i2 = LoFTREncoderLayer(d_model=128,nhead=8)
        
        self.self_attn_p3 = LoFTREncoderLayer(d_model=512,nhead=8)
        self.self_attn_i2p3 = LoFTREncoderLayer(d_model=512,nhead=8)
        self.self_attn_i3 = LoFTREncoderLayer(d_model=256,nhead=8)
        self.self_attn_p2i3 = LoFTREncoderLayer(d_model=256,nhead=8)
        
        self.self_attn_p4 = LoFTREncoderLayer(d_model=256,nhead=8)
        self.self_attn_i2p4 = LoFTREncoderLayer(d_model=256,nhead=8)
        self.self_attn_i4 = LoFTREncoderLayer(d_model=128,nhead=8)
        self.self_attn_p2i4 = LoFTREncoderLayer(d_model=128,nhead=8)
        
        self.position_encoding = PositionEncodingSine(d_model=32,max_shape=(128,128))
        # self.self_attn_p5 = LoFTREncoderLayer(d_model=128,nhead=8)
        # self.self_attn_i2p5 = LoFTREncoderLayer(d_model=128,nhead=8)
        # self.self_attn_i5 = LoFTREncoderLayer(d_model=64,nhead=8)
        # self.self_attn_p2i5 = LoFTREncoderLayer(d_model=64,nhead=8)
    def selfp(self,p,layer,num_view):
        p1 = torch.chunk(p,num_view,dim=0)
        p1 = [torch.reshape(p1[i],(1,p1[i].size()[0],p1[i].size()[1])) for i in range(num_view)]
        p1 = [layer(p1[i],p1[i]).squeeze() for i in range(num_view)]
        p2 = torch.cat(p1,dim=0)
        return p2
    
    def selfi(self,feat_i,layer,num_view):
        B,C,H,W = feat_i[0].size()
        
        for i in range(num_view):
            feat_i[i] = feat_i[i].reshape(B,C,H*W).permute(0,2,1)
            feat_i[i] = layer(feat_i[i],feat_i[i])
            feat_i[i] = feat_i[i].permute(0,2,1).reshape(B,C,H,W)
        return feat_i
    
    def crossp(self,p,layer):
        p1 = torch.chunk(p,2,dim=0)
        p1 = [torch.reshape(p1[i],(1,p1[i].size()[0],p1[i].size()[1])) for i in range(2)]
        p1 = [layer(p1[0],p1[1]).squeeze(),layer(p1[1],p1[0]).squeeze()]
        p2 = torch.cat(p1,dim=0)
        return p2
    
    def crossi(self,feat_i,layer):
        B,C,H,W = feat_i[0].size()
        for i in range(2):
            feat_i[i] = feat_i[i].reshape(B,C,H*W).permute(0,2,1)
            feat_i[i] = layer(feat_i[i],feat_i[i+1%2])
            feat_i[i] = feat_i[i].permute(0,2,1).reshape(B,C,H,W)
        return feat_i
    
    def forward(self, batch, rgbs, K, deps, vps=None):
        # Estimate Depth -- now for 1 and 2
        n_views = 2
        output = {}
        B, _, H, W = rgbs[0].shape

        # Encode features
        feat_p_encode = []
        feat_i_encode = []

        #pre stage
        feat_p = batch['features'].clone().detach() #全为1 (130982,1)
        for block_op in self.pcd_pre_stages:
            feat_p = block_op(feat_p, batch)# feat_p (130982,128)
        
        
        feat_i = [self.cnn_pre_stages(rgbs[i]) for i in range(n_views)] # feat_i [2](4,64,128,128)
        

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][0].squeeze())#(131072,128)
        feat_p2i = self.fuse_p2i_ds_0[0](feat_p2i)#(131072,64)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_0[1])# [2](4,64,128,128)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][0])#(130982,32,64)
        feat_i2p = self.fuse_i2p_ds_0[0](feat_i2p.max(1)[0])#(130982,128)类似于pointnet的最大池化
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_0[1])#(130982,128)

        feat_p = feat_p + self.selfp(feat_p,self.self_attn_p1,n_views)
        feat_i = feat_i + self.selfi(feat_i,self.self_attn_i1,n_views)
        feat_i2p = feat_i2p + self.selfp(feat_i2p,self.self_attn_i2p1,n_views)
        feat_p2i = feat_p2i + self.selfi(feat_p2i,self.self_attn_p2i1,n_views)
        
        
        feat_p = feat_p + feat_i2p#(130982,128)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,64,128,128)
        
        

        feat_p_encode.append(feat_p)# [1](130982,128)
        feat_i_encode.append(feat_i)# [1][2](4,64,128,128)

        #downsample 0
        for block_op in self.pcd_ds_0:
            feat_p = block_op(feat_p, batch)# (17420,256)
        feat_i = [self.cnn_ds_0(feat_i[i]) for i in range(n_views)]# [2](4,128,64,64)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())#(32768,256)
        feat_p2i = self.fuse_p2i_ds_1[0](feat_p2i)#(32768,128)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_1[1])# [2](4,128,64,64)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])# (17420,32,128)
        feat_i2p = self.fuse_i2p_ds_1[0](feat_i2p.max(1)[0])# (17420,256)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_1[1])# (17420,256)

        feat_p = feat_p + self.selfp(feat_p,self.self_attn_p2,n_views)
        feat_i = feat_i + self.selfi(feat_i,self.self_attn_i2,n_views)
        feat_i2p = feat_i2p + self.selfp(feat_i2p,self.self_attn_i2p2,n_views)
        feat_p2i = feat_p2i + self.selfi(feat_p2i,self.self_attn_p2i2,n_views)
        
        feat_p = feat_p + feat_i2p# (17420,256)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,128,64,64)

        feat_p_encode.append(feat_p)# [(130982,128)(17420,256)]
        feat_i_encode.append(feat_i)# [[2](4,64,128,128),[2](4,128,64,64)]

        #downsample 1
        for block_op in self.pcd_ds_1:
            feat_p = block_op(feat_p, batch)# (5895,512)
        feat_i = [self.cnn_ds_1(feat_i[i]) for i in range(n_views)]# [2](4,256,32,32)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][2].squeeze())#(8192,512)
        feat_p2i = self.fuse_p2i_ds_2[0](feat_p2i)#(8192,256)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_2[1])# [2](4,256,32,32)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][2])#(5895,32,256)
        feat_i2p = self.fuse_i2p_ds_2[0](feat_i2p.max(1)[0])#(5895,512)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_2[1])#(5895,512)
        
        feat_p = feat_p + self.selfp(feat_p,self.self_attn_p3,n_views)
        feat_i = feat_i + self.selfi(feat_i,self.self_attn_i3,n_views)
        feat_i2p = feat_i2p + self.selfp(feat_i2p,self.self_attn_i2p3,n_views)
        feat_p2i = feat_p2i + self.selfi(feat_p2i,self.self_attn_p2i3,n_views)

        feat_p = feat_p + feat_i2p#(5895,512)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,256,32,32)

        
        # upsample0
        for block_i, block_op in enumerate(self.pcd_up_0):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)#(17420,256)

        feat_i = [
            self.cnn_up_0(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-1][i]), dim=1)) for i in range(n_views)]#[2](4,128,64,64)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())#(32768,256)
        feat_p2i = self.fuse_p2i_up_0[0](feat_p2i)#(32768,128)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_up_0[1])#[2](4,128,64,64)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])# (17420,32,128)
        feat_i2p = self.fuse_i2p_up_0[0](feat_i2p.max(1)[0])# (17420,256)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_up_0[1])# (17420,256)

        feat_p = feat_p + self.selfp(feat_p,self.self_attn_p4,n_views)
        feat_i = feat_i + self.selfi(feat_i,self.self_attn_i4,n_views)
        feat_i2p = feat_i2p + self.selfp(feat_i2p,self.self_attn_i2p4,n_views)
        feat_p2i = feat_p2i + self.selfi(feat_p2i,self.self_attn_p2i4,n_views)
        
        feat_p = feat_p + feat_i2p # (17420,256)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]#[2](4,128,64,64)

        # upsample1
        for block_i, block_op in enumerate(self.pcd_up_1):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)#(130982,32)

        feat_i = [
            self.cnn_up_1(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-2][i]), dim=1)) for i in range(n_views)]#[2](4,32,128,128)

        pcs_X = batch['points_img']     #[2](4,16384,3)
        pcs_F = self.map(feat_i, feat_p, batch) #[2](4,16384,32)

        pcs_F = [
            self.position_encoding(pcs_F[0].permute(0,2,1).reshape(B,32,128,128)).reshape(B,32,16384).permute(0,2,1),
            self.position_encoding(pcs_F[1].permute(0,2,1).reshape(B,32,128,128)).reshape(B,32,16384).permute(0,2,1)
        ]
        
        pcs_F_sc4 = list(self.loftr_coarse4(pcs_F[0],pcs_F[1]))
        pcs_F = pcs_F+pcs_F_sc4
        
        if vps is not None:
            # Drop first viewpoint -- assumed to be identity transformation
            vps = vps[1:]
        elif self.align_cfg.algorithm == "weighted_procrustes":
            vps = []
            cor_loss = []
            for i in range(1, n_views):
                corr_i = get_correspondences(
                    P1=pcs_F[0],
                    P2=pcs_F[i],
                    P1_X=pcs_X[0],
                    P2_X=pcs_X[i],
                    num_corres=self.num_corres,
                    ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
                )
                Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

                vps.append(Rt_i)
                cor_loss.append(cor_loss_i)

                # add for visualization
                output[f"corres_0{i}"] = corr_i
                output[f"vp_{i}"] = Rt_i
        else:
            raise ValueError(f"How to align using {self.align_cfg.algorithm}?")

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        # Get RGB pointcloud as well for direct rendering
        pcs_rgb = [rgb.view(B, 3, -1).permute(0, 2, 1).contiguous() for rgb in rgbs]

        projs = []
        # get joint for all values
        if self.pointcloud_source == "joint":
            pcs_X_joint = torch.cat(pcs_X, dim=1)
            pcs_F_joint = torch.cat(pcs_F, dim=1)
            pcs_RGB_joint = torch.cat(pcs_rgb, dim=1)
            pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

        # Rasterize and Blend
        for i in range(n_views):
            if self.pointcloud_source == "other":
                # get joint for all values except the one
                pcs_X_joint = torch.cat(pcs_X[0:i] + pcs_X[i + 1: n_views], dim=1)
                pcs_F_joint = torch.cat(pcs_F[0:i] + pcs_F[i + 1: n_views], dim=1)
                pcs_RGB_joint = torch.cat(
                    pcs_rgb[0:i] + pcs_rgb[i + 1: n_views], dim=1
                )
                pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

            if i > 0:
                rot_joint_X = transform_points_Rt(pcs_X_joint, vps[i - 1])
                rot_joint_X = points_to_ndc(rot_joint_X, K, (H, W))
            else:
                rot_joint_X = points_to_ndc(pcs_X_joint, K, (H, W))
            projs.append(self.renderer(rot_joint_X, pcs_FRGB_joint))

        # Decode
        for i in range(n_views):
            proj_FRGB_i = projs[i]["feats"]
            proj_RGB_i = proj_FRGB_i[:, -3:]
            proj_F_i = proj_FRGB_i[:, :-3]

            output[f"rgb_render_{i}"] = proj_RGB_i
            output[f"ras_depth_{i}"] = projs[i]["depth"]
            output[f"cover_{i}"] = projs[i]["mask"].unsqueeze(1)  # useless
        
        return output

    def gather_p2i(self, feat_p, idx):
        feat_p = torch.cat((feat_p, torch.zeros_like(feat_p[:1, :])), 0)
        return feat_p[idx]

    def gather_i2p(self, feat_i, idx):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        feat_i2p = []
        for i in range(B):
            feat_i2p.append(src_feat_i[i].reshape(C,H*W).permute(1,0))
            feat_i2p.append(tgt_feat_i[i].reshape(C,H*W).permute(1,0))

        feat_i2p = torch.cat(feat_i2p, 0)
        feat_i2p = torch.cat((feat_i2p, torch.zeros_like(feat_i2p[:1, :])), 0)
        return feat_i2p[idx]  #N,16,C

    def fusep2i(self, feat_i, feat_p2i, layer):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        # src_feat_i = src_feat_i.reshape(B, C, H * W).permute(0, 2, 1)
        # tgt_feat_i = tgt_feat_i.reshape(B, C, H * W).permute(0, 2, 1)

        src_feat_p2i = []
        tgt_feat_p2i = []

        for i in range(2*B):
            if i % 2 == 0:
                src_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))#[4](1,16384,64)
            else:
                tgt_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))#[4](1,16384,64)

        src_feat_p2i = torch.vstack(src_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)#(4,64,128,128)
        tgt_feat_p2i = torch.vstack(tgt_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)#(4,64,128,128)

        src_feat_p2i = torch.cat([src_feat_i, src_feat_p2i], 1)#(4,128,128,128)
        tgt_feat_p2i = torch.cat([tgt_feat_i, tgt_feat_p2i], 1)#(4,128,128,128)

        src_feat_p2i = layer(src_feat_p2i)#(4,64,128,128)
        tgt_feat_p2i = layer(tgt_feat_p2i)#(4,64,128,128)

        return [src_feat_p2i, tgt_feat_p2i]

    def fusei2p(self, feat_p, feat_i2p, layer):
        feat_i2p = torch.cat([feat_p, feat_i2p], -1)
        feat_i2p = layer(feat_i2p)
        return feat_i2p


    def generate_pointclouds(self, K, deps, vps=None):
        n_views = len(deps)
        # generate pointclouds - generate grid once for efficiency
        B, _, H, W = deps[0].shape
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pcs_X = [
            grid_to_pointcloud(K_inv, deps[i], None, grid)[0] for i in range(n_views)
        ]

        if vps is not None:
            pcs_X_rot = [
                transform_points_Rt(pcs_X[i + 1], vps[i + 1], inverse=True, )
                for i in range(n_views - 1)
            ]
            pcs_X = pcs_X[0:1] + pcs_X_rot
            pcs_X = torch.cat(pcs_X, dim=1).detach().cpu()

        return pcs_X

    def get_feature_pcs(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert (
                feats[0].shape[-1] == deps[0].shape[-1]
        ), f"Same size {feats[0].shape} - {deps[0].shape}"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]
        return pcs_X, pcs_F, None
    
    
    
    
class PCReg_KPURes18_MSF_loftr9(nn.Module):
    def __init__(self, cfg):
        super(PCReg_KPURes18_MSF_loftr9, self).__init__()
        # set encoder decoder
        chan_in = 3
        self.cfg = cfg
        feat_dim = cfg.feat_dim

        # No imagenet pretraining
        pretrained = False

        encode_I = URes18Encoder1(chan_in, feat_dim, pretrained)
        encode_P = KPFCN(cfg)
        self.cnn_pre_stages = nn.Sequential(
            encode_I.inconv,
            encode_I.layer1
        )
        self.pcd_pre_stages = nn.ModuleList()
        for i in range(0,2):
            self.pcd_pre_stages.append(encode_P.encoder_blocks[i])

        self.cnn_ds_0 = encode_I.layer2
        self.pcd_ds_0 = nn.ModuleList()
        for i in range(2,5):
            self.pcd_ds_0.append(encode_P.encoder_blocks[i])

        self.cnn_ds_1 = encode_I.layer3
        self.pcd_ds_1 = nn.ModuleList()
        for i in range(5,8):
            self.pcd_ds_1.append(encode_P.encoder_blocks[i])

        self.cnn_up_0 = encode_I.up1
        self.pcd_up_0 = nn.ModuleList()
        for i in range(0, 2):
            self.pcd_up_0.append(encode_P.decoder_blocks[i])

        self.cnn_up_1 = nn.Sequential(
            encode_I.up2,
            encode_I.outconv
        )
        self.pcd_up_1 = nn.ModuleList()
        for i in range(2, 4):
            self.pcd_up_1.append(encode_P.decoder_blocks[i])

        # 0
        self.fuse_p2i_ds_0 = nn.ModuleList()
        self.fuse_p2i_ds_0.append(
            UnaryBlock(128, 64, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_0.append(
            nn.Sequential(
                nn.Conv2d(64*2, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_0 = nn.ModuleList()
        self.fuse_i2p_ds_0.append(
            UnaryBlock(64, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_0.append(
            UnaryBlock(128 * 2, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 1
        self.fuse_p2i_ds_1 = nn.ModuleList()
        self.fuse_p2i_ds_1.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_1.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_1 = nn.ModuleList()
        self.fuse_i2p_ds_1.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_1.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 2
        self.fuse_p2i_ds_2 = nn.ModuleList()
        self.fuse_p2i_ds_2.append(
            UnaryBlock(512, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_2.append(
            nn.Sequential(
                nn.Conv2d(256 * 2, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_2 = nn.ModuleList()
        self.fuse_i2p_ds_2.append(
            UnaryBlock(256, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_2.append(
            UnaryBlock(512 * 2, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        self.fuse_p2i_up_0 = nn.ModuleList()
        self.fuse_p2i_up_0.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_up_0.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
        ))
        self.fuse_i2p_up_0 = nn.ModuleList()
        self.fuse_i2p_up_0.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_up_0.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # self.fuse_p2i_up_1 = nn.ModuleList()
        # self.fuse_i2p_up_1 = nn.ModuleList()
        self.align_cfg = cfg
        self.renderer = PointsRenderer(cfg)
        self.num_corres = cfg.num_correspodances
        self.pointcloud_source = cfg.pointcloud_source
        self.map = Fusion_CATL(feat_dim)
        
        
        self.loftr_coarse4 = LocalFeatureTransformer({'d_model':32,
                'nhead':8,
                'layer_names':['self', 'cross'] * 4,
                'attention':'linear'})
        
        self.self_attn_p1 = LoFTREncoderLayer(d_model=128,nhead=8)
        self.self_attn_i2p1 = LoFTREncoderLayer(d_model=128,nhead=8)
        self.self_attn_i1 = LoFTREncoderLayer(d_model=64,nhead=8)
        self.self_attn_p2i1 = LoFTREncoderLayer(d_model=64,nhead=8)
        
        self.self_attn_p2 = LoFTREncoderLayer(d_model=256,nhead=8)
        self.self_attn_i2p2 = LoFTREncoderLayer(d_model=256,nhead=8)
        self.self_attn_i2 = LoFTREncoderLayer(d_model=128,nhead=8)
        self.self_attn_p2i2 = LoFTREncoderLayer(d_model=128,nhead=8)
        
        self.self_attn_p3 = LoFTREncoderLayer(d_model=512,nhead=8)
        self.self_attn_i2p3 = LoFTREncoderLayer(d_model=512,nhead=8)
        self.self_attn_i3 = LoFTREncoderLayer(d_model=256,nhead=8)
        self.self_attn_p2i3 = LoFTREncoderLayer(d_model=256,nhead=8)
        
        self.self_attn_p4 = LoFTREncoderLayer(d_model=256,nhead=8)
        self.self_attn_i2p4 = LoFTREncoderLayer(d_model=256,nhead=8)
        self.self_attn_i4 = LoFTREncoderLayer(d_model=128,nhead=8)
        self.self_attn_p2i4 = LoFTREncoderLayer(d_model=128,nhead=8)
        
        self.position_encoding = PositionEncodingSine(d_model=64,max_shape=(128,128))
        # self.self_attn_p5 = LoFTREncoderLayer(d_model=128,nhead=8)
        # self.self_attn_i2p5 = LoFTREncoderLayer(d_model=128,nhead=8)
        # self.self_attn_i5 = LoFTREncoderLayer(d_model=64,nhead=8)
        # self.self_attn_p2i5 = LoFTREncoderLayer(d_model=64,nhead=8)
    def selfp(self,p,layer,num_view):
        p1 = torch.chunk(p,num_view,dim=0)
        p1 = [torch.reshape(p1[i],(1,p1[i].size()[0],p1[i].size()[1])) for i in range(num_view)]
        p1 = [layer(p1[i],p1[i]).squeeze() for i in range(num_view)]
        p2 = torch.cat(p1,dim=0)
        return p2
    
    def selfi(self,feat_i,layer,num_view):
        B,C,H,W = feat_i[0].size()
        
        for i in range(num_view):
            feat_i[i] = feat_i[i].reshape(B,C,H*W).permute(0,2,1)
            feat_i[i] = layer(feat_i[i],feat_i[i])
            feat_i[i] = feat_i[i].permute(0,2,1).reshape(B,C,H,W)
        return feat_i
    
    def crossp(self,p,layer):
        p1 = torch.chunk(p,2,dim=0)
        p1 = [torch.reshape(p1[i],(1,p1[i].size()[0],p1[i].size()[1])) for i in range(2)]
        p1 = [layer(p1[0],p1[1]).squeeze(),layer(p1[1],p1[0]).squeeze()]
        p2 = torch.cat(p1,dim=0)
        return p2
    
    def crossi(self,feat_i,layer):
        B,C,H,W = feat_i[0].size()
        for i in range(2):
            feat_i[i] = feat_i[i].reshape(B,C,H*W).permute(0,2,1)
            feat_i[i] = layer(feat_i[i],feat_i[i+1%2])
            feat_i[i] = feat_i[i].permute(0,2,1).reshape(B,C,H,W)
        return feat_i
    
    def forward(self, batch, rgbs, K, deps, vps=None):
        # Estimate Depth -- now for 1 and 2
        n_views = 2
        output = {}
        B, _, H, W = rgbs[0].shape

        # Encode features
        feat_p_encode = []
        feat_i_encode = []

        #pre stage
        feat_p = batch['features'].clone().detach() #全为1 (130982,1)
        for block_op in self.pcd_pre_stages:
            feat_p = block_op(feat_p, batch)# feat_p (130982,128)
        
        
        feat_i = [self.cnn_pre_stages(rgbs[i]) for i in range(n_views)] # feat_i [2](4,64,128,128)
        feat_i[0] = self.position_encoding(feat_i[0])
        feat_i[1] = self.position_encoding(feat_i[1])

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][0].squeeze())#(131072,128)
        feat_p2i = self.fuse_p2i_ds_0[0](feat_p2i)#(131072,64)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_0[1])# [2](4,64,128,128)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][0])#(130982,32,64)
        feat_i2p = self.fuse_i2p_ds_0[0](feat_i2p.max(1)[0])#(130982,128)类似于pointnet的最大池化
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_0[1])#(130982,128)

        feat_p = feat_p + self.selfp(feat_p,self.self_attn_p1,n_views)
        feat_i = feat_i + self.selfi(feat_i,self.self_attn_i1,n_views)
        feat_i2p = feat_i2p + self.selfp(feat_i2p,self.self_attn_i2p1,n_views)
        feat_p2i = feat_p2i + self.selfi(feat_p2i,self.self_attn_p2i1,n_views)
        
        
        feat_p = feat_p + feat_i2p#(130982,128)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,64,128,128)
        
        

        feat_p_encode.append(feat_p)# [1](130982,128)
        feat_i_encode.append(feat_i)# [1][2](4,64,128,128)

        #downsample 0
        for block_op in self.pcd_ds_0:
            feat_p = block_op(feat_p, batch)# (17420,256)
        feat_i = [self.cnn_ds_0(feat_i[i]) for i in range(n_views)]# [2](4,128,64,64)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())#(32768,256)
        feat_p2i = self.fuse_p2i_ds_1[0](feat_p2i)#(32768,128)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_1[1])# [2](4,128,64,64)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])# (17420,32,128)
        feat_i2p = self.fuse_i2p_ds_1[0](feat_i2p.max(1)[0])# (17420,256)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_1[1])# (17420,256)

        feat_p = feat_p + self.selfp(feat_p,self.self_attn_p2,n_views)
        feat_i = feat_i + self.selfi(feat_i,self.self_attn_i2,n_views)
        feat_i2p = feat_i2p + self.selfp(feat_i2p,self.self_attn_i2p2,n_views)
        feat_p2i = feat_p2i + self.selfi(feat_p2i,self.self_attn_p2i2,n_views)
        
        feat_p = feat_p + feat_i2p# (17420,256)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,128,64,64)

        feat_p_encode.append(feat_p)# [(130982,128)(17420,256)]
        feat_i_encode.append(feat_i)# [[2](4,64,128,128),[2](4,128,64,64)]

        #downsample 1
        for block_op in self.pcd_ds_1:
            feat_p = block_op(feat_p, batch)# (5895,512)
        feat_i = [self.cnn_ds_1(feat_i[i]) for i in range(n_views)]# [2](4,256,32,32)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][2].squeeze())#(8192,512)
        feat_p2i = self.fuse_p2i_ds_2[0](feat_p2i)#(8192,256)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_2[1])# [2](4,256,32,32)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][2])#(5895,32,256)
        feat_i2p = self.fuse_i2p_ds_2[0](feat_i2p.max(1)[0])#(5895,512)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_2[1])#(5895,512)
        
        feat_p = feat_p + self.selfp(feat_p,self.self_attn_p3,n_views)
        feat_i = feat_i + self.selfi(feat_i,self.self_attn_i3,n_views)
        feat_i2p = feat_i2p + self.selfp(feat_i2p,self.self_attn_i2p3,n_views)
        feat_p2i = feat_p2i + self.selfi(feat_p2i,self.self_attn_p2i3,n_views)

        feat_p = feat_p + feat_i2p#(5895,512)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,256,32,32)

        
        # upsample0
        for block_i, block_op in enumerate(self.pcd_up_0):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)#(17420,256)

        feat_i = [
            self.cnn_up_0(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-1][i]), dim=1)) for i in range(n_views)]#[2](4,128,64,64)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())#(32768,256)
        feat_p2i = self.fuse_p2i_up_0[0](feat_p2i)#(32768,128)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_up_0[1])#[2](4,128,64,64)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])# (17420,32,128)
        feat_i2p = self.fuse_i2p_up_0[0](feat_i2p.max(1)[0])# (17420,256)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_up_0[1])# (17420,256)

        feat_p = feat_p + self.selfp(feat_p,self.self_attn_p4,n_views)
        feat_i = feat_i + self.selfi(feat_i,self.self_attn_i4,n_views)
        feat_i2p = feat_i2p + self.selfp(feat_i2p,self.self_attn_i2p4,n_views)
        feat_p2i = feat_p2i + self.selfi(feat_p2i,self.self_attn_p2i4,n_views)
        
        feat_p = feat_p + feat_i2p # (17420,256)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]#[2](4,128,64,64)

        # upsample1
        for block_i, block_op in enumerate(self.pcd_up_1):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)#(130982,32)

        feat_i = [
            self.cnn_up_1(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-2][i]), dim=1)) for i in range(n_views)]#[2](4,32,128,128)

        pcs_X = batch['points_img']     #[2](4,16384,3)
        pcs_F = self.map(feat_i, feat_p, batch) #[2](4,16384,32)

        
        
        pcs_F_sc4 = list(self.loftr_coarse4(pcs_F[0],pcs_F[1]))
        pcs_F = pcs_F+pcs_F_sc4
        
        if vps is not None:
            # Drop first viewpoint -- assumed to be identity transformation
            vps = vps[1:]
        elif self.align_cfg.algorithm == "weighted_procrustes":
            vps = []
            cor_loss = []
            for i in range(1, n_views):
                corr_i = get_correspondences(
                    P1=pcs_F[0],
                    P2=pcs_F[i],
                    P1_X=pcs_X[0],
                    P2_X=pcs_X[i],
                    num_corres=self.num_corres,
                    ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
                )
                Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

                vps.append(Rt_i)
                cor_loss.append(cor_loss_i)

                # add for visualization
                output[f"corres_0{i}"] = corr_i
                output[f"vp_{i}"] = Rt_i
        else:
            raise ValueError(f"How to align using {self.align_cfg.algorithm}?")

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        # Get RGB pointcloud as well for direct rendering
        pcs_rgb = [rgb.view(B, 3, -1).permute(0, 2, 1).contiguous() for rgb in rgbs]

        projs = []
        # get joint for all values
        if self.pointcloud_source == "joint":
            pcs_X_joint = torch.cat(pcs_X, dim=1)
            pcs_F_joint = torch.cat(pcs_F, dim=1)
            pcs_RGB_joint = torch.cat(pcs_rgb, dim=1)
            pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

        # Rasterize and Blend
        for i in range(n_views):
            if self.pointcloud_source == "other":
                # get joint for all values except the one
                pcs_X_joint = torch.cat(pcs_X[0:i] + pcs_X[i + 1: n_views], dim=1)
                pcs_F_joint = torch.cat(pcs_F[0:i] + pcs_F[i + 1: n_views], dim=1)
                pcs_RGB_joint = torch.cat(
                    pcs_rgb[0:i] + pcs_rgb[i + 1: n_views], dim=1
                )
                pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

            if i > 0:
                rot_joint_X = transform_points_Rt(pcs_X_joint, vps[i - 1])
                rot_joint_X = points_to_ndc(rot_joint_X, K, (H, W))
            else:
                rot_joint_X = points_to_ndc(pcs_X_joint, K, (H, W))
            projs.append(self.renderer(rot_joint_X, pcs_FRGB_joint))

        # Decode
        for i in range(n_views):
            proj_FRGB_i = projs[i]["feats"]
            proj_RGB_i = proj_FRGB_i[:, -3:]
            proj_F_i = proj_FRGB_i[:, :-3]

            output[f"rgb_render_{i}"] = proj_RGB_i
            output[f"ras_depth_{i}"] = projs[i]["depth"]
            output[f"cover_{i}"] = projs[i]["mask"].unsqueeze(1)  # useless
        
        return output

    def gather_p2i(self, feat_p, idx):
        feat_p = torch.cat((feat_p, torch.zeros_like(feat_p[:1, :])), 0)
        return feat_p[idx]

    def gather_i2p(self, feat_i, idx):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        feat_i2p = []
        for i in range(B):
            feat_i2p.append(src_feat_i[i].reshape(C,H*W).permute(1,0))
            feat_i2p.append(tgt_feat_i[i].reshape(C,H*W).permute(1,0))

        feat_i2p = torch.cat(feat_i2p, 0)
        feat_i2p = torch.cat((feat_i2p, torch.zeros_like(feat_i2p[:1, :])), 0)
        return feat_i2p[idx]  #N,16,C

    def fusep2i(self, feat_i, feat_p2i, layer):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        # src_feat_i = src_feat_i.reshape(B, C, H * W).permute(0, 2, 1)
        # tgt_feat_i = tgt_feat_i.reshape(B, C, H * W).permute(0, 2, 1)

        src_feat_p2i = []
        tgt_feat_p2i = []

        for i in range(2*B):
            if i % 2 == 0:
                src_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))#[4](1,16384,64)
            else:
                tgt_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))#[4](1,16384,64)

        src_feat_p2i = torch.vstack(src_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)#(4,64,128,128)
        tgt_feat_p2i = torch.vstack(tgt_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)#(4,64,128,128)

        src_feat_p2i = torch.cat([src_feat_i, src_feat_p2i], 1)#(4,128,128,128)
        tgt_feat_p2i = torch.cat([tgt_feat_i, tgt_feat_p2i], 1)#(4,128,128,128)

        src_feat_p2i = layer(src_feat_p2i)#(4,64,128,128)
        tgt_feat_p2i = layer(tgt_feat_p2i)#(4,64,128,128)

        return [src_feat_p2i, tgt_feat_p2i]

    def fusei2p(self, feat_p, feat_i2p, layer):
        feat_i2p = torch.cat([feat_p, feat_i2p], -1)
        feat_i2p = layer(feat_i2p)
        return feat_i2p


    def generate_pointclouds(self, K, deps, vps=None):
        n_views = len(deps)
        # generate pointclouds - generate grid once for efficiency
        B, _, H, W = deps[0].shape
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pcs_X = [
            grid_to_pointcloud(K_inv, deps[i], None, grid)[0] for i in range(n_views)
        ]

        if vps is not None:
            pcs_X_rot = [
                transform_points_Rt(pcs_X[i + 1], vps[i + 1], inverse=True, )
                for i in range(n_views - 1)
            ]
            pcs_X = pcs_X[0:1] + pcs_X_rot
            pcs_X = torch.cat(pcs_X, dim=1).detach().cpu()

        return pcs_X

    def get_feature_pcs(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert (
                feats[0].shape[-1] == deps[0].shape[-1]
        ), f"Same size {feats[0].shape} - {deps[0].shape}"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]
        return pcs_X, pcs_F, None
    
    
    
class PCReg_KPURes18_MSF_loftr10(nn.Module):
    def __init__(self, cfg):
        super(PCReg_KPURes18_MSF_loftr10, self).__init__()
        # set encoder decoder
        chan_in = 3
        self.cfg = cfg
        feat_dim = cfg.feat_dim

        # No imagenet pretraining
        pretrained = False

        encode_I = URes18Encoder1(chan_in, feat_dim, pretrained)
        encode_P = KPFCN(cfg)
        self.cnn_pre_stages = nn.Sequential(
            encode_I.inconv,
            encode_I.layer1
        )
        self.pcd_pre_stages = nn.ModuleList()
        for i in range(0,2):
            self.pcd_pre_stages.append(encode_P.encoder_blocks[i])

        self.cnn_ds_0 = encode_I.layer2
        self.pcd_ds_0 = nn.ModuleList()
        for i in range(2,5):
            self.pcd_ds_0.append(encode_P.encoder_blocks[i])

        self.cnn_ds_1 = encode_I.layer3
        self.pcd_ds_1 = nn.ModuleList()
        for i in range(5,8):
            self.pcd_ds_1.append(encode_P.encoder_blocks[i])

        self.cnn_up_0 = encode_I.up1
        self.pcd_up_0 = nn.ModuleList()
        for i in range(0, 2):
            self.pcd_up_0.append(encode_P.decoder_blocks[i])

        self.cnn_up_1 = nn.Sequential(
            encode_I.up2,
            encode_I.outconv
        )
        self.pcd_up_1 = nn.ModuleList()
        for i in range(2, 4):
            self.pcd_up_1.append(encode_P.decoder_blocks[i])

        # 0
        self.fuse_p2i_ds_0 = nn.ModuleList()
        self.fuse_p2i_ds_0.append(
            UnaryBlock(128, 64, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_0.append(
            nn.Sequential(
                nn.Conv2d(64*2, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_0 = nn.ModuleList()
        self.fuse_i2p_ds_0.append(
            UnaryBlock(64, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_0.append(
            UnaryBlock(128 * 2, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 1
        self.fuse_p2i_ds_1 = nn.ModuleList()
        self.fuse_p2i_ds_1.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_1.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_1 = nn.ModuleList()
        self.fuse_i2p_ds_1.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_1.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # 2
        self.fuse_p2i_ds_2 = nn.ModuleList()
        self.fuse_p2i_ds_2.append(
            UnaryBlock(512, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_ds_2.append(
            nn.Sequential(
                nn.Conv2d(256 * 2, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        )
        self.fuse_i2p_ds_2 = nn.ModuleList()
        self.fuse_i2p_ds_2.append(
            UnaryBlock(256, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_ds_2.append(
            UnaryBlock(512 * 2, 512, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        self.fuse_p2i_up_0 = nn.ModuleList()
        self.fuse_p2i_up_0.append(
            UnaryBlock(256, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_p2i_up_0.append(
            nn.Sequential(
                nn.Conv2d(128 * 2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
        ))
        self.fuse_i2p_up_0 = nn.ModuleList()
        self.fuse_i2p_up_0.append(
            UnaryBlock(128, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.fuse_i2p_up_0.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        # self.fuse_p2i_up_1 = nn.ModuleList()
        # self.fuse_i2p_up_1 = nn.ModuleList()
        self.align_cfg = cfg
        self.renderer = PointsRenderer(cfg)
        self.num_corres = cfg.num_correspodances
        self.pointcloud_source = cfg.pointcloud_source
        self.map = Fusion_CATL(feat_dim)
        
        
        self.loftr_coarse4 = LocalFeatureTransformer({'d_model':32,
                'nhead':8,
                'layer_names':['self', 'cross'] * 4,
                'attention':'linear'})
        
        self.self_attn_p1 = LoFTREncoderLayer(d_model=128,nhead=8)
        self.self_attn_i2p1 = LoFTREncoderLayer(d_model=128,nhead=8)
        self.self_attn_i1 = LoFTREncoderLayer(d_model=64,nhead=8)
        self.self_attn_p2i1 = LoFTREncoderLayer(d_model=64,nhead=8)
        
        self.self_attn_p2 = LoFTREncoderLayer(d_model=256,nhead=8)
        self.self_attn_i2p2 = LoFTREncoderLayer(d_model=256,nhead=8)
        self.self_attn_i2 = LoFTREncoderLayer(d_model=128,nhead=8)
        self.self_attn_p2i2 = LoFTREncoderLayer(d_model=128,nhead=8)
        
        self.self_attn_p3 = LoFTREncoderLayer(d_model=512,nhead=8)
        self.self_attn_i2p3 = LoFTREncoderLayer(d_model=512,nhead=8)
        self.self_attn_i3 = LoFTREncoderLayer(d_model=256,nhead=8)
        self.self_attn_p2i3 = LoFTREncoderLayer(d_model=256,nhead=8)
        
        self.self_attn_p4 = LoFTREncoderLayer(d_model=256,nhead=8)
        self.self_attn_i2p4 = LoFTREncoderLayer(d_model=256,nhead=8)
        self.self_attn_i4 = LoFTREncoderLayer(d_model=128,nhead=8)
        self.self_attn_p2i4 = LoFTREncoderLayer(d_model=128,nhead=8)
        
        self.position_encoding = PositionEncodingSine(d_model=64,max_shape=(128,128))
        # self.self_attn_p5 = LoFTREncoderLayer(d_model=128,nhead=8)
        # self.self_attn_i2p5 = LoFTREncoderLayer(d_model=128,nhead=8)
        # self.self_attn_i5 = LoFTREncoderLayer(d_model=64,nhead=8)
        # self.self_attn_p2i5 = LoFTREncoderLayer(d_model=64,nhead=8)
    def selfp(self,p,layer,num_view):
        p1 = torch.chunk(p,num_view,dim=0)
        p1 = [torch.reshape(p1[i],(1,p1[i].size()[0],p1[i].size()[1])) for i in range(num_view)]
        p1 = [layer(p1[i],p1[i]).squeeze() for i in range(num_view)]
        p2 = torch.cat(p1,dim=0)
        return p2
    
    def selfi(self,feat_i,layer,num_view):
        B,C,H,W = feat_i[0].size()
        
        for i in range(num_view):
            feat_i[i] = feat_i[i].reshape(B,C,H*W).permute(0,2,1)
            feat_i[i] = layer(feat_i[i],feat_i[i])
            feat_i[i] = feat_i[i].permute(0,2,1).reshape(B,C,H,W)
        return feat_i
    
    def crossp(self,p,layer):
        p1 = torch.chunk(p,2,dim=0)
        p1 = [torch.reshape(p1[i],(1,p1[i].size()[0],p1[i].size()[1])) for i in range(2)]
        p1 = [layer(p1[0],p1[1]).squeeze(),layer(p1[1],p1[0]).squeeze()]
        p2 = torch.cat(p1,dim=0)
        return p2
    
    def crossi(self,feat_i,layer):
        B,C,H,W = feat_i[0].size()
        for i in range(2):
            feat_i[i] = feat_i[i].reshape(B,C,H*W).permute(0,2,1)
            feat_i[i] = layer(feat_i[i],feat_i[i+1%2])
            feat_i[i] = feat_i[i].permute(0,2,1).reshape(B,C,H,W)
        return feat_i
    
    def forward(self, batch, rgbs, K, deps, vps=None):
        # Estimate Depth -- now for 1 and 2
        n_views = 2
        output = {}
        B, _, H, W = rgbs[0].shape

        # Encode features
        feat_p_encode = []
        feat_i_encode = []

        #pre stage
        feat_p = batch['features'].clone().detach() #全为1 (130982,1)
        for block_op in self.pcd_pre_stages:
            feat_p = block_op(feat_p, batch)# feat_p (130982,128)
        
        
        feat_i = [self.cnn_pre_stages(rgbs[i]) for i in range(n_views)] # feat_i [2](4,64,128,128)
        feat_i[0] = feat_i[0]+self.position_encoding(feat_i[0])
        feat_i[1] = feat_i[1]+self.position_encoding(feat_i[1])

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][0].squeeze())#(131072,128)
        feat_p2i = self.fuse_p2i_ds_0[0](feat_p2i)#(131072,64)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_0[1])# [2](4,64,128,128)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][0])#(130982,32,64)
        feat_i2p = self.fuse_i2p_ds_0[0](feat_i2p.max(1)[0])#(130982,128)类似于pointnet的最大池化
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_0[1])#(130982,128)

        feat_p = feat_p + self.selfp(feat_p,self.self_attn_p1,n_views)
        feat_i = feat_i + self.selfi(feat_i,self.self_attn_i1,n_views)
        feat_i2p = feat_i2p + self.selfp(feat_i2p,self.self_attn_i2p1,n_views)
        feat_p2i = feat_p2i + self.selfi(feat_p2i,self.self_attn_p2i1,n_views)
        
        
        feat_p = feat_p + feat_i2p#(130982,128)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,64,128,128)
        
        

        feat_p_encode.append(feat_p)# [1](130982,128)
        feat_i_encode.append(feat_i)# [1][2](4,64,128,128)

        #downsample 0
        for block_op in self.pcd_ds_0:
            feat_p = block_op(feat_p, batch)# (17420,256)
        feat_i = [self.cnn_ds_0(feat_i[i]) for i in range(n_views)]# [2](4,128,64,64)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())#(32768,256)
        feat_p2i = self.fuse_p2i_ds_1[0](feat_p2i)#(32768,128)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_1[1])# [2](4,128,64,64)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])# (17420,32,128)
        feat_i2p = self.fuse_i2p_ds_1[0](feat_i2p.max(1)[0])# (17420,256)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_1[1])# (17420,256)

        feat_p = feat_p + self.selfp(feat_p,self.self_attn_p2,n_views)
        feat_i = feat_i + self.selfi(feat_i,self.self_attn_i2,n_views)
        feat_i2p = feat_i2p + self.selfp(feat_i2p,self.self_attn_i2p2,n_views)
        feat_p2i = feat_p2i + self.selfi(feat_p2i,self.self_attn_p2i2,n_views)
        
        feat_p = feat_p + feat_i2p# (17420,256)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,128,64,64)

        feat_p_encode.append(feat_p)# [(130982,128)(17420,256)]
        feat_i_encode.append(feat_i)# [[2](4,64,128,128),[2](4,128,64,64)]

        #downsample 1
        for block_op in self.pcd_ds_1:
            feat_p = block_op(feat_p, batch)# (5895,512)
        feat_i = [self.cnn_ds_1(feat_i[i]) for i in range(n_views)]# [2](4,256,32,32)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][2].squeeze())#(8192,512)
        feat_p2i = self.fuse_p2i_ds_2[0](feat_p2i)#(8192,256)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_ds_2[1])# [2](4,256,32,32)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][2])#(5895,32,256)
        feat_i2p = self.fuse_i2p_ds_2[0](feat_i2p.max(1)[0])#(5895,512)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_ds_2[1])#(5895,512)
        
        feat_p = feat_p + self.selfp(feat_p,self.self_attn_p3,n_views)
        feat_i = feat_i + self.selfi(feat_i,self.self_attn_i3,n_views)
        feat_i2p = feat_i2p + self.selfp(feat_i2p,self.self_attn_i2p3,n_views)
        feat_p2i = feat_p2i + self.selfi(feat_p2i,self.self_attn_p2i3,n_views)

        feat_p = feat_p + feat_i2p#(5895,512)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]# [2](4,256,32,32)

        
        # upsample0
        for block_i, block_op in enumerate(self.pcd_up_0):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)#(17420,256)

        feat_i = [
            self.cnn_up_0(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-1][i]), dim=1)) for i in range(n_views)]#[2](4,128,64,64)

        feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())#(32768,256)
        feat_p2i = self.fuse_p2i_up_0[0](feat_p2i)#(32768,128)
        feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_up_0[1])#[2](4,128,64,64)

        feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])# (17420,32,128)
        feat_i2p = self.fuse_i2p_up_0[0](feat_i2p.max(1)[0])# (17420,256)
        feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_up_0[1])# (17420,256)

        feat_p = feat_p + self.selfp(feat_p,self.self_attn_p4,n_views)
        feat_i = feat_i + self.selfi(feat_i,self.self_attn_i4,n_views)
        feat_i2p = feat_i2p + self.selfp(feat_i2p,self.self_attn_i2p4,n_views)
        feat_p2i = feat_p2i + self.selfi(feat_p2i,self.self_attn_p2i4,n_views)
        
        feat_p = feat_p + feat_i2p # (17420,256)
        feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]#[2](4,128,64,64)

        # upsample1
        for block_i, block_op in enumerate(self.pcd_up_1):
            if block_i % 2 == 1:
                feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
            feat_p = block_op(feat_p, batch)#(130982,32)

        feat_i = [
            self.cnn_up_1(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                     feat_i_encode[-2][i]), dim=1)) for i in range(n_views)]#[2](4,32,128,128)

        pcs_X = batch['points_img']     #[2](4,16384,3)
        pcs_F = self.map(feat_i, feat_p, batch) #[2](4,16384,32)

        
        
        pcs_F_sc4 = list(self.loftr_coarse4(pcs_F[0],pcs_F[1]))
        pcs_F = pcs_F+pcs_F_sc4
        
        if vps is not None:
            # Drop first viewpoint -- assumed to be identity transformation
            vps = vps[1:]
        elif self.align_cfg.algorithm == "weighted_procrustes":
            vps = []
            cor_loss = []
            for i in range(1, n_views):
                corr_i = get_correspondences(
                    P1=pcs_F[0],
                    P2=pcs_F[i],
                    P1_X=pcs_X[0],
                    P2_X=pcs_X[i],
                    num_corres=self.num_corres,
                    ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
                )
                Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

                vps.append(Rt_i)
                cor_loss.append(cor_loss_i)

                # add for visualization
                output[f"corres_0{i}"] = corr_i
                output[f"vp_{i}"] = Rt_i
        else:
            raise ValueError(f"How to align using {self.align_cfg.algorithm}?")

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        # Get RGB pointcloud as well for direct rendering
        pcs_rgb = [rgb.view(B, 3, -1).permute(0, 2, 1).contiguous() for rgb in rgbs]

        projs = []
        # get joint for all values
        if self.pointcloud_source == "joint":
            pcs_X_joint = torch.cat(pcs_X, dim=1)
            pcs_F_joint = torch.cat(pcs_F, dim=1)
            pcs_RGB_joint = torch.cat(pcs_rgb, dim=1)
            pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

        # Rasterize and Blend
        for i in range(n_views):
            if self.pointcloud_source == "other":
                # get joint for all values except the one
                pcs_X_joint = torch.cat(pcs_X[0:i] + pcs_X[i + 1: n_views], dim=1)
                pcs_F_joint = torch.cat(pcs_F[0:i] + pcs_F[i + 1: n_views], dim=1)
                pcs_RGB_joint = torch.cat(
                    pcs_rgb[0:i] + pcs_rgb[i + 1: n_views], dim=1
                )
                pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

            if i > 0:
                rot_joint_X = transform_points_Rt(pcs_X_joint, vps[i - 1])
                rot_joint_X = points_to_ndc(rot_joint_X, K, (H, W))
            else:
                rot_joint_X = points_to_ndc(pcs_X_joint, K, (H, W))
            projs.append(self.renderer(rot_joint_X, pcs_FRGB_joint))

        # Decode
        for i in range(n_views):
            proj_FRGB_i = projs[i]["feats"]
            proj_RGB_i = proj_FRGB_i[:, -3:]
            proj_F_i = proj_FRGB_i[:, :-3]

            output[f"rgb_render_{i}"] = proj_RGB_i
            output[f"ras_depth_{i}"] = projs[i]["depth"]
            output[f"cover_{i}"] = projs[i]["mask"].unsqueeze(1)  # useless
        
        return output

    def gather_p2i(self, feat_p, idx):
        feat_p = torch.cat((feat_p, torch.zeros_like(feat_p[:1, :])), 0)
        return feat_p[idx]

    def gather_i2p(self, feat_i, idx):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        feat_i2p = []
        for i in range(B):
            feat_i2p.append(src_feat_i[i].reshape(C,H*W).permute(1,0))
            feat_i2p.append(tgt_feat_i[i].reshape(C,H*W).permute(1,0))

        feat_i2p = torch.cat(feat_i2p, 0)
        feat_i2p = torch.cat((feat_i2p, torch.zeros_like(feat_i2p[:1, :])), 0)
        return feat_i2p[idx]  #N,16,C

    def fusep2i(self, feat_i, feat_p2i, layer):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        # src_feat_i = src_feat_i.reshape(B, C, H * W).permute(0, 2, 1)
        # tgt_feat_i = tgt_feat_i.reshape(B, C, H * W).permute(0, 2, 1)

        src_feat_p2i = []
        tgt_feat_p2i = []

        for i in range(2*B):
            if i % 2 == 0:
                src_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))#[4](1,16384,64)
            else:
                tgt_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))#[4](1,16384,64)

        src_feat_p2i = torch.vstack(src_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)#(4,64,128,128)
        tgt_feat_p2i = torch.vstack(tgt_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)#(4,64,128,128)

        src_feat_p2i = torch.cat([src_feat_i, src_feat_p2i], 1)#(4,128,128,128)
        tgt_feat_p2i = torch.cat([tgt_feat_i, tgt_feat_p2i], 1)#(4,128,128,128)

        src_feat_p2i = layer(src_feat_p2i)#(4,64,128,128)
        tgt_feat_p2i = layer(tgt_feat_p2i)#(4,64,128,128)

        return [src_feat_p2i, tgt_feat_p2i]

    def fusei2p(self, feat_p, feat_i2p, layer):
        feat_i2p = torch.cat([feat_p, feat_i2p], -1)
        feat_i2p = layer(feat_i2p)
        return feat_i2p


    def generate_pointclouds(self, K, deps, vps=None):
        n_views = len(deps)
        # generate pointclouds - generate grid once for efficiency
        B, _, H, W = deps[0].shape
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pcs_X = [
            grid_to_pointcloud(K_inv, deps[i], None, grid)[0] for i in range(n_views)
        ]

        if vps is not None:
            pcs_X_rot = [
                transform_points_Rt(pcs_X[i + 1], vps[i + 1], inverse=True, )
                for i in range(n_views - 1)
            ]
            pcs_X = pcs_X[0:1] + pcs_X_rot
            pcs_X = torch.cat(pcs_X, dim=1).detach().cpu()

        return pcs_X

    def get_feature_pcs(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert (
                feats[0].shape[-1] == deps[0].shape[-1]
        ), f"Same size {feats[0].shape} - {deps[0].shape}"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]
        return pcs_X, pcs_F, None    
    
    
    
    
    
    
    
    
    