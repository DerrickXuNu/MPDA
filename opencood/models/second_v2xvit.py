import os

import numpy as np
import torch
import torch.nn as nn

from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.fuse_modules.fuse_utils import regroup
from opencood.models.fuse_modules.v2xvit_basic import V2XTransformer
from opencood.utils.common_utils import torch_tensor_to_numpy
from opencood.hypes_yaml.yaml_utils import save_yaml


class SecondV2XViT(nn.Module):
    def __init__(self, args):
        super(SecondV2XViT, self).__init__()

        self.save_feature_flag = False
        if 'save_features' in args:
            self.save_feature_flag = args['save_features']
            self.save_folder = args['output_folder']
        self.max_cav = args['max_cav']

        # mean_vfe
        self.mean_vfe = MeanVFE(args['mean_vfe'], 4)
        # sparse 3d backbone
        self.backbone_3d = VoxelBackBone8x(args['backbone_3d'],
                                           4, args['grid_size'])
        # height compression
        self.height_compression = HeightCompression(args['height_compression'])
        # base ben backbone
        self.backbone_2d = \
            BaseBEVBackbone(args['base_bev_backbone'],
                            args['height_compression']['feature_num'])
        self.fusion_net = V2XTransformer(args['transformer'])

        self.cls_head = nn.Conv2d(args['cls_head_dim'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(args['cls_head_dim'], 7 * args['anchor_number'],
                                  kernel_size=1)

    def forward(self, data_dict):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        spatial_correction_matrix = data_dict['spatial_correction_matrix']

        # B, max_cav, 3(dt dv infra), 1, 1
        prior_encoding =\
            data_dict['prior_encoding'].unsqueeze(-1).unsqueeze(-1)
        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'batch_size': torch.sum(record_len).cpu().numpy(),
                      'record_len': record_len}

        batch_dict = self.mean_vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.height_compression(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']
        if self.save_feature_flag:
            self.save_feature(spatial_features_2d, data_dict)

        # N, C, H, W -> B,  L, C, H, W
        regroup_feature, mask = regroup(spatial_features_2d,
                                        record_len,
                                        self.max_cav)
        # prior encoding added
        prior_encoding = prior_encoding.repeat(1, 1, 1,
                                               regroup_feature.shape[3],
                                               regroup_feature.shape[4])
        regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)

        # b l c h w -> b l h w c
        regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
        # transformer fusion
        fused_feature = self.fusion_net(regroup_feature,
                                        mask,
                                        spatial_correction_matrix)
        # b h w c -> b c h w
        fused_feature = fused_feature.permute(0, 3, 1, 2)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict

    def save_feature(self, spatial_2d, data_dict):
        """
        Save the features in the folder for later training.

        Parameters
        ----------
        spatial_2d : torch.tensor
            Spatial features, N C H W

        data_dict: dict
            Metadata.
        """
        index = 0

        for cav_id, cav_content in data_dict['raw_info'][0].items():
            scene = cav_content['scene']
            output_folder = os.path.join(self.save_folder, scene, cav_id)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            timestamp = cav_content['timestamp']

            if not cav_content['valid']:
                save_array = \
                    np.zeros_like(torch_tensor_to_numpy(spatial_2d[0]))
            else:
                save_array = torch_tensor_to_numpy(spatial_2d[index])
                index += 1

            # save the data
            save_yml_name = os.path.join(output_folder, timestamp + '.yaml')
            save_feature_name = os.path.join(output_folder, timestamp + '.npz')
            save_yaml(cav_content['yaml'], save_yml_name)
            np.savez_compressed(save_feature_name, save_array)

