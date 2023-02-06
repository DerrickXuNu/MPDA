import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.fuse_modules.fuse_utils import regroup
from opencood.models.fuse_modules.v2xvit_basic import V2XTransformer
from opencood.utils.common_utils import torch_tensor_to_numpy
from opencood.hypes_yaml.yaml_utils import save_yaml

# conv2d + bn + relu
class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, activation=True,
                 batch_norm=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k,
                              stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation:
            return F.relu(x, inplace=True)
        else:
            return x


# conv3d + bn + relu
class Conv3d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, batch_norm=True):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=k,
                              stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)

        return F.relu(x, inplace=True)


# Fully Connected Network
class FCN(nn.Module):

    def __init__(self, cin, cout):
        super(FCN, self).__init__()
        self.cout = cout
        self.linear = nn.Linear(cin, cout)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        # KK is the stacked k across batch
        kk, t, _ = x.shape
        x = self.linear(x.view(kk * t, -1))
        x = F.relu(self.bn(x))
        return x.view(kk, t, -1)


# Voxel Feature Encoding layer
class VFE(nn.Module):

    def __init__(self, cin, cout, T):
        super(VFE, self).__init__()
        assert cout % 2 == 0
        self.units = cout // 2
        self.fcn = FCN(cin, self.units)
        self.T = T

    def forward(self, x, mask):
        # point-wise feature
        pwf = self.fcn(x)
        # locally aggregated feature
        laf = torch.max(pwf, 1)[0].unsqueeze(1).repeat(1, self.T, 1)
        # point-wise concat feature
        pwcf = torch.cat((pwf, laf), dim=2)
        # apply mask
        mask = mask.unsqueeze(2).repeat(1, 1, self.units * 2)
        pwcf = pwcf * mask.float()

        return pwcf


# Stacked Voxel Feature Encoding
class SVFE(nn.Module):

    def __init__(self, T):
        super(SVFE, self).__init__()
        self.vfe_1 = VFE(7, 32, T)
        self.vfe_2 = VFE(32, 128, T)
        self.fcn = FCN(128, 128)

    def forward(self, x):
        mask = torch.ne(torch.max(x, 2)[0], 0)
        x = self.vfe_1(x, mask)
        x = self.vfe_2(x, mask)
        x = self.fcn(x)
        # element-wise max pooling
        x = torch.max(x, 1)[0]
        return x


# Convolutional Middle Layer
class CML(nn.Module):
    def __init__(self):
        super(CML, self).__init__()
        self.conv3d_1 = Conv3d(64, 128, 3, s=(2, 1, 1), p=(1, 1, 1))
        self.conv3d_2 = Conv3d(128, 128, 3, s=(1, 1, 1), p=(0, 1, 1))
        self.conv3d_3 = Conv3d(128, 256, 3, s=(2, 1, 1), p=(1, 1, 1))

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        return x

class VoxelNetV2XViT(nn.Module):
    def __init__(self, args):
        super(VoxelNetV2XViT, self).__init__()

        self.save_feature_flag = False
        if 'save_features' in args:
            self.save_feature_flag = args['save_features']
            self.save_folder = args['output_folder']
        self.max_cav = args['max_cav']

        self.svfe = PillarVFE(args['pillar_vfe'],
                              num_point_features=4,
                              voxel_size=args['voxel_size'],
                              point_cloud_range=args['lidar_range'])

        # self.svfe = SVFE(args['T'])
        self.cml = CML()

        self.N = args['N']
        self.D = args['D']
        self.H = args['H']
        self.W = args['W']
        self.T = args['T']
        self.anchor_num = args['anchor_num']

        self.fusion_net = V2XTransformer(args['transformer'])

        self.cls_head = nn.Conv2d(args['cls_head_dim'], args['anchor_num'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(args['cls_head_dim'], 7 * args['anchor_num'],
                                  kernel_size=1)

    def voxel_indexing(self, sparse_features, coords):
        dim = sparse_features.shape[-1]
        dtype = sparse_features.dtype
        dense_feature = Variable(
            torch.zeros(dim, self.N, self.D, self.H, self.W).cuda()).to(dtype)

        dense_feature[:, coords[:, 0], coords[:, 1], coords[:, 2],
        coords[:, 3]] = sparse_features.transpose(0, 1)

        return dense_feature.transpose(0, 1)


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

        if voxel_coords.is_cuda:
            record_len_tmp = record_len.cpu()

        record_len_tmp = list(record_len_tmp.numpy())

        self.N = sum(record_len_tmp)

        # feature learning network
        vwfs = self.svfe(batch_dict)['pillar_features']
        voxel_coords = torch_tensor_to_numpy(voxel_coords)
        vwfs = self.voxel_indexing(vwfs, voxel_coords)

        # convolutional middle network
        spatial_features_2d = self.cml(vwfs).view(self.N, -1, self.H, self.W)

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

