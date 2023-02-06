"""
Learnable Resizer
"""

import torch
from torch import nn

from opencood.models.fuse_modules.wg_fusion_modules import SwapFusionEncoder


class residual_block(nn.Module):
    def __init__(self, input_dim):
        super(residual_block, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 3, padding=1),
            nn.BatchNorm2d(input_dim),
            nn.LeakyReLU(),
            nn.Conv2d(input_dim, input_dim, 3, padding=1),
            nn.BatchNorm2d(input_dim)
        )

    def forward(self, x):
        x = x + self.module(x)
        return x


class LearnableResizer(nn.Module):
    def __init__(self, args):
        super(LearnableResizer, self).__init__()
        # channel selection
        self.channel_selector = nn.Conv2d(args['input_channel'],
                                          args['output_channel'],
                                          1)
        # window+grid attention
        self.wg_att_1 = SwapFusionEncoder(args['wg_att'])
        self.wg_att_2 = SwapFusionEncoder(args['wg_att'])


        self.res_blocks = nn.ModuleList()
        num_res = args['residual']['depth']
        input_channel = args['residual']['input_dim']

        # residual block
        for i in range(num_res):
            self.res_blocks.append(residual_block(input_channel))

    def forward(self, ego_feature, cav_feature):
        cav_feature = self.channel_selector(cav_feature)

        _, h, w = ego_feature.shape
        # self attention
        cav_feature_1 = self.wg_att_1(cav_feature)
        # naive feature resizer
        cav_feature_1 = torch.nn.functional.interpolate(cav_feature_1,
                                                      [h,
                                                       w],
                                                      mode='bilinear',
                                                      align_corners=False)
        cav_feature_2 = cav_feature_1
        for res_bloc in self.res_blocks:
            cav_feature_2 = res_bloc(cav_feature_2)
        cav_feature_2 += cav_feature_1
        cav_feature_2 = self.wg_att_2(cav_feature_2)

        # residual shortcut
        cav_feature_0 = torch.nn.functional.interpolate(cav_feature,
                                                      [h,
                                                       w],
                                                      mode='bilinear',
                                                      align_corners=False)
        return cav_feature_0 + cav_feature_2



