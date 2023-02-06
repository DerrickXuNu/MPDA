import torch
import torch.nn as nn

from opencood.models.fuse_modules.fuse_utils import regroup
from opencood.models.fuse_modules.v2xvit_basic import V2XTransformer
from opencood.models.sub_modules.resizer import LearnableResizer
from opencood.models.fuse_modules.wg_fusion_modules import CrossDomainFusionEncoder
from opencood.models.da_modules.classfier import DAImgHead


class V2XViTDA(nn.Module):
    def __init__(self, args):
        super(V2XViTDA, self).__init__()

        self.max_cav = args['max_cav']
        self.resizer = LearnableResizer(args['resizer'])
        self.cdt = CrossDomainFusionEncoder(args['cdt'])
        self.classifier = DAImgHead(256)

        self.fusion_net = V2XTransformer(args['transformer'])

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)

    def forward(self, data_dict):
        record_len = data_dict['record_len']
        spatial_correction_matrix = data_dict['spatial_correction_matrix']

        # B, max_cav, 3(dt dv infra), 1, 1
        prior_encoding =\
            data_dict['prior_encoding'].unsqueeze(-1).unsqueeze(-1)
        spatial_features_2d = data_dict['spatial_features']
        feature_masks = data_dict['feature_masks']

        b, _, c, h, w = spatial_features_2d.shape
        reshaped_spatial_feature_list = []

        # iterate through each batcth to resize others to ego space
        for i in range(b):
            cur_feature_batch, cur_feature_mask = spatial_features_2d[i], \
                                                  feature_masks[i]
            ego_feature_mask = cur_feature_mask[0]
            ego_feature = cur_feature_batch[0]
            ego_feature = ego_feature[:int(ego_feature_mask[0]),
                                      :int(ego_feature_mask[1]),
                                      :int(ego_feature_mask[2])]

            cav_feature_mask = cur_feature_mask[1:record_len[i]]
            cav_feature = cur_feature_batch[1:record_len[i]]

            if cav_feature.shape[0] > 0:
                cav_feature = cav_feature[:,
                                          :int(cav_feature_mask[0, 0]),
                                          :int(cav_feature_mask[0, 1]),
                                          :int(cav_feature_mask[0, 2])]
                cav_feature = self.resizer(ego_feature, cav_feature)
                ego_copy = ego_feature[None].repeat(cav_feature.shape[0],
                                                    1, 1, 1)
                cav_feature = self.cdt(ego_copy, cav_feature)

                reshaped_spatial_feature_list.append(torch.cat((ego_feature[None],
                                                               cav_feature),
                                                               dim=0))
            else:
                reshaped_spatial_feature_list.append(ego_feature[None])

        spatial_features_2d = torch.cat(reshaped_spatial_feature_list, dim=0)
        # for domain classifier
        da_feature = self.classifier(spatial_features_2d)

        # print("reshapling caused %f s" % (time.time() - start))
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
        fused_feature = self.fusion_net(regroup_feature, mask, spatial_correction_matrix)
        # b h w c -> b c h w
        fused_feature = fused_feature.permute(0, 3, 1, 2)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm,
                       'rm': rm,
                       'da_feature': da_feature,
                       'record_len': record_len}

        return output_dict

