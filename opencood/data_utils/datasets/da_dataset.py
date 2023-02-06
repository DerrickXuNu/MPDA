"""
A dataset class for domain adaption in v2x-vit
"""

import os
import math
import time
from collections import OrderedDict

import torch
import numpy as np
from numpy import random
from torch.utils.data import Dataset

import opencood
import opencood.data_utils.post_processor as post_processor
from opencood.data_utils.datasets.basedataset import BaseDataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils import box_utils


class DADataset(Dataset):
    def __init__(self, params, visualize, train=True):
        self.params = params
        self.base_dataset = BaseDataset(params, visualize, train, False)
        # data path for source model
        params['root_dir'] = params['root_dir_s']
        params['validate_dir'] = params['validate_dir_s']
        self.base_dataset_s = BaseDataset(params, visualize, train, True)

        # data path for the target domain model
        params['root_dir'] = params['root_dir_t']
        params['validate_dir'] = params['validate_dir_t']
        self.base_dataset_t_1 = BaseDataset(params, visualize, train, True)
        self.base_dataset_t = self.base_dataset_t_1

        # data path for the target domain 2 model
        if 'root_dir_t_2' not in params:
            params['root_dir'] = params['root_dir_t']
            params['validate_dir'] = params['validate_dir_t']
        else:
            params['root_dir'] = params['root_dir_t_2']
            params['validate_dir'] = params['validate_dir_t_2']
        self.base_dataset_t_2 = BaseDataset(params, visualize, train, True)

        self.h_max, self.w_max, self.c_max = 0, 0, 0
        self.max_cav = self.base_dataset_t.max_cav

        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = post_processor.build_postprocessor(
            params['postprocess'],
            train)

    def __len__(self):
        return self.base_dataset_s.len_record[-1]

    def random_select(self):
        x = random.randint(2)
        if x == 0:
            self.base_dataset_t = self.base_dataset_t_1
        else:
            self.base_dataset_t = self.base_dataset_t_2

    def __getitem__(self, idx):
        base_data_dict_s = \
            self.base_dataset_s.retrieve_base_data(idx,
                                                   cur_ego_pose_flag=True)
        base_data_dict_t = \
            self.base_dataset_t.retrieve_base_data(idx,
                                                   cur_ego_pose_flag=True)
        base_data_dict = self.base_dataset.retrieve_base_data(idx,
                                                              cur_ego_pose_flag=True)

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict_s.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break
        assert cav_id == list(base_data_dict_s.keys())[
            0], "The first element in the OrderedDict must be ego"
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        # check the maximum size in h, w, c
        self.c_max = max(base_data_dict_s[ego_id]['npy'].shape[0],
                         base_data_dict_t[ego_id]['npy'].shape[0])
        self.h_max = max(base_data_dict_s[ego_id]['npy'].shape[1],
                         base_data_dict_t[ego_id]['npy'].shape[1])
        self.w_max = max(base_data_dict_s[ego_id]['npy'].shape[2],
                         base_data_dict_t[ego_id]['npy'].shape[2])

        processed_features = []
        feature_masks = []
        object_stack = []
        object_id_stack = []
        projected_lidar_stack = []

        # prior knowledge for time delay correction and indicating data type
        # (V2V vs V2i)
        velocity = []
        time_delay = []
        infra = []
        spatial_correction_matrix = []

        for ((cav_id_s, selected_cav_base_s), (cav_id_t, selected_cav_base_t),
             (cav_id_b, selected_cav_base_b)) \
                in zip(base_data_dict_s.items(), base_data_dict_t.items(),
                       base_data_dict.items()):
            assert cav_id_s == cav_id_t
            # check if the cav is within the communication range with ego
            distance = \
                math.sqrt((selected_cav_base_s['params']['lidar_pose'][0] -
                           ego_lidar_pose[0]) ** 2 + (
                                  selected_cav_base_s['params'][
                                      'lidar_pose'][1] - ego_lidar_pose[
                                      1]) ** 2)
            if distance > opencood.data_utils.datasets.COM_RANGE:
                continue

            if cav_id_s == ego_id:
                selected_cav_base = selected_cav_base_s
            else:
                selected_cav_base = selected_cav_base_t
            start = time.time()
            selected_cav_processed = self.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose,
                selected_cav_base_b)
            # print("data processing %f s" % (time.time() - start))

            object_stack.append(selected_cav_processed['object_bbx_center'])
            object_id_stack += selected_cav_processed['object_ids']
            processed_features.append(
                selected_cav_processed['processed_features'])
            feature_masks.append(selected_cav_processed['feature_masks'])

            velocity.append(selected_cav_processed['velocity'])
            time_delay.append(float(selected_cav_base['time_delay']))
            spatial_correction_matrix.append(
                selected_cav_base['params']['spatial_correction_matrix'])
            infra.append(1 if int(cav_id) < 0 else 0)

            projected_lidar_stack.append(
                selected_cav_processed['projected_lidar'])

        # exclude all repetitive objects
        unique_indices = \
            [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = \
            np.zeros((self.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        cav_num = len(processed_features)
        for i in range(self.max_cav - cav_num):
            processed_features.append(np.zeros_like(processed_features[0]))
            feature_masks.append(np.zeros_like(feature_masks[0]))

        processed_features = np.vstack(processed_features)
        feature_masks = np.vstack(feature_masks)

        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()

        # generate targets label
        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=mask)

        # pad dv, dt, infra to max_cav
        velocity = velocity + (self.max_cav - len(velocity)) * [0.]
        time_delay = time_delay + (self.max_cav - len(time_delay)) * [0.]
        infra = infra + (self.max_cav - len(infra)) * [0.]
        spatial_correction_matrix = np.stack(spatial_correction_matrix)
        padding_eye = np.tile(np.eye(4)[None], (self.max_cav - len(
            spatial_correction_matrix), 1, 1))
        spatial_correction_matrix = np.concatenate(
            [spatial_correction_matrix, padding_eye], axis=0)

        processed_data_dict['ego'].update(
            {'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_ids': [object_id_stack[i] for i in unique_indices],
             'anchor_box': anchor_box,
             'spatial_features': processed_features,
             'label_dict': label_dict,
             'cav_num': cav_num,
             'velocity': velocity,
             'time_delay': time_delay,
             'infra': infra,
             'spatial_correction_matrix': spatial_correction_matrix,
             'feature_masks': feature_masks})

        processed_data_dict['ego'].update({'origin_lidar':
            np.vstack(
                projected_lidar_stack)})

        return processed_data_dict

    def get_item_single_car(self, selected_cav_base, ego_pose, base_for_vis):
        selected_cav_processed = {}

        # calculate the transformation matrix
        transformation_matrix = \
            selected_cav_base['params']['transformation_matrix']

        # retrieve objects under ego coordinates
        object_bbx_center, object_bbx_mask, object_ids = \
            self.post_processor.generate_object_center([selected_cav_base],
                                                       ego_pose)

        # filter lidar
        lidar_np = base_for_vis['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        # remove points that hit itself
        lidar_np = mask_ego_points(lidar_np)
        # project the lidar to ego space
        lidar_np[:, :3] = \
            box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                     transformation_matrix)
        lidar_np = mask_points_by_range(lidar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])

        # process the saved feature
        spatial_2d_features = selected_cav_base['npy']
        feature_mask = np.zeros(3)

        padding_features = np.zeros((self.c_max, self.h_max, self.w_max))
        padding_features[:spatial_2d_features.shape[0],
        :spatial_2d_features.shape[1],
        :spatial_2d_features.shape[2]] = spatial_2d_features

        feature_mask[0] = spatial_2d_features.shape[0]
        feature_mask[1] = spatial_2d_features.shape[1]
        feature_mask[2] = spatial_2d_features.shape[2]

        # velocity
        velocity = selected_cav_base['params']['ego_speed']
        # normalize veloccity by average speed 30 km/h
        velocity = velocity / 30

        selected_cav_processed.update(
            {'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
             'object_ids': object_ids,
             'projected_lidar': lidar_np,
             'processed_features': padding_features[None, ...],
             'feature_masks': feature_mask[None, ...],
             'velocity': velocity})

        return selected_cav_processed

    def reinitialize(self):
        self.base_dataset_t.reinitialize()
        self.base_dataset_s.reinitialize()

    def collate_batch_train(self, batch):
        start = time.time()
        # Intermediate fusion is different the other two
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        object_ids = []
        spatial_feature_list = []
        feature_mask_list = []

        # used to record different scenario
        record_len = []
        label_dict_list = []

        # used for PriorEncoding
        velocity = []
        time_delay = []
        infra = []

        # used for correcting the spatial transformation between delayed
        # timestamp and current timestamp
        spatial_correction_matrix_list = []

        origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            object_ids.append(ego_dict['object_ids'])

            spatial_feature_list.append(ego_dict['spatial_features'])
            feature_mask_list.append(ego_dict['feature_masks'])
            record_len.append(ego_dict['cav_num'])
            label_dict_list.append(ego_dict['label_dict'])

            velocity.append(ego_dict['velocity'])
            time_delay.append(ego_dict['time_delay'])
            infra.append(ego_dict['infra'])
            spatial_correction_matrix_list.append(
                ego_dict['spatial_correction_matrix'])

            origin_lidar.append(ego_dict['origin_lidar'])

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        spatial_features = \
            torch.from_numpy(np.array(spatial_feature_list)).float()
        feature_masks = torch.from_numpy(np.array(feature_mask_list))
        # [2, 3, 4, ..., M]
        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        label_torch_dict = \
            self.post_processor.collate_batch(label_dict_list)

        # (B, max_cav)
        velocity = torch.from_numpy(np.array(velocity))
        time_delay = torch.from_numpy(np.array(time_delay))
        infra = torch.from_numpy(np.array(infra))
        spatial_correction_matrix_list = \
            torch.from_numpy(np.array(spatial_correction_matrix_list)).float()
        # (B, max_cav, 3)
        prior_encoding = \
            torch.stack([velocity, time_delay, infra], dim=-1).float()

        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'spatial_features': spatial_features,
                                   'record_len': record_len,
                                   'label_dict': label_torch_dict,
                                   'object_ids': object_ids[0],
                                   'prior_encoding': prior_encoding,
                                   'spatial_correction_matrix': spatial_correction_matrix_list,
                                   'feature_masks': feature_masks})
        # print("collate time  %f s" % (time.time() - start))

        origin_lidar = \
            np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
        origin_lidar = torch.from_numpy(origin_lidar)
        output_dict['ego'].update({'origin_lidar': origin_lidar})

        return output_dict

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict = self.collate_batch_train(batch)

        # check if anchor box in the batch
        if batch[0]['ego']['anchor_box'] is not None:
            output_dict['ego'].update({'anchor_box':
                torch.from_numpy(np.array(
                    batch[0]['ego'][
                        'anchor_box']))})

        # save the transformation matrix (4, 4) to ego vehicle
        transformation_matrix_torch = \
            torch.from_numpy(np.identity(4)).float()
        output_dict['ego'].update({'transformation_matrix':
                                       transformation_matrix_torch})

        return output_dict

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_box_tensor, pred_score = \
            self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor

    def visualize_result(self, pred_box_tensor,
                         gt_tensor,
                         pcd,
                         show_vis,
                         save_path,
                         dataset=None):
        self.post_processor.visualize(pred_box_tensor,
                                      gt_tensor,
                                      pcd,
                                      show_vis,
                                      save_path,
                                      dataset=dataset)

if __name__ == '__main__':
    from opencood.hypes_yaml.yaml_utils import load_yaml

    params = load_yaml('../../hypes_yaml/v2xvit_da.yaml')

    opencda_dataset = DADataset(params, train=True, visualize=True)
    test = opencda_dataset.__getitem__(10)
    print(test)
