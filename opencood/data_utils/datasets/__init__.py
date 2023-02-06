from opencood.data_utils.datasets.intermediate_fusion_dataset import IntermediateFusionDataset
from opencood.data_utils.datasets.da_dataset import DADataset

__all__ = {
    'IntermediateFusionDataset': IntermediateFusionDataset,
    'DADataset': DADataset
}

# the final range for evaluation
GT_RANGE = [-140, -40, -3, 140, 40, 1]
# The communication range for cavs
COM_RANGE = 70


def build_dataset(dataset_cfg, visualize=False, train=True):
    dataset_name = dataset_cfg['fusion']['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in opencood/" \
                    f"data_utils/datasets/init.py"
    assert dataset_name in ['LateFusionDataset', 'EarlyFusionDataset',
                            'IntermediateFusionDataset', 'DADataset'], error_message

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train
    )

    return dataset
