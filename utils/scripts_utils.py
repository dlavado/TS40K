
import argparse
from typing import List
import numpy as np
import torch
import random
import pytorch_lightning as pl
import sys

from torchmetrics import MetricCollection, JaccardIndex, F1Score, Accuracy, Precision, Recall
import wandb

sys.path.insert(0, '..')

from core.criterions.dice_loss import BinaryDiceLoss, BinaryDiceLoss_BCE
from core.criterions.tversky_loss import FocalTverskyLoss, TverskyLoss
from core.criterions.w_mse import WeightedMSE
from core.criterions.geneo_loss import GENEO_Loss, Tversky_Wrapper_Loss



class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value


def fix_randomness(seed=0):
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    pl.seed_everything(hash("setting random seeds") % 2**32 - 1)


def main_arg_parser():
    parser = argparse.ArgumentParser(description="Process script arguments")

    parser.add_argument('--wandb_mode', type=str, default=None, help='Mode of the wandb.init function') # 'disabled' for no recording

    parser.add_argument('--wandb_sweep', action='store_true', default=None, help='If True, the script is run by wandb sweep')

    parser.add_argument('--dataset', type=str, default='ts40k', help='Dataset to use')

    parser.add_argument('--model', type=str, default='scenenet', help='Model to use')

    return parser


def _resolve_geneo_criterions(criterion_name):
    criterion_name = criterion_name.lower()
    if criterion_name == 'geneo':
        return GENEO_Loss
    elif criterion_name == 'geneo_tversky':
        return Tversky_Wrapper_Loss
    else:
        raise NotImplementedError(f'GENEO Criterion {criterion_name} not implemented')


def resolve_criterion(criterion_name):
    criterion_name = criterion_name.lower()
    if criterion_name == 'mse':
        return WeightedMSE
    elif criterion_name == 'dice':
        return BinaryDiceLoss
    elif criterion_name == 'dice_bce':
        return BinaryDiceLoss_BCE
    elif criterion_name == 'tversky':
        return TverskyLoss
    elif criterion_name == 'focal_tversky':
        return FocalTverskyLoss
    elif 'geneo' in criterion_name: 
        return _resolve_geneo_criterions(criterion_name)
    else:
        raise NotImplementedError(f'Criterion {criterion_name} not implemented')
    

def resolve_optimizer(optimizer_name:str, model, learning_rate):
        optimizer_name = optimizer_name.lower()
        if  optimizer_name == 'adam':
            return torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'lbfgs':
            return torch.optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=20)
        
        raise NotImplementedError(f'Optimizer {optimizer_name} not implemented')
    

def init_metrics(task='multiclass', tau=0.5, num_classes=2, ignore_index=-1):

    params = {'task': task, 'num_classes': num_classes, 'ignore_index': ignore_index, 'threshold': tau}
    # 'multidim_average': 'global'
    return MetricCollection([
        JaccardIndex(**params, average=None),
        # JaccardIndex(**params, average=None),
        F1Score(**params, average=None),
        Precision(**params, average='micro'),
        Recall(**params, average='micro'),
        Accuracy(**params, average='micro'),
    ])


def pointcloud_to_wandb(pcd:np.ndarray, input=None, gt=None) -> List[wandb.Object3D]:
    """
    Converts a point cloud to a wandb Object3D object.

    Parameters
    ----------

    `pcd` - np.ndarray:
        The point cloud to be converted. The shape of the point cloud must be (N, 3) or (N, 4) or (N, 6) where N is the number of points.

    `input` - np.ndarray:
        The input point cloud. The shape of the point cloud must be (N, 3) or (N, 4) or (N, 6) where N is the number of points.

    `gt` - np.ndarray:
        The ground truth point cloud. The shape of the point cloud must be (N, 3) or (N, 4) or (N, 6) where N is the number of points.

    Returns
    -------
    list of wandb.Object3D
        A list of Object3D objects that can be logged to wandb.
    """
    # Log point clouds to wandb
    point_clouds = []
    if input is not None:
        input_cloud = wandb.Object3D(input)
        point_clouds.append(input_cloud)

    if gt is not None:
        ground_truth = wandb.Object3D(gt)
        point_clouds.append(ground_truth)

    prediction = wandb.Object3D(pcd)
    point_clouds.append(prediction)
    
    return point_clouds
