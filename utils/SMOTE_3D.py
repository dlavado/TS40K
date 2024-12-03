

import torch
import numpy as np
from torch_cluster import knn
from typing import List, Tuple

import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
from utils.torch_knn import torch_knn

class SMOTE3D:
    def __init__(self, k_neighbors=5, sampling_strategy=1.0, smote_label=None, num_points_resampled=None):
        """
        Initialize the SMOTE3D instance.

        Parameters
        ----------

        k_neighbors : int
            Number of neighbors to consider when generating synthetic samples

        sampling_strategy : float
            Ratio of synthetic samples to generate relative to the number of minority samples

        smote_label : int
            Label of the minority class; if None, the minority class is identified automatically

        num_points_resampled : int
            Number of 3D points on each sample in order to "batchify" the samples
        """
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.smote_label = smote_label
        self.num_points_resampled = num_points_resampled

    def fit_resample(self, X:torch.Tensor, y:torch.Tensor, random_add=True) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Apply SMOTE (Synthetic Minority Over-sampling Technique) to the input data.

        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (N, num_points, 3)

        y : torch.Tensor
            Labels of shape (N, num_points)

        random_add : bool
            Whether to randomly add synthetic samples to the minority class;
            If False, the points selected to add synthetic samples are the closest to the original minority points
        Returns
        -------

        X_resampled : List[torch.Tensor]
            Resampled data of shape (N, num_points_resampled, 3); warning: num_points_resampled may be different for each sample, which is why we return a list of tensors

        y_resampled : List[torch.Tensor]
            Resampled labels of shape (N, num_points_resampled)
        """

        X_resampled = []
        y_resampled = []

        for i in range(len(X)):
            X_i = X[i] # sample is sent to device for faster computation
            y_i = y[i].long()

            if self.smote_label is not None:
                minority_class = self.smote_label
            else:
                # Identify minority class points
                class_freq = torch.bincount(y_i.reshape(-1))
                minority_class = torch.argmin(class_freq)
            
            # minority points
            minority_points = X_i[y_i == minority_class]

            if len(minority_points) == 0: # if no minority points, skip
                X_resampled.append(X_i)
                y_resampled.append(y_i)
                continue

            # Find k-nearest neighbors for each minority point
            edge_index = knn(minority_points, minority_points, self.k_neighbors).t()
            
            synthetic_points = []
            num_minority_samples = len(minority_points)
            
            for i in range(len(minority_points)):
                # Get neighbors for the current point
                neighbors = edge_index[edge_index[:, 0] == i][:, 1] # first we select the row where the current point is the source, then we select the target column

                # Randomly select neighbors
                neighbor_indices = torch.randperm(len(neighbors))[:self.k_neighbors - 1]
                for neighbor_idx in neighbor_indices:
                    # Generate synthetic points
                    alpha = torch.rand(3, device=X_i.device)
                    synthetic_point = minority_points[i] + alpha * (minority_points[neighbor_idx] - minority_points[i])
                    synthetic_points.append(synthetic_point)
            
            if len(synthetic_points) == 0: # if no synthetic points, skip
                X_resampled.append(X_i)
                y_resampled.append(y_i)
                continue

            synthetic_points = torch.stack(synthetic_points)
            num_synthetic_samples = len(synthetic_points)
            
            # Determine how many synthetic samples to add
            num_samples_to_add = int(self.sampling_strategy * min(num_synthetic_samples, num_minority_samples))
            #num_samples_to_add = int((self.sampling_strategy * num_minority_samples) - num_synthetic_samples)
            
            if num_samples_to_add > 0:
                # Randomly select synthetic samples to add    
                if random_add: 
                    indices_to_add = torch.randperm(num_synthetic_samples)[:num_samples_to_add]
                else:
                    # Calculate distances between synthetic points and original points
                    distances = torch.cdist(synthetic_points, minority_points)
                    # Sort distances in ascending order
                    _, indices = torch.sort(distances, dim=1)
                    # Select the indices of the closest original points
                    indices_to_add = indices[:, :num_samples_to_add].reshape(-1)
                
                synthetic_samples_to_add = synthetic_points[indices_to_add]
                synthetic_points = torch.cat([synthetic_points, synthetic_samples_to_add], dim=0)
                
            # Combine minority class points with synthetic points
            X_i_resampled = torch.cat([X_i, synthetic_points], dim=0)
            y_i_resampled = torch.cat([y_i, torch.full((len(synthetic_points),), fill_value=minority_class, device=y_i.device)], dim=0)

            X_resampled.append(X_i_resampled)
            y_resampled.append(y_i_resampled)
                
        return X_resampled, y_resampled
    

    def fit_resample_batch(self, X:torch.Tensor, y:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply SMOTE (Synthetic Minority Over-sampling Technique) to the input data.

        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (B, num_points, 3)

        y : torch.Tensor
            Labels of shape (B, num_points)

        Returns
        -------

        X_resampled : torch.Tensor
            Resampled data of shape (N, num_points_resampled, 3)

        y_resampled : torch.Tensor
            Resampled labels of shape (N, num_points_resampled)
        """

        smote_x, smote_y = self.fit_resample(X, y, random_add=True)

        if self.num_points_resampled is None:
            return smote_x, smote_y

        for i in range(len(smote_x)):
            if smote_x[i].shape[0] < self.num_points_resampled: # if the number of points is less than the desired number of points
                num_points_to_add = self.num_points_resampled - smote_x[i].shape[0]
                indices_to_add = torch.randint(0, smote_x[i].shape[0], (num_points_to_add,))
                smote_x[i] = torch.cat([smote_x[i], smote_x[i][indices_to_add]], dim=0)
                smote_y[i] = torch.cat([smote_y[i], smote_y[i][indices_to_add]], dim=0)
            elif smote_x[i].shape[0] > self.num_points_resampled: # if the number of points is greater than the desired number of points
                indices_to_keep = torch.randint(0, smote_x[i].shape[0], (self.num_points_resampled,))
                smote_x[i] = smote_x[i][indices_to_keep]
                smote_y[i] = smote_y[i][indices_to_keep]

        return torch.stack(smote_x), torch.stack(smote_y)

        




    


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '..')
    sys.path.insert(1, '../..')
    from core.datasets.TS40K import TS40K_FULL_Preprocessed
    from utils import constants
    from utils import pointcloud_processing as eda

    # %%

    ts40k = TS40K_FULL_Preprocessed(constants.TS40K_FULL_PREPROCESSED_PATH, split='fit', sample_types=['tower_radius'], transform=None, load_into_memory=True)

    # %%
    tup = ts40k.data
    X, Y = [], []
    for i in range(len(tup)):
        X.append(tup[i][0])
        Y.append(tup[i][1])

    X = torch.from_numpy(np.array(X))
    Y = torch.from_numpy(np.array(Y)).long()

    print(X.shape)
    print(print(Y.shape))

    # Example usage:
    # Assuming X is your 3D point cloud data (torch.Tensor) and y is the corresponding labels (torch.Tensor)
    # Initialize SMOTE3D instance
    smote = SMOTE3D(k_neighbors=3, sampling_strategy=0.5, smote_label=4, num_points_resampled=11024)

    # Apply SMOTE
    # X_resampled, y_resampled = smote.fit_resample(X, Y, True)
    X_resampled, y_resampled = smote.fit_resample_batch(X, Y)
    print(X_resampled.shape, y_resampled.shape)
    
    # %%
    # Visualize the resampled data
    idx = torch.randint(0, len(X), (1,)).item()
    idx = 2164
    print(f"Sample index: {idx}")

    xyz, y = X[idx], Y[idx]
    y = y.reshape(-1).numpy()
    xyz = xyz.squeeze().numpy()
    pynt = eda.np_to_ply(xyz)
    eda.color_pointcloud(pynt, y, use_preset_colors=True)
    eda.visualize_ply([pynt])

    xyz, y = X_resampled[idx], y_resampled[idx]
    y = y.reshape(-1).numpy()
    xyz = xyz.squeeze().numpy()
    pynt = eda.np_to_ply(xyz)
    eda.color_pointcloud(pynt, y, use_preset_colors=True)
    eda.visualize_ply([pynt])

# %%
