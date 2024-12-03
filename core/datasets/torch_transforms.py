
from typing import Any, List, Optional, Sequence, Tuple, Union
import torch
import numpy as np


import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')

from utils import voxelization as Vox
from utils import pointcloud_processing as eda


class Dict_to_Tuple:
    def __init__(self, omit:Union[str, list]=None) -> None:
        self.omit = omit

    def __call__(self, sample:dict):
        return tuple([sample[key] for key in sample.keys() if key not in self.omit])


class Add_Batch_Dim:

    def __call__(self, sample) -> Any:
        sample = list(sample)
        return tuple([s.unsqueeze(0) for s in sample])

class ToTensor:
    def __call__(self, sample):
        sample = list(sample)
        return tuple([torch.from_numpy(s.astype(np.float64)) for s in sample])

class To:
    def __init__(self, dtype:torch.dtype=torch.float32) -> None:
        self.dtype = dtype

    def __call__(self, sample):
        sample = list(sample)
        return tuple([s.to(self.dtype) for s in sample])

class ToDevice:

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, sample):
        if isinstance(sample, Sequence):
            return tuple([s.to(self.device) for s in sample])
        elif isinstance(sample, dict):
            return {key: value.to(self.device) for key, value in sample.items()}
        else:
            return sample.to(self.device)

class ToFullDense:
    """
    Transforms a Regression Dataset into a Belief Dataset.

    Essentially, any voxel that has tower points is given a belief of 1,
    in order to maximize the towers' geometry.
    For the input, the density is normalized to 1, so empty voxels have a value
    of 0 and 1 otherwise.

    It requires a discretization of raw LiDAR Point Clouds in Torch format.
    """

    def __init__(self, apply=[True, True]) -> None:
        
        self.apply = apply
    
    def densify(self, tensor:torch.Tensor):
        return (tensor > 0).to(tensor)

    def __call__(self, sample:torch.Tensor):

        vox, gt = [self.densify(tensor) if self.apply[i] else tensor for i, tensor in enumerate(sample) ]
        
        return vox, gt



class Voxelization:

    def __init__(self, keep_labels='all', vox_size:Tuple[int]=None, vxg_size:Tuple[int]=None) -> None:
        """
        Voxelizes raw LiDAR 3D point points in (N, 3) format 
        according to the provided discretization

        Parameters
        ----------
        `vox_size` - Tuple of 3 Ints:
            Size of the voxels to dicretize the point clouds
        `vxg_size` - Tuple of 3 Ints:
            Size of the voxelgrid used to discretize the point clouds

        One of the two parameters need to be provided, `vox_size` takes priority

        Returns
        -------
        A Voxelized 3D point cloud in Density/Probability mode
        """
        
        if vox_size is None and vxg_size is None:
            ValueError("Voxel size or Voxelgrid size must be provided")

        self.vox_size = vox_size
        self.vxg_size = vxg_size
        self.keep_labels = keep_labels


    def __call__(self, sample:torch.Tensor):
        
        pts, labels = sample

        if pts.dim() == 3: # batched point clouds
            pts = pts[0] # decapsulate the batch dimension
        if labels.dim() == 2: # batched labels
            labels = labels[0] # decapsulate the batch dimension

        if pts.shape[1] > 3: # if the point cloud has features
            pts = pts[:, :3] # keep only the xyz coordinates  

        vox, gt = Vox.torch_voxelize_pcd_gt(
            pts, labels, self.keep_labels,
            voxel_dims=self.vox_size, voxelgrid_dims=self.vxg_size
        )

        vox = vox[None]
        # gt = gt[None] # encapsulate the batch dimension  

        #print shapes
        # print(f"vox shape: {vox.shape}, gt shape: {gt.shape}")

        return vox, gt
    

class Voxelization_withPCD:

    def __init__(self, keep_labels=None, vox_size:Tuple[int]=None, vxg_size:Tuple[int]=None) -> None:
        """
        Voxelizes raw LiDAR 3D point points in `numpy` (N, 3) format 
        according to the provided discretization

        Parameters
        ----------
        `vox_size` - Tuple of 3 Ints:
            Size of the voxels to dicretize the point clouds
        `vxg_size` - Tuple of 3 Ints:
            Size of the voxelgrid used to discretize the point clouds

        One of the two parameters need to be provided, `vox_size` takes priority

        Returns
        -------
        A Voxelized 3D point cloud in Density/Probability mode
        """
        
        if vox_size is None and vxg_size is None:
            ValueError("Voxel size or Voxelgrid size must be provided")


        self.vox_size = vox_size
        self.vxg_size = vxg_size
        self.keep_labels = keep_labels


    def __call__(self, sample:torch.Tensor):
        
        pts, labels = sample
        point_feats = None

        if pts.dim() == 3: # batched point clouds
            pts = pts[0] # decapsulate the batch dimension
        if labels.dim() == 2: # batched labels
            labels = labels[0] # decapsulate the batch dimension

        if pts.shape[1] > 3: # if the point cloud has features
            point_feats = pts[:, 3:] # keep the features
            pts = pts[:, :3] # keep only the xyz coordinates  

        vox, gt, pt_locs = Vox.torch_voxelize_input_pcd(
            pts, labels, self.keep_labels,
            voxel_dims=self.vox_size, voxelgrid_dims=self.vxg_size
        )
        
        if point_feats is not None:
            pt_locs = torch.concatenate([pt_locs, point_feats], dim=-1) # add the features to the point locations

        #print shapes
        # print(f"vox shape: {vox.shape}, gt shape: {gt.shape}, pt_locs shape: {pt_locs.shape}")

        # pt_locs = pt_locs[None]
        vox = vox[None]
        # gt = gt[None] # encapsulate the batch dimension  
        
        return vox, gt, pt_locs
    

class EDP_Labels:
    def __call__(self, sample) -> Any:
        pcd, labels, *args = sample

        labels = self.edp_labels(labels)

        return pcd, labels, *args
    
    def edp_labels(self, labels:torch.Tensor) -> torch.Tensor:
        #cast each label to its corresponding EDP label
        new_labels = torch.tensor([eda.DICT_NEW_LABELS[label.item()] if label.item() >= 0 else label.item() for label in labels.squeeze()]).reshape(labels.shape)
        # print(f"labels NEW unique: {torch.unique(new_labels)}, labels shape: {new_labels.shape}")
        return new_labels
    

class Normalize_Labels:

    def __call__(self, sample) -> Any:
        """
        Normalize the labels to be between [0, num_classes-1]
        """

        pointcloud, labels, pt_locs = sample

        labels = self.normalize_labels(labels)

        return pointcloud, labels, pt_locs
    
    def normalize_labels(self, labels:torch.Tensor) -> torch.Tensor:
        """

        labels - tensor with shape (P,) and values in [0, C -1] not necessarily contiguous
        

        transform the labels to be between [0, num_classes-1] with contiguous values
        """

        unique_labels = torch.unique(labels)
        num_classes = unique_labels.shape[0]
        
        labels = labels.unsqueeze(-1) # shape = (P, 1)
        labels = (labels == unique_labels).float() # shape = (P, C)
        labels = labels * torch.arange(num_classes).to(labels.device) # shape = (P, C)
        labels = labels.sum(dim=-1).long() # shape = (P,)
       
        return labels


class Ignore_Label:

    def __init__(self, ignore_label:int) -> None:
        self.ignore_label = ignore_label

    def __call__(self, sample) -> Any:
        """
        Ignore the points with the ignore label
        """

        pointcloud, labels = sample

        mask = labels == self.ignore_label

        # if pointcloud.ndim >= 3:
        #     pointcloud[mask[None]] = -1 # only if 
        labels[mask] = -1 # ignore the points with the ignore label

        return pointcloud, labels   



class Remove_Label:

    def __init__(self, remove_label:int) -> None:
        self.remove_label = remove_label

    def __call__(self, sample) -> Any:
        """
        Remove the points with the remove label
        """

        pointcloud, labels = sample

        mask = labels != self.remove_label

        pointcloud = pointcloud[mask]
        labels = labels[mask]

        return pointcloud, labels

class Merge_Label:

    def __init__(self, merge_labels:dict[int, int]) -> None:
        """
        merge_labels = {
            label_to_transform: new_label
        }
        """
        self.merge_labels = merge_labels

    def __call__(self, sample) -> Any:
        """
        Merge the labels according to the provided dictionary
        """

        pointcloud, labels = sample

        for key, value in self.merge_labels.items():
            labels[labels == key] = value

        return pointcloud, labels

class Add_Normal_Vector:

    def __call__(self, sample) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add the normal vector to the point cloud
        """

        pointcloud, labels = sample

        normals = eda.estimate_normals(pointcloud.numpy())
        normals = torch.from_numpy(normals).to(pointcloud.device).to(torch.float32)
        pointcloud = torch.cat([pointcloud, normals], dim=-1)

        return pointcloud, labels


class Repeat_Points:

    def __init__(self, num_points:int) -> None:
        """
        Repeat the points in the point cloud until the number of points is equal to the number of points to sample;

        Useful for batch training.
        """
        self.num_points = num_points

    def __call__(self, sample) -> Any:

        pointcloud, labels = sample
        if pointcloud.ndim == 3: # batched point clouds
           point_dim = 1
        else:
            point_dim = 0
        if pointcloud.shape[point_dim] < self.num_points:
            # duplicate the points until the number of points is equal to the number of points to sample
            random_indices = torch.randint(0, pointcloud.shape[point_dim] - 1, size=(self.num_points - pointcloud.shape[point_dim],))

            if pointcloud.ndim == 3:    
                pointcloud = torch.cat([pointcloud, pointcloud[:, random_indices]], dim=point_dim)
                labels = torch.cat([labels, labels[:, random_indices]], dim=point_dim)
            else:
                pointcloud = torch.cat([pointcloud, pointcloud[random_indices]], dim=point_dim)
                labels = torch.cat([labels, labels[random_indices]], dim=point_dim)

        return pointcloud, labels
        


class Remove_Noise_DBSCAN:

    def __init__(self, eps=0.1, min_points=150) -> None:
        """
        Best parameters across samples: (tensor(0.1000), tensor(150.)) with mean score: 0.91946
        Best parameters across samples: (tensor(0.1000), tensor(150.)) with mean noise removal density: 0.95495

        These parameters assume that the points cloud is normalized to [0, 1]

        Parameters
        ----------

        `eps` - float:
            The maximum distance between two samples for one to be considered as in the neighborhood of the other.

        `min_points` - int:
            The number of samples in a neighborhood for a point to be considered as a core point.
        """
         
        self.eps = eps
        self.min_points = min_points

    def __call__(self, sample) -> Tuple[torch.Tensor, torch.Tensor]:

        pointcloud, labels = sample

        if pointcloud.ndim == 3: # batched point clouds (B, P, F) ; (B, P)
            data = torch.cat([pointcloud, labels], dim=-1)
            for i in range(pointcloud.shape[0]):
                data[i] = self.remove_noise(data[i])

        else: # single point cloud (P, F) ; (P,)
            data = torch.cat([pointcloud, labels.unsqueeze(-1)], dim=-1)
            data = self.remove_noise(data)

        return data[..., :-1], data[..., -1]
    

    def remove_noise(self, data:torch.Tensor) -> torch.Tensor:

        data = eda.remove_noise(data.cpu().numpy(), self.eps, self.min_points)
        return torch.from_numpy(data)
            


class Random_Point_Sampling:

    def __init__(self, num_points: Union[int, torch.Tensor]) -> None:
        self.num_points = num_points
        self.repeat = Repeat_Points(num_points)

    def __call__(self, sample) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly sample `num_points` from the point cloud
        """

        if isinstance(sample, tuple):
            pointcloud, labels = sample
        else:
            pointcloud, labels = sample[:, :, :-1], sample[:, :, -1]

        if pointcloud.shape[1] < self.num_points:
            pointcloud, labels = self.repeat((pointcloud, labels))
        
        else:
            random_indices = torch.randperm(pointcloud.shape[1])[:self.num_points]
            pointcloud = pointcloud[:, random_indices]
            labels = labels[:, random_indices]


        return pointcloud, labels
    

class Inverse_Density_Sampling:
    """
    Inverse Density Sampling:
    1. calcule the neighbors of each 3D point within a ball of radius `ball_radius`
    2. order the point indices by the number of neighbors
    3. the `num_points` points with the least number of neighbors are sampled
    """

    def __init__(self, num_points, ball_radius) -> None:
        self.num_points = num_points
        self.ball_radius = ball_radius

    def __call__(self, sample) -> Tuple[torch.Tensor, torch.Tensor]:

        if isinstance(sample, tuple):
            pointcloud, labels = sample
        else:
            pointcloud, labels = sample[..., :-1], sample[..., -1] # preprocessed sample

        idis_pointcloud = torch.empty((pointcloud.shape[0], self.num_points, pointcloud.shape[2]), device=pointcloud.device)
        idis_labels = torch.empty((pointcloud.shape[0], self.num_points), dtype=torch.long, device=pointcloud.device)

        if pointcloud.ndim == 3: # batched point clouds
            for i in range(pointcloud.shape[0]):
                knn_indices = self.inverse_density_sampling(pointcloud[i], self.num_points, self.ball_radius)
                idis_pointcloud[i] = pointcloud[i, knn_indices]
                idis_labels[i] = labels[i, knn_indices]
        else:
            # print(f"pointcloud shape: {pointcloud.shape}, labels shape: {labels.shape}")
            knn_indices = self.inverse_density_sampling(pointcloud, self.num_points, self.ball_radius)
            idis_pointcloud = pointcloud[:, knn_indices]
            idis_labels = labels[:, knn_indices]

        # print(f"idis_pointcloud shape: {idis_pointcloud.shape}, idis_labels shape: {idis_labels.shape}")

        return idis_pointcloud.squeeze(), idis_labels.squeeze()
    
    def inverse_density_sampling(self, pointcloud:torch.Tensor, num_points:int, ball_radius:float) -> torch.Tensor:
        from torch_cluster import radius

        pointcloud = pointcloud.squeeze() # shape = (B, P, 3) -> (P, 3)
        # print(f"pointcloud shape: {pointcloud.shape}")

        indices = radius(pointcloud, pointcloud, r=ball_radius, max_num_neighbors=pointcloud.shape[0]) # shape = (2, P^2)

        #print(f"indices shape: {indices.shape}")
        #print(f"indices: {indices}")
        
        # count the number of neighbors for each point
        num_neighbors = torch.bincount(indices[0], minlength=pointcloud.shape[0]) # shape = (P,)

        #print(f"num_neighbors shape: {num_neighbors.shape}; \nnum_neighbors: {num_neighbors}")

        # select the `num_points` points with the least number of neighbors
        knn_indices = torch.argsort(num_neighbors, dim=-1)[:num_points]

        #print(f"knn_indices shape: {knn_indices.shape}; \nknn_indices: {knn_indices}")

        return knn_indices

        

class Farthest_Point_Sampling:

    def __init__(self, num_points: Union[int, torch.Tensor], fps_labels=True) -> None:
        self.num_points = num_points # if tensor, then it is the batch size and corresponds to dim 0 of the input tensor
        self.fps_labels = fps_labels # if True, then the labels are also sampled with the point cloud
        self.repeat = Repeat_Points(num_points)


    def __call__(self, sample) -> Tuple[torch.Tensor, torch.Tensor]:
        from torch_cluster import fps

        if self.fps_labels:
            if isinstance(sample, tuple): 
                pointcloud, labels = sample 
            else:
                pointcloud, labels = sample[..., :-1], sample[..., -1]
        else:
            pointcloud, target = sample
            labels = torch.zeros_like(target)

        # if the number of points in the point cloud is less than the number of points to sample
        pointcloud, labels = self.repeat((pointcloud, labels))

        fps_indices = fps(pointcloud, batch=None, ratio=self.num_points/pointcloud.shape[0], random_start=True)

        pointcloud = pointcloud[fps_indices] # shape = (N, 3 + F + 1)

        if pointcloud.shape[0] < self.num_points: # if the number of points in the point cloud is less than the number of points to sample
            pointcloud = torch.cat([pointcloud, pointcloud[torch.randint(0, pointcloud.shape[0] - 1, size=(self.num_points - pointcloud.shape[0],))]], dim=0)
        elif pointcloud.shape[0] > self.num_points:
            pointcloud = pointcloud[:self.num_points] # shape = (N, 3 + F)

        if self.fps_labels:
            return pointcloud, labels[fps_indices]
        else:
            return pointcloud, target
    
    # code stolen from pytorch3d cuz their library does not install
    def sample_farthest_points_naive(self,
        points: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        K: Union[int, list, torch.Tensor] = 50,
        random_start_point: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Iterative farthest point sampling algorithm [1] to subsample a set of
        K points from a given pointcloud. At each iteration, a point is selected
        which has the largest nearest neighbor distance to any of the
        already selected points.

        Farthest point sampling provides more uniform coverage of the input
        point cloud compared to uniform random sampling.

        [1] Charles R. Qi et al, "PointNet++: Deep Hierarchical Feature Learning
            on Point Sets in a Metric Space", NeurIPS 2017.

        Args:
            points: (N, P, D) array containing the batch of pointclouds
            lengths: (N,) number of points in each pointcloud (to support heterogeneous
                batches of pointclouds)
            K: samples required in each sampled point cloud (this is typically << P). If
                K is an int then the same number of samples are selected for each
                pointcloud in the batch. If K is a tensor is should be length (N,)
                giving the number of samples to select for each element in the batch
            random_start_point: bool, if True, a random point is selected as the starting
                point for iterative sampling.

        Returns:
            selected_points: (N, K, D), array of selected values from points. If the input
                K is a tensor, then the shape will be (N, max(K), D), and padded with
                0.0 for batch elements where k_i < max(K).
            selected_indices: (N, K) array of selected indices. If the input
                K is a tensor, then the shape will be (N, max(K), D), and padded with
                -1 for batch elements where k_i < max(K).
        """
        N, P, D = points.shape
        device = points.device

        # Validate inputs
        if lengths is None:
            lengths = torch.full((N,), P, dtype=torch.int64, device=device)
        else:
            if lengths.shape != (N,):
                raise ValueError("points and lengths must have same batch dimension.")
            if lengths.max() > P:
                raise ValueError("Invalid lengths.")

        # TODO: support providing K as a ratio of the total number of points instead of as an int
        if isinstance(K, int):
            K = torch.full((N,), K, dtype=torch.int64, device=device)
        elif isinstance(K, list):
            K = torch.tensor(K, dtype=torch.int64, device=device)

        if K.shape[0] != N:
            raise ValueError("K and points must have the same batch dimension")

        # Find max value of K
        max_K = torch.max(K)

        # List of selected indices from each batch element
        all_sampled_indices = []

        for n in range(N):
            # Initialize an array for the sampled indices, shape: (max_K,)
            sample_idx_batch = torch.full(
                # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
                #  typing.Tuple[int, ...]]` but got `Tuple[Tensor]`.
                (max_K,),
                fill_value=-1,
                dtype=torch.int64,
                device=device,
            )

            # Initialize closest distances to inf, shape: (P,)
            # This will be updated at each iteration to track the closest distance of the
            # remaining points to any of the selected points
            closest_dists = points.new_full(
                # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
                #  typing.Tuple[int, ...]]` but got `Tuple[Tensor]`.
                (lengths[n],),
                float("inf"),
                dtype=torch.float32,
                device = device,
            )

            # Select a random point index and save it as the starting point
            selected_idx = torch.randint(0, lengths[n] - 1, device=device) if random_start_point else 0
            sample_idx_batch[0] = selected_idx

            # If the pointcloud has fewer than K points then only iterate over the min
            # pyre-fixme[6]: For 1st param expected `SupportsRichComparisonT` but got
            #  `Tensor`.
            # pyre-fixme[6]: For 2nd param expected `SupportsRichComparisonT` but got
            #  `Tensor`.
            k_n = min(lengths[n], K[n])

            # Iteratively select points for a maximum of k_n
            for i in range(1, k_n):
                # Find the distance between the last selected point
                # and all the other points. If a point has already been selected
                # it's distance will be 0.0 so it will not be selected again as the max.
                dist = points[n, selected_idx, :] - points[n, : lengths[n], :]
                # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
                #  `int`.
                dist_to_last_selected = (dist**2).sum(-1)  # (P - i)

                # If closer than currently saved distance to one of the selected
                # points, then updated closest_dists
                closest_dists = torch.min(dist_to_last_selected, closest_dists)  # (P - i)

                # The aim is to pick the point that has the largest
                # nearest neighbour distance to any of the already selected points
                selected_idx = torch.argmax(closest_dists)
                sample_idx_batch[i] = selected_idx

            # Add the list of points for this batch to the final list
            all_sampled_indices.append(sample_idx_batch)

        all_sampled_indices = torch.stack(all_sampled_indices, dim=0)

        # Gather the points
        all_sampled_points = self._masked_gather(points, all_sampled_indices)

        # Return (N, max_K, D) subsampled points and indices
        return all_sampled_points, all_sampled_indices
    
    def _masked_gather(self, points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        Helper function for torch.gather to collect the points at
        the given indices in idx where some of the indices might be -1 to
        indicate padding. These indices are first replaced with 0.
        Then the points are gathered after which the padded values
        are set to 0.0.

        Args:
            points: (N, P, D) float32 tensor of points
            idx: (N, K) or (N, P, K) long tensor of indices into points, where
                some indices are -1 to indicate padding

        Returns:
            selected_points: (N, K, D) float32 tensor of points
                at the given indices
        """

        if len(idx) != len(points):
            raise ValueError("points and idx must have the same batch dimension")

        N, P, D = points.shape

        if idx.ndim == 3:
            # Case: KNN, Ball Query where idx is of shape (N, P', K)
            # where P' is not necessarily the same as P as the
            # points may be gathered from a different pointcloud.
            K = idx.shape[2]
            # Match dimensions for points and indices
            idx_expanded = idx[..., None].expand(-1, -1, -1, D)
            points = points[:, :, None, :].expand(-1, -1, K, -1)
        elif idx.ndim == 2:
            # Farthest point sampling where idx is of shape (N, K)
            idx_expanded = idx[..., None].expand(-1, -1, D)
        else:
            raise ValueError("idx format is not supported %s" % repr(idx.shape))

        idx_expanded_mask = idx_expanded.eq(-1)
        idx_expanded = idx_expanded.clone()
        # Replace -1 values with 0 for gather
        idx_expanded[idx_expanded_mask] = 0
        # Gather points
        selected_points = points.gather(dim=1, index=idx_expanded)
        # Replace padded values
        selected_points[idx_expanded_mask] = 0.0
        return selected_points

        
class Normalize_PCD:

    def __init__(self, range=[0,1]) -> None:
        
        assert len(range) == 2
        assert range[0] < range[1]

        self.range = range



    def __call__(self, sample) -> torch.Tensor:
        """
        Normalize the point cloud to have zero mean and unit variance.
        """

        pointcloud, labels = sample

        pointcloud = self.normalize(pointcloud)

        return pointcloud, labels
    

    def normalize(self, pointcloud:torch.Tensor) -> torch.Tensor:
        """
        normalize = (x - min(x)) / (max(x) - min(x))
        now x \in pointcloud is such that x \in [0, 1] (i.e., range)
        """

        point_dim = 1 if pointcloud.dim() == 3 else 0

        xyz = pointcloud[..., :3]
            
        min_x = xyz.min(dim=point_dim, keepdim=True).values
        max_x = xyz.max(dim=point_dim, keepdim=True).values
        
        xyz = (xyz - min_x) / (max_x - min_x)

        # put pointcloud in range
        xyz = xyz * (self.range[1] - self.range[0]) + self.range[0]

        pointcloud[..., :3] = xyz

        return pointcloud

    def standardize(self, pointcloud:torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------

        `pointcloud` - torch.Tensor with shape ((B), P, 3)
            Point cloud to be normalized; Batch dim is optional
        """

        pointcloud = pointcloud.float()

        if pointcloud.dim() == 3: # batched point clouds
            centroid = pointcloud.mean(dim=1, keepdim=True)
            pointcloud = pointcloud - centroid
            max_dist:torch.Tensor = torch.sqrt((pointcloud ** 2).sum(dim=-1)).max(dim=1) # shape = (batch_size,)
            pointcloud = pointcloud / max_dist.values[:, None, None]

        else: # single point cloud
            centroid = pointcloud.mean(dim=0)
            pointcloud = pointcloud - centroid 
            max_dist = torch.sqrt((pointcloud ** 2).sum(dim=-1)).max()
            pointcloud = pointcloud / max_dist

        return pointcloud



class SMOTE_3D_Upsampling:
    
        def __init__(self, sampling_strategy=0.8, k=5, num_points_resampled=11024) -> None:
            self.sampling_strategy = sampling_strategy
            self.k = k
            self.num_points_resampled = num_points_resampled
    
        def __call__(self, sample) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Synthetic Minority Over-sampling Technique for 3D point clouds
            """
            from utils.SMOTE_3D import SMOTE3D
    
            if isinstance(sample, tuple):
                pointcloud, labels = sample
            else:
                pointcloud, labels = sample[..., :-1], sample[..., -1] # shape = (B, P, 3), (B, P)
    
            smote = SMOTE3D(k_neighbors=self.k, sampling_strategy=self.sampling_strategy, num_points_resampled=self.num_points_resampled)

            if pointcloud.dim() < 3: # not batched point clouds
                pointcloud, labels = smote.fit_resample_batch(pointcloud[None], labels[None])
            else:
                pointcloud, labels = smote.fit_resample_batch(pointcloud, labels)
                
            # the result is two lists of tensors, we need to convert them to tensors
            if len(pointcloud) > 1:
                raise ValueError("The SMOTE_3D output cannot be stacked, provide a single tensor")
            
            pointcloud, labels = pointcloud[0], labels[0] # shape = (P, 3), (P,)

            # pointcloud = pointcloud[None] # encapsulate the batch dimension

            return pointcloud, labels
        



    