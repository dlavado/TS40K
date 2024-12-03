
import math
import random
import shutil
from time import sleep
from typing import List, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import numpy as np
import laspy as lp
import gc
import torch.nn.functional as F
import torch

import sys

from tqdm import tqdm
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
import utils.pointcloud_processing as eda
from core.datasets.torch_transforms import Farthest_Point_Sampling, Normalize_PCD, To, Voxelization, Inverse_Density_Sampling, SMOTE_3D_Upsampling, Add_Normal_Vector, Remove_Noise_DBSCAN
import os

def list_dir(directory):
    return [entry.name for entry in os.scandir(directory)]


def edp_labels(labels:Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        #cast each label to its corresponding EDP label
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        
        labels = torch.tensor([eda.DICT_NEW_LABELS[label.item()] if label.item() >= 0 else label.item() for label in labels.squeeze()]).reshape(labels.shape)
        return labels


def build_data_samples(data_dirs:List[str], 
                       save_dir=os.getcwd(), 
                       sem_labels=True, 
                       fps=None, 
                       sample_types='all', 
                       objects_detect='all', 
                       data_split:dict={"fit": 0.7, "test": 0.3}):
    """

    Builds a dataset of voxelized point clouds in npy format according to 
    the las files in data_dirs and saves it in `save_dir`.

    The dataset will have the following structure:
    /save_dir
    | /tower_radius
    |     | - /fit
    |     | - /test
    | /2_towers
    |     | - /fit
    |     | - /test
    | /no_tower
    |     | - /fit
    |     | - /test

    In each 

    Parameters
    ----------

    `data_dirs` - str list : 
        list of directories with las files to be converted

    `save_dir` - str:
        directory where to save the npy files.

    `sem_labels` - bool:
        if True, semantic labels are casted to the new EDP labels

    `fps` - int:
        number of points to sample with Farthest Point Sampling
        if None, FPS is not applied

    `sample_types` - str list:
        list of the types of samples to be saved
        sample_types \in ['tower_radius', '2_towers', 'no_tower']
        or sample_types == 'all' for all types


    `data_split` - dict {"str": float} or int:
        if data_split is dict:
            split of the dataset into sub-folders; keys correspond to the folders and the values to the sample split
        elif data_split is int:
            data_split == 0 for no dataset split 
    """

    print(f"\n\nBuilding dataset in {save_dir}...")

    if sample_types is None or sample_types == 'all':
        sample_types = ['tower_radius', '2_towers', 'no_tower']

    if objects_detect == 'all':
        objects_detect = eda.DICT_OBJ_LABELS_NAMES.values()

    # Build Directories
    # /save_dir
    # | /tower_radius
    # |     | - /fit
    # |     | - /test
    # | /2_towers
    # |     | - /fit
    # |     | - /test
    # | /no_tower
    # |     | - /fit
    # |     | - /test
    for type in sample_types:
        cwd = os.path.join(save_dir, type)
        for folder in ['fit', 'test']:
            if not os.path.exists(os.path.join(cwd, folder)):
                os.makedirs(os.path.join(cwd, folder))

    # to resume processing from the last read .las file
    read_files = []
    pik_name = 'read_files.pickle'
    pik_path = os.path.join(save_dir, pik_name)
    if os.path.exists(pik_path):
        read_files = eda.load_pickle(pik_path)
 
    for cwd in data_dirs:
        print(f"\n\n\nReading files in {cwd}...")

        for las_file in os.listdir(cwd):
            filename = os.path.join(cwd, las_file)

            if filename in read_files:
                print(f"File {filename} already read...\n\n")
                continue

            if ".las" in filename:
                try:
                    las = lp.read(filename)
                except Exception as e:
                    print(f"Problem occurred while reading {filename}\n\n")
                    continue
            else:
                print(f"File {filename} is not a .las file...\n\n")
                continue

            print(f"\n\n\nReading...{filename}")

            xyz, classes = eda.las_to_numpy(las)


            t_samples = []
            tt_samples = [] 
            f_samples = []
            if np.any(classes == eda.POWER_LINE_SUPPORT_TOWER):
                if 'tower_radius' in sample_types:
                    t_samples = eda.crop_tower_samples(xyz, classes, radius=50)
                if '2_towers' in sample_types:
                    tt_samples = eda.crop_two_towers_samples(xyz, classes)    

            if 'no_tower' in sample_types:
                f_samples = eda.crop_ground_samples(xyz, classes)     

            xyz, classes = None, None
            del xyz, classes
            gc.collect()

            if len(t_samples) == 0 and len(tt_samples) == 0 and len(f_samples) == 0:
                print(f"No samples in {filename}\n\n")
                continue

            if len(sample_types) < 3:
                sample_types = ['tower_radius', '2_towers', 'no_tower'] # to avoid errors, some lists will be empty so no samples will be saved
                
            
            for sample_type, samples in zip(sample_types, [t_samples, tt_samples, f_samples]):
                fit_path = os.path.join(save_dir, sample_type, 'fit/')
                counter = len(os.listdir(fit_path))
                print(f"\n\nNumber of samples in {sample_type}: {len(samples)}")

                if fps is not None:
                    fps_sampler = Farthest_Point_Sampling(fps)

                for sample in samples:
                    
                    sample = torch.from_numpy(sample)
                    if fps is not None:
                        input, labels = fps_sampler(sample)
                    else:
                        input, labels = sample[:, :-1], sample[:, -1]

                    if sem_labels:
                        labels = edp_labels(labels)

                    uq_classes = torch.unique(labels) 
                    # quality check of the point cloud
                    if input.shape[0] < 500:
                        print(f"Sample has less than 500 points...\nSkipping...")
                        continue
                    elif torch.sum(uq_classes > 0) < 2:
                        print(f"Sample has less than 2 semantic classes...\nSkipping...")
                        continue
                    else:
                        print(f"Sample has {input.shape[0]} points and {uq_classes} classes")

                    # get bounding boxes
                    obj_boxes:list[dict] = []
                    # for obj_name in objects_detect:
                    #     print(f"\nGetting bounding boxes for {obj_name}...")
                    #     if sample_type == 'no_tower' and obj_name == 'Power line support tower':
                    #         continue
                    #     obj_label = list(eda.DICT_OBJ_LABELS_NAMES.keys())[list(eda.DICT_OBJ_LABELS_NAMES.values()).index(obj_name)]
                    #     obj_boxes += eda.extract_bounding_boxes(input.numpy(), labels.numpy(), obj_label, eps=3, min_samples=100)

                    sample_dict = {
                        'type' :            sample_type, # tower_radius, 2_towers, no_tower
                        'input_pcd' :       input, # torch.tensor  with shape (N, 3)
                        'semantic_labels' : labels, # torch.tensor with shape (N,)
                        'obj_boxes':        obj_boxes # list of dicts with keys: ['class_label', 'position', 'dimensions', 'rotation']
                    }

                    # print the sample info
                    print("------- Sample Info -------")
                    print(f"\tSample type: {sample_type}")
                    print(f"\tSample input shape: {input.shape}")
                    print(f"\tSample semantic labels shape: {labels.shape}")
                    print(f"\tSample number of 3D bounding boxes: {len(obj_boxes)}")
                    if len(obj_boxes) > 0:
                        print(f"\tSample random box: {obj_boxes[random.randint(0, len(obj_boxes)-1)]}")
                    print("---------------------------")

                    try:  
                        sample_path = os.path.join(fit_path, f"sample_{counter}.pt")           
                        torch.save(sample_dict, sample_path)
                        print(f"Saving... {sample_path}")
                        counter += 1
                    except Exception as e:
                        print(e)
                        print(f"Problem occurred while saving {filename}\n\n")
                        continue


                    # free up as much memory as possible
                    sample_dict, sample, input, labels, uq_classes, obj_boxes, sample_path = None, None, None, None, None, None, None
                    del sample_dict, sample, input, labels, uq_classes, obj_boxes, sample_path
                    gc.collect()
                    torch.cuda.empty_cache()
                    sleep(0.1)

            del las
            del t_samples
            del tt_samples
            del f_samples
            gc.collect()
            read_files.append(filename)
            eda.save_pickle(read_files, pik_path)
          

    if data_split == 0:
        return # all files in fit_path
    
    # split data
    print(f"\n\nSplitting data in {save_dir}...")
    
    for sample_type in sample_types:

        fit_path = os.path.join(save_dir, sample_type, 'fit')
        print(f"Splitting {fit_path}...")

        samples = os.listdir(fit_path)
        random.shuffle(samples)

        assert sum(list(data_split.values())) <= 1, "data splits should not surpass 1"

        split_sum = 0
        sample_size = len(samples)
        print(f"Number of total samples: {sample_size}")

        for folder, split in data_split.items():
            if folder == 'fit':
                split_sum += split
                continue

            cwd = os.path.join(save_dir, sample_type, folder)
            if not os.path.exists(cwd):
                os.makedirs(cwd)

            samples_split = samples[int(split_sum*sample_size):math.ceil((split_sum+split)*sample_size)]
            split_sum += split
            print(f"Samples in {folder}: {len(samples_split)}")

            for sample in samples_split:
                shutil.move(os.path.join(fit_path, sample), cwd)


def build_bounding_boxes(dataset_dir:str, objects_detect:list[str]='all'):
    """
    Takes the samples built in `build_data_samples`, extracts the bounding boxes in the samples according to the objects to detect
    and adds this information to the samples.
    """

    if objects_detect == 'all':
        objects_detect = eda.DICT_OBJ_LABELS_NAMES.values()

    original_objects_detect = objects_detect

    sample_types = ['tower_radius', '2_towers', 'no_tower']
    random.shuffle(sample_types)


    # process each sample at a time to avoid memory issues
    for sample_type in sample_types:
        sample_type_path = os.path.join(dataset_dir, sample_type)
        if sample_type == 'tower_radius':
            continue # already processed
        for split in os.listdir(sample_type_path):
            skip_2 = sample_type == '2_towers' and split == 'fit'
            skip_no_tower = sample_type == 'no_tower' and split == 'fit'
            split_path = os.path.join(sample_type_path, split)
            for sample in tqdm(os.listdir(split_path), desc=f"Processing {sample_type} {split} samples..."):
                sample_path = os.path.join(split_path, sample)
                i = int(sample.split('_')[-1].split('.')[0])

                if skip_2: # skip the first 2768 samples of 2_towers
                    if i == 2709:
                        skip_2 = False
                    else:
                        continue
                elif skip_no_tower: # skip the first 2768 samples of no_tower
                    if i == 13927:
                        skip_no_tower = False
                    else:
                        continue
                
                try:
                    sample_dict = torch.load(sample_path)
                except Exception as e:
                    print(f"Unreadable file: {sample_path}")
                    print(e)
                    continue
                    # exit()  

                print(f"\n\n\nProcessing... {sample_path}")      

                objects_detect = original_objects_detect
                obj_boxes:list[dict] = []

                input = sample_dict['input_pcd']
                labels = torch.squeeze(sample_dict['semantic_labels'])
                sample_dict['semantic_labels'] = labels
               
                for obj_name in objects_detect:
                    print(f"\nGetting bounding boxes for {obj_name}...")
                    if sample_type == 'no_tower' and obj_name == 'Power line support tower':
                        continue
                    obj_label = list(eda.DICT_OBJ_LABELS_NAMES.keys())[list(eda.DICT_OBJ_LABELS_NAMES.values()).index(obj_name)]
                        
                    obj_boxes += eda.extract_bounding_boxes(input.numpy(), labels.numpy(), obj_label, eps=3, min_samples=300)
                
                print(f"\n\n\nNumber of bounding boxes in sample: {len(obj_boxes)}")

                sample_dict['obj_boxes'] = obj_boxes

                # print the sample info
                if random.randint(0, 10) >= 8:
                    print("------- Sample Info -------")
                    print(f"\tSample type: {sample_type}")
                    print(f"\tSample input shape: {input.shape}")
                    print(f"\tSample semantic labels shape: {labels.shape}")
                    print(f"\tSample number of 3D bounding boxes: {len(obj_boxes)}")
                    if len(obj_boxes) > 0:
                        print(f"\tSample random box: {obj_boxes[random.randint(0, len(obj_boxes)-1)]}")
                    print("---------------------------")

                print(f"Saving... {sample_path}")
                torch.save(sample_dict, sample_path)

                # clean as much memory as possible
                sample_dict, input, labels, obj_boxes = None, None, None, None
                del sample_dict, input, labels, obj_boxes
                gc.collect()



def save_preprocessed_data(data_dir, save_dir):
    """
    Save preprocessed data in save_dir from the data in data_dir of TS40K_FULL;
    Keeps directory structure of TS40K_FULL and fit/test distribution
    
    """
    # composed = Compose([
    #                     Normalize_PCD(),
    #                     # Farthest_Point_Sampling(fps_points),
    #                     To(torch.float32),
    #                 ]) 

    composed = Compose([
                        # Voxelization(vxg_size=(64, 64, 64)),
                        # Normalize_PCD(),
                        # Farthest_Point_Sampling(fps_points, fps_labels=False),
                        Inverse_Density_Sampling(10_000, 0.05),
                        # SMOTE_3D_Upsampling(k=5, sampling_strategy=0.6),
                        Normalize_PCD(),
                        To(torch.float32),
                    ])

    # for sample_type in ['tower_radius', 'no_tower', '2_towers']:
    for sample_type in os.listdir(data_dir):

        if '.' in sample_type:
            continue # skip files

        sample_type_path = os.path.join(data_dir, sample_type)
        save_type_path = os.path.join(save_dir, sample_type)

        if not os.path.exists(save_type_path):
            os.makedirs(save_type_path)

        for split in os.listdir(sample_type_path):
            if split == 'test':
                continue
            #     transform = None
            else:
                transform = composed
            # split_path = os.path.join(sample_type_path, split)
            save_split_path = os.path.join(save_type_path, split)

            if not os.path.exists(save_split_path):
                os.makedirs(save_split_path)

            # dm = TS40K_FULL_Preprocessed(data_dir, split=split, sample_types=[sample_type], transform=transform, load_into_memory=True)
            dm = TS40K_FULL(data_dir, split=split, sample_types=[sample_type], task='sem_seg', transform=composed, load_into_memory=False)

            # get folder count
            # folder_count = 0
            folder_count = len(os.listdir(save_split_path)) # on reruns, the current sample probably killed the process so it cannot be computed
            for i in tqdm(range(folder_count, len(dm)), desc=f"Saving {sample_type} {split} samples..."):
                sample = dm[i]
                sample_path = os.path.join(save_split_path, dm.data_files[i])
                sample_path = os.path.join(save_split_path, f"sample_{i}.pt")
                torch.save(sample, sample_path)

   

def process_ts40k_for_mmlab_pcdet_framework(data_dir, save_dir, fps_points, normalize=True):
    """
    Save preprocessed data in save_dir from the data in data_dir of TS40K_FULL;
    https://github.com/open-mmlab/OpenPCDet?tab=readme-ov-file

    The format of the sample files is as follows:
        ├── custom\n
        │   │   │── ImageSets\n
        │   │   │   │── train.txt\n
        │   │   │   │── val.txt\n
        │   │   │── points\n
        │   │   │   │── 000000.npy\n
        │   │   │   │── 999999.npy\n
        │   │   │── labels\n
        │   │   │   │── 000000.txt\n
        │   │   │   │── 999999.txt\n

    The format of the .txt files for the labels is as follows:\n
    format: [x y z dx dy dz heading_angle category_name]

    1.50 1.46 0.10 5.12 1.85 4.13 1.56 Vehicle\n
    5.54 0.57 0.41 1.08 0.74 1.95 1.57 Pedestrian

    The training and validation splits are defined in the train.txt and val.txt files respectively.
    where we list the point cloud file names without the extension.\n
    e.g.\n
    000000\n
    000001\n
    000002\n
    ...
    """

    composed = Compose([
                        # Normalize_PCD(),
                        # Farthest_Point_Sampling(fps_points, fps_labels=False),
                        To(torch.float32),
                        ]) 
    
    labels_path = os.path.join(save_dir, 'labels')
    points_path = os.path.join(save_dir, 'points')
    imagesets_path = os.path.join(save_dir, 'ImageSets')
    train_file_path = os.path.join(imagesets_path, 'train.txt')
    val_file_path = os.path.join(imagesets_path, 'val.txt')

    if not os.path.exists(labels_path):
        os.makedirs(labels_path)
    if not os.path.exists(points_path):
        os.makedirs(points_path)
    if not os.path.exists(imagesets_path):
        os.makedirs(imagesets_path)

    for split in ['fit', 'test']:

        dm = TS40K_FULL(data_dir, split=split, sample_types='all', task='obj_det', transform=composed, load_into_memory=False)

        save_file = train_file_path if split == 'fit' else val_file_path

        if os.path.exists(save_file):
            with open(save_file, "r") as f:
                num_samples = sum(1 for _ in f) # get number of samples in val.txt
        else:
            num_samples = 0
            file = open(save_file, 'w')
            file.close() # create file if it does not exist
           

        for i in tqdm(range(num_samples, len(dm)), desc=f"Saving {split} samples..."):
            points, bboxes = dm[i]
            if bboxes.shape[0] > 0 and points.shape[0] >= fps_points:  # only save samples with objects and with more than fps_points
                sample_path = os.path.join(points_path, f"{i:06d}.npy")

                points = points.numpy()
                if normalize:
                    min_xyz = np.min(points, axis=0)
                    max_xyz = np.max(points, axis=0)
                    points = (points - min_xyz) / (max_xyz - min_xyz)
                    
                # the sample needs to be in the format unified normative coordinate, i.e., y, x, z for some reason, this framework is stopid
                np.save(sample_path, points)
                # write on save_file
                with open(save_file, "a") as f:
                    f.write(f"{i:06d}\n")
                # write on labelssample_dict['obj_boxes']
                label_path = os.path.join(labels_path, f"{i:06d}.txt")
                with open(label_path, "w") as f:
                    #bboxes is a (O, 8) tensor with the following format: [x, y, z, dx, dy, dz, heading_angle, label]
                    for bbox in bboxes:
                        if bbox.shape[0] < 8:
                            print(sample_path)
                            print(f"Bounding box with less than 8 elements: {bbox}")
                            # exit()
                        x, y, z, dx, dy, dz, heading_angle, label = bbox
                        # normalize the bounding box to the point cloud
                        if normalize:
                            x, y, z = (x - min_xyz[0]) / (max_xyz[0] - min_xyz[0]), (y - min_xyz[1]) / (max_xyz[1] - min_xyz[1]), (z - min_xyz[2]) / (max_xyz[2] - min_xyz[2])
                            dx, dy, dz = dx / (max_xyz[0] - min_xyz[0]), dy / (max_xyz[1] - min_xyz[1]), dz / (max_xyz[2] - min_xyz[2])
                        f.write(f"{x} {y} {z} {dx} {dy} {dz} {heading_angle} {eda.DICT_OBJ_LABELS_NAMES[int(label)].replace(' ', '-')}\n")

            # release memory
            points, bboxes = None, None
            del points, bboxes
            gc.collect()


def save_normalized_data(data_dir, save_dir):

    # for sample_type in ['tower_radius', 'no_tower', '2_towers']:
    for sample_type in os.listdir(data_dir):

        sample_type_path = os.path.join(data_dir, sample_type)
        save_type_path = os.path.join(save_dir, sample_type)

        if '.' in sample_type:
            continue  # skip files

        if not os.path.exists(save_type_path):
            os.makedirs(save_type_path)

        for split in os.listdir(sample_type_path):
            # split_path = os.path.join(sample_type_path, split)
            save_split_path = os.path.join(save_type_path, split)

            if not os.path.exists(save_split_path):
                os.makedirs(save_split_path)

            dm = TS40K_FULL(data_dir, split=split, sample_types=[sample_type], task='sem_seg', transform=None, load_into_memory=False)

            # get folder count
            # folder_count = 0
            folder_count = len(os.listdir(save_split_path))
            for i in tqdm(range(folder_count, len(dm)), desc=f"Saving {sample_type} {split} samples..."):
                sample_dict = dm._get_dict(i)
                points = sample_dict['input_pcd']
                bboxes = dm._bboxes_to_tensor(sample_dict['obj_boxes'])
                
                min_points, max_points = torch.min(points, axis=0).values, torch.max(points, axis=0).values
                points = (points - min_points) / (max_points - min_points)
                sample_dict['input_pcd'] = points

                if len(bboxes) > 0:
                    # normalize bboxes
                    bboxes[:, 0] = (bboxes[:, 0] - min_points[0]) / (max_points[0] - min_points[0]) # x
                    bboxes[:, 1] = (bboxes[:, 1] - min_points[1]) / (max_points[1] - min_points[1]) # y
                    bboxes[:, 2] = (bboxes[:, 2] - min_points[2]) / (max_points[2] - min_points[2]) # z
                    bboxes[:, 3] = bboxes[:, 3] / (max_points[0] - min_points[0]) # dx
                    bboxes[:, 4] = bboxes[:, 4] / (max_points[1] - min_points[1]) # dy
                    bboxes[:, 5] = bboxes[:, 5] / (max_points[2] - min_points[2]) # dz

                sample_dict['obj_boxes'] = bboxes

                # if torch.rand(1) > 0.9:
                #     print(f"Sample {i} has been normalized...")
                #     print(f"Sample input shape: {points.shape} range of values: {torch.min(points, axis=0).values}, {torch.max(points, axis=0).values}")
                #     print(f"semantics: {torch.unique(sample_dict['semantic_labels'])}")
                #     print(f"Sample number of 3D bounding boxes: {len(bboxes)}")
                #     if len(bboxes) > 0:
                #         print(f"Sample random box: {bboxes[random.randint(0, len(bboxes)-1)]}")

                sample_path = os.path.join(save_split_path, f"sample_{i}.pt")
                torch.save(sample_dict, sample_path)


class TS40K(Dataset):

    def __init__(self, dataset_path, split='fit', transform=None, min_points=None, load_into_memory=True) -> None:
        """
        Initializes the TS40K dataset

        Parameters
        ----------
        `dataset_path` - str:
            path to the directory with the voxelized point clouds crops in npy format

        `split` - str:
            split of the dataset to access 
            split \in [fit, test] 
            
        `transform` - (None, torch.Transform) :
            transformation to apply to the point clouds

        `min_points` - int:
            minimum number of points in the point cloud

        `load_into_memory` - bool:
            if True, loads the entire dataset into memory
        """
        super().__init__()

        self.transform = transform
        self.split = split

        self.dataset_path = os.path.join(dataset_path, split)

        self.data_files:np.ndarray = np.array([file for file in os.listdir(self.dataset_path)
                        if os.path.isfile(os.path.join(self.dataset_path, file)) and ('.npy' in file or '.pt' in file)])
        
        if min_points:
            self.data_files = np.array([file for file in self.data_files if np.load(os.path.join(self.dataset_path, file)).shape[0] >= min_points])
    
        self.load_into_memory = False
        if load_into_memory:
            self._load_data_into_memory()
            self.load_into_memory = True

    def __len__(self):
        return len(self.data_files)

    def __str__(self) -> str:
        return f"TS40K {self.split} Dataset with {len(self)} samples"
    
    def _load_data_into_memory(self):
        self.data = []
        for i in tqdm(range(len(self)), desc="Loading data into memory..."):
            self.data.append(self.__getitem__(i))

    def set_transform(self, new_transform):
        self.transform = new_transform
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # data[i]

        if self.load_into_memory:
            return self.data[idx]

        if torch.is_tensor(idx):
            idx = idx.tolist()

        npy_path = os.path.join(self.dataset_path, self.data_files[idx])

        try:
            npy = np.load(npy_path)
        except:
            print(f"Unreadable file: {npy_path}")
        
        sample = (npy[None, :, 0:-1], npy[None, :, -1]) # xyz-coord (1, N, 3); label (1, N) 


        if self.transform:
            sample = self.transform(sample)
            #print(f"Transformed sample: {sample[0].shape}, {sample[1].shape}, {sample[2].shape}")
            return sample
        
        return sample
    

class TS40K_FULL(Dataset):

    def __init__(self, dataset_path, split='fit', sample_types='all', task="sem_seg", transform=None, load_into_memory=True) -> None:
        """
        Initializes the TS40K dataset.

        There are different types of samples in the dataset:
        - tower_radius: samples with a single tower in the center
        - 2_towers: samples with two towers and a power line between them
        - no_tower: samples with no towers

        There are also two types of tasks available:
        - sem_seg: semantic segmentation
        - obj_det: object detection, which inlcudes objects of the classes: `power_line` and `tower` 

        Parameters
        ----------

        `dataset_path` - str:
            path to the directory with the TS40K dataset

        `split` - str:
            split of the dataset to access 
            split \in [fit, test]

        `sample_types` - str list:
            list of the types of samples to be used
            sample_types \in ['tower_radius', '2_towers', 'no_tower']
            or sample_types == 'all' for all types

        `task` - str:
            task to perform
            task \in ['sem_seg', 'obj_det']

        `transform` - (None, torch.Transform) :
            transformation to apply to the point clouds

        `load_into_memory` - bool:
            if True, loads the entire dataset into memory

        """
        super().__init__()

        
        if task not in ['sem_seg', 'obj_det']:
            raise ValueError(f"Task {task} not supported. Task should be one of ['sem_seg', 'obj_det']")

        if sample_types != 'all' and not isinstance(sample_types, list):
            raise ValueError(f"sample_types should be a list of strings or 'all'")

    
        self.transform = transform
        self.split = split
        self.task = task

        self.dataset_path = dataset_path

        if sample_types == 'all':
            sample_types = ['tower_radius', '2_towers', 'no_tower']

        self.data_files = []

        for type in sample_types:
            type_path = os.path.join(self.dataset_path, type, split)
            self.data_files += [os.path.join(type_path, file) for file in os.listdir(type_path)
                                 if os.path.isfile(os.path.join(type_path, file)) and ('.npy' in file or '.pt' in file)]

        self.data_files = np.array(self.data_files)
        
        # if min_points:
        #     # if min_points is not None, filter out samples with less than min_points
        #     self.data_files = np.array([file for file in self.data_files if torch.load(os.path.join(self.dataset_path, file))['input_pcd'].shape[0] >= min_points])
    
        self.load_into_memory = False
        if load_into_memory:
            self._load_data_into_memory()
            self.load_into_memory = True
            
    def __len__(self):
        return len(self.data_files)

    def __str__(self) -> str:
        return f"TS40K FULL {self.split} Dataset with {len(self)} samples"
    
    def _load_data_into_memory(self):
        self.data = []
        for i in tqdm(range(len(self)), desc="Loading data into memory..."):
            self.data.append(self.__getitem__(i))

    
    def _bboxes_to_tensor(self, bboxes:list[dict]):
        """
        Converts a list of bounding boxes to a tensor of shape (O, 8) with the following format:
        [x, y, z, dx, dy, dz, heading_angle, class_label]

        Parameters
        ----------
        `bboxes` - list[dict]:
            list of bounding boxes with the following keys:
            ['class_label', 'position', 'dimensions', 'rotation']

        Returns
        -------
        `boxes` - torch.Tensor:
            tensor with shape (O, 8) with the bounding boxes
        """
        if len(bboxes) == 0:
            return torch.tensor([])
        
        boxes = torch.zeros((len(bboxes), 8))

        for i, bbox in enumerate(bboxes):
            pos   = torch.tensor([bbox['position']['x'], bbox['position']['y'], bbox['position']['z']])
            dims  = torch.tensor([bbox['dimensions']['width'], bbox['dimensions']['height'], bbox['dimensions']['length']])
            angle = torch.tensor([bbox['rotation']])
            label = torch.tensor([bbox['class_label']])

            cat = torch.cat((pos, dims, angle, label), axis=0) # [x, y, z, dx, dy, dz, heading_angle, class_label], shape (8,)
            boxes[i] = cat 
        

        return boxes


    def _get_dict(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_path = self.data_files[idx]

        try:
            if sample_path.endswith('.pt'):
                sample_dict = torch.load(sample_path)
            elif sample_path.endswith('.npy'):
                sample_dict = np.load(sample_path, allow_pickle=True).item()
            else:
                raise ValueError(f"File {sample_path} is not a .pt or .npy file")
        
        except Exception as e:
            print(f"Unreadable file: {sample_path}")
            print(e)
            return self[random.randint(0, len(self)-1)]
        
        return sample_dict
    
    def _get_file_path(self, idx):
        return os.path.join(self.dataset_path, self.data_files[idx])

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The format of the sample files is as follows:
        sample_dict = {
            'type' :            sample_type, # tower_radius, 2_towers, no_tower
            'input_pcd' :       input, # torch.tensor  with shape (N, 3)
            'semantic_labels' : labels[None], # torch.tensor with shape (N, 1)
            'obj_boxes':        obj_boxes # list of dicts with keys: ['class_label', 'position', 'dimensions', 'rotation']
        }        
        """
        # data[i]

        if self.load_into_memory:
            return self.data[idx]

        sample_dict = self._get_dict(idx)
        
        if isinstance(sample_dict, dict):
            x = sample_dict['input_pcd']
            if self.task == "sem_seg":
                y = sample_dict['semantic_labels']
                y = torch.squeeze(y) # reshape to (N,)
                sample = (x, y) # xyz-coord (N, 3); label (N,)
            else:
                y = sample_dict['obj_boxes']
                if not isinstance(y, torch.Tensor):
                    y = self._bboxes_to_tensor(y)
                sample = (x, y) # xyz-coord (N, 3); label (O, 8)

            if self.transform:
                # send sample to gpu
                sample = self.transform(sample)
                #print(f"Transformed sample: {sample[0].shape}, {sample[1].shape}, {sample[2].shape}")
        
        else: # data was preprocessed as saved as a tuple
            x, y = sample_dict
            sample = (x.squeeze(), y.squeeze())
        
        return sample


class TS40K_FULL_Preprocessed(Dataset):
    """
    The preprocesssed data follows the following transformations:
    - Normalize_PCD() : normalize the point cloud to have mean 0 and std 1
    - Farthest_Point_Sampling(fps_points) : sample fps_points from the point cloud with a total of 10K points
    - To(torch.float32) : cast the point cloud to float32

    This results in a datasets with similar structure to the original TS40K_FULL dataset, but with the preprocessed data.

    The targets are specific to the sem_seg task, for others, different preprocessing should be applied.
    """

    def __init__(self, dataset_path:str, split='fit', sample_types='all', transform=None, load_into_memory=True, use_full_test_set=False) -> None:
        super().__init__()

        if sample_types != 'all' and not isinstance(sample_types, list):
            raise ValueError(f"sample_types should be a list of strings or 'all'")
        
        self.dataset_path = dataset_path
        self.transform = transform

        if split == 'test' and use_full_test_set:
            self.dataset_path = self.dataset_path.replace('-Preprocessed', '')
            self.ts40k_full = TS40K_FULL(self.dataset_path, split='test', sample_types=sample_types, task='sem_seg', transform=transform, load_into_memory=load_into_memory)
        else:
            self.ts40k_full = None

        if sample_types == 'all':
            sample_types = ['tower_radius', '2_towers', 'no_tower']

        self.data_files = []

        for type in sample_types:
            type_path = os.path.join(self.dataset_path, type, split)
            self.data_files += [os.path.join(type_path, file) for file in os.listdir(type_path)
                                 if os.path.isfile(os.path.join(type_path, file)) and ('.npy' in file or '.pt' in file)]

        self.data_files = np.array(self.data_files)


        self.load_into_memory = False
        if load_into_memory:
            self._load_data_into_memory()
            self.load_into_memory = True


    def _load_data_into_memory(self):
        self.data = []
        for i in tqdm(range(len(self)), desc="Loading data into memory..."):
            self.data.append(self.__getitem__(i))


    def __len__(self):
        return len(self.data_files)
    
    def _get_file_path(self, idx) -> str:
        return os.path.join(self.dataset_path, self.data_files[idx])


    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # data[i]

        if self.ts40k_full:
            return self.ts40k_full[idx]

        if self.load_into_memory:
            return self.data[idx]

        if torch.is_tensor(idx):
            idx = idx.tolist()

        pt_path = self.data_files[idx]
        
        try:
            pt = torch.load(pt_path)  # xyz-coord (1, N, 3); label (1, N)
        except:
            print(f"Unreadable file: {pt_path}")

        if self.transform:
            pt = self.transform(pt)
            #print(f"Transformed sample: {sample[0].shape}, {sample[1].shape}, {sample[2].shape}")

        return pt
       
            

def main():
    # import utils.voxelization as Vox
    # print(torch.__version__)
    # print(torch.cuda.is_available())
    # print(torch.cuda.get_device_name(0))
    
    TS40K_DIR = constants.TS40K_PATH
    # TS40K_DIR = os.path.join(constants.TOSH_PATH, "TS40K-Dataset")

    LAS_DIRS = [
        #os.path.join(TS40K_DIR, "LIDAR-2022"),
        os.path.join(TS40K_DIR, "LIDAR-2024"), 
        #os.path.join(TS40K_DIR, "Labelec_LAS")
    ]
    
    # build_data_samples(LAS_DIRS, 
    #                    save_dir = os.path.join(TS40K_DIR, "TS40K-FULL"), 
    #                    sem_labels=True, 
    #                    fps=None, 
    #                    sample_types='all',
    #                    objects_detect=['Power line support tower', 'Power lines', 'Medium Vegetation'], 
    #                    data_split = 0 #{"fit": 0.8, "test": 0.2}
    #                 )
    
    # build_bounding_boxes(os.path.join(TS40K_DIR, "TS40K-FULL"), 
    #                      objects_detect=["Power line support tower", "Power lines", 'Medium Vegetation'])

    # save_preprocessed_data(os.path.join(TS40K_DIR, "TS40K-FULL"),
    #                        os.path.join(TS40K_DIR, "TS40K-FULL-Preprocessed-IDIS"),
    #                     )

    # save_preprocessed_data(constants.TS40K_FULL_PREPROCESSED_PATH,
    #                        os.path.join(TS40K_DIR, "TS40K-FULL-Preprocessed-SMOTE"),
    #                     )

    # process_ts40k_for_mmlab_pcdet_framework(os.path.join(TS40K_DIR, "TS40K-FULL"),
    #                                         "/home/didi/VSCode/Philosophy-of-Doctors/OpenPCDet/data/ts40k/",
    #                                         fps_points=10000)    

    # save_normalized_data(os.path.join(TS40K_DIR, "TS40K-FULL"),
    #                      os.path.join(TS40K_DIR, "TS40K-FULL-Normalized")
    #                     )    
    

    # input("Press Enter to continue...")

    # ts40k = TS40K_FULL_Preprocessed(
    #     constants.TS40K_FULL_PREPROCESSED_IDIS_PATH,
    #     split='fit',
    # )

    # transform = Compose([
    #             Normalize_PCD([0, 1]),
    #             Add_Normal_Vector(),
    #         ])

    # for split in ['fit', 'test']:
    #     for sample_type in ['tower_radius', '2_towers', 'no_tower']:
    #         ts40k = TS40K_FULL(constants.TS40K_FULL_PATH, 
    #                            split=split, 
    #                            sample_types=[sample_type], 
    #                            task='sem_seg', transform=transform, load_into_memory=False)
    #         print(f"TS40K {split} {sample_type} Dataset has {len(ts40k)} samples")

    #         for i in tqdm(range(len(ts40k)), desc=f"Processing {split} {sample_type} samples..."):
    #             sample_path = ts40k._get_file_path(i)
    #             sample_dict = ts40k._get_dict(i)

    #             transformed_sample = ts40k[i][0]

    #             new_sample_path = sample_path.replace('TS40K-FULL', 'TS40K-FULL-Normals-Normalized')
                
    #             if os.path.exists(new_sample_path): # skip already processed samples
    #                 continue
    #             os.makedirs(os.path.dirname(new_sample_path), exist_ok=True)
    #             sample_dict['normal_vectors'] = transformed_sample[:, 3:]
    #             sample_dict['input_pcd'] = transformed_sample[:, :3]
    #             # print(transformed_sample[:, 3:].shape)
    #             # print(torch.min(transformed_sample[:, :3], axis=0).values)
    #             # print(torch.max(transformed_sample[:, :3], axis=0).values)
    #             # input("Press Enter to continue...")

    #             for key in sample_dict.keys():
    #                 if isinstance(sample_dict[key], torch.Tensor):
    #                     sample_dict[key] = sample_dict[key].squeeze().to(torch.float16)

    #             torch.save(sample_dict, new_sample_path)

    # for i in tqdm(range(len(ts40k)), desc="Visualizing samples..."):
    #     xyz, y = ts40k[i]
    #     print(f"Sample {i} has {xyz.shape[0]} points")
    #     if xyz.shape[0] < 10000:
    #         print(f"Sample {i} has less than 10K points: {xyz.shape[0]}")
    #         file_path = ts40k.data_files[i]
    #         # eliminate the file
    #         os.remove(file_path)

    # input("Press Enter to continue...")

   
    composed = Compose([
                        # Farthest_Point_Sampling(10000),
                        # Random_Point_Sampling(10000),
                        # Inverse_Density_Sampling(10000, 0.5),
                        Normalize_PCD(),
                        Remove_Noise_DBSCAN(),
                        # To(torch.float32),
                    ])
    
   
    # ts40k = TS40K_FULL(constants.TS40K_FULL_PATH, 
    #                    split='fit', 
    #                    sample_types=['tower_radius'], 
    #                    task='sem_seg', transform=composed, load_into_memory=False)

    ts40k = TS40K_FULL_Preprocessed(
        constants.TS40K_FULL_PREPROCESSED_PATH, 
        split='fit', 
        sample_types='all', 
        transform=composed, 
        load_into_memory=False
    )

    class_freqs = torch.zeros(6)

    for i in tqdm(range(0, len(ts40k)), desc="Computing class frequencies..."):
        _, y = ts40k[i]
        y = y.squeeze().long()
        class_freqs += torch.bincount(y, minlength=6)

    # print(class_freqs)
    # tensor([0.0382, 0.3490, 0.4521, 0.1543, 0.0036, 0.0028])
    # ALL DATASET DENSITIES: tensor([0.0167, 0.4656, 0.4590, 0.0494, 0.0022, 0.0072])
    # TOWERED SAMPLES DENSITIES: tensor([0.0255, 0.4056, 0.5244, 0.0248, 0.0073, 0.0124])
    print(class_freqs / torch.sum(class_freqs))
    
    for idx in range(len(ts40k)): 
        xyz, y = ts40k[idx]
        y = y.reshape(-1).numpy()
        xyz = xyz.squeeze().numpy()
        pynt = eda.np_to_ply(xyz)
        eda.color_pointcloud(pynt, y, use_preset_colors=True)
        eda.visualize_ply([pynt])





if __name__ == "__main__":
    from utils import constants
    from torchvision.transforms import Compose
    from core.datasets.torch_transforms import Normalize_PCD, Farthest_Point_Sampling, Random_Point_Sampling, Inverse_Density_Sampling
    
    main()

# %%
