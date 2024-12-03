import torch
from torch.utils.data import Dataset
import laspy as lp
import os
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
import utils.pointcloud_processing as eda

def convert_las_to_laz(las_file_paths:list[str]):
    import subprocess
    new_file_paths = []
    for las_file_path in las_file_paths:
        laz_file_path = las_file_path.replace(".las", ".laz")
        
        # Run the LASzip command to convert .las to .laz
        subprocess.run(["laszip", "-i", las_file_path, "-o", laz_file_path], check=True)
        
        # Check if the .laz file was created successfully
        if os.path.exists(laz_file_path):
            # print(f"{laz_file_path} converted succesfully!")
            new_file_paths.append(laz_file_path)
            os.remove(las_file_path)
            # print(f"Removed {las_file_path}")
        else:
            print(f"Failed to convert {las_file_path}")
            input("Continue?")

    return new_file_paths


def spatial_chunk_iterator(las: lp.LasReader, chunk_size: int, bins: int = 5):
    chunk_per_bin = chunk_size // bins

    for points in las.chunk_iterator(chunk_size):
        x_coords, y_coords, z_coords = points.x, points.y, points.z
        
        # Calculate the continuity of points along X and Y:
        # sort by coord; calc consecutive diff; more contiguous axes will have less var in consecutive diff;
        x_var = np.var(np.diff(np.sort(x_coords)))
        y_var = np.var(np.diff(np.sort(y_coords)))

        # Choose the axis with the lower variance => more continuity
        if x_var < y_var:
            sorted_indices = np.lexsort((z_coords, y_coords, x_coords))
        else:
            sorted_indices = np.lexsort((z_coords, x_coords, y_coords))
        sorted_points = points[sorted_indices]

        for i in range(0, len(sorted_points), chunk_per_bin):
            offset = min(i + chunk_per_bin, len(sorted_points))
            yield sorted_points[i:offset] # yileds las objects


class Labelec_Dataset(Dataset):
    """
    Labelec Dataset class for PyTorch

    Parameters
    ----------

    `las_data_dir` - str :
        directory where the dataset is stored in LAS format

    `split` - str :
        split of the dataset to use. Options: 'fit', 'test'

    `save_chunks` - bool :
        whether to save the chunks of the las files; if False, the chunks are assumed to be saved already

    `chunk_size` - int :
        size of the chunks to process

    `bins` - int :
        number of bins to divide the chunk into; chunk_size // bins is the size of sample

    `transform` - (None, torch.Transform) :
        transformation to apply to the point clouds

    `load_into_memory` - bool :
        whether to load the entire dataset into memory
    """
    def __init__(self,
                las_data_dir,
                split = 'fit',
                save_chunks = False,
                include_rgb = True,
                chunk_size = 10_000_000,
                bins = 5,
                transform = None,
                load_into_memory = False,
            ) -> None:
        super().__init__()

        las_dir = os.path.join(las_data_dir, split)
        self.chunk_dir = os.path.join(las_dir, 'chunks/')
        self.split = split
        self.transform = transform
        self.include_rgb = include_rgb

        if save_chunks:
            assert isinstance(bins, (float, int)) and isinstance(chunk_size, (float, int)) and bins <= chunk_size

            las_file_paths = np.array([
                    os.path.join(las_dir, file_name) for file_name in os.listdir(las_dir) if file_name.endswith('.las') or file_name.endswith('.laz')
                ])
            
            self.chunk_file_paths = []
            
            for las_file in las_file_paths:
                chunk_file_paths = self._save_chunks(las_file, chunk_size, bins, to_laz=True)
                self.chunk_file_paths += chunk_file_paths
        else:
            self.chunk_file_paths = [
                os.path.join(self.chunk_dir, file_name) for file_name in os.listdir(self.chunk_dir)
            ]

        self.chunk_file_paths = np.array(self.chunk_file_paths)

        self.load_into_memory = False
        if load_into_memory:
            self._load_data_into_memory()
            self.load_into_memory = True

    def __len__(self):
        return len(self.chunk_file_paths)

    def __str__(self) -> str:
        return f"Labelec {self.split} Dataset in {self.chunk_dir} with {len(self)} samples"
    
    def _load_data_into_memory(self):
        self.data = []
        for i in tqdm(range(len(self)), desc="Loading data into memory..."):
            self.data.append(self.__getitem__(i))

    def __getitem__(self, index):
        # print(f"Loading chunk {index}...")
        
        if self.load_into_memory:
            return self.data[index]

        # upload las file
        chunk_las_file = self.chunk_file_paths[index]
        with lp.open(chunk_las_file) as chunk_las:
            chunk_las = chunk_las.read()
            feats = ['classification']
            if self.include_rgb:
                feats.append('rgb')
            chunk = eda.las_to_numpy(chunk_las, include_feats=feats) #xyz;rgb;label

        chunk = torch.from_numpy(chunk)

        sample = (chunk[:, :-1], chunk[:, -1])

        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def _save_chunks(self, las_file_path: str, chunk_size: int, bins: int = 5, to_laz=True) -> list[str]:
        file_paths = []
        file_counter = 0

        with lp.open(las_file_path, mode='r') as las_open:
            # header for the chunk las files
            header = lp.LasHeader(point_format=las_open.header.point_format, version=las_open.header.version)

            for _, las_chunk in tqdm(enumerate(spatial_chunk_iterator(las_open, chunk_size, bins)), desc=f"Saving Chunks of {las_file_path}..."):
                
                # if file_counter < len(list(os.listdir(self.chunk_dir))): # skip already saved chunks
                #     file_counter += 1
                #     continue

                file_name = f"chunk_{file_counter}.laz" if to_laz else f"chunk_{file_counter}.las"
                file_path = os.path.join(self.chunk_dir, file_name)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                with lp.open(file_path, header=header, mode="w") as las_write:
                    buffer_chunk = lp.PackedPointRecord.zeros(point_count=len(las_chunk), point_format=header.point_format)
                    buffer_chunk.copy_fields_from(las_chunk)
                    las_write.write_points(buffer_chunk)

                file_counter += 1
                file_paths.append(file_path)

        return file_paths
    


class Labelec_Preprocessed(Dataset):
    """
    Labelec Preprocessed Dataset class for PyTorch;

    The preprocesssed data follows the following transformations:
    - Normalize_PCD([0, 1]) : normalize the point cloud to [0, 1]
    - Farthest_Point_Sampling(10_000) : sample fps_points from the point cloud with a total of 10K points
    - To(torch.float32) : cast the point cloud to float32

    This results in a datasets with similar structure to the original Labelec dataset, but with the preprocessed data.

    Parameters
    ----------
    `data_dir` - str :
        directory where the dataset is stored in .pt format

    `split` - str :
        split of the dataset to use. Options: 'fit', 'test'

    `transform` - (None, torch.Transform) :
        transformation to apply to the point clouds

    `load_into_memory` - bool :
        whether to load the entire dataset into memory
    """
    def __init__(self,
                data_dir,
                split = 'fit',
                transform = None,
                load_into_memory = False,
            ) -> None:
        super().__init__()

        self.split = split
        self.transform = transform

        self.data_dir = os.path.join(data_dir, split)
        self.data_file_paths = [
            os.path.join(self.data_dir, file_name) for file_name in os.listdir(self.data_dir)
        ]

        self.load_into_memory = False
        if load_into_memory:
            self._load_data_into_memory()
            self.load_into_memory = True

    def __len__(self):
        return len(self.data_file_paths)

    def __str__(self) -> str:
        return f"Labelec {self.split} Dataset in {self.data_dir} with {len(self)} samples"
    
    def _load_data_into_memory(self):
        self.data = []
        for i in tqdm(range(len(self)), desc="Loading data into memory..."):
            self.data.append(self.__getitem__(i))

    def __getitem__(self, index):
        # print(f"Loading chunk {index}...")
        
        if self.load_into_memory:
            return self.data[index]

        # upload las file
        data_file_path = self.data_file_paths[index]
        sample = torch.load(data_file_path)

        if self.transform:
            sample = self.transform(sample)
        
        return sample

if __name__ == '__main__':
    from utils import constants as consts
    import core.datasets.torch_transforms as tt
    from torchvision.transforms import Compose
    from tqdm import tqdm

    transform = Compose([
        tt.Merge_Label({2:3, 1:3}), # merge low vegetation and ground with medium vegetation
    ])

    LABELEC_DIR = consts.LABELEC_RGB_DIR

    split = 'test'

    labelec = Labelec_Preprocessed(
        data_dir=consts.LABELEC_RGB_PREPROCESSED,
        split=split,
        transform=transform,
        load_into_memory=False
    )

    class_freqs = torch.zeros(6)
    # # CLASS DENSITY: tensor([0.0541, 0.0006, 0.3098, 0.6208, 0.0061, 0.0085])
    for i in tqdm(range(len(labelec)), desc="Calculating class frequencies..."):
        _, y = labelec[i]
        class_freqs += torch.bincount(y.long(), minlength=6)

    print(class_freqs / torch.sum(class_freqs))

    # for i in range(len(labelec)):
    #     x, y = labelec[i]
    #     print(x.shape, y.shape)

    #     print(torch.unique(y))
    #     print(torch.min(x, dim=0).values, torch.max(x, dim=0).values)
    #     print(torch.mean(x, dim=0), torch.std(x, dim=0))


    #     eda.plot_pointcloud(x.numpy()[:, :3], y.numpy(), window_name="Labelec Preprocessed", use_preset_colors=True)

    input("Continue?")

    transform = Compose([
        tt.EDP_Labels(),
        # tt.Merge_Label({eda.LOW_VEGETATION: eda.MEDIUM_VEGETAION}),
        # tt.Repeat_Points(100_000),
        tt.Farthest_Point_Sampling(10_000),
        tt.Normalize_PCD([0, 1]),
        tt.To(torch.float32),
    ])

    split = 'test'

    labelec = Labelec_Dataset(
        las_data_dir=LABELEC_DIR,
        split=split,
        save_chunks=True,
        chunk_size=20_000_000,
        bins=200,
        transform=transform,
        load_into_memory=False
    )

    # save preprocessed data
    save_path = os.path.join(LABELEC_DIR, f"Preprocessed/{split}")
    os.makedirs(save_path, exist_ok=True)

    for i in tqdm(range(len(labelec))):
        x, y = labelec[i]
        print(x.shape, y.shape)
        file_id = labelec.chunk_file_paths[i].split('/')[-1].replace('.laz', '')
        save_file_path = os.path.join(save_path, f"sample_{file_id}.pt")
        torch.save((x, y), save_file_path)
        # if not os.path.exists(save_file_path):
        #     torch.save((x, y), save_file_path)
        # else:
        #     print(f"File {save_file_path} already exists!")

    # for i in range(len(labelec)):
    #     x, y = labelec[i]
    #     print(x.shape, y.shape)

    #     if torch.isnan(x).any():
    #         ValueError("NANs in x")
    #     if torch.isnan(y).any():
    #         ValueError("NANs in y")

    #     print(torch.unique(y))
    #     print(torch.min(x, dim=0).values, torch.max(x, dim=0).values)
    #     print(torch.mean(x, dim=0), torch.std(x, dim=0))


    # rgb = x[:, 3:]
    # print(x[:10, :])
    # print(torch.unique(y))
