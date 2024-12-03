

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
from core.datasets.Labelec import Labelec_Dataset as Labelec, Labelec_Preprocessed

def collate_fn(batch):
    coords_list = []
    feats_list = []
    labels = []
    batch_vector = []

    for i, (x, label) in enumerate(batch):
        # Assuming coords are in the first C_coords columns and feats in the remaining columns
        C_coords = 3  # or however many columns `coords` have

        coords = x[:, :C_coords]         # Extract coords
        feats = x[:, C_coords:]          # Extract feats

        coords_list.append(coords)       # Shape: (N_i, C_coords)
        feats_list.append(feats)         # Shape: (N_i, C_feats)
        labels.append(label)             # Shape: (N_i, 1) or (N_i,)
        batch_vector.append(torch.full((x.shape[0],), i, dtype=torch.long))

    # Concatenate each list into a single tensor
    coords = torch.cat(coords_list, dim=0)  # Shape: (B * N, C_coords)
    feats = torch.cat(feats_list, dim=0)    # Shape: (B * N, C_feats)
    labels = torch.cat(labels, dim=0)       # Shape: (B * N, 1) or (B * N,)
    batch_vector = torch.cat(batch_vector, dim=0)  # Shape: (B * N,)

    # Return a dictionary with the separate components
    return {
        'coords': coords,
        'feats': feats,
        'batch_vector': batch_vector,
        'labels': labels
    }


class LitLabelec(pl.LightningDataModule):

    def __init__(self, 
                 las_data_dir, 
                 save_chunks=False, 
                 chunk_size=10_000_000, 
                 bins=5, 
                 transform=None, 
                 test_transform=None,
                 add_rgb=True,
                 load_into_memory=False,
                 batch_size=1,
                 val_split=0.2,
                 num_workers=8   
            ):
        
        super().__init__()
        self.las_data_dir = las_data_dir
        self.transform = transform
        self.test_transform = test_transform        
        self.load_into_memory = load_into_memory
        self.add_rgb = add_rgb
        if save_chunks:
            self._build_dataset(chunk_size, bins)
        
        self.save_hyperparameters()

    def _build_dataset(self, chunk_size, bins): #Somewhat equivalent to `prepare_data` hook of LitDataModule
        Labelec(las_data_dir=self.las_data_dir, split='fit', save_chunks=True, include_rgb=self.add_rgb, chunk_size=chunk_size, bins=bins, transform=self.transform, load_into_memory=self.load_into_memory) 
        Labelec(las_data_dir=self.las_data_dir, split='test', save_chunks=True, include_rgb=self.add_rgb, chunk_size=chunk_size, bins=bins, transform=self.test_transform, load_into_memory=self.load_into_memory)           
   
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.fit_ds = Labelec(las_data_dir=self.las_data_dir, split='fit', save_chunks=False, include_rgb=self.add_rgb, transform=self.transform, load_into_memory=self.load_into_memory)
            self.train_dataset, self.val_dataset = random_split(self.fit_ds, [1 - self.hparams.val_split, self.hparams.val_split])
        
        elif stage == 'test':
            self.test_ds = Labelec(las_data_dir=self.las_data_dir, split='test', save_chunks=False, include_rgb=self.add_rgb, transform=self.test_transform, load_into_memory=self.load_into_memory)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
    

class LitLabelec_Preprocessed(pl.LightningDataModule):

    def __init__(self, 
                 data_dir, 
                 transform=None, 
                 test_transform=None,
                 load_into_memory=False,
                 batch_size=1,
                 val_split=0.2,
                 num_workers=8   
            ):
        
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.test_transform = test_transform        
        self.load_into_memory = load_into_memory
        
        self.save_hyperparameters() 

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.fit_ds = Labelec_Preprocessed(data_dir=self.data_dir, split='fit', transform=self.transform, load_into_memory=self.load_into_memory)
            self.train_dataset, self.val_dataset = random_split(self.fit_ds, [1 - self.hparams.val_split, self.hparams.val_split])
        elif stage == 'test' or stage == 'predict':
            self.test_ds = Labelec_Preprocessed(data_dir=self.data_dir, split='test', transform=self.test_transform, load_into_memory=self.load_into_memory)
        else:
            raise ValueError(f"Invalid stage: {stage}")   

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
    


if __name__ == '__main__':
    from utils import constants as consts
    import core.datasets.torch_transforms as tt
    from torchvision.transforms import Compose
    from tqdm import tqdm
    import os
    import torch

    transform = Compose([
        tt.EDP_Labels(),
        # tt.Merge_Label({eda.LOW_VEGETATION: eda.MEDIUM_VEGETAION}),
        # tt.Repeat_Points(100_000),
        tt.Farthest_Point_Sampling(100_000),
        tt.Normalize_PCD([0, 1]),
        tt.To(torch.float32),
    ])

    lit_labelec = LitLabelec(las_data_dir=consts.LABELEC_RGB_DIR, 
                             save_chunks=False, 
                             chunk_size=10_000_000, 
                             bins=5, 
                             transform=transform, 
                             test_transform=transform, 
                             load_into_memory=False, 
                             batch_size=16, 
                             val_split=0.2, 
                             num_workers=8
                        )

    split = 'fit'

    # save preprocessed data
    # save preprocessed data
    save_path = os.path.join(consts.LABELEC_RGB_DIR, f"Preprocessed/{split}")
    os.makedirs(save_path, exist_ok=True)

    
    lit_labelec.setup(stage=split)
    batch_size  = 16
    fit_loader = DataLoader(lit_labelec.fit_ds, batch_size=16, num_workers=8, pin_memory=True, shuffle=True)

    for i, batch in tqdm(enumerate(fit_loader)):
        
        for j in range(len(batch)):
            x, y = batch[j]
            idx = batch_size * i + j
            file_id = lit_labelec.fit_ds.chunk_file_paths[idx].split('/')[-1].replace('.laz', '')
            torch.save((x, y), os.path.join(save_path, f"sample_{file_id}.pt"))


    

