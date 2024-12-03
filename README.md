# TS40K Dataset - A 3D Point Cloud Dataset on Transmission Network Systems in Rural Environments

## Overview

The [TS40K] is a collection of 3D point clouds obtained from LiDAR scans. It is designed for research and development in point cloud segmentation and classification, with a special focus on power grid inspection and safety.

This repository contains easy-to-use DataLoaders (in Pytorch) and Data Modules (in Lightining).

## Table of Contents
- [Dataset Description](#dataset-description)
- [Data Format](#data-format)
- [How to Download](#how-to-download)
- [Citation](#citation)
- [Contact](#contact)

## Dataset Description
Data is collected using UAV-mounted LiDAR sensors from a BEV's perpective and saved in `.las` format. 
Notably, the use of UAVs results in capturing data from a birds-eye view perspective, this leads to good data characteristics for learning models, such as high point density, absence of object occlusion, and homogeneous object density.

To safeguard the transmission network topology, we partition the data into 3 types of samples:

- **Tower-radius**: Includes the environment around a power-line support tower, providing a comprehensive view of the surroundings relevant to the tower’s location.
- **Power-line**: Focuses on power lines as the main actors, featuring two towers at opposite sides. This sample type offers insights into the spatial relationships of power lines and their supporting structures.
- **No-tower**: Represents rural terrain without supporting towers but potentially includes power lines. This sample type provides context for areas where transmission infrastructure is absent.

On average, each sample type has a length of 70, 100, and 90 meters, respectively.

#### Overview of TS40K
![image](https://github.com/user-attachments/assets/e6037d68-b290-4ccb-8a7c-334e8dcc9eb8)

## Data Format
Upon Processing the original `.las` scans, data is stored in dictionaries containing in `.pt` format as follows:

```
sample_dict = {
    'type' :            sample_type,  # str \in [tower_radius, 2_towers, no_tower]
    'input_pcd' :       input,  # torch.tensor with shape (N, 3)
    'semantic_labels' : labels[None],  # torch.tensor with shape (N, 1)
    'obj_boxes':        obj_boxes  # list of dicts with keys: ['class_label', 'position', 'dimensions', 'rotation']
}

sample_dict = torch.load(sample_0.pt)
```


## Data Organization
```
/TS40K-Dataset/TS40K-FULL/
                   └── tower_radius/
                   |     └── fit/
                   |          ├ sample_x.pt
                   |          └ sample_y.pt
                   |          ...
                   |     └── test/
                   |          ...                    
                   └── tower_radius/
                   |     └── fit/
                   |     └── test/
                   └── tower_radius/
                   |     └── fit/
                   |     └── test/
```

## How to Download
To access the TS40K Dataset, you are welcome to request access to the data owners by email:
- André Coelho:   Andre.Coelho@edp.com
- Ricardo Santos: RICARDOVIEIRA.SANTOS@edp.com

**Disclaimer**:
The dataset is available for **academic research purposes** only. To access the data, please contact the data owners using your institution's email address.

**Commercial use** of the dataset is **not authorized** without prior consent from the data owners.

For further inquiries or special requests, please reach out to the dataset owners.

## Citation

Our work has been accepted into WACV 2025, please consider citing our work with that reference once it is available!
For now, we leave the ArXiv [paper](https://arxiv.org/abs/2405.13989)

```
@article{lavado2024ts40k,
  title={TS40K: a 3D Point Cloud Dataset of Rural Terrain and Electrical Transmission System},
  author={Lavado, Diogo and Soares, Cl{\'a}udia and Micheletti, Alessandra and Santos, Ricardo and Coelho, Andr{\'e} and Santos, Jo{\~a}o},
  journal={arXiv preprint arXiv:2405.13989},
  year={2024}
}
```

## Contact

For any inquiries reagarding this repository, trouble contating the data owners, or collaborations, please contact me at [d.lavado@campus.fct.unl.pt](mailto:d.lavado@campus.fct.unl.pt)

