
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
import os

sys.path.append("..")
import utils.pointcloud_processing as eda
import utils.constants as constants
from core.datasets.TS40K import TS40K_FULL, TS40K_FULL_Preprocessed
import pprint

"""
------------ RAW DATA INFO ------------
DICT_OBJ_LABELS_NAMES = {
    0: "Noise",
    1: "Ground",
    2: "Low Vegetation",
    3: "Medium Vegetation",
    4: "Power line support tower",
    5: "Power lines",
}

{
    'edp_labels': {'class_counts': array([ 13481700,  13016212, 257742475,  15911922, 147776233,  24763336,
               0,   8496314,         0,         0, 985331645, 389797309,
       516260421,         0,         0,  11195549,  19974279,     40036,
               0, 191795140,         0,         0], dtype=int32),
                'class_density': array([5.19409405e-03, 5.01475551e-03, 9.93004337e-02, 6.13038559e-03,
                                        5.69337438e-02, 9.54056953e-03, 0.00000000e+00, 3.27337458e-03,
                                        0.00000000e+00, 0.00000000e+00, 3.79618686e-01, 1.50177195e-01,
                                        1.98899633e-01, 0.00000000e+00, 0.00000000e+00, 4.31330874e-03,
                                        7.69548972e-03, 1.54246682e-05, 0.00000000e+00, 7.38929064e-02,
                                        0.00000000e+00, 0.00000000e+00]),
                'num_points': 2595582571, -> 2595 million points
                'num_samples': 494},

    'semantic_labels': {'class_counts': array([  34994226, 1434869260,  921969652,  172539569,   11195549, 20014315], dtype=int32),
                        'class_density': array([0.01348222, 0.55281203, 0.35520721, 0.06647431, 0.00431331, 0.00771091]),
                        'num_points': 2595582571,
                        'num_samples': 494
                    }
}

------------ PROCESSED DATA INFO ------------
Splitting data in /media/didi/TOSHIBA EXT/TS40K-Dataset/TS40K-FULL...
Splitting /media/didi/TOSHIBA EXT/TS40K-Dataset/TS40K-FULL/tower_radius/fit...
Number of total samples: 3052
Samples in test: 611
Splitting /media/didi/TOSHIBA EXT/TS40K-Dataset/TS40K-FULL/2_towers/fit...
Number of total samples: 2991
Samples in test: 599
Splitting /media/didi/TOSHIBA EXT/TS40K-Dataset/TS40K-FULL/no_tower/fit...
Number of total samples: 14251
Samples in test: 2851
Overall: 20294


{'2_towers': {'fit': {'class_counts': tensor([ 3538154,  5061452, 10161845,  4042658,   437323,   678568]),
                      'class_density': tensor([0.1479, 0.2116, 0.4248, 0.1690, 0.0183, 0.0284]),
                      'num_points': 23920000,
                      'num_samples': 2392},
              'test': {'class_counts': tensor([ 863410, 1264788, 2576624, 1006467,  109053,  169658]),
                       'class_density': tensor([0.1441, 0.2111, 0.4302, 0.1680, 0.0182, 0.0283]),
                       'num_points': 5990000,
                       'num_samples': 599}},
 'no_tower': {'fit': {'class_counts': tensor([ 4986491, 42352925, 47586630, 16403202,        0,  2670752]),
                      'class_density': tensor([0.0437, 0.3715, 0.4174, 0.1439, 0.0000, 0.0234]),
                      'num_points': 114000000,
                      'num_samples': 11400},
              'test': {'class_counts': tensor([ 1268096, 10799701, 11732319,  4034021,        0,   675863]),
                       'class_density': tensor([0.0445, 0.3788, 0.4115, 0.1415, 0.0000, 0.0237]),
                       'num_points': 28510000,
                       'num_samples': 2851}},
 'overall': {'fit': {'class_counts': tensor([11390445, 53358272, 68601841, 24270866,   750550,  3958026]),
                     'class_density': tensor([0.0702, 0.3287, 0.4226, 0.1495, 0.0046, 0.0244]),
                     'num_points': 162330000,
                     'num_samples': 16233},
             'test': {'class_counts': tensor([ 2827385, 13596240, 17013818,  5985846,   187982,   998729]),
                      'class_density': tensor([0.0696, 0.3348, 0.4190, 0.1474, 0.0046, 0.0246]),
                      'num_points': 40610000,
                      'num_samples': 4061}},
 'tower_radius': {'fit': {'class_counts': tensor([ 2865800,  5943895, 10853366,  3825006,   313227,   608706]),
                          'class_density': tensor([0.1174, 0.2435, 0.4446, 0.1567, 0.0128, 0.0249]),
                          'num_points': 24410000,
                          'num_samples': 2441},
                  'test': {'class_counts': tensor([ 695879, 1531751, 2704875,  945358,   78929,  153208]),
                           'class_density': tensor([0.1139, 0.2507, 0.4427, 0.1547, 0.0129, 0.0251]),
                           'num_points': 6110000,
                           'num_samples': 611}}}

Num of bboxes:
Database Power-line-support-tower: 7937
Database Power-lines: 12776
Database Medium-Vegetation: 16873
Overall: 37686
"""


def generate_density_plots():
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    data = {
        'power_line': {
            'fit': {
                'class_density': np.array([0.1479, 0.2116, 0.4248, 0.1690, 0.0183, 0.0284]),
            },
            'test': {
                'class_density': np.array([0.1441, 0.2111, 0.4302, 0.1680, 0.0182, 0.0283]),
            }
        },
        'no_tower': {
            'fit': {
                'class_density': np.array([0.0437, 0.3715, 0.4174, 0.1439, 0.0000, 0.0234]),
            },
            'test': {
                'class_density': np.array([0.0445, 0.3788, 0.4115, 0.1415, 0.0000, 0.0237]),
            }
        },
        'overall': {
            'fit': {
                'class_density': np.array([0.0702, 0.3287, 0.4226, 0.1495, 0.0046, 0.0244]),
            },
            'test': {
                'class_density': np.array([0.0696, 0.3348, 0.4190, 0.1474, 0.0046, 0.0246]),
            }
        },
        'tower_radius': {
            'fit': {
                'class_density': np.array([0.1174, 0.2435, 0.4446, 0.1567, 0.0128, 0.0249]),
            },
            'test': {
                'class_density': np.array([0.1139, 0.2507, 0.4427, 0.1547, 0.0129, 0.0251]),
            }
        }
    }

    DICT_OBJ_LABELS_NAMES = {
        0: "Noise",
        1: "Ground",
        2: "Low Veg.",
        3: "Medium Veg.",
        4: "Support tower",
        5: "Power lines",
    }

    # Assigning colors to each sample type
    colors = {'tower_radius': 'r', 'power_line': 'g', 'no_tower': 'b'}

    # Plot class densities for each class label
    fig, ax = plt.subplots()
    bar_width = 0.2
    index = np.arange(len(DICT_OBJ_LABELS_NAMES))

    for i, class_label in enumerate(DICT_OBJ_LABELS_NAMES.values()):
        for j, (sample_type, color) in enumerate(colors.items()):
            densities = data[sample_type]['fit']['class_density']
            ax.bar(index[i] + j * bar_width, densities[i], bar_width, color=color)

    # Creating legend patches for each color and sample type
    legend_patches = [Patch(color=color, label=sample_type) for sample_type, color in colors.items()]

    # ax.set_xlabel('Class Labels')
    # ax.set_ylabel('Class Density')
    # ax.set_title('Class Densities for Each Class Label')
    ax.set_xticks(index + (bar_width * (len(colors) - 1)) / 2)
    ax.set_xticklabels(DICT_OBJ_LABELS_NAMES.values())
    
    ax.legend(handles=legend_patches, fontsize=16)

    # Remove the upper and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    
    plt.yticks(fontsize=16)
    plt.xticks(rotation=60, fontsize=20)
    plt.tight_layout()
    # plt.savefig('ts40k_class_densities.png')
    plt.show()

    # Assigning colors to each sample type
    colors = {'fit': 'r', 'test': 'g'}

    # Plot overall class densities
    fig, ax = plt.subplots()
    bar_width = 0.35
    index = np.arange(len(DICT_OBJ_LABELS_NAMES))

    class_densities_fit = data['overall']['fit']['class_density']
    class_densities_test = data['overall']['test']['class_density']

    ax.bar(index - bar_width/2, class_densities_fit, bar_width, color=colors['fit'], label='Train')
    ax.bar(index + bar_width/2, class_densities_test, bar_width, color=colors['test'], label='Test')

    # ax.set_xlabel('Class Labels')
    # ax.set_ylabel('Class Density')
    # ax.set_title('Overall Class Densities')
    ax.set_xticks(index)
    ax.set_xticklabels([DICT_OBJ_LABELS_NAMES[i] for i in range(len(DICT_OBJ_LABELS_NAMES))])
    ax.legend(fontsize=16)

    # Remove the upper and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.yticks(fontsize=16)

    plt.xticks(rotation=60, fontsize=20)
    plt.tight_layout()
    # plt.savefig('ts40k_train_test_distribution.png')
    plt.show()


def get_info_on_ts40k_full(data_path):

    sample_types = ['tower_radius', '2_towers', 'no_tower']

    num_classes = len(eda.DICT_OBJ_LABELS_NAMES) # 6

    init_dict = {
        'class_counts' : torch.zeros(num_classes, dtype=torch.long),
        'num_samples' : 0,
        'num_points' : 0 ,
        'num_bboxes' : 0 ,
        'sample_lengths' : []
    }

    # init dataset  info dict
    dataset_info = {}
    dataset_info['overall'] = {
                                'fit': copy.deepcopy(init_dict),
                                'test': copy.deepcopy(init_dict)
                            }

    for sample_type in sample_types:
        dataset_info[sample_type] = {
                                    'fit': copy.deepcopy(init_dict),
                                    'test': copy.deepcopy(init_dict)
                                }
    
    
    for split in ['fit', 'test']:
        dataset = TS40K_FULL(data_path, split=split, task='sem_seg', sample_types=sample_types, transform=None, load_into_memory=False)
        
        for i in tqdm(range(len(dataset)), desc=f"Reading TS40K {split} dataset"):
            """
            format:
            sample_dict = {
                'type' :            sample_type, # tower_radius, 2_towers, no_tower
                'input_pcd' :       input, # torch.tensor  with shape (N, 3)
                'semantic_labels' : labels[None], # torch.tensor with shape (N, 1)
                'obj_boxes':        obj_boxes # list of dicts with keys: ['class_label', 'position', 'dimensions', 'rotation']
            }    
            """
            sample = dataset._get_dict(i)

            sample_type : str = sample['type']
            sem_target : torch.Tensor = sample['semantic_labels']
            sem_target = sem_target.reshape(-1) # ensures that removes batch or channel dims

            pcd = sample['input_pcd']
            distances = torch.max(pcd, dim=0).values - torch.min(pcd, dim=0).values # get length of each axis

            # print(f"{sample_type =}, {sem_target.unique() =}")2

            # get class counts
            counts = torch.bincount(sem_target, minlength=num_classes)

            assert sum(counts) == len(sem_target), f"{sum(counts) =}, {len(sem_target) =}"

            # update dataset_info
            dataset_info[sample_type][split]['class_counts'] += counts
            dataset_info[sample_type][split]['num_samples'] += 1
            dataset_info[sample_type][split]['num_points'] += len(sem_target)
            dataset_info[sample_type][split]['num_bboxes'] += len(sample['obj_boxes'])
            dataset_info[sample_type][split]['sample_lengths'].append(distances)

            # update overall dataset info
            dataset_info['overall'][split]['class_counts'] += counts
            dataset_info['overall'][split]['num_samples'] += 1
            dataset_info['overall'][split]['num_points'] += len(sem_target)
            dataset_info['overall'][split]['num_bboxes'] += len(sample['obj_boxes'])
            dataset_info['overall'][split]['sample_lengths'].append(distances)


    # update class density
    for key in dataset_info.keys():
        for split in ['fit', 'test']:
            dataset_info[key][split]['class_density'] = dataset_info[key][split]['class_counts'] / dataset_info[key][split]['num_points']

            dataset_info[key][split]['sample_lengths'] = torch.stack(dataset_info[key][split]['sample_lengths'])
            
    return dataset_info


def get_info_on_ts40k_full_preprocessed(data_path):
    sample_types = ['tower_radius', '2_towers', 'no_tower']

    num_classes = len(eda.DICT_OBJ_LABELS_NAMES) # 6

    init_dict = {
        'class_counts' : torch.zeros(num_classes, dtype=torch.long),
        'num_samples' : 0,
        'num_points' : 0,
        'sample_lengths' : []
    }

    # init dataset  info dict
    dataset_info = {}
    dataset_info['overall'] = {
                                'fit': copy.deepcopy(init_dict),
                                'test': copy.deepcopy(init_dict)
                            }

    for sample_type in sample_types:
        dataset_info[sample_type] = {
                                    'fit': copy.deepcopy(init_dict),
                                    'test': copy.deepcopy(init_dict)
                                }
    
        for split in ['fit', 'test']:
            dataset = TS40K_FULL_Preprocessed(data_path, split=split, sample_types=[sample_type], load_into_memory=False)
        
            for i in tqdm(range(len(dataset)), desc=f"Reading TS40K {split} {sample_type} dataset"):
                """
                format:
                tuple = (input, labels)
                    input : torch.tensor with shape (N, 3)

                    labels : torch.tensor with shape (N,)  
                """
                pcd, sem_target = dataset[i]

                sem_target = sem_target.to(torch.int)

                distances = torch.max(pcd, dim=0).values - torch.min(pcd, dim=0).values # get length of each axis

                # print(f"{sem_target.unique() =}")

                # get class counts
                counts = torch.bincount(sem_target, minlength=num_classes)

                assert sum(counts) == len(sem_target), f"{sum(counts) =}, {len(sem_target) =}"

                # update dataset_info
                dataset_info[sample_type][split]['class_counts'] += counts
                dataset_info[sample_type][split]['num_samples'] += 1
                dataset_info[sample_type][split]['num_points'] += len(sem_target)
                dataset_info[sample_type][split]['sample_lengths'].append(distances)

                # update overall dataset info
                dataset_info['overall'][split]['class_counts'] += counts
                dataset_info['overall'][split]['num_samples'] += 1
                dataset_info['overall'][split]['num_points'] += len(sem_target)
                dataset_info['overall'][split]['sample_lengths'].append(distances)

    # update class density
    for key in dataset_info.keys():
        for split in ['fit', 'test']:
            dataset_info[key][split]['class_density'] = dataset_info[key][split]['class_counts'] / dataset_info[key][split]['num_points']

            dataset_info[key][split]['sample_lengths'] = torch.stack(dataset_info[key][split]['sample_lengths'])

    return dataset_info

def get_info_on_ts40k_raw(data_paths:list[str]):
    """
    Get info on the TS40K dataset raw data.
    This data is composed of .las files that contain [x, y, z, label] information.
    """
    import laspy as lp

    num_classes = len(eda.DICT_EDP_LABELS)
    num_sem_classes = len(eda.DICT_OBJ_LABELS_NAMES)

    dataset_dict = {
        'edp_labels' : {
            'class_counts' : np.zeros(num_classes, dtype=np.int32),
            'num_samples' : 0,
            'num_points' : 0,
            'lidar_lengths' : [] 
        },
        'semantic_labels' : {
            'class_counts' : np.zeros(num_sem_classes, dtype=np.int32),
            'num_samples' : 0,
            'num_points' : 0
        },
    }

    for cwd in data_paths:

        print(f"\n\n\nReading files in {cwd}...")

        for las_file in os.listdir(cwd):
            filename = os.path.join(cwd, las_file)

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

            pcd, classes = eda.las_to_numpy(las)

            distances = np.max(pcd, axis=0) - np.min(pcd, axis=0)

            sem_labels = np.array([eda.DICT_NEW_LABELS[c] for c in classes])

            # get class counts
            counts = np.bincount(classes, minlength=num_classes)
            sem_counts = np.bincount(sem_labels, minlength=num_sem_classes)

            assert sum(counts) == len(classes), f"{sum(counts) =}, {len(classes) =}"
            assert sum(sem_counts) == len(sem_labels), f"{sum(sem_counts) =}, {len(sem_labels) =}"

            # update info_dict
            dataset_dict['edp_labels']['class_counts'] += counts
            dataset_dict['edp_labels']['num_samples'] += 1
            dataset_dict['edp_labels']['num_points'] += len(classes)
            dataset_dict['edp_labels']['lidar_lengths'].append(distances)

            dataset_dict['semantic_labels']['class_counts'] += sem_counts
            dataset_dict['semantic_labels']['num_samples'] += 1
            dataset_dict['semantic_labels']['num_points'] += len(sem_labels)


    # update class density
    for key in dataset_dict.keys():
        dataset_dict[key]['class_density'] = dataset_dict[key]['class_counts'] / dataset_dict[key]['num_points']

    dataset_dict['edp_labels']['lidar_lengths'] = np.array(dataset_dict['edp_labels']['lidar_lengths'])
    
    return dataset_dict


def visualize_raw_ts40k(las_dir):
    #las_dir = os.path.join(TS40K_DIR, "Labelec_LAS")
    for las in os.listdir(las_dir):

        filename = os.path.join(las_dir, las)

        if ".las" in filename:
            try:
                las = lp.read(filename)
            except Exception as e:
                print(f"Problem occurred while reading {filename}\n\n")
                continue
        else:
            continue
        
        xyz, classes = eda.las_to_numpy(las)
        print(np.unique(classes))
        pynt = eda.np_to_ply(xyz)
        sem_labels = np.array([eda.DICT_NEW_LABELS[c] for c in classes])
        eda.color_pointcloud(pynt, sem_labels, use_preset_colors=True)
        eda.visualize_ply([pynt]) 


def visualize_full_ts40k(data_path, sample_type='all', task='sem_seg'):


    dm = TS40K_FULL(data_path, split='fit', task=task, sample_types=sample_type, transform=None, load_into_memory=False)
    
    idxs = np.arange(len(dm))
    np.random.shuffle(idxs)

    for i in idxs:
        if task == 'sem_seg':
            xyz, y = dm[i]
            y = y.reshape(-1).numpy()
            xyz = xyz.squeeze().numpy()
            pynt = eda.np_to_ply(xyz)
            eda.color_pointcloud(pynt, y, use_preset_colors=True)
            eda.visualize_ply([pynt])
        else: # objdet
            sample = dm._get_dict(i)
            xyz = sample['input_pcd'].squeeze().numpy()
            bboxes = sample['obj_boxes']
            eda.visualize_ply_with_bboxes(xyz, bboxes)

        # ans = input("Press 'q' to exit...")
        # if ans == 'q':
        #     print("Exiting...")
        #     break


if __name__ == '__main__':
    import laspy as lp

    data_path = constants.TS40K_FULL_PATH
    data_path_preprocessed = constants.TS40K_FULL_PREPROCESSED_PATH
    TS40K_DIR = constants.TS40K_PATH

    LAS_DIRS = [
        os.path.join(TS40K_DIR, "LIDAR-2022"),
        os.path.join(TS40K_DIR, "LIDAR-2024"), 
        # os.path.join(TS40K_DIR, "Labelec_LAS")
    ]
    

    # visualize_full_ts40k(data_path, sample_type=['2_towers'], task='sem_seg')

    #visualize_raw_ts40k(LAS_DIRS[0])

    # ----------------------------------------------

    # dataset_info = get_info_on_ts40k_raw(
    #     [os.path.join(TS40K_DIR, "LIDAR-2022"), os.path.join(TS40K_DIR, "Labelec_LAS")]
    # )

    # lengths = dataset_info['edp_labels']['lidar_lengths']
    # print(f"Average lengths: {np.mean(lengths, axis=0)}")
    # print(f"Max lengths: {np.max(lengths, axis=0)}")
    # print(f"Min lengths: {np.min(lengths, axis=0)}")

    # pprint.pprint(dataset_info)
    # input("Press Enter to continue...")

    # ----------------------------------------------

    # dataset_info = get_info_on_ts40k_full(data_path)

    # for key in dataset_info.keys():
    #     for split in ['fit', 'test']:
    #         print(f"{key} {split} dataset:")
    #         print(f"{dataset_info[key][split]['num_samples'] =}")
    #         print(f"{dataset_info[key][split]['num_points'] =}")
    #         lengths = dataset_info[key][split]['sample_lengths']
    #         print(f"Average lengths: {torch.mean(lengths, dim=0)}")
    #         print(f"Max lengths: {torch.max(lengths, dim=0).values}")
    #         print(f"Min lengths: {torch.min(lengths, dim=0).values}")
    #         print("\n\n")
    # print dataset info dict in a nice way
    # pprint.pprint(dataset_info)
    #generate_plots(dataset_info)


    # ----------------------------------------------

    generate_density_plots()



        
            
            






