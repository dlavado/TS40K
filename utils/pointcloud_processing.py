"""
In this file we will import the LiDAR data
into dataframes, validate and transform the data
in order to perform EDA on it.

@author: d.lavado

"""
from re import U
from typing import List, Union
import numpy as np
import pandas as pd
import os
import laspy as lp
import open3d as o3d
import matplotlib.pyplot as plt


DICT_EDP_LABELS = {
    0: "Created, unclassified",
    1: "Unclassified",
    2: "Ground",
    3: "Low vegetation",
    4: "Medium vegetation",
    5: "Natural obstacle",
    6: "Human structures",
    7: "Low point (noise)",
    8: "Model keypoints (masspoints)",
    9: "Water",
    10: "Rail",
    11: "Road surface",
    12: "Overlap points",
    13: "Medium reliability",
    14: "Low reliability",
    15: "Power line support tower",
    16: "Main power line",
    17: "Other power line",
    18: "Fiber optic cable",
    19: "Not rated (object to be classified)",
    20: "Not rated (object to be classified)",
    21: "Incidents",
}

# classes of point clouds:
CREATED = 0
UNCLASSIFIED = 1
GROUND = 2
LOW_VEGETATION = 3
MEDIUM_VEGETAION = 4
NATURAL_OBSTACLE = 5
HUMAN_STRUCTURES = 6
LOW_POINT = 7
MODEL_KEYPOINTS = 8
WATER = 9
RAIL = 10
ROAD_SURFACE = 11
OVERLAP_POINTS = 12
MEDIUM_RELIABILITY = 13
LOW_RELIABILITY = 14
POWER_LINE_SUPPORT_TOWER = 15
MAIN_POWER_LINE = 16
OTHER_POWER_LINE = 17
FIBER_OPTIC_CABLE = 18
NOT_RATED_OBJ_TBC = 19
NOT_RATED_OBJ_TBIG = 20
INCIDENTS = 21

DICT_NEW_LABELS = {
    CREATED : 0,
    UNCLASSIFIED : 0,
    MODEL_KEYPOINTS : 0,
    MEDIUM_RELIABILITY : 0,
    LOW_RELIABILITY : 0,  
    NOT_RATED_OBJ_TBIG : 0, 
    LOW_POINT : 0, # noise


    NOT_RATED_OBJ_TBC : 1,
    RAIL : 1,
    WATER : 1, 
    GROUND: 1, # ground

    LOW_VEGETATION : 2,
    OVERLAP_POINTS : 2,
    ROAD_SURFACE : 2,
    MEDIUM_VEGETAION : 3, # vegetation
    NATURAL_OBSTACLE : 3,

    INCIDENTS : 3,
    HUMAN_STRUCTURES : 3, # obstacles
   
    POWER_LINE_SUPPORT_TOWER : 4,

    MAIN_POWER_LINE : 5,
    OTHER_POWER_LINE: 5,
    FIBER_OPTIC_CABLE : 5, # power lines
}

DICT_NEW_LABELS_COLORS = {
    0 : [0, 0, 0],      # noise -> black
    1 : [0.58, 0.3, 0], # ground -> brown
    2 : [0, 0.5, 0],    # low vegetation -> dark green
    3 : [0, 1, 0],      # medium vegetation -> green
    4 : [0, 0, 1],      # power line support tower -> blue
    5 : [1, 0.5, 0],    # power lines -> orange  
}


DICT_OBJ_LABELS_NAMES = {
    0: "Noise",
    1: "Ground",
    2: "Low Vegetation",
    3: "Medium Vegetation",
    4: "Power line support tower",
    5: "Power lines",
}


"""
las.props:
['X', 'Y', 'Z', 'intensity', 'return_number', 'number_of_returns', 'scan_direction_flag', 'edge_of_flight_line', 
'classification', 'synthetic', 'key_point', 'withheld', 'scan_angle_rank', 'user_data', 'point_source_id', 'gps_time']
"""

def las_to_numpy(las, include_feats = ['classification', 'rgb'], header=False):
    """
    Converts .las object into a numpy file.\n

    Parameters
    ----------
    las: laspy 
        .las object to be converted
    
    include_feats: list[str] or 'all'
        features to include in the numpy file;
        for instance, ['classification', 'intensity', 'rgb'] are available features.
        if 'all' is passed, all features are included.

    header: bool
        if True, returns the header of the numpy array with the coloumn names
    """

    feats = list(las.point_format.dimension_names)
    feats = [f.lower() for f in feats]

    if include_feats == 'all':
        include_feats = feats
        include_feats.remove('x')
        include_feats.remove('y')
        include_feats.remove('z') 
   
    xyz = np.vstack((las.x, las.y, las.z)).transpose()
    if header:
        header = ['x', 'y', 'z']
    
    
    for feat in include_feats:
        if feat == 'classification':
            continue  # classification will be added at the end
        
        if feat == 'rgb':
            rgb = np.vstack((las.red, las.green, las.blue)).transpose()
            rgb = process_labelec_rgb(rgb)
            xyz = np.append(xyz, rgb, axis=1)
            continue
        
        if feat not in feats:
            raise ValueError(f"Feature {feat} not available in the las file.")
        
        if header:
            header += [feat]
        
        feat = np.array(las[feat])
        xyz = np.append(xyz, feat.reshape(-1, 1), axis=1)


    
    if 'classification' in include_feats:
        classes = np.array(las.classification)
        xyz = np.append(xyz, classes.reshape(-1, 1), axis=1)
        if header:
            header += ['classification']

    if header:
        return xyz, header

    return xyz
    

def process_labelec_rgb(rgb:np.ndarray) -> np.ndarray:
    """
    Processes the RGB data of the point cloud wrt Labelec's data.\n

    Parameters
    ----------
    `rgb` - (N, 3) ndarray:
        RGB data of the point cloud

    Returns
    -------
    `rgb` - (N, 3) ndarray \in [0, 1]:
        processed RGB data of the point cloud
    """
    rgb = rgb / 256 # normalize the rgb values
    rgb = rgb / 255 # normalize the rgb values
    return rgb

def np_to_ply(xyz:np.ndarray, save=False, filename="pcd.ply"):
    """
    Converts numpy ndarray into ply format in order to use open3D lib\n
    Parameters
    ----------
    `xyz` - numpy ndarray: 
        np to be converted
    [`save` - boolean]: 
        save ? saves the ply obj as filename in Data_Sample folder
    [`filename` - str]: 
        name of the ply obj if save == True
        
    Returns
    -------
    pcd - ply: 
        the corresponding ply obj
    """
    assert xyz.ndim == 2 and xyz.shape[1] == 3, xyz.shape
    pcd = o3d.geometry.PointCloud()
    # for i in range(xyz.shape[0]):
    #     pcd.points.append(xyz[i])
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if save:
        o3d.io.write_point_cloud(os.getcwd() + "/../Data_sample/" + filename, pcd)
    return pcd


def estimate_normals(pcd:Union[np.ndarray, o3d.geometry.PointCloud]) -> np.ndarray:
    """
    Estimates the normals of the point cloud\n

    Parameters
    ----------

    `pcd` - o3d PointCloud:
        point cloud to estimate the normals

    Returns
    -------
    `normals` - np.ndarray:
        the normals of the point cloud
    """

    if type(pcd) == np.ndarray:
        pcd = np_to_ply(pcd[:, :3])

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return np.array(pcd.normals)


def ply_to_np(pcd):
    return np.array(pcd.points)


def visualize_ply(pcd_load, window_name='Open3D'):
    """
    Plots the points clouds given as arg.

    Parameters
    ----------
    `pcd_load` - list of ply objs:
        N point clouds to be visualized
    """
   
    o3d.visualization.draw_geometries(pcd_load, window_name=window_name)


def weights_to_colors(weights:np.ndarray, cmap:str='viridis'):
    """
    Converts the weights into colors\n

    Parameters
    ----------
    `weights` - (N,) ndarray:
        weights of the points
    `cmap` - str:
        colormap to be used

    Returns
    -------
    `colors` - (N, 3) ndarray:
        colors of the points
    """
    # normalize weights
    weights = weights.squeeze()
    weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
    colors = plt.get_cmap(cmap)(weights)
    return colors[:, :3]


def plot_pointcloud(xyz: np.ndarray, classes:np.ndarray=None, rgb=None, window_name='Open3D', cmap:str=None, use_preset_colors=False):
    """
    Plots the point cloud in 3D\n

    Parameters
    ----------
    `xyz` - (N, 3) ndarray: 
        xyz coords of the point cloud
    `classes` - (N,) ndarray:
        classes of the points in the point cloud
    `rgb` - (N, 3) ndarray:
        colors of the points in the point cloud
    `window_name` - str:
        name of the window to be displayed
    `use_preset_colors` - bool:
        if True, uses the colors defined in DICT_NEW_LABELS_COLORS

    Pre-Conds
    ---------
    Either rgb or classes must be not None
    """
    pcd = np_to_ply(xyz)
    color_pointcloud(pcd, classes, colors=rgb, use_preset_colors=use_preset_colors)
    visualize_ply([pcd], window_name=window_name)
    

def color_pointcloud(pcd, classes=None, colors=None, use_preset_colors=False):
    """
    Colors the given point cloud.\n

    Parameters
    ----------
    `pcd` - PointCloud : 
        point cloud data to be colored with n points;
    `classes` - ndarray: 
        classification of the points in pcd; 
    `colors` - np.ndarray: 
       array with shape (n, 3) containing the colors for each point in pcd;
    `use_preset_colors` - bool: 
        if True, uses the colors defined in DICT_NEW_LABELS_COLORS;

    Pre-Conds
    ---------
    class_color.shape == (22, 3);
    np_colors.shape == (len(classes), 3) -> pickle file shape;

    Returns
    -------
    `np_colors` - np.ndarray:
        The colors for each class in format (N, 3)
    """

    if colors is None: # if colors are not provided, 
        if classes is None:
            raise ValueError("Either classes or colors needs to be not None.")
        
        unique_classes = np.unique(classes)

        if use_preset_colors:
            use_preset_colors = np.array(
                [DICT_NEW_LABELS_COLORS[x] for x in unique_classes])
        else:
            # define random colors for each class & normalize them
            use_preset_colors = np.random.choice(
                256, size=(len(unique_classes), 3)) / 255

        colors = np.empty((len(classes), 3))
        for i, c in enumerate(classes):
            colors[i, :] = use_preset_colors[np.where(unique_classes == c)[0][0]]

        # assert colors.shape == (len(classes), 3)

    # assign colors to point cloud according to their class
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return colors

def get_tower_files(files_dirs = [], print_info=True):
    """
    Returns a list with all the .las files that have towers
    """
    tower_files = []
    tower_points = 0
    total_points = 0
    #files_dirs = ["media/didi/TOSHIBA\ EXT/LIDAR", "../Data_sample/"]
    

    for dir in files_dirs:
       
        os.chdir(dir)

        for las_file in os.listdir("."):
            twp, ttp = 0, 0
            filename = os.getcwd() + "/" + las_file
            if ".las" in filename:
                las = lp.read(filename)
            else:
                continue

            #print(f"las_file point format params:\n {list(las.point_format.dimension_names)}")

            xyz, classes = las_to_numpy(las)

            if np.any(classes == POWER_LINE_SUPPORT_TOWER):
                tower_files += [filename]
                twp = len(classes[classes == POWER_LINE_SUPPORT_TOWER])

            ttp = las.header.point_count
            if print_info:
                print(filename + "\n" +
                        f"file total points: {ttp} \n" +
                        f"file tower points: {twp}\nratio: {twp / ttp}\n")

            tower_points += twp
            total_points += ttp
        if print_info:
            print(f"\nnum of files with tower: {len(tower_files)}\n" +
                    f"total points: {total_points} \n" +
                    f"tower points: {tower_points}\nratio: {tower_points / total_points}\n")

    return tower_files


def merge_pcds(xyz:List[np.ndarray], classes:List[np.ndarray]) -> Union[np.ndarray, np.ndarray]:

    merge = None 

    assert len(xyz) == len(classes)

    for i in range(len(xyz)):
        pcd = np.concatenate((xyz[i], classes[i].reshape(-1, 1)), axis = 1)
        if merge is None:
            merge = pcd
        else:
            merge = np.concatenate((merge, pcd), axis=0)
    
    if merge is None:
        return None, None
    
    return merge[:, :-1], merge[:, -1]


def describe_data(X, y=None, column_names=['x', 'y', 'z'], vis=True):
    """
    Describes the data X and returns the corresponding dataframe.

    Parameters
    ----------
    `X` : ndarray
        Array to be described.
    `y` : ndarray, optional
        Array with the classes of X.
    `column_names` : list, optional
        Names of the columns of the dataframe to be returned (excluding the column for y).

    Returns
    -------
    df : pandas.DataFrame
        Dataframe of the described data with y included if y is not None.

    Preconditions
    -------------
    If y is not None, the length of X must be equal to the length of y.
    """
    df = pd.DataFrame(X, columns=column_names)
    if y is not None:
        df["class"] = y
    if vis:
        print(df.describe())
    return df


def normalize_xyz(data:np.ndarray):
    """
    Normalizes the data\n

    Parameters
    ----------
        `data`: ndarray with the data to be scaled

    Returns
    -------
        `scaler`: the scaler of the data
        `scaled_xyz`: the scaled data
    """
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    xyz_shape = data.shape
    scaled_xyz = scaler.fit_transform(data.reshape(-1, data.shape[-1]))
    return scaler, scaled_xyz.reshape(xyz_shape)


def xyz_centroid(xyz:np.ndarray) -> np.ndarray:
    return np.mean(xyz, axis=0)

def select_object(classes:np.ndarray, obj_classes:List[int]) -> np.ndarray:
    """
    Selects the points with the desired classes.\n

    Parameters
    ----------
    `classes` - (N,) ndarray: 
        classes of the points

    `obj_class` - list:
        classes of the desired objects

    Returns
    -------
    `mask` - (N,) ndarray:
        mask for the selected points
    """
    mask = np.isin(classes, obj_classes)
    return mask


def euclidean_distance(x:np.ndarray, y:np.ndarray, axis=None):
    return np.linalg.norm(x - y, axis=axis)


def dbscan(pcd, eps, min_points, visual=False):
    labels = np.array(pcd.cluster_dbscan(
        eps=eps, min_points=min_points, print_progress=visual))

    if len(labels) == 0:
        return np.array([])
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(
        labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = [0, 0, 0, 0]  # -1 means noise
    return colors[:, :3]


def remove_noise(pcd:np.ndarray, eps=15, min_points=100) -> np.ndarray:
    """
    Removes the noise from the point cloud.\n

    Parameters
    ----------
    `pcd` - (N, 3 + M) ndarray:
        point cloud data with the xyz coords and M features
    `eps` - int:
        neigh distance for DBSCAN algorithm
    `min_points` - int:
        min number of neigh for DBSCAN algorithm

    Returns
    -------
    `clean_pcd` - (K, 3 + M) ndarray:
        point cloud data without the noise; K <= N
    """

    pcd_points = pcd[:, :3]
    ply = np_to_ply(pcd_points)

    labels = np.array(ply.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

    mask = labels != -1

    clean_pcd = pcd[mask]
    
    return clean_pcd

def extract_towers(pcd_towers:np.ndarray, eps=15, min_points=100) -> List[np.ndarray]:
    """
    Extracts the points of each individual tower (or structure whose instances can be segregated using DBSCAN).\n

    Parameters
    ----------
    `pcd_towers` - numpy array: 
        point cloud containing SOLELY the intended structures to be extracted
    `eps` - int: 
        neigh distance for DBSCAN algorithm
    `min_points` - int: 
        min number of neigh for DBSCAN algorithm
    (these 2 params are optimized to segregate towers)

    Returns
    -------
        list with N numpy arrays, each numpy contains an instance (xyz coords) of the intended structure.
    """
    
    pcd_tower_points = pcd_towers[:, :3]
    pcd_tower_colors = dbscan(np_to_ply(pcd_tower_points), eps, min_points, False)

    if pcd_tower_colors.size == 0:
        return []

    # each color represents a cluster from the DBSCAN
    tower_colors = np.unique(pcd_tower_colors, axis=0)
    
    # black is reserved for noise in DBSCAN ;
    tower_colors = tower_colors[(tower_colors != [0, 0, 0]).any(axis=1)]
    #assert len(pcd_tower_colors) == len(pcd_tower_points)

    if len(tower_colors) == 0: #no clusters from DBSCAN
        return []

    tower_xyz_rgb = np.append(pcd_tower_points, pcd_tower_colors, axis=1)
    # assert tower_xyz_rgb.shape == (len(pcd_tower_points), 6), tower_xyz_rgb.shape  # x, y, z, r, g, b
    df_tower = describe_data(tower_xyz_rgb, column_names=['x', 'y', 'z', 'r', 'g', 'b'], vis=False)
    group_rgb = df_tower.groupby(['r', 'g', 'b'])

    # towers will contain numpys with the coords of each individual tower in the .las file
    towers = [np.array(group_rgb[['x', 'y', 'z']].get_group(tuple(color)))
              for color in tower_colors]
    # there are as many towers at the end as clusters in the DBSCAN
    # assert len(towers) == len(tower_colors)

    return towers


def crop_tower_radius(pcd:np.ndarray, xyz_tower:np.ndarray, radius=1.0):
    """
    Selects points from the original point cloud around a given tower considering a given radius\n

    Parameters
    ----------
    `pcd` - (N, 3 + M) ndarray:
        point cloud data with the xyz coords and M features
    
    `xyz_tower` - (N, 3) ndarray:
        xyz coords of the tower to be cropped
    
    `radius` - float:
        radius around the tower

    Returns
    -------
    `rad` - (N', 3 + M) ndarray:
        points of the original point cloud around the tower at radius distance
    """
    
    if radius == 0:
        # radius = height of the tower
        radius = np.max(xyz_tower[:, 2]) - np.min(xyz_tower[:, 2])

    baricenter = np.mean(xyz_tower, axis=0) # shape (3,)
    # disregard the z coord
    rad = pcd[np.sum(np.power((pcd[:, :2] - baricenter[:2]), 2), axis=1) <= radius*radius]

    return rad

def crop_tower_samples(pcd:np.ndarray, radius=15) -> List[np.ndarray]:
    """
    Generates samples considering the towers and the area around them.\n

    Parameters
    ----------

    `pcd` - (N, 3 + M) ndarray:
        point cloud data with the xyz coords and M features

    `radius` - int:
        radius around the tower

    Returns
    -------
    `samples` - List[np.ndarray]:
        list with the samples of the point cloud considering the towers
    """

    xyz = pcd[:, :3]
    classes = pcd[:, -1]

    tower_mask = select_object(classes, [POWER_LINE_SUPPORT_TOWER])
    if np.sum(tower_mask) == 0:
        return []
    
    xyz_tower = xyz[tower_mask] # point cloud with only the towers
    towers = extract_towers(xyz_tower)

    samples = []
    for tower in towers:
        crop = crop_tower_radius(pcd, tower, radius=radius)
        samples.append(crop)

    return samples


def crop_two_towers(pcd, xyz_tower1, xyz_tower2):
    """
    Selects the point cloud area between the two given towers.\n

    Parameters
    ----------
    `pcd` - (N, 3 + M) ndarray:
        point cloud data with the xyz coords and M features

    `xyz_tower1` - (N, 3) ndarray:
        xyz coords of tower1

    `xyz_tower2` - (N, 3) ndarray:
        xyz coords of tower2

    Returns
    -------
    `ret` - (N', 3 + M) ndarray:
        area of the original point cloud that is between the given towers
    """

    # determine the min and max points in between the towers to crop the point cloud
    tt = np.concatenate((xyz_tower1, xyz_tower2))
    min1 = np.min(tt, axis=0)
    max2 = np.max(tt, axis=0)

    ret = pcd[((min1[:2] <= pcd[:, :2]) & (pcd[:, :2] <= max2[:2])).all(axis=1)]

    return ret    

def crop_ground_samples(pcd:np.ndarray, step_per_cloud=50):
    """
    Crops the ground samples of the point cloud\n

    Parameters
    ----------

    `pcd` - (N, 3 + M) ndarray:
        point cloud data with the xyz coords and M features
    
    `step_per_cloud` - int:
        the stride in which we will crop the ground samples

    Returns
    -------
    `samples` - List[np.ndarray]:
        list with the ground samples of the point cloud
    """

    xyz = pcd[:, :3]
    classes = pcd[:, -1]

    xyz_min = np.min(xyz, axis=0)
    xyz_max = np.max(xyz, axis=0)
    max_coord = np.argmax(xyz_max - xyz_min)
    samples = []
    
    for i in np.arange(xyz_min[max_coord], xyz_max[max_coord], step_per_cloud):
        mask = (xyz[:, max_coord] >= i) & (xyz[:, max_coord] < i + step_per_cloud)
        if POWER_LINE_SUPPORT_TOWER in classes[mask] or len(classes[mask]) <= 10_000:
            continue # we disregard the samples with towers or with less than 10_000 points
        ground = pcd[mask]
        samples.append(ground)
    
               
    return samples


def crop_two_towers_samples(pcd:np.ndarray) -> List[np.ndarray]:
    """
    Generates samples considering two towers and the area between them.\n


    Parameters
    ----------
    `pcd` - (N, 3 + M) ndarray:
        point cloud data with the xyz coords and M features

    Returns
    -------
    `samples` - List[np.ndarray]:
        list with the samples of the point cloud considering two towers
    """

    xyz = pcd[:, :3]
    classes = pcd[:, -1]

    tower_mask = select_object(classes, [POWER_LINE_SUPPORT_TOWER])
    pcd_tower = xyz[tower_mask] # point cloud with only the towers
    towers = extract_towers(pcd_tower)

    if len(towers) == 1:
        return [crop_tower_radius(pcd, towers[0], radius=0)]
    elif len(towers) == 0:
        return []

    samples = []

    avg_points = np.array([np.mean(tower, axis=0) for tower in towers])

    for i in range(len(towers)):
        eucs = np.array([euclidean_distance(avg_points[i], avg_points[j]) for j in range(len(towers))])
        # idx is the index of the closest tower to towers[i]
        idx = np.argmin(eucs[eucs > 0]) # eucs == 0 is the distance with itself
        if idx >= i:
            idx += 1 # we need to consider the index of the tower in the original list

        two_tower_crop = crop_two_towers(pcd, towers[i], towers[idx])
        tower_one_radius = crop_tower_radius(pcd, towers[i], radius=0)        
        tower_two_radius = crop_tower_radius(pcd, towers[idx], radius=0)
        
        final_crop = np.concatenate((two_tower_crop, tower_one_radius, tower_two_radius), axis=0)
        final_crop = np.unique(final_crop, axis=0) # remove duplicated points
        samples.append(final_crop)

    return samples


if __name__ == "__main__":
    import constants

    TS40K_DIR = constants.TS40K_PATH
    LAS_FILES = [
        os.path.join(TS40K_DIR, "Labelec_LAS"),
        os.path.join(TS40K_DIR, "LIDAR-2022"),
        os.path.join(TS40K_DIR, "LIDAR-2024"),
        os.path.join(TS40K_DIR, "Labelec_LAC_RGB_2024"),
        constants.LAS_RGB_PROCESSED,
        constants.LAS_RGB_ORIGINALS,
    ]


    ##############################
    # REMOVE NOISE FROM POINT CLOUDS
    ##############################




    ##############################
    # DEVELOP CHUNKS FOR LABELEC DATASET
    ##############################
    # curr_las_dir = LAS_FILES[-3] # New data from Labelec
    # curr_las_dir = os.path.join(curr_las_dir, 'fit/')

    # num_samples = {
    #     "tower_radius": 0,
    #     "two_towers": 0,
    #     "ground_samples": 0,
    # }

    # chunk_size = 20_000_000

    # for f in os.listdir(curr_las_dir):
    #     f_path = os.path.join(curr_las_dir, f)
    #     if os.path.isfile(f_path) and ('.las' in f_path or '.laz' in f_path):
    #         print(f"processing file {f_path}...")
    #         with lp.open(f_path) as las:
    #             for points in spatial_chunk_iterator(las, chunk_size, 40):
    #             # for points in las.chunk_iterator(chunk_size):
    #                 pcd = las_to_numpy(points, include_feats=['classification', 'rgb'])
    #                 print(pcd.shape)
    #                 normals = estimate_normals(pcd)
    #                 print(normals.shape)
    #                 classes = pcd[:, -1].astype(int)
    #                 classes = np.array([DICT_NEW_LABELS[c] for c in classes])
    #                 # pcd[:, -1] = classes
    #                 rgb = pcd[:, 3:-1]
    #                 rgb = rgb / 256 # normalize the rgb values
    #                 rgb = rgb / 255 # normalize the rgb values
    #                 pcd[:, 3:-1] = rgb
    #                 uq_classes, uq_count = np.unique(classes, return_counts=True)
    #                 print(uq_classes)
    #                 for i, c in enumerate(uq_classes):
    #                     print(f"class {DICT_OBJ_LABELS_NAMES[c]} has {uq_count[i]/np.sum(uq_count)} ratio")
    #                 print(np.min(rgb, axis=0), np.max(rgb, axis=0))

    #                 plot_pointcloud(pcd[:, :3], classes, None, window_name=f"chunk", use_preset_colors=True)
    #                 # plot_pointcloud(pcd[:, :3], None, rgb, window_name=f"chunk rgb")
                    
                    
    #                 # tower_radius = crop_tower_samples(pcd, radius=30)
    #                 # two_towers = crop_two_towers_samples(pcd)
    #                 # ground_samples = crop_ground_samples(pcd, step_per_cloud=50)
                    
    #                 # print(f"num of towers: {len(tower_radius)}")
    #                 # print(f"num of two towers: {len(two_towers)}")
    #                 # print(f"num of ground samples: {len(ground_samples)}")
    #                 # num_samples["tower_radius"] += len(tower_radius)
    #                 # num_samples["two_towers"] += len(two_towers)
    #                 # num_samples["ground_samples"] += len(ground_samples)

    #                 # for i, tower in enumerate(ground_samples):
    #                 #     print(f"tower {i} has {tower.shape} shape")
    #                 #     tower_class = tower[:, -1].astype(int)
    #                 #     tower_class = np.array([DICT_NEW_LABELS[c] for c in tower_class])
    #                 #     plot_pointcloud(tower[:, :3], tower_class, None, window_name=f"tower {i} classification", use_preset_colors=True)
    #                 #     # print(np.min(tower[:, 3:-1], axis=0), np.max(tower[:, 3:-1], axis=0))
    #                 #     plot_pointcloud(tower[:, :3], None, rgb=tower[:, 3:-1], window_name=f"tower {i} rgb")

                    

    #         print(f"{num_samples}")
    #         input("Press enter to continue...") 
    #         print(f_path)
    #         las = lp.read(f_path)
    #         dim_names = las.point_format.dimension_names
    #         dim_names = list(dim_names)
    #         print(dim_names)
        

    #         pcd, header = las_to_numpy(las, include_feats=['classification', 'rgb'], header=True)

    #         print(header)
    #         print(pcd.shape)

    #         # tower_radius = crop_tower_samples(pcd, radius=15)

    #         # print(f"num of towers: {len(tower_radius)}")
    #         # for i, tower in enumerate(tower_radius):
    #         #     print(f"tower {i} has {len(tower)} points")
    #         #     plot_pointcloud(tower[:, :3], tower[:, -1], window_name=f"tower {i}")
                
    #         xyz = pcd[:, :3]
    #         xyz = (xyz - np.min(xyz, axis=0)) / (np.max(xyz, axis=0) - np.min(xyz, axis=0))
    #         classes = pcd[:, -1].astype(int)
    #         classes = np.array([DICT_NEW_LABELS[c] for c in classes])
    #         print(np.unique(classes))
            
    #         red, green, blue = pcd[:, header.index('red')], pcd[:, header.index('green')], pcd[:, header.index('blue')]
    #         rgb = np.array([red, green, blue]).T
    #         rgb = (rgb / 256) / 255
    #         print(np.min(rgb, axis=0), np.max(rgb, axis=0))
    #         print(np.mean(rgb, axis=0), np.std(rgb, axis=0))
    #         # input("Press enter to continue...")
    #         ply = np_to_ply(xyz)
    #         if np.std(rgb, axis=0)[0] > 0.1:
    #             color_pointcloud(ply, None, colors=rgb)
    #             visualize_ply([ply])
    #         # color_pointcloud(ply, classes, use_preset_colors=True)
    #         # visualize_ply([ply])
    
