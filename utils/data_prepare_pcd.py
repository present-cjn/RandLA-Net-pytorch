# -*- coding: utf-8 -*-
"""
Time:     2023/3/16 17:07
Author:   cjn
Version:  1.0.0
File:     data_prepare_pcd.py
Describe: 
"""
import pickle
import numpy as np
from os.path import join, exists
from sklearn.neighbors import KDTree
from helper_tool import DataProcessing as DP


def load_pcd_data(file_path):
    """
    载入一个pcd文件，从第12行开始是数据
    :param file_path:
    :return: numpy数据
    """
    points_I = []
    with open(file_path, 'r') as f:
        data = f.readlines()
        for line in data[11:len(data)-1]:
            str = line.split(' ')
            points_I.append(str)

    return np.array(points_I)


file_path = r'E:\data\cloud'
output_path = r'E:\data\cloud\result'
filename = 'cloud_map'

points = load_pcd_data(join(file_path, filename+'.pcd'))[:, 0:3]
points = points.astype(np.float32)
sub_points = DP.grid_sub_sampling(points, grid_size=0.06)
search_tree = KDTree(sub_points)
proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
proj_inds = proj_inds.astype(np.int32)

KDTree_save = join(output_path, filename + '.pkl')
proj_save = join(output_path, filename + '_proj.pkl')


np.save(join(output_path, filename + '.npy'), sub_points)
with open(KDTree_save, 'wb') as f:
    pickle.dump(search_tree, f)
with open(proj_save, 'wb') as f:
    pickle.dump([proj_inds], f)

