import os
import pickle
import numpy as np
from pytorch3d.ops import knn_points, knn_gather

import yaml

def read_pkl(path):
    return pickle.load(open(path, 'rb'))

def normalize():
    pass

if __name__ == '__main__':
    data_root = './data/'
    points = read_pkl(os.path.join(data_root, 'points_by_cat.pkl'))
    point_labs = read_pkl(os.path.join(data_root, 'point_labels_by_cat.pkl'))