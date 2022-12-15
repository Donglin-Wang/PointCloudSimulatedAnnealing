import pickle
from collections import Counter

import numpy as np

if __name__ == '__main__':
    RAW_ROOT = './data/shapenet_part/raw/'
    PROCESSED_ROOT = './data/shapenet_part/processed/'

    '''Testing the distribution of labels among each category'''
    points = np.load('./data/shapenet_part/raw/points.npy', allow_pickle=True)
    point_labels = np.load('./data/shapenet_part/raw/point_labels.npy', allow_pickle=True)
    shape_labels = np.load('./data/shapenet_part/raw/shape_labels.npy')
    print('Num of shapes: ', shape_labels.shape)

    cat_dist = {shape:Counter() for shape in shape_labels}
    for i, label in enumerate(point_labels):
        cat_dist[shape_labels[i]] += Counter(label)
    print(cat_dist)

    '''Processing Shapenet Part raw data'''
    points = np.load('./data/shapenet_part/raw/points.npy', allow_pickle=True)
    point_labels = np.load('./data/shapenet_part/raw/point_labels.npy', allow_pickle=True)
    shape_labels = np.load('./data/shapenet_part/raw/shape_labels.npy')
    
    points_by_cat = {shape:[] for shape in shape_labels}
    point_labels_by_cat = {shape:[] for shape in shape_labels}

    for point, point_label, shape_label in zip(points, point_labels, shape_labels):
        idx = np.random.choice(np.arange(len(point)), 2048)
        points_by_cat[shape_label].append(point[idx])
        point_labels_by_cat[shape_label].append(point_label[idx])

    points_by_cat = {shape:np.stack(data) for shape, data in points_by_cat.items()}
    point_labels_by_cat = {shape:np.stack(data) for shape, data in point_labels_by_cat.items()}


    pickle.dump(points_by_cat, open(PROCESSED_ROOT + 'points_by_cat.pkl', 'wb'))
    pickle.dump(point_labels_by_cat, open(PROCESSED_ROOT + 'point_labels_by_cat.pkl', 'wb'))
    print({shape:data.shape for shape, data in points_by_cat.items()})
    print({shape:data.shape for shape, data in point_labels_by_cat.items()})
  
    temp_index = {shape:[] for shape in points_by_cat.keys()}
    for cat, instances in point_labels_by_cat.items():
        max_part_num = 0
        part_num = []
        for label in instances:
            cur_part_num = len(np.unique(label))
            part_num.append(cur_part_num)
            max_part_num = max(cur_part_num, max_part_num)
        part_num = np.array(part_num)
        index = np.arange(len(instances))[part_num == max_part_num]
        temp_index[cat] = index
    
    print(list(temp_index.keys()))
    pickle.dump(temp_index, open(PROCESSED_ROOT + 'temp_index.pkl', 'wb'))

    '''Writing meta-data'''
    cat_temp_num = {key:len(data) for key, data in temp_index.items()}
    cat_part_num = {}
    for cat, label in point_labels.items():
        cat_part_num[cat] = np.unique(label).tolist()
    
    meta = {'cat_dist': cat_dist, 'cat_temp_num': cat_temp_num, 'cat_part_num': cat_part_num}
