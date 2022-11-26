import os
import pickle
import numpy as np
import torch
from pytorch3d.ops import knn_points, knn_gather
import trimesh

CATEGORIES = []
COLORS = np.array([[235, 87, 87], [242, 153, 74], [242, 201, 76],
    [33, 150, 83], [47, 128, 237], [155, 81, 224]])

def read_pkl(path):
    return pickle.load(open(path, 'rb'))

def get_knn(labeled_pcds, labels, unlabeled_pcds, K = 1):
    n_unlabeled = unlabeled_pcds.shape[0]
    labeled_pcds = torch.Tensor(labeled_pcds).unsqueeze(0).repeat(n_unlabeled, 1, 1)
    labels = torch.Tensor(labels).long().repeat(n_unlabeled, 1).unsqueeze(-1)
    unlabeled_pcds = torch.Tensor(unlabeled_pcds)
    _, idx, _ = knn_points(unlabeled_pcds, labeled_pcds, K = K)
    print(labels.shape)
    pred = knn_gather(labels, idx)
    pred, _ = torch.mode(pred)
    return pred

def visualize(pcd, labels):
    if isinstance(pcd, torch.Tensor):
        pcd = pcd.detach().cpu().numpy()
    pcd = trimesh.points.PointCloud(pcd, colors=COLORS[labels])
    pcd.scene().show()

def get_support_query(pcds, pcd_labs, n_q):
    idx = np.random.choice(np.arange(len(pcds)), size=n_q + 1, replace=False)
    return pcds[idx[0]], pcd_labs[idx[0]], pcds[idx[1:]], pcd_labs[idx[1:]]

if __name__ == '__main__':
    data_root = './data/'
    pcds_dict = read_pkl(os.path.join(data_root, 'points_by_cat.pkl'))
    pcd_labs = read_pkl(os.path.join(data_root, 'point_labels_by_cat.pkl'))

    support, support_labs, query, query_labs = get_support_query(pcds_dict['Airplane'], pcd_labs['Airplane'], 4) 
    labels = get_knn(support, support_labs, query)
    for pcd, pcd_lab in zip(query, query_labs):
        visualize(pcd, pcd_lab)