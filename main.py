import os
import pickle

from anneal import *

import numpy as np
import torch
from pytorch3d.ops import knn_points, knn_gather
from pytorch3d.structures import Pointclouds
import trimesh
from tqdm import tqdm

DATA_ROOT = './data/'
CATEGORIES = []
COLORS = np.array([[235, 87, 87], [242, 153, 74], [242, 201, 76],
                   [33, 150, 83], [47, 128, 237], [155, 81, 224], [0, 0, 0]])
BATCH_SIZE = 64
CAT_TO_NUM_PARTS = {'Motorbike': 6, 'Guitar': 3, 'Rocket': 3, 'Cap': 2, 'Bag': 2, 'Airplane': 4, 'Lamp': 4,
                    'Car': 4, 'Skateboard': 3, 'Table': 3, 'Mug': 2, 'Knife': 2, 'Chair': 4, 'Laptop': 2, 'Pistol': 3, 'Earphone': 3}


def numpy_to_tensor(*arrs):
    result = []
    for arr in arrs:
        result.append(torch.Tensor(arr))
    return result


def tensor_to_numpy():
    pass


def read_pkl(path):
    return pickle.load(open(path, 'rb'))


def get_knn(labeled_pcds, labels, unlabeled_pcds, K=1):
    n_unlabeled = unlabeled_pcds.shape[0]
    labeled_pcds, labels, unlabeled_pcds = numpy_to_tensor(
        labeled_pcds, labels, unlabeled_pcds)
    labeled_pcds = labeled_pcds.repeat(n_unlabeled, 1, 1)
    labels = labels.long().repeat(n_unlabeled, 1).unsqueeze(-1)
    _, idx, _ = knn_points(unlabeled_pcds, labeled_pcds, K=K)
    pred = knn_gather(labels, idx).squeeze(-1)
    pred, _ = torch.mode(pred)
    return pred


def visualize(pcd, labels):
    if isinstance(pcd, torch.Tensor):
        pcd = pcd.detach().cpu().numpy()
    pcd = trimesh.points.PointCloud(pcd, colors=COLORS[labels])
    pcd.scene().show()


def get_support_query(pcds, pcd_labs, n_q):
    idx = np.random.choice(np.arange(len(pcds)), size=n_q + 1, replace=False)
    return pcds[[idx[0]]], pcd_labs[[idx[0]]], pcds[idx[1:]], pcd_labs[idx[1:]]


def get_roi_knn(pcds, pcd_labs, n_parts, K=32):

    pcds, pcd_labs = numpy_to_tensor(pcds, pcd_labs)
    pcd_labs = pcd_labs.long().unsqueeze(-1)

    _, idx, _ = knn_points(pcds, pcds, K=K)
    new_labs = knn_gather(pcd_labs, idx).squeeze(-1)

    combo_to_roi_pcds = {(i, j): [[] for _ in range(pcds.shape[0])]
                         for i in range(n_parts)
                         for j in range(i + 1, n_parts)}
    combo_to_roi_idxs = {(i, j): []
                         for i in range(n_parts)
                         for j in range(i + 1, n_parts)}
    combo_to_roi_labs = {(i, j): [[] for _ in range(pcds.shape[0])]
                         for i in range(n_parts)
                         for j in range(i + 1, n_parts)}

    for i, pcd_lab in enumerate(new_labs):
        for j, pnt_lab in enumerate(pcd_lab):
            unique_labs = tuple(pnt_lab.unique().numpy())
            if (len(unique_labs)) == 2:
                combo_to_roi_pcds[unique_labs][i].append(pcds[i][j])
                combo_to_roi_labs[unique_labs][i].append(pcd_labs[i][j][0])
                combo_to_roi_idxs[unique_labs].append((i, j))

    for k, roi_pcds in combo_to_roi_pcds.items():
        for i, roi_pcd in enumerate(roi_pcds):
            combo_to_roi_pcds[k][i] = torch.stack(
                    roi_pcd) if roi_pcd else torch.tensor([])
            combo_to_roi_labs[k][i] = torch.tensor(combo_to_roi_labs[k][i])

    combo_to_roi_pcds = {k: Pointclouds(v)
                         for k, v in combo_to_roi_pcds.items()}

    return idx, combo_to_roi_idxs, combo_to_roi_pcds, combo_to_roi_labs


def calc_accuracy():
    pass

def calc_iou(pred, target):
    mask = pred == target
    unique_labs = torch.unique(target)
    total_iou = 0.0
    for lab in unique_labs:
        n_pred = torch.sum(pred == lab)
        n_gt = torch.sum(target == lab)
        n_intersect = torch.sum((target == lab) * mask)
        n_union = n_pred + n_gt - n_intersect
        if n_union == 0:
            total_iou += 1
        else:
            total_iou += n_intersect * 1.0 / n_union
    
    return total_iou / len(unique_labs)


def main():
    category = "Car"
    pcds_dict = read_pkl(os.path.join(DATA_ROOT, 'points_by_cat.pkl'))
    pcd_labs = read_pkl(os.path.join(DATA_ROOT, 'point_labels_by_cat.pkl'))
    support_pcds, support_labs, query_pcds, query_labs = get_support_query(
        pcds_dict[category], pcd_labs[category], 5)
    query_pcds = torch.tensor(query_pcds).float()
    query_labs = torch.tensor(query_labs).long()
    pred = get_knn(support_pcds, support_labs, query_pcds, K=3)
    new_pred = pred.clone()
    # Get initial pred accurancy
    support_closest_idx, support_roi_idxs, support_roi_pcds, support_roi_labs = get_roi_knn(support_pcds, support_labs, 4)
    query_closest_idx, query_roi_idxs, query_roi_pcds, query_roi_labs = get_roi_knn(query_pcds, pred, 4)

    for combo, roi_pcd in support_roi_pcds.items():
        if roi_pcd.isempty(): continue
        roi_pcd = roi_pcd.points_list()[0]
        roi_labs = support_roi_labs[combo][0]
        for i in tqdm(range(len(query_pcds)), leave=False):
            # visualize(query_pcds[i], pred[i])
            anneal = Anneal(roi_pcd, roi_labs, query_pcds[i], new_pred[i], query_closest_idx[i])
            anneal.anneal()
    # print(torch.equal(pred, new_pred))
            # visualize(query_pcds[i], new_query_lab)
    for i in range(len(new_pred)):
        visualize(query_pcds[i], pred[i])
        visualize(query_pcds[i], new_pred[i])
    
    avg_iou, new_avg_iou = 0, 0
    for p, gt in zip(pred, query_labs):
        avg_iou += calc_iou(p, gt)
    avg_iou /= len(pred)
    print(avg_iou)
    for p, gt in zip(new_pred, query_labs):
        new_avg_iou += calc_iou(p, gt)
    new_avg_iou /= len(new_pred)
    print(new_avg_iou)

if __name__ == '__main__':
    data_root = './data/'
    pcds_dict = read_pkl(os.path.join(data_root, 'points_by_cat.pkl'))
    pcd_labs = read_pkl(os.path.join(data_root, 'point_labels_by_cat.pkl'))

    # new_labs, combo_to_roi_idxs, combo_to_roi_pcds = get_roi_knn(
    #     pcds_dict['Guitar'], pcd_labs['Guitar'], 4)
    # visualize(pcds_dict['Guitar'][0], new_labs[0])
    # for _, roi_pcds in combo_to_roi_pcds.items():
    #     for roi_pcd in roi_pcds:
    #         if not roi_pcd.isempty():
    #             for pcd in roi_pcd.points_list():
    #                 print(pcd.shape)
    # main()
    cnt = 0
    num = 0
    for k, v in pcds_dict.items():
        cnt += len(v)
        num += 1
    print(cnt / num)
