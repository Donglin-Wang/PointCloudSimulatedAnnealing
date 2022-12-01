import os
import pickle
import numpy as np
import torch
from pytorch3d.ops import knn_points, knn_gather
from pytorch3d.structures import Pointclouds
import trimesh

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
    labeled_pcds = labeled_pcds.unsqueeze(0).repeat(n_unlabeled, 1, 1)
    labels = labels.long().repeat(n_unlabeled, 1).unsqueeze(-1)
    unlabeled_pcds = unlabeled_pcds
    _, idx, _ = knn_points(unlabeled_pcds, labeled_pcds, K=K)
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

    for i, pcd_lab in enumerate(new_labs):
        for j, pnt_lab in enumerate(pcd_lab):
            unique_labs = tuple(pnt_lab.unique().numpy())
            if (len(unique_labs)) == 2:
                combo_to_roi_pcds[unique_labs][i].append(pcds[i][j])
                combo_to_roi_idxs[unique_labs].append((i, j))

    for k, roi_pcds in combo_to_roi_pcds.items():
        for i, roi_pcd in enumerate(roi_pcds):
            combo_to_roi_pcds[k][i] = torch.stack(
                roi_pcd) if roi_pcd else torch.tensor([])
    
    combo_to_roi_pcds = {k:Pointclouds(v) for k, v in combo_to_roi_pcds.items()}

    pcd_labs = pcd_labs.squeeze(-1)
    for _, roi_idxs in combo_to_roi_idxs.items():
        for (i, j) in roi_idxs:
            pcd_labs[i][j] = 6

    return pcd_labs, combo_to_roi_idxs, combo_to_roi_pcds


def get_roi_ball(pcds, pcd_labs):
    pcds, pcd_labs = numpy_to_tensor(pcds, pcd_labs)
    pcd_labs = pcd_labs.long().unsqueeze(-1)
    # _, idx, _ = ball_query(pcds, pcds, radius=0.1)

    haha = pcds[:, :, None].expand(-1, -1, 10, -1)
    # labs = knn_gather(pcd_labs, idx)
    torch.Tensor().gather()
    pass


if __name__ == '__main__':
    data_root = './data/'
    pcds_dict = read_pkl(os.path.join(data_root, 'points_by_cat.pkl'))
    pcd_labs = read_pkl(os.path.join(data_root, 'point_labels_by_cat.pkl'))

    new_labs, combo_to_roi_idxs, combo_to_roi_pcds = get_roi_knn(
        pcds_dict['Guitar'], pcd_labs['Guitar'], 4)
    visualize(pcds_dict['Guitar'][0], new_labs[0])
    for _, roi_pcds in combo_to_roi_pcds.items():
        for roi_pcd in roi_pcds:
            if not roi_pcd.isempty():
                for pcd in roi_pcd.points_list():
                    print(pcd.shape)
