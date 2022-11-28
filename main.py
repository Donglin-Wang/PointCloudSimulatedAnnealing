import os
import pickle
import numpy as np
import torch
from pytorch3d.ops import knn_points, knn_gather, ball_query, padded_to_packed
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
import trimesh

CATEGORIES = []
COLORS = np.array([[235, 87, 87], [242, 153, 74], [242, 201, 76],
                   [33, 150, 83], [47, 128, 237], [155, 81, 224], [0, 0, 0]])
BATCH_SIZE = 64


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


def get_roi_knn(pcds, pcd_labs, K=5):
    print(pcds.shape)
    pcds, pcd_labs = numpy_to_tensor(pcds, pcd_labs)
    pcd_labs = pcd_labs.long().unsqueeze(-1)
    _, idx, _ = knn_points(pcds, pcds, K=K)
    new_labs = knn_gather(pcd_labs, idx).squeeze(-1)
    rois = []
    for i, pcd_lab in enumerate(new_labs):
        for j, pnt_lab in enumerate(pcd_lab):
            if (len(pnt_lab.unique())) > 1:
                rois.append((i, j))

    pcd_labs = pcd_labs.squeeze(-1)
    for (i, j) in rois:
        pcd_labs[i][j] = 6
    
    
    return pcd_labs

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

    # a, b = numpy_to_tensor(pcds_dict['Airplane'], pcd_labs['Airplane'])
    # print(a.shape, b.shape)

    # support, support_labs, query, query_labs = get_support_query(
    #     pcds_dict['Airplane'], pcd_labs['Airplane'], 4)
    # labels = get_knn(support, support_labs, query)
    # for pcd, pcd_lab in zip(query, query_labs):
    #     visualize(pcd, pcd_lab)

    new_labs = get_roi_knn(pcds_dict['Airplane'], pcd_labs['Airplane'])
    visualize(pcds_dict['Airplane'][0], new_labs[0])
    # labs = pcd_labs['Airplane']
    # labs[mask] = 6
    # visualize(pcds_dict['Airplane'][0], labs[0])
