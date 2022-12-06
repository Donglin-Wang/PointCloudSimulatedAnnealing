import math
import random

import torch
from tqdm import tqdm
from pytorch3d.ops import knn_points, knn_gather
from pytorch3d.loss import chamfer_distance


class Anneal:
    def __init__(self, support_roi, support_roi_labs, query_pcd, query_lab, query_neigh_idx, start_pnt_idx=None):
        # Get the coordinate from file
        self.query_pcd = query_pcd
        self.query_lab = query_lab
        self.query_neigh_idx = query_neigh_idx
        self.support_roi = support_roi
        self.support_roi_labs = support_roi_labs
        self.anchor_idx = random.randint(0, len(support_roi) - 1)
        self.anchor = support_roi[self.anchor_idx]
        self.support_roi = support_roi

        # Initialize
        self.cur =  start_pnt_idx if start_pnt_idx else self.init_solu()
        self.cur_fit = self.calc_fitness(self.cur)
        self.best = self.cur
        self.best_fit = self.cur_fit
        
        # Set parameters
        self.init_temp = torch.tensor(math.sqrt(2048))
        self.cur_temp = torch.tensor(math.sqrt(2048))
        self.alpha = 0.9995
        self.max_iter = 1000  # number of inner loop n_{in}
        self.num_exp = 2  # number of outer loop n_{out} / random restarts
        self.log = []

    def init_solu(self):
        return random.randint(0, len(self.query_pcd) - 1)

    # Change the path randomly each time
    def tweak(self, solu):
        rand_neigh_idx = random.randint(0, len(self.query_neigh_idx[solu]) - 1)
        return self.query_neigh_idx[solu][rand_neigh_idx]

    # Calculate the full distance of the trip
    def calc_fitness(self, solu):
        cur_query_pnt = self.query_pcd[solu]
        cur_roi = self.support_roi + (self.anchor - cur_query_pnt)
        fitness = chamfer_distance(cur_roi.unsqueeze(0), self.query_pcd.unsqueeze(0))
        return fitness[0]

    # Start annealing
    def anneal(self):
        # Outer loop that executes random restarts
        for _ in range(self.num_exp):
            # Inner loop for the Simulated Annealing iterations
            for _ in range(self.max_iter):
                new_solu = self.tweak(self.best)
                new_fit = self.calc_fitness(new_solu)
                # If a neighboring solution is better, the new solution is accepted
                if new_fit < self.cur_fit:
                    self.cur, self.cur_fit = new_solu, new_fit
                    if new_fit < self.best_fit:
                        # If the new solution is better than the best, the best is updated
                        self.best, self.best_fit = self.cur, self.cur_fit
                # If a worse solution appears
                else:
                    # The value range of math.exp() is (0,1)
                    accept_thresh = torch.exp(-torch.abs(new_fit -
                                             self.cur_fit) / self.cur_temp)
                    # Use Simulated Annealing algorithm to decide whether to transfer or not based on probability
                    if random.random() < accept_thresh:
                        self.cur, self.cur_fit = new_solu, new_fit
                # Reduce temperature and annealing, 0 < alpha < 1. The larger the r, the slower the temperature
                # goes down.
                self.cur_temp *= self.alpha
                self.log.append(self.best_fit)
            self.cur_temp = self.init_temp
            self.cur = self.init_solu()
        self.write_labs()
        return self.query_lab
    
    def write_labs(self):
        cur_query_pnt = self.query_pcd[self.best]
        best_roi = self.support_roi + (self.anchor - cur_query_pnt)
        _, idx, _ = knn_points(best_roi.unsqueeze(0), self.query_pcd.unsqueeze(0), K=1)
        idx = idx[0,:,0]
        for i, j in enumerate(idx):
            self.query_lab[j] = self.support_roi_labs[i]
        


if __name__ == '__main__':
    # Check the support roi shape
    # Check the the
    pass
