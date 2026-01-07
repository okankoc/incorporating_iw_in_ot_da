import torch
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

import utils

# TODO: Improve this function, it is too slow
def compute_pseudolabels(Z, num_target, num_classes, soft=True):
    # Check pseudolabeling accuracy based on clustering output
    # Compute the 'ultrametric distance'!
    dists = torch.zeros(num_target, num_classes, dtype=torch.int)
    for i in range(num_target):
        idx_s = num_target + torch.arange(num_classes)
        idx_pt = i
        found_all = False
        for j, row in enumerate(Z):
            if found_all == False:
                # Keep track of indices of the point and the source conditionals
                if int(row[0]) == idx_pt:
                    for k in range(num_classes):
                        if int(row[1]) == idx_s[k] and dists[i, k] == 0:
                            dists[i, k] = row[-1] - 1
                    idx_pt = num_target + num_classes + j
                if int(row[1]) == idx_pt:
                    for k in range(num_classes):
                        if int(row[1]) == idx_s[k] and dists[i, k] == 0:
                            dists[i, k] = row[-1] - 1
                    idx_pt = num_target + num_classes + j
                found_all = True
                for k in range(num_classes):
                    if dists[i, k] == 0:
                        found_all = False
                    if int(row[0]) == idx_s[k] or int(row[1]) == idx_s[k]:
                        idx_s[k] = num_target + num_classes + j
    # print(dists)
    if soft is True:
        # Expected label = 0 times first column + 1 times second column
        dists = torch.tensor(dists, dtype=torch.float)
        return torch.nn.functional.softmin(dists, dim=1)
    else:
        y_pred = torch.argmin(dists, dim=1)
        return utils.one_hot(y_pred, num_classes)


def compute_cluster(source_feat, target_feat, method):
    num_targets = target_feat.shape[0]
    num_classes = len(source_feat)
    dist_targets = torch.cdist(target_feat, target_feat, p=2)
    dist_mat = torch.zeros(num_targets + num_classes, num_targets + num_classes)
    dist_mat[:num_targets, :num_targets] = dist_targets

    for i in range(num_classes):
        for j in range(num_classes):
            dists = torch.cdist(source_feat[i], source_feat[j], p=2)
            min_dist = torch.min(dists)
            dist_mat[num_targets + i, num_targets + j] = min_dist
            # print(dist_mat[num_targets+i, num_targets+j])

    for i in range(num_classes):
        dist_to_source = torch.cdist(target_feat, source_feat[i], p=2)
        min_dist_to_source, _ = torch.min(dist_to_source, dim=1)
        # print(min_dist_to_source)
        # Expand target distances with source cond as a new node
        dist_mat[num_targets + i, :num_targets] = min_dist_to_source
        dist_mat[:num_targets, num_targets + i] = min_dist_to_source
    # Perform single linkage hierarchical clustering
    y = dist_mat[torch.nonzero(torch.triu(dist_mat, diagonal=1), as_tuple=True)]
    Z = linkage(
        y.detach().numpy(), method, metric="euclidean", optimal_ordering=True
    )
    return Z


def plot_cluster(Z, num_targets, num_classes):
    plt.figure()

    # Plot the dendrogram
    def llf(idx):
        if idx < num_targets:
            return str(idx)
        elif idx >= num_targets:
            return "S" + str(idx - num_targets)

    dendrogram(Z, labels=None, leaf_label_func=llf)
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    plt.show()
