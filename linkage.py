import torch
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

import utils

def compute_pseudolabels(Z, num_targets, num_classes, soft=True):
    coph_dists = cophenet(Z)
    # Convert to square matrix
    ultra_dist_mat = squareform(coph_dists)
    ultra_dists_to_source = torch.tensor(ultra_dist_mat[num_targets:, :-num_classes])

    # print(dists)
    if soft is True:
        return torch.nn.functional.softmin(ultra_dists_to_source, dim=0).T
    else:
        y_pred = torch.argmin(ultra_dists_to_source, dim=0)
        return utils.one_hot(y_pred.T, num_classes)


@torch.no_grad()
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
    Z = linkage(y, method, metric="euclidean", optimal_ordering=True)
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
