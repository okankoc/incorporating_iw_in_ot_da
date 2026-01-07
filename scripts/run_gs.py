import torch
import numpy as np
import ot
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage


def compute_single_linkage_clustering(source_feat, target_feat):
    num_targets = target_feat.shape[0]
    num_classes = len(source_feat)
    dist_targets = torch.tensor(cdist(target_feat, target_feat, metric="euclidean"))
    dist_mat = torch.zeros(num_targets + num_classes, num_targets + num_classes)
    dist_mat[:num_targets, :num_targets] = dist_targets

    for i in range(num_classes):
        for j in range(num_classes):
            dists = torch.tensor(
                cdist(source_feat[i], source_feat[j], metric="euclidean")
            )
            # w_dist = ot.emd2_1d(source_feat[i], source_feat[j])
            min_dist = torch.min(dists)
            dist_mat[num_targets + i, num_targets + j] = min_dist
            # print(dist_mat[num_targets+i, num_targets+j])

    for i in range(num_classes):
        dist_to_source = cdist(target_feat, source_feat[i], metric="euclidean")
        dist_to_source = torch.tensor(dist_to_source)
        min_dist_to_source, _ = torch.min(dist_to_source, dim=1)
        # print(min_dist_to_source)
        # Expand target distances with source cond as a new node
        dist_mat[num_targets + i, :num_targets] = min_dist_to_source
        dist_mat[:num_targets, num_targets + i] = min_dist_to_source
    # Perform single linkage hierarchical clustering
    y = dist_mat[torch.nonzero(torch.triu(dist_mat, diagonal=1), as_tuple=True)]
    Z = linkage(y, method="single", metric="euclidean", optimal_ordering=True)
    return Z


def plot_linkage_clustering(Z, num_targets, num_classes):
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


def gen_gauss_covariates(mean, var, num_samples):
    if var.numel() == 1:
        return mean + torch.sqrt(var) * torch.randn(num_samples)
    dim = mean.shape[0]
    return mean[:, torch.newaxis] + torch.linalg.cholesky(var) @ torch.randn(
        dim, num_samples
    )


def opt_wrr_rule_iter_ls(X_source, X_target, y_source, thresh, theta0=None):
    dim = X_source.shape[-1]
    if theta0 is not None:
        theta = torch.clone(theta0).detach()
    else:
        theta = torch.zeros(dim)
    X_pq = torch.cat((X_source, X_target))
    num_source = X_source.shape[0]
    num_target = X_target.shape[0]
    loss = torch.nn.MSELoss()
    prev_loss = 0.0
    total_loss = 100.0
    w_source = torch.ones(num_source) / num_source
    w_target = torch.ones(num_target) / num_target
    scale_bound = 0.5

    while abs(total_loss - prev_loss) > thresh:
        prev_loss = total_loss
        pred_source = X_source @ theta
        pred_target = X_target @ theta
        Gamma = ot.emd_1d(pred_source, pred_target)
        M_pq = torch.cat(
            (
                torch.cat((torch.diag(w_source), -Gamma), dim=1),
                torch.cat((-Gamma.T, torch.diag(w_target)), dim=1),
            ),
            dim=0,
        )
        R = num_source * (X_pq.T @ M_pq @ X_pq) / scale_bound
        theta = torch.linalg.solve(X_source.T @ X_source + R, X_source.T @ y_source)
        source_loss = loss(pred_source, y_source)

        # Compute the WRR bound for squared distance
        cost_mat = ot.utils.euclidean_distances(
            pred_source[:, torch.newaxis], pred_target[:, torch.newaxis], squared=True
        )
        ot_cost = torch.sum(Gamma * cost_mat)
        total_loss = source_loss + scale_bound * ot_cost

        print(
            f"Total loss: {total_loss}, source loss: {source_loss}, ot dist: {ot_cost}, theta: {theta.detach().numpy()}"
        )
    return (theta, X_target @ theta)


def opt_wrr_rule_with_sgd(X_source, X_target, y_source, thresh, theta0=None):
    # Optimize the Wasserstein-regularized Risk (WRR)
    dim = X_source.shape[-1]
    if theta0 is not None:
        theta = theta0.clone().detach()
    else:
        theta = torch.zeros(dim)
    theta.requires_grad = True
    opt = torch.optim.Adam([theta], lr=1e-3)
    loss = torch.nn.MSELoss()
    num_source = X_source.shape[0]
    num_target = X_target.shape[0]
    w_source = torch.ones(num_source) / num_source
    w_target = torch.ones(num_target) / num_target

    prev_loss = 0.0
    total_loss = 100.0
    while abs(total_loss - prev_loss) > thresh:
        prev_loss = total_loss
        pred_source = X_source @ theta
        pred_target = X_target @ theta
        Gamma = ot.emd_1d(pred_source, pred_target)
        # ot_dist = ot_loss(w_source, pred_source, w_target, pred_target)
        source_loss = loss(pred_source, y_source)

        # Compute the WRR bound for squared distance
        cost_mat = ot.utils.euclidean_distances(
            pred_source[:, torch.newaxis], pred_target[:, torch.newaxis], squared=True
        )
        ot_dist = torch.sum(Gamma * cost_mat)
        total_loss = source_loss + 0.5 * ot_dist
        total_loss.backward()

        print(
            f"Total loss: {total_loss}, source loss: {source_loss}, ot dist: {ot_dist}, theta: {theta.detach().numpy()}"
        )
        opt.step()
        opt.zero_grad()
    theta.requires_grad = False
    return (theta, X_target @ theta)


def opt_pseudolabels_iter_ls(X_source, X_target, y_source, thresh, theta0=None):
    dim = X_source.shape[-1]
    if theta0 is not None:
        theta = torch.clone(theta0).detach()
    else:
        theta = torch.zeros(dim)
    X_pq = torch.cat((X_source, X_target))
    num_source = X_source.shape[0]
    num_target = X_target.shape[0]
    loss = torch.nn.MSELoss()
    prev_loss = 0.0
    total_loss = 100.0
    w_source = torch.ones(num_source) / num_source
    w_target = torch.ones(num_target) / num_target

    while abs(total_loss - prev_loss) > thresh:
        prev_loss = total_loss
        pred_source = X_source @ theta
        pred_target = X_target @ theta

        pred1_source = pred_source[: num_source // 2]
        pred2_source = pred_source[num_source // 2 :]
        pred1_target = pred_target[: num_target // 2]
        pred2_target = pred_target[num_target // 2 :]
        source_feat_cond = [pred1_source[:, None], pred2_source[:, None]]
        target_feat_cond = [pred1_target[:, None], pred2_target[:, None]]
        target_feat = torch.hstack((pred1_target, pred2_target))[:, None]
        Z = compute_single_linkage_clustering(source_feat_cond, target_feat)
        # plot_linkage_clustering(Z, num_targets=target_feat.shape[0], num_classes=2)
        y_pseudo = compute_linkage_pseudolabels(Z, num_target=target_feat.shape[0])
        # Scale 0,1 to 1, -1
        y_pseudo = -2 * y_pseudo + 1

        X_full = torch.vstack((X_source, X_target))
        y_full = torch.hstack((y_source, y_pseudo))
        theta, res, _, _ = torch.linalg.lstsq(X_full, y_full)
        total_loss = loss(X_full @ theta, y_full)
        print(f"Total loss: {total_loss}")
    return (theta, X_target @ theta)


def compute_linkage_pseudolabels(Z, num_target, soft=True):
    # Check pseudolabeling accuracy based on clustering output
    # Compute the 'ultrametric distance'!
    dists = torch.zeros(num_target, 2, dtype=torch.int)
    for i in range(num_target):
        idx_s1 = num_target
        idx_s2 = num_target + 1
        idx_pt = i
        found_both = False
        for j, row in enumerate(Z):
            if found_both == False:
                # Keep track of indices of the point and the source conditionals
                if int(row[0]) == idx_pt:
                    if int(row[1]) == idx_s1 and dists[i, 0] == 0:
                        dists[i, 0] = row[-1] - 1
                    if int(row[1]) == idx_s2 and dists[i, 1] == 0:
                        dists[i, 1] = row[-1] - 1
                    idx_pt = num_target + 2 + j
                if int(row[1]) == idx_pt:
                    if int(row[0]) == idx_s1 and dists[i, 0] == 0:
                        dists[i, 0] = row[-1] - 1
                    if int(row[0]) == idx_s2 and dists[i, 1] == 0:
                        dists[i, 1] = row[-1] - 1
                    idx_pt = num_target + 2 + j
                if dists[i, 0] != 0 and dists[i, 1] != 0:
                    found_both = True
                if int(row[0]) == idx_s1 or int(row[1]) == idx_s1:
                    idx_s1 = num_target + 2 + j
                if int(row[0]) == idx_s2 or int(row[1]) == idx_s2:
                    idx_s2 = num_target + 2 + j
    # print(dists)
    if soft is True:
        # Expected label = 0 times first column + 1 times second column
        dists = torch.tensor(dists, dtype=torch.float)
        return torch.nn.functional.softmin(dists, dim=1)[:, -1]
    else:
        return torch.argmin(dists, dim=1)


def check_gradual_shift_in_linear_classification():
    """
    TODO: Test also weighted WRR whenever both source and target cannot be both seperated!
    TODO: Should we feed in the whole set of source points to the clusterer?
    """

    torch.manual_seed(2)
    num_target = 10
    num_source = 10
    dim = 2

    mean1_source = 2 * torch.ones(dim)
    mean2_source = -2 * torch.ones(dim)
    var1_source = 0.4 * torch.eye(dim)
    var2_source = var1_source
    x1_source = gen_gauss_covariates(
        mean1_source, var1_source, num_samples=num_source // 2
    ).T
    x2_source = gen_gauss_covariates(
        mean2_source, var2_source, num_samples=num_source // 2
    ).T
    y1_source = torch.ones(num_source // 2)
    y2_source = -torch.ones(num_source // 2)

    x_source = torch.vstack((x1_source, x2_source))
    X_source = torch.hstack((torch.ones(num_source, 1), x_source))
    y_source = torch.hstack((y1_source, y2_source))
    theta_ls, res, _, _ = torch.linalg.lstsq(X_source, y_source)
    pred_source_ls = X_source @ theta_ls

    # Check source accuracy!
    loss = torch.nn.CrossEntropyLoss()
    print(f"CE loss: {loss(pred_source_ls, y_source)}")
    y_hat_source_ls = 2 * (pred_source_ls > 0) - 1
    acc = torch.sum(y_hat_source_ls == y_source) / num_source
    print(f"Accuracy % = {acc}")

    # generate gradual shift
    num_shifts = 5
    shift_direction_1 = 10 * torch.tensor([0.0, 1.0])
    shift_direction_2 = 10 * torch.tensor([0.0, 1.0])
    var1_target = var1_source
    var2_target = var2_source
    x_target = torch.zeros(num_target // 2, 2, 2)
    num_gen = num_target // num_shifts
    for i in range(num_shifts):
        mean1_target = mean1_source - (i / num_shifts) * shift_direction_1
        mean2_target = mean2_source + (i / num_shifts) * shift_direction_2
        x_target[i * num_gen // 2 : (i + 1) * num_gen // 2, :, 0] = (
            gen_gauss_covariates(mean1_target, var1_target, num_samples=num_gen // 2).T
        )
        x_target[i * num_gen // 2 : (i + 1) * num_gen // 2, :, 1] = (
            gen_gauss_covariates(mean2_target, var2_target, num_samples=num_gen // 2).T
        )
    x_target = torch.vstack((x_target[:, :, 0], x_target[:, :, 1]))
    y1_target = torch.ones(num_target // 2)
    y2_target = -torch.ones(num_target // 2)
    y_target = torch.hstack((y1_target, y2_target))
    X_target = torch.hstack((torch.ones(num_target, 1), x_target))

    # Check target accuracy, it should be poor for LS!
    pred_target_ls = X_target @ theta_ls
    print(f"CE loss: {loss(pred_target_ls, y_target)}")
    y_hat_target_ls = 2 * (pred_target_ls > 0) - 1
    acc = torch.sum(y_hat_target_ls == y_target) / num_target
    print(f"Accuracy % = {acc}")

    # Viz source and target inputs!
    plt.scatter(x1_source[:, 0], x1_source[:, 1], s=20, c="b", marker="*")
    plt.scatter(x2_source[:, 0], x2_source[:, 1], s=20, c="b", marker="^")
    plt.scatter(x_target[:, 0], x_target[:, 1], s=20, c="r", marker="+")

    # Draw the decision boundary where predictions are zero!
    x_plot = torch.arange(start=-5, end=5, step=0.05)
    y_plot = (-theta_ls[0] - theta_ls[1] * x_plot) / theta_ls[2]
    plt.plot(x_plot, y_plot, c="b", linestyle="--")

    # Test WRR-based optimizer
    theta_wrr, pred_target_wrr = opt_wrr_rule_iter_ls(
        X_source, X_target, y_source, thresh=1e-4
    )
    # Interestingly, the SGD based optimizer does not work well here!
    # Seems to get stuck at local optima?
    # print('Trying WRR opt with SGD...')
    # theta_wrr, pred_target_wrr = opt_wrr_rule_with_sgd(X_source, X_target, y_source, thresh=1e-4, theta0=theta_ls)

    # Draw the decision boundary where predictions are zero!
    x_plot = torch.arange(start=-1, end=1, step=0.05)
    y_plot = (-theta_wrr[0] - theta_wrr[1] * x_plot) / theta_wrr[2]
    plt.plot(x_plot, y_plot, c="r", linestyle="--")

    # Check target accuracy, it should be good for WRR!
    print(f"CE loss: {loss(pred_target_wrr, y_target)}")
    y_hat_target_wrr = 2 * (pred_target_wrr > 0) - 1
    acc = torch.sum(y_hat_target_wrr == y_target) / num_target
    print(f"Accuracy % = {acc}")

    # Test linkage-clustering based pseudolabeler
    theta_pl, pred_target_pl = opt_pseudolabels_iter_ls(
        X_source, X_target, y_source, thresh=1e-4, theta0=theta_ls
    )
    print(f"CE loss: {loss(pred_target_pl, y_target)}")
    y_hat_target_pl = 2 * (pred_target_pl > 0) - 1
    acc = torch.sum(y_hat_target_pl == y_target) / num_target
    print(f"Accuracy % = {acc}")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    check_gradual_shift_in_linear_classification()
