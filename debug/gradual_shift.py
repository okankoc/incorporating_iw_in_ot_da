import torch
import matplotlib.pyplot as plt

import linkage

@torch.no_grad()
def calc_gradual_shift(
    loss_fun, pred_source, pred_target, y_source, y_target, num_classes
):
    num_targets = pred_target.shape[0]
    pred_source_cond = []
    source_means = torch.zeros(num_classes, num_classes)
    for i in range(num_classes):
        cond = pred_source[torch.argmax(y_source, dim=1) == i]
        pred_source_cond.append(cond)
        source_means[i] = torch.mean(cond, dim=0)

    ##### PLOT DISTANCES FROM THE SOURCE CONDITIONAL MEANS
    cond_dist, correct_labeled = compute_cond_dist(pred_target, y_target, source_means, num_classes)
    plot_distances(cond_dist, correct_labeled, num_classes)
    plt.show()

    ##### LINKAGE METHOD TO CHECK IF THERE IS GRADUAL SHIFT
    Z = linkage.compute_cluster(pred_source_cond, pred_target, method='single')
    # linkage.plot_cluster(Z, num_targets, num_classes)
    y_pseudo = linkage.compute_pseudolabels(Z, num_targets, num_classes, soft=False)
    pseudo_loss = loss_fun(y_pseudo, y_target)
    print(f"linkage_pseudo_loss: {pseudo_loss.item()}")
    pseudo_acc = (
        torch.sum(torch.argmax(y_pseudo, dim=1) == torch.argmax(y_target, dim=1))
        / num_targets
    )
    print(f"linkage acc: {100 * pseudo_acc}")


def compute_cond_dist(pred_target, y_target, source_means, num_classes):
    dists = {i: {j: [] for j in range(num_classes)} for i in range(num_classes)}
    correct = {i: [] for i in range(num_classes)}
    pred_labels = pred_target.argmax(dim=1, keepdim=True)
    for i in range(num_classes):
        mask = (torch.argmax(y_target, dim=1) == i)
        cond = pred_target[mask]
        for j in range(num_classes):
            dists[i][j].append(torch.norm(cond - source_means[i], dim=1))
        correct[i].append(pred_labels[mask])

    for i in range(num_classes):
        for j in range(num_classes):
            dists[i][j] = torch.concatenate(dists[i][j], axis=0)
        correct[i] = torch.concatenate(correct[i], axis=0)
    return dists, correct


def plot_distances(dist, correct_labeled, num_classes):
    # Plot the error-bar plot
    colors_without_red = ["blue", "green", "yellow", "orange", "purple", "cyan", "magenta", "brown", "lime", "pink"]
    fig = plt.figure(figsize=(10, 8))
    plt.title(f"Distance of target conditionals to source class mean")
    for i in range(num_classes):
        sort_dist, sort_idx = torch.sort(dist[i][i], descending=False)
        colors = ["red" if value == 0 else colors_without_red[i % 10] for value in correct_labeled[i][sort_idx]]
        x_vec = torch.range(1, len(sort_idx))
        plt.scatter(x_vec, sort_dist, c=colors, label=str(i))
