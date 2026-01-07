import torch
import ot
import geomloss
import numpy as np
import utils
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


def debug_method(config, method, model, loss_fun, scenario, fabric, idx_des):
    num_batches = len(scenario.source_test_dataloader)
    idx_des = idx_des % num_batches
    idx = 0
    for (X_train, y_train), (X_shift, y_shift) in zip(
        scenario.source_test_dataloader, scenario.target_test_dataloader
    ):
        if idx == idx_des:
            print("============================================")
            print(f"Debugging/validating on {idx}'th test batch")
            y_train = utils.one_hot(y_train, scenario.num_classes)
            y_shift = utils.one_hot(y_shift, scenario.num_classes)
            debug_model(
                config["debug_options"],
                model,
                loss_fun,
                fabric,
                X_train,
                y_train,
                X_shift,
                y_shift,
            )
            if config["validate"] is True:
                method.validate(model, fabric, X_train, y_train, X_shift)
            break
            print("============================================")
        idx += 1


# Debugging by printing Wasserstein-based bounds for all methods!
def debug_model(
    config, model, loss_fun, fabric, X_source, y_source, X_target, y_target
):
    pred_source = model(X_source)
    pred_target = model(X_target)
    target_loss = loss_fun(pred_target, y_target)
    print(f"target_loss: {target_loss.item()}")
    if config["calc_wrr"]:
        source_loss = loss_fun(pred_source, y_source)
        ot_cost, ot_mat = calc_ot(pred_source, pred_target, fabric)
        wrr = source_loss + ot_cost
        print(
            f"WRR: {wrr.item()}, ot_cost: {ot_cost.item()}, source_loss: {source_loss.item()}"
        )
    if config["calc_weighted_wrr"]:
        w_wrr, w_ot_mat, w_source, w_source_loss = calc_weighted_wrr(
            model, fabric, loss_fun, pred_source, pred_target, y_source, reg=1e-1
        )
        w_ot_cost = w_wrr - w_source_loss
        print(
            f"W-WRR: {w_wrr.item()}, w_ot_cost: {w_ot_cost.item()}, w_source_loss: {w_source_loss.item()}"
        )

        # Get the top 5 weights
        if config["verbose_weighted_wrr"]:
            vals, w_idx = torch.sort(w_source, descending=True)
            source_losses = loss_fun(pred_source, y_source, reduction="none")
            num_select = 5
            print(
                f"Top {num_select} source weights: {vals[:num_select].detach().numpy()}, with total weight: {torch.sum(vals[:num_select])}"
            )
            print(
                f"Their labels: {torch.argmax(y_source[w_idx[:num_select]], dim=1).detach().numpy()}"
            )
            print(f"Their losses: {source_losses[w_idx[:num_select]].detach().numpy()}")
            # Print their labels + num of points in batch with same label
            num_source = y_source.shape[0]
            print(
                f"Num of label proportions in batch: {(torch.sum(y_source, dim=0) / num_source).detach().numpy()}"
            )
            print(
                f"Average loss per label: {torch.sum(source_losses[:, None] * y_source, dim=0) / torch.sum(y_source, dim=0)}"
            )
            y_preds = torch.argmax(pred_target, dim=1)
            y_preds_hot = utils.one_hot(y_preds, model.num_classes)
            num_target = pred_target.shape[0]
            print(
                f"Num of predicted target class proportions in batch: {(torch.sum(y_preds_hot, dim=0) / num_target).detach().numpy()}"
            )

    if config["calc_entanglement"]:
        y_dist = torch.cdist(y_source, y_target)
        entanglement = torch.sum(ot_mat * y_dist)
        print(f"Entanglement: {entanglement}")
        y_acc = (y_dist == 0).to(torch.float)
        print(f"OT acc: {torch.sum(ot_mat * y_acc)}")
        # weighted_entanglement = torch.sum(w_ot_mat * y_dist)
        # print(f"Weighted entanglement: {weighted_entanglement}")
        # print(f"Weighted OT acc: {torch.sum(w_ot_mat * y_acc)}")

        # Check entanglement/accuracy of selective alignment
        cost_mat = torch.cdist(pred_source, pred_target, p=2)
        std_cost, mean_cost = torch.std_mean(cost_mat)
        thresh = mean_cost - std_cost
        filter_mat = cost_mat < thresh
        filt_ent = torch.sum(ot_mat * filter_mat * y_dist) / torch.sum(
            ot_mat * filter_mat
        )
        filt_acc = torch.sum(ot_mat * filter_mat * y_acc) / torch.sum(
            ot_mat * filter_mat
        )
        print(f"Distance filtered entanglement/acc: {filt_ent}/{filt_acc}")

        std_margin, mean_margin, margin, correct = calc_margin(pred_source, y_source)
        thresh = mean_margin + std_margin
        filter_vec_margin = margin > thresh
        filt_ent = torch.sum(
            ot_mat[correct][filter_vec_margin] * y_dist[correct][filter_vec_margin]
        )
        filt_ent /= torch.sum(ot_mat[correct][filter_vec_margin])
        filt_acc = torch.sum(
            ot_mat[correct][filter_vec_margin] * y_acc[correct][filter_vec_margin]
        )
        filt_acc /= torch.sum(ot_mat[correct][filter_vec_margin])
        print(f"Margin filtered entanglement/acc: {filt_ent}/{filt_acc}")

    if config["calc_margin"]:
        std_margin, avg_margin, _, _ = calc_margin(pred_source, y_source)
        print(f"Source margin mean: {avg_margin}, std: {std_margin}")
        std_margin, avg_margin, _, _ = calc_margin(pred_target, y_target)
        print(f"Target margin mean: {avg_margin}, std: {std_margin}")
    if config["calc_grad_info"]:
        calc_grad_info(model, loss_fun, fabric, pred_source, pred_target, y_source)
    if config["calc_weight_info"]:
        weight_norms = []
        for name, param in model.named_parameters():
            if "weight" in name:
                weight_norms.append(
                    torch.linalg.vector_norm(param.data, ord=2, dim=None).item()
                )
        print(f"Weight norms for each layer: {weight_norms}")
    if config["calc_label_shift"]:
        calc_w_distance_label_shift(y_source, y_target, model.num_classes)
    if config["calc_gradual_shift"]:
        calc_gradual_shift(loss_fun, pred_source, pred_target, y_source, y_target, model.num_classes)


def calc_gradual_shift(loss_fun, pred_source, pred_target, y_source, y_target, num_classes):
    num_classes = 10
    num_targets = pred_target.shape[0]
    pred_source_cond = []
    for i in range(num_classes):
        cond = pred_source[torch.argmax(y_source, dim=1) == i]
        pred_source_cond.append(cond)
    Z = compute_single_linkage_clustering(pred_source_cond, pred_target)
    # plot_linkage_clustering(Z, num_targets, num_classes)
    y_pseudo = compute_linkage_pseudolabels(Z, num_targets, num_classes, soft=False)

    pseudo_loss = loss_fun(y_pseudo, y_target)
    print(f"linkage_pseudo_loss: {pseudo_loss.item()}")
    pseudo_acc = torch.sum(torch.argmax(y_pseudo, dim=1) == torch.argmax(y_target, dim=1)) / num_targets
    print(f"linkage acc: {100 * pseudo_acc}")


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
    plt.show()


def compute_single_linkage_clustering(source_feat, target_feat):
    num_targets = target_feat.shape[0]
    num_classes = len(source_feat)
    dist_targets = torch.cdist(target_feat, target_feat, p=2)
    dist_mat = torch.zeros(num_targets + num_classes, num_targets + num_classes)
    dist_mat[:num_targets, :num_targets] = dist_targets

    for i in range(num_classes):
        for j in range(num_classes):
            dists = torch.cdist(source_feat[i], source_feat[j], p=2)
            # w_dist = ot.emd2_1d(source_feat[i], source_feat[j])
            min_dist = torch.min(dists)
            dist_mat[num_targets+i, num_targets+j] = min_dist
            # print(dist_mat[num_targets+i, num_targets+j])

    for i in range(num_classes):
        dist_to_source = torch.cdist(target_feat, source_feat[i], p=2)
        min_dist_to_source, _ = torch.min(dist_to_source, dim=1)
        # print(min_dist_to_source)
        # Expand target distances with source cond as a new node
        dist_mat[num_targets+i, :num_targets] = min_dist_to_source
        dist_mat[:num_targets, num_targets+i] = min_dist_to_source
    # Perform single linkage hierarchical clustering
    y = dist_mat[torch.nonzero(torch.triu(dist_mat, diagonal=1), as_tuple=True)]
    Z = linkage(y.detach().numpy(), method="single", metric='euclidean', optimal_ordering=True)
    return Z


def compute_linkage_pseudolabels(Z, num_target, num_classes, soft=True):
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


def calc_grad_info(model, loss_fun, fabric, pred_source, pred_target, y_source):
    model.zero_grad()
    source_loss = loss_fun(pred_source, y_source)
    fabric.backward(source_loss, retain_graph=True)
    grad_source_norms = []
    grad_source = []
    for name, param in model.named_parameters():
        if "weight" in name:
            grad_source.append(param.grad)
            grad_source_norms.append(
                torch.linalg.vector_norm(param.grad, ord=2, dim=None).item()
            )

    model.zero_grad()
    ot_cost, _ = calc_ot(pred_source, pred_target, fabric)
    fabric.backward(ot_cost)
    grad_ot_norms = []
    grad_ot = []
    grad_total_norms = []
    idx = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            grad_ot.append(param.grad)
            grad_ot_norms.append(
                torch.linalg.vector_norm(param.grad, ord=2, dim=None).item()
            )
            grad_total_norms.append(
                torch.linalg.vector_norm(
                    param.grad + grad_source[idx], ord=2, dim=None
                ).item()
            )
            idx += 1
    print(
        f"Grad norms avg. (source/OT/WRR): {np.mean(grad_source_norms)}/{np.mean(grad_ot_norms)}/{np.mean(grad_total_norms)}"
    )

    angles = torch.zeros(len(grad_total_norms))
    for i in range(len(grad_total_norms)):
        inner_prod = torch.sum(grad_source[i] * grad_ot[i]) / (
            grad_source_norms[i] * grad_ot_norms[i]
        )
        angles[i] = torch.acos(inner_prod)
    print(f"Avg. angle between grads : {torch.mean(angles) * 180.0 / torch.pi}")


# Assuming Euclidean distance is to be used for W_{1,l} computation,
# the result is equal to \sqrt(2) / 2 * \sum_i |p_i - q_i|
def calc_w_distance_label_shift(y_source, y_target, num_classes):
    p_y = torch.sum(y_source, dim=0)
    p_y /= torch.sum(p_y)
    q_y = torch.sum(y_target, dim=0)
    q_y /= torch.sum(q_y)
    w_1_euclidean_dist = np.sqrt(2) * torch.sum(torch.abs(p_y - q_y)) / 2
    print(f"W1_distance_labels: {w_1_euclidean_dist}")


def calc_margin(preds, labels):
    # Get correct points with matching labels
    # Find the gap between max and second max (by sorting for now)
    pred_sorted_val, pred_sorted_ind = torch.sort(preds, dim=1, descending=True)
    correct = pred_sorted_ind[:, 0] == labels.argmax(1)
    margin = pred_sorted_val[correct, 0] - pred_sorted_val[correct, 1]
    std_margin, mean_margin = torch.std_mean(margin)
    return std_margin, mean_margin, margin, correct


def calc_weighted_wrr(model, fabric, loss_fun, f_source, f_target, y_source, reg):
    num_target = f_target.shape[0]
    source_loss = loss_fun(f_source, y_source)
    # loss matrix
    w_target = torch.ones(num_target, device=fabric.device) / num_target
    cost_mat = torch.cdist(f_source, f_target, 2)
    source_losses = loss_fun(f_source, y_source, reduction="none")
    cost_mat = cost_mat + source_losses[:, None]

    # ot_mat = torch.softmax(-cost_mat / reg, dim=0) * w_target[None, :]
    num_source = y_source.shape[0]
    w_source = torch.ones(num_source, device=fabric.device) / num_source
    reg_m = (1.0, 100.0)
    # ot_mat = ot.sinkhorn_unbalanced(
    #     w_source, w_target, cost_mat, reg, reg_m, method="sinkhorn_stabilized"
    # )
    ot_mat = ot.unbalanced.mm_unbalanced(
        w_source, w_target, cost_mat, reg_m, div="kl", numItermax=1000
    )

    loss = torch.sum(ot_mat * cost_mat)
    w_source = torch.sum(ot_mat, dim=1)
    w_source_loss = torch.sum(w_source * source_losses)
    return loss, ot_mat, w_source, w_source_loss


def calc_ot(f_source, f_target, fabric, reg=1e-6):
    num_source = f_source.shape[0]
    num_target = f_target.shape[0]

    ### Geomloss
    # ot_loss = geomloss.SamplesLoss(loss="sinkhorn", p=1, blur=reg)
    # cost = ot_loss(f_source, f_target)

    # The problem with using POT is that Sinkhorn is not converging for low reg.
    # and it doesn't seem possible to get a numerically accurate ot_mat from geomloss!
    w_source = torch.ones(num_source, device=fabric.device) / num_source
    num_target = f_target.shape[0]
    w_target = torch.ones(num_target, device=fabric.device) / num_target
    cost_mat = torch.cdist(f_source, f_target, p=2)
    ot_mat = ot.emd(w_source, w_target, cost_mat, numItermax=5000)
    cost = torch.sum(ot_mat * cost_mat)

    return cost, ot_mat
