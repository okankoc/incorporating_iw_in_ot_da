import os
import torch
import ot
import geomloss
import numpy as np
import matplotlib.pyplot as plt

import utils
from debug.gradual_shift import calc_gradual_shift


class Debugger:
    def __init__(self, scenario):
        self.source_dl = utils.ForeverDataIterator(scenario.source_test_dataloader)
        self.target_dl = utils.ForeverDataIterator(scenario.target_test_dataloader)
        margin_dict = {'source': {'mean': [], 'std': []}, 'target': {'mean': [], 'std': []}}
        grad_dict = {'source_norm': [], 'ot_norm': [], 'angle': []}
        self.metrics = {'target_loss': [], 'wrr': [], 'w2r2': [],
                        'margin': margin_dict, 'grad': grad_dict}

    def calc_metrics(self, config, model, loss_fun, scenario, fabric):
        X_train, y_train = next(self.source_dl)
        X_shift, y_shift = next(self.target_dl)
        print("============================================")
        print(f"Debugging/validating on test batch")
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
            self.metrics,
        )
        print("============================================")


    # Assuming only one algorithm and one run
    def save_metrics_plot(self, config):
        num_batches = len(self.metrics['wrr'])
        folder_name = os.path.join("results", "debug")
        os.makedirs(folder_name, exist_ok=True)

        # Saving WRR and weighted WRR values together with target loss in a separate plot
        fig, ax = plt.subplots()
        ax.plot(np.arange(num_batches), np.array(self.metrics['target_loss']), label='target_loss')
        if config['calc_wrr'] is True:
            ax.plot(np.arange(num_batches), np.array(self.metrics['wrr']), label='wrr')
        if config['calc_weighted_wrr'] is True:
            ax.plot(np.arange(num_batches), np.array(self.metrics['w2r2']), label='weighted_wrr')
        ax.legend(loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.savefig(os.path.join(folder_name, "wrr_test_vals.pdf"), format="pdf")

        # Saving margin and target loss in a separate plot
        if config['calc_margin'] is True:
            fig, ax = plt.subplots()
            ax.plot(np.arange(num_batches), np.array(self.metrics['target_loss']), label='target_loss')
            ax.errorbar(x=np.arange(num_batches), y=self.metrics['margin']['source']['mean'], yerr=self.metrics['margin']['source']['std'], label='source_margin')
            ax.errorbar(x=np.arange(num_batches), y=self.metrics['margin']['target']['mean'], yerr=self.metrics['margin']['target']['std'], label='target_margin')
            ax.legend(loc="lower right")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.savefig(os.path.join(folder_name, "margin_test_vals.pdf"), format="pdf")

        # Saving gradient norms and target loss in a separate plot
        if config['calc_grad_info'] is True:
            fig, ax = plt.subplots()
            ax.plot(np.arange(num_batches), np.array(self.metrics['target_loss']), label='target_loss')
            ax.plot(np.arange(num_batches), np.array(self.metrics['grad']['source_norm']), label='source_grad_norm')
            ax.plot(np.arange(num_batches), np.array(self.metrics['grad']['ot_norm']), label='ot_grad_norm')
            ax.plot(np.arange(num_batches), np.array(self.metrics['grad']['angle']), label='grad_angle')
            ax.legend(loc="lower right")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.savefig(os.path.join(folder_name, "grad_test_vals.pdf"), format="pdf")


# Debugging by printing Wasserstein-based bounds for all methods!
def debug_model(
    config, model, loss_fun, fabric, X_source, y_source, X_target, y_target, metrics
):
    pred_source = model(X_source)
    pred_target = model(X_target)
    target_loss = loss_fun(pred_target, y_target)
    metrics['target_loss'].append(target_loss.item())
    print(f"target_loss: {target_loss.item()}")
    if config["calc_wrr"]:
        source_loss = loss_fun(pred_source, y_source)
        ot_cost, ot_mat = calc_ot(pred_source, pred_target, fabric)
        wrr = source_loss + ot_cost
        metrics['wrr'].append(wrr.item())
        print(
            f"WRR: {wrr.item()}, ot_cost: {ot_cost.item()}, source_loss: {source_loss.item()}"
        )
    if config["calc_weighted_wrr"]:
        w_wrr = debug_weighted_wrr(config, model, fabric, loss_fun, pred_source, pred_target, y_source)
        metrics['w2r2'].append(w_wrr.item())
    if config["calc_entanglement"]:
        calc_entanglement(y_source, y_target, ot_mat)
    if config["calc_margin"]:
        std_margin, avg_margin, _, _ = calc_margin(pred_source, y_source)
        metrics['margin']['source']['mean'].append(avg_margin.item())
        metrics['margin']['source']['std'].append(std_margin.item())
        print(f"Source margin mean: {avg_margin}, std: {std_margin}")
        std_margin, avg_margin, _, _ = calc_margin(pred_target, y_target)
        print(f"Target margin mean: {avg_margin}, std: {std_margin}")
        metrics['margin']['target']['mean'].append(avg_margin.item())
        metrics['margin']['target']['std'].append(std_margin.item())
    if config["calc_grad_info"]:
        mean_source_grad_norm, mean_ot_grad_norm, mean_angle = calc_grad_info(model, loss_fun, fabric, pred_source,
                                                                              pred_target, y_source)
        metrics['grad']['source_norm'].append(mean_source_grad_norm.item())
        metrics['grad']['ot_norm'].append(mean_ot_grad_norm.item())
        metrics['grad']['angle'].append(mean_angle.item())
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
        calc_gradual_shift(
            loss_fun, pred_source, pred_target, y_source, y_target, model.num_classes
        )


def debug_weighted_wrr(config, model, fabric, loss_fun, pred_source, pred_target, y_source):
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
    return w_wrr


def calc_entanglement(y_source, y_target, ot_mat):
    y_dist = torch.cdist(y_source, y_target)
    entanglement = torch.sum(ot_mat * y_dist)
    print(f"Entanglement: {entanglement}")
    y_acc = (y_dist == 0).to(torch.float)
    print(f"OT acc: {torch.sum(ot_mat * y_acc)}")
    # weighted_entanglement = torch.sum(w_ot_mat * y_dist)
    # print(f"Weighted entanglement: {weighted_entanglement}")
    # print(f"Weighted OT acc: {torch.sum(w_ot_mat * y_acc)}")

    # Check entanglement of ultrametric-based coupling
    # import linkage
    # num_source, num_target = pred_source.shape[0], pred_target.shape[0]
    # ultra_dist_mat = linkage.compute_soft_cluster(pred_source, pred_target)
    # w_source = torch.ones(num_source, device=fabric.device) / num_source
    # w_target = torch.ones(num_target, device=fabric.device) / num_target
    # ultra_ot_mat = ot.emd(w_source, w_target, ultra_dist_mat, numItermax=5000)
    # ultra_entanglement = torch.sum(ultra_ot_mat * y_dist)
    # print(f"Ultrametric-OT entanglement: {ultra_entanglement}")

    # Check entanglement/accuracy of selective alignment
    # check_selective_alignment(pred_source, pred_target, y_source, y_target, ot_mat)


def check_selective_alignment(pred_source, pred_target, y_source, y_target, ot_mat):
    y_dist = torch.cdist(y_source, y_target)
    y_acc = (y_dist == 0).to(torch.float)
    cost_mat = torch.cdist(pred_source, pred_target, p=2)
    std_cost, mean_cost = torch.std_mean(cost_mat)
    thresh = mean_cost - std_cost
    filter_mat = cost_mat < thresh
    filt_ent = torch.sum(ot_mat * filter_mat * y_dist) / torch.sum(ot_mat * filter_mat)
    filt_acc = torch.sum(ot_mat * filter_mat * y_acc) / torch.sum(ot_mat * filter_mat)
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
    mean_source_grad_norm = np.mean(grad_source_norms)
    mean_ot_grad_norm = np.mean(grad_ot_norms)
    print(
        f"Grad norms avg. (source/OT/WRR): {mean_source_grad_norm}/{mean_ot_grad_norm}/{np.mean(grad_total_norms)}"
    )

    angles = torch.zeros(len(grad_total_norms))
    for i in range(len(grad_total_norms)):
        inner_prod = torch.sum(grad_source[i] * grad_ot[i]) / (
            grad_source_norms[i] * grad_ot_norms[i]
        )
        angles[i] = torch.acos(inner_prod)
    mean_angle = torch.mean(angles)
    print(f"Avg. angle between grads : {mean_angle * 180.0 / torch.pi}")
    return mean_source_grad_norm, mean_ot_grad_norm, mean_angle


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


def calc_weighted_wrr(
    model, fabric, loss_fun, f_source, f_target, y_source, reg, method="mm"
):
    num_target = f_target.shape[0]
    # loss matrix
    w_target = torch.ones(num_target, device=fabric.device) / num_target
    cost_mat = torch.cdist(f_source, f_target, 2)
    source_losses = loss_fun(f_source, y_source, reduction="none")
    cost_mat = cost_mat + source_losses[:, None]

    # ot_mat = torch.softmax(-cost_mat / reg, dim=0) * w_target[None, :]
    num_source = y_source.shape[0]
    w_source = torch.ones(num_source, device=fabric.device) / num_source
    reg_m = (1.0, 100.0)
    if method == "mm":
        ot_mat = ot.unbalanced.mm_unbalanced(
            w_source, w_target, cost_mat, reg_m, div="kl", numItermax=1000
        )
    else:
        print("Using sinkhorn unbalanced...")
        ot_mat = ot.sinkhorn_unbalanced(
            w_source, w_target, cost_mat, reg, reg_m, method="sinkhorn_stabilized"
        )

    loss = torch.sum(ot_mat * cost_mat)
    w_source = torch.sum(ot_mat, dim=1)
    w_source_loss = torch.sum(w_source * source_losses)
    return loss, ot_mat, w_source, w_source_loss


def calc_ot(f_source, f_target, fabric, reg=1e-6, use_geomloss=False):
    num_source = f_source.shape[0]
    num_target = f_target.shape[0]

    ### Geomloss
    if use_geomloss is True:
        ot_loss = geomloss.SamplesLoss(loss="sinkhorn", p=1, blur=reg)
        cost = ot_loss(f_source, f_target)
    else:
        # The problem with using POT is that Sinkhorn is not converging for low reg.
        # and it doesn't seem possible to get a numerically accurate ot_mat from geomloss!
        w_source = torch.ones(num_source, device=fabric.device) / num_source
        num_target = f_target.shape[0]
        w_target = torch.ones(num_target, device=fabric.device) / num_target
        cost_mat = torch.cdist(f_source, f_target, p=2)
        ot_mat = ot.emd(w_source, w_target, cost_mat, numItermax=5000)
        cost = torch.sum(ot_mat * cost_mat)
    return cost, ot_mat
