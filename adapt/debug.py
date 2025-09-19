import torch
import ot
import geomloss


# Debugging by printing Wasserstein-based bounds for all methods!
def debug_model(config, model, loss_fun, fabric, X_source, y_source, X_target, y_target):
    pred_source = model(X_source)
    pred_target = model(X_target)
    target_loss = loss_fun(pred_target, y_target)
    print(f"target_loss: {target_loss.item()}")
    if config['calc_wrr'] == True:
        source_loss = loss_fun(pred_source, y_source)
        ot_cost, ot_mat = calc_ot(pred_source, pred_target, fabric)
        wrr = source_loss + ot_cost
        print(f"WRR: {wrr.item()}, ot_cost: {ot_cost.item()}, source_loss: {source_loss.item()}")
    if config['calc_weighted_wrr'] == True:
        w_wrr, w_ot_mat, w_source_loss = calc_weighted_wrr(model, fabric, loss_fun, pred_source, pred_target, y_source, config['wrr_entropy_reg'])
        w_ot_cost = w_wrr - w_source_loss
        print(f"W-WRR: {w_wrr.item()}, w_ot_cost: {w_ot_cost.item()}, w_source_loss: {w_source_loss.item()}")
    if config['calc_entanglement'] == True:
        y_dist = torch.cdist(y_source, y_target)
        entanglement = torch.sum(ot_mat * y_dist)
        weighted_entanglement = torch.sum(w_ot_mat * y_dist)
        print(f"Entanglement: {entanglement} Weighted entanglement: {weighted_entanglement}")
        y_acc = (y_dist == 0).to(torch.float)
        print(f"OT acc: {torch.sum(ot_mat * y_acc)}, Weighted OT acc: {torch.sum(w_ot_mat * y_acc)}")
    if config['calc_margin'] == True:
        std_margin, avg_margin = calc_margin(pred_source, y_source)
        print(f"Source margin mean: {avg_margin}, std: {std_margin}")
        std_margin, avg_margin = calc_margin(pred_target, y_target)
        print(f"Target margin mean: {avg_margin}, std: {std_margin}")
    if config['calc_grad_info'] == True:
        num_layers = len(model.net)
        model.zero_grad()
        source_loss = loss_fun(pred_source, y_source)
        fabric.backward(source_loss, retain_graph=True)
        grad_source_norms = torch.zeros(num_layers)
        grad_source = []
        for i, w in enumerate(model.parameters()):
            grad_source.append(w.grad)
            grad_source_norms[i] = torch.linalg.vector_norm(w.grad, ord=2, dim=None).item()
        print(f"Source loss grad norms avg.: {torch.mean(grad_source_norms)}")
        model.zero_grad()
        ot_cost, _ = calc_ot(pred_source, pred_target, fabric)
        fabric.backward(ot_cost)
        grad_ot_norms = torch.zeros(num_layers)
        grad_ot = []
        for i, w in enumerate(model.parameters()):
            grad_ot.append(w.grad)
            grad_ot_norms[i] = torch.linalg.vector_norm(w.grad, ord=2, dim=None).item()
        print(f"OT cost grad norms avg.: {torch.mean(grad_ot_norms)}")
        angles = torch.zeros(num_layers)
        for i, w in enumerate(model.parameters()):
            inner_prod = torch.sum(grad_source[i] * grad_ot[i]) / (grad_source_norms[i] * grad_ot_norms[i])
            angles[i] = torch.acos(inner_prod)
        print(f"Avg. angle between grads : {torch.mean(angles) * 180.0 / torch.pi}")


def calc_margin(preds, labels):
    # Get correct points with matching labels
    # Find the gap between max and second max (by sorting for now)
    pred_sorted_val, pred_sorted_ind = torch.sort(preds, dim=1, descending=True)
    correct = (pred_sorted_ind[:, 0] == labels.argmax(1))
    margin = pred_sorted_val[correct, 0] - pred_sorted_val[correct, 1]
    return torch.std_mean(margin)


def calc_weighted_wrr(model, fabric, loss_fun, f_source, f_target, y_source, reg):
    num_target = f_target.shape[0]
    source_loss = loss_fun(f_source, y_source)
    # loss matrix
    w_target = torch.ones(num_target, device=fabric.device) / num_target
    cost_mat = torch.cdist(f_source, f_target, 2)
    source_losses = loss_fun(f_source, y_source, reduction='none')
    cost_mat += cost_mat + source_losses[:, None]
    ot_mat = torch.softmax(-cost_mat / reg, dim=0) * w_target[None, :]
    loss = torch.sum(ot_mat * cost_mat)
    w_source = torch.sum(ot_mat, dim=1)
    w_source_loss = torch.sum(w_source * source_losses)
    return loss, ot_mat, w_source_loss


def calc_ot(f_source, f_target, fabric, reg=1e-6):
    num_source = f_source.shape[0]
    num_target = f_target.shape[0]

    ### Python crashes regularly with POT so switching to GeomLoss
    ot_loss = geomloss.SamplesLoss(loss="sinkhorn", p=1, blur=reg)
    cost = ot_loss(f_source, f_target)

    # The problem with using POT is that Sinkhorn is not converging for low reg.
    # and it doesn't seem possible to get a numerically accurate ot_mat from geomloss!
    w_source = torch.ones(num_source, device=fabric.device) / num_source
    num_target = f_target.shape[0]
    w_target = torch.ones(num_target, device=fabric.device) / num_target
    cost_mat = torch.cdist(f_source, f_target, p=2)
    ot_mat = ot.emd(w_source, w_target, cost_mat, numItermax=5000)

    return cost, ot_mat
