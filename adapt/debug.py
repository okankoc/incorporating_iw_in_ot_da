import torch
import ot
import geomloss


# Debugging by printing Wasserstein-based bounds for all methods!
def debug_model(config, model, loss_fun, fabric, X_source, y_source, X_target, y_target):
    pred_source = model(X_source)
    pred_target = model(X_target)
    source_loss = loss_fun(pred_source, y_source)

    num_source = pred_source.shape[0]
    w_source = torch.ones(num_source, device=fabric.device) / num_source
    num_target = pred_target.shape[0]
    w_target = torch.ones(num_target, device=fabric.device) / num_target
    # The problem with using POT is that Sinkhorn is not converging for low reg.
    cost_mat = torch.cdist(pred_source, pred_target)
    ot_mat = ot.emd(w_source, w_target, cost_mat, numItermax=5000)
    ot_cost = calc_ot(pred_source, pred_target)
    loss = source_loss + ot_cost

    print(f"WRR: {loss.item()}, ot_cost: {ot_cost.item()}, source_loss: {source_loss.item()}")

    if config['calc_entanglement'] == True:
        entanglement = torch.sum(ot_mat * torch.cdist(y_source, y_target))
        print(f"Entanglement: {entanglement}")
    if config['calc_margin'] == True:
        # Get correct points with matching labels
        # Find the gap between max and second max (by sorting for now)
        pred_sorted_val, pred_sorted_ind = torch.sort(pred_source, dim=1, descending=True)
        correct = (pred_sorted_ind[:, 0] == y_source.argmax(1))
        margin = pred_sorted_val[correct, 0] - pred_sorted_val[correct, 1]
        std_margin, avg_margin = torch.std_mean(margin)
        print(f"Margin mean: {avg_margin}, std: {std_margin}")
    if config['calc_grad_norms'] == True:
        grad_norms = []
        for w in model.parameters():
            grad_norms.append(torch.linalg.vector_norm(w.grad, ord=2, dim=None).item())
        print(f"Grad norms: {grad_norms}")


def calc_ot(f_source, f_target, reg=1e-6):
    num_source = f_source.shape[0]
    num_target = f_target.shape[0]

    ### Python crashes regularly with POT so switching to GeomLoss
    ot_loss = geomloss.SamplesLoss(loss="sinkhorn", p=1, blur=reg)
    cost = ot_loss(f_source, f_target)
    return cost
