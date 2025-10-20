import numpy as np
import torch
import ot
import geomloss
from scipy.spatial.distance import cdist
import matplotlib
import matplotlib.pyplot as plt

from importance_weighting import apply_importance_weighting as apply_iw

matplotlib.use(backend="QtAgg", force=True)


def regression_fun_1d(x):
    # Watch out for values close to 0
    return -(torch.sin(torch.pi * x) / (torch.pi * x))


def gen_gauss_covariates(mean, var, num_samples):
    if var.numel() == 1:
        return mean + torch.sqrt(var) * torch.randn(num_samples)
    dim = mean.size
    return mean[:, torch.newaxis] + torch.linalg.cholesky(var) @ torch.random.randn(
        dim, num_samples
    )


def gen_labels(fun, x, noise_var):
    num_pts = x.shape[0]
    y_act = fun(x)
    y_noise = y_act + torch.sqrt(noise_var) * torch.randn(num_pts)
    return (torch.tensor(y_noise), torch.tensor(y_act))


def opt_wrr_rule_with_sgd(
    theta0,
    X_source,
    X_target,
    y_source,
    method="emd",
    blur=1e-4,
    weight=False,
    use_geom=False,
    thresh=1e-3,
    report_every=10,
):
    # Optimize the Wasserstein-regularized Risk (WRR)
    theta = torch.tensor(theta0, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=1e-3)
    loss = torch.nn.MSELoss(reduction="none")
    num_source = X_source.shape[0]
    num_target = X_target.shape[0]
    w_source = torch.ones(num_source) / num_source
    w_target = torch.ones(num_target) / num_target
    ot_loss = geomloss.SamplesLoss(loss="sinkhorn", p=1, blur=blur)

    print("============ SGD ==========")
    prev_loss = 0.0
    total_loss = 1.0
    iter_idx = 1
    while abs(total_loss - prev_loss) > thresh:
        prev_loss = total_loss
        pred_source = X_source @ theta[:, torch.newaxis]
        pred_target = X_target @ theta[:, torch.newaxis]
        cost_mat = ot.utils.euclidean_distances(pred_source, pred_target, squared=False)
        if weight is False:
            source_loss = loss(pred_source[:, 0], y_source) @ w_source
            if use_geom is True:
                ot_dist = ot_loss(w_source, pred_source, w_target, pred_target)
            else:
                if method == "sinkhorn":
                    ot_dist = ot.sinkhorn2(w_source, w_target, cost_mat, blur)
                elif method == "emd":
                    ot_dist = ot.emd2(w_source, w_target, cost_mat)
        else:
            source_loss, ot_dist = compute_weighted_wrr(
                w_source, loss, pred_source, pred_target, y_source
            )
        total_loss = source_loss + ot_dist
        total_loss.backward()
        opt.step()
        opt.zero_grad()
        # print(f"Standard OT cost: {ot_dist}")
        if iter_idx % report_every == 0:
            print(
                f"Total loss: {total_loss}, source loss: {source_loss}, ot dist: {ot_dist}, theta: {theta}"
            )
        iter_idx += 1
    return (theta, X_target @ theta)


def opt_wrr_rule_iter_ls(theta0, X_source, X_target, y_source, method="emd", blur=1e-4):
    theta = theta0
    X_pq = torch.cat((X_source, X_target))
    num_source = X_source.shape[0]
    num_target = X_target.shape[0]
    w_source = torch.ones(num_source) / num_source
    w_target = torch.ones(num_target) / num_target
    loss = torch.nn.MSELoss()
    print("============ ITER LS ==========")
    for j in range(11):
        pred_source = X_source @ theta
        pred_target = X_target @ theta
        cost_mat = torch.tensor(
            cdist(
                pred_source[:, torch.newaxis].detach(),
                pred_target[:, torch.newaxis].detach(),
                metric="sqeuclidean",
            )
        )
        if method == "sinkhorn":
            Gamma = ot.sinkhorn(w_source, w_target, cost_mat, blur)
        elif method == "emd":
            Gamma = ot.emd(w_source, w_target, cost_mat)
        ot_cost = torch.sqrt(torch.sum(Gamma * cost_mat))
        M_pq = torch.cat(
            (
                torch.cat((torch.diag(w_source), -Gamma), dim=1),
                torch.cat((-Gamma.T, torch.diag(w_target)), dim=1),
            ),
            dim=0,
        )
        R = (X_pq.T @ M_pq @ X_pq) / (2 * ot_cost)
        theta = torch.linalg.solve(X_source.T @ X_source + R, X_source.T @ y_source)
        source_loss = loss(pred_source, y_source)
        total_loss = source_loss + ot_cost
        print(
            f"Total loss: {total_loss}, source loss: {source_loss}, ot dist: {ot_cost}, theta: {theta}"
        )
    return (theta, X_target @ theta)


def test_that_wrr_rule_can_be_optimized_in_linear_regression():
    """
    Check WRR performance with W1-distance on Euclidean distance loss. Optimize using gradient descent
    Check above using squared distance. Optimize a) with gradient descent, b) explicitly and compare
    """

    torch.manual_seed(0)
    num_source = 150
    s2_noise = torch.tensor([1 / 16])
    x_source = gen_gauss_covariates(
        mean=torch.tensor([1.0]), var=torch.tensor([1 / 4.0]), num_samples=num_source
    )
    y_source, y_source_clean = gen_labels(regression_fun_1d, x_source, s2_noise)

    X_source = torch.vstack((torch.ones((num_source)), x_source)).T
    theta_ls, res, _, _ = torch.linalg.lstsq(X_source, y_source)
    pred_source_ls = X_source @ theta_ls

    # generate test samples
    num_target = 50
    x_target = gen_gauss_covariates(
        mean=torch.tensor([2.0]), var=torch.tensor([1 / 16.0]), num_samples=num_target
    )
    X_target = torch.vstack((torch.ones(num_target), x_target)).T
    pred_target_ls = X_target @ theta_ls

    theta_wrr_sgd, pred_target_wrr_sgd = opt_wrr_rule_with_sgd(
        theta_ls,
        X_source,
        X_target,
        y_source,
        method="sinkhorn",
        blur=1e-4,
        weight=False,
        use_geom=True,
        thresh=1e-4,
        report_every=10,
    )
    # TODO: Why in sinkhorn we get a jumping phenomenon?
    theta_wrr_iter_ls, pred_target_wrr_iter_ls = opt_wrr_rule_iter_ls(
        theta_ls, X_source, X_target, y_source, method="sinkhorn", blur=1e-4
    )

    y_target, _ = gen_labels(regression_fun_1d, x_target, s2_noise)
    loss = torch.nn.MSELoss()
    risk_target_ls = loss(pred_target_ls, y_target)
    risk_target_wrr_sgd = loss(pred_target_wrr_sgd, y_target)
    risk_target_wrr_iter_ls = loss(pred_target_wrr_iter_ls, y_target)

    print(f"LS Test risk: {risk_target_ls}")
    print(f"WRR-sgd Test risk: {risk_target_wrr_sgd}")
    print(f"WRR-iter-ls Test risk: {risk_target_wrr_iter_ls}")

    plt.scatter(x_source, y_source, c="red", s=15)
    plt.scatter(x_source, y_source_clean, c="black", s=15)
    plt.scatter(x_source, pred_source_ls, c="blue", s=15)
    plt.scatter(x_target, pred_target_wrr_sgd.detach().numpy(), c="brown", s=15)
    plt.scatter(x_target, pred_target_wrr_iter_ls.detach().numpy(), c="purple", s=15)
    plt.legend(["noisy labels", "actual labels", "lstsq", "WRR-sgd", "WRR-iter-ls"])
    plt.show()


def test_that_dual_variables_in_OT_correspond_to_effect_of_IW():
    num_x = 10
    num_y = 10
    dim = 2
    torch.manual_seed(1)
    x = torch.randn(num_x, dim)
    y = torch.randn(num_y, dim)
    a = torch.rand(num_x)
    a /= torch.sum(a)
    a.requires_grad = True
    b = torch.rand(num_y)
    b /= torch.sum(b)
    p = 1
    blur = 0.001
    ot_loss = geomloss.SamplesLoss(loss="sinkhorn", p=p, blur=blur)
    # ot_cost = ot_loss(a, x, b, y)
    # ot_cost.backward()

    cost_mat = torch.tensor(cdist(x, y, metric="euclidean"))
    ot_cost = ot.emd2(a, b, cost_mat)
    ot_cost.backward()

    # Modify p randomly using a vector theta that sums up to 1
    da = torch.randn(num_x)
    da -= da.mean()
    lamb = 0.001
    a2 = a + lamb * da

    diff_auto = torch.dot(a.grad, a2 - a)
    print(f"Auto-diff of OT costs: {diff_auto}")

    # Get dual variables using geomloss
    potentials = geomloss.SamplesLoss(loss="sinkhorn", p=p, blur=blur, potentials=True)
    f, _ = potentials(a, x, b, y)
    f[0] -= f[0].mean()
    print(f"Potential from geomloss: {f}")

    # Check dual variables using POT library
    # G, log = ot.sinkhorn(a, b, cost_mat, reg=0.001, log=True)
    # f = log['u']
    # f = torch.log(f)
    # f -= f.mean()
    # print(f"Potential from POT: {f}")

    # Get f and dot with theta to calculate the derivative
    diff_deriv = torch.dot(f[0], a2 - a)

    diff_deriv = diff_deriv.detach().numpy()
    diff_auto = diff_auto.detach().numpy()
    print(f"Potential based derivative of OT costs: {diff_deriv}")

    ot_cost2 = ot.emd2(a2, b, cost_mat)
    # ot_cost2 = ot_loss(a2, x, b, y)
    diff_exact = ot_cost2 - ot_cost
    diff_exact = diff_exact.detach().numpy()
    print(f"Exact difference of OT costs: {diff_exact}")

    # TODO: If we compute the total derivative it should match with diff_auto!
    # assert np.allclose(diff_deriv, diff_auto, rtol=1e-2) is True
    assert np.allclose(diff_auto, diff_exact, rtol=1e-2) is True


def compute_weighted_wrr(
    w_source, loss_fun, pred_source, pred_target, y_source, num_iter=30, blur=1e-4
):
    # loss matrix
    num_source = pred_source.shape[0]
    num_target = pred_target.shape[0]
    w_target = torch.ones(num_target) / num_target

    # Now minimize the OT using weights
    w_source.requires_grad = True
    opt = torch.optim.SGD([w_source], lr=1e-3, momentum=0.98)
    potentials = geomloss.SamplesLoss(loss="sinkhorn", p=1, blur=blur, potentials=True)
    losses = loss_fun(pred_source[:, 0], y_source)
    for i in range(num_iter):
        f, g = potentials(w_source, pred_source, w_target, pred_target)
        f0 = f - f.mean()
        w_source.grad = f0[0] + losses
        opt.step()
        opt.zero_grad()
        w_source.requires_grad = False
        w_source[w_source < 0.0] = 0.0
        w_source /= w_source.sum()
        w_source.requires_grad = True
        source_loss = losses @ w_source
        ot_dist = f[0] @ w_source + g @ w_target
        print(
            f"WRR: {source_loss + ot_dist}, derivative_norm: {(f0[0] + losses).norm()}"
        )
    return source_loss, ot_dist


def compute_weighted_ot(x_source, x_target, use_geom=False, num_iter=30, blur=1e-4):
    # loss matrix
    num_source = x_source.shape[0]
    num_target = x_target.shape[0]
    a = torch.ones(num_source) / num_source
    b = torch.ones(num_target) / num_target
    if use_geom is True:
        ot_loss = geomloss.SamplesLoss(loss="sinkhorn", p=1, blur=blur)
        ot_cost = ot_loss(a, x_source, b, x_target)
        potentials = geomloss.SamplesLoss(
            loss="sinkhorn", p=1, blur=blur, potentials=True
        )
        a.requires_grad = True
    else:
        cost_mat = torch.tensor(cdist(x_source, x_target, metric="euclidean"))
        a.requires_grad = True
        ot_cost = ot.emd2(a, b, cost_mat)
    print(f"Standard OT cost: {ot_cost}")

    # Now minimize the OT using weights
    opt = torch.optim.SGD([a], lr=1e-3, momentum=0.98)
    for i in range(num_iter):
        if use_geom is True:
            f, _ = potentials(a, x_source, b, x_target)
            f -= f.mean()
            a.grad = f[0]
        opt.step()
        opt.zero_grad()
        a.requires_grad = False
        a[a < 0.0] = 0.0
        a /= a.sum()
        a.requires_grad = True
        if use_geom is True:
            ot_cost_weighted = ot_loss(a, x_source, b, x_target)
            print(
                f"Weighted OT cost: {ot_cost_weighted}, derivative_norm: {f[0].norm()}"
            )
        else:
            ot_cost_weighted = ot.emd2(a, b, cost_mat)
            ot_cost_weighted.backward()
            print(
                f"Weighted OT cost: {ot_cost_weighted}, derivative_norm: {a.grad.norm()}"
            )
    # ot_mat = ot.emd(a, b, cost_mat)
    # ot_cost = torch.sum(ot_mat * cost_mat)
    # print(f"Check: ot_cost: {ot_cost}")
    return ot_cost_weighted, a


def test_that_weighted_OT_can_be_computed():
    # Create two distributions
    # using Gaussians such that
    # the W1-distance is significant
    # but the weighted distance converges to zero
    # as the number of samples increase!
    import geomloss

    torch.manual_seed(0)
    dim = 2
    num_source = 60
    num_target = 50

    mu_source = torch.rand(1, dim)
    cov_source = 0.1 * torch.eye(dim)

    mu_target = torch.rand(1, dim)
    mat_rand = torch.randn(dim, dim)
    cov_target = 0.1 * mat_rand @ mat_rand.T

    x_source = (
        mu_source.repeat(num_source, 1)
        + torch.randn(num_source, dim) @ torch.linalg.cholesky_ex(cov_source)[0]
    )
    x_target = (
        mu_target.repeat(num_target, 1)
        + torch.randn(num_target, dim) @ torch.linalg.cholesky_ex(cov_target)[0]
    )

    print(f"mu_source = {mu_source}, mu_target = {mu_target}")
    ot_cost_weighted, w = compute_weighted_ot(
        x_source, x_target, use_geom=True, num_iter=100
    )
    print(ot_cost_weighted)

    # Check explicit solution
    cost_mat = torch.tensor(cdist(x_source, x_target, metric="euclidean"))
    epsilon = 1e-4
    q = torch.ones(num_target) / num_target
    ot_mat = torch.softmax(-cost_mat / epsilon, dim=0) * q[None, :]
    # Implicitly optimized weights
    w = torch.sum(ot_mat, dim=1)
    ot_cost_alt = torch.sum(ot_mat * cost_mat)

    print(ot_cost_alt)

    ######## --------------------------
    # Repeat the same for optimizing gamma and weights w for the WRR cost function
    torch.manual_seed(2)
    num_source = 150
    s2_noise = torch.tensor([1 / 16])
    x_source = gen_gauss_covariates(
        mean=torch.tensor([1.0]), var=torch.tensor([1 / 4.0]), num_samples=num_source
    )
    y_source, _ = gen_labels(regression_fun_1d, x_source, s2_noise)

    X_source = torch.vstack((torch.ones((num_source)), x_source)).T
    theta_ls, _, _, _ = torch.linalg.lstsq(X_source, y_source)
    pred_source = X_source @ theta_ls

    # generate test samples
    num_target = 50
    x_target = gen_gauss_covariates(
        mean=torch.tensor([1.5]), var=torch.tensor([1 / 16.0]), num_samples=num_target
    )
    X_target = torch.vstack((torch.ones(num_target), x_target)).T
    pred_target = X_target @ theta_ls

    # Optimize solution
    w_source = torch.ones(num_source) / num_source
    loss_fun = torch.nn.MSELoss(reduction="none")
    source_loss_w, ot_dist_w = compute_weighted_wrr(
        w_source,
        loss_fun,
        pred_source[:, torch.newaxis],
        pred_target[:, torch.newaxis],
        y_source,
        num_iter=200,
        blur=1e-4,
    )
    wrr_weighted = source_loss_w + ot_dist_w
    print(wrr_weighted)

    # Calculate explicit solution
    cost_mat = torch.tensor(
        cdist(
            pred_source[:, torch.newaxis],
            pred_target[:, torch.newaxis],
            metric="euclidean",
        )
    )
    losses = loss_fun(pred_source, y_source)
    cost_mat += losses[:, None]
    epsilon = 1e-4
    q = torch.ones(num_target) / num_target
    ot_mat = torch.softmax(-cost_mat / epsilon, dim=0) * q[None, :]
    # Implicitly optimized weights
    # w = torch.sum(ot_mat, dim=1)
    wrr_cost_weighted_alt = torch.sum(ot_mat * cost_mat)
    print(wrr_cost_weighted_alt)


def compare_weighted_wrr_opt_to_iw_in_linear_regression():
    torch.manual_seed(10)
    num_source = 150
    s2_noise = torch.tensor([1 / 16])
    x_source = gen_gauss_covariates(
        mean=torch.tensor([1.0]), var=torch.tensor([1 / 4.0]), num_samples=num_source
    )
    y_source, y_source_clean = gen_labels(regression_fun_1d, x_source, s2_noise)

    X_source = torch.vstack((torch.ones((num_source)), x_source)).T
    theta_ls, res, _, _ = torch.linalg.lstsq(X_source, y_source)
    pred_source_ls = X_source @ theta_ls

    # generate test samples
    num_target = 50
    x_target = gen_gauss_covariates(
        mean=torch.tensor([2.0]), var=torch.tensor([1 / 16.0]), num_samples=num_target
    )
    X_target = torch.vstack((torch.ones(num_target), x_target)).T
    pred_target_ls = X_target @ theta_ls

    theta_wrr, pred_target_wrr = opt_wrr_rule_with_sgd(
        theta_ls,
        X_source,
        X_target,
        y_source,
        method="sinkhorn",
        blur=1e-4,
        weight=True,
        use_geom=True,
        thresh=1e-4,
        report_every=10,
    )
    theta_iw, pred_target_iw = apply_iw(
        X_source, X_target, y_source, method="kmm", max_weight=10
    )

    loss = torch.nn.MSELoss()
    y_target, _ = gen_labels(regression_fun_1d, x_target, s2_noise)
    risk_target_ls = loss(pred_target_ls, y_target)
    risk_target_wrr = loss(pred_target_wrr, y_target)
    risk_target_iw = loss(pred_target_iw, y_target)

    print(f"LS Test risk: {risk_target_ls}")
    print(f"WRR Test risk: {risk_target_wrr}")
    print(f"IW Test risk: {risk_target_iw}")

    plt.scatter(x_source, y_source, c="red", s=15)
    plt.scatter(x_source, y_source_clean, c="black", s=15)
    plt.scatter(x_source, pred_source_ls, c="blue", s=15)
    plt.scatter(x_target, pred_target_wrr.detach().numpy(), c="brown", s=15)
    plt.scatter(x_target, pred_target_iw.detach().numpy(), c="purple", s=15)
    plt.legend(["noisy labels", "actual labels", "lstsq", "WRR", "IW"])
    plt.show()


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    # test_that_dual_variables_in_OT_correspond_to_effect_of_IW()
    test_that_weighted_OT_can_be_computed()
    # test_that_wrr_rule_can_be_optimized_in_linear_regression()
    # compare_weighted_wrr_opt_to_iw_in_linear_regression()
