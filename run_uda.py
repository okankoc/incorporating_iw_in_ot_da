"""Script for testing unsupervised domain adaptation algorithms in several different distribution shift scenarios."""
import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from lightning import Fabric

# Code from this repo
import utils
import shifts
import adapt
import loss
from debug import debug_method
from config.local_config import setup_local_config
from config.cluster_config import setup_cluster_config
from load_model import load_model, init_model, pretrain_model


def reset_all(seed):
    # Python & NumPy
    np.random.seed(seed)

    # PyTorch CPU/CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism (optional but recommended for fair comps)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_algorithm(config, name, model, loss_fun, opt, scenario, fabric):
    # Prepare adaptation methods
    if name == "wrr":
        alg = adapt.wrr.WRR(config["wrr"], fabric, model, loss_fun, opt)
    if name == "weighted_wrr":
        alg = adapt.weighted_wrr.WeightedWRR(config["weighted_wrr"], fabric, model, loss_fun, opt)
    if name == "cons_wrr":
        alg = adapt.constrained_wrr.ConstrainedWRR(config["cons_wrr"], fabric, model, loss_fun, opt)
    if name == "lje":
        alg = adapt.oracle.OracleLJE(fabric, model, loss_fun, opt)
    if name == "cc":
        alg = adapt.oracle.OracleCC(config["cc"], fabric, model, loss_fun, opt)
    if name == "erm":
        alg = adapt.erm.ERM(model, fabric, loss_fun, opt)
    if name == "dann":
        alg = adapt.dann.DANN(config["dann"], fabric, model, loss_fun, scenario)
    if name == "fdal":
        alg = adapt.fdal.FDAL(config["fdal"], fabric, model, loss_fun, scenario)
    if name == "reverse_kl":
        alg = adapt.reverse_kl.ReverseKL(config["reverse_kl"], fabric, model, loss_fun, opt)
        model = alg.model # model is probabilistic representation network in reverse-kl case
    return alg, model


def run_uda(config, fabric):
    methods = config["algs"]
    num_methods = len(methods)
    num_epochs = config["num_epochs"]
    num_runs = config["num_runs"]
    results = torch.zeros(
        num_methods, num_runs, num_epochs+1, 4)

    # Run adaptation
    scenario = shifts.init_scenario(config, fabric)
    model = init_model(config, scenario)
    scenario = setup_fabric_dataloaders(fabric, scenario)
    loss_fun = init_loss(config)
    opt = init_opt(config, model)
    results = pretrain_model(model, config, fabric, scenario, loss_fun, opt, results)

    # saving loss_source, acc_source, loss_target, acc_target
    for i in range(num_methods):
        for j in range(num_runs):
            reset_all(seed=j)
            model = load_model(config, fabric, scenario)
            loss_fun = init_loss(config)
            opt = init_opt(config, model)
            alg, model = init_algorithm(config, methods[i], model, loss_fun, opt, scenario, fabric)
            print("===============================")
            print(f"Algorithm {alg.name}, run number: {j}")
            for epoch in range(num_epochs):
                batch_idx = 0
                print(f"Epoch {epoch+1}")
                for (X_train, y_train), (X_shift, y_shift) in zip(
                    scenario.source_dataloader, scenario.target_dataloader
                ):
                    y_train = utils.one_hot(y_train, scenario.num_classes)
                    y_shift = utils.one_hot(y_shift, scenario.num_classes)
                    if batch_idx % 10 == 0:
                        print(f"Batch id: {batch_idx}")
                    alg.adapt(model, fabric, X_train, y_train, X_shift, y_shift)
                    if config["debug"] is True and (
                        batch_idx % config["print_every_n"] == 0
                    ):
                        debug_method(
                            config,
                            alg,
                            model,
                            loss_fun,
                            scenario,
                            fabric,
                            batch_idx / config["print_every_n"],
                        )
                    batch_idx += 1

                print("===============================")
                print(f"Algorithm {alg.name}")
                results[i, j, epoch+1, :] = utils.report_metrics(
                    scenario,
                    model,
                    loss_fun,
                    config["report_source_train_risk"],
                    config["report_target_train_risk"],
                    fabric,
                )
    return results


def init_loss(config):
    if config["loss"] == "margin":
        loss_fun = loss.MarginLoss()
    elif config["loss"] == "euclidean":
        loss_fun = loss.EuclideanLoss()
    elif config["loss"] == "cross-entropy":
        loss_fun = loss.CELoss()
    else:
        raise Exception("Loss function not implemented!")
    return loss_fun


# For now we assume that all algorithms share the optimizer, but we can change that later
def init_opt(config, model):
    if config["optimizer"] == "adam":
        opt = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
    elif config["optimizer"] == "sgd":
        opt = torch.optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )
    else:
        raise Exception("Unknown optimizer!")
    return opt


def setup_fabric_dataloaders(fabric, scenario):
    scenario.source_dataloader = fabric.setup_dataloaders(scenario.source_dataloader)
    scenario.target_dataloader = fabric.setup_dataloaders(scenario.target_dataloader)
    scenario.source_test_dataloader = fabric.setup_dataloaders(
        scenario.source_test_dataloader
    )
    scenario.target_test_dataloader = fabric.setup_dataloaders(
        scenario.target_test_dataloader
    )
    return scenario


def save_results(results, config):
    os.makedirs("results", exist_ok=True)
    model_name = config["model"]
    scenario_name = config["scenario"]
    methods = config["algs"]
    num_methods, num_runs, num_epochs, _ = results.shape
    metrics = ["loss_source", "acc_source", "loss_target", "acc_target"]

    data = {'config': config}
    for m, metric in enumerate(metrics):
        data[metric] = {}
        save_name = "results/" + scenario_name + "_" + model_name + "_" + metric
        fig, ax = plt.subplots()
        for i, method in enumerate(methods):
            data[metric][method] = results[i, :, :, m]
            stds, means = torch.std_mean(results[i, :, :, m], dim=0)
            ax.errorbar(x=np.arange(num_epochs), y=means, yerr=stds, label=method)
        ax.legend(loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.savefig(save_name + ".pdf", format="pdf")
    torch.save(data, "results/" + scenario_name + "_" + model_name + ".pth")


def run_on_cluster():
    config = setup_cluster_config()
    fabric = Fabric(accelerator=config["device"], devices="auto", strategy="auto")
    fabric.launch()

    if fabric.global_rank != 0:
        import builtins
        builtins.print = lambda *args, **kwargs: None

    print(f"Fabric device: {fabric.device}")
    reset_all(seed=0)

    # Run all experiments
    models = ["ConvNet", "ResNet"]
    dann_layers = ["flatten", -1] # ignored for ResNet
    scenarios = ['MNIST_TO_USPS', 'USPS_TO_MNIST', 'MNIST_TO_MNIST_M', 'SVHN_TO_MNIST']
    for i, model in enumerate(models):
        if model == "ResNet":
            config["optimizer"] = "sgd"
            config["learning_rate"] = 1e-4
        for scenario in scenarios:
            config["model"] = model
            config["scenario"] = scenario
            config["dann"]["layer_to_apply_disc"] = dann_layers[i]
            res = run_uda(config, fabric)
            save_results(res, config)


def run_on_local():
    config = setup_local_config()
    fabric = Fabric(accelerator=config["device"], devices="auto", strategy="auto")
    fabric.launch()

    if fabric.global_rank != 0:
        import builtins

        builtins.print = lambda *args, **kwargs: None

    print(f"Fabric device: {fabric.device}")
    reset_all(seed=0)

    res = run_uda(config, fabric)
    save_results(res, config)



if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    # torch.set_printoptions(precision=3, sci_mode=False)
    # torch.autograd.set_detect_anomaly(True)

    run_on_local()
    # run_on_cluster()
