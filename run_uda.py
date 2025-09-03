"""Script for testing unsupervised domain adaptation algorithms in several different distribution shift scenarios."""

import copy
import types
import numpy as np
import torch
import torchvision.models
import ot
import geomloss
from torch import nn
import matplotlib
import matplotlib.pyplot as plt
from lightning import Fabric

import utils
import shifts
from adapt import WRR
from models.conv import ConvNet, ConvNet2, LeNet, SmallCNN
from models.mlp import MultiLayerPerceptron as MLP

# Necessary in mac osx to be able close figures in emacs
matplotlib.use(backend="QtAgg", force=True)


def reset_all(seed):
    # Python & NumPy
    np.random.seed(seed)

    # PyTorch CPU/CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism (optional but recommended for fair comps)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()
        self.reduction = 'mean'

    def forward(self, x, y):
        losses = torch.sqrt(torch.sum((x - y) ** 2, dim=1))
        if self.reduction == 'mean':
            return torch.mean(losses)
        return losses


def get_methods(model, loss_fun, fabric):
    # Prepare adaptation methods
    methods = []
    methods.append(
        WRR(
            fabric,
            model,
            loss_fun,
            learning_rate=1e-4,
            weight=False,
            p=1,
            scale=1.0,
            reg=1e-2,
            debug=True)
    )
    methods.append(
        WRR(
            fabric,
            model,
            loss_fun,
            learning_rate=1e-4,
            weight=True,
            p=1,
            scale=1.0,
            reg=1e-2,
            debug=True))
    return methods


def run_uda(methods, model, scenario, loss_fun, num_epochs, fabric):
    # Run adaptation
    for method in methods:
        print("===============================")
        print(f"Method {method.name}")
        reset_all(seed=1)
        for epoch in range(num_epochs):
            batch_idx = 0
            print(f"Epoch {epoch+1}")
            for (X_train, y_train), (X_shift, y_shift) in zip(scenario.source_dataloader, scenario.target_dataloader):
                y_train = utils.one_hot(y_train, scenario.num_classes)
                y_shift = utils.one_hot(y_shift, scenario.num_classes)
                if batch_idx % 10 == 0:
                    print(f"Batch id: {batch_idx}")
                batch_idx += 1
                method.adapt(model, fabric, X_train, y_train, X_shift, y_shift)

            print("===============================")
            print(f"Method {method.name}")
            utils.report_acc(scenario, model, loss_fun)
        model.restore_params()


def init_scenario(dataloader_options, fabric):
    # scenario = shifts.MNIST_to_USPS(dataloader_options, use_sampler=True, class_balanced=False)
    # scenario = shifts.USPS_to_MNIST(dataloader_options, use_sampler=True)
    scenario = shifts.MNIST_to_MNIST_M(dataloader_options, preprocess=False)
    # scenario = shifts.SVHN_to_MNIST(dataloader_options, class_balanced=False)
    # scenario = shifts.CIFAR_CORRUPT(dataloader_options, corruptions=["fog", "frost", "snow"])
    # scenario = shifts.PORTRAITS(dataloader_options, size=(32,32), train_ratio=0.8)
    # scenario = shifts.OFFICEHOME(dataloader_options, size=(224,224))
    scenario.source_dataloader = fabric.setup_dataloaders(scenario.source_dataloader)
    scenario.target_dataloader = fabric.setup_dataloaders(scenario.target_dataloader)
    scenario.source_test_dataloader = fabric.setup_dataloaders(scenario.source_test_dataloader)
    scenario.target_test_dataloader = fabric.setup_dataloaders(scenario.target_test_dataloader)
    return scenario


def init_model(scenario):
    # model = MLP(layer_sizes=[scenario.input_size, 200, 100, scenario.num_classes], f_nonlinear=nn.ReLU())
    # model = ConvNet(num_classes=scenario.num_classes)
    # model = ConvNet2(num_classes=scenario.num_classes)
    # model = LeNet(num_classes=scenario.num_classes)
    # model = SmallCNN(num_classes=scenario.num_classes)
    # load_model(model, scenario)
    model = init_resnet(scenario.num_classes)
    return model


def init_resnet(num_classes):
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.num_classes = num_classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # for OfficeHomeDataset
    model.name = "RESNET18"

    @torch.no_grad()
    def save_params(model):
        model.state = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def restore_params(model):
        model.load_state_dict(model.state)
        return dict(model.named_parameters())

    model.save_params = types.MethodType(save_params, model)
    model.restore_params = types.MethodType(restore_params, model)
    return model


def run_uda_experiments(fabric, num_epochs, pretrain=False):
    loss_fun = EuclideanLoss()
    # loss_fun = nn.CrossEntropyLoss()
    dataloader_options = {"batch_size": 64, "shuffle": False, "drop_last": True}
    scenario = init_scenario(dataloader_options, fabric)
    model = init_model(scenario)
    if pretrain is True:
        model = utils.train_model_on_source(model, loss_fun, scenario, num_epochs=5, fabric=fabric)
    else:
        fabric.setup(model)
    model.save_params()
    # Report initial performance of a loaded source-trained model
    utils.report_acc(scenario, model, loss_fun)
    methods = get_methods(model, loss_fun, fabric)
    run_uda(methods, model, scenario, loss_fun, num_epochs, fabric)


def check_shared_support():
    loss_fun = nn.CrossEntropyLoss()
    device = "mps"
    dataloader_options = {"batch_size": 64, "shuffle": False, "drop_last": True}
    scenario = init_scenario(dataloader_options)
    model = init_model(scenario, device)
    model = utils.train_model_on_source(model, loss_fun, scenario, device, num_epochs=5)

    # Compute weighted OT and note the difference!
    num_batches_max = 5
    batch_id = 0
    for (X_train, y_train), (X_shift, y_shift) in zip(scenario.source_dataloader, scenario.target_dataloader):
        X_train, X_shift, y_train, y_shift = (
            X_train.to(device),
            X_shift.to(device),
            utils.one_hot(y_train.to(device), scenario.num_classes),
            utils.one_hot(y_shift.to(device), scenario.num_classes),
        )
        f_source = model(X_train)
        f_target = model(X_shift)
        ot_loss = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=1e-4)
        total_cost = ot_loss(f_source, f_target)
        print(f"Standard OT with cost {total_cost} on batch {batch_id}")
        # TODO: Reimplement this with explicit solution if needed!
        compute_weighted_ot(f_source, f_target, p=2, blur=1e-4, device=device)

        batch_id += 1
        if batch_id == num_batches_max:
            break


if __name__ == "__main__":
    fabric = Fabric(accelerator="auto", devices="auto", strategy="auto")
    fabric.launch()
    print(f"Fabric device: {fabric.device}")
    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(precision=4, sci_mode=False)
    reset_all(seed=1)
    run_uda_experiments(fabric, num_epochs=1)
