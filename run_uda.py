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


def get_methods(config, model, loss_fun, fabric):
    # Prepare adaptation methods
    methods = []
    if 'wrr' in config['algs']:
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
                debug=False)
        )
    elif 'weighted_wrr' in config['algs']:
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


def run_uda(config, methods, model, scenario, loss_fun, num_epochs, fabric):
    # Run adaptation
    for method in methods:
        print("===============================")
        print(f"Method {method.name}")
        reset_all(seed=1)
        for epoch in range(config['num_epochs']):
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


def init_scenario(config, fabric):
    if config['scenario'] == 'MNIST_to_USPS':
        scenario = shifts.MNIST_to_USPS(dataloader_options, use_sampler=True, class_balanced=config['class_balanced'])
    elif config['scenario'] == 'USPS_to_MNIST':
        scenario = shifts.USPS_to_MNIST(dataloader_options, use_sampler=True)
    elif config['scenario'] == 'MNIST_to_MNIST_M':
        scenario = shifts.MNIST_to_MNIST_M(dataloader_options, preprocess=False)
    elif config['scenario'] == 'SVHN_to_MNIST':
        scenario = shifts.SVHN_to_MNIST(dataloader_options, class_balanced=config['class_balanced'])
    elif config['scenario'] == 'CIFAR10C':
        scenario = shifts.CIFAR_CORRUPT(dataloader_options, corruptions=["fog", "frost", "snow"])
    elif config['scenario'] == 'PORTRAITS':
        scenario = shifts.PORTRAITS(dataloader_options, size=(32,32), train_ratio=0.8)
    elif config['scenario'] == 'OFFICEHOME':
        scenario = shifts.OFFICEHOME(dataloader_options, size=(224,224))
    else:
        raise Exception('Unknown scenario')
    scenario.source_dataloader = fabric.setup_dataloaders(scenario.source_dataloader)
    scenario.target_dataloader = fabric.setup_dataloaders(scenario.target_dataloader)
    scenario.source_test_dataloader = fabric.setup_dataloaders(scenario.source_test_dataloader)
    scenario.target_test_dataloader = fabric.setup_dataloaders(scenario.target_test_dataloader)
    return scenario


def init_model(config, scenario):
    if config['model'] == 'MLP':
        model = MLP(layer_sizes=[scenario.input_size, 200, 100, scenario.num_classes], f_nonlinear=nn.ReLU())
    elif config['model'] == 'ConvNet':
        model = ConvNet(num_classes=scenario.num_classes)
    elif config['model'] == 'ConvNet2':
        model = ConvNet2(num_classes=scenario.num_classes)
    elif config['model'] == 'LeNet':
        model = LeNet(num_classes=scenario.num_classes)
    elif config['model'] == 'SmallCNN':
        model = SmallCNN(num_classes=scenario.num_classes)
    elif config['model'] == 'ResNet':
        model = init_resnet(config['resnet_size'], scenario.num_classes)
    return model


def init_resnet(size, num_classes):
    if size == 18:
        model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        model.name = "RESNET18"
    elif size == 50:
        model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        model.name = "RESNET50"
    else:
        raise Exception('Resnet size not allowed!')
    model.num_classes = num_classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

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


def run_uda_experiments(fabric, config):
    loss_fun = config['loss']
    scenario = init_scenario(config, fabric)
    model = init_model(config, scenario)
    if config['pretrain'] is True:
        model = utils.train_model_on_source(model, loss_fun, scenario, num_epochs=config['num_pretrain_epochs'], fabric=fabric)
    else:
        fabric.setup(model)
    model.save_params()
    # Report initial performance of a loaded source-trained model
    utils.report_acc(scenario, model, loss_fun)
    methods = get_methods(config, model, loss_fun, fabric)
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
    fabric = Fabric(accelerator="cpu", devices="auto", strategy="auto")
    fabric.launch()
    print(f"Fabric device: {fabric.device}")
    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(precision=4, sci_mode=False)

    config = {
        # Experiment details
        seed: 1,

        # Model and optimizer (MLP, ConvNet, ConvNet2, LeNet, SmallCNN, ResNet)
        model: 'MLP',
        resnet_size: 18, # 18 or 50
        pretrain: False
        num_pretrain_epochs: 1 # if pretrain is True
        loss: EuclideanLoss, # nn.CrossEntropyLoss()
        optimizer: 'adam', # alternative: sgd
        learning_rate: 1e-3, # use 1e-4 for ResNets or a learning scheduler
        num_epochs: 1,

        # Data loader options
        batch_size: 64,
        shuffle: False,
        drop_last: True,

        # Distribution shift scenario (MNIST_to_USPS, CIFAR10C, ...)
        scenario: 'MNIST_to_USPS',
        class_balanced: False,

        # Algorithms to compare against
        algs: ['wrr', 'weighted_wrr']
    }

    reset_all(seed=config['seed'])
    run_uda_experiments(fabric, config)
