"""Script for testing unsupervised domain adaptation algorithms in several different distribution shift scenarios."""

import copy
import types
import numpy as np
import torch
import torchvision.models
import geomloss
from torch import nn
import matplotlib
import matplotlib.pyplot as plt
from lightning import Fabric

# Code from this repo
import utils
import shifts
from loss import MarginLoss, EuclideanLoss, CELoss
from adapt.wrr import WRR
from adapt.weighted_wrr import WeightedWRR
from adapt.constrained_wrr import ConstrainedWRR
from adapt.oracle import OracleLJE, OracleCC
from adapt.erm import ERM
from adapt.dann import DANN
from adapt.fdal import FDAL
from adapt.reverse_kl import ReverseKL
from adapt.debug import debug_model
from models.conv import ConvNet, ConvNet2, LeNet, SmallCNN, ConvDomainClassifier
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


def init_algorithm(config, name, model, loss_fun, opt, fabric):
    # Prepare adaptation methods
    if name == 'wrr':
        alg = WRR(config['wrr'], fabric, model, loss_fun, opt)
    if name == 'weighted_wrr':
        alg = WeightedWRR(config['weighted_wrr'], fabric, model, loss_fun, opt)
    if name == 'cons_wrr':
        alg = ConstrainedWRR(config['cons_wrr'], fabric, model, loss_fun, opt)
    if name == 'lje':
        alg = OracleLJE(model, fabric, loss_fun, opt)
    if name == 'cc':
        alg = OracleCC(config['cc'], fabric, model, loss_fun, opt)
    if name == 'erm':
        alg = ERM(model, fabric, loss_fun, opt)
    if name == 'dann':
        alg = DANN(config['dann'], fabric, model, loss_fun)
    if name == 'fdal':
        alg = FDAL(config['fdal'], fabric, model, loss_fun)
    if name == 'reverse-kl':
        alg = ReverseKL(config['reverse_kl'], fabric, model, loss_fun, opt)
    return alg


def run_uda(config, fabric):
    # Run adaptation
    loss_fun = config['loss']
    scenario = init_scenario(config, fabric)
    methods = config['algs']
    num_methods = len(methods)
    num_epochs = config['num_epochs']
    num_runs = config['num_runs']
    results = torch.zeros(num_methods, num_runs, num_epochs, 4) # saving loss_source, acc_source, loss_target, acc_target
    pretrain_model(config, fabric, scenario, loss_fun)
    for i in range(num_methods):
        for j in range(num_runs):
            model = init_model(config, fabric, scenario, loss_fun)
            opt = init_opt(config, model)
            alg = init_algorithm(config, methods[i], model, loss_fun, opt, fabric)
            print("===============================")
            print(f"Algorithm {alg.name}")
            reset_all(seed=j)
            for epoch in range(num_epochs):
                batch_idx = 0
                print(f"Epoch {epoch+1}")
                for (X_train, y_train), (X_shift, y_shift) in zip(scenario.source_dataloader, scenario.target_dataloader):
                    y_train = utils.one_hot(y_train, scenario.num_classes)
                    y_shift = utils.one_hot(y_shift, scenario.num_classes)
                    if batch_idx % 10 == 0:
                        print(f"Batch id: {batch_idx}")
                    alg.adapt(model, fabric, X_train, y_train, X_shift, y_shift)
                    if config['debug'] is True and (batch_idx % config['print_every_n'] == 0):
                        debug_method(config, alg, model, loss_fun, scenario, fabric, batch_idx / config['print_every_n'])
                    batch_idx += 1

                print("===============================")
                print(f"Algorithm {alg.name}")
                results[i, j, epoch, :] = utils.report_metrics(scenario, model, loss_fun, config['report_source_train_risk'], config['report_target_train_risk'])
    return results


def debug_method(config, method, model, loss_fun, scenario, fabric, idx_des):
    num_batches = len(scenario.source_test_dataloader.dataset) / config['test_batch_size']
    idx_des = idx_des % num_batches
    idx = 0
    for (X_train, y_train), (X_shift, y_shift) in zip(scenario.source_test_dataloader, scenario.target_test_dataloader):
        if idx == idx_des:
            print("============================================")
            print(f"Debugging/validating on {idx}'th test batch")
            y_train = utils.one_hot(y_train, scenario.num_classes)
            y_shift = utils.one_hot(y_shift, scenario.num_classes)
            debug_model(config['debug_options'], model, loss_fun, fabric, X_train, y_train, X_shift, y_shift)
            if config['validate'] is True:
                method.validate(config, model, fabric, X_train, y_train, X_shift)
            break
            print("============================================")
        idx += 1


def init_scenario(config, fabric):
    dataloader_options = {"batch_size": config['batch_size'], "shuffle": False, "drop_last": True}
    test_dataloader_options = {"batch_size": config['test_batch_size'], "shuffle": False, "drop_last": True}
    if config['scenario'] == 'MNIST_TO_USPS':
        scenario = shifts.MNIST_to_USPS(dataloader_options, test_dataloader_options, use_sampler=True, class_balanced=config['class_balanced'])
    elif config['scenario'] == 'USPS_TO_MNIST':
        scenario = shifts.USPS_to_MNIST(dataloader_options, test_dataloader_options, use_sampler=True)
    elif config['scenario'] == 'MNIST_TO_MNIST_M':
        scenario = shifts.MNIST_to_MNIST_M(dataloader_options, test_dataloader_options, preprocess=False)
    elif config['scenario'] == 'SVHN_TO_MNIST':
        scenario = shifts.SVHN_to_MNIST(dataloader_options, test_dataloader_options, class_balanced=config['class_balanced'])
    elif config['scenario'] == 'CIFAR10C':
        scenario = shifts.CIFAR_CORRUPT(dataloader_options, test_dataloader_options, corruptions=["fog", "frost", "snow"])
    elif config['scenario'] == 'PORTRAITS':
        scenario = shifts.PORTRAITS(dataloader_options, test_dataloader_options, size=(32,32), train_ratio=0.8)
    elif config['scenario'] == 'OFFICEHOME':
        scenario = shifts.OFFICEHOME(dataloader_options, test_dataloader_options, size=(224,224))
    else:
        raise Exception('Unknown scenario')
    scenario.source_dataloader = fabric.setup_dataloaders(scenario.source_dataloader)
    scenario.target_dataloader = fabric.setup_dataloaders(scenario.target_dataloader)
    scenario.source_test_dataloader = fabric.setup_dataloaders(scenario.source_test_dataloader)
    scenario.target_test_dataloader = fabric.setup_dataloaders(scenario.target_test_dataloader)
    return scenario


# For now we assume that all algorithms share the optimizer, but we can change that later
def init_opt(config, model):
    if config['optimizer'] == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])
    else:
        raise Exception('Unknown optimizer!')
    return opt


def init_lazy_modules(model, scenario):
    for X_source in scenario.source_dataloader:
        model(X_source[0])
        break


def pretrain_model(config, fabric, scenario, loss_fun):
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

    init_lazy_modules(model, scenario)

    print(f"Initialized model {model.name}")
    folder_path = "save_files/" + scenario.name + "/"
    save_path = folder_path + model.name + ".pth"
    try:
        # Load parameters from a file
        model.load_state_dict(torch.load(save_path, weights_only=True))
        print(f"Saved model found! Loading parameters from file: {save_path}")
        model = fabric.setup(model)
    except:
        print(f"Saved model NOT found!")
        if config['pretrain'] is True:
            opt = init_opt(config, model)
            model, opt = fabric.setup(model, opt)
            print(f"Pretraining {config['num_pretrain_epochs']} epochs...")
            if config['pretrain_on_both'] is True:
                print('========= DEBUG MODE ON: USING TARGET LABELS TO PRETRAIN LJE ORACLE MODEL =======')
                model = utils.train_model_on_source_and_target(config, model, loss_fun, scenario, opt, fabric)
            else:
                model = utils.train_model_on_source(config, model, loss_fun, scenario, opt, fabric)

    # Report initial performance
    utils.report_metrics(scenario, model, loss_fun, config['report_source_train_risk'], config['report_target_train_risk'])


def init_model(config, fabric, scenario, loss_fun):
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

    print(f"Initialized model {model.name}")
    folder_path = "save_files/" + scenario.name + "/"
    save_path = folder_path + model.name + ".pth"
    try:
        # Load parameters from a file
        model.load_state_dict(torch.load(save_path, weights_only=True))
        print(f"Saved model found! Loading parameters from file: {save_path}")
    except:
        print(f"Saved model NOT found!")

    model.train()
    if config['adapt_only_last_layer'] == True:
        num_layers = len(list(model.parameters()))
        for i, p in enumerate(model.parameters()):
            if i < num_layers - 1:
                p.requires_grad = False
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


def plot_results(results, config):
    os.makedirs("results", exist_ok=True)
    methods = config['algs']
    num_methods, num_runs, num_epochs, _ = results.shape
    metrics = ['loss_source', 'acc_source', 'loss_target', 'acc_target']

    for m, metric in enumerate(metrics):
        save_name = "results/" + scenario.name + "_" + model.name + "_" + metric
        fig, ax = plt.subplots()
        for i, method in enumerate(methods):
            stds, means = torch.std_mean(results[i, :, :, m], dim=0)
            ax.errorbar(x=np.arange(num_epochs), y=means, yerr=stds, label=method)
        ax.legend(loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.savefig(save_name + ".pdf", format="pdf")


def setup_config():
    config = {
        # Experiment details
        'device': 'auto', # 'cpu' or 'auto' to find gpu automatically

        # Model and optimizer (MLP, ConvNet, ConvNet2, LeNet, SmallCNN, ResNet)
        'model': 'ConvNet',
        'resnet_size': 18, # 18 or 50
        'pretrain': True,
        'num_pretrain_epochs': 5, # if pretrain is True
        'loss': MarginLoss(), #MarginLoss(), EuclideanLoss(), CELoss()
        'optimizer': 'adam', # alternatives: adam, sgd
        'learning_rate': 1e-3, # use 1e-4 for ResNets or a learning scheduler
        'momentum': 0.9, # for SGD
        'weight_decay': 0.0,

        # Data loader options
        'batch_size': 64,

        # Distribution shift scenario (MNIST_TO_USPS, CIFAR10C, ...)
        'scenario': 'MNIST_TO_USPS',
        'class_balanced': False,
        'num_epochs': 5,
        'num_runs': 3,
        'algs': ['cc', 'wrr', 'weighted_wrr', 'dann'], # wrr, weighted_wrr, cons_wrr, lje, erm, cc, dann, fdal, reverse-kl

        # Debugging algorithms
        'debug': True,
        'print_every_n': 100,
        'report_source_train_risk': False,
        'report_target_train_risk': False,
        'pretrain_on_both': False, # starting from a model that 'cheats'!
        'adapt_only_last_layer': False,

        # Test set dataloader options
        'test_batch_size': 512,
        'checkpoint': False,
        'validate': False,
    }

    debug_config = {
        'calc_label_shift': False,
        'calc_entanglement': False,
        'calc_margin': False,
        'calc_wrr': True,
        'calc_weighted_wrr': False,
        'verbose_weighted_wrr': False,
        'calc_weight_info': False,
        'calc_grad_info': False,
        }

    config['debug_options'] = debug_config

    # Algorithms and their hyperparameters/options
    config = setup_alg_config(config)
    return config


def setup_alg_config(config):
    wrr_config = {
        'scale': 1.0,
        'norm': 2,
        'entropy_reg': 1e-3,
        'propagate_labels': False,
        'print_info': False,
        }

    weighted_wrr_config = {
        'scale': 1.0,
        'entropy_reg': 1e-1,
        'add_source_loss': True,
        'separate_optim': True,
        'uot_alg': 'mm', # sinkhorn or mm
        'uot_init': False, # initialize MM with semi-relaxed UOT
        'uot_iter_max': 1000,
        'autograd_at_convergence': True,
        'reg_m': (1.0, 100.0),
        'print_info': False,
        }

    cons_wrr_config = {
        'norm': 2,
        'entropy_reg': 1e-3,
        'scale': 1.0,
        'thresh': 0.01
        }

    dann_config = {
        'layer_to_apply_disc': 'flatten', #'flatten' for Conv or -2 for MLP
        'discriminator': ConvDomainClassifier(), # ConvDomainClassifier() or MLP([100, 10, 2], nn.ReLU())
        'learning_rate': 1e-3, # for the internal optimizer
        'weight_decay': 0.0,
        'num_epochs': 2,
        'num_batches': 1000, # rough estimate should be enough
        }

    fdal_config = {
        'juncture': -1, # For now we keep backbone/taskhead juncture at the last layer only
        'auxhead': ConvDomainClassifier(), # This seems to be necessary to prevent blowing up!
        'grl': {"max_iters": 3000, "hi": 0.6, "auto_step": True},
        'divergence': 'pearson',
        'learning_rate': 1e-3, # for the internal optimizer
        'weight_decay': 0.0,
        'clip_grad_val': 10,
    }

    cc_config = {
        'entropy_reg': 1e-3,
        'norm': 2,
        'mode': 'joint', # or 'weighted_joint' or 'conditional'
        'add_source_loss': False, # only for weighted_joint
        }

    reverse_kl_config = {
        'alpha_reverse': 0.1,
        'alpha_forward': 0.1,
        'augment_softmax': 0.0,
    }

    config['wrr'] = wrr_config
    config['weighted_wrr'] = weighted_wrr_config
    config['cons_wrr'] = cons_wrr_config
    config['dann'] = dann_config
    config['fdal'] = fdal_config
    config['cc'] = cc_config
    config['reverse_kl'] = reverse_kl_config
    return config


def run_all_experiments(config, fabric):
    # Run all experiments
    models = ['MLP', 'ConvNet']
    scenarios = ['MNIST_TO_USPS', 'USPS_TO_MNIST', 'MNIST_TO_MNIST_M', 'SVHN_TO_MNIST']
    for model in models:
        for scenario in scenarios:
            config['model'] = model
            config['scenario'] = scenario
            res = run_uda(config, fabric)
            plot_results(res, config['algs'])


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    # torch.set_printoptions(precision=3, sci_mode=False)
    # torch.autograd.set_detect_anomaly(True)

    config = setup_config()
    fabric = Fabric(accelerator=config['device'], devices="auto", strategy="auto")
    fabric.launch()

    if fabric.global_rank != 0:
        import builtins
        builtins.print = lambda *args, **kwargs: None

    print(f"Fabric device: {fabric.device}")
    reset_all(seed=0)

    res = run_uda(config, fabric)
    # plot_results(res, config['algs'])
    # run_all_experiments(config, fabric)
