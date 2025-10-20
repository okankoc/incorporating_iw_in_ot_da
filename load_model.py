import copy
import types
import torch
import torchvision
from torch import nn

import utils
from models.conv import ConvNet, ConvNet2, LeNet, SmallCNN
from models.mlp import MultiLayerPerceptron as MLP


def load_model(config, fabric, scenario):
    model = init_model(config, scenario)
    folder_path = "save_files/" + scenario.name + "/"
    save_path = folder_path + model.name + ".pth"
    try:
        # Load parameters from a file
        model.load_state_dict(torch.load(save_path, weights_only=True))
        print(f"Saved model found! Loading parameters from file: {save_path}")
    except:
        print(f"Saved model {model.name} NOT found!")
    model.train()
    if config["adapt_only_last_layer"]:
        num_layers = len(list(model.parameters()))
        for i, p in enumerate(model.parameters()):
            if i < num_layers - 1:
                p.requires_grad = False
    return model


def init_model(config, scenario):
    if config["model"] == "MLP":
        model = MLP(
            layer_sizes=[scenario.input_size, 200, 100, scenario.num_classes],
            f_nonlinear=nn.ReLU(),
        )
    elif config["model"] == "ConvNet":
        model = ConvNet(num_classes=scenario.num_classes)
    elif config["model"] == "ConvNet2":
        model = ConvNet2(num_classes=scenario.num_classes)
    elif config["model"] == "LeNet":
        model = LeNet(num_classes=scenario.num_classes)
    elif config["model"] == "SmallCNN":
        model = SmallCNN(num_classes=scenario.num_classes)
    elif config["model"] == "ResNet":
        model = init_resnet(
            config["resnet_size"], scenario.num_channels, scenario.num_classes
        )
    else:
        raise Exception("Model not found")
    init_lazy_modules(model, scenario)
    disable_inplace_activations(model)
    print(f"Initialized model {model.name}")
    return model


def disable_inplace_activations(model):
    for m in model.modules():
        if isinstance(m, (nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SiLU)):
            if hasattr(m, "inplace") and m.inplace:
                m.inplace = False


def init_resnet(size, num_inp_channels, num_classes):
    if size == 18:
        model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        model.name = "RESNET18"
    elif size == 50:
        model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        model.name = "RESNET50"
    else:
        raise Exception("Resnet size not allowed!")
    model.num_classes = num_classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if num_inp_channels == 1:
        # Average the pretrained RGB filters to
        # get a single-channel equivalent
        avg_weight = model.conv1.weight.sum(dim=1, keepdim=True) / 3
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.conv1.weight.data = avg_weight

    def track_features(model, layer_id):
        # Ignores the layer_id!
        # Register hooks for the layers you're interested in
        def fun(module, inputs, outputs):
            model.features = inputs[0]

        hook = model.fc.register_forward_hook(fun)

    @torch.no_grad()
    def save_params(model):
        model.state = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def restore_params(model):
        model.load_state_dict(model.state)
        return dict(model.named_parameters())

    model.track_features = types.MethodType(track_features, model)
    model.save_params = types.MethodType(save_params, model)
    model.restore_params = types.MethodType(restore_params, model)
    return model


def init_lazy_modules(model, scenario):
    model = model.to("cpu")
    for X_source, y_source in scenario.source_dataloader:
        model(X_source.to("cpu"))
        break
    return model


def init_lazy_discriminator(discr, model, scenario, use_features):
    model = model.to("cpu")
    discr = discr.to("cpu")
    for X_source, y_source in scenario.source_dataloader:
        if use_features is True:
            model(X_source.to("cpu"))
            discr(model.features)
        else:
            discr(model(X_source.to("cpu")))
        break
    return discr


def pretrain_model(model, config, fabric, scenario, loss_fun, opt, res):
    folder_path = "save_files/" + scenario.name + "/"
    save_path = folder_path + model.name + ".pth"
    try:
        # Load parameters from a file
        model.load_state_dict(torch.load(save_path, weights_only=True))
        print(f"Saved model found! Loading parameters from file: {save_path}")
        model = fabric.setup(model)
    except:
        print(f"Saved model {model.name} NOT found!")
        if config["pretrain"] is True:
            model, opt = fabric.setup(model, opt)
            print(f"Pretraining {config['num_pretrain_epochs']} epochs...")
            if config["pretrain_on_both"] is True:
                print(
                    "========= DEBUG MODE ON: USING TARGET LABELS TO PRETRAIN LJE ORACLE MODEL ======="
                )
                model = utils.train_model_on_source_and_target(
                    config, model, loss_fun, scenario, opt, fabric
                )
            else:
                model = utils.train_model_on_source(
                    config, model, loss_fun, scenario, opt, fabric
                )

    # Report initial performance
    res_pretrain = utils.report_metrics(
        scenario,
        model,
        loss_fun,
        config["report_source_train_risk"],
        config["report_target_train_risk"],
        fabric,
    )
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j, 0, :] = res_pretrain
    return res
