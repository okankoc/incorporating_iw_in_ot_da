def setup_cluster_config():
    config = {
        # Experiment details
        "device": "auto",  # 'cpu' or 'auto' to find gpu automatically
        # Model and optimizer (MLP, ConvNet, ConvNet2, LeNet, SmallCNN, ResNet)
        "models": ["ResNet"],
        "resnet_size": 18,  # 18 or 50
        "pretrain": True,
        "num_pretrain_epochs": 5,  # if pretrain is True
        "loss": "margin",  # margin, euclidean or cross-entropy
        "optimizer": "adam",  # alternatives: adam, sgd
        "learning_rate": 1e-3,  # use 1e-4 for ResNets or a learning scheduler
        "momentum": 0.9,  # for SGD
        "weight_decay": 0.0,
        "num_epochs": 5,
        "num_runs": 3,
        "algs": [
            "wrr",
            "weighted_wrr",
            "lje",
            "erm",
            "cc",
            "dann",
            "reverse_kl",
        ],  # wrr, weighted_wrr, cons_wrr, lje, erm, cc, dann, fdal, reverse-kl
        # Debugging algorithms
        "debug": False,
        "debug_every_n": 50,
        "n_batches_per_epoch": -1,  # if -1 uses all batches
        "report_source_train_risk": False,
        "report_target_train_risk": False,
        "pretrain_on_both": False,  # starting from a model that 'cheats'!
        "adapt_only_last_layer": False,
        "checkpoint": False,
        "validate": False,
    }

    scenario_config = {
        # Data loader options
        "batch_size": 64,
        # Test set dataloader options
        "test_batch_size": 512,
        "shuffle": True,
        "cifar-10-corruptions": ["fog", "frost", "snow"],
        # Distribution shift scenario (MNIST_TO_USPS, CIFAR10C, ...)
        "scenarios": [
            "MNIST_TO_USPS",
            "USPS_TO_MNIST",
            "MNIST_TO_MNIST_M",
            "SVHN_TO_MNIST",
        ],
    }

    debug_config = {
        "calc_label_shift": False,
        "calc_entanglement": False,
        "calc_margin": False,
        "calc_wrr": True,
        "calc_weighted_wrr": True,
        "verbose_weighted_wrr": False,
        "calc_weight_info": False,
        "calc_grad_info": False,
    }

    config["scenario_options"] = scenario_config
    config["debug_options"] = debug_config

    # Algorithms and their hyperparameters/options
    config = setup_alg_config(config)
    return config


def setup_alg_config(config):
    wrr_config = {
        "scale": 1.0,
        "norm": 2,
        "entropy_reg": 1e-3,
        "propagate_labels": False,
        "print_info": False,
    }

    weighted_wrr_config = {
        "scale": 1.0,
        "entropy_reg": 1e-1,  # only for sinkhorn uot
        "add_source_loss": True,
        "separate_optim": True,
        "uot_alg": "mm",  # sinkhorn or mm
        "uot_init": False,  # initialize MM with semi-relaxed UOT
        "uot_iter_max": 1000,
        "autograd_at_convergence": True,
        "reg_m": (1.0, 100.0),
        "print_info": False,
    }

    cons_wrr_config = {"norm": 2, "entropy_reg": 1e-3, "scale": 1.0, "thresh": 0.01}

    dann_config = {
        "conv_feat_layer": "flatten",
        "mlp_feat_layer": -2,  # ignored for ResNets
        "discriminator": "conv",  # conv or mlp
        "learning_rate": 1e-3,  # for the internal optimizer
        "weight_decay": 0.0,
        "num_epochs": 2,
        "num_batches": 1000,  # rough estimate should be enough
    }

    fdal_config = {
        "juncture": -1,  # For now we keep backbone/taskhead juncture at the last layer only
        "auxhead": "conv",  # conv or none
        "grl": {"max_iters": 3000, "hi": 0.6, "auto_step": True},
        "divergence": "pearson",
        "learning_rate": 1e-3,  # for the internal optimizer
        "weight_decay": 0.0,
        "clip_grad_val": 10,
    }

    cc_config = {
        "entropy_reg": 1e-3,
        "norm": 2,
        "mode": "joint",  # or 'weighted_joint' or 'conditional'
        "add_source_loss": False,  # only for weighted_joint
    }

    reverse_kl_config = {
        "alpha_reverse": 0.1,
        "alpha_forward": 0.1,
        "augment_softmax": 0.0,
    }

    config["wrr"] = wrr_config
    config["weighted_wrr"] = weighted_wrr_config
    config["cons_wrr"] = cons_wrr_config
    config["dann"] = dann_config
    config["fdal"] = fdal_config
    config["cc"] = cc_config
    config["reverse_kl"] = reverse_kl_config
    return config
