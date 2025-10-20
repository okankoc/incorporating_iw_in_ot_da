def init_scenario(config, fabric):
    dataloader_options = {
        "batch_size": config["batch_size"],
        "shuffle": False,
        "drop_last": False,
    }
    test_dataloader_options = {
        "batch_size": config["test_batch_size"],
        "shuffle": False,
        "drop_last": False,
    }
    if config["scenario"] == "MNIST_TO_USPS":
        scenario = MNIST_to_USPS(dataloader_options, test_dataloader_options)
    elif config["scenario"] == "USPS_TO_MNIST":
        scenario = USPS_to_MNIST(dataloader_options, test_dataloader_options)
    elif config["scenario"] == "MNIST_TO_MNIST_M":
        scenario = MNIST_to_MNIST_M(
            dataloader_options, test_dataloader_options, preprocess=False
        )
    elif config["scenario"] == "SVHN_TO_MNIST":
        scenario = SVHN_to_MNIST(dataloader_options, test_dataloader_options)
    elif config["scenario"] == "CIFAR-10-C":
        scenario = CIFAR_CORRUPT(
            dataloader_options,
            test_dataloader_options,
            corruptions=config["cifar-10-corruptions"],
        )
    elif config["scenario"] == "PORTRAITS":
        scenario = PORTRAITS(
            dataloader_options, test_dataloader_options, size=(32, 32), train_ratio=0.8
        )
    elif config["scenario"] == "OFFICEHOME":
        scenario = OFFICEHOME(
            dataloader_options, test_dataloader_options, size=(224, 224)
        )
    else:
        raise Exception("Unknown scenario")
    return scenario
