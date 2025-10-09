import os
import time
import torch
import numpy as np
import glog as log


def one_hot(x, k):
    return torch.nn.functional.one_hot(x, k).float()


def test(dataloader, model, loss_fun):
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    num_points = 0
    # We can't use this when we use a sampler to up/downsample the dataset!!!
    # num_inputs = len(dataloader.dataset)
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fun(pred, one_hot(y, model.num_classes)).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            num_points += y.shape[0]
    test_loss /= num_batches
    correct /= num_points
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct


def report_metrics(scenario, model, loss_fun, report_source_train, report_target_train):
    # These are very slow
    if report_source_train:
        print(f"Reporting accuracy/loss on source {scenario.source_name} training dataset...")
        test(scenario.source_dataloader, model, loss_fun)

    if report_target_train:
        print(f"Reporting accuracy/loss on target {scenario.target_name} training dataset...")
        test(scenario.target_dataloader, model, loss_fun)

    print(f"Reporting accuracy/loss on {scenario.source_name} test dataset...")
    loss_source, acc_source = test(scenario.source_test_dataloader, model, loss_fun)

    print(f"Reporting accuracy/loss on {scenario.target_name} test dataset...")
    loss_target, acc_target = test(scenario.target_test_dataloader, model, loss_fun)
    return torch.tensor([loss_source, acc_source, loss_target, acc_target])


def train_model_on_source(config, model, loss_fun, scenario, opt, fabric):
    folder_path = "save_files/" + scenario.name + "/"
    save_path = folder_path + model.name + ".pth"
    try:
        # Load parameters from a file
        model.load_state_dict(torch.load(save_path, weights_only=True))
        model = fabric.setup(model)
        log.info(f"Saved model found! Loading parameters from file: {save_path}")
    except:
        log.info(f"Either model has changed or save file {save_path} not found. Training from scratch...")
        model, opt = fabric.setup(model, opt)
        train(
            scenario.source_dataloader,
            model,
            loss_fun,
            opt,
            config['num_pretrain_epochs'],
            fabric,
            report_every=10,
        )
        # Report accuracy/loss on whole training dataset
        test(scenario.source_dataloader, model, loss_fun)
        log.info(f"Saving parameters to file: {save_path}")
        os.makedirs(folder_path, exist_ok=True)
        torch.save(model.state_dict(), save_path)
    return model


# Checks that the same network architecture used for both source *and* target can achieve high accuracy
def train_model_on_source_and_target(config, model, loss_fun, scenario, opt, fabric):
    folder_path = "save_files/" + scenario.name + "/train_on_both/"
    save_path = folder_path + model.name + ".pth"
    try:
        # Load parameters from a file
        model.load_state_dict(torch.load(save_path, weights_only=True))
        model = fabric.setup(model)
        log.info(f"Saved model found! Loading parameters from file: {save_path}")
    except:
        log.info(f"Save file {save_path} not found. Training from scratch...")
        combined_data = torch.utils.data.ConcatDataset([scenario.source_data, scenario.target_data])
        dataloader = torch.utils.data.DataLoader(combined_data, **scenario.dataloader_options)
        model, opt = fabric.setup(model, opt)
        train(
            dataloader,
            model,
            loss_fun,
            opt,
            config['num_pretrain_epochs'],
            fabric,
            report_every=10,
        )
        # Report accuracy/loss on whole training dataset
        test(scenario.source_dataloader, model, loss_fun)
        log.info(f"Saving parameters to file: {save_path}")
        os.makedirs(folder_path, exist_ok=True)
        torch.save(model.state_dict(), save_path)
    return model


def train(dataloader, model, loss_fun, optimizer, num_epochs, fabric, report_every=1, report_metrics=True):
    size = len(dataloader.dataset)
    t0 = time.perf_counter()
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        for batch, (X, y) in enumerate(dataloader):
            y = one_hot(y, model.num_classes)
            loss = loss_fun(model(X), y)
            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            if batch % report_every == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f} epoch:{epoch+1} [{current:>5d}/{size:>5d}]")
        if report_metrics is True:
            print("Train dataset metrics:")
            test(dataloader, model, loss_fun)
    print(f"Method took {time.perf_counter() - t0} sec")
