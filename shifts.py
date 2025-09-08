import os
import pandas
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader, random_split, ConcatDataset
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler
from torchvision import datasets
from torchvision.transforms import v2

import utils


class USPS_to_MNIST:
    def __init__(self, dataloader_options, test_dataloader_options, use_sampler):
        self.name = "USPS_TO_MNIST"
        self.num_classes = 10
        self.dataloader_options = dataloader_options
        self.source_name = "USPS"
        self.target_name = "MNIST"
        self.transforms_target = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize(size=[16, 16], antialias=True)]
        )
        self.transforms_source = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.input_size = 256
        self.source_data = datasets.USPS(
            root="data/USPS",
            train=True,
            download=True,
            transform=self.transforms_source,
        )
        self.target_data = datasets.MNIST(root="data", train=True, download=True, transform=self.transforms_target)

        # Load both datasets
        if use_sampler is True:
            sampler = RandomSampler(self.source_data, replacement=True, num_samples=len(self.target_data))
            self.source_dataloader = DataLoader(self.source_data, sampler=sampler, **dataloader_options)
        else:
            self.source_dataloader = DataLoader(self.source_data, **dataloader_options)
        self.target_dataloader = DataLoader(self.target_data, **dataloader_options)

        test_data = datasets.USPS(root="data/USPS", train=False, download=True, transform=self.transforms_source)
        self.source_test_dataloader = DataLoader(test_data, **test_dataloader_options)
        test_data = datasets.MNIST(root="data", train=False, download=True, transform=self.transforms_target)
        self.target_test_dataloader = DataLoader(test_data, **test_dataloader_options)
        # calc_w_distance_label_shift(self)
        self.labels = np.array([str(i) for i in range(10)])


class MNIST_to_USPS:
    def __init__(self, dataloader_options, test_dataloader_options, use_sampler, class_balanced):
        self.name = "MNIST_TO_USPS"
        self.num_classes = 10
        self.dataloader_options = dataloader_options
        self.source_name = "MNIST"
        self.target_name = "USPS"
        self.transforms_target = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.transforms_source = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize(size=[16, 16], antialias=True)]
        )
        self.input_size = 256
        self.source_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=self.transforms_source,
        )
        self.target_data = datasets.USPS(root="data/USPS", train=True, download=True, transform=self.transforms_target)
        source_test_data = datasets.MNIST(root="data", train=False, download=True, transform=self.transforms_source)
        target_test_data = datasets.USPS(root="data/USPS", train=False, download=True, transform=self.transforms_target)

        # Load both datasets
        if use_sampler is True:
            if class_balanced is True:
                source_class_weights = 1.0 / torch.bincount(torch.tensor(self.source_data.targets)).float()
                source_sample_weights = source_class_weights[self.source_data.targets]
                source_sampler = WeightedRandomSampler(
                    weights=source_sample_weights, num_samples=len(source_sample_weights), replacement=True
                )
                self.source_dataloader = DataLoader(self.source_data, sampler=source_sampler, **dataloader_options)
                target_class_weights = 1.0 / torch.bincount(torch.tensor(self.target_data.targets)).float()
                target_sample_weights = target_class_weights[self.target_data.targets]
                target_sampler = WeightedRandomSampler(
                    weights=target_sample_weights, num_samples=len(self.source_data), replacement=True
                )
                self.target_dataloader = DataLoader(self.target_data, sampler=target_sampler, **dataloader_options)

                source_class_weights = 1.0 / torch.bincount(torch.tensor(source_test_data.targets)).float()
                source_sample_weights = source_class_weights[source_test_data.targets]
                source_sampler = WeightedRandomSampler(
                    weights=source_sample_weights, num_samples=len(source_sample_weights), replacement=True
                )
                self.source_test_dataloader = DataLoader(source_test_data, sampler=source_sampler, **test_dataloader_options)
                target_class_weights = 1.0 / torch.bincount(torch.tensor(target_test_data.targets)).float()
                target_sample_weights = target_class_weights[target_test_data.targets]
                target_sampler = WeightedRandomSampler(
                    weights=target_sample_weights, num_samples=len(source_test_data), replacement=True
                )
                self.target_test_dataloader = DataLoader(target_test_data, sampler=target_sampler, **test_dataloader_options)
            else:
                self.source_dataloader = DataLoader(self.source_data, **dataloader_options)
                self.source_test_dataloader = DataLoader(source_test_data, **test_dataloader_options)
                target_sampler = RandomSampler(self.target_data, replacement=True, num_samples=len(self.source_data))
                self.target_dataloader = DataLoader(self.target_data, sampler=target_sampler, **dataloader_options)
                target_test_sampler = RandomSampler(
                    target_test_data, replacement=True, num_samples=len(source_test_data)
                )
                self.target_test_dataloader = DataLoader(
                    target_test_data, sampler=target_test_sampler, **test_dataloader_options
                )
        else:
            self.source_dataloader = DataLoader(self.source_data, **dataloader_options)
            self.target_dataloader = DataLoader(self.target_data, **dataloader_options)
            self.source_test_dataloader = DataLoader(source_test_data, **test_dataloader_options)
            self.target_test_dataloader = DataLoader(target_test_data, **test_dataloader_options)
        # calc_w_distance_label_shift(self)
        self.labels = np.array([str(i) for i in range(10)])


class MNIST_to_MNIST_M:
    def __init__(self, dataloader_options, test_dataloader_options, preprocess):
        self.name = "MNIST_TO_MNISTM"
        if preprocess is True:
            self.process_MNIST_M_labels(root="data/MNIST-M", use_train=True)
            self.process_MNIST_M_labels(root="data/MNIST-M", use_train=False)
        self.num_classes = 10
        self.input_size = 3072
        self.dataloader_options = dataloader_options
        self.source_name = "MNIST"
        self.target_name = "MNIST_M"
        self.transforms_source = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize(size=(32, 32)), self.StackTransform()]
        )
        self.transforms_target = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.source_data = datasets.MNIST(root="data", train=True, download=True, transform=self.transforms_source)
        self.target_data = datasets.ImageFolder(root="data/MNIST-M/train", transform=self.transforms_target)

        # Load both datasets
        self.source_dataloader = DataLoader(self.source_data, **dataloader_options)
        self.target_dataloader = DataLoader(self.target_data, **dataloader_options)

        source_test_data = datasets.MNIST(root="data", train=False, download=True, transform=self.transforms_source)
        self.source_test_dataloader = DataLoader(source_test_data, **test_dataloader_options)
        target_test_data = datasets.ImageFolder(root="data/MNIST-M/test", transform=self.transforms_target)
        self.target_test_dataloader = DataLoader(target_test_data, **test_dataloader_options)
        # calc_w_distance_label_shift(self)
        self.labels = np.array([str(i) for i in range(10)])

    def process_MNIST_M_labels(self, root, use_train):
        if use_train:
            folder = "train"
        else:
            folder = "test"
        labels_file = os.path.join(root, folder + "_labels.txt")
        out = pandas.read_csv(labels_file, header=None, names=["name", "label"], sep=" ")
        for label in out["label"].unique():
            os.makedirs(os.path.join(root, folder, str(label)), exist_ok=True)
        for index, elem in out.iterrows():
            os.rename(
                src=os.path.join(root, folder, elem["name"]),
                dst=os.path.join(root, folder, str(elem["label"]), elem["name"]),
            )

    class StackTransform:
        # Transform MNIST to SVHN RGB-shape (this is a bit dumb)
        def __call__(self, x):
            return torch.stack((x[0], x[0], x[0]), dim=0)


class SVHN_to_MNIST:
    def __init__(self, dataloader_options, test_dataloader_options, class_balanced):
        self.name = "SVHN_TO_MNIST"
        self.dataloader_options = dataloader_options
        self.source_name = "SVHN"
        self.target_name = "MNIST"
        self.input_size = 3072
        self.num_classes = 10
        self.transforms_source = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.transforms_target = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize(size=(32, 32)), self.StackTransform()]
        )
        self.source_data = datasets.SVHN(
            root="data/SVHN", split="train", download=True, transform=self.transforms_source
        )
        self.target_data = datasets.MNIST(root="data", train=True, download=True, transform=self.transforms_target)
        source_test_data = datasets.SVHN(
            root="data/SVHN", split="test", download=True, transform=self.transforms_source
        )
        target_test_data = datasets.MNIST(root="data", train=False, download=True, transform=self.transforms_target)

        # Load both datasets
        if class_balanced is True:
            source_class_weights = 1.0 / torch.bincount(torch.tensor(self.source_data.labels)).float()
            source_sample_weights = source_class_weights[self.source_data.labels]
            source_sampler = WeightedRandomSampler(
                weights=source_sample_weights, num_samples=len(source_sample_weights), replacement=True
            )
            self.source_dataloader = DataLoader(self.source_data, sampler=source_sampler, **dataloader_options)
            target_class_weights = 1.0 / torch.bincount(torch.tensor(self.target_data.targets)).float()
            target_sample_weights = target_class_weights[self.target_data.targets]
            target_sampler = WeightedRandomSampler(
                weights=target_sample_weights, num_samples=len(target_sample_weights), replacement=True
            )
            self.target_dataloader = DataLoader(self.target_data, sampler=target_sampler, **dataloader_options)

            source_class_weights = 1.0 / torch.bincount(torch.tensor(source_test_data.labels)).float()
            source_sample_weights = source_class_weights[source_test_data.labels]
            source_test_sampler = WeightedRandomSampler(
                weights=source_sample_weights, num_samples=len(source_sample_weights), replacement=True
            )
            self.source_test_dataloader = DataLoader(
                source_test_data, sampler=source_test_sampler, **test_dataloader_options
            )
            target_class_weights = 1.0 / torch.bincount(torch.tensor(target_test_data.targets)).float()
            target_sample_weights = target_class_weights[target_test_data.targets]
            target_test_sampler = WeightedRandomSampler(
                weights=target_sample_weights, num_samples=len(target_sample_weights), replacement=True
            )
            self.target_test_dataloader = DataLoader(
                target_test_data, sampler=target_test_sampler, **test_dataloader_options
            )
        else:
            self.source_dataloader = DataLoader(self.source_data, **dataloader_options)
            self.target_dataloader = DataLoader(self.target_data, **dataloader_options)
            self.source_test_dataloader = DataLoader(source_test_data, **test_dataloader_options)
            self.target_test_dataloader = DataLoader(target_test_data, **test_dataloader_options)
        # calc_w_distance_label_shift(self)
        self.labels = np.array([str(i) for i in range(10)])

    class StackTransform:
        # Transform MNIST to SVHN RGB-shape (this is a bit dumb)
        def __call__(self, x):
            return torch.stack((x[0], x[0], x[0]), dim=0)


class CIFAR_CORRUPT:
    def __init__(self, dataloader_options, test_dataloader_options, corruptions):
        self.name = "CIFAR10_TO_CIFAR10C"
        self.dataloader_options = dataloader_options
        self.source_name = "CIFAR10"
        self.target_name = "CIFAR10C"
        self.input_size = 3072
        self.num_classes = 10
        self.transforms_source = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.transforms_target = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.source_data = datasets.CIFAR10(root="data/CIFAR10", train=True, transform=self.transforms_source)

        X_target = torch.tensor([])
        y_target = torch.tensor([])
        for corruption in corruptions:
            X = torch.from_numpy(np.load("data/CIFAR10-C/" + corruption + ".npy")).permute((0, 3, 1, 2))
            y = torch.from_numpy(np.load("data/CIFAR10-C/labels.npy")).long()
            X_target = torch.concatenate((X_target, X))
            y_target = torch.concatenate((y_target, y))
        num_targets = X_target.shape[0]
        idx_shuffle = torch.randperm(num_targets)
        X_target = X_target[idx_shuffle]
        y_target = y_target[idx_shuffle].long()
        num_test_use = int(X_target.shape[0] * 0.8)
        X_target_use, X_target_test = X_target[:num_test_use], X_target[num_test_use:]
        y_target_use, y_target_test = y_target[:num_test_use], y_target[num_test_use:]

        self.target_data = utils.GenericDataset(X_target_use, y_target_use, transform=self.transforms_target)

        # Load both datasets
        self.source_dataloader = DataLoader(self.source_data, **dataloader_options)
        self.target_dataloader = DataLoader(self.target_data, **dataloader_options)

        source_test_data = datasets.CIFAR10(
            root="data/CIFAR10", train=False, download=True, transform=self.transforms_source
        )
        self.source_test_dataloader = DataLoader(source_test_data, **test_dataloader_options)
        target_test_data = utils.GenericDataset(X_target_test, y_target_test, transform=self.transforms_target)
        self.target_test_dataloader = DataLoader(target_test_data, **test_dataloader_options)
        # calc_w_distance_label_shift(self)
        self.labels = np.array(
            ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        )


# For now the scenario we consider: training on Art, Clipart, Product -> testing on Real World
class OFFICEHOME:
    def __init__(self, dataloader_options, test_dataloader_options, size=(32, 32), train_ratio=0.8, grayscale=False):
        self.name = "OFFICEHOME"
        self.num_classes = 65
        self.dataloader_options = dataloader_options
        self.input_size = size[0] * size[1]
        operations = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize(size)]
        if grayscale is True:
            operations.append(v2.Grayscale(1))
        transforms = v2.Compose(operations)
        # art_data = datasets.ImageFolder(root='data/OfficeHomeDataset/Art', transform=transforms)
        # clipart_data = datasets.ImageFolder(root='data/OfficeHomeDataset/Clipart', transform=transforms)
        product_data = datasets.ImageFolder(root="data/OfficeHomeDataset/Product", transform=transforms)
        real_data = datasets.ImageFolder(root="data/OfficeHomeDataset/Real World", transform=transforms)

        # source_data = ConcatDataset([art_data, clipart_data, product_data])
        source_data = product_data
        target_data = real_data

        train_size = int(train_ratio * len(source_data))
        test_size = int(len(source_data)) - train_size
        train_source, test_source = random_split(source_data, [train_size, test_size])

        train_size = int(train_ratio * len(target_data))
        test_size = int(len(target_data)) - train_size
        train_target, test_target = random_split(target_data, [train_size, test_size])

        # Load both datasets
        self.source_dataloader = DataLoader(train_source, **dataloader_options)
        self.target_dataloader = DataLoader(train_target, **dataloader_options)
        self.source_test_dataloader = DataLoader(test_source, **test_dataloader_options)
        self.target_test_dataloader = DataLoader(test_target, **test_dataloader_options)

        self.source_name = "product"
        self.target_name = "real"


class PORTRAITS:
    def __init__(self, dataloader_options, size=(32, 32), train_ratio=0.8):
        self.name = "PORTRAITS"
        self.num_classes = 2
        self.dataloader_options = dataloader_options
        self.input_size = size[0] * size[1]
        transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Grayscale(1), v2.Resize(size)])
        dataset = ImageFolderWithFilenames(root="data/Portraits", transform=transforms)

        # Print first year of dataset
        _, _, filename = dataset.get_item_with_year(0)
        first_year = filename.split("_")[0]
        print(f"First year of source dataset: {first_year}")

        # Check mid-index year and report
        mid_idx = int(len(dataset) / 2)
        _, _, filename = dataset.get_item_with_year(mid_idx)
        year = filename.split("_")[0]
        self.source_name = "P1900-" + str(year)
        self.target_name = "P" + str(year) + "-2010"
        print(f"Year separating source from target datasets: {year}")

        # Print last year of dataset
        _, _, filename = dataset.get_item_with_year(-1)
        final_year = filename.split("_")[0]
        print(f"Final year of target dataset: {final_year}")

        subset1_idx = list(range(0, mid_idx))
        subset2_idx = list(range(mid_idx, len(dataset)))
        subset1 = Subset(dataset, subset1_idx)
        subset2 = Subset(dataset, subset2_idx)

        train_size = int(train_ratio * len(subset1))
        test_size = int(len(subset1)) - train_size
        train_source, test_source = random_split(subset1, [train_size, test_size])
        train_size = int(train_ratio * len(subset2))
        test_size = int(len(subset2)) - train_size
        train_target, test_target = random_split(subset2, [train_size, test_size])

        # Load both datasets
        self.source_dataloader = DataLoader(train_source, **dataloader_options)
        self.target_dataloader = DataLoader(train_target, **dataloader_options)
        self.source_test_dataloader = DataLoader(test_source, **test_dataloader_options)
        self.target_test_dataloader = DataLoader(test_target, **test_dataloader_options)
        # calc_w_distance_label_shift(self)
        self.labels = np.array(["Female", "Male"])


# Assuming Euclidean distance is to be used for W_{1,l} computation,
# the result is equal to \sqrt(2) / 2 * \sum_i |p_i - q_i|
def calc_w_distance_label_shift(scenario):
    num_cond_source = torch.zeros(scenario.num_classes)
    num_cond_target = torch.zeros(scenario.num_classes)
    for (X_train, y_train), (X_shift, y_shift) in zip(scenario.source_dataloader, scenario.target_dataloader):
        for i in range(scenario.num_classes):
            num_cond_source[i] += torch.sum(y_train == i)
            num_cond_target[i] += torch.sum(y_shift == i)
    p_y = num_cond_source / torch.sum(num_cond_source)
    q_y = num_cond_target / torch.sum(num_cond_target)
    w_1_euclidean_dist = np.sqrt(2) * torch.sum(torch.abs(p_y - q_y)) / 2
    print(f"W1_distance_labels for {scenario.name} is {w_1_euclidean_dist}")


class ImageFolderWithFilenames(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []

        # Traverse the folder and store image paths and filenames
        for root_dir, _, files in os.walk(self.root):
            for file_name in files:
                if file_name.endswith((".png", ".jpg", ".jpeg")):
                    self.images.append(os.path.join(root_dir, file_name))
                    # Extract the folder name as label
                    label = os.path.basename(root_dir)
                    if label == "F":
                        self.labels.append(0)
                    else:
                        self.labels.append(1)

        # Sort the images and labels based on file names
        self.images, self.labels = zip(
            *sorted(
                zip(self.images, self.labels),
                key=lambda x: os.path.basename(x[0]),  # Sorting based on image file path (x[0])
            )
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]  # Get the corresponding label
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_item_with_year(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]  # Get the corresponding label
        if self.transform:
            image = self.transform(image)

        # Get the filename from the path
        filename = os.path.basename(image_path)

        return image, label, filename  # return the image and the filename
