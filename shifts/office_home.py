import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2


# For now the scenario we consider: training on Art, Clipart, Product -> testing on Real World
class OFFICEHOME:
    def __init__(
        self,
        dataloader_options,
        test_dataloader_options,
        size=(32, 32),
        train_ratio=0.8,
        grayscale=False,
    ):
        self.name = "OFFICEHOME"
        self.num_channels = 3
        self.num_classes = 65
        self.input_size = size[0] * size[1]
        operations = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size),
        ]
        if grayscale is True:
            operations.append(v2.Grayscale(1))
        transforms = v2.Compose(operations)
        # art_data = datasets.ImageFolder(root='data/OfficeHomeDataset/Art', transform=transforms)
        # clipart_data = datasets.ImageFolder(root='data/OfficeHomeDataset/Clipart', transform=transforms)
        product_data = datasets.ImageFolder(
            root="data/OfficeHomeDataset/Product", transform=transforms
        )
        real_data = datasets.ImageFolder(
            root="data/OfficeHomeDataset/Real World", transform=transforms
        )

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
