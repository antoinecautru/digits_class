import numpy as np
import torch
from torchvision import datasets, transforms
import random

def get_modified_dataset(base_dataset):

    class ModifiedDataset(base_dataset):
        def __init__(
            self,
            root,
            img_size=56,
            random_shift=False,
            scramble_image=False,
            noise=0.0,
            # add some parameters here if you like, or pass them through the kwargs 
            *args,
            **kwargs,
        ):

            super().__init__(root, *args, **kwargs)
            assert img_size >= 28 # 28 is the default size of MNIST images
            self.img_size = img_size
            self.scramble_image = scramble_image
            assert noise >= 0.0
            self.noise = noise

            if random_shift:
                rng = random.Random(433) # set seed
                self.r_idxs = [
                    rng.randrange(img_size - 28 + 1) for _ in range(len(self))
                ]
                self.c_idxs = [
                    rng.randrange(img_size - 28 + 1) for _ in range(len(self))
                ]
            else:
                self.r_idxs = [(img_size - 28) // 2] * len(self)
                self.c_idxs = self.r_idxs
            self.torch_rng = torch.Generator()
            self.torch_rng.manual_seed(2147483647)
            self.shuffle_idxs = torch.randperm(img_size**2, generator=self.torch_rng)
            # ...

        def __getitem__(self, index):
            sample = super().__getitem__(index)
            image, label = sample
            if self.img_size > 28:
                new_image = torch.full((1, self.img_size, self.img_size), image.min())
                c_idx = self.c_idxs[index]
                r_idx = self.r_idxs[index]
                new_image[:, c_idx : c_idx + 28, r_idx : r_idx + 28] = image
                image = new_image

            if self.noise:
                self.torch_rng.manual_seed(2147433433 + index)
                image = image + self.noise * torch.randn(
                    image.shape, generator=self.torch_rng
                )

            if self.scramble_image:
                image = image.view(-1)[self.shuffle_idxs].reshape(
                    1, self.img_size, self.img_size
                )
            return (image, label)

    return ModifiedDataset

def get_dataloaders(
    base_dataset,
    batch_size,
    **kwargs,
):

    dataset_cls = get_modified_dataset(base_dataset)
    if base_dataset == datasets.FashionMNIST:
        mean = 0.286041
        std = 0.353024
        labels_map = lambda label: {
            0: "T-Shirt",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle Boot",
        }[label]
        path = "./data/FMNIST/"
    elif base_dataset == datasets.MNIST:
        mean = 0.1307
        std = 0.3081
        labels_map = lambda label: label
        path = "./data/MNIST/"
    else:
        raise NotImplementedError

    
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    train_set = dataset_cls(
        root=path,
        train=True,
        download=True,
        transform=transform,
        **kwargs,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
    )
    val_set = dataset_cls(
        root=path,
        train=False,
        download=True,
        transform=transform,
        **kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=True,
    )
    
    return train_loader, val_loader

