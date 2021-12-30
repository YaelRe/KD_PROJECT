import torchvision


class Cifar10_with_indices(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        return data, target, index
