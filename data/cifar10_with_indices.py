import torchvision
from torchvision.transforms import transforms


class Cifar10_with_indices(torchvision.datasets.CIFAR10):
    def __init__(self, data, train_mode):
        super().__init__(root=data)
        if train_mode == True:
            self.cifar10 = torchvision.datasets.CIFAR10(root=data,
                                                        train=True,
                                                        download=True,
                                                        transform=transforms.Compose([
                                                        transforms.RandomCrop(32, padding=4),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor() ]))
        else:
            self.cifar10 = torchvision.datasets.CIFAR10(
                                                        root=data,
                                                        train=False,
                                                        download=True,
                                                        transform=transforms.Compose([
                                                            transforms.ToTensor(),
                                                            # normalize,
                                                        ]))

    def __getitem__(self, index):
        data, target = self.cifar10[index]
        return (data, target, index)