import torch
from torchvision.transforms import transforms
from data.cifar10_with_indices import Cifar10_with_indices


def get_loaders(dataset, data, batch_size, val_batch_size, workers):
    dataset = Cifar10_with_indices
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

    train_dataset = dataset(
        root=data,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
        ]))
    train_batch_sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(train_dataset),
                                                        batch_size=batch_size, drop_last=False)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=workers,
                                              pin_memory=False)

    test_dataset = dataset(
        root=data,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            # normalize,
        ]))

    test_batch_sampler = torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(test_dataset),
                                                       batch_size=val_batch_size, drop_last=False)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, num_workers=workers,
                                             pin_memory=False)

    return trainloader, testloader, None
