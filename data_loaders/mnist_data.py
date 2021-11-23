import torch
from torchvision.transforms import transforms


def get_loaders(dataset, data, batch_size, val_batch_size, workers):
    train_dataset = dataset(
        root=data,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ]))

    train_batch_sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(train_dataset),
                                                        batch_size=batch_size, drop_last=False)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=workers)

    test_dataset = dataset(
        root=data,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ]))

    test_batch_sampler = torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(test_dataset),
                                                       batch_size=val_batch_size, drop_last=False)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, num_workers=workers)


    return trainloader, testloader, None
