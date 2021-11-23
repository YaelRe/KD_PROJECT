import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import RandomSampler
from torchvision import datasets, transforms

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}


def inception_preproccess(input_size, normalize=__imagenet_stats):
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ])


def scale_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    if scale_size != input_size:
        t_list = [transforms.Resize(scale_size)] + t_list

    return transforms.Compose(t_list)


def get_transform(augment=True, input_size=224):
    normalize = __imagenet_stats
    scale_size = int(input_size / 0.875)
    if augment:
        return inception_preproccess(input_size=input_size, normalize=normalize)
    else:
        return scale_crop(input_size=input_size, scale_size=scale_size, normalize=normalize)


def get_loaders(dataset, data, batch_size, val_batch_size, workers):
    # TODO: pin-memory currently broken for distributed
    pin_memory = False
    # TODO: datasets.ImageNet
    val_data = datasets.ImageFolder(root=os.path.join(data, 'val'), transform=get_transform(False))
    val_sampler = RandomSampler(val_data)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=val_batch_size, sampler=val_sampler,
                                             num_workers=workers, pin_memory=pin_memory)

    train_data = datasets.ImageFolder(root=os.path.join(data, 'train'),
                                      transform=get_transform())
    train_sampler = RandomSampler(train_data)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler,
                                               num_workers=workers, pin_memory=pin_memory)
    return train_loader, val_loader, None
