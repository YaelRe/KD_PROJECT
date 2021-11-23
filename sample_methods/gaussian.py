import torch

from sample_methods.sample import Sample


class Gaussian(Sample):
    def __init__(self, noise_sd, **kwargs):
        super(Gaussian, self).__init__()
        self.noise_sd = noise_sd

    def generate_distribution(self, data, sample_num=1, target=None):
        self.data = data
        self.samples = [self.data + torch.randn_like(self.data) * self.noise_sd for _ in range(sample_num)]
        return

    def add_perturbations(self, data_list, target_list=None):
        samples = [data + torch.randn_like(data) * self.noise_sd for data in data_list]
        return samples

    def sample(self, data, target=None):
        return data + torch.randn_like(data) * self.noise_sd
