class Sample:
    def __init__(self):
        self.distribution = None
        self.data = None
        self.samples = None

    def generate_distribution(self, data, sample_num=1, target=None):
        raise NotImplementedError('You need to define a generate_distribution method!')

    def sample(self, data, target=None):
        self.generate_distribution(data, sample_num=1)
        return self.samples.pop()

    def generate_samples(self, data, sample_num=1, target=None):
        self.generate_distribution(data, sample_num, target)
        samples = self.samples
        del self.samples
        return samples

    def add_perturbations(self, data_list, target_list=None):
        raise NotImplementedError('You need to define a generate_pertubations method!')

