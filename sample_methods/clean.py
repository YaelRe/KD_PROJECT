from sample_methods.sample import Sample


class Clean(Sample):
    def __init__(self, **kwargs):
        super(Clean, self).__init__()

    def generate_distribution(self, data, sample_num=1, target=None):
        self.data = data
        self.samples = [data for _ in range(sample_num)]
        return

    def add_perturbations(self, data_list, target_list=None):
        return data_list

    def sample(self, data, target=None):
        return data
