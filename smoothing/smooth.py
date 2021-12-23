import torch
import pandas as pd

from sample_methods.clean import Clean
from sample_methods.gaussian import Gaussian
from run import batch_index

normalize_Cifar10 = {'mean': torch.tensor([0.491, 0.482, 0.447]), 'std': torch.tensor([0.247, 0.243, 0.262])}
normalize_Cifar10_no_var = {'mean': torch.tensor([0.491, 0.482, 0.447]), 'std': torch.tensor([1, 1, 1])}


class Smooth:
    def __init__(self, base_model, noise_sd=0.0, m_forward=1, smooth="mcpredict", normalization='cifar10'):
        self.base_model = base_model
        num_classes = base_model.num_classes
        self.num_classes = num_classes
        if normalization == 'cifar10':
            normalization = normalize_Cifar10
        elif normalization == 'cifar10_no_var':
            normalization = normalize_Cifar10_no_var
        else:
            normalization = None
        self.normalize = (lambda x: x) if normalization is None else (
            lambda x: ((x - normalization['mean'].to(x).view(1, -1, 1, 1)) /
                       normalization['std'].to(x).view(1, -1, 1, 1)))

        self.noise_sd = torch.tensor(noise_sd)
        self.m_forward = m_forward

        self.monte_carlo_predict = None
        if smooth == 'mcpredict':
            self.monte_carlo_predict = self.monte_carlo_smooth_predict
        elif smooth == 'mcepredict':
            self.monte_carlo_predict = self.monte_carlo_expectation_predict
        else:
            raise ValueError('Wrong smooth method!')

        self.sample_method = Clean()
        if self.noise_sd > 0:
            self.sample_method = Gaussian(self.noise_sd)

    def forward(self, x):
        x_sample = self.sample_method.sample(x)
        x_sample = self.normalize(x_sample)
        return self.base_model(x_sample)

    # def __call__(self, x):
    #     return self.forward(x)

    def predict(self, x, output, maxk, calc_prob=False, mode=None):
        _, pred = output.topk(maxk, 1, True, True)
        outputs, hist, predict = self.monte_carlo_predict(x, maxk, pred)

        if mode is not None:
            stacked_outputs = torch.stack(outputs)
            stacked_outputs = stacked_outputs.detach().cpu()
            df = pd.DataFrame(
                stacked_outputs.reshape([stacked_outputs.shape[0] * stacked_outputs.shape[1], stacked_outputs.shape[2]]))
            df['batch_number'] = str(batch_index)
            output_file_name = mode + '_output.csv'
            df.to_csv(output_file_name, mode='a', index=False)

        pred_prob = -1
        pred_prob_var = -1
        if calc_prob:
            pred_prob = self.calc_pred_prob(x, predict, outputs, hist)
            pred_prob_var = pred_prob * (1 - pred_prob)

        return predict, pred_prob, pred_prob_var

    def generate_outputs(self, x, m_output):
        samples = self.sample_method.generate_samples(x, m_output)
        samples_n = [self.normalize(x_sample) for x_sample in samples]
        outputs = None
        with torch.no_grad():
            outputs = [self.base_model.forward(x_sample) for x_sample in samples_n]
        del samples
        del samples_n
        return outputs

    def calc_pred_prob(self, x, predict, outputs, hist):
        pred = predict[:, 0]
        if self.noise_sd == 0:
            return 0
        histogram = hist
        if hist is None:
            histogram = self.generate_histogram(outputs, 1, self.m_forward).view(-1, self.num_classes)
        m_maxk = histogram.gather(1, predict)
        m_prominent = m_maxk[:,0]
        pred_prob = torch.div(m_prominent.float(), self.m_forward)
        pred_prob = torch.mean(pred_prob).item()
        return pred_prob

    def monte_carlo_smooth_predict(self, x, maxk, pred):
        outputs = self.generate_outputs(x, self.m_forward)
        # with torch.no_grad():
        if self.m_forward < 2:
            _, predictions = outputs[0].topk(maxk, 1, True, True)
            return outputs, None, predictions

        output_hist = self.generate_histogram(outputs, 1, self.m_forward)
        histogram = output_hist.view(-1, self.num_classes)
        predict = histogram.topk(maxk, 1, True, True)[1]

        return outputs, histogram, predict

    def monte_carlo_expectation_predict(self, x, maxk, pred):
        self.eval()
        # with torch.no_grad():
        outputs = self.generate_outputs(x, self.m_forward)

        if self.m_forward < 2:
            x_n = self.normalize(x)
            x_sample = self.sample_method.generate_samples(x_n, self.m_forward)[0]
            output = self.base_model.forward(x_sample)
            _, predictions = output.topk(maxk, 1, True, True)
            return outputs, None, predictions

        output_list = [output_i.unsqueeze(dim=0).to(x) for output_i in outputs]
        outs = torch.cat(output_list)
        output = outs.mean(dim=0)
        _, predictions = output.topk(maxk, 1, True, True)
        return outputs, None, predictions

    def base_model_predict(self, x, maxk, pred):
        return pred

    def generate_histogram(self, outputs, maxk, m_hist):
        predictions = [output.topk(maxk, 1, True, True) for output in outputs]
        predictions = [pred.unsqueeze(dim=2) for _, pred in predictions]
        predictions_tensor = torch.cat(predictions, dim=2).view(-1, m_hist)
        batch_predictions = list(predictions_tensor.split(1, dim=0))
        batch_hists = [torch.histc(m_predictions,
                                   bins=self.num_classes,
                                   min=0,
                                   max=(self.num_classes - 1)) for m_predictions in batch_predictions]
        histogram = torch.cat(batch_hists, dim=0).view(-1, maxk, self.num_classes)
        return histogram

    # def gaussian_sample(self, x):
    #     if self.noise_sd > 0:
    #         return x + torch.randn_like(x) * self.noise_sd
    #     return x

    # def clean_sample(self, x):
    #     return x

    def set_mc_sample_method(self, method, first_sampling=True, second_sampling=False, final_sampling=False):
        if first_sampling:
            self.sample_method = method
        if second_sampling:
            self.second_sample_method = method
        if final_sampling:
            self.final_sample_method = method

    def data_parallel(self, gpus):
        self.base_model = torch.nn.DataParallel(self.base_model, gpus)
        return self

    def train(self, mode=True):
        self.base_model.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def requires_grad(self, requires=True):
        for param in self.base_model.parameters():
            param.requires_grad = requires

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.base_model = self.base_model.to(*args, **kwargs)
        self.noise_sd = self.noise_sd.to(*args, **kwargs).detach()

        return self

    def parameters(self, *args, **kwargs):
        return self.base_model.parameters(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.base_model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.base_model.load_state_dict(*args, **kwargs)

    def modules(self, *args, **kwargs):
        return self.base_model.modules(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        self.base_model.zero_grad(*args, **kwargs)
