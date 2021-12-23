import torch
import pandas as pd
import numpy as np


class TeacherData:
    def __init__(self):
        self.num_classes = 10
        self.m_forward = 8


    def read_teacher_outputs(self, mode):
        csv_file_name = mode + '_output.csv'
        df = pd.read_csv(csv_file_name)
        return df

    def get_teacher_output_by_batch_number(self, outputs_df, batch_index):
        outputs_df = outputs_df.loc[outputs_df['batch_number'] == str(batch_index), outputs_df.columns != 'batch_number']
        outputs = outputs_df.to_numpy()
        outputs_to_tensor = torch.from_numpy(np.vstack(outputs).astype(np.float))
        return outputs_to_tensor.reshape([self.m_forward,
                                          int(outputs_to_tensor.shape[0]/self.m_forward),
                                          outputs_to_tensor.shape[1]]) # 8,256,10


    def generate_prediction_smoothong_outputs(self, outputs):
        maxk = 1  # remove it later..
        num_classes = 10  # TODO: move it to class definition
        predictions = [output.topk(maxk, 1, True, True) for output in outputs]
        predictions = [pred.unsqueeze(dim=2) for _, pred in predictions]
        predictions_tensor = torch.cat(predictions, dim=2).view(-1, self.m_forward)
        batch_predictions = list(predictions_tensor.split(1, dim=0))
        batch_hists = [torch.histc(m_predictions.float(),
                                   bins=self.num_classes,
                                   min=0,
                                   max=(self.num_classes - 1)) for m_predictions in batch_predictions]
        histogram = torch.cat(batch_hists, dim=0).view(-1, maxk, self.num_classes)
        histogram = torch.div(histogram, float(self.m_forward))
        return histogram





