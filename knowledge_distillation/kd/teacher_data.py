import torch
import pandas as pd
import numpy as np


class TeacherData:
    def __init__(self, data_dic, m_forward=8):
        self.num_classes = 10
        self.m_forward = m_forward
        self.clean_outputs = None
        self.perturb_outputs = None

        self._read_teacher_outputs(data_dic)

    def _read_teacher_outputs(self, data_dic):
        if data_dic['clean_data'] is True:
            csv_file_name = './teacher_data/clean_train_data_output.csv'
            clean_df = pd.read_csv(csv_file_name)
            clean_df = clean_df.drop(clean_df[clean_df.image_indices == 'image_indices'].index)
            clean_df["image_indices"] = pd.to_numeric(clean_df["image_indices"])
            self.clean_outputs = clean_df

        if data_dic['perturb_data'] is True:
            csv_file_name = './teacher_data/perturb_train_data_output.csv'
            perturb_df = pd.read_csv(csv_file_name)
            perturb_df = perturb_df.drop(perturb_df[perturb_df.image_indices == 'image_indices'].index)
            perturb_df["image_indices"] = pd.to_numeric(perturb_df["image_indices"])
            self.perturb_outputs = perturb_df

    def _get_teacher_output_by_image_indices(self, outputs_df: pd.DataFrame, image_indices: list):
        batch_histogram_outputs = outputs_df.loc[outputs_df['image_indices'].isin(image_indices)]
        batch_histogram_outputs = batch_histogram_outputs.drop(['image_indices', 'batch_number'], axis=1)

        outputs = batch_histogram_outputs.to_numpy()
        outputs_to_tensor = torch.from_numpy(np.vstack(outputs).astype(np.float))
        return outputs_to_tensor

    def _generate_prediction_smoothing_outputs(self, outputs):
        histogram = torch.div(outputs, float(self.m_forward))
        return histogram

    def get_predictions_by_image_indices(self, mode: str, image_indices: list):
        df = None
        if mode == 'clean':
            df = self.clean_outputs
        if mode == 'perturb':
            df = self.perturb_outputs
        df_batch = self._get_teacher_output_by_image_indices(df, image_indices)
        return self._generate_prediction_smoothing_outputs(df_batch)




