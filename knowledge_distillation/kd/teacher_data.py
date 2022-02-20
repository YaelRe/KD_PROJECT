import torch
import pandas as pd
import numpy as np


def load_teacher_data_from_csv(file_name):
    clean_df = pd.read_csv(file_name)
    clean_df = clean_df.drop(clean_df[clean_df.image_indices == 'image_indices'].index)
    clean_df["image_indices"] = pd.to_numeric(clean_df["image_indices"])
    return clean_df


class TeacherData:
    def __init__(self, data_dic, m_forward=512):
        self.num_classes = 10
        self.m_forward = m_forward
        self.clean_train_outputs = None
        self.clean_test_outputs = None
        self.perturb_train_outputs = None
        self.perturb_test_outputs = None

        self._read_teacher_outputs(data_dic)

    def _read_teacher_outputs(self, data_dic):
        # in local: ./teacher_data... in server: 'knowledge_distillation/teacher_data/...
        if data_dic['clean_train_data'] is True:
            csv_file_name = 'knowledge_distillation/teacher_data/clean_train_data_output.csv'
            self.clean_train_outputs = load_teacher_data_from_csv(csv_file_name)

        if data_dic['perturb_train_data'] is True:
            csv_file_name = 'knowledge_distillation/teacher_data/perturb_train_data_output.csv'
            self.perturb_train_outputs = load_teacher_data_from_csv(csv_file_name)

        if data_dic['clean_test_data'] is True:
            csv_file_name = 'knowledge_distillation/teacher_data/clean_test_data_output.csv'
            self.clean_test_outputs = load_teacher_data_from_csv(csv_file_name)

        if data_dic['perturb_test_data'] is True:
            csv_file_name = 'knowledge_distillation/teacher_data/perturb_test_data_output.csv'
            self.perturb_test_outputs = load_teacher_data_from_csv(csv_file_name)

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
        if mode == 'clean_train':
            df = self.clean_train_outputs
        if mode == 'perturb_train':
            df = self.perturb_train_outputs
        if mode == 'clean_test':
            df = self.clean_test_outputs
        if mode == 'perturb_test':
            df = self.perturb_test_outputs
        df_batch = self._get_teacher_output_by_image_indices(df, image_indices)
        return self._generate_prediction_smoothing_outputs(df_batch)
