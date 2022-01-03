import pandas as pd

clean_data_list = ['clean_data_output_train_Jan-02-2022.csv', 'clean_data_output_train_Jan-03-2022.csv']
perturb_data_list = ['perturb_data_output_train_Jan-02-2022.csv', 'perturb_data_output_train_Jan-03-2022.csv']

combined_clean_data = pd.concat([pd.read_csv(f) for f in clean_data_list])
combined_perturb_data = pd.concat([pd.read_csv(f) for f in perturb_data_list])

combined_clean_data.to_csv("clean_train_data_output.csv", index=False)
combined_perturb_data.to_csv("perturb_train_data_output.csv", index=False)