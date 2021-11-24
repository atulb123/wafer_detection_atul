import os

from data_validation.check_data_attributes import CheckDataAttributes
from train.model_selection import ModelSelection


class TrainFLow:

    def __init__(self):
        self.file_name_check = CheckDataAttributes(os.getcwd() + "/train_test_files/train_raw_dataset", ",")
        self.model_selection = ModelSelection()

    def train_flow(self):
        self.file_name_check.check_file_name()
        self.file_name_check.check_column_count_and_null(type_of_test="train")
        self.file_name_check.consolidate_data(type_of_test="train")
        self.model_selection.clean_training_data()
        self.model_selection.generate_best_model_for_training()
        self.model_selection.create_clusters_and_select_best_model()
