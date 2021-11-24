from data_validation.check_data_attributes import CheckDataAttributes
import os

from test.predict_test_flow import PredictTestFlow
from train.model_selection import ModelSelection


class TestFlow:
    def __init__(self):
        self.file_name_check = CheckDataAttributes(os.getcwd() + "/train_test_files/test_raw_dataset", ",")
        self.model_selection = ModelSelection()
        self.predict_test_flow = PredictTestFlow()

    def test_flow(self):
        self.file_name_check.check_file_name()
        self.file_name_check.check_column_count_and_null(type_of_test="test")
        self.file_name_check.consolidate_data(type_of_test="test")
        self.model_selection.clean_test_data()
        self.predict_test_flow.predict_scores_for_test_dataset()
