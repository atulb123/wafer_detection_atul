import pandas as pd
import os, re
import shutil
from json_read_write.JsonReader import JsonReader
from os_level_operations.create_delete_folder import CreateDeleteFolder
from os_level_operations.read_folder import ReadFolder


class CheckDataAttributes:

    def __init__(self, path, sep):
        self.json_reader = JsonReader()
        self.file_path = path
        self.sep = sep
        self.create_delete_folder = CreateDeleteFolder()
        self.training_data_schema = os.getcwd() + "/schema_training.json"
        self.test_data_schema = os.getcwd() + "/schema_prediction.json"
        self.read_folder = ReadFolder()

    def check_column_count(self, file_name, reference_file_path):
        df = pd.read_csv(self.file_path + "/" + file_name)
        return len(df.columns) == len(
            self.json_reader.get_value_from_json_file(reference_file_path, "ColName").keys())

    def check_complete_column_is_not_null(self, file_name):
        df = pd.read_csv(self.file_path + "/" + file_name)
        for column in df.columns:
            if df[column].isnull().sum() == pd.read_csv(self.file_path + "/" + file_name).shape[0]:
                import ipdb
                ipdb.set_trace()
                return False
            else:
                pass
        return True

    def check_file_name(self):
        try:
            self.create_delete_folder.create_folders(["good_files", "bad_files"])
            list_of_files = self.read_folder.return_list_of_file(self.file_path)
            for file in list_of_files:
                if re.match(self.json_reader.get_value_from_json_file(self.training_data_schema, "file_name_regex"),
                            file) == None:
                    shutil.copy(self.file_path + "/" + file, os.getcwd() + "/train_test_files/bad_files/" + file)
                else:
                    shutil.copy(self.file_path + "/" + file, os.getcwd() + "/train_test_files/good_files/" + file)
        except BaseException as msg:
            raise BaseException(msg)

    def check_column_count_and_null(self, type_of_test="train"):
        reference_path = self.training_data_schema if type_of_test == "train" else self.test_data_schema
        try:
            list_of_files = self.read_folder.return_list_of_file(os.getcwd() + "/train_test_files/good_files")
            for file in list_of_files:
                if self.check_column_count(file, reference_path) and self.check_complete_column_is_not_null(file):
                    pass
                else:
                    shutil.move(os.getcwd() + "/train_test_files/good_files/" + file,
                                os.getcwd() + "/train_test_files/bad_files/" + file)
        except BaseException as msg:
            raise BaseException(msg)

    def consolidate_data(self, type_of_test="train"):
        try:
            list_of_dataframes = []
            list_of_files = self.read_folder.return_list_of_file(os.getcwd() + "/train_test_files/good_files")
            for file in list_of_files:
                list_of_dataframes.append(
                    pd.read_csv(os.getcwd() + "/train_test_files/good_files/" + file, sep=self.sep))
            df = pd.concat(list_of_dataframes, axis=0)
            if type_of_test == "train":
                df.to_csv(os.getcwd() + "/train_test_files/training_files/raw_train_file.csv", index=None)
            else:
                df.to_csv(os.getcwd() + "/train_test_files/test_files/raw_test_file.csv", index=None)
        except BaseException as msg:
            raise BaseException(msg)
