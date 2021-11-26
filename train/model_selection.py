import os
import pandas as pd
from sklearn.model_selection import train_test_split
from data_transformation.data_engineering import DataEngineering
from json_read_write.JsonReader import JsonReader
from train.create_best_model import CreateBestModel

from train.create_clusters import Clustering


class ModelSelection:

    def __init__(self):
        self.train_file = os.getcwd() + "/train_test_files/training_files/raw_train_file.csv"
        self.test_file = os.getcwd() + "/train_test_files/test_files/raw_test_file.csv"
        self.data_engineering = DataEngineering()
        self.json_reader = JsonReader()
        self.clustering = Clustering()
        self.create_best_model = CreateBestModel()

    def clean_training_data(self, sep=","):
        self.df = pd.read_csv(self.train_file, sep=sep)
        self.df = self.data_engineering.knn_null_imputer(df=self.df, exclude_columns=["Wafer", "Good/Bad"])
        self.df.to_csv(os.getcwd() + "/train_test_files/training_files/clean_train_file.csv", index=None)

    def clean_test_data(self, sep=","):
        self.df = pd.read_csv(self.test_file, sep=sep)
        self.df = self.data_engineering.knn_null_imputer(df=self.df, exclude_columns=["Wafer"])
        self.df.to_csv(os.getcwd() + "/train_test_files/test_files/clean_test_file.csv", index=None)

    def generate_best_model_for_training(self):
        df = pd.read_csv(os.getcwd() + "/train_test_files/training_files/clean_train_file.csv")
        df = df.drop(
            columns=self.json_reader.get_value_from_json_file(os.getcwd() + "/schema_training.json", "exclude_column"))
        self.x = df.drop(
            columns=self.json_reader.get_value_from_json_file(os.getcwd() + "/schema_training.json", "outcome_column"))
        self.y = df[self.json_reader.get_value_from_json_file(os.getcwd() + "/schema_training.json", "outcome_column")]
        self.x, self.y = self.create_best_model.handle_imbalanced_dataset(self.x, self.y, 1)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, train_size=.75)

    def create_clusters_and_select_best_model(self):
        self.no_of_clusters = self.clustering.get_cluster_count(self.x)
        self.x = self.clustering.create_cluster_groups(self.x)
        df_with_cluster_group = self.x
        df_with_cluster_group[
            self.json_reader.get_value_from_json_file(os.getcwd() + "/schema_training.json", "outcome_column")] = self.y
        no_of_clusters = self.x["cluster_group"].unique()
        for cluster in no_of_clusters:
            outcome_column = self.json_reader.get_value_from_json_file(os.getcwd() + "/schema_training.json",
                                                                       "outcome_column")
            include_columns = df_with_cluster_group.drop(columns=["cluster_group", outcome_column]).columns
            df_temp = df_with_cluster_group[df_with_cluster_group["cluster_group"] == cluster]
            x, y = df_temp[include_columns], df_temp[outcome_column]
            x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                                test_size=.25)
            self.create_best_model.get_best_model(x_train, x_test, y_train, y_test, cluster)
