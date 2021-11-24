import pickle, os
import pandas as pd


class PredictTestFlow:
    def predict_scores_for_test_dataset(self):
        self.df = pd.read_csv(os.getcwd() + "/train_test_files/test_files/clean_test_file.csv")
        self.df = self.get_clustering_group(self.df, ["Wafer"])
        self.get_predictions(self.df, exclude_column=["cluster_group", "Wafer"])

    def get_clustering_group(self, df, exclude_column=[]):
        with open(os.getcwd() + "/trained_models/clustering_model/kmean_model.sav", "rb") as f:
            knn_model = pickle.load(f)
            df["cluster_group"] = knn_model.fit_predict(df.drop(columns=exclude_column))
            return df

    def get_predictions(self, df, exclude_column=[]):
        y_pred = []
        for i in range(df.shape[0]):
            with open(os.getcwd() + "/trained_models/prediction_models/rf_model_cluster_" + str(
                    df["cluster_group"][i]) + ".sav", "rb") as f:
                pred_model = pickle.load(f)
                y_pred.append(pred_model.predict([list(df.drop(columns=exclude_column).iloc[i, :])])[0])
        df["Good/Bad"] = y_pred
        df.drop(columns="cluster_group", inplace=True)
        df.to_csv(os.getcwd() + "/train_test_files/test_files/prediction_output.csv", index=None)
