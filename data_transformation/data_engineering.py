from sklearn.impute import KNNImputer
import pandas as pd


class DataEngineering:
    def knn_null_imputer(self, df, n_neighbours=3, exclude_columns=[]):
        if len(exclude_columns) > 0:
            knn_imputer = KNNImputer(n_neighbors=n_neighbours)
            included_columns = list(df.columns)
            for i in exclude_columns:
                included_columns.remove(i)
            df_new = df[included_columns]
            df_new = pd.DataFrame(knn_imputer.fit_transform(df_new), columns=df_new.columns)
            df_new[exclude_columns] = df[exclude_columns]
            return df_new
        else:
            knn_imputer = KNNImputer(n_neighbors=n_neighbours)
            df = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)
            return df

    def check_column_with_single_value(self, df, exclude_columns=[]):
        if len(exclude_columns) > 0:
            included_columns = list(df.columns)
            for i in exclude_columns:
                included_columns.remove(i)
            for column in included_columns:
                if len(df[column].unique()) == 1:
                    df.drop(columns=column, inplace=True)
        else:
            for column in df.columns:
                if len(df[column].unique()) == 1:
                    df.drop(columns=column, inplace=True)
        return df
