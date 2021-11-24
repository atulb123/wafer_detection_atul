import os

from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import pickle


class CreateBestModel:
    def get_best_model_parameters(self, x_train, y_train):
        random_forest_model = RandomForestClassifier()
        xg_boost_model = XGBClassifier()
        random_forest_param = {"criterion": ["gini", "entropy"]}
        xg_boost_param = {'learning_rate': [0.5, 0.1, 0.01, 0.001]}
        res = dict()
        for model in [(random_forest_model, "rf", random_forest_param),
                      (xg_boost_model, "xg", xg_boost_param)]:
            grid_search_cv = GridSearchCV(model[0], model[-1])
            grid_search_cv.fit(x_train, y_train)
            res[model[1]] = grid_search_cv.best_params_
        return res

    def get_best_model(self, x_train, x_test, y_train, y_test, cluster_number):
        res = self.get_best_model_parameters(x_train, y_train)
        rf_train_model = RandomForestClassifier(criterion=res.get("rf").get('criterion'))
        xg_train_model = XGBClassifier(learning_rate=res.get('xg').get('learning_rate'),
                                       objective='binary:logistic')
        rf_train_model.fit(x_train, y_train)
        xg_train_model.fit(x_train, y_train)
        rf_y_pred = rf_train_model.predict(x_test)
        xg_y_pred = xg_train_model.predict(x_test)
        if len(y_test.unique()) == 1:
            print("RF_ACCURACY_SCORE", accuracy_score(y_test, rf_y_pred))
            print("XG_ACCURACY_SCORE", accuracy_score(y_test, xg_y_pred))
            best_model = "rf" if accuracy_score(y_test, rf_y_pred) >= accuracy_score(y_test, xg_y_pred) else "xg"
        else:
            print("RF_ROC_AUC_SCORE", roc_auc_score(y_test, rf_y_pred))
            print("XG_ROC_AUC_SCORE", roc_auc_score(y_test, xg_y_pred))
            best_model = "rf" if roc_auc_score(y_test, rf_y_pred) >= roc_auc_score(y_test, xg_y_pred) else "xg"
        if best_model == "xg":
            with open(
                    os.getcwd() + "/trained_models/prediction_models/xg_model_cluster_" + str(cluster_number) + ".sav",
                    "wb") as f:
                pickle.dump(xg_train_model, f)
        else:
            with open(
                    os.getcwd() + "/trained_models/prediction_models/rf_model_cluster_" + str(cluster_number) + ".sav",
                    "wb") as f:
                pickle.dump(rf_train_model, f)

    def handle_imbalanced_dataset(self, x, y, ratio):
        ros = RandomOverSampler(ratio)
        x_new, y_new = ros.fit_resample(x, y)
        return x_new, y_new
