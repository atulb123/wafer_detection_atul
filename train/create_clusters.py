import os

from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt
import pickle


class Clustering:
    def get_cluster_count(self, data):
        wcss = []
        for i in range(1, 12):
            kmean = KMeans(n_clusters=i)
            kmean.fit(data)
            wcss.append(kmean.inertia_)
        plt.plot(range(1, 12), wcss)
        plt.savefig(os.getcwd() + "/trained_models/clustering_model/knee_plot.png")
        self.cluster_count = int(KneeLocator(range(1, 12), wcss, curve='convex', direction='decreasing').knee)
        return self.cluster_count

    def create_cluster_groups(self, x):
        knn_model = KMeans(self.cluster_count)
        clusters = knn_model.fit_predict(x)
        x["cluster_group"] = clusters
        self.save_model(knn_model, os.getcwd() + "/trained_models/clustering_model/kmean_model.sav", "wb")
        return x

    def save_model(self, model, path, mode):
        with open(path, mode) as fb:
            pickle.dump(model,fb)
