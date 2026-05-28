'''
src/clustering
==============
Algoritma clustering untuk pengelompokan profil kesehatan provinsi.

Kelas
-----
KMeansClustering    : hard clustering berbasis centroid
GMMClustering       : soft clustering berbasis probabilitas Gaussian

Penggunaan cepat
----------------
from src.clustering import KMeansClustering, GMMClustering

km  = KMeansClustering(n_clusters=4)
labels = km.fit_predict(scaled_feature_matrix)
profiles = km.get_cluster_profiles(feature_mart_df)
'''

from .kmeans import KMeansClustering
from .gmm    import GMMClustering

__all__ = ["KMeansClustering", "GMMClustering"]