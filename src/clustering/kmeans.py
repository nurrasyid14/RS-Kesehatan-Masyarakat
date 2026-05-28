'''
src/clustering/kmeans.py
=========================
KMeansClustering — hard clustering berbasis centroid K-Means.

Referensi
---------
Aggarwal, C. C. (2016). Recommender Systems: The Textbook.
    Springer. Ch. 7 — Clustering-Based Collaborative Filtering.

MacQueen, J. (1967). Some methods for classification and analysis
    of multivariate observations. Proceedings of the 5th Berkeley
    Symposium on Mathematical Statistics and Probability.

Catatan desain
--------------
- Input selalu berupa feature matrix numerik (numpy array atau DataFrame).
  Kolom Provinsi harus sudah dipisahkan sebelum masuk fit().
- Elbow dan silhouette analysis dijalankan secara terpisah dari fit()
  agar tidak memperlambat pipeline utama.
- get_cluster_profiles() mengembalikan statistik per cluster
  dalam format yang bisa langsung dikonsumsi RegionalProfiler.
'''

import os
import warnings
import pickle

import numpy as np
import pandas as pd

from sklearn.cluster         import KMeans
from sklearn.metrics         import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    silhouette_samples,
)


# ---------------------------------------------------------
# Konstanta default
# ---------------------------------------------------------

DEFAULT_N_CLUSTERS   = 4
DEFAULT_MAX_ITER     = 300
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_INIT       = 10
ELBOW_K_RANGE        = range(2, 11)


class KMeansClustering:
    '''
    K-Means clustering untuk pengelompokan profil kesehatan provinsi.

    Parameters
    ----------
    n_clusters : int
        Jumlah cluster. Default 4.
    max_iter : int
        Maksimum iterasi per run. Default 300.
    random_state : int
        Seed untuk reprodusibilitas. Default 42.
    n_init : int
        Jumlah inisialisasi berbeda yang dijalankan. Default 10.
    '''

    def __init__(
        self,
        n_clusters:   int = DEFAULT_N_CLUSTERS,
        max_iter:     int = DEFAULT_MAX_ITER,
        random_state: int = DEFAULT_RANDOM_STATE,
        n_init:       int = DEFAULT_N_INIT,
    ) -> None:

        self.n_clusters   = n_clusters
        self.max_iter     = max_iter
        self.random_state = random_state
        self.n_init       = n_init

        self._model:  KMeans | None    = None
        self._labels: np.ndarray | None = None
        self._X:      np.ndarray | None = None

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _to_array(self, X) -> np.ndarray:
        '''Konversi DataFrame/array ke numpy 2D float array.'''
        if isinstance(X, pd.DataFrame):
            return X.select_dtypes(include=np.number).values.astype(float)
        return np.array(X, dtype=float)

    def _assert_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "Model belum dilatih. Panggil fit() atau fit_predict() terlebih dahulu."
            )

    # --------------------------------------------------
    # Core
    # --------------------------------------------------

    def fit(self, X) -> "KMeansClustering":
        '''
        Latih model K-Means.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_provinces, n_features)
            Feature matrix yang sudah di-scale.

        Returns
        -------
        self
        '''
        self._X = self._to_array(X)

        self._model = KMeans(
            n_clusters   = self.n_clusters,
            max_iter     = self.max_iter,
            random_state = self.random_state,
            n_init       = self.n_init,
        )
        self._model.fit(self._X)
        self._labels = self._model.labels_

        return self

    def predict(self, X) -> np.ndarray:
        '''
        Prediksi cluster untuk data baru.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of int, shape (n_samples,)
        '''
        self._assert_fitted()
        return self._model.predict(self._to_array(X))

    def fit_predict(self, X) -> np.ndarray:
        '''
        Latih model dan kembalikan label cluster sekaligus.

        Returns
        -------
        np.ndarray of int, shape (n_provinces,)
        '''
        self.fit(X)
        return self._labels

    # --------------------------------------------------
    # Evaluasi
    # --------------------------------------------------

    def evaluate_clustering(self) -> dict:
        '''
        Hitung metrik kualitas clustering pada data training.

        Returns
        -------
        dict
            silhouette_score    : [-1, 1], lebih tinggi lebih baik
            davies_bouldin      : [0, ∞],  lebih rendah lebih baik
            calinski_harabasz   : [0, ∞],  lebih tinggi lebih baik
            inertia             : WCSS K-Means
            n_clusters          : jumlah cluster aktual
        '''
        self._assert_fitted()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sil = silhouette_score(self._X, self._labels)
            db  = davies_bouldin_score(self._X, self._labels)
            ch  = calinski_harabasz_score(self._X, self._labels)

        return {
            "silhouette_score":  round(float(sil), 4),
            "davies_bouldin":    round(float(db),  4),
            "calinski_harabasz": round(float(ch),  4),
            "inertia":           round(float(self._model.inertia_), 4),
            "n_clusters":        self.n_clusters,
        }

    def elbow_method(
        self,
        X,
        k_range: range = ELBOW_K_RANGE,
    ) -> pd.DataFrame:
        '''
        Hitung WCSS (inertia) untuk berbagai nilai k.

        Digunakan untuk visualisasi elbow curve — pemilihan k optimal
        diserahkan ke pengguna / RegionalProfiler.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        k_range : range
            Rentang nilai k yang diuji. Default range(2, 11).

        Returns
        -------
        pd.DataFrame
            Kolom: k | inertia
        '''
        X_arr = self._to_array(X)
        results = []

        for k in k_range:
            km = KMeans(
                n_clusters   = k,
                max_iter     = self.max_iter,
                random_state = self.random_state,
                n_init       = self.n_init,
            )
            km.fit(X_arr)
            results.append({"k": k, "inertia": round(km.inertia_, 4)})

        return pd.DataFrame(results)

    def silhouette_analysis(
        self,
        X,
        k_range: range = ELBOW_K_RANGE,
    ) -> pd.DataFrame:
        '''
        Hitung silhouette score rata-rata dan per-sample
        untuk berbagai nilai k.

        Returns
        -------
        pd.DataFrame
            Kolom: k | mean_silhouette | min_silhouette | max_silhouette
        '''
        X_arr   = self._to_array(X)
        results = []

        for k in k_range:
            km = KMeans(
                n_clusters   = k,
                random_state = self.random_state,
                n_init       = self.n_init,
            )
            labels = km.fit_predict(X_arr)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                samples = silhouette_samples(X_arr, labels)

            results.append({
                "k":               k,
                "mean_silhouette": round(float(samples.mean()), 4),
                "min_silhouette":  round(float(samples.min()),  4),
                "max_silhouette":  round(float(samples.max()),  4),
            })

        return pd.DataFrame(results)

    # --------------------------------------------------
    # Tuning
    # --------------------------------------------------

    def tune_hyperparameters(
        self,
        X,
        k_range:    range = ELBOW_K_RANGE,
        n_init_list: list = [10, 20, 30],
        metric:     str   = "silhouette",
    ) -> dict:
        '''
        Grid search sederhana atas (k, n_init).

        Parameters
        ----------
        X : array-like
        k_range : range
            Nilai k yang diuji.
        n_init_list : list of int
            Nilai n_init yang diuji.
        metric : str
            "silhouette" (default) atau "calinski_harabasz".

        Returns
        -------
        dict
            best_k | best_n_init | best_score | results_df
        '''
        X_arr = self._to_array(X)
        rows  = []

        for k in k_range:
            for n_init in n_init_list:

                km = KMeans(
                    n_clusters   = k,
                    max_iter     = self.max_iter,
                    random_state = self.random_state,
                    n_init       = n_init,
                )
                labels = km.fit_predict(X_arr)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    if metric == "silhouette":
                        score = silhouette_score(X_arr, labels)
                    else:
                        score = calinski_harabasz_score(X_arr, labels)

                rows.append({
                    "k":      k,
                    "n_init": n_init,
                    "score":  round(float(score), 4),
                })

        results_df = pd.DataFrame(rows)

        best_row = results_df.loc[results_df["score"].idxmax()]

        return {
            "best_k":      int(best_row["k"]),
            "best_n_init": int(best_row["n_init"]),
            "best_score":  float(best_row["score"]),
            "metric":      metric,
            "results_df":  results_df,
        }

    # --------------------------------------------------
    # Profiling
    # --------------------------------------------------

    def get_cluster_centroids(self) -> pd.DataFrame:
        '''
        Kembalikan centroid tiap cluster.

        Returns
        -------
        pd.DataFrame
            Index = cluster_id (0-based).
            Kolom = fitur (kolom integer jika feature names tidak tersedia).
        '''
        self._assert_fitted()
        return pd.DataFrame(
            self._model.cluster_centers_,
            index=pd.Index(range(self.n_clusters), name="cluster_id"),
        )

    def get_cluster_profiles(
        self,
        feature_df:   pd.DataFrame,
        province_col: str = "Provinsi",
    ) -> pd.DataFrame:
        '''
        Statistik deskriptif per cluster dari feature DataFrame asli.

        Parameters
        ----------
        feature_df : pd.DataFrame
            DataFrame dengan kolom numerik dan opsional kolom Provinsi.
            Urutan baris harus sama dengan X yang di-fit.
        province_col : str
            Nama kolom provinsi. Jika tidak ada, diabaikan.

        Returns
        -------
        pd.DataFrame
            MultiIndex (cluster_id, stat) di index,
            kolom = fitur numerik.
            stat : mean | std | min | max | count
        '''
        self._assert_fitted()

        df = feature_df.copy().reset_index(drop=True)
        df["cluster_id"] = self._labels

        numeric_cols = [
            c for c in df.select_dtypes(include=np.number).columns
            if c != "cluster_id"
        ]

        profiles = (
            df.groupby("cluster_id")[numeric_cols]
            .agg(["mean", "std", "min", "max", "count"])
        )

        return profiles

    # --------------------------------------------------
    # Persistence
    # --------------------------------------------------

    def save_model(self, path: str) -> None:
        '''
        Simpan model ke file pickle.

        Parameters
        ----------
        path : str
            Path output, e.g. "models/kmeans_k4.pkl"
        '''
        self._assert_fitted()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)

        print(f"[KMeansClustering] Model disimpan → {path}")

    @classmethod
    def load_model(cls, path: str) -> "KMeansClustering":
        '''
        Muat model dari file pickle.

        Parameters
        ----------
        path : str

        Returns
        -------
        KMeansClustering
        '''
        with open(path, "rb") as f:
            obj = pickle.load(f)

        if not isinstance(obj, cls):
            raise TypeError(f"File bukan KMeansClustering: {type(obj)}")

        print(f"[KMeansClustering] Model dimuat ← {path}")
        return obj