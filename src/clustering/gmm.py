'''
src/clustering/gmm.py
======================
GMMClustering — soft clustering berbasis Gaussian Mixture Model.

Referensi
---------
Aggarwal, C. C. (2016). Recommender Systems: The Textbook.
    Springer. Ch. 7.

McLachlan, G. J., & Peel, D. (2000). Finite Mixture Models.
    Wiley Series in Probability and Statistics.

Catatan desain — soft vs hard clustering
-----------------------------------------
GMM menghasilkan probabilitas membership, bukan label tunggal.
Provinsi bisa "sebagian" masuk dua cluster — ini berguna untuk
rekomendasi kebijakan di wilayah transisional.

get_soft_profiles() memanfaatkan probabilitas ini untuk menghasilkan
weighted summary per cluster, berbeda dari get_cluster_profiles()
KMeans yang strict per label.

Pemilihan covariance_type
--------------------------
"full"    : setiap cluster punya matriks kovarians sendiri.
            Paling fleksibel, paling banyak parameter.
"tied"    : semua cluster berbagi satu matriks kovarians.
"diag"    : diagonal saja — fitur dianggap independen per cluster.
"spherical": satu variance per cluster. Paling sederhana, mirip K-Means.

Default "full" untuk data kesehatan yang cenderung punya korelasi
antar dimensi (misalnya daerah dengan barrier tinggi cenderung
punya workforce rendah).
'''

import os
import warnings
import pickle

import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


# ---------------------------------------------------------
# Konstanta default
# ---------------------------------------------------------

DEFAULT_N_COMPONENTS  = 4
DEFAULT_COV_TYPE      = "full"
DEFAULT_MAX_ITER      = 200
DEFAULT_RANDOM_STATE  = 42
DEFAULT_REG_COVAR     = 1e-6
BIC_K_RANGE           = range(2, 11)

COV_TYPES = ["full", "tied", "diag", "spherical"]


class GMMClustering:
    '''
    Gaussian Mixture Model clustering untuk profil kesehatan provinsi.

    Parameters
    ----------
    n_components : int
        Jumlah komponen Gaussian (cluster). Default 4.
    covariance_type : str
        Tipe matriks kovarians: "full" | "tied" | "diag" | "spherical".
        Default "full".
    max_iter : int
        Maksimum iterasi EM. Default 200.
    random_state : int
        Seed reprodusibilitas. Default 42.
    reg_covar : float
        Regularisasi diagonal kovarians untuk kestabilan numerik.
        Default 1e-6.
    '''

    def __init__(
        self,
        n_components:    int   = DEFAULT_N_COMPONENTS,
        covariance_type: str   = DEFAULT_COV_TYPE,
        max_iter:        int   = DEFAULT_MAX_ITER,
        random_state:    int   = DEFAULT_RANDOM_STATE,
        reg_covar:       float = DEFAULT_REG_COVAR,
    ) -> None:

        if covariance_type not in COV_TYPES:
            raise ValueError(
                f"covariance_type harus salah satu dari {COV_TYPES}, "
                f"bukan '{covariance_type}'"
            )

        self.n_components    = n_components
        self.covariance_type = covariance_type
        self.max_iter        = max_iter
        self.random_state    = random_state
        self.reg_covar       = reg_covar

        self._model:  GaussianMixture | None = None
        self._labels: np.ndarray | None      = None
        self._proba:  np.ndarray | None      = None
        self._X:      np.ndarray | None      = None

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _to_array(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.select_dtypes(include=np.number).values.astype(float)
        return np.array(X, dtype=float)

    def _assert_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "Model belum dilatih. Panggil fit() atau fit_predict()."
            )

    def _build_model(self) -> GaussianMixture:
        return GaussianMixture(
            n_components    = self.n_components,
            covariance_type = self.covariance_type,
            max_iter        = self.max_iter,
            random_state    = self.random_state,
            reg_covar       = self.reg_covar,
        )

    # --------------------------------------------------
    # Core
    # --------------------------------------------------

    def fit(self, X) -> "GMMClustering":
        '''
        Latih model GMM menggunakan algoritma Expectation-Maximization.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_provinces, n_features)

        Returns
        -------
        self
        '''
        self._X     = self._to_array(X)
        self._model = self._build_model()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(self._X)

        self._labels = self._model.predict(self._X)
        self._proba  = self._model.predict_proba(self._X)

        if not self._model.converged_:
            warnings.warn(
                f"[GMMClustering] EM tidak konvergen dalam {self.max_iter} iterasi. "
                f"Pertimbangkan menambah max_iter atau mengubah covariance_type.",
                UserWarning,
            )

        return self

    def predict(self, X) -> np.ndarray:
        '''
        Hard assignment: kembalikan label cluster untuk data baru.

        Returns
        -------
        np.ndarray of int, shape (n_samples,)
        '''
        self._assert_fitted()
        return self._model.predict(self._to_array(X))

    def predict_proba(self, X) -> np.ndarray:
        '''
        Soft assignment: probabilitas membership per cluster.

        Returns
        -------
        np.ndarray, shape (n_samples, n_components)
            Setiap baris menjumlah ke 1.0.
        '''
        self._assert_fitted()
        return self._model.predict_proba(self._to_array(X))

    def fit_predict(self, X) -> np.ndarray:
        '''
        Latih dan kembalikan label cluster (hard assignment).

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
        Metrik kualitas clustering pada data training.

        Returns
        -------
        dict
            silhouette_score    : [-1, 1]
            davies_bouldin      : [0, ∞]
            calinski_harabasz   : [0, ∞]
            bic                 : Bayesian Information Criterion
            aic                 : Akaike Information Criterion
            converged           : bool
            n_iter              : jumlah iterasi EM yang dibutuhkan
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
            "bic":               round(float(self._model.bic(self._X)),  4),
            "aic":               round(float(self._model.aic(self._X)),  4),
            "converged":         bool(self._model.converged_),
            "n_iter":            int(self._model.n_iter_),
            "n_components":      self.n_components,
            "covariance_type":   self.covariance_type,
        }

    def bic_analysis(
        self,
        X,
        k_range:       range = BIC_K_RANGE,
        cov_types:     list  = None,
    ) -> pd.DataFrame:
        '''
        Hitung BIC untuk berbagai (k, covariance_type).

        BIC lebih rendah = model lebih baik (penalti kompleksitas lebih ketat).

        Parameters
        ----------
        X : array-like
        k_range : range
        cov_types : list of str
            Default: semua 4 tipe kovarians.

        Returns
        -------
        pd.DataFrame
            Kolom: k | covariance_type | bic
        '''
        X_arr     = self._to_array(X)
        cov_types = cov_types or COV_TYPES
        rows      = []

        for k in k_range:
            for cov in cov_types:
                try:
                    gm = GaussianMixture(
                        n_components    = k,
                        covariance_type = cov,
                        max_iter        = self.max_iter,
                        random_state    = self.random_state,
                        reg_covar       = self.reg_covar,
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        gm.fit(X_arr)

                    rows.append({
                        "k":               k,
                        "covariance_type": cov,
                        "bic":             round(float(gm.bic(X_arr)), 4),
                    })
                except Exception:
                    pass

        return (
            pd.DataFrame(rows)
            .sort_values("bic")
            .reset_index(drop=True)
        )

    def aic_analysis(
        self,
        X,
        k_range:   range = BIC_K_RANGE,
        cov_types: list  = None,
    ) -> pd.DataFrame:
        '''
        Hitung AIC untuk berbagai (k, covariance_type).

        AIC lebih rendah = fit lebih baik.
        Kurang keras dibanding BIC — cenderung memilih k lebih besar.

        Returns
        -------
        pd.DataFrame
            Kolom: k | covariance_type | aic
        '''
        X_arr     = self._to_array(X)
        cov_types = cov_types or COV_TYPES
        rows      = []

        for k in k_range:
            for cov in cov_types:
                try:
                    gm = GaussianMixture(
                        n_components    = k,
                        covariance_type = cov,
                        max_iter        = self.max_iter,
                        random_state    = self.random_state,
                        reg_covar       = self.reg_covar,
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        gm.fit(X_arr)

                    rows.append({
                        "k":               k,
                        "covariance_type": cov,
                        "aic":             round(float(gm.aic(X_arr)), 4),
                    })
                except Exception:
                    pass

        return (
            pd.DataFrame(rows)
            .sort_values("aic")
            .reset_index(drop=True)
        )

    # --------------------------------------------------
    # Tuning
    # --------------------------------------------------

    def tune_hyperparameters(
        self,
        X,
        k_range:        range = BIC_K_RANGE,
        cov_types:      list  = None,
        reg_covar_list: list  = None,
        metric:         str   = "bic",
    ) -> dict:
        '''
        Grid search atas (k, covariance_type, reg_covar).

        Parameters
        ----------
        metric : str
            "bic" (default) atau "aic".
            Untuk BIC/AIC, best = minimum score.

        Returns
        -------
        dict
            best_k | best_covariance_type | best_reg_covar
            | best_score | metric | results_df
        '''
        X_arr          = self._to_array(X)
        cov_types      = cov_types      or COV_TYPES
        reg_covar_list = reg_covar_list or [1e-6, 1e-4, 1e-2]
        rows           = []

        for k in k_range:
            for cov in cov_types:
                for reg in reg_covar_list:
                    try:
                        gm = GaussianMixture(
                            n_components    = k,
                            covariance_type = cov,
                            max_iter        = self.max_iter,
                            random_state    = self.random_state,
                            reg_covar       = reg,
                        )
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            gm.fit(X_arr)

                        score = (
                            gm.bic(X_arr) if metric == "bic"
                            else gm.aic(X_arr)
                        )

                        rows.append({
                            "k":               k,
                            "covariance_type": cov,
                            "reg_covar":       reg,
                            "score":           round(float(score), 4),
                        })
                    except Exception:
                        pass

        results_df = pd.DataFrame(rows)

        # BIC/AIC: lebih rendah lebih baik
        best_row = results_df.loc[results_df["score"].idxmin()]

        return {
            "best_k":               int(best_row["k"]),
            "best_covariance_type": str(best_row["covariance_type"]),
            "best_reg_covar":       float(best_row["reg_covar"]),
            "best_score":           float(best_row["score"]),
            "metric":               metric,
            "results_df":           results_df,
        }

    # --------------------------------------------------
    # Profiling
    # --------------------------------------------------

    def get_cluster_probabilities(
        self,
        province_col: str       = "Provinsi",
        provinces:    list[str] = None,
    ) -> pd.DataFrame:
        '''
        Distribusi probabilitas membership per provinsi.

        Parameters
        ----------
        province_col : str
        provinces : list of str
            Nama provinsi sesuai urutan baris X. Opsional.

        Returns
        -------
        pd.DataFrame
            Kolom: [provinsi?] | cluster_0 | cluster_1 | ... | hard_label
        '''
        self._assert_fitted()

        n = len(self._proba)
        cols = {
            f"cluster_{i}": np.round(self._proba[:, i], 4)
            for i in range(self.n_components)
        }
        cols["hard_label"] = self._labels

        df = pd.DataFrame(cols)

        if provinces is not None and len(provinces) == n:
            df.insert(0, province_col, provinces)

        return df

    def get_soft_profiles(
        self,
        feature_df:   pd.DataFrame,
        province_col: str = "Provinsi",
    ) -> pd.DataFrame:
        '''
        Weighted cluster profiles — tiap baris provinsi dikontribusikan
        ke semua cluster sesuai probabilitasnya.

        Berbeda dari hard profiling KMeans yang strict per label,
        soft profiles menangkap wilayah transisional dengan lebih baik.

        Parameters
        ----------
        feature_df : pd.DataFrame
            DataFrame numerik dengan urutan baris sama seperti X.

        Returns
        -------
        pd.DataFrame
            Index = cluster_id (0-based).
            Kolom = fitur numerik.
            Nilai = weighted mean berdasarkan probabilitas.
        '''
        self._assert_fitted()

        df = feature_df.copy().reset_index(drop=True)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        profiles = {}

        for k in range(self.n_components):
            weights = self._proba[:, k]                     # shape (n,)
            w_sum   = weights.sum()

            if w_sum == 0:
                profiles[k] = {c: np.nan for c in numeric_cols}
            else:
                profiles[k] = {
                    c: round(
                        float(np.average(df[c].values, weights=weights)),
                        4
                    )
                    for c in numeric_cols
                }

        return pd.DataFrame(profiles).T.rename_axis("cluster_id")

    # --------------------------------------------------
    # Persistence
    # --------------------------------------------------

    def save_model(self, path: str) -> None:
        '''Simpan model ke file pickle.'''
        self._assert_fitted()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)

        print(f"[GMMClustering] Model disimpan → {path}")

    @classmethod
    def load_model(cls, path: str) -> "GMMClustering":
        '''Muat model dari file pickle.'''
        with open(path, "rb") as f:
            obj = pickle.load(f)

        if not isinstance(obj, cls):
            raise TypeError(f"File bukan GMMClustering: {type(obj)}")

        print(f"[GMMClustering] Model dimuat ← {path}")
        return obj