'''
src/profiler/content_representation.py
========================================
ContentRepresentation — membangun representasi numerik provinsi
dan menghitung similarity matrix.

Referensi
---------
Aggarwal, C. C. (2016). Recommender Systems: The Textbook.
    Springer. Ch. 4 — Content-Based Recommender Systems.

Catatan desain
--------------
Kelas ini bekerja pada feature mart yang sudah di-scale (output
normalizer.normalize()). Jika di-feed data mentah, normalize_vectors()
harus dipanggil terlebih dahulu.

Reduksi dimensi (PCA, t-SNE, UMAP) bersifat opsional dan ditujukan
untuk eksplorasi / visualisasi — bukan input clustering utama.
'''

import os
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import (
    cosine_similarity    as sk_cosine,
    euclidean_distances,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


PROVINCE_COL = "Provinsi"


class ContentRepresentation:
    '''
    Representasi vektor konten tiap provinsi dan similarity matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Feature mart yang sudah di-scale. Harus memiliki kolom Provinsi.
    '''

    def __init__(self, df: pd.DataFrame) -> None:

        if PROVINCE_COL not in df.columns:
            raise KeyError(f"Kolom '{PROVINCE_COL}' tidak ditemukan.")

        self._df         = df.copy()
        self._provinces  = df[PROVINCE_COL].values
        self._feature_cols = [
            c for c in df.columns
            if c != PROVINCE_COL
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        self._vectors: np.ndarray | None    = None
        self._sim_matrix: np.ndarray | None = None

    # --------------------------------------------------
    # Feature vectors
    # --------------------------------------------------

    def build_feature_vectors(self) -> np.ndarray:
        '''
        Bangun matrix fitur P = [P_1, P_2, ..., P_n]
        di mana P_i = [x_1, x_2, ..., x_m] untuk provinsi i.

        Returns
        -------
        np.ndarray, shape (n_provinces, n_features)
        '''
        self._vectors = (
            self._df[self._feature_cols]
            .values
            .astype(float)
        )
        return self._vectors

    def normalize_vectors(
        self,
        method: str = "minmax"
    ) -> np.ndarray:
        '''
        Normalisasi ulang feature vectors in-place.

        Berguna jika df yang diberikan belum di-scale,
        atau perlu re-normalisasi setelah penggabungan sumber.

        Parameters
        ----------
        method : str
            "minmax" (default) — scale ke [0, 1].
            "l2"               — unit norm per baris (untuk cosine).

        Returns
        -------
        np.ndarray, shape (n_provinces, n_features)
        '''
        if self._vectors is None:
            self.build_feature_vectors()

        if method == "minmax":
            scaler = MinMaxScaler()
            self._vectors = scaler.fit_transform(self._vectors)

        elif method == "l2":
            norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._vectors = self._vectors / norms

        else:
            raise ValueError(f"method harus 'minmax' atau 'l2', bukan '{method}'")

        return self._vectors

    # --------------------------------------------------
    # Similarity
    # --------------------------------------------------

    def compute_similarity_matrix(
        self,
        metric: str = "cosine"
    ) -> pd.DataFrame:
        '''
        Hitung similarity matrix N×N antar semua provinsi.

        Parameters
        ----------
        metric : str
            "cosine" (default) atau "euclidean".

        Returns
        -------
        pd.DataFrame, shape (n_provinces, n_provinces)
            Index dan kolom = nama provinsi.
        '''
        if self._vectors is None:
            self.build_feature_vectors()

        if metric == "cosine":
            sim = self.cosine_similarity()
        elif metric == "euclidean":
            sim = self.euclidean_similarity()
        else:
            raise ValueError(f"metric harus 'cosine' atau 'euclidean'")

        self._sim_matrix = sim

        return pd.DataFrame(
            sim,
            index=self._provinces,
            columns=self._provinces,
        )

    def cosine_similarity(self) -> np.ndarray:
        '''
        Cosine similarity:
            cos(θ) = (A · B) / (|A| |B|)

        Returns
        -------
        np.ndarray, shape (n_provinces, n_provinces)
        '''
        if self._vectors is None:
            self.build_feature_vectors()

        return sk_cosine(self._vectors)

    def euclidean_similarity(self) -> np.ndarray:
        '''
        Similarity berbasis jarak Euclidean:
            sim(A, B) = 1 / (1 + d(A, B))

        Nilai range [0, 1]. Lebih tinggi = lebih mirip.

        Returns
        -------
        np.ndarray, shape (n_provinces, n_provinces)
        '''
        if self._vectors is None:
            self.build_feature_vectors()

        dist = euclidean_distances(self._vectors)
        return 1.0 / (1.0 + dist)

    def get_top_similar(
        self,
        province:  str,
        top_n:     int   = 5,
        metric:    str   = "cosine",
        threshold: float = 0.0,
    ) -> pd.DataFrame:
        '''
        Kembalikan top-N provinsi paling mirip dengan provinsi target.

        Parameters
        ----------
        province : str
            Nama provinsi target.
        top_n : int
            Jumlah tetangga terdekat. Default 5.
        metric : str
            "cosine" atau "euclidean".
        threshold : float
            Hanya tampilkan pasangan dengan similarity ≥ threshold.

        Returns
        -------
        pd.DataFrame
            Kolom: province | similarity_score
        '''
        sim_df = self.compute_similarity_matrix(metric=metric)

        if province not in sim_df.index:
            raise KeyError(
                f"Provinsi '{province}' tidak ditemukan. "
                f"Tersedia: {list(sim_df.index)}"
            )

        row = sim_df.loc[province].drop(labels=province)
        row = row[row >= threshold]

        top = (
            row
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        top.columns = ["province", "similarity_score"]
        top["similarity_score"] = top["similarity_score"].round(4)

        return top

    # --------------------------------------------------
    # Content profiles
    # --------------------------------------------------

    def generate_content_profiles(self) -> pd.DataFrame:
        '''
        Buat semantic tag per provinsi berdasarkan nilai relatif
        tiap dimensi terhadap median nasional.

        Tag yang dihasilkan per fitur:
            "high_{feature}"  jika nilai > median + 0.5 × IQR
            "low_{feature}"   jika nilai < median − 0.5 × IQR
            (tidak di-tag jika berada di tengah)

        Returns
        -------
        pd.DataFrame
            Kolom: provinsi | tags (list) | summary (str)
        '''
        if self._vectors is None:
            self.build_feature_vectors()

        X   = self._vectors
        q25 = np.percentile(X, 25, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        iqr = q75 - q25
        med = np.median(X, axis=0)

        upper = med + 0.5 * iqr
        lower = med - 0.5 * iqr

        rows = []

        for i, prov in enumerate(self._provinces):

            tags = []

            for j, feat in enumerate(self._feature_cols):

                val = X[i, j]

                if val > upper[j]:
                    tags.append(f"high_{feat}")
                elif val < lower[j]:
                    tags.append(f"low_{feat}")

            rows.append({
                PROVINCE_COL: prov,
                "tags":       tags,
                "summary":    ", ".join(tags) if tags else "typical",
            })

        return pd.DataFrame(rows)

    # --------------------------------------------------
    # Dimensionality reduction
    # --------------------------------------------------

    def reduce_dimensions(
        self,
        method:     str = "pca",
        n_components: int = 2,
    ) -> pd.DataFrame:
        '''
        Reduksi dimensi untuk eksplorasi dan visualisasi.

        Parameters
        ----------
        method : str
            "pca"   — Principal Component Analysis (default, cepat).
            "tsne"  — t-SNE (lebih baik untuk visualisasi non-linear).
            "umap"  — UMAP (perlu install umap-learn).
        n_components : int
            Jumlah dimensi target. Default 2.

        Returns
        -------
        pd.DataFrame
            Kolom: Provinsi | dim_1 | dim_2 [| dim_3 ...]
        '''
        if self._vectors is None:
            self.build_feature_vectors()

        X = self._vectors

        if method == "pca":

            reducer = PCA(n_components=n_components, random_state=42)
            reduced = reducer.fit_transform(X)

            explained = reducer.explained_variance_ratio_
            print(
                f"[ContentRepresentation] PCA explained variance: "
                + ", ".join(f"PC{i+1}={v:.1%}" for i, v in enumerate(explained))
            )

        elif method == "tsne":

            try:
                from sklearn.manifold import TSNE
            except ImportError:
                raise ImportError("sklearn sudah terinstal — cek versi.")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reducer = TSNE(
                    n_components = n_components,
                    random_state = 42,
                    perplexity   = min(30, len(X) - 1),
                )
                reduced = reducer.fit_transform(X)

        elif method == "umap":

            try:
                import umap
            except ImportError:
                raise ImportError(
                    "umap-learn tidak terinstal. "
                    "Jalankan: pip install umap-learn"
                )

            reducer = umap.UMAP(
                n_components = n_components,
                random_state = 42,
            )
            reduced = reducer.fit_transform(X)

        else:
            raise ValueError(
                f"method harus 'pca', 'tsne', atau 'umap', bukan '{method}'"
            )

        dim_cols = [f"dim_{i+1}" for i in range(n_components)]
        df_out = pd.DataFrame(reduced, columns=dim_cols)
        df_out.insert(0, PROVINCE_COL, self._provinces)

        return df_out

    # --------------------------------------------------
    # Persistence
    # --------------------------------------------------

    def save_similarity_matrix(self, path: str) -> None:
        '''Simpan similarity matrix ke CSV.'''

        if self._sim_matrix is None:
            self.compute_similarity_matrix()

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        df = pd.DataFrame(
            self._sim_matrix,
            index   = self._provinces,
            columns = self._provinces,
        )
        df.to_csv(path)
        print(f"[ContentRepresentation] Similarity matrix disimpan → {path}")

    def load_similarity_matrix(self, path: str) -> pd.DataFrame:
        '''
        Muat similarity matrix dari CSV.

        Returns
        -------
        pd.DataFrame (n_provinces × n_provinces)
        '''
        df = pd.read_csv(path, index_col=0)
        self._sim_matrix = df.values
        print(f"[ContentRepresentation] Similarity matrix dimuat ← {path}")
        return df