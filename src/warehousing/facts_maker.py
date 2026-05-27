'''
src/warehousing/facts_maker.py
================================
FactsMaker — membangun fact tables dari engineered & scaled DataFrame.

Fact tables yang dihasilkan
----------------------------
fact_healthcare_raw          : long-format semua indikator per provinsi
fact_healthcare_feature_mart : composite index per provinsi (wide)
fact_clustering              : placeholder untuk hasil clustering
fact_similarity              : cosine similarity antar provinsi
fact_recommendation          : placeholder untuk hasil rekomendasi

Catatan desain — fact_clustering & fact_recommendation
--------------------------------------------------------
Kedua tabel ini diinisialisasi sebagai placeholder kosong (skema saja)
karena nilainya baru tersedia setelah modul clustering/ dan recommender/
berjalan. FactsMaker menyediakan method update untuk mengisi keduanya
setelah pipeline downstream selesai.
'''

import numpy as np
import pandas as pd

from .features_mart   import build_feature_mart, FEATURE_MART_COLS
from .similarity_score import build_similarity

PROVINCE_COL = "Provinsi"


class FactsMaker:
    '''
    Membangun fact tables dari hasil prep pipeline.

    Parameters
    ----------
    engineered_df : pd.DataFrame
        Output feature_engineer.engineer() — kolom mentah + index.
    scaled_df : pd.DataFrame
        Output normalizer.normalize() — versi ter-scale dari engineered_df.

    Penggunaan
    ----------
    maker = FactsMaker(engineered_df, scaled_df)
    facts = maker.make_all_facts()

    facts["raw"]          → fact_healthcare_raw
    facts["feature_mart"] → fact_healthcare_feature_mart
    facts["clustering"]   → fact_clustering (placeholder)
    facts["similarity"]   → fact_similarity
    facts["recommendation"] → fact_recommendation (placeholder)
    '''

    def __init__(
        self,
        engineered_df: pd.DataFrame,
        scaled_df:     pd.DataFrame,
    ) -> None:

        for name, df in [("engineered_df", engineered_df), ("scaled_df", scaled_df)]:
            if PROVINCE_COL not in df.columns:
                raise KeyError(
                    f"Kolom '{PROVINCE_COL}' tidak ditemukan di {name}."
                )

        self._eng   = engineered_df.copy()
        self._scaled = scaled_df.copy()

    # --------------------------------------------------
    # fact_healthcare_raw
    # --------------------------------------------------

    def make_healthcare_raw_fact(self) -> pd.DataFrame:
        '''
        Long-format: satu baris per (provinsi × indikator).

        Schema: province_id | provinsi | indicator | value
        '''

        numeric_cols = [
            c for c in self._eng.columns
            if c != PROVINCE_COL
            and pd.api.types.is_numeric_dtype(self._eng[c])
        ]

        # Buat province_id dari urutan abjad
        prov_sorted = sorted(self._eng[PROVINCE_COL].unique())
        prov_id_map = {p: i + 1 for i, p in enumerate(prov_sorted)}

        melted = self._eng[[PROVINCE_COL] + numeric_cols].melt(
            id_vars=PROVINCE_COL,
            value_vars=numeric_cols,
            var_name="indicator",
            value_name="value",
        )

        melted.insert(
            0,
            "province_id",
            melted[PROVINCE_COL].map(prov_id_map)
        )

        return melted.sort_values(
            ["province_id", "indicator"]
        ).reset_index(drop=True)

    # --------------------------------------------------
    # fact_healthcare_feature_mart
    # --------------------------------------------------

    def make_feature_mart(self) -> pd.DataFrame:
        '''
        Wide-format composite index per provinsi.

        Schema: province_id | provinsi | [6 index cols]
        Menggunakan engineered_df (pre-scale) agar nilai index
        masih interpretatif (bukan 0-1 normalized).
        '''
        return build_feature_mart(self._eng)

    # --------------------------------------------------
    # fact_clustering (placeholder)
    # --------------------------------------------------

    def make_clustering_fact(self) -> pd.DataFrame:
        '''
        Placeholder untuk hasil clustering.
        Diisi oleh clustering pipeline setelah K-Means / GMM selesai.

        Schema: province_id | provinsi | cluster_id | cluster_method
                | [engineered feature cols] | [scaled feature cols]
        '''

        prov_sorted = sorted(self._eng[PROVINCE_COL].unique())
        prov_id_map = {p: i + 1 for i, p in enumerate(prov_sorted)}

        # Feature cols — hanya index cols untuk efisiensi
        index_cols = [
            c for c in self._eng.columns
            if c.endswith("_index") and c != PROVINCE_COL
        ]
        scaled_index_cols = [
            c for c in self._scaled.columns
            if c.endswith("_index") and c != PROVINCE_COL
        ]

        eng_sub   = self._eng[[PROVINCE_COL] + index_cols].copy()
        scale_sub = self._scaled[scaled_index_cols].copy()
        scale_sub.columns = [c + "_scaled" for c in scale_sub.columns]

        fact = pd.concat(
            [eng_sub.reset_index(drop=True), scale_sub.reset_index(drop=True)],
            axis=1
        )

        fact = fact.sort_values(PROVINCE_COL).reset_index(drop=True)
        fact.insert(0, "province_id", fact[PROVINCE_COL].map(prov_id_map))

        # Placeholder kolom clustering
        fact["cluster_id"]     = pd.NA
        fact["cluster_method"] = pd.NA

        return fact

    # --------------------------------------------------
    # fact_similarity
    # --------------------------------------------------

    def make_similarity_fact(self, threshold: float = 0.0) -> pd.DataFrame:
        '''
        Cosine similarity matrix antar provinsi berbasis scaled feature mart.

        Schema: province_a | province_b | similarity_score

        Parameters
        ----------
        threshold : float
            Pasangan di bawah threshold tidak disimpan. Default 0.0.
        '''

        # Gunakan scaled feature mart sebagai basis
        scaled_mart = build_feature_mart(self._scaled)

        return build_similarity(scaled_mart, threshold=threshold)

    # --------------------------------------------------
    # fact_recommendation (placeholder)
    # --------------------------------------------------

    def make_recommendation_fact(self) -> pd.DataFrame:
        '''
        Placeholder untuk output recommendation engine.
        Diisi oleh hybrid recommender setelah pipeline selesai.

        Schema: province_id | provinsi | recommendation | score | source
        '''

        prov_sorted = sorted(self._eng[PROVINCE_COL].unique())

        fact = pd.DataFrame({
            "province_id":    [i + 1 for i in range(len(prov_sorted))],
            PROVINCE_COL:     prov_sorted,
            "recommendation": pd.NA,
            "score":          pd.NA,
            "source":         pd.NA,   # "content_based" | "case_based" | "hybrid"
        })

        return fact

    # --------------------------------------------------
    # make_all_facts
    # --------------------------------------------------

    def make_all_facts(
        self,
        similarity_threshold: float = 0.0
    ) -> dict[str, pd.DataFrame]:
        '''
        Bangun semua fact tables sekaligus.

        Returns
        -------
        dict dengan key:
            "raw", "feature_mart", "clustering",
            "similarity", "recommendation"
        '''

        print("\n[FactsMaker] Building fact tables...")

        facts = {
            "raw":            self.make_healthcare_raw_fact(),
            "feature_mart":   self.make_feature_mart(),
            "clustering":     self.make_clustering_fact(),
            "similarity":     self.make_similarity_fact(similarity_threshold),
            "recommendation": self.make_recommendation_fact(),
        }

        labels = {
            "raw":            "fact_healthcare_raw",
            "feature_mart":   "fact_healthcare_feature_mart",
            "clustering":     "fact_clustering",
            "similarity":     "fact_similarity",
            "recommendation": "fact_recommendation",
        }

        for key, fact_df in facts.items():
            print(
                f"  {labels[key]:35s} → "
                f"{len(fact_df)} baris, "
                f"{len(fact_df.columns)} kolom"
            )

        return facts

    # --------------------------------------------------
    # Update methods (dipanggil oleh downstream pipeline)
    # --------------------------------------------------

    @staticmethod
    def update_clustering_fact(
        fact_clustering:  pd.DataFrame,
        cluster_labels:   pd.Series | np.ndarray,
        method:           str,
    ) -> pd.DataFrame:
        '''
        Isi kolom cluster_id dan cluster_method pada fact_clustering.

        Dipanggil oleh src/clustering setelah K-Means / GMM selesai.

        Parameters
        ----------
        fact_clustering : pd.DataFrame
            Output make_clustering_fact().
        cluster_labels : array-like
            Label cluster per baris, urutan sama dengan fact_clustering.
        method : str
            "kmeans" atau "gmm".
        '''

        df = fact_clustering.copy()
        df["cluster_id"]     = cluster_labels
        df["cluster_method"] = method

        return df

    @staticmethod
    def update_recommendation_fact(
        fact_recommendation: pd.DataFrame,
        province:            str,
        recommendation:      str,
        score:               float,
        source:              str,
    ) -> pd.DataFrame:
        '''
        Tambahkan satu baris rekomendasi ke fact_recommendation.

        Dipanggil oleh src/recommender setelah hybrid ranking selesai.
        '''

        df = fact_recommendation.copy()

        mask = df[PROVINCE_COL] == province

        df.loc[mask, "recommendation"] = recommendation
        df.loc[mask, "score"]          = score
        df.loc[mask, "source"]         = source

        return df