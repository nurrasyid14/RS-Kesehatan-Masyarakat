'''
src/recommender/content_based.py
==================================
ContentBasedRecommender — rekomendasi berbasis kemiripan profil wilayah.

Prinsip
-------
Provinsi dengan profil kesehatan mirip → mendapat rekomendasi
kebijakan yang serupa. Kemiripan diukur dari 5 composite index:

    facility_index, workforce_capacity_index, disease_burden_index,
    insurance_coverage_index, accessibility_barrier_index

Alur
----
    scaled_mart_df
        ↓ build_content_vectors()
    feature vectors P_i = [x_1, ..., x_n]
        ↓ compute_similarity_matrix()
    similarity matrix N×N
        ↓ rank_similar_regions()
    top-K provinsi paling mirip
        ↓ generate_recommendations()
    rekomendasi kebijakan berbasis kemiripan

Referensi
---------
Aggarwal (2016), Ch. 4 — Content-Based Recommender Systems.

Data empiris
------------
Dari 38 provinsi BPS 2025:
  - Riau ↔ Lampung: similarity 0.9987 (tertinggi)
  - Jawa Barat ↔ Jawa Timur: 0.994
  - Papua Pegunungan ↔ Papua Barat Daya: 0.927 (terendah Papua cluster)
'''

import os
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing    import MinMaxScaler


PROVINCE_COL = "Provinsi"

# Kolom index yang digunakan sebagai content vector
CONTENT_COLS = [
    "facility_index",
    "workforce_capacity_index",
    "disease_burden_index",
    "insurance_coverage_index",
    "accessibility_barrier_index",
]

# Metadata kebijakan per kondisi dominan
# Format: condition_key → {tag, policy, description, sdg_target}
POLICY_KNOWLEDGE: dict[str, dict] = {
    "low_facility_index": {
        "tag":         "TAMBAH_FASILITAS",
        "policy":      "Penambahan dan pemerataan fasilitas kesehatan (RS dan Puskesmas)",
        "description": "Rasio fasilitas per penduduk di bawah median nasional.",
        "sdg_target":  "3.b",
        "priority":    2,
    },
    "low_workforce_capacity_index": {
        "tag":         "DISTRIBUSI_TENAGA",
        "policy":      "Redistribusi atau rekrutmen tenaga kesehatan ke wilayah kekurangan",
        "description": "Kapasitas tenaga kesehatan di bawah median nasional.",
        "sdg_target":  "3.c",
        "priority":    2,
    },
    "high_disease_burden_index": {
        "tag":         "PENGENDALIAN_PENYAKIT",
        "policy":      "Penguatan program pengendalian penyakit menular (TBC, HIV, Malaria, DBD, Kusta)",
        "description": "Indeks beban penyakit di atas median nasional.",
        "sdg_target":  "3.3",
        "priority":    1,
    },
    "low_insurance_coverage_index": {
        "tag":         "PERLUASAN_JKN",
        "policy":      "Perluasan kepesertaan Jaminan Kesehatan Nasional (BPJS PBI / Jamkesda)",
        "description": "Cakupan jaminan kesehatan di bawah median nasional.",
        "sdg_target":  "3.8",
        "priority":    2,
    },
    "high_accessibility_barrier_index": {
        "tag":         "AKSES_LAYANAN",
        "policy":      "Intervensi hambatan akses layanan (subsidi biaya, transportasi, waktu tunggu)",
        "description": "Indeks hambatan akses di atas median nasional.",
        "sdg_target":  "3.1",
        "priority":    1,
    },
}

# Threshold: deviasi relatif terhadap median untuk memicu tag kebijakan
HIGH_RATIO = 1.10   # > 110% median → "high"
LOW_RATIO  = 0.90   # < 90% median  → "low"


# ---------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------

def _policy_tags_for_province(
    row:    pd.Series,
    median: pd.Series,
    cols:   list[str],
) -> list[dict]:
    '''
    Hasilkan list policy dict untuk satu provinsi berdasarkan
    posisi indeksnya relatif terhadap median nasional.
    '''
    tags = []

    for col in cols:

        val = float(row[col])
        med = float(median[col])

        if med == 0:
            continue

        ratio = val / med

        if col == "disease_burden_index" and ratio > HIGH_RATIO:
            key = "high_disease_burden_index"
        elif col == "accessibility_barrier_index" and ratio > HIGH_RATIO:
            key = "high_accessibility_barrier_index"
        elif col == "facility_index" and ratio < LOW_RATIO:
            key = "low_facility_index"
        elif col == "workforce_capacity_index" and ratio < LOW_RATIO:
            key = "low_workforce_capacity_index"
        elif col == "insurance_coverage_index" and ratio < LOW_RATIO:
            key = "low_insurance_coverage_index"
        else:
            key = None

        if key and key in POLICY_KNOWLEDGE:
            entry = POLICY_KNOWLEDGE[key].copy()
            entry["condition"] = key
            entry["index_value"]  = round(val, 4)
            entry["median_value"] = round(med, 4)
            tags.append(entry)

    # Urutkan berdasarkan priority (1 = paling mendesak)
    tags.sort(key=lambda x: x["priority"])
    return tags


# ---------------------------------------------------------
# ContentBasedRecommender
# ---------------------------------------------------------

class ContentBasedRecommender:
    '''
    Rekomendasi kebijakan berbasis kemiripan profil kesehatan wilayah.

    Parameters
    ----------
    feature_mart : pd.DataFrame
        Feature mart pre-scale (interpretatif) dengan kolom *_index.
    scaled_mart : pd.DataFrame
        Versi ter-scale dari feature_mart untuk perhitungan similarity.
    similarity_metric : str
        "cosine" (default) atau "euclidean".
    top_k : int
        Jumlah provinsi paling mirip yang dipertimbangkan. Default 5.
    '''

    def __init__(
        self,
        feature_mart:      pd.DataFrame,
        scaled_mart:       pd.DataFrame,
        similarity_metric: str = "cosine",
        top_k:             int = 5,
    ) -> None:

        for name, df in [("feature_mart", feature_mart), ("scaled_mart", scaled_mart)]:
            if PROVINCE_COL not in df.columns:
                raise KeyError(f"Kolom '{PROVINCE_COL}' tidak ditemukan di {name}.")

        self._feat     = feature_mart.copy().reset_index(drop=True)
        self._scaled   = scaled_mart.copy().reset_index(drop=True)
        self._metric   = similarity_metric
        self._top_k    = top_k

        self._content_cols = [c for c in CONTENT_COLS if c in self._scaled.columns]
        self._provinces    = self._scaled[PROVINCE_COL].values

        self._vectors:    np.ndarray | None = None
        self._sim_matrix: np.ndarray | None = None
        self._median:     pd.Series  | None = None

    # --------------------------------------------------
    # Build
    # --------------------------------------------------

    def build_content_vectors(self) -> np.ndarray:
        '''
        Bangun feature matrix P = [P_1, ..., P_n].

        P_i = [facility_index, workforce_capacity_index,
               disease_burden_index, insurance_coverage_index,
               accessibility_barrier_index]

        Returns
        -------
        np.ndarray, shape (n_provinces, n_features)
        '''
        self._vectors = (
            self._scaled[self._content_cols]
            .values
            .astype(float)
        )

        # Pre-compute median dari feature asli (pre-scale)
        feat_cols = [c for c in self._content_cols if c in self._feat.columns]
        self._median = self._feat[feat_cols].median()

        return self._vectors

    def compute_similarity_matrix(self) -> pd.DataFrame:
        '''
        Hitung similarity matrix N×N.

        Returns
        -------
        pd.DataFrame — index dan kolom = nama provinsi.
        '''
        if self._vectors is None:
            self.build_content_vectors()

        if self._metric == "cosine":
            sim = self.cosine_similarity()
        elif self._metric == "euclidean":
            sim = self.euclidean_similarity()
        else:
            raise ValueError(f"metric harus 'cosine' atau 'euclidean'")

        self._sim_matrix = sim

        return pd.DataFrame(
            sim,
            index   = self._provinces,
            columns = self._provinces,
        )

    def cosine_similarity(self) -> np.ndarray:
        '''cos(θ) = (A·B) / (|A||B|). Range [−1, 1], biasanya [0, 1].'''
        if self._vectors is None:
            self.build_content_vectors()
        return cosine_similarity(self._vectors)

    def euclidean_similarity(self) -> np.ndarray:
        '''sim(A,B) = 1 / (1 + d(A,B)). Range (0, 1].'''
        if self._vectors is None:
            self.build_content_vectors()
        dist = euclidean_distances(self._vectors)
        return 1.0 / (1.0 + dist)

    # --------------------------------------------------
    # Ranking
    # --------------------------------------------------

    def rank_similar_regions(
        self,
        province: str,
        top_n:    int   = None,
        threshold: float = 0.0,
    ) -> pd.DataFrame:
        '''
        Urutkan provinsi paling mirip dengan provinsi target.

        Parameters
        ----------
        province : str
        top_n : int
            Default = self._top_k.
        threshold : float
            Hanya tampilkan similarity ≥ threshold.

        Returns
        -------
        pd.DataFrame
            Kolom: rank | province | similarity_score
        '''
        if self._sim_matrix is None:
            self.compute_similarity_matrix()

        top_n = top_n or self._top_k

        prov_list = list(self._provinces)
        if province not in prov_list:
            raise KeyError(
                f"Provinsi '{province}' tidak ditemukan. "
                f"Tersedia: {prov_list}"
            )

        idx    = prov_list.index(province)
        scores = self._sim_matrix[idx].copy()
        scores[idx] = -1   # exclude self

        # Filter threshold
        valid  = np.where(scores >= threshold)[0]
        top_idx = valid[np.argsort(scores[valid])[::-1][:top_n]]

        rows = [
            {
                "rank":             i + 1,
                "province":         self._provinces[j],
                "similarity_score": round(float(scores[j]), 4),
            }
            for i, j in enumerate(top_idx)
        ]

        return pd.DataFrame(rows)

    # --------------------------------------------------
    # Recommendations
    # --------------------------------------------------

    def generate_recommendations(
        self,
        province: str,
        top_n:    int = None,
    ) -> list[dict]:
        '''
        Hasilkan rekomendasi kebijakan untuk satu provinsi
        berdasarkan kondisi provinsi mirip terdekat.

        Logika
        ------
        1. Temukan top-K provinsi paling mirip.
        2. Identifikasi kondisi dominan PROVINSI TARGET sendiri.
        3. Untuk setiap kondisi, ambil contoh provinsi mirip yang
           memiliki kondisi serupa sebagai "precedent".
        4. Beri skor = similarity × (1 / priority).

        Returns
        -------
        list of dict, diurutkan descending berdasarkan score.
        '''
        if self._sim_matrix is None:
            self.compute_similarity_matrix()
        if self._median is None:
            self.build_content_vectors()

        prov_list = list(self._provinces)
        if province not in prov_list:
            raise KeyError(f"Provinsi '{province}' tidak ditemukan.")

        top_n  = top_n or self._top_k
        ranked = self.rank_similar_regions(province, top_n=top_n)

        # Ambil baris provinsi target dari feature_mart asli
        target_row = self._feat[
            self._feat[PROVINCE_COL] == province
        ].iloc[0]

        # Kondisi dominan target
        target_policies = _policy_tags_for_province(
            target_row, self._median,
            [c for c in self._content_cols if c in self._feat.columns]
        )

        # Ambil precedent dari provinsi paling mirip
        precedents: dict[str, list[str]] = {}

        for _, sim_row in ranked.iterrows():
            sim_prov = sim_row["province"]
            sim_score = float(sim_row["similarity_score"])

            sim_feat_row = self._feat[
                self._feat[PROVINCE_COL] == sim_prov
            ].iloc[0]

            sim_policies = _policy_tags_for_province(
                sim_feat_row, self._median,
                [c for c in self._content_cols if c in self._feat.columns]
            )

            for p in sim_policies:
                key = p["tag"]
                if key not in precedents:
                    precedents[key] = []
                precedents[key].append(
                    f"{sim_prov} (sim={sim_score:.3f})"
                )

        # Gabungkan: hanya rekomendasikan kondisi yang ada di target
        # ATAU muncul di ≥2 provinsi mirip (strong signal)
        target_tags = {p["tag"] for p in target_policies}
        strong_signal_tags = {
            tag for tag, provs in precedents.items()
            if len(provs) >= 2
        }
        relevant_tags = target_tags | strong_signal_tags

        recommendations = []

        for p in target_policies:
            if p["tag"] not in relevant_tags:
                continue

            score = 1.0 / p["priority"]
            if p["tag"] in precedents:
                # Boost berdasarkan jumlah precedent
                score += 0.1 * len(precedents[p["tag"]])

            rec = {
                "province":    province,
                "tag":         p["tag"],
                "policy":      p["policy"],
                "description": p["description"],
                "sdg_target":  p["sdg_target"],
                "priority":    p["priority"],
                "score":       round(score, 4),
                "source":      "content_based",
                "precedents":  precedents.get(p["tag"], []),
            }
            recommendations.append(rec)

        # Tambahkan sinyal kuat dari provinsi mirip yang tidak ada di target
        for tag in strong_signal_tags - target_tags:
            if tag in POLICY_KNOWLEDGE:
                p = POLICY_KNOWLEDGE[tag]
                rec = {
                    "province":    province,
                    "tag":         tag,
                    "policy":      p["policy"],
                    "description": f"Muncul di {len(precedents[tag])} provinsi dengan profil serupa.",
                    "sdg_target":  p["sdg_target"],
                    "priority":    p["priority"],
                    "score":       round(0.05 * len(precedents[tag]), 4),
                    "source":      "content_based_signal",
                    "precedents":  precedents[tag],
                }
                recommendations.append(rec)

        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations

    def explain_recommendation(
        self,
        province:   str,
        tag:        str,
        top_n:      int = 3,
    ) -> dict:
        '''
        Penjelasan eksplisit mengapa satu rekomendasi diberikan.

        Returns
        -------
        dict
            province | tag | reason | similar_provinces
            | index_comparison | sdg_target
        '''
        if self._sim_matrix is None:
            self.compute_similarity_matrix()

        ranked = self.rank_similar_regions(province, top_n=top_n)

        # Nilai indeks target vs median
        target_row = self._feat[self._feat[PROVINCE_COL] == province].iloc[0]
        comparison = {}

        for col in [c for c in self._content_cols if c in self._feat.columns]:
            comparison[col] = {
                "provinsi": round(float(target_row[col]), 4),
                "median":   round(float(self._median[col]), 4),
                "ratio":    round(float(target_row[col]) / max(float(self._median[col]), 1e-9), 3),
            }

        policy_info = POLICY_KNOWLEDGE.get(tag, {})

        return {
            "province":          province,
            "tag":               tag,
            "policy":            policy_info.get("policy", ""),
            "reason":            policy_info.get("description", ""),
            "sdg_target":        policy_info.get("sdg_target", ""),
            "similar_provinces": ranked["province"].tolist(),
            "index_comparison":  comparison,
        }

    # --------------------------------------------------
    # Filter
    # --------------------------------------------------

    def filter_recommendations(
        self,
        recommendations: list[dict],
        min_score:       float = 0.1,
        max_per_sdg:     int   = 2,
        deduplicate:     bool  = True,
    ) -> list[dict]:
        '''
        Saring rekomendasi: hapus duplikat, relevansi rendah,
        dan batasi per SDG target.

        Parameters
        ----------
        min_score : float
            Hapus rekomendasi dengan score < min_score.
        max_per_sdg : int
            Maksimum rekomendasi per SDG target.
        deduplicate : bool
            Hapus duplikat berdasarkan tag.
        '''
        seen_tags: set[str]       = set()
        sdg_count: dict[str, int] = {}
        filtered                  = []

        for rec in sorted(recommendations, key=lambda x: x["score"], reverse=True):

            if rec["score"] < min_score:
                continue

            if deduplicate and rec["tag"] in seen_tags:
                continue

            sdg = rec.get("sdg_target", "")
            if sdg_count.get(sdg, 0) >= max_per_sdg:
                continue

            seen_tags.add(rec["tag"])
            sdg_count[sdg] = sdg_count.get(sdg, 0) + 1
            filtered.append(rec)

        return filtered

    # --------------------------------------------------
    # Export
    # --------------------------------------------------

    def export_recommendations(
        self,
        recommendations: list[dict],
        output_path:     str,
        fmt:             str = "json",
    ) -> None:
        '''
        Ekspor rekomendasi ke JSON atau CSV.

        Parameters
        ----------
        fmt : str
            "json" (default) atau "csv".
        '''
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if fmt == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(recommendations, f, ensure_ascii=False, indent=2)

        elif fmt == "csv":
            # Flatten list fields for CSV
            rows = []
            for rec in recommendations:
                r = rec.copy()
                r["precedents"] = "; ".join(r.get("precedents", []))
                rows.append(r)
            pd.DataFrame(rows).to_csv(output_path, index=False)

        else:
            raise ValueError(f"fmt harus 'json' atau 'csv'")

        print(f"[ContentBasedRecommender] Exported → {output_path}")

    # --------------------------------------------------
    # Visualisasi
    # --------------------------------------------------

    def visualize_similarity(
        self,
        province:    str = None,
        output_path: str = None,
        top_n:       int = 10,
    ) -> None:
        '''
        Visualisasi similarity: heatmap (semua provinsi) atau
        bar chart (top-N similar untuk satu provinsi).

        Memerlukan matplotlib dan seaborn.

        Parameters
        ----------
        province : str
            Jika None, tampilkan heatmap penuh.
        output_path : str
            Simpan ke file jika diberikan.
        '''
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError(
                "matplotlib dan seaborn diperlukan. "
                "Jalankan: pip install matplotlib seaborn"
            )

        if self._sim_matrix is None:
            self.compute_similarity_matrix()

        if province:
            # Bar chart top-N
            ranked = self.rank_similar_regions(province, top_n=top_n)

            fig, ax = plt.subplots(figsize=(8, 5))
            colors  = [
                "#534AB7" if s >= 0.95 else "#AFA9EC"
                for s in ranked["similarity_score"]
            ]
            ax.barh(
                ranked["province"][::-1],
                ranked["similarity_score"][::-1],
                color=colors[::-1],
            )
            ax.set_xlabel("Cosine Similarity")
            ax.set_title(
                f"Provinsi paling mirip dengan {province}",
                fontsize=11, fontweight="500",
            )
            ax.axvline(0.95, color="#E8593C", linestyle="--", linewidth=0.8,
                       label="threshold 0.95")
            ax.legend(fontsize=9)
            ax.set_xlim(0, 1.05)
            plt.tight_layout()

        else:
            # Full heatmap (subsetted to 15 provinces for readability)
            n  = min(len(self._provinces), 15)
            sm = self._sim_matrix[:n, :n]
            pv = self._provinces[:n]

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                sm,
                xticklabels = pv,
                yticklabels = pv,
                cmap        = "RdPu",
                vmin        = 0.8,
                vmax        = 1.0,
                linewidths  = 0.3,
                ax          = ax,
                annot       = True,
                fmt         = ".2f",
                annot_kws   = {"size": 7},
            )
            ax.set_title(
                "Similarity heatmap (15 provinsi pertama)",
                fontsize=11, fontweight="500",
            )
            plt.xticks(rotation=45, ha="right", fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"[ContentBasedRecommender] Viz disimpan → {output_path}")
        else:
            plt.show()

        plt.close()