'''
src/profiler/regional_profiler.py
===================================
RegionalProfiler — semantic profiling cluster dan provinsi pasca-clustering.

Referensi
---------
Aggarwal, C. C. (2016). Recommender Systems: The Textbook.
    Springer. Ch. 7.

WHO (2023). Health SDG Profile: Driving SDG progress through
    better health data. World Health Organization.

Catatan desain — SDG 3 mapping
-------------------------------
SDG 3 "Good Health and Well-Being" memiliki 13 target utama (3.1–3.d).
Mapping dilakukan ke subset target yang relevan dengan data BPS:

    Target 3.1  : AKI (mortalitas ibu) — proxy: barrier akses
    Target 3.3  : Penyakit menular (TBC, HIV, Malaria, DBD, Kusta)
    Target 3.4  : Penyakit tidak menular & kesehatan jiwa — proxy: workforce
    Target 3.8  : UHC (Universal Health Coverage) — proxy: insurance coverage
    Target 3.b  : Akses obat & tenaga kesehatan — proxy: facility + workforce
    Target 3.c  : Tenaga kesehatan & kapasitas — proxy: workforce_capacity_index

generate_policy_tags() menghasilkan tag yang dapat langsung dikonsumsi
recommendation engine sebagai item descriptor.
'''

import json
import os

import numpy as np
import pandas as pd


PROVINCE_COL = "Provinsi"

# Indeks komposit yang diharapkan tersedia
INDEX_COLS = [
    "facility_index",
    "workforce_capacity_index",
    "disease_burden_index",
    "insurance_coverage_index",
    "accessibility_barrier_index",
    "treatment_effectiveness_index",
]

# ---------------------------------------------------------
# SDG 3 mapping
# ---------------------------------------------------------

SDG3_MAP: dict[str, dict] = {
    "facility_index": {
        "target":    "3.b",
        "label":     "Access to Medicines & Health Facilities",
        "direction": "high_is_good",
    },
    "workforce_capacity_index": {
        "target":    "3.c",
        "label":     "Health Workforce & Capacity",
        "direction": "high_is_good",
    },
    "disease_burden_index": {
        "target":    "3.3",
        "label":     "Communicable Disease Burden",
        "direction": "low_is_good",
    },
    "insurance_coverage_index": {
        "target":    "3.8",
        "label":     "Universal Health Coverage (UHC)",
        "direction": "high_is_good",
    },
    "accessibility_barrier_index": {
        "target":    "3.1",
        "label":     "Healthcare Access Barriers",
        "direction": "low_is_good",
    },
    "treatment_effectiveness_index": {
        "target":    "3.3",
        "label":     "Treatment Effectiveness (TB)",
        "direction": "high_is_good",
    },
}

# Threshold untuk tag semantik — nilai relatif terhadap mean nasional
HIGH_THRESHOLD  = 0.75   # > 75th percentile → "high"
LOW_THRESHOLD   = 0.25   # < 25th percentile → "low"

# Policy tag templates per kondisi yang ditemukan
POLICY_TEMPLATE: dict[str, dict] = {
    "low_facility_index": {
        "tag":         "TAMBAH_FASILITAS",
        "description": "Penambahan RS dan/atau Puskesmas diprioritaskan",
        "sdg_target":  "3.b",
    },
    "low_workforce_capacity_index": {
        "tag":         "DISTRIBUSI_TENAGA",
        "description": "Redistribusi atau rekrutmen tenaga kesehatan",
        "sdg_target":  "3.c",
    },
    "high_disease_burden_index": {
        "tag":         "PENGENDALIAN_PENYAKIT",
        "description": "Program penanggulangan penyakit menular diperkuat",
        "sdg_target":  "3.3",
    },
    "low_insurance_coverage_index": {
        "tag":         "PERLUASAN_JKN",
        "description": "Perluasan cakupan Jaminan Kesehatan Nasional",
        "sdg_target":  "3.8",
    },
    "high_accessibility_barrier_index": {
        "tag":         "AKSES_LAYANAN",
        "description": "Intervensi hambatan akses (biaya, transportasi, waktu)",
        "sdg_target":  "3.1",
    },
    "low_treatment_effectiveness_index": {
        "tag":         "KUALITAS_PENGOBATAN",
        "description": "Peningkatan kualitas dan kepatuhan pengobatan",
        "sdg_target":  "3.3",
    },
}


class RegionalProfiler:
    '''
    Semantic profiling cluster dan provinsi pasca-clustering.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Feature mart (pre-scale, interpretatif).
        Harus memiliki kolom Provinsi dan kolom *_index.
    cluster_labels : array-like of int
        Label cluster per provinsi (urutan sama dengan feature_df).
    cluster_method : str
        "kmeans" atau "gmm". Disimpan sebagai metadata.
    '''

    def __init__(
        self,
        feature_df:     pd.DataFrame,
        cluster_labels,
        cluster_method: str = "kmeans",
    ) -> None:

        if PROVINCE_COL not in feature_df.columns:
            raise KeyError(f"Kolom '{PROVINCE_COL}' tidak ditemukan.")

        self._df     = feature_df.copy().reset_index(drop=True)
        self._labels = np.array(cluster_labels, dtype=int)
        self._method = cluster_method

        if len(self._labels) != len(self._df):
            raise ValueError(
                f"Panjang cluster_labels ({len(self._labels)}) "
                f"tidak cocok dengan feature_df ({len(self._df)} baris)."
            )

        self._df["cluster_id"] = self._labels

        # Kolom index yang tersedia
        self._index_cols = [
            c for c in INDEX_COLS if c in self._df.columns
        ]

        # Nomor cluster unik
        self._cluster_ids = sorted(self._df["cluster_id"].unique())

    # --------------------------------------------------
    # Cluster-level profiling
    # --------------------------------------------------

    def generate_cluster_profiles(self) -> pd.DataFrame:
        '''
        Deskripsi semantic per cluster berdasarkan nilai rata-rata
        tiap composite index dibandingkan rata-rata nasional.

        Returns
        -------
        pd.DataFrame
            Index = cluster_id.
            Kolom = [index_cols..., n_provinces, semantic_label, description]
        '''
        stats = self.summarize_cluster_statistics()

        rows = []

        for cid in self._cluster_ids:

            row = {"cluster_id": cid}

            for col in self._index_cols:
                row[col] = round(
                    float(stats.loc[(cid, "mean"), col]), 4
                )

            provinces = self._df.loc[
                self._df["cluster_id"] == cid, PROVINCE_COL
            ].tolist()

            row["n_provinces"] = len(provinces)
            row["provinces"]   = provinces

            # Semantic label
            dominant = self.identify_dominant_features(cluster_id=cid)
            row["dominant_features"] = dominant
            row["semantic_label"]    = self._build_semantic_label(dominant)
            row["description"]       = self._build_description(dominant)

            rows.append(row)

        return pd.DataFrame(rows).set_index("cluster_id")

    def summarize_cluster_statistics(self) -> pd.DataFrame:
        '''
        Statistik deskriptif per cluster: mean, std, min, max, count.

        Returns
        -------
        pd.DataFrame
            MultiIndex (cluster_id, stat).
            Kolom = index_cols.
        '''
        return (
            self._df.groupby("cluster_id")[self._index_cols]
            .agg(["mean", "std", "min", "max", "count"])
        )

    def identify_dominant_features(
        self,
        cluster_id: int,
        top_n:      int = 3,
    ) -> list[str]:
        '''
        Identifikasi fitur pembeda utama cluster relatif terhadap
        rata-rata nasional.

        Sebuah fitur dianggap "dominant" jika cluster mean-nya
        menyimpang paling jauh dari grand mean (dalam satuan std).

        Parameters
        ----------
        cluster_id : int
        top_n : int
            Jumlah fitur teratas. Default 3.

        Returns
        -------
        list of str
            Format: "high_{feature}" atau "low_{feature}"
        '''
        subset = self._df[self._df["cluster_id"] == cluster_id][self._index_cols]
        all_   = self._df[self._index_cols]

        cluster_mean = subset.mean()
        grand_mean   = all_.mean()
        grand_std    = all_.std().replace(0, 1e-9)

        z_scores = (cluster_mean - grand_mean) / grand_std

        # Urutkan berdasarkan magnitude absolut
        top_features = z_scores.abs().nlargest(top_n).index

        result = []
        for feat in top_features:
            prefix = "high" if z_scores[feat] > 0 else "low"
            result.append(f"{prefix}_{feat}")

        return result

    def compare_clusters(self) -> pd.DataFrame:
        '''
        Tabel perbandingan antar cluster — mean per index col.

        Returns
        -------
        pd.DataFrame
            Baris = cluster_id, Kolom = index_cols + n_provinces
        '''
        cluster_means = (
            self._df.groupby("cluster_id")[self._index_cols]
            .mean()
            .round(4)
        )

        counts = (
            self._df.groupby("cluster_id")
            .size()
            .rename("n_provinces")
        )

        return cluster_means.join(counts)

    # --------------------------------------------------
    # Policy & SDG
    # --------------------------------------------------

    def generate_policy_tags(self) -> pd.DataFrame:
        '''
        Hasilkan tag kebijakan otomatis per cluster berdasarkan
        kondisi dominan yang terdeteksi.

        Tags mengacu pada POLICY_TEMPLATE dan threshold percentile.

        Returns
        -------
        pd.DataFrame
            Kolom: cluster_id | policy_tags | sdg_targets | priority_count
        '''
        national_q = {
            col: {
                "q25": float(self._df[col].quantile(LOW_THRESHOLD)),
                "q75": float(self._df[col].quantile(HIGH_THRESHOLD)),
            }
            for col in self._index_cols
        }

        rows = []

        for cid in self._cluster_ids:

            subset      = self._df[self._df["cluster_id"] == cid]
            cluster_mean = subset[self._index_cols].mean()

            tags        = []
            sdg_targets = set()

            for col in self._index_cols:

                val       = cluster_mean[col]
                sdg_info  = SDG3_MAP.get(col, {})
                direction = sdg_info.get("direction", "high_is_good")

                # Tentukan kondisi berdasarkan arah dan threshold
                if direction == "high_is_good":
                    condition = f"low_{col}" if val < national_q[col]["q25"] else None
                else:
                    condition = f"high_{col}" if val > national_q[col]["q75"] else None

                if condition and condition in POLICY_TEMPLATE:
                    policy = POLICY_TEMPLATE[condition]
                    tags.append(policy["tag"])
                    sdg_targets.add(policy["sdg_target"])

            rows.append({
                "cluster_id":     cid,
                "policy_tags":    tags,
                "sdg_targets":    sorted(sdg_targets),
                "priority_count": len(tags),
            })

        return pd.DataFrame(rows).set_index("cluster_id")

    def map_sdg3_indicators(self) -> pd.DataFrame:
        '''
        Peta kondisi tiap cluster terhadap target SDG 3.

        Returns
        -------
        pd.DataFrame
            Index = cluster_id.
            Kolom = SDG target string (3.1, 3.3, ...).
            Nilai = "needs_attention" | "on_track" | "data_limited"
        '''
        national_med = self._df[self._index_cols].median()
        sdg_targets  = sorted({v["target"] for v in SDG3_MAP.values()})

        rows = []

        for cid in self._cluster_ids:

            subset       = self._df[self._df["cluster_id"] == cid]
            cluster_mean = subset[self._index_cols].mean()
            row          = {"cluster_id": cid}

            # Inisialisasi
            for t in sdg_targets:
                row[t] = "data_limited"

            for col, sdg_info in SDG3_MAP.items():

                if col not in self._index_cols:
                    continue

                target    = sdg_info["target"]
                direction = sdg_info["direction"]
                val       = cluster_mean[col]
                med       = national_med[col]

                if direction == "high_is_good":
                    status = "on_track" if val >= med else "needs_attention"
                else:
                    status = "on_track" if val <= med else "needs_attention"

                # "needs_attention" menang atas "on_track"
                if row[target] != "needs_attention":
                    row[target] = status

            rows.append(row)

        return pd.DataFrame(rows).set_index("cluster_id")

    # --------------------------------------------------
    # Province-level summary
    # --------------------------------------------------

    def create_region_summary(
        self,
        province: str,
    ) -> dict:
        '''
        Ringkasan lengkap satu provinsi: profil indeks, cluster,
        policy tags, dan SDG status.

        Parameters
        ----------
        province : str
            Nama provinsi.

        Returns
        -------
        dict
            province | cluster_id | index_values | policy_tags
            | sdg_status | dominant_features
        '''
        mask = self._df[PROVINCE_COL] == province

        if not mask.any():
            raise KeyError(
                f"Provinsi '{province}' tidak ditemukan. "
                f"Tersedia: {sorted(self._df[PROVINCE_COL].unique())}"
            )

        row      = self._df[mask].iloc[0]
        cid      = int(row["cluster_id"])

        # Index values
        index_values = {
            col: round(float(row[col]), 4)
            for col in self._index_cols
            if col in row.index
        }

        # Policy tags untuk cluster ini
        policy_df = self.generate_policy_tags()
        tags      = policy_df.loc[cid, "policy_tags"] if cid in policy_df.index else []

        # SDG status untuk cluster ini
        sdg_df   = self.map_sdg3_indicators()
        sdg_row  = sdg_df.loc[cid].to_dict() if cid in sdg_df.index else {}

        # Dominant features
        dominant = self.identify_dominant_features(cid)

        return {
            "province":          province,
            "cluster_id":        cid,
            "cluster_method":    self._method,
            "index_values":      index_values,
            "dominant_features": dominant,
            "policy_tags":       tags,
            "sdg_status":        sdg_row,
        }

    # --------------------------------------------------
    # Export & visualisasi
    # --------------------------------------------------

    def export_profiles(
        self,
        output_dir: str,
        fmt:        str = "csv",
    ) -> None:
        '''
        Ekspor hasil profiling ke folder output.

        Files yang dihasilkan
        ---------------------
        cluster_profiles.csv / .json
        cluster_policy_tags.csv / .json
        cluster_sdg3_map.csv / .json
        cluster_comparison.csv / .json

        Parameters
        ----------
        output_dir : str
        fmt : str
            "csv" (default) atau "json".
        '''
        os.makedirs(output_dir, exist_ok=True)

        tables = {
            "cluster_profiles":    self.generate_cluster_profiles(),
            "cluster_policy_tags": self.generate_policy_tags(),
            "cluster_sdg3_map":    self.map_sdg3_indicators(),
            "cluster_comparison":  self.compare_clusters(),
        }

        for name, df in tables.items():

            path = os.path.join(output_dir, f"{name}.{fmt}")

            if fmt == "csv":
                df.to_csv(path)
            elif fmt == "json":
                df.to_json(path, orient="index", indent=2, force_ascii=False)
            else:
                raise ValueError(f"fmt harus 'csv' atau 'json'")

            print(f"[RegionalProfiler] Exported → {path}")

    def visualize_profiles(
        self,
        output_path: str = None,
    ) -> None:
        '''
        Visualisasi radar chart per cluster berbasis composite index.

        Memerlukan matplotlib. Jika output_path diberikan, simpan ke file;
        jika tidak, tampilkan ke layar.

        Parameters
        ----------
        output_path : str, optional
            Path file gambar, e.g. "output/cluster_radar.png"
        '''
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib tidak terinstal. "
                "Jalankan: pip install matplotlib"
            )

        comparison = self.compare_clusters()[self._index_cols]
        labels     = [c.replace("_index", "").replace("_", "\n") for c in self._index_cols]
        n          = len(labels)
        angles     = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles    += angles[:1]  # tutup lingkaran

        fig, ax = plt.subplots(
            figsize    = (8, 8),
            subplot_kw = {"polar": True},
        )

        colors = plt.cm.tab10.colors

        for i, cid in enumerate(comparison.index):

            values  = comparison.loc[cid].tolist()
            values += values[:1]

            ax.plot(angles, values, color=colors[i % 10], linewidth=2, label=f"Cluster {cid}")
            ax.fill(angles, values, color=colors[i % 10], alpha=0.15)

        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
        ax.set_ylim(0, None)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        ax.set_title(
            f"Regional Cluster Profiles ({self._method.upper()})",
            pad=20,
            fontsize=12,
        )

        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"[RegionalProfiler] Visualisasi disimpan → {output_path}")
        else:
            plt.show()

        plt.close()

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    @staticmethod
    def _build_semantic_label(dominant_features: list[str]) -> str:
        '''Buat label singkat dari daftar dominant features.'''

        if not dominant_features:
            return "balanced"

        parts = []
        for feat in dominant_features[:2]:        # maks 2 fitur di label
            level, *name_parts = feat.split("_")
            name = " ".join(name_parts).replace("index", "").strip()
            parts.append(f"{level.upper()} {name}")

        return " / ".join(parts)

    @staticmethod
    def _build_description(dominant_features: list[str]) -> str:
        '''Buat deskripsi naratif dari dominant features.'''

        label_map = {
            "facility_index":              "ketersediaan fasilitas",
            "workforce_capacity_index":    "kapasitas tenaga kesehatan",
            "disease_burden_index":        "beban penyakit menular",
            "insurance_coverage_index":    "cakupan jaminan kesehatan",
            "accessibility_barrier_index": "hambatan akses layanan",
            "treatment_effectiveness_index": "efektivitas pengobatan",
        }

        parts = []
        for feat in dominant_features:
            level, *name_parts = feat.split("_", 1)
            col  = "_".join(name_parts)
            desc = label_map.get(col, col.replace("_", " "))
            parts.append(
                f"{'tinggi' if level == 'high' else 'rendah'} {desc}"
            )

        return "Karakteristik: " + "; ".join(parts) + "."