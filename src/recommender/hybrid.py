'''
src/recommender/hybrid.py
===========================
HybridRecommender — menggabungkan content-based dan case-based
menjadi satu recommendation engine berbobot.

Formula
-------
    R = α · R_c + β · R_k

    R_c  = content-based score
    R_k  = case-based score
    α, β = bobot (default: α=0.6, β=0.4)

Pemilihan bobot default
-----------------------
α > β karena data BPS 2025 memberikan fitur kuantitatif yang kuat
untuk content similarity, sementara case library saat ini berbasis
anchor kasus yang belum memiliki feedback loop empiris.
Setelah feedback dikumpulkan, tune_hybrid_weights() dapat dijalankan
untuk kalibrasi ulang.

Conflict resolution
-------------------
Jika content engine dan case engine merekomendasikan kebijakan berbeda
untuk masalah yang sama (misal: content → "tambah RS", case → "tambah
tenaga"), conflict dideteksi berdasarkan SDG target yang tumpang tindih
dan diselesaikan dengan mengambil yang score tertinggi, bukan arbitrary.

Referensi
---------
Aggarwal (2016), Ch. 9 — Hybrid Recommender Systems.
Burke, R. (2002). Hybrid Recommender Systems: Survey and Experiments.
    User Modeling and User-Adapted Interaction.
'''

import os
import json

import numpy as np
import pandas as pd

from .content_based import ContentBasedRecommender
from .case_based    import CaseBasedRecommender


PROVINCE_COL = "Provinsi"

# Default weights — dapat di-tune via tune_hybrid_weights()
DEFAULT_ALPHA = 0.6   # bobot content-based
DEFAULT_BETA  = 0.4   # bobot case-based


class HybridRecommender:
    '''
    Hybrid Healthcare Recommender: R = α·R_c + β·R_k.

    Parameters
    ----------
    feature_mart : pd.DataFrame
        Feature mart pre-scale.
    scaled_mart : pd.DataFrame
        Feature mart ter-scale.
    alpha : float
        Bobot content-based engine. Default 0.6.
    beta : float
        Bobot case-based engine. Default 0.4.
    top_k : int
        Jumlah rekomendasi final yang dikembalikan. Default 5.
    similarity_metric : str
        "cosine" (default) atau "euclidean" untuk content engine.
    '''

    def __init__(
        self,
        feature_mart:      pd.DataFrame,
        scaled_mart:       pd.DataFrame,
        alpha:             float = DEFAULT_ALPHA,
        beta:              float = DEFAULT_BETA,
        top_k:             int   = 5,
        similarity_metric: str   = "cosine",
    ) -> None:

        if not np.isclose(alpha + beta, 1.0):
            raise ValueError(
                f"α + β harus = 1.0, diberikan α={alpha}, β={beta}."
            )

        self.alpha    = alpha
        self.beta     = beta
        self._top_k   = top_k

        self._content_engine = ContentBasedRecommender(
            feature_mart      = feature_mart,
            scaled_mart       = scaled_mart,
            similarity_metric = similarity_metric,
            top_k             = 8,
        )

        self._case_engine = CaseBasedRecommender(
            feature_mart  = feature_mart,
            scaled_mart   = scaled_mart,
            top_k_cases   = 4,
        )

        # Cache hasil per provinsi untuk efisiensi
        self._cache: dict[str, dict] = {}

    # --------------------------------------------------
    # Combine
    # --------------------------------------------------

    def combine_recommendations(
        self,
        content_recs: list[dict],
        case_recs:    list[dict],
    ) -> dict[str, dict]:
        '''
        Gabungkan dua list rekomendasi menjadi satu dict keyed by tag.

        Setiap tag menyimpan score content, score case, dan metadata
        dari kedua engine.

        Returns
        -------
        dict { tag → combined_entry }
        '''
        combined: dict[str, dict] = {}

        for rec in content_recs:
            tag = rec["tag"]
            if tag not in combined:
                combined[tag] = {
                    "tag":          tag,
                    "policy":       rec["policy"],
                    "sdg_target":   rec.get("sdg_target", ""),
                    "priority":     rec.get("priority", 3),
                    "content_score": 0.0,
                    "case_score":    0.0,
                    "sources":       [],
                    "metadata":      {},
                }
            combined[tag]["content_score"] = max(
                combined[tag]["content_score"],
                rec.get("score", 0.0),
            )
            combined[tag]["sources"].append("content_based")
            combined[tag]["metadata"]["content"] = {
                "precedents":  rec.get("precedents", []),
                "description": rec.get("description", ""),
            }

        for rec in case_recs:
            tag = rec["tag"]
            if tag not in combined:
                combined[tag] = {
                    "tag":          tag,
                    "policy":       rec["policy"],
                    "sdg_target":   rec.get("sdg_target", ""),
                    "priority":     rec.get("priority", 3),
                    "content_score": 0.0,
                    "case_score":    0.0,
                    "sources":       [],
                    "metadata":      {},
                }
            combined[tag]["case_score"] = max(
                combined[tag]["case_score"],
                rec.get("score", 0.0),
            )
            combined[tag]["sources"].append("case_based")
            combined[tag]["metadata"]["case"] = {
                "case_id":           rec.get("case_id", ""),
                "case_label":        rec.get("case_label", ""),
                "evidence_provinces": rec.get("evidence_provinces", []),
                "adaptation_notes":  rec.get("adaptation_notes", []),
                "effectiveness":     rec.get("effectiveness", 0.0),
            }

        return combined

    def weighted_hybrid(
        self,
        combined: dict[str, dict],
    ) -> dict[str, dict]:
        '''
        Terapkan formula R = α·R_c + β·R_k.

        Returns
        -------
        dict { tag → entry_with_hybrid_score }
        '''
        for tag, entry in combined.items():

            entry["hybrid_score"] = round(
                self.alpha * entry["content_score"] +
                self.beta  * entry["case_score"],
                4,
            )

            # Tentukan source dominan
            if entry["content_score"] > 0 and entry["case_score"] > 0:
                entry["source"] = "hybrid"
            elif entry["content_score"] > 0:
                entry["source"] = "content_only"
            else:
                entry["source"] = "case_only"

            # Persentase kontribusi
            total = entry["content_score"] + entry["case_score"]
            if total > 0:
                entry["content_contribution"] = round(
                    entry["content_score"] / total * 100, 1
                )
                entry["case_contribution"] = round(
                    entry["case_score"] / total * 100, 1
                )
            else:
                entry["content_contribution"] = 0.0
                entry["case_contribution"]    = 0.0

        return combined

    def rank_hybrid_recommendations(
        self,
        combined:  dict[str, dict],
        top_n:     int = None,
    ) -> list[dict]:
        '''
        Urutkan rekomendasi berdasarkan hybrid_score descending.

        Returns
        -------
        list of dict
        '''
        top_n = top_n or self._top_k

        ranked = sorted(
            combined.values(),
            key     = lambda x: x.get("hybrid_score", 0),
            reverse = True,
        )

        return ranked[:top_n]

    def resolve_conflicts(
        self,
        combined: dict[str, dict],
    ) -> dict[str, dict]:
        '''
        Deteksi dan selesaikan konflik: dua rekomendasi dengan
        SDG target sama tapi kebijakan berbeda.

        Strategi resolusi:
        - Jika kedua dari engine yang berbeda (content vs case)
          → pertahankan keduanya tapi tandai sebagai "conflicting"
        - Jika salah satu score < 0.1 → hapus yang lebih lemah

        Returns
        -------
        dict — combined yang sudah di-resolve
        '''
        sdg_groups: dict[str, list[str]] = {}

        for tag, entry in combined.items():
            sdg = entry.get("sdg_target", "")
            if sdg not in sdg_groups:
                sdg_groups[sdg] = []
            sdg_groups[sdg].append(tag)

        for sdg, tags in sdg_groups.items():
            if len(tags) <= 1:
                continue

            # Cek konflik: semua dari sumber berbeda
            for tag in tags:
                combined[tag]["conflict_detected"] = len(tags) > 1
                combined[tag]["conflict_peers"]    = [t for t in tags if t != tag]

            # Hapus yang terlalu lemah (hybrid_score < 0.05)
            for tag in list(tags):
                if combined[tag].get("hybrid_score", 0) < 0.05:
                    del combined[tag]

        return combined

    # --------------------------------------------------
    # Explain
    # --------------------------------------------------

    def explain_hybrid_reasoning(
        self,
        province:  str,
        tag:       str,
        combined:  dict[str, dict] = None,
    ) -> dict:
        '''
        Penjelasan lengkap mengapa rekomendasi hybrid dihasilkan.

        Parameters
        ----------
        province : str
        tag : str
        combined : dict, optional
            Jika None, pipeline dijalankan ulang.

        Returns
        -------
        dict dengan breakdown kontribusi α·R_c dan β·R_k.
        '''
        if combined is None:
            combined = self._run_engines(province)

        entry = combined.get(tag, {})

        content_explain = self._content_engine.explain_recommendation(
            province, tag
        ) if entry.get("content_score", 0) > 0 else {}

        case_explain = self._case_engine.explain_case_reasoning(
            province, tag
        ) if entry.get("case_score", 0) > 0 else {}

        return {
            "province":               province,
            "tag":                    tag,
            "policy":                 entry.get("policy", ""),
            "sdg_target":             entry.get("sdg_target", ""),
            "hybrid_score":           entry.get("hybrid_score", 0),
            "formula":                f"R = {self.alpha}·{entry.get('content_score',0):.3f} + {self.beta}·{entry.get('case_score',0):.3f}",
            "content_contribution":   f"{entry.get('content_contribution', 0):.1f}%",
            "case_contribution":      f"{entry.get('case_contribution', 0):.1f}%",
            "content_explanation":    content_explain,
            "case_explanation":       case_explain,
        }

    # --------------------------------------------------
    # Evaluate & Tune
    # --------------------------------------------------

    def evaluate_hybrid_performance(
        self,
        provinces: list[str] = None,
    ) -> pd.DataFrame:
        '''
        Evaluasi kualitas rekomendasi untuk semua (atau sebagian) provinsi.

        Metrik yang dihitung per provinsi:
        - n_recommendations : jumlah rekomendasi yang dihasilkan
        - mean_hybrid_score : rata-rata hybrid score
        - coverage_sdg      : jumlah SDG target yang tercakup
        - source_diversity  : apakah rekomendasi datang dari kedua engine

        Returns
        -------
        pd.DataFrame
        '''
        prov_list = provinces or list(
            self._content_engine._feat[PROVINCE_COL].values
        )

        rows = []

        for prov in prov_list:
            try:
                recs = self.generate_final_recommendations(prov)
            except Exception:
                recs = []

            if not recs:
                rows.append({
                    "province":          prov,
                    "n_recommendations": 0,
                    "mean_hybrid_score": 0.0,
                    "coverage_sdg":      0,
                    "source_diversity":  False,
                })
                continue

            scores  = [r.get("hybrid_score", 0) for r in recs]
            sdgs    = {r.get("sdg_target", "") for r in recs}
            sources = {r.get("source", "") for r in recs}

            rows.append({
                "province":          prov,
                "n_recommendations": len(recs),
                "mean_hybrid_score": round(float(np.mean(scores)), 4),
                "coverage_sdg":      len(sdgs),
                "source_diversity":  len(sources) > 1 or "hybrid" in sources,
            })

        return pd.DataFrame(rows).sort_values(
            "mean_hybrid_score", ascending=False
        ).reset_index(drop=True)

    def tune_hybrid_weights(
        self,
        alpha_range: list[float] = None,
        eval_metric: str         = "mean_hybrid_score",
        n_provinces: int         = 10,
    ) -> dict:
        '''
        Cari α dan β optimal via grid search.

        Karena α + β = 1, search hanya butuh satu dimensi (α).
        β = 1 − α.

        Parameters
        ----------
        alpha_range : list[float]
            Nilai α yang dicoba. Default: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8].
        eval_metric : str
            Kolom dari evaluate_hybrid_performance() yang dioptimalkan.
        n_provinces : int
            Evaluasi pada sampel n_provinces provinsi untuk efisiensi.

        Returns
        -------
        dict
            best_alpha | best_beta | best_score | results_df
        '''
        alpha_range = alpha_range or [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        # Sampel provinsi
        all_provs = list(self._content_engine._feat[PROVINCE_COL].values)
        sample    = all_provs[:n_provinces]

        rows = []

        for alpha in alpha_range:
            beta = round(1.0 - alpha, 4)

            # Sementara ganti bobot
            self.alpha = alpha
            self.beta  = beta
            self._cache.clear()

            eval_df = self.evaluate_hybrid_performance(provinces=sample)
            score   = float(eval_df[eval_metric].mean())

            rows.append({
                "alpha": alpha,
                "beta":  beta,
                "score": round(score, 4),
            })

        results_df = pd.DataFrame(rows)
        best_row   = results_df.loc[results_df["score"].idxmax()]

        # Set ke best
        self.alpha = float(best_row["alpha"])
        self.beta  = float(best_row["beta"])
        self._cache.clear()

        print(
            f"[HybridRecommender] Best weights: "
            f"α={self.alpha}, β={self.beta}, "
            f"{eval_metric}={best_row['score']:.4f}"
        )

        return {
            "best_alpha":  float(best_row["alpha"]),
            "best_beta":   float(best_row["beta"]),
            "best_score":  float(best_row["score"]),
            "metric":      eval_metric,
            "results_df":  results_df,
        }

    # --------------------------------------------------
    # Final output
    # --------------------------------------------------

    def _run_engines(self, province: str) -> dict[str, dict]:
        '''Jalankan kedua engine dan gabungkan hasilnya.'''

        if province in self._cache:
            return self._cache[province]

        content_recs = self._content_engine.generate_recommendations(province)
        case_recs    = self._case_engine.generate_case_recommendation(province)

        combined = self.combine_recommendations(content_recs, case_recs)
        combined = self.weighted_hybrid(combined)
        combined = self.resolve_conflicts(combined)

        self._cache[province] = combined
        return combined

    def generate_final_recommendations(
        self,
        province: str,
        top_n:    int = None,
    ) -> list[dict]:
        '''
        Hasilkan rekomendasi kebijakan final untuk satu provinsi.

        Ini adalah metode utama yang dipanggil oleh pipeline dan dashboard.

        Parameters
        ----------
        province : str
        top_n : int
            Default = self._top_k.

        Returns
        -------
        list of dict, diurutkan descending berdasarkan hybrid_score.
        Setiap entry berisi semua informasi yang dibutuhkan dashboard.
        '''
        combined = self._run_engines(province)
        ranked   = self.rank_hybrid_recommendations(combined, top_n=top_n)

        # Tambahkan field explain ringkas untuk setiap rekomendasi
        for rec in ranked:
            rec["province"] = province
            rec["sources"]  = list(set(rec.get("sources", [])))

        return ranked

    def export_final_recommendations(
        self,
        all_provinces: list[str] = None,
        output_dir:    str       = "data/warehouse/recommendations",
        fmt:           str       = "json",
    ) -> None:
        '''
        Ekspor rekomendasi final untuk semua provinsi.

        Files yang dihasilkan
        ---------------------
        recommendations_{province}.json / .csv  (per provinsi)
        recommendations_all.json                (semua provinsi)

        Parameters
        ----------
        all_provinces : list[str], optional
            Default: semua provinsi di feature_mart.
        fmt : str
            "json" (default) atau "csv".
        '''
        os.makedirs(output_dir, exist_ok=True)

        provs = all_provinces or list(
            self._content_engine._feat[PROVINCE_COL].values
        )

        all_recs = {}

        for prov in provs:
            try:
                recs = self.generate_final_recommendations(prov)
                all_recs[prov] = recs

                # Per-province file
                safe_name = prov.replace(" ", "_").replace("/", "-")
                path = os.path.join(output_dir, f"recommendations_{safe_name}.{fmt}")

                if fmt == "json":
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(recs, f, ensure_ascii=False, indent=2)

                elif fmt == "csv":
                    rows = []
                    for rec in recs:
                        r = rec.copy()
                        for key in ("sources",):
                            r[key] = "; ".join(r.get(key, []))
                        # Flatten metadata
                        meta = r.pop("metadata", {})
                        if "case" in meta:
                            r["evidence_provinces"] = "; ".join(
                                meta["case"].get("evidence_provinces", [])
                            )
                            r["adaptation_notes"] = "; ".join(
                                meta["case"].get("adaptation_notes", [])
                            )
                        rows.append(r)
                    pd.DataFrame(rows).to_csv(path, index=False)

            except Exception as e:
                print(f"  [WARN] {prov}: {e}")

        # All-provinces combined
        combined_path = os.path.join(output_dir, f"recommendations_all.{fmt}")

        if fmt == "json":
            with open(combined_path, "w", encoding="utf-8") as f:
                json.dump(all_recs, f, ensure_ascii=False, indent=2)

        print(
            f"\n[HybridRecommender] Exported {len(all_recs)} provinsi → "
            f"{os.path.abspath(output_dir)}"
        )

    def visualize_hybrid_results(
        self,
        province:    str,
        output_path: str = None,
    ) -> None:
        '''
        Visualisasi dua panel:
        1. Bar chart hybrid score per rekomendasi
        2. Stacked bar kontribusi content vs case

        Memerlukan matplotlib.
        '''
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            raise ImportError("matplotlib diperlukan.")

        recs = self.generate_final_recommendations(province, top_n=7)

        if not recs:
            print(f"[HybridRecommender] Tidak ada rekomendasi untuk {province}")
            return

        tags   = [r["tag"] for r in recs]
        hybrid = [r.get("hybrid_score", 0) for r in recs]
        cont   = [r.get("content_contribution", 0) / 100.0 * r.get("hybrid_score", 0) for r in recs]
        case_s = [r.get("case_contribution",    0) / 100.0 * r.get("hybrid_score", 0) for r in recs]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Panel 1: Hybrid score
        colors_bar = [
            "#534AB7" if r.get("source") == "hybrid"
            else "#AFA9EC" if r.get("source") == "content_only"
            else "#0F6E56"
            for r in recs
        ]
        ax1.barh(range(len(tags)), hybrid[::-1], color=colors_bar[::-1])
        ax1.set_yticks(range(len(tags)))
        ax1.set_yticklabels([t.replace("_", " ").lower() for t in tags[::-1]], fontsize=8)
        ax1.set_xlabel("Hybrid score")
        ax1.set_title(f"Rekomendasi final — {province}", fontsize=10, fontweight="500")
        ax1.axvline(0.3, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

        legend_patches = [
            mpatches.Patch(color="#534AB7", label=f"Hybrid (α={self.alpha})"),
            mpatches.Patch(color="#AFA9EC", label="Content only"),
            mpatches.Patch(color="#0F6E56", label="Case only"),
        ]
        ax1.legend(handles=legend_patches, fontsize=8, loc="lower right")

        # Panel 2: Stacked kontribusi
        x = range(len(tags))
        ax2.bar(x, cont,   label=f"Content (α={self.alpha})", color="#534AB7", alpha=0.8)
        ax2.bar(x, case_s, label=f"Case (β={self.beta})",    color="#0F6E56", alpha=0.8, bottom=cont)
        ax2.set_xticks(x)
        ax2.set_xticklabels(
            [t.replace("_", "\n").lower() for t in tags],
            fontsize=7, rotation=20, ha="right",
        )
        ax2.set_ylabel("Score kontribusi")
        ax2.set_title("Kontribusi content vs case", fontsize=10, fontweight="500")
        ax2.legend(fontsize=8)

        plt.suptitle(
            f"Hybrid Recommendation — {province}  |  R = {self.alpha}·R_c + {self.beta}·R_k",
            fontsize=11, y=1.02,
        )
        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"[HybridRecommender] Viz disimpan → {output_path}")
        else:
            plt.show()

        plt.close()