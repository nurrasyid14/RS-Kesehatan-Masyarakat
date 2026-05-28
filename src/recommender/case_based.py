'''
src/recommender/case_based.py
================================
CaseBasedRecommender — rekomendasi berbasis memori kasus wilayah.

Prinsip
-------
"Provinsi dengan kondisi serupa pernah ditangani bagaimana?"

Sistem mencari kasus historis yang mirip, lalu mengadaptasi solusinya
ke konteks wilayah baru. Ini adalah policy memory system.

Siklus CBR (Aamodt & Plaza, 1994):
    Retrieve → Reuse → Revise → Retain

Case Library
------------
Diinisialisasi dari data nyata BPS 2025 menggunakan provinsi-provinsi
dengan profil ekstrem sebagai "anchor cases":

    Jawa Barat / Jawa Timur  → high disease burden, high facility
    Papua / Maluku Utara     → high workforce per population
    Kepulauan Bangka Belitung → high accessibility barrier
    Nusa Tenggara Timur      → low workforce
    Jambi / Maluku           → low insurance coverage
    Kalimantan Utara         → low facility, high insurance

Setiap case berisi: problem (kondisi dominan), intervention (kebijakan),
outcome (target SDG), dan evidence (provinsi yang sudah menjalankan).

Referensi
---------
Aamodt, A., & Plaza, E. (1994). Case-Based Reasoning: Foundational Issues,
    Methodological Variations, and System Approaches. AI Communications.
Aggarwal (2016), Ch. 9 — Hybrid Recommender Systems.
'''

import os
import json
import copy

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


PROVINCE_COL = "Provinsi"

CONTENT_COLS = [
    "facility_index",
    "workforce_capacity_index",
    "disease_burden_index",
    "insurance_coverage_index",
    "accessibility_barrier_index",
]

# ---------------------------------------------------------
# Case Library — basis data BPS 2025
# ---------------------------------------------------------
# Setiap case merepresentasikan satu pola masalah yang telah diobservasi
# pada provinsi tertentu beserta intervensi yang direkomendasikan.
# "evidence_provinces": provinsi yang memiliki profil serupa di data BPS.

DEFAULT_CASE_LIBRARY: list[dict] = [

    # ── Beban penyakit tinggi + fasilitas memadai ────────
    {
        "case_id":    "CASE_001",
        "label":      "High disease burden, adequate facilities",
        "problem": {
            "high_disease_burden_index":     True,
            "high_facility_index":           True,
            "low_workforce_capacity_index":  False,
        },
        "intervention": [
            {
                "tag":        "INTENSIFIKASI_PROGRAM",
                "policy":     "Intensifikasi program pengendalian penyakit berbasis fasilitas yang ada",
                "sdg_target": "3.3",
                "priority":   1,
            },
            {
                "tag":        "PELATIHAN_KLINIS",
                "policy":     "Pelatihan klinisi untuk penatalaksanaan TBC, HIV, dan Malaria",
                "sdg_target": "3.3",
                "priority":   2,
            },
        ],
        "outcome":            "Penurunan angka kesakitan penyakit menular",
        "evidence_provinces": ["Jawa Timur", "Jawa Barat", "Jawa Tengah"],
        "sdg_targets":        ["3.3"],
        "effectiveness":      0.80,
    },

    # ── Tenaga tinggi, fasilitas rendah ─────────────────
    {
        "case_id":    "CASE_002",
        "label":      "High workforce, low facilities",
        "problem": {
            "high_workforce_capacity_index": True,
            "low_facility_index":            True,
            "high_disease_burden_index":     False,
        },
        "intervention": [
            {
                "tag":        "PEMBANGUNAN_FASKES",
                "policy":     "Pembangunan Puskesmas baru di daerah terpencil",
                "sdg_target": "3.b",
                "priority":   1,
            },
            {
                "tag":        "OPTIMASI_TENAGA",
                "policy":     "Optimasi penempatan tenaga kesehatan ke fasilitas baru",
                "sdg_target": "3.c",
                "priority":   2,
            },
        ],
        "outcome":            "Peningkatan cakupan layanan kesehatan dasar",
        "evidence_provinces": ["Papua", "Maluku Utara"],
        "sdg_targets":        ["3.b", "3.c"],
        "effectiveness":      0.72,
    },

    # ── Hambatan akses tinggi ────────────────────────────
    {
        "case_id":    "CASE_003",
        "label":      "High accessibility barriers",
        "problem": {
            "high_accessibility_barrier_index": True,
            "low_insurance_coverage_index":     False,
        },
        "intervention": [
            {
                "tag":        "SUBSIDI_TRANSPORT",
                "policy":     "Program subsidi transportasi ke fasilitas kesehatan",
                "sdg_target": "3.1",
                "priority":   1,
            },
            {
                "tag":        "TELEMEDICINE",
                "policy":     "Pengembangan layanan telemedicine dan Puskesmas keliling",
                "sdg_target": "3.8",
                "priority":   2,
            },
        ],
        "outcome":            "Peningkatan utilisasi layanan kesehatan",
        "evidence_provinces": ["Kepulauan Bangka Belitung", "Bali", "Lampung"],
        "sdg_targets":        ["3.1", "3.8"],
        "effectiveness":      0.68,
    },

    # ── Coverage JKN rendah ──────────────────────────────
    {
        "case_id":    "CASE_004",
        "label":      "Low insurance coverage",
        "problem": {
            "low_insurance_coverage_index":  True,
            "high_disease_burden_index":     False,
        },
        "intervention": [
            {
                "tag":        "SOSIALISASI_JKN",
                "policy":     "Kampanye masif pendaftaran BPJS PBI bagi masyarakat tidak mampu",
                "sdg_target": "3.8",
                "priority":   1,
            },
            {
                "tag":        "JAMKESDA",
                "policy":     "Pengembangan Jaminan Kesehatan Daerah (Jamkesda) sebagai pelengkap JKN",
                "sdg_target": "3.8",
                "priority":   2,
            },
        ],
        "outcome":            "Peningkatan cakupan UHC",
        "evidence_provinces": ["Jambi", "Maluku", "Riau"],
        "sdg_targets":        ["3.8"],
        "effectiveness":      0.75,
    },

    # ── Tenaga rendah + coverage rendah ─────────────────
    {
        "case_id":    "CASE_005",
        "label":      "Low workforce and low insurance coverage",
        "problem": {
            "low_workforce_capacity_index": True,
            "low_insurance_coverage_index": True,
        },
        "intervention": [
            {
                "tag":        "BEASISWA_NAKES",
                "policy":     "Program beasiswa ikatan dinas tenaga kesehatan untuk daerah 3T",
                "sdg_target": "3.c",
                "priority":   1,
            },
            {
                "tag":        "NUSANTARA_SEHAT",
                "policy":     "Penguatan program Nusantara Sehat dan internship wajib daerah terpencil",
                "sdg_target": "3.c",
                "priority":   1,
            },
            {
                "tag":        "PERLUASAN_JKN",
                "policy":     "Perluasan kepesertaan JKN bersamaan dengan penguatan SDM",
                "sdg_target": "3.8",
                "priority":   2,
            },
        ],
        "outcome":            "Peningkatan akses dan cakupan layanan kesehatan",
        "evidence_provinces": ["Nusa Tenggara Timur", "Kalimantan Selatan", "Sumatera Selatan"],
        "sdg_targets":        ["3.c", "3.8"],
        "effectiveness":      0.70,
    },

    # ── Fasilitas rendah, hambatan tinggi ────────────────
    {
        "case_id":    "CASE_006",
        "label":      "Low facility, high barrier",
        "problem": {
            "low_facility_index":               True,
            "high_accessibility_barrier_index": True,
        },
        "intervention": [
            {
                "tag":        "PUSKESMAS_TERPENCIL",
                "policy":     "Pembangunan prioritas Puskesmas rawat inap di kecamatan terpencil",
                "sdg_target": "3.b",
                "priority":   1,
            },
            {
                "tag":        "AMBULANS_DESA",
                "policy":     "Pengadaan ambulans desa dan sistem rujukan komunitas",
                "sdg_target": "3.1",
                "priority":   2,
            },
        ],
        "outcome":            "Penurunan kematian akibat keterlambatan pertolongan",
        "evidence_provinces": ["Papua Tengah", "Kepulauan Bangka Belitung"],
        "sdg_targets":        ["3.b", "3.1"],
        "effectiveness":      0.65,
    },

    # ── Profil seimbang, coverage tinggi ─────────────────
    {
        "case_id":    "CASE_007",
        "label":      "Balanced profile, high insurance",
        "problem": {
            "high_insurance_coverage_index":     True,
            "high_disease_burden_index":         False,
            "low_accessibility_barrier_index":   True,
        },
        "intervention": [
            {
                "tag":        "KUALITAS_LAYANAN",
                "policy":     "Peningkatan mutu layanan dan akreditasi fasilitas kesehatan",
                "sdg_target": "3.8",
                "priority":   2,
            },
            {
                "tag":        "PREVENTIF_PROMOTIF",
                "policy":     "Penguatan program preventif dan promotif kesehatan masyarakat",
                "sdg_target": "3.4",
                "priority":   3,
            },
        ],
        "outcome":            "Pemeliharaan status kesehatan yang sudah baik",
        "evidence_provinces": ["Kalimantan Utara", "Sulawesi Barat", "Papua Pegunungan"],
        "sdg_targets":        ["3.8", "3.4"],
        "effectiveness":      0.85,
    },
]


# ---------------------------------------------------------
# CaseBasedRecommender
# ---------------------------------------------------------

class CaseBasedRecommender:
    '''
    Rekomendasi kebijakan berbasis memori kasus historis wilayah.

    Siklus CBR: Retrieve → Reuse (Adapt) → Revise → Retain

    Parameters
    ----------
    feature_mart : pd.DataFrame
        Feature mart pre-scale dengan kolom *_index.
    scaled_mart : pd.DataFrame
        Feature mart ter-scale untuk perhitungan jarak.
    case_library : list[dict], optional
        Library kasus. Default: DEFAULT_CASE_LIBRARY (7 kasus dari BPS 2025).
    similarity_threshold : float
        Kasus dengan similarity < threshold tidak digunakan. Default 0.5.
    top_k_cases : int
        Jumlah kasus teratas yang di-retrieve. Default 3.
    '''

    def __init__(
        self,
        feature_mart:         pd.DataFrame,
        scaled_mart:          pd.DataFrame,
        case_library:         list[dict] = None,
        similarity_threshold: float      = 0.5,
        top_k_cases:          int        = 3,
    ) -> None:

        self._feat      = feature_mart.copy().reset_index(drop=True)
        self._scaled    = scaled_mart.copy().reset_index(drop=True)
        self._threshold = similarity_threshold
        self._top_k     = top_k_cases
        self._provinces = self._feat[PROVINCE_COL].values

        self._content_cols = [c for c in CONTENT_COLS if c in self._feat.columns]
        self._median       = self._feat[self._content_cols].median()

        self._case_library: list[dict] = (
            copy.deepcopy(case_library)
            if case_library
            else copy.deepcopy(DEFAULT_CASE_LIBRARY)
        )

        # Build case vectors untuk similarity search
        self._case_vectors: np.ndarray = self._build_case_vectors()

    # --------------------------------------------------
    # Case Library
    # --------------------------------------------------

    def build_case_library(self) -> list[dict]:
        '''
        Kembalikan case library yang sedang aktif.

        Returns
        -------
        list of dict
        '''
        return self._case_library

    def _build_case_vectors(self) -> np.ndarray:
        '''
        Konversi setiap case menjadi binary/float vector
        berdasarkan kondisi problem-nya.

        Urutan dimensi = CONTENT_COLS:
            [facility, workforce, disease, insurance, barrier]
        Encoding: +1 = "high_{col}", -1 = "low_{col}", 0 = tidak disebutkan.
        '''
        rows = []

        for case in self._case_library:
            prob = case["problem"]
            vec  = []

            for col in self._content_cols:
                if prob.get(f"high_{col}", False):
                    vec.append(1.0)
                elif prob.get(f"low_{col}", False):
                    vec.append(-1.0)
                else:
                    vec.append(0.0)

            rows.append(vec)

        return np.array(rows, dtype=float)

    def _province_to_problem_vector(self, province: str) -> np.ndarray:
        '''
        Konversi profil provinsi ke binary problem vector
        dalam ruang yang sama dengan case vectors.

        Encoding: +1 jika > 110% median, -1 jika < 90% median, 0 otherwise.
        '''
        row = self._feat[self._feat[PROVINCE_COL] == province].iloc[0]
        vec = []

        for col in self._content_cols:
            val = float(row[col])
            med = float(self._median[col])
            if med == 0:
                vec.append(0.0)
                continue

            ratio = val / med
            if ratio > 1.10:
                vec.append(1.0)
            elif ratio < 0.90:
                vec.append(-1.0)
            else:
                vec.append(0.0)

        return np.array(vec, dtype=float).reshape(1, -1)

    # --------------------------------------------------
    # Retrieve
    # --------------------------------------------------

    def retrieve_similar_cases(
        self,
        province: str,
        top_n:    int = None,
    ) -> list[dict]:
        '''
        Retrieve kasus paling mirip dengan masalah provinsi target.

        Similaritas dihitung antara problem vector provinsi dan
        problem vector setiap case dalam library.

        Returns
        -------
        list of dict, setiap entry = case + similarity_score.
        Diurutkan descending.
        '''
        top_n    = top_n or self._top_k
        prov_vec = self._province_to_problem_vector(province)

        # Handle all-zero vector (profil "biasa" tanpa kondisi ekstrem)
        if np.all(prov_vec == 0):
            # Gunakan similarity berbasis jarak Euclidean pada scaled index
            return self._retrieve_by_feature_distance(province, top_n)

        sims = self.compute_case_similarity(prov_vec)
        results = []

        for i, sim in enumerate(sims):
            if sim >= self._threshold:
                case = copy.deepcopy(self._case_library[i])
                case["similarity_score"] = round(float(sim), 4)
                results.append(case)

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_n]

    def _retrieve_by_feature_distance(
        self,
        province: str,
        top_n:    int,
    ) -> list[dict]:
        '''
        Fallback: retrieve menggunakan jarak cosine pada scaled features
        ketika profil provinsi tidak memiliki kondisi ekstrem.
        '''
        scaled_cols = [c for c in self._content_cols if c in self._scaled.columns]

        prov_scaled = self._scaled[
            self._scaled[PROVINCE_COL] == province
        ][scaled_cols].values

        if len(prov_scaled) == 0:
            return []

        results = []

        for i, case in enumerate(self._case_library):
            # Buat vector dari evidence provinces
            evid_provs = case.get("evidence_provinces", [])
            evid_rows  = self._scaled[
                self._scaled[PROVINCE_COL].isin(evid_provs)
            ][scaled_cols]

            if evid_rows.empty:
                continue

            evid_mean = evid_rows.mean(axis=0).values.reshape(1, -1)
            sim       = float(cosine_similarity(prov_scaled, evid_mean)[0][0])

            if sim >= self._threshold:
                case_copy = copy.deepcopy(case)
                case_copy["similarity_score"] = round(sim, 4)
                results.append(case_copy)

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_n]

    def compute_case_similarity(
        self,
        prov_vec: np.ndarray,
    ) -> np.ndarray:
        '''
        Hitung similarity antara satu problem vector
        dan semua case vectors.

        Menggunakan dot-product similarity pada space ternary {-1, 0, +1}.

        Returns
        -------
        np.ndarray, shape (n_cases,)
        '''
        n_dims = self._case_vectors.shape[1]

        # Matching score: proporsi dimensi yang arah-nya sama
        scores = []

        for cv in self._case_vectors:
            match   = np.sum((prov_vec[0] == cv) & (cv != 0))
            defined = np.sum(cv != 0)
            score   = float(match) / max(defined, 1)
            scores.append(score)

        return np.array(scores)

    # --------------------------------------------------
    # Adapt (Reuse + Revise)
    # --------------------------------------------------

    def adapt_previous_solution(
        self,
        case:     dict,
        province: str,
    ) -> dict:
        '''
        Sesuaikan solusi case dengan konteks provinsi baru.

        Adaptasi yang dilakukan:
        - Tambahkan konteks geografis (pulau/wilayah)
        - Sesuaikan prioritas berdasarkan severity kondisi
        - Tambahkan catatan adaptasi

        Returns
        -------
        dict — case yang sudah diadaptasi
        '''
        adapted = copy.deepcopy(case)

        row = self._feat[self._feat[PROVINCE_COL] == province].iloc[0]

        # Cari kondisi paling parah di provinsi ini
        severity_notes = []

        for col in self._content_cols:
            val = float(row[col])
            med = float(self._median[col])
            if med == 0:
                continue
            ratio = val / med

            if col in ("disease_burden_index", "accessibility_barrier_index") and ratio > 1.25:
                severity_notes.append(
                    f"{col.replace('_index','').replace('_',' ')} sangat tinggi ({ratio:.1f}× median)"
                )
            elif col in ("facility_index", "workforce_capacity_index", "insurance_coverage_index") and ratio < 0.75:
                severity_notes.append(
                    f"{col.replace('_index','').replace('_',' ')} sangat rendah ({ratio:.1f}× median)"
                )

        adapted["adapted_for"]       = province
        adapted["adaptation_notes"]  = severity_notes
        adapted["original_case_id"]  = case["case_id"]

        # Boost priority pada intervensi yang relevan dengan severity
        for interv in adapted["intervention"]:
            if any("sangat" in note for note in severity_notes):
                interv["priority"] = max(1, interv["priority"] - 1)

        return adapted

    # --------------------------------------------------
    # Generate
    # --------------------------------------------------

    def generate_case_recommendation(
        self,
        province: str,
        top_n:    int = None,
    ) -> list[dict]:
        '''
        Hasilkan rekomendasi berbasis kasus historis untuk satu provinsi.

        Alur: Retrieve → Adapt → Format

        Returns
        -------
        list of dict, diurutkan berdasarkan similarity × effectiveness.
        '''
        cases = self.retrieve_similar_cases(province, top_n=top_n)

        if not cases:
            return []

        recommendations = []

        for case in cases:

            adapted  = self.adapt_previous_solution(case, province)
            sim_score = case.get("similarity_score", 0.5)
            eff_score = case.get("effectiveness",    0.7)

            for interv in adapted["intervention"]:
                score = sim_score * eff_score * (1.0 / interv["priority"])
                rec = {
                    "province":          province,
                    "tag":               interv["tag"],
                    "policy":            interv["policy"],
                    "sdg_target":        interv["sdg_target"],
                    "priority":          interv["priority"],
                    "score":             round(score, 4),
                    "source":            "case_based",
                    "case_id":           adapted["original_case_id"],
                    "case_label":        adapted["label"],
                    "evidence_provinces": adapted["evidence_provinces"],
                    "similarity_score":  sim_score,
                    "effectiveness":     eff_score,
                    "adaptation_notes":  adapted["adaptation_notes"],
                }
                recommendations.append(rec)

        # Dedup by tag — ambil yang score tertinggi
        seen:  dict[str, dict] = {}
        for rec in sorted(recommendations, key=lambda x: x["score"], reverse=True):
            if rec["tag"] not in seen:
                seen[rec["tag"]] = rec

        result = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
        return result

    def explain_case_reasoning(
        self,
        province: str,
        tag:      str,
    ) -> dict:
        '''
        Jelaskan kenapa case dan intervensi tertentu dipilih.

        Returns
        -------
        dict
            province | tag | retrieved_cases | matching_conditions
            | adaptation_notes | evidence_provinces
        '''
        cases = self.retrieve_similar_cases(province)

        matching_conditions = []
        prov_vec = self._province_to_problem_vector(province)[0]

        for col, val in zip(self._content_cols, prov_vec):
            if val == 1.0:
                matching_conditions.append(f"high_{col}")
            elif val == -1.0:
                matching_conditions.append(f"low_{col}")

        relevant_cases = [
            c for c in cases
            if any(i["tag"] == tag for i in c["intervention"])
        ]

        return {
            "province":             province,
            "tag":                  tag,
            "matching_conditions":  matching_conditions,
            "retrieved_cases":      [c["case_id"] for c in cases],
            "relevant_cases":       [c["case_id"] for c in relevant_cases],
            "evidence_provinces":   [
                p for c in relevant_cases
                for p in c.get("evidence_provinces", [])
            ],
            "adaptation_notes":     [
                note
                for c in relevant_cases
                for note in self.adapt_previous_solution(c, province).get("adaptation_notes", [])
            ],
        }

    # --------------------------------------------------
    # Retain
    # --------------------------------------------------

    def update_case_library(
        self,
        new_case:   dict,
        validate:   bool = True,
    ) -> None:
        '''
        Tambahkan kasus baru ke library (fase Retain dalam siklus CBR).

        Parameters
        ----------
        new_case : dict
            Harus memiliki key: case_id, label, problem, intervention,
            outcome, evidence_provinces, effectiveness.
        validate : bool
            Jika True, validasi struktur sebelum menambahkan.
        '''
        required_keys = {"case_id", "label", "problem", "intervention",
                         "outcome", "evidence_provinces"}

        if validate:
            missing = required_keys - set(new_case.keys())
            if missing:
                raise ValueError(
                    f"Kunci yang wajib tidak ada: {missing}"
                )

            # Cek duplikat case_id
            existing_ids = {c["case_id"] for c in self._case_library}
            if new_case["case_id"] in existing_ids:
                raise ValueError(
                    f"case_id '{new_case['case_id']}' sudah ada di library."
                )

        self._case_library.append(copy.deepcopy(new_case))
        self._case_vectors = self._build_case_vectors()

        print(
            f"[CaseBasedRecommender] Case '{new_case['case_id']}' "
            f"ditambahkan. Total: {len(self._case_library)} kasus."
        )

    def evaluate_case_effectiveness(self) -> pd.DataFrame:
        '''
        Ringkasan efektivitas setiap case dalam library.

        Returns
        -------
        pd.DataFrame
            Kolom: case_id | label | effectiveness | n_interventions
                   | n_evidence_provinces | sdg_targets
        '''
        rows = []

        for case in self._case_library:
            rows.append({
                "case_id":              case["case_id"],
                "label":                case["label"],
                "effectiveness":        case.get("effectiveness", None),
                "n_interventions":      len(case["intervention"]),
                "n_evidence_provinces": len(case.get("evidence_provinces", [])),
                "sdg_targets":          ", ".join(case.get("sdg_targets", [])),
            })

        return pd.DataFrame(rows).sort_values("effectiveness", ascending=False)

    # --------------------------------------------------
    # Export
    # --------------------------------------------------

    def export_case_recommendations(
        self,
        recommendations: list[dict],
        output_path:     str,
        fmt:             str = "json",
    ) -> None:
        '''Ekspor rekomendasi ke JSON atau CSV.'''

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if fmt == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(recommendations, f, ensure_ascii=False, indent=2)

        elif fmt == "csv":
            rows = []
            for rec in recommendations:
                r = rec.copy()
                for key in ("evidence_provinces", "adaptation_notes"):
                    r[key] = "; ".join(r.get(key, []))
                rows.append(r)
            pd.DataFrame(rows).to_csv(output_path, index=False)

        print(f"[CaseBasedRecommender] Exported → {output_path}")

    # --------------------------------------------------
    # Visualisasi
    # --------------------------------------------------

    def visualize_case_network(
        self,
        output_path: str = None,
    ) -> None:
        '''
        Visualisasi jaringan kasus — menampilkan hubungan antar case
        berdasarkan kesamaan evidence provinces.

        Memerlukan matplotlib.
        '''
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            raise ImportError("matplotlib diperlukan.")

        # Hitung overlap evidence provinces
        n     = len(self._case_library)
        fig, ax = plt.subplots(figsize=(10, 7))

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        cx     = np.cos(angles) * 3.0
        cy     = np.sin(angles) * 3.0

        colors = plt.cm.tab10.colors

        # Gambar edge berdasarkan shared evidence provinces
        for i in range(n):
            for j in range(i + 1, n):
                evid_i = set(self._case_library[i].get("evidence_provinces", []))
                evid_j = set(self._case_library[j].get("evidence_provinces", []))
                shared = len(evid_i & evid_j)

                if shared > 0:
                    ax.plot(
                        [cx[i], cx[j]], [cy[i], cy[j]],
                        color="gray", linewidth=shared * 0.5, alpha=0.4,
                    )

        # Gambar node
        for i, case in enumerate(self._case_library):
            ax.scatter(cx[i], cy[i], s=300, color=colors[i % 10], zorder=5)
            ax.text(
                cx[i], cy[i] + 0.35,
                case["case_id"],
                ha="center", va="bottom", fontsize=8, fontweight="500",
            )
            ax.text(
                cx[i], cy[i] - 0.35,
                case["label"][:30],
                ha="center", va="top", fontsize=7, color="gray",
            )

        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.axis("off")
        ax.set_title(
            "Case Network — tebal garis = banyak shared evidence provinces",
            fontsize=10, fontweight="500",
        )
        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"[CaseBasedRecommender] Viz disimpan → {output_path}")
        else:
            plt.show()

        plt.close()