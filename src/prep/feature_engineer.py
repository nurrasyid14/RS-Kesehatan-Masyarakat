'''
src/prep/feature_engineer.py
=============================
Step 3 — Pembentukan composite index per dimensi kesehatan.

Index yang dihasilkan
---------------------
Index                       Makna
--------------------------  -------------------------------------------
facility_index              Ketersediaan fasilitas kesehatan
workforce_capacity_index    Kapasitas tenaga kesehatan
disease_burden_index        Beban penyakit (semakin tinggi = lebih berat)
insurance_coverage_index    Cakupan jaminan kesehatan
accessibility_barrier_index Hambatan akses (semakin tinggi = lebih berat)
treatment_effectiveness_idx Efektivitas pengobatan

Catatan desain — polaritas
--------------------------
Tidak semua kolom dalam satu dimensi "searah".
Contoh: disease_burden_index
  - HIV kasus baru, kusta, malaria, DBD → POSITIF (naik = lebih buruk)
  - Keberhasilan pengobatan TBC         → NEGATIF (naik = lebih baik)
    → diinvert: burden_TBC_tx = 1 − keberhasilan_tbc

Kolom yang perlu diinvert ditandai di INVERT_COLS.
Inversi hanya berlaku jika nilai sudah dalam skala 0–1 (rasio).
Jika belum, inversi tidak diterapkan dan peringatan ditampilkan.
'''

import warnings
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Konfigurasi dimensi
# ---------------------------------------------------------

# Setiap entry mendefinisikan satu composite index.
# "keywords"    : substring yang dicari di nama kolom (lowercase)
# "exclude"     : substring yang TIDAK boleh ada di nama kolom
# "invert"      : substring yang kolom-nya perlu diinvert (1 − x)
# "output_col"  : nama kolom index yang dihasilkan

DIMENSION_CONFIG: list[dict] = [
    {
        "output_col": "facility_index",
        "keywords":   ["rumah_sakit", "puskesmas"],
        "exclude":    [],
        "invert":     [],
    },
    {
        "output_col": "workforce_capacity_index",
        "keywords":   ["tenaga"],
        "exclude":    [],
        "invert":     [],
    },
    {
        "output_col": "disease_burden_index",
        "keywords":   ["tbc", "hiv", "kusta", "malaria", "dbd"],
        "exclude":    [],
        # Keberhasilan pengobatan TBC → nilainya makin tinggi = makin BAIK
        # Untuk burden index, kita invert agar polaritas konsisten
        "invert":     ["keberhasilan"],
    },
    {
        "output_col": "insurance_coverage_index",
        "keywords":   ["bpjs", "jamkesda", "asuransi", "perusahaan", "pbi"],
        "exclude":    [],
        "invert":     [],
    },
    {
        "output_col": "accessibility_barrier_index",
        "keywords":   [
            "biaya_berobat", "biaya_transport", "sarana_transportasi",
            "waktu_tunggu", "mengobati_sendiri", "pendamping",
            "tidak_perlu", "lainnya",
            # fallback dari nama kolom mentah
            "tidak_berobat",
        ],
        "exclude":    ["_index"],
        "invert":     [],
    },
    {
        "output_col": "treatment_effectiveness_index",
        # Proxy: keberhasilan pengobatan TBC saja untuk saat ini
        # Bisa diperluas jika ada data recovery rate lain
        "keywords":   ["keberhasilan"],
        "exclude":    [],
        "invert":     [],
    },
]

PROVINCE_COL  = "Provinsi"
INDEX_SUFFIX  = "_index"


# ---------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------

def _match_cols(
    all_cols:  list[str],
    keywords:  list[str],
    exclude:   list[str],
    assigned:  set[str]
) -> list[str]:
    '''
    Kembalikan kolom yang:
    - Mengandung minimal satu keyword
    - Tidak mengandung satu pun kata exclude
    - Belum ditugaskan ke index lain
    - Bukan kolom Provinsi atau kolom index yang sudah ada
    '''
    matched = []

    for col in all_cols:

        cl = col.lower()

        if col in assigned:
            continue
        if col == PROVINCE_COL:
            continue
        if cl.endswith(INDEX_SUFFIX):
            continue
        if not any(k in cl for k in keywords):
            continue
        if any(e in cl for e in exclude):
            continue

        matched.append(col)

    return matched


def _safe_invert(
    series: pd.Series,
    col:    str
) -> pd.Series:
    '''
    Invert kolom rasio: nilai_baru = 1 − nilai_lama.
    Hanya aman jika semua nilai dalam [0, 1].
    Jika tidak, kembalikan series asli + warning.
    '''
    max_val = series.dropna().max()
    min_val = series.dropna().min()

    if max_val > 1.0 or min_val < 0.0:
        warnings.warn(
            f"[feature_engineer] Kolom '{col}' tidak dalam skala [0,1] "
            f"(min={min_val:.2f}, max={max_val:.2f}). "
            f"Inversi dilewati — normalisasi dulu sebelum feature engineering "
            f"jika inversi diperlukan.",
            UserWarning,
            stacklevel=3
        )
        return series

    return 1.0 - series


def _build_index(
    df:      pd.DataFrame,
    cols:    list[str],
    invert:  list[str]
) -> pd.Series:
    '''
    Hitung rata-rata kolom (dengan inversi untuk kolom yang sesuai).
    '''
    working = df[cols].copy()

    for col in cols:
        if any(inv in col.lower() for inv in invert):
            working[col] = _safe_invert(working[col], col)

    return working.mean(axis=1)


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------

def engineer(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    '''
    Buat composite index per dimensi kesehatan.

    Parameters
    ----------
    df : pd.DataFrame
        Output dari imputer.impute().
        Kolom sudah bersih dan tanpa NaN.
    verbose : bool
        Cetak kolom yang dikontribusikan ke tiap index.

    Returns
    -------
    pd.DataFrame
        DataFrame original + kolom index baru.
    '''

    df = df.copy()

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    assigned: set[str] = set()

    for cfg in DIMENSION_CONFIG:

        cols = _match_cols(
            numeric_cols,
            cfg["keywords"],
            cfg["exclude"],
            assigned
        )

        if not cols:
            if verbose:
                print(
                    f"[feature_engineer] {cfg['output_col']:35s} "
                    f"→ tidak ada kolom yang cocok, dilewati"
                )
            continue

        df[cfg["output_col"]] = _build_index(df, cols, cfg["invert"])

        # Kolom yang berkontribusi ke treatment_effectiveness_index
        # boleh juga berkontribusi ke disease_burden_index (shared)
        # → jangan ditandai assigned untuk kasus tersebut
        if cfg["output_col"] != "treatment_effectiveness_index":
            assigned.update(cols)

        if verbose:
            print(
                f"[feature_engineer] {cfg['output_col']:35s} "
                f"← {len(cols)} kolom"
            )
            for c in cols:
                inv_mark = " [inv]" if any(
                    inv in c.lower() for inv in cfg["invert"]
                ) else ""
                print(f"    {c}{inv_mark}")

    return df
