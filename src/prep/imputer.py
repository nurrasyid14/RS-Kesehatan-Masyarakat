'''
src/prep/imputer.py
===================
Step 2 — Imputasi missing value.

Strategi imputasi per dimensi
------------------------------
Dimensi          Kolom target            Strategi
--------         -------------           ---------
Fasilitas        RS, Puskesmas           median
Tenaga           Tenaga *                median
Penyakit         TBC, HIV, Kusta, dll.   median
Jaminan          *_ratio (insurance)     mean
Hambatan         *_ratio (barrier)       mean
Index (feat.)    *_index                 mean

Alasan pemilihan strategi
--------------------------
- median  : untuk count data (RS, tenaga) yang sering miring kanan
            karena provinsi besar (DKI, Jawa Timur) dominan
- mean    : untuk rasio & proporsi yang distribusinya lebih simetris

Missing yang tersisa setelah imputasi (misalnya seluruh kolom kosong)
akan dibiarkan agar terdeteksi saat validasi downstream.
'''

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Peta dimensi → keyword kolom → strategi
# ---------------------------------------------------------

IMPUTATION_MAP: list[dict] = [
    {
        "label":    "fasilitas",
        "keywords": ["rumah_sakit", "puskesmas", "rs_"],
        "strategy": "median",
    },
    {
        "label":    "tenaga",
        "keywords": ["tenaga"],
        "strategy": "median",
    },
    {
        "label":    "penyakit",
        "keywords": ["tbc", "hiv", "kusta", "malaria", "dbd"],
        "strategy": "median",
    },
    {
        "label":    "jaminan",
        "keywords": ["bpjs", "jamkesda", "asuransi", "perusahaan", "pbi"],
        "strategy": "mean",
    },
    {
        "label":    "hambatan",
        "keywords": [
            "biaya_berobat", "biaya_transport", "sarana_transportasi",
            "waktu_tunggu", "mengobati_sendiri", "pendamping",
            "tidak_perlu", "lainnya", "barrier"
        ],
        "strategy": "mean",
    },
    {
        "label":    "index",
        "keywords": ["_index"],
        "strategy": "mean",
    },
]

PROVINCE_COL = "Provinsi"


# ---------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------

def _match_cols(
    df_cols:  list[str],
    keywords: list[str],
    assigned: set[str]
) -> list[str]:
    '''
    Kembalikan kolom numerik yang mengandung salah satu keyword
    dan belum ditugaskan ke dimensi lain.
    '''
    return [
        col for col in df_cols
        if col not in assigned
        and col != PROVINCE_COL
        and any(k in col.lower() for k in keywords)
    ]


def _fill_with_strategy(
    df:       pd.DataFrame,
    cols:     list[str],
    strategy: str
) -> pd.DataFrame:
    '''Terapkan strategi imputasi pada kolom tertentu.'''

    if strategy == "median":
        fill_values = df[cols].median()
    elif strategy == "mean":
        fill_values = df[cols].mean()
    else:
        raise ValueError(f"Strategi tidak dikenal: '{strategy}'")

    df[cols] = df[cols].fillna(fill_values)
    return df


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------

def impute(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    '''
    Imputasi missing value pada DataFrame yang sudah dibersihkan
    secara struktural.

    Parameters
    ----------
    df : pd.DataFrame
        Output dari structural_cleaner.clean().
    verbose : bool
        Jika True, cetak ringkasan kolom yang diimputasi.

    Returns
    -------
    pd.DataFrame
        DataFrame tanpa missing value pada kolom yang ter-cover
        oleh IMPUTATION_MAP.
    '''

    df = df.copy()

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    assigned: set[str] = set()

    for rule in IMPUTATION_MAP:

        cols = _match_cols(numeric_cols, rule["keywords"], assigned)

        if not cols:
            continue

        # Hanya imputasi kolom yang benar-benar punya NaN
        missing_cols = [c for c in cols if df[c].isna().any()]

        if missing_cols:
            df = _fill_with_strategy(df, missing_cols, rule["strategy"])

            if verbose:
                print(
                    f"[imputer] {rule['label']:12s} "
                    f"({rule['strategy']:6s}) → {len(missing_cols)} kolom"
                )

        assigned.update(cols)

    # Fallback: kolom numerik yang belum ter-cover → mean
    uncovered = [
        c for c in numeric_cols
        if c not in assigned and df[c].isna().any()
    ]

    if uncovered:
        df = _fill_with_strategy(df, uncovered, "mean")

        if verbose:
            print(
                f"[imputer] fallback     (mean  ) → "
                f"{len(uncovered)} kolom: {uncovered}"
            )

    return df
