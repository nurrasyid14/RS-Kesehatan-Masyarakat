'''
src/prep/structural_cleaner.py
==============================
Step 1 — Pembersihan struktural raw DataFrame.

Tanggung jawab
--------------
- Strip whitespace & newline dari nama kolom
- Rename kolom ke format snake_case standar
- Konversi invalid values (NA, –, kosong) → np.nan
- Konversi kolom numerik secara aman
- Konversi kolom persentase → rasio (÷ 100)
  dan rename dengan suffix _ratio

Tidak melakukan
---------------
- Imputasi missing value  → imputer.py
- Feature engineering     → feature_engineer.py
- Scaling                 → normalizer.py
'''

import re
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Konstanta
# ---------------------------------------------------------

INVALID_VALUES = ["NA", "NaN", "N/A", "-", "–", ""]

PROVINCE_COL = "Provinsi"


# ---------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------

def _strip_column_names(df: pd.DataFrame) -> pd.DataFrame:
    '''Strip whitespace dan newline dari semua nama kolom.'''
    df.columns = [
        col.strip().replace("\n", " ")
        for col in df.columns
    ]
    return df


def _replace_invalid(df: pd.DataFrame) -> pd.DataFrame:
    '''Ganti semua representasi invalid menjadi np.nan.'''
    return df.replace(INVALID_VALUES, np.nan)


def _convert_numerics(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Konversi kolom non-Provinsi ke numerik secara aman.
    Koma desimal (,) dikonversi ke titik (.) terlebih dahulu.
    Karakter non-numerik selain titik dan minus dihapus.
    '''
    df = df.copy()

    for col in df.columns:

        if col == PROVINCE_COL:
            continue

        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(",", ".", regex=False)
            .str.replace(r"[^\d.\-]", "", regex=True)
        )

        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _convert_percentage_to_ratio(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Kolom yang mengandung "_ratio" pada nama aslinya
    (sudah di-rename dari Persentase) dibagi 100.

    Kolom dengan kata "Persentase" di nama-nya juga
    di-rename dengan pola:
        "Persentase ..." → "..._ratio"
    '''
    df = df.copy()
    rename_map = {}

    for col in df.columns:

        if "persentase" in col.lower():

            df[col] = df[col] / 100.0

            new_name = (
                re.sub(r"(?i)persentase\s*", "", col)
                .strip()
                .lower()
                .replace(" ", "_")
            )
            if not new_name.endswith("_ratio"):
                new_name += "_ratio"

            rename_map[col] = new_name

        # Kolom sudah bernama *_ratio tapi nilainya masih 0–100
        # (heuristik: max value > 1 → belum dibagi 100)
        elif col.lower().endswith("_ratio"):

            if df[col].dropna().max() > 1.0:
                df[col] = df[col] / 100.0

    df.rename(columns=rename_map, inplace=True)
    return df


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Bersihkan DataFrame secara struktural.

    Urutan operasi
    --------------
    1. Strip nama kolom
    2. Ganti invalid values → NaN
    3. Konversi numerik
    4. Konversi persentase → rasio

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame hasil load CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame bersih, siap untuk imputasi.
    '''

    df = df.copy()
    df = _strip_column_names(df)
    df = _replace_invalid(df)
    df = _convert_numerics(df)
    df = _convert_percentage_to_ratio(df)

    return df
