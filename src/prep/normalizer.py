'''
src/prep/normalizer.py
======================
Step 4 — Normalisasi / scaling fitur.

Metode yang tersedia
--------------------
"minmax"    MinMaxScaler   → output [0, 1]
            Cocok untuk distance-based (K-Means, cosine similarity)

"standard"  StandardScaler → output μ=0, σ=1
            Cocok untuk GMM dan algoritma yang sensitif terhadap variance

Kolom yang tidak di-scale
--------------------------
- Provinsi (string identifier)
- Kolom non-numerik lainnya

Output
------
DataFrame dengan index integer di-reset, kolom Provinsi
dikembalikan ke posisi pertama.
'''

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler


PROVINCE_COL    = "Provinsi"
VALID_METHODS   = ("minmax", "standard")


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------

def normalize(
    df:     pd.DataFrame,
    method: str = "minmax"
) -> pd.DataFrame:
    '''
    Scale seluruh kolom numerik DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output dari feature_engineer.engineer().
    method : str
        "minmax" (default) atau "standard".

    Returns
    -------
    pd.DataFrame
        DataFrame ter-scale dengan Provinsi tetap di kolom pertama.

    Raises
    ------
    ValueError
        Jika method tidak dikenal.
    '''

    if method not in VALID_METHODS:
        raise ValueError(
            f"method harus salah satu dari {VALID_METHODS}, "
            f"bukan '{method}'"
        )

    df = df.copy()

    # Pisahkan kolom non-numerik agar tidak ikut di-scale
    non_numeric = df.select_dtypes(exclude=np.number).columns.tolist()
    numeric     = df.select_dtypes(include=np.number).columns.tolist()

    province_series = None

    if PROVINCE_COL in non_numeric:
        province_series = df[PROVINCE_COL].reset_index(drop=True)
        non_numeric.remove(PROVINCE_COL)

    # Kolom non-numerik lain (jika ada) dipertahankan apa adanya
    other_non_numeric = df[non_numeric] if non_numeric else None

    # ---------------------------------------------------------
    # Scaling
    # ---------------------------------------------------------

    scaler = MinMaxScaler() if method == "minmax" else StandardScaler()

    scaled_array = scaler.fit_transform(df[numeric])

    scaled_df = pd.DataFrame(
        scaled_array,
        columns=numeric
    )

    # ---------------------------------------------------------
    # Gabungkan kembali
    # ---------------------------------------------------------

    if province_series is not None:
        scaled_df.insert(0, PROVINCE_COL, province_series)

    if other_non_numeric is not None:
        for col in other_non_numeric.columns:
            scaled_df[col] = other_non_numeric[col].values

    return scaled_df
