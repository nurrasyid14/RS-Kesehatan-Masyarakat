'''
src/warehousing/similarity_score.py
=====================================
Menghitung similarity matrix antar provinsi berbasis feature mart.

Metode: Cosine Similarity
--------------------------
Dipilih karena:
- Robust terhadap perbedaan skala absolut antar provinsi
- Umum digunakan dalam content-based recommendation
  (Aggarwal, 2016, Ch. 4)
- Bekerja baik pada fitur yang sudah di-scale ke [0,1]

Output: fact_similarity
-----------------------
province_a | province_b | similarity_score

Hanya pasangan (a, b) dengan a < b yang disimpan (upper triangle)
untuk menghindari duplikasi. Diagonal (a == a) dikecualikan.
'''

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

PROVINCE_COL = "Provinsi"


def build_similarity(
    scaled_mart: pd.DataFrame,
    threshold:   float = 0.0,
) -> pd.DataFrame:
    '''
    Hitung cosine similarity antar semua pasangan provinsi.

    Parameters
    ----------
    scaled_mart : pd.DataFrame
        Feature mart yang sudah di-scale (output normalizer).
        Harus memiliki kolom Provinsi dan kolom numerik index.
    threshold : float
        Pasangan dengan similarity < threshold tidak disimpan.
        Default 0.0 → simpan semua pasangan.

    Returns
    -------
    pd.DataFrame
        fact_similarity:
        province_a | province_b | similarity_score
        Diurutkan descending berdasarkan similarity_score.
    '''

    if PROVINCE_COL not in scaled_mart.columns:
        raise KeyError(f"Kolom '{PROVINCE_COL}' tidak ditemukan.")

    provinces = scaled_mart[PROVINCE_COL].values
    feature_cols = [
        c for c in scaled_mart.columns
        if c not in (PROVINCE_COL, "province_id")
        and pd.api.types.is_numeric_dtype(scaled_mart[c])
    ]

    if not feature_cols:
        raise ValueError(
            "Tidak ada kolom numerik ditemukan untuk menghitung similarity."
        )

    matrix = scaled_mart[feature_cols].values
    sim_matrix = cosine_similarity(matrix)

    n = len(provinces)
    rows = []

    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim_matrix[i, j])
            if score >= threshold:
                rows.append({
                    "province_a":       provinces[i],
                    "province_b":       provinces[j],
                    "similarity_score": round(score, 6),
                })

    fact_similarity = (
        pd.DataFrame(rows)
        .sort_values("similarity_score", ascending=False)
        .reset_index(drop=True)
    )

    return fact_similarity