'''
src/warehousing/features_mart.py
=================================
Helper — membangun feature mart dari engineered DataFrame.

Feature mart adalah ringkasan kompak dari seluruh composite index
per provinsi. Ini menjadi input utama clustering dan recommendation.

Output: fact_healthcare_feature_mart
-------------------------------------
province_id | provinsi | facility_index | workforce_capacity_index
            | disease_burden_index | insurance_coverage_index
            | accessibility_barrier_index | treatment_effectiveness_index
'''

import pandas as pd

PROVINCE_COL = "Provinsi"

FEATURE_MART_COLS = [
    "facility_index",
    "workforce_capacity_index",
    "disease_burden_index",
    "insurance_coverage_index",
    "accessibility_barrier_index",
    "treatment_effectiveness_index",
]


def build_feature_mart(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Ekstrak composite index per provinsi menjadi feature mart.

    Parameters
    ----------
    df : pd.DataFrame
        Output dari feature_engineer.engineer().
        Harus sudah memiliki kolom *_index.

    Returns
    -------
    pd.DataFrame
        fact_healthcare_feature_mart:
        province_id | provinsi | [6 index cols]
    '''

    available = [c for c in FEATURE_MART_COLS if c in df.columns]
    missing   = set(FEATURE_MART_COLS) - set(available)

    if missing:
        print(
            f"  [FeaturesMart] Kolom index tidak ditemukan, di-skip → "
            + ", ".join(sorted(missing))
        )

    mart = df[[PROVINCE_COL] + available].copy()
    mart = mart.sort_values(PROVINCE_COL).reset_index(drop=True)
    mart.insert(0, "province_id", mart.index + 1)

    return mart