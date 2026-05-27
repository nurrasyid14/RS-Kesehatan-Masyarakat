'''
src/warehousing
===============
Data warehouse layer — menghasilkan dimension tables dan fact tables
dari data yang sudah melalui prep pipeline.

Struktur output
---------------
data/warehouse/
    dimensions/
        dim_facilities.csv
        dim_workforce.csv
        dim_disease.csv
        dim_insurance.csv
        dim_healthcare_barrier.csv
    facts/
        fact_healthcare_raw.csv
        fact_healthcare_feature_mart.csv
        fact_clustering.csv
        fact_similarity.csv
        fact_recommendation.csv

Modul
-----
dimensions_maker    : DimensionMaker — 5 dimension tables
facts_maker         : FactsMaker — 5 fact tables + feature mart
similarity_score    : Similarity matrix antar provinsi
features_mart       : Feature mart builder (helper FactsMaker)

Penggunaan cepat
----------------
from src.warehousing import run_warehousing

run_warehousing(engineered_df, scaled_df)
'''

from .dimensions_maker import DimensionMaker
from .facts_maker      import FactsMaker
from .similarity_score import build_similarity
from .features_mart    import build_feature_mart, FEATURE_MART_COLS


__all__ = [
    "DimensionMaker",
    "FactsMaker",
    "run_warehousing",
]