'''
src/prep
========
ETL & preprocessing pipeline untuk Hybrid Health Recommender System.

Referensi:
    Charu Aggarwal, Recommender Systems: The Textbook, 2016, Springer.

Modul
-----
structural_cleaner   : standardisasi struktural raw DataFrame
merger               : penggabungan 5 sumber BPS
imputer              : imputasi missing value
feature_engineer     : pembentukan composite index per dimensi
normalizer           : scaling fitur untuk algoritma downstream

Penggunaan cepat
----------------
from src.prep import run_pipeline

scaled_df, engineered_df, cleaned_df = run_pipeline("data/Data Kesehatan Masyarakat (Merged).csv")
'''

from .structural_cleaner import clean
from .imputer            import impute
from .feature_engineer   import engineer
from .normalizer         import normalize
from .merger             import merge_sources



__all__ = [
    "clean",
    "impute",
    "engineer",
    "normalize",
    "merge_sources",
]
