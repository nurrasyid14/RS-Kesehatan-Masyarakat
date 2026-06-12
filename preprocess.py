'''
src/prep/pipeline.py
====================
Orchestrator pipeline ETL & preprocessing.

Merangkai 4 langkah prep secara berurutan:

    raw CSV
        ↓ structural_cleaner.clean()
    cleaned_df
        ↓ imputer.impute()
    imputed_df
        ↓ feature_engineer.engineer()
    engineered_df
        ↓ normalizer.normalize()
    scaled_df

Output file (default: folder yang sama dengan input CSV)
---------------------------------------------------------
cleaned_data.csv
engineered_data.csv
scaled_data.csv
'''

import os
import pandas as pd

from src.prep.structural_cleaner import clean
from src.prep.imputer            import impute
from src.prep.feature_engineer   import engineer
from src.prep.normalizer         import normalize


# ---------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------

def _save(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    print(f"  saved -> {path}")


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------

def run_pipeline(
    file_path:      str,
    scaling_method: str  = "minmax",
    save_outputs:   bool = True,
    verbose:        bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Jalankan full preprocessing pipeline.

    Parameters
    ----------
    file_path : str
        Path ke raw CSV (hasil merge BPS atau Merged.csv).
    scaling_method : str
        "minmax" (default) atau "standard".
    save_outputs : bool
        Jika True, simpan cleaned/engineered/scaled ke folder input.
    verbose : bool
        Teruskan ke modul imputer dan feature_engineer untuk logging.

    Returns
    -------
    tuple[raw_df, cleaned_df, engineered_df, scaled_df]
    '''

    output_dir = os.path.dirname(os.path.abspath(file_path))

    # ── Step 0 : Load ────────────────────────────────────
    raw_df = pd.read_csv(file_path)

    # ── Step 1 : Clean ───────────────────────────────────
    cleaned_df = clean(raw_df)

    # ── Step 2 : Impute ──────────────────────────────────
    imputed_df = impute(cleaned_df, verbose=verbose)

    # ── Step 3 : Feature Engineering ─────────────────────
    engineered_df = engineer(imputed_df, verbose=verbose)

    # ── Step 4 : Normalize ───────────────────────────────
    scaled_df = normalize(engineered_df, method=scaling_method)

    # ── Save ─────────────────────────────────────────────
    if save_outputs:

        print("\n" + "=" * 50)
        print("PREP PIPELINE — OUTPUT FILES")
        print("=" * 50)

        _save(cleaned_df,    os.path.join(output_dir, "cleaned_data.csv"))
        _save(engineered_df, os.path.join(output_dir, "engineered_data.csv"))
        _save(scaled_df,     os.path.join(output_dir, "scaled_data.csv"))

        print("=" * 50 + "\n")

    return raw_df, cleaned_df, engineered_df, scaled_df
