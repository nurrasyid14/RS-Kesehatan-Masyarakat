'''
src/warehousing/warehouse.py
=============================
Orchestrator warehousing — merangkai DimensionMaker dan FactsMaker,
lalu mengekspor semua tabel ke data/warehouse/.

Dipanggil dari warehousing.py di root project.

Output layout
-------------
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
'''

import os
import pandas as pd

from src.warehousing.dimensions_maker import DimensionMaker
from src.warehousing.facts_maker      import FactsMaker


# ---------------------------------------------------------
# File name mapping
# ---------------------------------------------------------

DIM_FILENAMES = {
    "facilities": "dim_facilities.csv",
    "workforce":  "dim_workforce.csv",
    "disease":    "dim_disease.csv",
    "insurance":  "dim_insurance.csv",
    "barrier":    "dim_healthcare_barrier.csv",
}

FACT_FILENAMES = {
    "raw":            "fact_healthcare_raw.csv",
    "feature_mart":   "fact_healthcare_feature_mart.csv",
    "clustering":     "fact_clustering.csv",
    "similarity":     "fact_similarity.csv",
    "recommendation": "fact_recommendation.csv",
}


# ---------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------

def _ensure_dirs(dim_dir: str, fact_dir: str) -> None:
    os.makedirs(dim_dir,  exist_ok=True)
    os.makedirs(fact_dir, exist_ok=True)


def _save_tables(
    tables:    dict[str, pd.DataFrame],
    filenames: dict[str, str],
    output_dir: str,
    label:     str,
) -> None:
    '''Simpan dict tabel ke folder output, cetak path ke console.'''

    print(f"\n  [{label}]")

    for key, df in tables.items():
        filename = filenames[key]
        path = os.path.join(output_dir, filename)
        df.to_csv(path, index=False)
        print(f"    saved -> {path}")


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------

def run_warehousing(
    engineered_df:        pd.DataFrame,
    scaled_df:            pd.DataFrame,
    warehouse_dir:        str   = "data/warehouse",
    similarity_threshold: float = 0.0,
    verbose:              bool  = True,
) -> dict[str, dict[str, pd.DataFrame]]:
    '''
    Jalankan full warehousing pipeline.

    Langkah
    -------
    1. DimensionMaker.make_all_dimensions()  → 5 dim tables
    2. FactsMaker.make_all_facts()           → 5 fact tables
    3. Export semua ke data/warehouse/

    Parameters
    ----------
    engineered_df : pd.DataFrame
        Output feature_engineer.engineer().
    scaled_df : pd.DataFrame
        Output normalizer.normalize().
    warehouse_dir : str
        Root folder warehouse. Default "data/warehouse".
    similarity_threshold : float
        Pasangan provinsi dengan similarity < threshold tidak disimpan.
    verbose : bool
        Cetak progress ke console.

    Returns
    -------
    dict:
        {
            "dimensions": { "facilities": df, "workforce": df, ... },
            "facts":      { "raw": df, "feature_mart": df, ... }
        }
    '''

    # Intercept and rename workforce_index to workforce_capacity_index dynamically
    if "workforce_index" in engineered_df.columns:
        engineered_df = engineered_df.rename(columns={"workforce_index": "workforce_capacity_index"})
    if "workforce_index" in scaled_df.columns:
        scaled_df = scaled_df.rename(columns={"workforce_index": "workforce_capacity_index"})

    dim_dir  = os.path.join(warehouse_dir, "dimensions")
    fact_dir = os.path.join(warehouse_dir, "facts")

    _ensure_dirs(dim_dir, fact_dir)

    # ── Step 1 : Dimensions ──────────────────────────────
    dim_maker  = DimensionMaker(engineered_df)
    dimensions = dim_maker.make_all_dimensions()

    # ── Step 2 : Facts ───────────────────────────────────
    fact_maker = FactsMaker(engineered_df, scaled_df)
    facts      = fact_maker.make_all_facts(
        similarity_threshold=similarity_threshold
    )

    # ── Step 3 : Export ──────────────────────────────────
    if verbose:
        print("\n" + "=" * 56)
        print("WAREHOUSING — EXPORT")
        print("=" * 56)

    _save_tables(dimensions, DIM_FILENAMES,  dim_dir,  "dimensions")
    _save_tables(facts,      FACT_FILENAMES, fact_dir, "facts")

    if verbose:
        print("\n" + "=" * 56)
        print(f"Warehouse ready -> {os.path.abspath(warehouse_dir)}")
        print("=" * 56 + "\n")

    return {"dimensions": dimensions, "facts": facts}


if __name__ == "__main__":
    import sys
    # Load engineered and scaled data from local paths and run warehousing pipeline
    try:
        engineered_data_path = "data/engineered_data.csv"
        scaled_data_path = "data/scaled_data.csv"
        
        if not os.path.exists(engineered_data_path) or not os.path.exists(scaled_data_path):
            print(f"[ERROR] Required input files not found: {engineered_data_path} or {scaled_data_path}")
            print("Please run preprocess.py first.")
            sys.exit(1)
            
        print("[Warehouse CLI] Loading data and running warehousing pipeline...")
        eng_df = pd.read_csv(engineered_data_path)
        scaled_df = pd.read_csv(scaled_data_path)
        
        if "workforce_index" in eng_df.columns:
            eng_df = eng_df.rename(columns={"workforce_index": "workforce_capacity_index"})
        if "workforce_index" in scaled_df.columns:
            scaled_df = scaled_df.rename(columns={"workforce_index": "workforce_capacity_index"})
        
        run_warehousing(eng_df, scaled_df)
        print("[Warehouse CLI] Finished successfully.")
    except Exception as e:
        print(f"[ERROR] Warehouse CLI failed: {e}")
        sys.exit(1)