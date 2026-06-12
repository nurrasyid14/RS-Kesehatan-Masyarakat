'''
clustering.py
=============
Entry point pipeline clustering.

Membaca scaled feature mart dari warehouse, menjalankan K-Means dan GMM,
lalu menyimpan hasilnya kembali ke warehouse dan ke folder models/.

Flow
----
    data/warehouse/facts/fact_clustering.csv          (placeholder)
    data/warehouse/facts/fact_healthcare_feature_mart.csv
    data/scaled_data.csv
        ↓
    tune → fit → evaluate
        ↓
    KMeansClustering  +  GMMClustering
        ↓
    data/warehouse/facts/fact_clustering.csv          (updated)
    data/warehouse/facts/fact_clustering_gmm.csv      (soft labels)
    data/warehouse/facts/fact_cluster_probabilities.csv
    models/kmeans_k{k}.pkl
    models/gmm_k{k}_{cov}.pkl
    data/warehouse/facts/cluster_evaluation.csv

Menjalankan
-----------
    python clustering.py
    python clustering.py --method kmeans --k 4
    python clustering.py --method gmm --k 4 --cov full
    python clustering.py --tune          # jalankan hyperparameter tuning dulu
'''

import argparse
import os
import sys

import numpy as np
import pandas as pd

from src.clustering   import KMeansClustering, GMMClustering
from src.warehousing.facts_maker import FactsMaker


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATA_DIR        = "data"
WAREHOUSE_DIR   = os.path.join(DATA_DIR, "warehouse")
FACTS_DIR       = os.path.join(WAREHOUSE_DIR, "facts")
MODELS_DIR      = "models"

SCALED_PATH     = os.path.join(DATA_DIR, "scaled_data.csv")
FEAT_MART_PATH  = os.path.join(FACTS_DIR, "fact_healthcare_feature_mart.csv")
CLUSTERING_PATH = os.path.join(FACTS_DIR, "fact_clustering.csv")

# Output
KM_RESULT_PATH  = os.path.join(FACTS_DIR, "fact_clustering.csv")
GMM_RESULT_PATH = os.path.join(FACTS_DIR, "fact_clustering_gmm.csv")
PROBA_PATH      = os.path.join(FACTS_DIR, "fact_cluster_probabilities.csv")
EVAL_PATH       = os.path.join(FACTS_DIR, "cluster_evaluation.csv")

PROVINCE_COL    = "Provinsi"

# Kolom yang dipakai sebagai feature matrix
# (kolom *_index_scaled dari fact_clustering placeholder)
SCALED_SUFFIX   = "_scaled"


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _check_inputs() -> None:
    missing = [p for p in [SCALED_PATH, FEAT_MART_PATH, CLUSTERING_PATH]
               if not os.path.exists(p)]
    if missing:
        print("\n[ERROR] File input tidak ditemukan:")
        for p in missing:
            print(f"  {p}")
        print(
            "\nPastikan preprocess.py dan warehousing.py sudah dijalankan:\n"
            "  python preprocess.py\n"
            "  python warehousing.py\n"
        )
        sys.exit(1)


def _load_feature_matrix(
    fact_clustering: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    '''
    Ekstrak feature matrix dari fact_clustering.

    Menggunakan kolom *_scaled (hasil normalisasi)
    sebagai input algoritma.

    Returns
    -------
    (subset_df, X_array, scaled_cols)
    '''
    scaled_cols = [
        c for c in fact_clustering.columns
        if c.endswith(SCALED_SUFFIX)
    ]

    if not scaled_cols:
        raise ValueError(
            "Tidak ada kolom *_scaled ditemukan di fact_clustering. "
            "Pastikan warehousing.py menghasilkan kolom tersebut."
        )

    X = fact_clustering[scaled_cols].values.astype(float)

    return fact_clustering, X, scaled_cols


def _print_separator(title: str) -> None:
    print("\n" + "=" * 58)
    print(f"  {title}")
    print("=" * 58)


def _save_evaluation(
    km_eval:  dict,
    gmm_eval: dict,
    path:     str,
) -> None:
    '''Simpan metrik evaluasi kedua model ke satu CSV.'''
    rows = [
        {"model": "kmeans", **km_eval},
        {"model": "gmm",    **gmm_eval},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  saved -> {path}")


# ---------------------------------------------------------
# Sub-pipeline: KMeans
# ---------------------------------------------------------

def run_kmeans(
    fact_clustering: pd.DataFrame,
    X:               np.ndarray,
    n_clusters:      int,
    tune:            bool,
    k_range:         range,
) -> tuple[KMeansClustering, pd.DataFrame, dict]:
    '''
    Jalankan pipeline KMeans lengkap.

    Returns
    -------
    (model, updated_fact_clustering, eval_dict)
    '''
    _print_separator("K-MEANS CLUSTERING")

    km = KMeansClustering(n_clusters=n_clusters)

    # Tuning opsional
    if tune:
        print("  [tune] Mencari k dan n_init optimal...")
        result = km.tune_hyperparameters(X, k_range=k_range)
        n_clusters = result["best_k"]
        n_init     = result["best_n_init"]
        print(
            f"  [tune] best_k={n_clusters}, "
            f"best_n_init={n_init}, "
            f"silhouette={result['best_score']:.4f}"
        )
        km = KMeansClustering(n_clusters=n_clusters, n_init=n_init)
    else:
        print(f"  k = {n_clusters} (fixed)")

    # Elbow & silhouette summary (ringkas, tidak block pipeline)
    print("\n  [elbow]")
    elbow_df = km.elbow_method(X, k_range=k_range)
    print(elbow_df.to_string(index=False))

    # Fit
    print(f"\n  [fit] K-Means k={n_clusters}...")
    labels = km.fit_predict(X)

    # Evaluate
    eval_dict = km.evaluate_clustering()
    print(
        f"\n  [eval] silhouette={eval_dict['silhouette_score']:.4f}  "
        f"davies_bouldin={eval_dict['davies_bouldin']:.4f}  "
        f"calinski_harabasz={eval_dict['calinski_harabasz']:.4f}"
    )

    # Update fact table
    updated = FactsMaker.update_clustering_fact(
        fact_clustering = fact_clustering,
        cluster_labels  = labels,
        method          = "kmeans",
    )

    # Tampilkan distribusi cluster
    dist = updated.groupby("cluster_id")[PROVINCE_COL].count()
    print("\n  [distribusi cluster]")
    for cid, count in dist.items():
        provinces = updated.loc[
            updated["cluster_id"] == cid, PROVINCE_COL
        ].tolist()
        print(f"    cluster {cid}: {count} provinsi -> {provinces}")

    return km, updated, eval_dict


# ---------------------------------------------------------
# Sub-pipeline: GMM
# ---------------------------------------------------------

def run_gmm(
    fact_clustering:  pd.DataFrame,
    X:                np.ndarray,
    n_components:     int,
    covariance_type:  str,
    tune:             bool,
    k_range:          range,
) -> tuple[GMMClustering, pd.DataFrame, pd.DataFrame, dict]:
    '''
    Jalankan pipeline GMM lengkap.

    Returns
    -------
    (model, updated_fact_clustering, proba_df, eval_dict)
    '''
    _print_separator("GMM CLUSTERING")

    gmm = GMMClustering(
        n_components    = n_components,
        covariance_type = covariance_type,
    )

    # Tuning opsional
    if tune:
        print("  [tune] Mencari k, covariance_type, reg_covar optimal (BIC)...")
        result = gmm.tune_hyperparameters(X, k_range=k_range)
        n_components    = result["best_k"]
        covariance_type = result["best_covariance_type"]
        reg_covar       = result["best_reg_covar"]
        print(
            f"  [tune] best_k={n_components}, "
            f"cov={covariance_type}, "
            f"reg={reg_covar}, "
            f"BIC={result['best_score']:.4f}"
        )
        gmm = GMMClustering(
            n_components    = n_components,
            covariance_type = covariance_type,
            reg_covar       = reg_covar,
        )
    else:
        print(f"  k = {n_components}, covariance_type = {covariance_type} (fixed)")

    # BIC analysis ringkas
    print("\n  [BIC analysis]")
    bic_df = gmm.bic_analysis(X, k_range=k_range, cov_types=[covariance_type])
    print(bic_df.head(5).to_string(index=False))

    # Fit
    print(f"\n  [fit] GMM k={n_components} ({covariance_type})...")
    labels = gmm.fit_predict(X)

    # Evaluate
    eval_dict = gmm.evaluate_clustering()
    print(
        f"\n  [eval] silhouette={eval_dict['silhouette_score']:.4f}  "
        f"BIC={eval_dict['bic']:.4f}  "
        f"AIC={eval_dict['aic']:.4f}  "
        f"converged={eval_dict['converged']}"
    )

    # Update fact table
    updated = FactsMaker.update_clustering_fact(
        fact_clustering = fact_clustering.copy(),
        cluster_labels  = labels,
        method          = "gmm",
    )

    # Probabilitas membership
    provinces = fact_clustering[PROVINCE_COL].values
    proba_df  = gmm.get_cluster_probabilities(provinces=list(provinces))

    # Tampilkan distribusi cluster
    dist = updated.groupby("cluster_id")[PROVINCE_COL].count()
    print("\n  [distribusi cluster]")
    for cid, count in dist.items():
        provinces_list = updated.loc[
            updated["cluster_id"] == cid, PROVINCE_COL
        ].tolist()
        print(f"    cluster {cid}: {count} provinsi -> {provinces_list}")

    return gmm, updated, proba_df, eval_dict


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def clustering_pipeline(
    n_clusters:      int  = 4,
    covariance_type: str  = "full",
    tune:            bool = False,
    k_range:         range = range(2, 8),
    save_models:     bool = True,
) -> dict:
    '''
    Full clustering pipeline — K-Means dan GMM.

    Parameters
    ----------
    n_clusters : int
        Jumlah cluster awal. Diabaikan jika tune=True.
    covariance_type : str
        Tipe kovarians GMM. Diabaikan jika tune=True.
    tune : bool
        Jika True, jalankan hyperparameter search sebelum fit.
    k_range : range
        Rentang k untuk tuning/elbow/BIC analysis.
    save_models : bool
        Simpan model ke models/ jika True.

    Returns
    -------
    dict
        km_model, gmm_model, km_fact, gmm_fact,
        proba_df, km_eval, gmm_eval
    '''
    _check_inputs()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FACTS_DIR,  exist_ok=True)

    # ── Load ─────────────────────────────────────────────
    _print_separator("LOAD DATA")
    fact_clustering = pd.read_csv(CLUSTERING_PATH)
    print(f"  fact_clustering : {fact_clustering.shape[0]} provinsi")

    fact_clustering, X, scaled_cols = _load_feature_matrix(fact_clustering)
    print(f"  feature matrix  : {X.shape[0]} × {X.shape[1]}")
    print(f"  features        : {[c.replace(SCALED_SUFFIX,'') for c in scaled_cols]}")

    # ── K-Means ───────────────────────────────────────────
    km_model, km_fact, km_eval = run_kmeans(
        fact_clustering = fact_clustering,
        X               = X,
        n_clusters      = n_clusters,
        tune            = tune,
        k_range         = k_range,
    )

    # ── GMM ───────────────────────────────────────────────
    gmm_model, gmm_fact, proba_df, gmm_eval = run_gmm(
        fact_clustering  = fact_clustering,
        X                = X,
        n_components     = n_clusters,
        covariance_type  = covariance_type,
        tune             = tune,
        k_range          = k_range,
    )

    # ── Save facts ────────────────────────────────────────
    _print_separator("EXPORT")

    km_fact.to_csv(KM_RESULT_PATH, index=False)
    print(f"  saved -> {KM_RESULT_PATH}")

    gmm_fact.to_csv(GMM_RESULT_PATH, index=False)
    print(f"  saved -> {GMM_RESULT_PATH}")

    proba_df.to_csv(PROBA_PATH, index=False)
    print(f"  saved -> {PROBA_PATH}")

    _save_evaluation(km_eval, gmm_eval, EVAL_PATH)

    # ── Save models ───────────────────────────────────────
    if save_models:
        km_k   = km_model.n_clusters
        gmm_k  = gmm_model.n_components
        gmm_cv = gmm_model.covariance_type

        km_model.save_model(
            os.path.join(MODELS_DIR, f"kmeans_k{km_k}.pkl")
        )
        gmm_model.save_model(
            os.path.join(MODELS_DIR, f"gmm_k{gmm_k}_{gmm_cv}.pkl")
        )

    _print_separator("SELESAI")
    print(f"  K-Means  -> k={km_model.n_clusters}, silhouette={km_eval['silhouette_score']:.4f}")
    print(f"  GMM      -> k={gmm_model.n_components}, BIC={gmm_eval['bic']:.4f}\n")

    return {
        "km_model":  km_model,
        "gmm_model": gmm_model,
        "km_fact":   km_fact,
        "gmm_fact":  gmm_fact,
        "proba_df":  proba_df,
        "km_eval":   km_eval,
        "gmm_eval":  gmm_eval,
    }


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clustering pipeline — K-Means & GMM"
    )
    parser.add_argument(
        "--k", type=int, default=4,
        help="Jumlah cluster (default: 4). Diabaikan jika --tune."
    )
    parser.add_argument(
        "--cov", type=str, default="full",
        choices=["full", "tied", "diag", "spherical"],
        help="Covariance type GMM (default: full)."
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Jalankan hyperparameter search sebelum fit."
    )
    parser.add_argument(
        "--k-min", type=int, default=2,
        help="k minimum untuk search/elbow (default: 2)."
    )
    parser.add_argument(
        "--k-max", type=int, default=7,
        help="k maksimum untuk search/elbow (default: 7)."
    )
    parser.add_argument(
        "--no-save-models", action="store_true",
        help="Jangan simpan model .pkl."
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = _parse_args()

    clustering_pipeline(
        n_clusters      = args.k,
        covariance_type = args.cov,
        tune            = args.tune,
        k_range         = range(args.k_min, args.k_max + 1),
        save_models     = not args.no_save_models,
    )
