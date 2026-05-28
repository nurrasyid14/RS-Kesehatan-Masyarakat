'''
profiling.py
============
Entry point pipeline profiling.

Membaca hasil clustering dari warehouse, menjalankan
ContentRepresentation dan RegionalProfiler, lalu mengekspor
semua output ke data/warehouse/profiles/.

Flow
----
    data/warehouse/facts/fact_clustering.csv          (kmeans labels)
    data/warehouse/facts/fact_clustering_gmm.csv      (gmm labels)
    data/warehouse/facts/fact_healthcare_feature_mart.csv
    data/scaled_data.csv
        ↓
    ContentRepresentation  →  similarity matrix, content profiles, dim-reduksi
    RegionalProfiler       →  cluster profiles, policy tags, SDG mapping
        ↓
    data/warehouse/profiles/
        similarity_matrix.csv
        content_profiles.csv
        pca_projection.csv
        cluster_profiles.csv           (kmeans)
        cluster_profiles_gmm.csv       (gmm)
        cluster_policy_tags.csv
        cluster_sdg3_map.csv
        cluster_comparison.csv
        region_summaries.json
        cluster_radar.png              (opsional, butuh matplotlib)

Menjalankan
-----------
    python profiling.py
    python profiling.py --method kmeans         # profil hanya dari kmeans
    python profiling.py --method gmm
    python profiling.py --similarity euclidean
    python profiling.py --no-viz                # skip radar chart
'''

import argparse
import json
import os
import sys

import pandas as pd

from src.profiler import ContentRepresentation, RegionalProfiler


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATA_DIR        = "data"
WAREHOUSE_DIR   = os.path.join(DATA_DIR, "warehouse")
FACTS_DIR       = os.path.join(WAREHOUSE_DIR, "facts")
PROFILES_DIR    = os.path.join(WAREHOUSE_DIR, "profiles")

SCALED_PATH     = os.path.join(DATA_DIR, "scaled_data.csv")
FEAT_MART_PATH  = os.path.join(FACTS_DIR, "fact_healthcare_feature_mart.csv")
KM_FACT_PATH    = os.path.join(FACTS_DIR, "fact_clustering.csv")
GMM_FACT_PATH   = os.path.join(FACTS_DIR, "fact_clustering_gmm.csv")

PROVINCE_COL    = "Provinsi"


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _check_inputs(method: str) -> None:

    required = [SCALED_PATH, FEAT_MART_PATH]

    if method in ("kmeans", "both"):
        required.append(KM_FACT_PATH)
    if method in ("gmm", "both"):
        required.append(GMM_FACT_PATH)

    missing = [p for p in required if not os.path.exists(p)]

    if missing:
        print("\n[ERROR] File input tidak ditemukan:")
        for p in missing:
            print(f"  {p}")
        print(
            "\nPastikan pipeline sebelumnya sudah dijalankan:\n"
            "  python preprocess.py\n"
            "  python warehousing.py\n"
            "  python clustering.py\n"
        )
        sys.exit(1)


def _print_separator(title: str) -> None:
    print("\n" + "=" * 58)
    print(f"  {title}")
    print("=" * 58)


def _load_cluster_labels(path: str, method: str) -> pd.Series:
    '''
    Baca cluster_id dari fact_clustering CSV.
    Diurutkan berdasarkan Provinsi agar konsisten dengan feature_mart.
    '''
    df = pd.read_csv(path)

    if "cluster_id" not in df.columns:
        raise ValueError(
            f"Kolom 'cluster_id' tidak ditemukan di {path}. "
            "Pastikan clustering.py sudah dijalankan."
        )

    if df["cluster_id"].isna().any():
        raise ValueError(
            f"Kolom 'cluster_id' masih mengandung NA di {path}. "
            "Pastikan clustering pipeline berjalan sampai selesai."
        )

    return (
        df.sort_values(PROVINCE_COL)
        .reset_index(drop=True)["cluster_id"]
        .astype(int)
    )


def _align_feature_mart(
    feature_mart:    pd.DataFrame,
    cluster_labels:  pd.Series,
    clustering_fact: pd.DataFrame,
) -> pd.DataFrame:
    '''
    Pastikan urutan baris feature_mart sesuai dengan clustering_fact
    (keduanya sudah sort by Provinsi — ini hanya verifikasi).
    '''
    prov_fm   = feature_mart.sort_values(PROVINCE_COL)[PROVINCE_COL].values
    prov_cl   = clustering_fact.sort_values(PROVINCE_COL)[PROVINCE_COL].values

    if list(prov_fm) != list(prov_cl):
        raise ValueError(
            "Urutan provinsi di feature_mart dan fact_clustering tidak sama. "
            "Periksa apakah kedua file berasal dari run pipeline yang sama."
        )

    return feature_mart.sort_values(PROVINCE_COL).reset_index(drop=True)


# ---------------------------------------------------------
# Sub-pipeline: ContentRepresentation
# ---------------------------------------------------------

def run_content_representation(
    scaled_df:        pd.DataFrame,
    similarity_metric: str,
    output_dir:       str,
) -> ContentRepresentation:
    '''
    Bangun feature vectors, hitung similarity matrix, reduce dimensions.

    Returns
    -------
    ContentRepresentation (sudah fit)
    '''
    _print_separator("CONTENT REPRESENTATION")

    cr = ContentRepresentation(scaled_df)

    # Feature vectors
    vectors = cr.build_feature_vectors()
    print(f"  feature vectors : {vectors.shape[0]} provinsi × {vectors.shape[1]} fitur")

    # Similarity matrix
    print(f"\n  [similarity] metric = {similarity_metric}")
    sim_df = cr.compute_similarity_matrix(metric=similarity_metric)
    sim_path = os.path.join(output_dir, "similarity_matrix.csv")
    cr.save_similarity_matrix(sim_path)

    # Content profiles (semantic tags per provinsi)
    print("\n  [content profiles]")
    content_profiles = cr.generate_content_profiles()
    cp_path = os.path.join(output_dir, "content_profiles.csv")
    content_profiles.to_csv(cp_path, index=False)
    print(f"  saved → {cp_path}")

    # PCA projection (2D untuk visualisasi / downstream)
    print("\n  [PCA 2D projection]")
    pca_df   = cr.reduce_dimensions(method="pca", n_components=2)
    pca_path = os.path.join(output_dir, "pca_projection.csv")
    pca_df.to_csv(pca_path, index=False)
    print(f"  saved → {pca_path}")

    return cr


# ---------------------------------------------------------
# Sub-pipeline: RegionalProfiler
# ---------------------------------------------------------

def run_regional_profiler(
    feature_mart:   pd.DataFrame,
    cluster_labels: pd.Series,
    method:         str,
    output_dir:     str,
    visualize:      bool,
) -> RegionalProfiler:
    '''
    Profiling cluster, policy tags, SDG mapping, region summaries.

    Returns
    -------
    RegionalProfiler (sudah fit)
    '''
    _print_separator(f"REGIONAL PROFILER ({method.upper()})")

    rp = RegionalProfiler(
        feature_df     = feature_mart,
        cluster_labels = cluster_labels,
        cluster_method = method,
    )

    # Cluster profiles
    print("\n  [cluster profiles]")
    profiles = rp.generate_cluster_profiles()
    print(profiles[["n_provinces", "semantic_label", "description"]].to_string())

    # Comparison table
    print("\n  [cluster comparison]")
    comparison = rp.compare_clusters()
    print(comparison.to_string())

    # Policy tags
    print("\n  [policy tags]")
    policy = rp.generate_policy_tags()
    for cid, row in policy.iterrows():
        print(
            f"    cluster {cid}: "
            + (", ".join(row["policy_tags"]) if row["policy_tags"] else "— tidak ada prioritas —")
        )

    # SDG 3 mapping
    print("\n  [SDG 3 mapping]")
    sdg = rp.map_sdg3_indicators()
    print(sdg.to_string())

    # Export semua tabel
    suffix    = "" if method == "kmeans" else f"_{method}"
    prof_dir  = output_dir

    rp.export_profiles(output_dir=prof_dir, fmt="csv")

    # Rename file yang punya suffix supaya tidak overwrite antar method
    if suffix:
        for base in ["cluster_profiles", "cluster_policy_tags",
                     "cluster_sdg3_map", "cluster_comparison"]:
            src  = os.path.join(prof_dir, f"{base}.csv")
            dest = os.path.join(prof_dir, f"{base}{suffix}.csv")
            if os.path.exists(src):
                os.replace(src, dest)

    # Region summaries — satu JSON per provinsi
    print("\n  [region summaries]")
    provinces = feature_mart[PROVINCE_COL].tolist()
    summaries = {}

    for prov in provinces:
        summaries[prov] = rp.create_region_summary(prov)

    summary_path = os.path.join(output_dir, f"region_summaries{suffix}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"  saved → {summary_path}")

    # Radar chart
    if visualize:
        print("\n  [visualisasi radar]")
        radar_path = os.path.join(output_dir, f"cluster_radar{suffix}.png")
        try:
            rp.visualize_profiles(output_path=radar_path)
        except ImportError as e:
            print(f"  [skip] {e}")

    return rp


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def profiling_pipeline(
    method:            str  = "both",
    similarity_metric: str  = "cosine",
    visualize:         bool = True,
) -> dict:
    '''
    Full profiling pipeline.

    Parameters
    ----------
    method : str
        "kmeans" | "gmm" | "both" (default).
    similarity_metric : str
        "cosine" (default) atau "euclidean".
    visualize : bool
        Buat radar chart jika True (butuh matplotlib).

    Returns
    -------
    dict
        content_repr, km_profiler (opsional), gmm_profiler (opsional)
    '''
    _check_inputs(method)
    os.makedirs(PROFILES_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────
    _print_separator("LOAD DATA")

    scaled_df    = pd.read_csv(SCALED_PATH)
    feature_mart = pd.read_csv(FEAT_MART_PATH)

    # Pastikan scaled_df hanya berisi kolom index untuk ContentRepresentation
    index_cols = [
        c for c in scaled_df.columns
        if c.endswith("_index") or c == PROVINCE_COL
    ]
    scaled_mart = scaled_df[index_cols].copy() if index_cols else scaled_df

    print(f"  scaled_mart     : {scaled_mart.shape[0]} provinsi × {scaled_mart.shape[1]-1} fitur")
    print(f"  feature_mart    : {feature_mart.shape[0]} provinsi × {feature_mart.shape[1]-1} fitur")

    result = {}

    # ── Content Representation ────────────────────────────
    cr = run_content_representation(
        scaled_df         = scaled_mart,
        similarity_metric = similarity_metric,
        output_dir        = PROFILES_DIR,
    )
    result["content_repr"] = cr

    # ── Regional Profiler — KMeans ────────────────────────
    if method in ("kmeans", "both"):

        km_fact    = pd.read_csv(KM_FACT_PATH)
        km_labels  = _load_cluster_labels(KM_FACT_PATH, "kmeans")
        feat_aligned = _align_feature_mart(feature_mart, km_labels, km_fact)

        rp_km = run_regional_profiler(
            feature_mart   = feat_aligned,
            cluster_labels = km_labels,
            method         = "kmeans",
            output_dir     = PROFILES_DIR,
            visualize      = visualize,
        )
        result["km_profiler"] = rp_km

    # ── Regional Profiler — GMM ───────────────────────────
    if method in ("gmm", "both"):

        gmm_fact   = pd.read_csv(GMM_FACT_PATH)
        gmm_labels = _load_cluster_labels(GMM_FACT_PATH, "gmm")
        feat_aligned = _align_feature_mart(feature_mart, gmm_labels, gmm_fact)

        rp_gmm = run_regional_profiler(
            feature_mart   = feat_aligned,
            cluster_labels = gmm_labels,
            method         = "gmm",
            output_dir     = PROFILES_DIR,
            visualize      = visualize,
        )
        result["gmm_profiler"] = rp_gmm

    # ── Summary ───────────────────────────────────────────
    _print_separator("SELESAI")
    print(f"  Output → {os.path.abspath(PROFILES_DIR)}")
    files = sorted(os.listdir(PROFILES_DIR))
    for f in files:
        print(f"    {f}")
    print()

    return result


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profiling pipeline — ContentRepresentation & RegionalProfiler"
    )
    parser.add_argument(
        "--method", type=str, default="both",
        choices=["kmeans", "gmm", "both"],
        help="Method clustering yang diprofil (default: both)."
    )
    parser.add_argument(
        "--similarity", type=str, default="cosine",
        choices=["cosine", "euclidean"],
        help="Metrik similarity (default: cosine)."
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Skip pembuatan radar chart."
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = _parse_args()

    profiling_pipeline(
        method            = args.method,
        similarity_metric = args.similarity,
        visualize         = not args.no_viz,
    )
