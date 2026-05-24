'''
====================================================================
HYBRID HEALTH RECOMMENDER SYSTEM
Preprocessing Pipeline (Step 0 - 3)

Referensi:
Charu Aggarwal, Recommender Systems: The Textbook, 2016, Springer.

Dataset:
Badan Pusat Statistik (BPS) Indonesia
https://www.bps.go.id/id/statistics-table?subject=522

Pipeline:
0. Load Raw Data
1. Cleaning
2. Feature Engineering
3. Scaling

Author:
NRP 3325600018
====================================================================
'''

import re
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# =========================================================
# STEP 0 — LOAD RAW DATA
# =========================================================

def load_raw_data(file_path: str) -> pd.DataFrame:
    '''
    Load raw CSV data.

    Parameters
    ----------
    file_path : str
        Path to CSV file.

    Returns
    -------
    pd.DataFrame
    '''

    df = pd.read_csv(file_path)

    return df


# =========================================================
# STEP 1 — DATA CLEANING
# =========================================================

def cleaning(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Cleaning Strategy
    -----------------
    1. Replace:
        "NA", "NaN", "N/A", "-" → np.nan

    2. Convert numeric columns safely

    3. Percentage columns:
        - convert percentage to ratio
        - rename:
            "Persentase BPJS"
            → "BPJS_ratio"

    4. Fill missing numeric values using column mean

    Returns
    -------
    Cleaned DataFrame
    '''

    df = df.copy()

    # -----------------------------------------------------
    # Standardize Missing Values
    # -----------------------------------------------------

    df.replace(
        ["NA", "NaN", "N/A", "-", ""],
        np.nan,
        inplace=True
    )

    # -----------------------------------------------------
    # Clean Column Names
    # -----------------------------------------------------

    df.columns = [
        col.strip().replace("\n", " ")
        for col in df.columns
    ]

    # -----------------------------------------------------
    # Convert Numeric Columns
    # -----------------------------------------------------

    for col in df.columns:

        if col.lower() != "provinsi":

            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace(r"[^\d\.\-]", "", regex=True)
            )

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # -----------------------------------------------------
    # Percentage → Ratio
    # -----------------------------------------------------

    rename_dict = {}

    for col in df.columns:

        if "Persentase" in col:

            df[col] = df[col] / 100

            new_col = (
                re.sub(r"Persentase\s*", "", col)
                .strip()
                .replace(" ", "_")
                .lower()
            ) + "_ratio"

            rename_dict[col] = new_col

    df.rename(columns=rename_dict, inplace=True)

    # -----------------------------------------------------
    # Fill Missing Numeric Values
    # -----------------------------------------------------

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    df[numeric_cols] = df[numeric_cols].fillna(
        df[numeric_cols].mean()
    )

    return df


# =========================================================
# STEP 2 — FEATURE ENGINEERING
# =========================================================

def feature_engineering(
    df: pd.DataFrame,
    population_col: str = None
) -> pd.DataFrame:
    '''
    Feature Engineering Strategy
    ----------------------------

    Derived Features:
    1. Healthcare Facility Index
    2. Healthcare Workforce Index
    3. Disease Burden Index
    4. Insurance Coverage Index

    Optional:
    - Per population ratio features

    Parameters
    ----------
    population_col : str
        Column name for population.

    Returns
    -------
    Feature-engineered DataFrame
    '''

    df = df.copy()

    # =====================================================
    # FACILITY INDEX
    # =====================================================

    facility_keywords = [
        "rumah_sakit",
        "puskesmas"
    ]

    facility_cols = [
        col for col in df.columns
        if any(k in col.lower() for k in facility_keywords)
    ]

    if facility_cols:

        df["facility_index"] = (
            df[facility_cols]
            .mean(axis=1)
        )

    # =====================================================
    # WORKFORCE INDEX
    # =====================================================

    workforce_keywords = [
        "tenaga"
    ]

    workforce_cols = [
        col for col in df.columns
        if any(k in col.lower() for k in workforce_keywords)
    ]

    if workforce_cols:

        df["workforce_index"] = (
            df[workforce_cols]
            .mean(axis=1)
        )

    # =====================================================
    # DISEASE BURDEN INDEX
    # =====================================================

    disease_keywords = [
        "tbc",
        "hiv",
        "kusta",
        "malaria",
        "dbd"
    ]

    disease_cols = [
        col for col in df.columns
        if any(k in col.lower() for k in disease_keywords)
    ]

    if disease_cols:

        df["disease_burden_index"] = (
            df[disease_cols]
            .mean(axis=1)
        )

    # =====================================================
    # INSURANCE COVERAGE INDEX
    # =====================================================

    insurance_keywords = [
        "bpjs",
        "jamkesda",
        "asuransi",
        "perusahaan"
    ]

    insurance_cols = [
        col for col in df.columns
        if any(k in col.lower() for k in insurance_keywords)
    ]

    if insurance_cols:

        df["insurance_coverage_index"] = (
            df[insurance_cols]
            .mean(axis=1)
        )

    # =====================================================
    # POPULATION NORMALIZATION (OPTIONAL)
    # =====================================================

    if population_col and population_col in df.columns:

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns

        excluded_cols = [
            population_col
        ]

        target_cols = [
            col for col in numeric_cols
            if col not in excluded_cols
        ]

        for col in target_cols:

            df[f"{col}_per_100k"] = (
                df[col] / df[population_col]
            ) * 100000

    return df


# =========================================================
# STEP 3 — FEATURE SCALING
# =========================================================

def scaling(
    df: pd.DataFrame,
    method: str = "minmax"
) -> pd.DataFrame:
    '''
    Scaling Features

    Methods
    -------
    - minmax
    - standard

    Returns
    -------
    Scaled DataFrame
    '''

    df = df.copy()

    # Preserve Province Column
    province_col = None

    if "Provinsi" in df.columns:

        province_col = df["Provinsi"]

        df = df.drop(columns=["Provinsi"])

    # -----------------------------------------------------
    # Select Scaler
    # -----------------------------------------------------

    if method == "minmax":

        scaler = MinMaxScaler()

    elif method == "standard":

        scaler = StandardScaler()

    else:

        raise ValueError(
            "method must be 'minmax' or 'standard'"
        )

    # -----------------------------------------------------
    # Scaling
    # -----------------------------------------------------

    scaled_array = scaler.fit_transform(df)

    scaled_df = pd.DataFrame(
        scaled_array,
        columns=df.columns
    )

    # Restore Province Column
    if province_col is not None:

        scaled_df.insert(
            0,
            "Provinsi",
            province_col.values
        )

    return scaled_df

# =========================================================
# SAVE OUTPUT CSV
# =========================================================

import os

def save_processed_data(
    original_file_path: str,
    cleaned_df: pd.DataFrame,
    scaled_df: pd.DataFrame,
    engineered_df: pd.DataFrame = None
):
    '''
    Save processed datasets into the same folder
    as the original CSV file.

    Output Files
    ------------
    - cleaned_data.csv
    - engineered_data.csv
    - scaled_data.csv

    Parameters
    ----------
    original_file_path : str
        Original raw CSV path.

    cleaned_df : pd.DataFrame
        Cleaned dataset.

    engineered_df : pd.DataFrame
        Feature engineered dataset.

    scaled_df : pd.DataFrame
        Scaled dataset.
    '''

    # -----------------------------------------------------
    # Get Original Directory
    # -----------------------------------------------------

    output_dir = os.path.dirname(
        os.path.abspath(original_file_path)
    )

    # -----------------------------------------------------
    # Define Output Paths
    # -----------------------------------------------------

    cleaned_path = os.path.join(
        output_dir,
        "cleaned_data.csv"
    )

    scaled_path = os.path.join(
        output_dir,
        "scaled_data.csv"
    )

    engineered_path = os.path.join(
        output_dir,
        "engineered_data.csv"
    )

    # -----------------------------------------------------
    # Save CSV Files
    # -----------------------------------------------------

    cleaned_df.to_csv(
        cleaned_path,
        index=False
    )

    scaled_df.to_csv(
        scaled_path,
        index=False
    )

    if engineered_df is not None:

        engineered_df.to_csv(
            engineered_path,
            index=False
        )

    # -----------------------------------------------------
    # Console Output
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("FILES SAVED SUCCESSFULLY")
    print("=" * 60)

    print(f"Cleaned Data     : {cleaned_path}")

    if engineered_df is not None:
        print(f"Engineered Data  : {engineered_path}")

    print(f"Scaled Data      : {scaled_path}")


# =========================================================
# FULL PIPELINE
# =========================================================

def preprocess_pipeline(
    file_path: str,
    scaling_method: str = "minmax",
    population_col: str = None
):
    '''
    Full preprocessing pipeline.

    Flow:
    -----
    raw data
        ↓
    cleaning
        ↓
    feature engineering
        ↓
    scaling

    Returns
    -------
    raw_df
    cleaned_df
    engineered_df
    scaled_df
    '''

    # Step 0
    raw_df = load_raw_data(file_path)

    # Step 1
    cleaned_df = cleaning(raw_df)

    # Step 2
    engineered_df = feature_engineering(
        cleaned_df,
        population_col=population_col
    )

    # Step 3
    scaled_df = scaling(
        engineered_df,
        method=scaling_method
    )

    # SAVE OUTPUT FILES
    save_processed_data(
        original_file_path=FILE_PATH,
        cleaned_df=cleaned_df,
        engineered_df=engineered_df,
        scaled_df=scaled_df
    )


    return (
        raw_df,
        cleaned_df,
        engineered_df,
        scaled_df
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    FILE_PATH = "data/Data Kesehatan Masyarakat (Merged).csv"

    (
        raw_data,
        cleaned_data,
        engineered_data,
        scaled_data
    ) = preprocess_pipeline(
        file_path=FILE_PATH,
        scaling_method="minmax",
        population_col=None
    )

    print("=" * 60)
    print("RAW DATA")
    print("=" * 60)
    print(raw_data.head())

    print("\n" + "=" * 60)
    print("CLEANED DATA")
    print("=" * 60)
    print(cleaned_data.head())

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERED DATA")
    print("=" * 60)
    print(engineered_data.head())

    print("\n" + "=" * 60)
    print("SCALED DATA")
    print("=" * 60)
    print(scaled_data.head())