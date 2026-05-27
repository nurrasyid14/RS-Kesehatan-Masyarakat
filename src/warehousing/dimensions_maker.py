'''
src/warehousing/dimensions_maker.py
====================================
DimensionMaker — membangun 5 dimension tables dari engineered DataFrame.

Setiap dimensi merepresentasikan satu aspek profil kesehatan provinsi:

    dim_facilities          : ketersediaan fasilitas kesehatan
    dim_workforce           : kapasitas tenaga kesehatan
    dim_disease             : beban penyakit menular
    dim_insurance           : cakupan jaminan kesehatan
    dim_healthcare_barrier  : hambatan akses layanan

Setiap tabel memiliki:
    - province_id   : surrogate key (integer, urutan abjad)
    - provinsi      : natural key
    - kolom-kolom dimensi dari engineered DataFrame
    - kolom composite index (hasil feature_engineer)

Kolom dikutip langsung dari hasil structural_cleaner → imputer → engineer,
sehingga nama kolom harus cocok dengan output pipeline prep.
'''

import pandas as pd
import numpy as np

PROVINCE_COL = "Provinsi"


# ---------------------------------------------------------
# Peta kolom per dimensi
# ---------------------------------------------------------
# Setiap entry: list kolom yang DICARI di DataFrame.
# Kolom yang tidak ditemukan di-skip dengan peringatan,
# bukan error — supaya tidak pecah jika kolom source berubah nama.

FACILITIES_COLS = [
    "Jumlah Rumah Sakit Umum",
    "Jumlah Rumah Sakit Khusus",
    "Jumlah Puskesmas Rawat Inap",
    "Jumlah Puskesmas Non Rawat Inap",
    "facility_index",
]

WORKFORCE_COLS = [
    "Tenaga Medis",
    "Jumlah Tenaga Medis",
    "Tenaga Kebidanan",
    "Tenaga Kefarmasian",
    "Tenaga Kesehatan Masyarakat",
    "Tenaga Kesehatan Lingkungan",
    "Tenaga Gizi",
    "Jumlah Tenaga Kesehatan Psikologi Klinis",
    "Jumlah Tenaga Keterapian Fisik",
    "Jumlah Tenaga Keteknisan Medis",
    "Jumlah Tenaga Teknik Biomedika",
    "Jumlah Tenaga Kesehatan Tradisional",
    "workforce_capacity_index",
]

DISEASE_COLS = [
    "Angka Penemuan TBC",
    "Keberhasilan Pengobatan TBC",
    "HIV/AIDS Kasus Baru",
    "Kasus Baru Kusta per 100.000",
    "Malaria per 1.000",
    "DBD per 100.000",
    "disease_burden_index",
    "treatment_effectiveness_index",
]

INSURANCE_COLS = [
    "BPJS PBI",
    "BPJS Non-PBI",
    "Jamkesda",
    "Asuransi Swasta",
    "Perusahaan/Kantor",
    "insurance_coverage_index",
]

BARRIER_COLS = [
    "Tidak Punya Biaya Berobat",
    "Tidak Ada Biaya Transport",
    "Tidak Ada Sarana Transportasi",
    "Waktu Tunggu Lama",
    "Mengobati Sendiri",
    "Tidak Ada Pendamping",
    "Merasa Tidak Perlu",
    "Lainnya",
    "Jumlah",
    "accessibility_barrier_index",
]


# ---------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------

def _resolve_cols(df: pd.DataFrame, wanted: list[str]) -> list[str]:
    '''
    Kembalikan subset wanted yang benar-benar ada di df.
    Cari secara case-insensitive sebagai fallback.
    '''
    df_cols_lower = {c.lower(): c for c in df.columns}
    resolved = []

    for col in wanted:
        if col in df.columns:
            resolved.append(col)
        elif col.lower() in df_cols_lower:
            resolved.append(df_cols_lower[col.lower()])

    return resolved


def _make_dim(
    df:        pd.DataFrame,
    dim_cols:  list[str],
    dim_name:  str,
) -> pd.DataFrame:
    '''
    Buat satu dimension table.

    Kolom output:
        province_id | provinsi | <dim_cols>
    '''
    resolved = _resolve_cols(df, dim_cols)

    missing = set(dim_cols) - set(resolved)
    if missing:
        print(
            f"  [DimensionMaker] {dim_name}: "
            f"{len(missing)} kolom tidak ditemukan, di-skip → "
            + ", ".join(sorted(missing))
        )

    subset = df[[PROVINCE_COL] + resolved].copy()

    # Surrogate key
    subset = subset.sort_values(PROVINCE_COL).reset_index(drop=True)
    subset.insert(0, "province_id", subset.index + 1)

    return subset


# ---------------------------------------------------------
# DimensionMaker
# ---------------------------------------------------------

class DimensionMaker:
    '''
    Membangun 5 dimension tables dari engineered DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output dari feature_engineer.engineer() atau imputer.impute().
        Provinsi harus ada sebagai kolom.

    Penggunaan
    ----------
    maker = DimensionMaker(engineered_df)
    dims  = maker.make_all_dimensions()

    dims["facilities"]  → dim_facilities DataFrame
    dims["workforce"]   → dim_workforce DataFrame
    ...
    '''

    def __init__(self, df: pd.DataFrame) -> None:

        if PROVINCE_COL not in df.columns:
            raise KeyError(
                f"Kolom '{PROVINCE_COL}' tidak ditemukan. "
                f"Pastikan DataFrame berasal dari prep pipeline."
            )

        self._df = df.copy()

    # --------------------------------------------------

    def make_facilities_dimension(self) -> pd.DataFrame:
        '''dim_facilities — ketersediaan fasilitas kesehatan.'''
        return _make_dim(self._df, FACILITIES_COLS, "dim_facilities")

    def make_workforce_dimension(self) -> pd.DataFrame:
        '''dim_workforce — kapasitas tenaga kesehatan.'''
        return _make_dim(self._df, WORKFORCE_COLS, "dim_workforce")

    def make_disease_dimension(self) -> pd.DataFrame:
        '''dim_disease — beban penyakit menular.'''
        return _make_dim(self._df, DISEASE_COLS, "dim_disease")

    def make_insurance_dimension(self) -> pd.DataFrame:
        '''dim_insurance — cakupan jaminan kesehatan.'''
        return _make_dim(self._df, INSURANCE_COLS, "dim_insurance")

    def make_barrier_dimension(self) -> pd.DataFrame:
        '''dim_healthcare_barrier — hambatan akses layanan.'''
        return _make_dim(self._df, BARRIER_COLS, "dim_healthcare_barrier")

    # --------------------------------------------------

    def make_all_dimensions(self) -> dict[str, pd.DataFrame]:
        '''
        Bangun semua dimensi sekaligus.

        Returns
        -------
        dict dengan key: "facilities", "workforce", "disease",
                         "insurance", "barrier"
        '''
        print("\n[DimensionMaker] Building dimension tables...")

        dims = {
            "facilities": self.make_facilities_dimension(),
            "workforce":  self.make_workforce_dimension(),
            "disease":    self.make_disease_dimension(),
            "insurance":  self.make_insurance_dimension(),
            "barrier":    self.make_barrier_dimension(),
        }

        for name, dim_df in dims.items():
            print(
                f"  dim_{name:20s} → "
                f"{len(dim_df)} baris, "
                f"{len(dim_df.columns)} kolom"
            )

        return dims