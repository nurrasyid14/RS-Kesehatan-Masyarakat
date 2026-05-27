'''
src/prep/merger.py
==================
Step 1b — Penggabungan 5 sumber data BPS menjadi satu DataFrame.

Digunakan ketika pipeline memulai dari 5 file Excel terpisah
(bukan dari Merged CSV yang sudah digabung).

Jika sudah menggunakan "Data Kesehatan Masyarakat (Merged).csv",
modul ini bisa dilewati — structural_cleaner langsung menerima
DataFrame gabungan tersebut.

Sumber data
-----------
1. Fasilitas   : Jumlah RS dan Puskesmas
2. Tenaga      : Jumlah tenaga kesehatan
3. Penyakit    : Kasus penyakit per provinsi
4. Jaminan     : Persentase kepesertaan jaminan kesehatan
5. Hambatan    : Alasan tidak berobat jalan
'''

import pandas as pd

PROVINCE_COL = "Provinsi"


# ---------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------

def _normalize_province_name(series: pd.Series) -> pd.Series:
    '''
    Normalisasi nama provinsi untuk konsistensi join.
    - Strip whitespace
    - Title case
    '''
    return series.str.strip().str.title()


def _load_and_tag(
    file_path: str,
    source_tag: str,
    sheet_name: int | str = 0
) -> pd.DataFrame:
    '''
    Load satu file Excel, normalisasi kolom Provinsi.

    Parameters
    ----------
    file_path : str
        Path ke file xlsx.
    source_tag : str
        Label sumber data (untuk logging/debugging).
    sheet_name : int | str
        Sheet index atau nama.
    '''

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception as e:
        raise IOError(
            f"Gagal membaca {source_tag} dari {file_path}: {e}"
        )

    if PROVINCE_COL not in df.columns:
        raise KeyError(
            f"Kolom '{PROVINCE_COL}' tidak ditemukan di {source_tag}. "
            f"Kolom yang tersedia: {list(df.columns)}"
        )

    df[PROVINCE_COL] = _normalize_province_name(df[PROVINCE_COL])

    return df


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------

def merge_sources(
    facilities_path:  str,
    workforce_path:   str,
    disease_path:     str,
    insurance_path:   str,
    barrier_path:     str,
    how:              str = "outer"
) -> pd.DataFrame:
    '''
    Gabungkan 5 file BPS menjadi satu DataFrame berbasis Provinsi.

    Parameters
    ----------
    facilities_path : str
        Path xlsx: Jumlah RS & Puskesmas
    workforce_path : str
        Path xlsx: Jumlah Tenaga Kesehatan
    disease_path : str
        Path xlsx: Kasus Penyakit
    insurance_path : str
        Path xlsx: Kepesertaan Jaminan Kesehatan
    barrier_path : str
        Path xlsx: Alasan Tidak Berobat Jalan
    how : str
        Tipe merge pandas ("outer" | "inner" | "left").
        Default "outer" untuk mempertahankan semua provinsi.

    Returns
    -------
    pd.DataFrame
        DataFrame gabungan dengan Provinsi sebagai kunci.
    '''

    sources = {
        "fasilitas":  facilities_path,
        "tenaga":     workforce_path,
        "penyakit":   disease_path,
        "jaminan":    insurance_path,
        "hambatan":   barrier_path,
    }

    frames = {
        tag: _load_and_tag(path, tag)
        for tag, path in sources.items()
    }

    merged = frames["fasilitas"]

    for tag in ["tenaga", "penyakit", "jaminan", "hambatan"]:

        merged = merged.merge(
            frames[tag],
            on=PROVINCE_COL,
            how=how,
            suffixes=("", f"_{tag}")
        )

    # Hapus kolom duplikat yang muncul karena suffix
    dup_cols = [
        col for col in merged.columns
        if col.endswith(("_tenaga", "_penyakit", "_jaminan", "_hambatan"))
        and col.replace("_tenaga", "").replace("_penyakit", "")
              .replace("_jaminan", "").replace("_hambatan", "")
        in merged.columns
    ]
    merged.drop(columns=dup_cols, inplace=True)

    return merged
