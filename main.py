import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import urllib.parse
import plotly.express as px
import plotly.graph_objects as go
from src.recommender import HybridRecommender

# Set page configuration
st.set_page_config(
    page_title="Sistem Rekomendasi Kesehatan Masyarakat Indonesia",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom premium styling (Glassmorphism & Slate theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header Banner styling */
    .header-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        border-radius: 16px;
        padding: 40px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        border: 1px solid #312e81;
    }
    .header-title {
        color: #f8fafc;
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 10px;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 16px;
        font-weight: 400;
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Glassmorphic Cards */
    .card-glass {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        color: #f8fafc;
        margin-bottom: 20px;
    }
    
    /* Clickable Grid Cards for Provinces */
    .grid-container {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }
    
    .province-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px;
        text-align: left;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-decoration: none;
        color: #f8fafc !important;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 140px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .province-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 20px -5px rgba(0, 0, 0, 0.3);
        border-color: #38bdf8;
    }
    
    .province-box h4 {
        margin: 0 0 10px 0;
        font-size: 18px;
        font-weight: 600;
        letter-spacing: -0.3px;
    }
    
    .province-box p {
        margin: 0 0 15px 0;
        font-size: 13px;
        color: #94a3b8;
    }
    
    .badge-deficit {
        align-self: flex-start;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* KPI Metric Cards */
    .kpi-container {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        margin-bottom: 25px;
    }
    
    .kpi-card {
        flex: 1;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .kpi-value {
        font-size: 32px;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 5px;
    }
    .kpi-label {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Recommendations styling */
    .rec-card {
        background: #1e1b4b;
        border-left: 6px solid #6366f1;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 15px;
        border-top: 1px solid #312e81;
        border-right: 1px solid #312e81;
        border-bottom: 1px solid #312e81;
    }
    .rec-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .rec-title {
        font-size: 16px;
        font-weight: 600;
        color: #f8fafc;
    }
    .rec-score {
        background: #312e81;
        color: #818cf8;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }
    .rec-desc {
        font-size: 13px;
        color: #cbd5e1;
        line-height: 1.5;
        margin: 0;
    }
    
    /* Back link style */
    .back-btn-container {
        margin-bottom: 20px;
    }
    
    /* Formatted table styling */
    .table-container {
        border: 1px solid #334155;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Progress Bars for SDG */
    .sdg-bar-container {
        margin-bottom: 12px;
    }
    .sdg-bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        margin-bottom: 4px;
    }
    .sdg-bar-outer {
        background-color: #334155;
        border-radius: 10px;
        height: 8px;
        width: 100%;
        overflow: hidden;
    }
    .sdg-bar-inner {
        height: 100%;
        border-radius: 10px;
    }
    
    /* Style grid card wrapper buttons in Tab 2 */
    .grid-card-wrapper div.stButton > button {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        border: 1px solid #1e293b !important;
        border-radius: 12px !important;
        padding: 18px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        text-align: left !important;
        width: 100% !important;
        min-height: 140px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        white-space: pre-wrap !important;
    }
    .grid-card-wrapper div.stButton > button:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 12px 20px -5px rgba(0, 0, 0, 0.3) !important;
        border-color: #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Core Logic & Calculations
# -------------------------------------------------------------

def calculate_deficits(scaled_df):
    """
    Menghitung Indeks Defisit Layanan Kesehatan berdasarkan 4 pilar utama.
    Tidak menggunakan formula severity score yang di-arbitrary.
    Defisit = average dari (1 - supply_index) + disease_burden_index
    """
    df = scaled_df.copy()
    
    # supply indices (higher is better, so deficit = 1 - index)
    df["facility_deficit"] = 1.0 - df["facility_index_scaled"]
    df["workforce_deficit"] = 1.0 - df["workforce_capacity_index_scaled"]
    df["insurance_deficit"] = 1.0 - df["insurance_coverage_index_scaled"]
    
    # disease index (higher is worse, so deficit = index)
    df["disease_deficit"] = df["disease_burden_index_scaled"]
    
    # Deficit Index (Rata-rata dari keempat aspek defisit tersebut)
    df["deficit_index"] = (df["facility_deficit"] + df["workforce_deficit"] + df["insurance_deficit"] + df["disease_deficit"]) / 4.0
    
    # Tentukan tingkat defisit
    def get_tier(score):
        if score >= 0.55:
            return "Defisit Sangat Tinggi"
        elif score >= 0.45:
            return "Defisit Tinggi"
        elif score >= 0.35:
            return "Defisit Sedang"
        else:
            return "Defisit Rendah"
            
    df["deficit_tier"] = df["deficit_index"].apply(get_tier)
    return df

class SDGMapper:
    """
    Inline mapper untuk menghitung capaian target SDG3 berdasarkan feature mart,
    diselaraskan dengan status GMM agar tidak terjadi kontradiksi visual.
    """
    def __init__(self, scaled_df, region_summaries):
        self.scaled_df = scaled_df
        self.region_summaries = region_summaries

    def calculate_sdg_scores(self, province: str) -> dict[str, float]:
        row = self.scaled_df[self.scaled_df["Provinsi"] == province]
        if row.empty:
            return {"3.3": 50.0, "3.8": 50.0, "3.c": 50.0, "3.d": 50.0}
        row = row.iloc[0]
        
        # Load statuses
        prov_summary = self.region_summaries.get(province, {})
        sdg_status_ref = prov_summary.get("sdg_status", {})
        
        status_3_3 = sdg_status_ref.get("3.3", "data_limited")
        status_3_8 = sdg_status_ref.get("3.8", "data_limited")
        status_3_c = sdg_status_ref.get("3.c", "data_limited")
        status_3_d = sdg_status_ref.get("3.b", "data_limited") # fallback to 3.b status
        
        # Base values (scaled indices)
        val_3_3 = 1.0 - row.get("disease_burden_index_scaled", 0.5)
        val_3_8 = (row.get("facility_index_scaled", 0.5) + row.get("insurance_coverage_index_scaled", 0.5)) / 2.0
        val_3_c = row.get("workforce_capacity_index_scaled", 0.5)
        val_3_d = (row.get("facility_index_scaled", 0.5) + row.get("workforce_capacity_index_scaled", 0.5) + (1.0 - row.get("disease_burden_index_scaled", 0.5))) / 3.0
        
        def align_score(status: str, val: float) -> float:
            status_lower = status.lower()
            if "on_track" in status_lower or "on track" in status_lower:
                # Aligned to optimal/good range [75%, 98%]
                return 75.0 + 23.0 * val
            elif "needs_attention" in status_lower or "needs attention" in status_lower or "attention" in status_lower:
                # Aligned to concern range [35%, 68%]
                return 35.0 + 33.0 * val
            else:
                # Aligned to average range [50%, 72%]
                return 50.0 + 22.0 * val
                
        return {
            "3.3": float(np.clip(align_score(status_3_3, val_3_3), 0.0, 100.0)),
            "3.8": float(np.clip(align_score(status_3_8, val_3_8), 0.0, 100.0)),
            "3.c": float(np.clip(align_score(status_3_c, val_3_c), 0.0, 100.0)),
            "3.d": float(np.clip(align_score(status_3_d, val_3_d), 0.0, 100.0))
        }

    def calculate_sdg(self, province: str) -> dict[str, float]:
        return self.calculate_sdg_scores(province)

# -------------------------------------------------------------
# Data Loading (Cached)
# -------------------------------------------------------------

@st.cache_data
def load_data():
    # Load Facts & Clustered Data
    feat_mart = pd.read_csv('data/warehouse/facts/fact_healthcare_feature_mart.csv')
    scaled_df_raw = pd.read_csv('data/warehouse/facts/fact_clustering_gmm.csv')
    
    # Calculate deficit indicators
    scaled_df = calculate_deficits(scaled_df_raw)
    
    # Load profile reference summaries
    with open('data/warehouse/profiles/region_summaries_gmm.json', 'r', encoding='utf-8') as f:
        region_summaries = json.load(f)
        
    # Load Dimensions
    dim_facilities = pd.read_csv('data/warehouse/dimensions/dim_facilities.csv')
    dim_workforce = pd.read_csv('data/warehouse/dimensions/dim_workforce.csv')
    dim_disease = pd.read_csv('data/warehouse/dimensions/dim_disease.csv')
    dim_insurance = pd.read_csv('data/warehouse/dimensions/dim_insurance.csv')
    
    # Load clean dataset to extract raw sub-indicators
    cleaned_df = pd.read_csv('data/cleaned_data.csv')
    
    # Enrich dimension tables with raw metrics from cleaned data
    # 1. Disease
    disease_cols = [c for c in cleaned_df.columns if "Penyakit" in c or "TBC" in c or "HIV" in c or "Kusta" in c or "Malaria" in c or "DBD" in c]
    dim_disease_enriched = pd.merge(dim_disease, cleaned_df[["Provinsi"] + disease_cols], on="Provinsi")
    
    # 2. Insurance
    insurance_cols = [c for c in cleaned_df.columns if "jaminan_kesehatan" in c or "bpjs" in c or "jamkesda" in c or "asuransi" in c]
    dim_insurance_enriched = pd.merge(dim_insurance, cleaned_df[["Provinsi"] + insurance_cols], on="Provinsi")
    
    # 3. Access Barrier
    barrier_cols = [c for c in cleaned_df.columns if "alasan_utama_tidak_berobat_jalan" in c or "tidak_berobat_jalan" in c]
    dim_barrier = cleaned_df[["Provinsi"] + barrier_cols].copy()
    
    return (feat_mart, scaled_df, region_summaries, 
            dim_facilities, dim_workforce, dim_disease_enriched, dim_insurance_enriched, dim_barrier)

# Load data tables
(feat_mart, scaled_df, region_summaries, 
 dim_facilities, dim_workforce, dim_disease, dim_insurance, dim_barrier) = load_data()

# Instantiate HybridRecommender using cached resource
@st.cache_resource
def get_recommender(feat_mart, scaled_df):
    scaled_mart = pd.DataFrame()
    scaled_mart["Provinsi"] = scaled_df["Provinsi"]
    scaled_mart["facility_index"] = scaled_df["facility_index_scaled"]
    scaled_mart["workforce_capacity_index"] = scaled_df["workforce_capacity_index_scaled"]
    scaled_mart["disease_burden_index"] = scaled_df["disease_burden_index_scaled"]
    scaled_mart["insurance_coverage_index"] = scaled_df["insurance_coverage_index_scaled"]
    scaled_mart["province_id"] = scaled_df["province_id"]
    
    # Re-order to match feature mart exactly
    scaled_mart = scaled_mart[feat_mart.columns]
    
    return HybridRecommender(feat_mart, scaled_mart)

recommender = get_recommender(feat_mart, scaled_df)
sdg_mapper = SDGMapper(scaled_df, region_summaries)

# Cluster label mappings
CLUSTER_LABELS = {
    0: "Cluster 0: HIGH insurance coverage / LOW facility",
    1: "Cluster 1: LOW workforce capacity / HIGH facility",
    2: "Cluster 2: HIGH disease burden / HIGH facility",
    3: "Cluster 3: LOW insurance coverage / LOW disease burden"
}

CLUSTER_DESCS = {
    0: "Karakteristik: tinggi cakupan jaminan kesehatan; rendah ketersediaan fasilitas; tinggi kapasitas tenaga kesehatan.",
    1: "Karakteristik: rendah kapasitas tenaga kesehatan; tinggi ketersediaan fasilitas; rendah beban penyakit menular.",
    2: "Karakteristik: tinggi beban penyakit menular; tinggi ketersediaan fasilitas; rendah kapasitas tenaga kesehatan.",
    3: "Karakteristik: rendah cakupan jaminan kesehatan; rendah beban penyakit menular; rendah ketersediaan fasilitas."
}

# -------------------------------------------------------------
# Main Header Layout
# -------------------------------------------------------------

st.markdown("""
<div class="header-container">
    <div class="header-title">Analisis Rekomendasi & Profil Kesehatan Masyarakat</div>
    <div class="header-subtitle">
        Sistem analisis spasial berbasis integrasi data warehouse BPS 2025, pemodelan clustering GMM, 
        dan rekomendasi kebijakan hybrid untuk akselerasi target Sustainable Development Goal 3 (SDG3).
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation via Tabs (Suppressed Sidebars)
tab1, tab2, tab3 = st.tabs([
    "📊 Dashboard Nasional", 
    "🗺️ Dashboard Provinsi & Rekomendasi", 
    "📈 Evaluasi & Metrik Wilayah"
])

# -------------------------------------------------------------
# Tab 1: Dashboard Nasional
# -------------------------------------------------------------
with tab1:
    st.markdown("### Ringkasan Kondisi Kesehatan Nasional")
    
    # Top KPI Metrics
    avg_deficit = scaled_df["deficit_index"].mean()
    total_provinces = len(scaled_df)
    high_deficit_count = len(scaled_df[scaled_df["deficit_index"] >= 0.45])
    
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    with kpi_col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{avg_deficit:.3f}</div>
            <div class="kpi-label">Rata-rata Defisit Layanan Nasional</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_provinces}</div>
            <div class="kpi-label">Total Provinsi Teranalisis</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{high_deficit_count}</div>
            <div class="kpi-label">Provinsi Berstatus Defisit Tinggi</div>
        </div>
        """, unsafe_allow_html=True)
        
    # National Deficit Distribution Chart
    col_chart1, col_chart2 = st.columns([2, 1])
    with col_chart1:
        st.markdown("##### Sebaran Defisit Kesehatan per Provinsi (Urutan Tinggi ke Rendah)")
        sorted_deficit_df = scaled_df.sort_values("deficit_index", ascending=False)
        fig_bar = px.bar(
            sorted_deficit_df,
            x="Provinsi",
            y="deficit_index",
            color="deficit_index",
            color_continuous_scale="RdYlGn_r",
            labels={"deficit_index": "Indeks Defisit Kesehatan", "Provinsi": "Provinsi"},
            template="plotly_dark",
            height=400
        )
        fig_bar.update_layout(margin=dict(l=20, r=20, t=10, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col_chart2:
        st.markdown("##### Distribusi Profil Wilayah (Clustering GMM)")
        cluster_counts = scaled_df["cluster_id"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster ID", "Jumlah Provinsi"]
        cluster_counts["Label"] = cluster_counts["Cluster ID"].map(CLUSTER_LABELS)
        
        fig_pie = px.pie(
            cluster_counts,
            values="Jumlah Provinsi",
            names="Label",
            color="Cluster ID",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            template="plotly_dark",
            hole=0.4
        )
        fig_pie.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    # Cluster Information Section
    st.markdown("### Karakteristik & Prioritas Rekomendasi Cluster")
    cols_cl = st.columns(2)
    
    # Render Cluster Profiles details
    for i, cl_id in enumerate([0, 1, 2, 3]):
        col_idx = i % 2
        with cols_cl[col_idx]:
            st.markdown(f"""
            <div class="card-glass">
                <h4 style="color: #38bdf8; margin-top: 0;">{CLUSTER_LABELS[cl_id]}</h4>
                <p style="font-size: 14px; line-height: 1.6;">{CLUSTER_DESCS[cl_id]}</p>
                <div style="background-color: #0f172a; padding: 10px; border-radius: 6px; font-size: 13px; color: #cbd5e1; border: 1px solid #334155;">
                    <b>Rencana Intervensi Utama:</b> {
                        "Pembangunan & Pemerataan Sarana Fisik (RS & Puskesmas)" if cl_id == 0 else
                        "Redistribusi, Insentif & Penguatan Distribusi Tenaga Medis" if cl_id == 1 else
                        "Akselerasi Pengendalian Penyakit Menular, Distribusi Nakes, & Perluasan JKN" if cl_id == 2 else
                        "Peluasan Cakupan Kepesertaan Jaminan Kesehatan Nasional (JKN/BPJS)"
                    }
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Radar Chart
    if os.path.exists("data/warehouse/profiles/cluster_radar_gmm.png"):
        st.markdown("### Visualisasi Profil Radar Karakteristik Cluster")
        col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
        with col_img2:
            st.image("data/warehouse/profiles/cluster_radar_gmm.png", caption="Visualisasi Radar Perbandingan Cluster GMM", use_container_width=True)

    # Interactive Cluster Characteristics Visualization
    st.markdown("### Analisis Perbandingan Indeks Karakteristik Cluster")
    st.write("Visualisasi interaktif perbandingan nilai rata-rata dari empat pilar indeks kesehatan utama di setiap cluster:")

    features_map = {
        "facility_index_scaled": "Kapasitas Fasilitas",
        "workforce_capacity_index_scaled": "Tenaga Kesehatan",
        "disease_burden_index_scaled": "Beban Penyakit",
        "insurance_coverage_index_scaled": "Jaminan Kesehatan"
    }
    
    cluster_means = scaled_df.groupby("cluster_id")[list(features_map.keys())].mean().reset_index()
    cluster_means["Cluster"] = cluster_means["cluster_id"].map(lambda x: f"Cluster {x}")
    
    # Melt dataframe for plotting
    melted_df = cluster_means.melt(
        id_vars=["Cluster"],
        value_vars=list(features_map.keys()),
        var_name="Indikator",
        value_name="Nilai Rata-rata"
    )
    melted_df["Indikator"] = melted_df["Indikator"].map(features_map)
    
    fig_cluster = px.bar(
        melted_df,
        x="Cluster",
        y="Nilai Rata-rata",
        color="Indikator",
        barmode="group",
        labels={"Nilai Rata-rata": "Nilai Rata-rata Indeks (0-1)", "Cluster": "Cluster"},
        color_discrete_sequence=px.colors.qualitative.Pastel,
        template="plotly_dark",
        height=400
    )
    fig_cluster.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

# -------------------------------------------------------------
# Tab 2: Dashboard Provinsi & Rekomendasi
# -------------------------------------------------------------
with tab2:
    # Synchronize session state and query parameters
    qp = st.query_params.get("selected_province", None)
    if isinstance(qp, list):
        qp = qp[0] if qp else None
        
    if "selected_province" not in st.session_state:
        st.session_state.selected_province = qp
        
    # Sync query params if they differ
    if qp != st.session_state.selected_province:
        st.session_state.selected_province = qp
        
    selected_province = st.session_state.selected_province
    
    if selected_province:
        # -----------------------------------------------------
        # Detail Profile View for Selected Province
        # -----------------------------------------------------
        # Clear selected_province button
        if st.button("← Kembali ke Daftar Provinsi", key="btn_back"):
            st.session_state.selected_province = None
            st.query_params.clear()
            st.rerun()
            
        # Get province data
        prov_scaled = scaled_df[scaled_df["Provinsi"] == selected_province]
        if prov_scaled.empty:
            st.error(f"Data untuk provinsi {selected_province} tidak ditemukan.")
        else:
            row_scaled = prov_scaled.iloc[0]
            deficit_score = row_scaled["deficit_index"]
            deficit_tier = row_scaled["deficit_tier"]
            cluster_id = int(row_scaled["cluster_id"])
            cluster_label = CLUSTER_LABELS[cluster_id]
            
            # Determine color for the deficit
            deficit_color = (
                "#dc2626" if deficit_tier == "Defisit Sangat Tinggi" else
                "#ea580c" if deficit_tier == "Defisit Tinggi" else
                "#d97706" if deficit_tier == "Defisit Sedang" else
                "#16a34a"
            )
            
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.5); border-left: 8px solid {deficit_color}; padding: 25px; border-radius: 12px; margin-bottom: 25px; border-top: 1px solid #334155; border-right: 1px solid #334155; border-bottom: 1px solid #334155;">
                <span class="badge-deficit" style="background-color: {deficit_color}; margin-bottom: 10px; display: inline-block;">{deficit_tier}</span>
                <h2 style="margin: 5px 0 10px 0; color: #f8fafc; font-size: 32px; font-weight: 700;">Profil Provinsi {selected_province}</h2>
                <div style="color: #38bdf8; font-size: 15px; font-weight: 500; margin-bottom: 5px;">
                    {cluster_label}
                </div>
                <div style="color: #94a3b8; font-size: 14px;">
                    Indeks Defisit Layanan Kesehatan: <b>{deficit_score:.4f}</b> (Skala 0-1. Rata-rata dari defisit fasilitas, nakes, asuransi, dan beban penyakit)
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Divide into columns for SDG targets & hybrid recommendations
            col_left, col_right = st.columns([1, 1.2])
            
            with col_left:
                st.markdown("### SDG 3 Target Capaian")
                sdg_scores = sdg_mapper.calculate_sdg(selected_province)
                
                # Fetch status details from profile reference
                prov_summary = region_summaries.get(selected_province, {})
                sdg_status_ref = prov_summary.get("sdg_status", {})
                
                # Render beautiful custom gauges/bars
                for target_code, target_name, target_desc in [
                    ("3.3", "Infectious Disease Control", "Penanggulangan HIV, TBC, Malaria, DBD, Kusta"),
                    ("3.8", "Universal Health Coverage", "Akses fasilitas dan cakupan jaminan BPJS Kesehatan"),
                    ("3.c", "Health Workforce Capacity", "Ketersediaan dan pemerataan rasio tenaga kesehatan"),
                    ("3.d", "Health System Capacity", "Ketahanan sistem kesehatan, fasilitas, dan nakes")
                ]:
                    score = sdg_scores.get(target_code, 50.0)
                    status_lbl = sdg_status_ref.get(target_code, "data_limited").replace("_", " ").upper()
                    
                    status_color = (
                        "#16a34a" if "ON TRACK" in status_lbl else
                        "#ea580c" if "ATTENTION" in status_lbl or "NEEDS" in status_lbl else
                        "#475569"
                    )
                    
                    st.markdown(f"""
                    <div class="card-glass" style="padding: 16px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-weight: 600; font-size: 14px; color: #f8fafc;">Target {target_code} - {target_name}</span>
                            <span style="font-size: 11px; background-color: {status_color}; color: #ffffff; padding: 2px 8px; border-radius: 4px; font-weight: 600;">{status_lbl}</span>
                        </div>
                        <p style="font-size: 12px; color: #94a3b8; margin: 0 0 10px 0;">{target_desc}</p>
                        <div class="sdg-bar-container">
                            <div class="sdg-bar-label">
                                <span style="font-size: 11px; color: #cbd5e1;">Skor Indeks Capaian</span>
                                <span style="font-weight: 600; font-size: 12px; color: #38bdf8;">{score:.1f}%</span>
                            </div>
                            <div class="sdg-bar-outer">
                                <div class="sdg-bar-inner" style="width: {score}%; background-color: #38bdf8;"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            with col_right:
                st.markdown("### Rekomendasi Kebijakan (Engine Hybrid)")
                
                # Run recommender system
                recs = recommender.generate_final_recommendations(selected_province)
                
                if not recs:
                    st.info("Tidak ada rekomendasi khusus yang teridentifikasi.")
                else:
                    st.markdown("<p style='font-size:14px; color:#cbd5e1; margin-bottom: 15px;'>Urutan rekomendasi kebijakan kesehatan hasil integrasi kemiripan fitur spasial dan memori kasus historis daerah lain:</p>", unsafe_allow_html=True)
                    for idx, rec in enumerate(recs):
                        tag = rec["tag"]
                        policy = rec["policy"]
                        score = rec["hybrid_score"]
                        sources = rec.get("sources", [])
                        meta = rec.get("metadata", {})
                        
                        desc = ""
                        details = ""
                        
                        if "content" in meta and meta["content"]:
                            desc = meta["content"].get("description", "")
                            precedents = meta["content"].get("precedents", [])
                            if precedents:
                                details += f"<b>Daerah Kemiripan:</b> {', '.join(precedents[:4])}<br/>"
                        
                        if "case" in meta and meta["case"]:
                            case_lbl = meta["case"].get("case_label", "")
                            notes = meta["case"].get("adaptation_notes", [])
                            if notes:
                                details += f"<b>Catatan Adaptasi Kasus:</b> {', '.join(notes)}<br/>"
                            if not desc:
                                desc = f"Berdasarkan adaptasi kasus: {case_lbl}"
                                
                        source_lbl = ", ".join([s.replace("_", " ") for s in sources]).upper()
                        
                        card_html = (
                            f'<div class="rec-card">'
                            f'<div class="rec-header">'
                            f'<span class="rec-title">{idx+1}. {policy}</span>'
                            f'<span class="rec-score">Skor: {score:.3f}</span>'
                            f'</div>'
                            f'<p class="rec-desc" style="margin-bottom: 8px;">{desc}</p>'
                        )
                        if details:
                            card_html += f'<div style="font-size: 11px; color:#94a3b8; line-height: 1.4; border-top:1px solid #334155; padding-top: 6px; margin-top: 6px;">{details}</div>'
                        card_html += (
                            f'<div style="font-size: 10px; color: #818cf8; font-weight:600; margin-top: 5px; text-transform: uppercase;">'
                            f'Engine: {source_lbl}'
                            f'</div>'
                            f'</div>'
                        )
                        st.markdown(card_html, unsafe_allow_html=True)

            # -----------------------------------------------------
            # Detailed Sub-Tables of Raw Indicators per Dimension
            # -----------------------------------------------------
            st.markdown("### Rincian Indikator Mentah per Dimensi Kesehatan")
            
            # Dimension 1: Fasilitas
            exp_fac = st.expander("🏨 1. Fasilitas Kesehatan & Sarana Prasarana")
            with exp_fac:
                prov_fac = dim_facilities[dim_facilities["Provinsi"] == selected_province]
                if prov_fac.empty:
                    st.write("Data tidak tersedia.")
                else:
                    row_fac = prov_fac.iloc[0]
                    st.markdown(f"**Indeks Kapasitas Fasilitas Kesehatan: `{row_fac['facility_index']:.2f}`**")
                    
                    # Display metrics
                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    m_col1.metric("RS Umum", int(row_fac.get("Jumlah Rumah Sakit Umum", 0)))
                    m_col2.metric("RS Khusus", int(row_fac.get("Jumlah Rumah Sakit Khusus", 0)))
                    m_col3.metric("Puskesmas Rawat Inap", int(row_fac.get("Jumlah Puskesmas Rawat Inap", 0)))
                    m_col4.metric("Puskesmas Non Rawat Inap", int(row_fac.get("Jumlah Puskesmas Non Rawat Inap", 0)))
                    
                    st.dataframe(prov_fac, use_container_width=True, hide_index=True)
            
            # Dimension 2: Tenaga Kesehatan
            exp_wf = st.expander("🧑‍⚕️ 2. Ketersediaan & kecukupan Tenaga Kesehatan")
            with exp_wf:
                prov_wf = dim_workforce[dim_workforce["Provinsi"] == selected_province]
                if prov_wf.empty:
                    st.write("Data tidak tersedia.")
                else:
                    row_wf = prov_wf.iloc[0]
                    st.markdown(f"**Indeks Kapasitas Tenaga Kesehatan: `{row_wf['workforce_capacity_index']:.2f}`**")
                    
                    # Metrics
                    w_col1, w_col2, w_col3, w_col4 = st.columns(4)
                    w_col1.metric("Tenaga Medis", int(row_wf.get("Tenaga Medis", 0)))
                    w_col2.metric("Tenaga Kebidanan", int(row_wf.get("Tenaga Kebidanan", 0)))
                    w_col3.metric("Tenaga Kefarmasian", int(row_wf.get("Tenaga Kefarmasian", 0)))
                    w_col4.metric("Tenaga Gizi", int(row_wf.get("Tenaga Gizi", 0)))
                    
                    st.dataframe(prov_wf, use_container_width=True, hide_index=True)
                    
            # Dimension 3: Penyakit
            exp_dis = st.expander("🦠 3. Beban Penyakit Menular & Epidemiologi")
            with exp_dis:
                prov_dis = dim_disease[dim_disease["Provinsi"] == selected_province]
                if prov_dis.empty:
                    st.write("Data tidak tersedia.")
                else:
                    row_dis = prov_dis.iloc[0]
                    st.markdown(f"**Indeks Beban Penyakit: `{row_dis['disease_burden_index']:.2f}`** (Semakin tinggi semakin parah)")
                    
                    d_col1, d_col2, d_col3 = st.columns(3)
                    d_col1.metric("HIV/AIDS Kasus Baru", int(row_dis.get("Jumlah Kasus Penyakit - HIV/AIDS Kasus Baru", 0)))
                    d_col2.metric("TBC Discovery Rate", f"{row_dis.get('Jumlah Kasus Penyakit - Angka Penemuan TBC', 0)}%")
                    d_col3.metric("Malaria per 1.000 Penduduk", f"{row_dis.get('Jumlah Kasus Penyakit - Angka Kesakitan Malaria per 1.000 Penduduk', 0)}")
                    
                    st.dataframe(prov_dis, use_container_width=True, hide_index=True)
                    
            # Dimension 4: Jaminan Kesehatan
            exp_ins = st.expander("💳 4. Jaminan Kesehatan & Proteksi Finansial")
            with exp_ins:
                prov_ins = dim_insurance[dim_insurance["Provinsi"] == selected_province]
                if prov_ins.empty:
                    st.write("Data tidak tersedia.")
                else:
                    row_ins = prov_ins.iloc[0]
                    st.markdown(f"**Indeks Cakupan Jaminan Finansial Kesehatan: `{row_ins['insurance_coverage_index']:.4f}`**")
                    
                    # Ratios
                    i_col1, i_col2, i_col3 = st.columns(3)
                    i_col1.metric("Rasio BPJS PBI", f"{row_ins.get('penduduk_yang_memiliki_jaminan_kesehatan_menurut_jenis_jaminan_-_bpjs_kesehatan_penerima_bantuan_iuran_(pbi)_ratio', 0)*100:.1f}%")
                    i_col2.metric("Rasio BPJS Non-PBI", f"{row_ins.get('penduduk_yang_memiliki_jaminan_kesehatan_menurut_jenis_jaminan_-_bpjs_kesehatan_non-penerima_bantuan_iuran_(non-pbi)_ratio', 0)*100:.1f}%")
                    i_col3.metric("Rasio Asuransi Swasta", f"{row_ins.get('penduduk_yang_memiliki_jaminan_kesehatan_menurut_jenis_jaminan_-_asuransi_swasta_ratio', 0)*100:.2f}%")
                    
                    st.dataframe(prov_ins, use_container_width=True, hide_index=True)
                    
            # Dimension 5: Hambatan Akses
            exp_bar = st.expander("🚧 5. Hambatan Akses Layanan Kesehatan")
            with exp_bar:
                prov_bar = dim_barrier[dim_barrier["Provinsi"] == selected_province]
                if prov_bar.empty:
                    st.write("Data tidak tersedia.")
                else:
                    row_bar = prov_bar.iloc[0]
                    st.markdown("**Proporsi Alasan Utama Masyarakat Tidak Berobat Jalan meskipun Memiliki Keluhan Kesehatan:**")
                    
                    b_col1, b_col2, b_col3 = st.columns(3)
                    b_col1.metric("Tidak Ada Biaya Berobat", f"{row_bar.get('penduduk_yang_mempunyai_keluhan_kesehatan_selama_sebulan_terakhir_dan_tidak_berobat_jalan_menurut_alasan_utama_tidak_berobat_jalan_-_tidak_punya_biaya_berobat_ratio', 0)*100:.1f}%")
                    b_col2.metric("Tidak Ada Biaya Transport", f"{row_bar.get('penduduk_yang_mempunyai_keluhan_kesehatan_selama_sebulan_terakhir_dan_tidak_berobat_jalan_menurut_alasan_utama_tidak_berobat_jalan_-_tidak_ada_biaya_transport_ratio', 0)*100:.1f}%")
                    b_col3.metric("Mengobati Sendiri (Swamedikasi)", f"{row_bar.get('penduduk_yang_mempunyai_keluhan_kesehatan_selama_sebulan_terakhir_dan_tidak_berobat_jalan_menurut_alasan_utama_tidak_berobat_jalan_-_mengobati_sendiri_ratio', 0)*100:.1f}%")
                    
                    st.dataframe(prov_bar, use_container_width=True, hide_index=True)
                    
    else:
        # -----------------------------------------------------
        # Grid View of 38 Provinces
        # -----------------------------------------------------
        st.markdown("### Dashboard Wilayah (38 Provinsi)")
        st.write("Gunakan filter di bawah untuk mencari atau mengurutkan provinsi berdasarkan tingkat defisit layanan kesehatan.")
        
        # Controls Row
        ctrl_col1, ctrl_col2 = st.columns([2, 1])
        with ctrl_col1:
            search_query = st.text_input("Cari Provinsi:", "").strip()
        with ctrl_col2:
            sort_option = st.selectbox(
                "Urutkan Berdasarkan:",
                [
                    "Alfabetis (Provinsi)",
                    "Defisit Layanan (Tinggi ke Rendah)",
                    "Defisit Layanan (Rendah ke Tinggi)",
                    "Cluster Pengelompokan"
                ]
            )
            
        # Apply Search Filter
        filtered_df = scaled_df.copy()
        if search_query:
            filtered_df = filtered_df[filtered_df["Provinsi"].str.contains(search_query, case=False)]
            
        # Apply Sorting
        if sort_option == "Alfabetis (Provinsi)":
            filtered_df = filtered_df.sort_values("Provinsi")
        elif sort_option == "Defisit Layanan (Tinggi ke Rendah)":
            filtered_df = filtered_df.sort_values("deficit_index", ascending=False)
        elif sort_option == "Defisit Layanan (Rendah ke Tinggi)":
            filtered_df = filtered_df.sort_values("deficit_index", ascending=True)
        elif sort_option == "Cluster Pengelompokan":
            filtered_df = filtered_df.sort_values("cluster_id")
            
        # Grid rendering using HTML CSS or native column buttons
        cols = st.columns(4)
        for i, (idx, row) in enumerate(filtered_df.reset_index(drop=True).iterrows()):
            col = cols[i % 4]
            with col:
                prov_name = row["Provinsi"]
                def_score = row["deficit_index"]
                def_tier = row["deficit_tier"]
                cl_id = int(row["cluster_id"])
                
                # Color based on tier
                color_hex = (
                    "#dc2626" if def_tier == "Defisit Sangat Tinggi" else
                    "#ea580c" if def_tier == "Defisit Tinggi" else
                    "#d97706" if def_tier == "Defisit Sedang" else
                    "#16a34a"
                )
                
                # Emojis for status
                emoji = (
                    "🔴" if def_tier == "Defisit Sangat Tinggi" else
                    "🟠" if def_tier == "Defisit Tinggi" else
                    "🟡" if def_tier == "Defisit Sedang" else
                    "🟢"
                )
                
                clean_name = prov_name.replace(" ", "_").replace("/", "_")
                wrapper_class = f"btn-wrap-{clean_name}"
                
                st.markdown(f"""
                <style>
                    .{wrapper_class} div.stButton > button {{
                        border-left: 6px solid {color_hex} !important;
                    }}
                </style>
                <div class="grid-card-wrapper {wrapper_class}">
                """, unsafe_allow_html=True)
                
                button_label = f"{emoji} {prov_name}\n\nCluster {cl_id}\nDefisit: {def_score:.3f}\n{def_tier.replace('Defisit ', '')}"
                
                if st.button(button_label, key=f"btn_{clean_name}", use_container_width=True):
                    st.session_state.selected_province = prov_name
                    st.query_params["selected_province"] = prov_name
                    st.rerun()
                    
                st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------
# Tab 3: Evaluasi & Metrik Wilayah
# -------------------------------------------------------------
with tab3:
    st.markdown("### Evaluasi Model & Kemiripan Antar Wilayah")
    
    tab_eval1, tab_eval2, tab_eval3 = st.tabs([
        "🔬 Evaluasi Model Clustering",
        "🤝 Matriks Kemiripan (Cosine)",
        "📋 Data Feature Mart Lengkap"
    ])
    
    with tab_eval1:
        st.markdown("##### Metrik Performa Algoritma Clustering")
        st.write("Perbandingan performa evaluasi antara K-Means dan Gaussian Mixture Model (GMM) pada dataset indikator kesehatan Indonesia:")
        
        # Load cluster evaluation metrics
        if os.path.exists("data/warehouse/facts/cluster_evaluation.csv"):
            eval_df = pd.read_csv("data/warehouse/facts/cluster_evaluation.csv")
            st.dataframe(eval_df, use_container_width=True, hide_index=True)
            
            st.markdown("""
            **Analisis Hasil Evaluasi:**
            - **K-Means** menunjukkan nilai *Silhouette Score* (`0.345`) dan *Davies-Bouldin Index* (`0.875`) yang secara teoritis lebih solid untuk partisi keras.
            - **GMM** dipilih untuk profil rekomendasi final karena mampu mengakomodasi *soft-clustering* (probabilitas keanggotaan wilayah pada tiap karakteristik masalah) dengan nilai log-likelihood yang optimal.
            """)
        else:
            st.info("File metrik evaluasi model tidak ditemukan.")
            
    with tab_eval2:
        st.markdown("##### Pencarian Wilayah dengan Kemiripan Tertinggi (Matriks Cosine Similarity)")
        st.write("Kemiripan wilayah dihitung menggunakan Cosine Similarity dari vektor pilar kesehatan. Wilayah dengan kemiripan tinggi cenderung memiliki pola kebutuhan kebijakan intervensi yang sejenis.")
        
        if os.path.exists("data/warehouse/facts/fact_similarity.csv"):
            sim_df = pd.read_csv("data/warehouse/facts/fact_similarity.csv")
            
            # Interactive lookup
            lookup_prov = st.selectbox("Pilih Provinsi untuk Melihat Wilayah Serupa:", sorted(scaled_df["Provinsi"].unique()))
            
            matches = sim_df[(sim_df["province_a"] == lookup_prov) | (sim_df["province_b"] == lookup_prov)].copy()
            
            # Clean dataframe for visualization
            matches["Wilayah Pembanding"] = matches.apply(
                lambda r: r["province_b"] if r["province_a"] == lookup_prov else r["province_a"], 
                axis=1
            )
            matches = matches[["Wilayah Pembanding", "similarity_score"]].sort_values("similarity_score", ascending=False).reset_index(drop=True)
            matches.columns = ["Provinsi Tetangga", "Tingkat Kemiripan (Cosine)"]
            
            st.markdown(f"**Top 10 Provinsi Paling Serupa dengan {lookup_prov}:**")
            st.dataframe(matches.head(10), use_container_width=True, hide_index=True)
            
            # Plot similarity bar chart
            fig_sim = px.bar(
                matches.head(10),
                x="Provinsi Tetangga",
                y="Tingkat Kemiripan (Cosine)",
                range_y=[0.8, 1.0],
                color="Tingkat Kemiripan (Cosine)",
                color_continuous_scale="Blues",
                template="plotly_dark",
                height=350
            )
            fig_sim.update_layout(margin=dict(l=20, r=20, t=10, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_sim, use_container_width=True)
        else:
            st.info("File data kemiripan spasial tidak ditemukan.")
            
    with tab_eval3:
        st.markdown("##### Tabel Data Feature Mart Lengkap")
        st.write("Data gabungan seluruh indeks indikator mentah dari 38 provinsi di Indonesia:")
        
        # Display full feature mart table
        st.dataframe(
            scaled_df[["Provinsi", "facility_index_scaled", "workforce_capacity_index_scaled", "disease_burden_index_scaled", "insurance_coverage_index_scaled", "deficit_index", "deficit_tier"]], 
            use_container_width=True,
            hide_index=True
        )
