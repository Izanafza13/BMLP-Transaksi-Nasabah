import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

import pipeline as pl

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Segmentasi Transaksi Nasabah (BMLP)", page_icon="📊", layout="wide")
sns.set_theme(style="whitegrid")
BLUE, ORANGE = "#1b7a9c", "#d95f02"
CLUSTER_COLORS = {0: BLUE, 1: ORANGE}


@st.cache_resource(show_spinner="Menyiapkan data dan melatih model...")
def get_artifacts():
    return pl.build_artifacts()


@st.cache_data(show_spinner="Menghitung silhouette untuk k = 2–10...")
def get_silhouettes(_art):
    return pl.silhouette_by_k(_art.X_scaled)


art = get_artifacts()
df = art.df

st.title("📊 Segmentasi & Klasifikasi Transaksi Nasabah")
st.caption("Proyek akhir kelas *Belajar Machine Learning untuk Pemula* (Dicoding × DBS Foundation) — "
           "clustering K-Means dan klasifikasi Decision Tree / Random Forest.")
st.warning(
    "**Demo pembelajaran, bukan sistem deteksi penipuan.** Dua cluster pada model ini pada dasarnya "
    "hanya dibedakan oleh kota (kode `Location`), dan label klasifikasi berasal dari cluster tersebut, "
    "sehingga akurasi 100% bersifat wajar dan bukan bukti kemampuan prediksi. Lihat tab *Keterbatasan*."
)

tab_ringkas, tab_eda, tab_cluster, tab_klas, tab_coba, tab_batas = st.tabs(
    ["🏠 Ringkasan", "🔎 Eksplorasi Data", "🧩 Clustering", "🌲 Klasifikasi", "🧪 Coba Prediksi", "⚠️ Keterbatasan"]
)

# ------------------------------------------------------------------ Ringkasan
with tab_ringkas:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data setelah dibersihkan", f"{len(df):,}")
    c2.metric("Jumlah fitur", len(pl.FEATURE_ORDER))
    c3.metric("Silhouette (k = 2)", f"{art.silhouette:.3f}")
    c4.metric("Ukuran cluster", " / ".join(str(int((art.labels == k).sum())) for k in (0, 1)))

    st.subheader("Alur proyek")
    st.markdown(
        """
1. **Data:** 2.512 transaksi keuangan (sumber dataset: materi kelas Dicoding). Setelah menghapus data kosong,
   duplikat, kolom ID/IP/tanggal, dan outlier (IQR), tersisa 1.945 baris.
2. **Preprocessing:** *label encoding* untuk fitur kategorikal, *standard scaling* untuk fitur numerik,
   dan *binning* usia menjadi `AgeGroup` (Muda, Dewasa, Lansia).
3. **Clustering:** K-Means dengan k = 2 (dipilih lewat silhouette score) menghasilkan kolom `Target`.
4. **Klasifikasi:** `Target` dipakai sebagai label untuk Decision Tree dan Random Forest
   (hyperparameter tuning dengan GridSearchCV pada notebook).
        """
    )
    st.info("Model dilatih ulang saat aplikasi dibuka (datanya kecil), dengan pengaturan yang sama seperti notebook: "
            "`random_state=42`, split 80:20 *stratified*, dan hyperparameter terbaik Random Forest dari GridSearchCV.")

# ------------------------------------------------------------------ EDA
with tab_eda:
    st.subheader("Cuplikan data (nilai asli)")
    st.dataframe(df.head(20), width="stretch")

    st.subheader("Distribusi fitur numerik")
    cols = st.columns(3)
    for i, col in enumerate(pl.NUM_COLS):
        fig, ax = plt.subplots(figsize=(4.2, 3))
        sns.histplot(df[col], bins=25, color=BLUE, ax=ax)
        ax.set_title(col, loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        sns.despine()
        cols[i % 3].pyplot(fig)
        plt.close(fig)

    st.subheader("Fitur kategorikal")
    pick = st.selectbox("Pilih fitur", [c for c in pl.CAT_COLS if c != "Location"])
    counts = df[pick].value_counts()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(counts.index.astype(str), counts.values, color=BLUE)
    ax.set_ylabel("Jumlah")
    sns.despine()
    st.pyplot(fig)
    plt.close(fig)

# ------------------------------------------------------------------ Clustering
with tab_cluster:
    st.subheader("Memilih jumlah cluster")
    sil = get_silhouettes(art)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(sil.index, sil.values, marker="o", color=BLUE, lw=2)
    ax.scatter([2], [sil.loc[2]], color=ORANGE, zorder=3, s=70, label="k = 2 (dipakai)")
    ax.set_xlabel("Jumlah cluster (k)")
    ax.set_ylabel("Silhouette score")
    ax.legend(frameon=False)
    sns.despine()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Visualisasi cluster (PCA 2 komponen)")
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for k in (0, 1):
        m = art.labels == k
        ax.scatter(art.pca_coords[m, 0], art.pca_coords[m, 1], s=12, alpha=0.6,
                   color=CLUSTER_COLORS[k], label=f"Cluster {k}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False)
    sns.despine()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Profil tiap cluster")
    prof = df.assign(Cluster=art.labels).groupby("Cluster")[pl.NUM_COLS].mean().round(2)
    st.dataframe(prof.T, width="stretch")
    st.caption("Rata-rata fitur numerik hampir sama di kedua cluster.")

    st.subheader("Apa yang membedakan kedua cluster?")
    tbl = pl.cities_by_cluster(art)
    st.dataframe(tbl, width="stretch", height=300)
    st.markdown(
        "Setiap kota masuk ke **satu** cluster saja. Kode encoding kota (urutan abjad) 0–21 menjadi "
        "Cluster 0 dan 22–42 menjadi Cluster 1. Jadi pembeda utamanya adalah **kode lokasi**, "
        "bukan pola transaksi."
    )

# ------------------------------------------------------------------ Klasifikasi
with tab_klas:
    st.subheader("Evaluasi pada data uji (20%)")
    model_name = st.radio("Model", list(art.reports), horizontal=True)
    rep = art.reports[model_name]
    c1, c2 = st.columns([1.2, 1])
    with c1:
        rep_df = pd.DataFrame(rep).T.round(3)
        st.dataframe(rep_df, width="stretch")
        st.metric("Akurasi", f"{rep['accuracy']:.3f}")
    with c2:
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        sns.heatmap(art.confusions[model_name], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Cluster 0", "Cluster 1"], yticklabels=["Cluster 0", "Cluster 1"])
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Fitur terpenting (Random Forest)")
    top = art.rf_importance.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(top.index, top.values, color=BLUE)
    ax.set_xlabel("Importance")
    sns.despine()
    st.pyplot(fig)
    plt.close(fig)
    loc_share = art.rf_importance[art.rf_importance.index.str.startswith("Location_")].sum()
    st.caption(f"Fitur kota (`Location_*`) menyumbang {loc_share:.0%} dari seluruh importance, "
               "karena label klasifikasi berasal dari cluster yang dibedakan oleh kota.")

# ------------------------------------------------------------------ Coba prediksi
with tab_coba:
    st.subheader("Coba prediksi satu transaksi")
    st.caption("Isi data transaksi, lalu lihat cluster K-Means dan prediksi klasifikasi.")
    with st.form("form_prediksi"):
        a, b, c = st.columns(3)
        amount = a.number_input("Nominal transaksi", float(df.TransactionAmount.min()),
                                float(df.TransactionAmount.max()), float(df.TransactionAmount.median()))
        age = b.number_input("Usia nasabah", int(df.CustomerAge.min()), int(df.CustomerAge.max()),
                             int(df.CustomerAge.median()))
        duration = c.number_input("Durasi transaksi (detik)", float(df.TransactionDuration.min()),
                                  float(df.TransactionDuration.max()), float(df.TransactionDuration.median()))
        d, e, f = st.columns(3)
        login = d.number_input("Percobaan login", int(df.LoginAttempts.min()), int(df.LoginAttempts.max()),
                               int(df.LoginAttempts.median()))
        balance = e.number_input("Saldo rekening", float(df.AccountBalance.min()),
                                 float(df.AccountBalance.max()), float(df.AccountBalance.median()))
        ttype = f.selectbox("Tipe transaksi", sorted(df.TransactionType.unique()))
        g, h, i = st.columns(3)
        loc = g.selectbox("Kota", sorted(df.Location.unique()))
        channel = h.selectbox("Kanal", sorted(df.Channel.unique()))
        occ = i.selectbox("Pekerjaan", sorted(df.CustomerOccupation.unique()))
        submitted = st.form_submit_button("Prediksi")

    if submitted:
        res = pl.predict_one(art, {
            "TransactionAmount": amount, "CustomerAge": float(age), "TransactionDuration": duration,
            "LoginAttempts": float(login), "AccountBalance": balance, "TransactionType": ttype,
            "Location": loc, "Channel": channel, "CustomerOccupation": occ,
        })
        r1, r2, r3 = st.columns(3)
        r1.metric("Cluster K-Means", f"Cluster {res['cluster_kmeans']}")
        r2.metric("Random Forest", f"Cluster {res['rf_pred']}",
                  f"keyakinan {res['rf_proba'][res['rf_pred']]:.0%}", delta_color="off")
        r3.metric("Kelompok usia", res["age_group"])
        st.info(f"Kota **{loc}** termasuk ke Cluster {res['cluster_kmeans']}. Karena pembeda utama cluster adalah "
                "kota, mengubah nilai lain biasanya tidak mengubah hasil. Coba ganti kota untuk melihat perbedaannya.")

# ------------------------------------------------------------------ Keterbatasan
with tab_batas:
    st.subheader("Keterbatasan yang perlu diketahui")
    st.markdown(
        """
- **Cluster ditentukan oleh kota.** `Location` di-*label encoding* menurut urutan abjad lalu di-*scaling*;
  nilainya berskala besar dibanding fitur lain sehingga mendominasi jarak K-Means.
  Hasilnya kedua cluster hampir identik pada nominal, usia, durasi, dan saldo.
- **Label klasifikasi berasal dari clustering**, bukan label nyata (misalnya penipuan atau bukan).
  Karena itu Decision Tree dan Random Forest mencapai akurasi 100% dengan mudah.
- **Dataset tidak memuat label penipuan** yang dipakai dalam proyek ini, sehingga aplikasi ini
  **tidak** dapat dipakai untuk mendeteksi transaksi mencurigakan.
- **Langkah perbaikan yang masuk akal:** meng-*cluster* tanpa `Location` (atau memakai *one-hot encoding*),
  mengevaluasi dengan metrik internal lain, lalu menafsirkan cluster dari pola nominal, saldo, dan durasi.
        """
    )
