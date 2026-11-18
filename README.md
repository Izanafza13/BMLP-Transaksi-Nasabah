# Segmentasi & Klasifikasi Transaksi Nasabah 📊

Proyek akhir kelas **Belajar Machine Learning untuk Pemula** (Dicoding × Coding Camp powered by DBS Foundation, jalur Data Scientist): clustering **K-Means** dan klasifikasi **Decision Tree / Random Forest** pada data transaksi keuangan, dilengkapi aplikasi **Streamlit**.

> ⚠️ **Demo pembelajaran, bukan sistem deteksi penipuan.** Lihat bagian [Keterbatasan](#keterbatasan).

**Demo:** _(isi tautan Streamlit setelah deploy)_

## Alur Proyek
1. **Data:** 2.512 transaksi keuangan (dataset dari materi kelas Dicoding). Setelah menghapus data kosong, duplikat, kolom ID/IP/tanggal, dan outlier (IQR), tersisa **1.945** baris.
2. **Preprocessing:** *label encoding* (kategorikal), *standard scaling* (numerik), dan *binning* usia menjadi `AgeGroup` (Muda, Dewasa, Lansia).
3. **Clustering:** K-Means, k = 2 (silhouette score ≈ 0,572; ukuran cluster 980 dan 965).
4. **Klasifikasi:** hasil cluster (`Target`) dipakai sebagai label untuk Decision Tree dan Random Forest (GridSearchCV; parameter terbaik `n_estimators=50`, `max_depth=None`, `min_samples_split=2`).

## Isi Aplikasi
| Tab | Isi |
|---|---|
| Ringkasan | Alur proyek dan metrik utama |
| Eksplorasi Data | Cuplikan data, distribusi fitur |
| Clustering | Silhouette per k, visualisasi PCA, profil cluster |
| Klasifikasi | Classification report, confusion matrix, fitur terpenting |
| Coba Prediksi | Form input satu transaksi → cluster dan prediksi |
| Keterbatasan | Hal yang perlu diketahui sebelum menafsirkan hasil |

Model **dilatih ulang saat aplikasi dibuka** (datanya kecil), dengan `random_state=42` dan split 80:20 *stratified*, sehingga tidak bergantung pada versi scikit-learn yang dipakai saat menyimpan model di notebook.

## Struktur Direktori
```
bmlp-transaksi-nasabah
├── app.py                          # aplikasi Streamlit
├── pipeline.py                     # preprocessing, clustering, klasifikasi, prediksi
├── data
│   └── data_clustering_inverse.csv # data bersih dengan nilai asli + kolom Target
├── notebooks
│   ├── 01_clustering.ipynb
│   └── 02_klasifikasi.ipynb
├── requirements.txt
└── README.md
```

## Menjalankan Secara Lokal
```
git clone https://github.com/Izanafza13/BMLP-Transaksi-Nasabah.git
cd bmlp-transaksi-nasabah
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
Aplikasi terbuka di `http://localhost:8501`.

## Keterbatasan
- **Cluster ditentukan oleh kota.** `Location` di-*label encoding* menurut urutan abjad lalu di-*scaling*, sehingga mendominasi jarak K-Means. Kota dengan kode 0–21 masuk Cluster 0 dan 22–42 masuk Cluster 1; fitur lain (nominal, usia, durasi, saldo) hampir identik di kedua cluster.
- **Label klasifikasi berasal dari clustering**, bukan label nyata. Akurasi 100% pada Decision Tree dan Random Forest karena itu wajar dan bukan bukti kemampuan prediksi.
- **Tidak dapat dipakai untuk mendeteksi penipuan**, karena proyek ini tidak memakai label penipuan yang sebenarnya.
- **Pengembangan lanjutan:** clustering tanpa `Location` atau dengan *one-hot encoding*, evaluasi dengan metrik lain, dan interpretasi cluster dari pola nominal, saldo, dan durasi.

## Catatan Notebook
Notebook adalah hasil pengerjaan submission Dicoding. Gambar contoh output dari soal sudah dihapus; teks instruksi soal masih ada.

## Teknologi
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Streamlit.
