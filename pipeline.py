"""Logika pipeline: preprocessing, clustering, klasifikasi, dan prediksi.

Preprocessing direkonstruksi dari data_clustering_inverse.csv (nilai asli),
sehingga hasilnya sama dengan yang dibuat di notebook Clustering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path(__file__).parent / "data" / "data_clustering_inverse.csv"

NUM_COLS = ["TransactionAmount", "CustomerAge", "TransactionDuration", "LoginAttempts", "AccountBalance"]
CAT_COLS = ["TransactionType", "Location", "Channel", "CustomerOccupation", "AgeGroup"]
FEATURE_ORDER = [
    "TransactionAmount", "TransactionType", "Location", "Channel", "CustomerAge",
    "CustomerOccupation", "TransactionDuration", "LoginAttempts", "AccountBalance", "AgeGroup",
]
AGE_LABELS = ["Muda", "Dewasa", "Lansia"]
# Hyperparameter terbaik dari GridSearchCV di notebook Klasifikasi
RF_PARAMS = {"n_estimators": 50, "max_depth": None, "min_samples_split": 2, "random_state": 42}


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def age_group_edges(df: pd.DataFrame) -> list[float]:
    """Batas usia untuk AgeGroup (qcut 3 kelompok pada data)."""
    _, edges = pd.qcut(df["CustomerAge"], q=3, retbins=True, labels=AGE_LABELS)
    return list(edges)


def assign_age_group(age: float, edges: list[float]) -> str:
    if age <= edges[1]:
        return AGE_LABELS[0]
    if age <= edges[2]:
        return AGE_LABELS[1]
    return AGE_LABELS[2]


@dataclass
class Artifacts:
    df: pd.DataFrame
    encoders: dict
    scaler: StandardScaler
    X_scaled: pd.DataFrame
    kmeans: KMeans
    flip: bool
    labels: np.ndarray
    silhouette: float
    pca_coords: np.ndarray
    age_edges: list
    X_onehot: pd.DataFrame
    y: pd.Series
    dt: DecisionTreeClassifier
    rf: RandomForestClassifier
    X_test: pd.DataFrame
    y_test: pd.Series
    reports: dict = field(default_factory=dict)
    confusions: dict = field(default_factory=dict)
    rf_importance: pd.Series | None = None


def build_artifacts(df: pd.DataFrame | None = None) -> Artifacts:
    df = load_data() if df is None else df.copy()

    # ---- Preprocessing (label encoding + standard scaling)
    encoders = {c: LabelEncoder().fit(df[c]) for c in CAT_COLS}
    scaler = StandardScaler().fit(df[NUM_COLS])

    scaled = pd.DataFrame(scaler.transform(df[NUM_COLS]), columns=NUM_COLS, index=df.index)
    encoded = pd.DataFrame({c: encoders[c].transform(df[c]) for c in CAT_COLS}, index=df.index)
    X_scaled = pd.concat([scaled, encoded], axis=1)[FEATURE_ORDER]

    # ---- Clustering (K-Means, k = 2)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_scaled)
    labels = kmeans.labels_.copy()
    # Samakan penomoran cluster dengan kolom Target di data (0/1 bisa tertukar antar versi)
    flip = bool((labels == df["Target"].to_numpy()).mean() < 0.5)
    if flip:
        labels = 1 - labels
    sil = float(silhouette_score(X_scaled, labels))
    pca_coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    # ---- Klasifikasi (one-hot encoding pada data asli, seperti notebook Klasifikasi)
    feats = df.drop(columns="Target")
    X_onehot = pd.get_dummies(feats, columns=CAT_COLS, drop_first=True)
    y = df["Target"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_onehot, y, test_size=0.2, random_state=42, stratify=y
    )
    dt = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)
    rf = RandomForestClassifier(**RF_PARAMS).fit(X_tr, y_tr)

    reports, confusions = {}, {}
    for name, model in {"Decision Tree": dt, "Random Forest (tuned)": rf}.items():
        pred = model.predict(X_te)
        reports[name] = classification_report(y_te, pred, output_dict=True, zero_division=0)
        confusions[name] = confusion_matrix(y_te, pred)

    importance = pd.Series(rf.feature_importances_, index=X_onehot.columns).sort_values(ascending=False)

    return Artifacts(
        df=df, encoders=encoders, scaler=scaler, X_scaled=X_scaled, kmeans=kmeans, flip=flip,
        labels=labels, silhouette=sil, pca_coords=pca_coords, age_edges=age_group_edges(df),
        X_onehot=X_onehot, y=y, dt=dt, rf=rf, X_test=X_te, y_test=y_te,
        reports=reports, confusions=confusions, rf_importance=importance,
    )


def silhouette_by_k(X_scaled: pd.DataFrame, k_range=range(2, 11)) -> pd.Series:
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
        scores[k] = silhouette_score(X_scaled, km.labels_)
    return pd.Series(scores, name="silhouette")


def predict_one(art: Artifacts, row: dict) -> dict:
    """Prediksi satu transaksi baru. `row` berisi 9 fitur mentah (tanpa AgeGroup)."""
    row = dict(row)
    row["AgeGroup"] = assign_age_group(row["CustomerAge"], art.age_edges)
    new = pd.DataFrame([row])[FEATURE_ORDER]

    # K-Means: label encoding + scaling seperti data latih
    num = pd.DataFrame(art.scaler.transform(new[NUM_COLS]), columns=NUM_COLS)
    cat = pd.DataFrame({c: art.encoders[c].transform(new[c]) for c in CAT_COLS})
    x = pd.concat([num, cat], axis=1)[FEATURE_ORDER]
    cluster = int(art.kmeans.predict(x)[0])
    if art.flip:
        cluster = 1 - cluster

    # Random Forest & Decision Tree: one-hot disejajarkan dengan kolom latih
    feats = art.df.drop(columns="Target")
    combined = pd.get_dummies(pd.concat([feats, new], ignore_index=True), columns=CAT_COLS, drop_first=True)
    x_oh = combined.iloc[[-1]].reindex(columns=art.X_onehot.columns, fill_value=0)
    proba = art.rf.predict_proba(x_oh)[0]
    return {
        "age_group": row["AgeGroup"],
        "cluster_kmeans": cluster,
        "rf_pred": int(art.rf.predict(x_oh)[0]),
        "rf_proba": {int(c): float(p) for c, p in zip(art.rf.classes_, proba)},
        "dt_pred": int(art.dt.predict(x_oh)[0]),
    }


def cities_by_cluster(art: Artifacts) -> pd.DataFrame:
    """Tabel kota per cluster (menunjukkan cluster dipisahkan oleh kode lokasi)."""
    t = pd.crosstab(art.df["Location"], art.labels)
    t.columns = [f"Cluster {c}" for c in t.columns]
    t["Kode encoding"] = [int(art.encoders["Location"].transform([c])[0]) for c in t.index]
    return t.sort_values("Kode encoding")
