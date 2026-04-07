import pandas as pd
import numpy as np
import hashlib
import os
import joblib
import networkx as nx
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────────────────────
# LOAD PRE-TRAINED MODEL  ← NEW (trained by ml.py, not re-trained here)
# ──────────────────────────────────────────────────────────────
_MODEL      = None
_FEAT_COLS  = None
_THRESHOLDS = {}

if os.path.exists("model/model.pkl"):
    _MODEL      = joblib.load("model/model.pkl")
    _FEAT_COLS  = joblib.load("model/feature_cols.pkl")
    _THRESHOLDS = joblib.load("model/thresholds.pkl")
    print("✅ Pre-trained model loaded from model/model.pkl")
else:
    print("⚠️  No pre-trained model found — will train fresh on each call (run ml.py first)")

# ──────────────────────────────────────────────────────────────
# UNIFIED DATE PARSER  (fixes Issue 7 — same logic as ml.py)
# ──────────────────────────────────────────────────────────────
DATE_FORMATS = ["%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%b-%Y"]

def parse_date(val):
    if pd.isnull(val):
        return pd.NaT
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(str(val).strip(), fmt)
        except ValueError:
            continue
    try:
        return pd.to_datetime(val, dayfirst=True)
    except Exception:
        return pd.NaT


# ──────────────────────────────────────────────────────────────
# MAIN INFERENCE FUNCTION
# ──────────────────────────────────────────────────────────────
def run_model(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # ── 1. ENSURE ALL REQUIRED COLUMNS EXIST  ← NEW ───────
    # Handles mixed-source DataFrames (PDF + CSV + Excel merged)
    required_defaults = {
        "gst_amount":  0.0,
        "po_number":   np.nan,
        "grn_number":  np.nan,
        "buyer_id":    "UNKNOWN",
        "lender_id":   "UNKNOWN",
        "supplier_id": "UNKNOWN",
        "GSTIN":       "NULL",
    }
    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default

    # ── 2. CLEAN NUMERICS ─────────────────────────────────
    df["amount"]     = pd.to_numeric(df["amount"],     errors="coerce")
    df["gst_amount"] = pd.to_numeric(df["gst_amount"], errors="coerce").fillna(0)
    df = df.dropna(subset=["amount"])

    if df.empty:
        return df

    # ── 3. UNIFIED DATE PARSING ───────────────────────────
    if "date" in df.columns:
        df["date"] = df["date"].apply(parse_date)
    else:
        df["date"] = pd.NaT

    # ── 4. EXACT DUPLICATE  (SHA-256) ─────────────────────
    def generate_hash(row):
        data = str(row["GSTIN"]) + str(row["amount"]) + str(row["date"])
        return hashlib.sha256(data.encode()).hexdigest()

    df["invoice_hash"]  = df.apply(generate_hash, axis=1)
    df["duplicate_flag"] = df.duplicated(subset=["invoice_hash"], keep=False).astype(int)

    # ── 5. NEAR DUPLICATE  (NLP TF-IDF cosine) ────────────
    df["fingerprint"] = (
        df["GSTIN"].astype(str) + " "
        + df["amount"].astype(str) + " "
        + df["date"].astype(str) + " "
        + df["supplier_id"].astype(str)
    )

    if len(df) > 1:
        try:
            vectorizer   = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
            tfidf_matrix = vectorizer.fit_transform(df["fingerprint"])
            sim_matrix   = cosine_similarity(tfidf_matrix)

            near_dup_flags = []
            for i in range(len(df)):
                row_scores    = sim_matrix[i].copy()
                row_scores[i] = 0
                near_dup_flags.append(1 if row_scores.max() >= 0.92 else 0)

            df["near_duplicate"] = near_dup_flags
        except Exception:
            df["near_duplicate"] = 0
    else:
        df["near_duplicate"] = 0

    # ── 6. FEATURE ENGINEERING ────────────────────────────
    df["supplier_invoice_count"] = df.groupby("supplier_id")["invoice_id"].transform("count")
    df["supplier_avg_amount"]    = df.groupby("supplier_id")["amount"].transform("mean")
    df["amount_deviation"]       = abs(df["amount"] - df["supplier_avg_amount"])
    df["lender_count"]           = df.groupby("GSTIN")["lender_id"].transform("nunique")

    # PO / GRN validation
    df["po_grn_match"] = (df["po_number"].notna() & df["grn_number"].notna()).astype(int)
    df["po_missing"]   = df["po_number"].isna().astype(int)
    df["grn_missing"]  = df["grn_number"].isna().astype(int)

    # GST consistency
    df["expected_gst"]      = df["amount"] * 0.18
    df["gst_deviation"]     = abs(df["gst_amount"] - df["expected_gst"]) / (df["expected_gst"] + 1)
    df["gst_mismatch_flag"] = (df["gst_deviation"] > 0.05).astype(int)

    # ── 7. GRAPH FEATURES  ← NEW (matches ml.py features) ─
    G = nx.DiGraph()
    for _, row in df.iterrows():
        s = "SUP_" + str(row["supplier_id"])
        b = "BUY_" + str(row["buyer_id"])
        l = "LEN_" + str(row["lender_id"])
        if pd.notna(row["supplier_id"]) and pd.notna(row["buyer_id"]):
            G.add_edge(s, b)

        if pd.notna(row["buyer_id"]) and pd.notna(row["lender_id"]):
            G.add_edge(b, l)

    # Cycle detection
    try:
        cycles         = list(nx.simple_cycles(G))
        circular_nodes = set(n for c in cycles for n in c)
    except Exception:
        circular_nodes = set()

    df["in_circular_cycle"] = df["supplier_id"].apply(
        lambda x: 1 if ("SUP_" + str(x)) in circular_nodes else 0
    )

    # Degree centrality
    try:
        deg_centrality = nx.degree_centrality(G)
    except Exception:
        deg_centrality = {}

    df["graph_centrality"] = df["supplier_id"].apply(
        lambda x: deg_centrality.get("SUP_" + str(x), 0)
    )

    # Cascade exposure size
    try:
        comp_map = {}
        for comp in nx.connected_components(G.to_undirected()):
            for node in comp:
                comp_map[node] = len(comp)
    except Exception:
        comp_map = {}

    df["cascade_exposure_size"] = df["supplier_id"].apply(
        lambda x: comp_map.get("SUP_" + str(x), 1)
    )

    # ── 8. DYNAMIC THRESHOLDS ─────────────────────────────
    # Use saved thresholds if available, else compute from current data
    high_amount_threshold = _THRESHOLDS.get("high_amount_threshold", df["amount"].quantile(0.90))
    high_freq_threshold   = _THRESHOLDS.get("high_freq_threshold",   df["supplier_invoice_count"].quantile(0.90))

    df["high_amount_flag"]    = (df["amount"]                 > high_amount_threshold).astype(int)
    df["high_frequency_flag"] = (df["supplier_invoice_count"] > high_freq_threshold).astype(int)

    # ── 9. FEATURE MATRIX  ← same columns as ml.py ────────
    FEATURE_COLS = _FEAT_COLS or [
        "amount",
        "supplier_invoice_count",
        "supplier_avg_amount",
        "amount_deviation",
        "lender_count",
        "duplicate_flag",
        "near_duplicate",
        "po_grn_match",
        "po_missing",
        "grn_missing",
        "gst_mismatch_flag",
        "gst_deviation",
        "in_circular_cycle",
        "graph_centrality",
        "cascade_exposure_size",
    ]

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    features = df[FEATURE_COLS].fillna(0)

    # ── 10. ML INFERENCE  ← loads saved model, no re-training
    if _MODEL is not None:
        preds = _MODEL.predict(features)
    else:
        # Fallback: train fresh if model file not found
        fallback_model = IsolationForest(contamination=0.1, random_state=42)
        preds = fallback_model.fit_predict(features)

    df["ml_flag"] = (preds == -1).astype(int)

    # ── 11. RISK SCORE ─────────────────────────────────────
    df["risk_score"] = (
        df["duplicate_flag"]      * 25
        + df["near_duplicate"]    * 20
        + df["ml_flag"]           * 20
        + df["in_circular_cycle"] * 15   # ← added (was missing before)
        + (df["lender_count"] > 1).astype(int) * 10
        + df["gst_mismatch_flag"] * 10
        + df["po_missing"]        * 8
        + df["grn_missing"]       * 7
    ).clip(upper=100)

    df["final_flag"] = (df["risk_score"] >= 50).astype(int)

    # ── 12. EXPLAINABILITY ────────────────────────────────
    def explain(row):
        reasons = []
        if row["duplicate_flag"] == 1:    reasons.append("Exact Duplicate")
        if row["near_duplicate"] == 1:    reasons.append("Near Duplicate (NLP)")
        if row["ml_flag"] == 1:           reasons.append("ML Anomaly")
        if row["in_circular_cycle"] == 1: reasons.append("Circular Trading")
        if row["lender_count"] > 1:       reasons.append("Multi-Lender")
        if row["gst_mismatch_flag"] == 1: reasons.append("GST Mismatch")
        if row["po_missing"] == 1:        reasons.append("PO Missing")
        if row["grn_missing"] == 1:       reasons.append("GRN Missing")
        return ", ".join(reasons)

    df["reason"] = df.apply(explain, axis=1)

    return df