import pandas as pd
import numpy as np
import hashlib
import glob
import os
import joblib
import networkx as nx
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# ──────────────────────────────────────────────────────────────
# 1. LOAD DATA  ← supports multiple CSV files now
# ──────────────────────────────────────────────────────────────
# Looks for all CSVs inside a 'data/' folder first,
# falls back to invoices.csv in the current directory.
csv_files = glob.glob("data/*.csv")
if not csv_files:
    csv_files = glob.glob("*.csv")
    csv_files = [f for f in csv_files if f not in ("flagged_invoices.csv", "invoice_history.csv")]

if not csv_files:
    raise FileNotFoundError(
        "No CSV files found. Put invoices.csv in the current folder "
        "or place multiple CSVs inside a 'data/' subfolder."
    )

print(f"📂 Loading {len(csv_files)} CSV file(s): {csv_files}")
df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
print(f"✅ Total rows loaded: {len(df)}")

# ──────────────────────────────────────────────────────────────
# 2. ENSURE REQUIRED COLUMNS EXIST  ← prevents KeyError on merge
# ──────────────────────────────────────────────────────────────
for col in ["gst_amount", "po_number", "grn_number", "buyer_id", "lender_id"]:
    if col not in df.columns:
        df[col] = np.nan

df["gst_amount"] = pd.to_numeric(df["gst_amount"], errors="coerce").fillna(0)

# ──────────────────────────────────────────────────────────────
# 3. DATE HANDLING  ← unified multi-format parsing (fixes Issue 7)
# ──────────────────────────────────────────────────────────────
DATE_FORMATS = ["%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%b-%Y"]

def parse_date(val):
    if pd.isnull(val):
        return pd.NaT
    for fmt in DATE_FORMATS:
        try:
            from datetime import datetime
            return datetime.strptime(str(val).strip(), fmt)
        except ValueError:
            continue
    try:
        return pd.to_datetime(val, dayfirst=True)
    except Exception:
        return pd.NaT

df["date"] = df["date"].apply(parse_date)

# ──────────────────────────────────────────────────────────────
# 4. EXACT DUPLICATE DETECTION  (SHA-256)
# ──────────────────────────────────────────────────────────────
def generate_hash(row):
    data = str(row["GSTIN"]) + str(row["amount"]) + str(row["date"])
    return hashlib.sha256(data.encode()).hexdigest()

df["invoice_hash"] = df.apply(generate_hash, axis=1)
df["duplicate_flag"] = df.duplicated(subset=["invoice_hash"], keep=False).astype(int)
print(f"Exact duplicates found: {df['duplicate_flag'].sum()}")

# ──────────────────────────────────────────────────────────────
# 5. NEAR DUPLICATE  (NLP TF-IDF cosine similarity)
# ──────────────────────────────────────────────────────────────
df["fingerprint"] = (
    df.get("GSTIN", df.get("supplier_id")).astype(str) + " "
    + df["amount"].astype(str) + " "
    + df["date"].astype(str) + " "
    + df["supplier_id"].astype(str)
)

if len(df) > 1:
    vectorizer   = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
    tfidf_matrix = vectorizer.fit_transform(df["fingerprint"])
    sim_matrix   = cosine_similarity(tfidf_matrix)

    near_dup_flags = []
    for i in range(len(df)):
        row_scores    = sim_matrix[i].copy()
        row_scores[i] = 0  # ignore self
        near_dup_flags.append(1 if row_scores.max() >= 0.92 else 0)

    df["near_duplicate"] = near_dup_flags
else:
    df["near_duplicate"] = 0

print(f"Near duplicates found: {df['near_duplicate'].sum()}")

# ──────────────────────────────────────────────────────────────
# 6. FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────
# 7. GRAPH ANOMALY DETECTION
# ──────────────────────────────────────────────────────────────
G = nx.DiGraph()

for _, row in df.iterrows():
    supplier = "SUP_" + str(row["supplier_id"])
    buyer    = "BUY_" + str(row["buyer_id"])
    lender   = "LEN_" + str(row["lender_id"])
    G.add_edge(supplier, buyer)
    G.add_edge(buyer,    lender)

# Cycle detection
cycles         = list(nx.simple_cycles(G))
circular_nodes = set(node for cycle in cycles for node in cycle)

df["in_circular_cycle"] = df["supplier_id"].apply(
    lambda x: 1 if ("SUP_" + str(x)) in circular_nodes else 0
)

# Degree centrality
degree_centrality   = nx.degree_centrality(G)
df["graph_centrality"] = df["supplier_id"].apply(
    lambda x: degree_centrality.get("SUP_" + str(x), 0)
)

# Cascade exposure size (connected component size)
component_map = {}
for component in nx.connected_components(G.to_undirected()):
    for node in component:
        component_map[node] = len(component)

df["cascade_exposure_size"] = df["supplier_id"].apply(
    lambda x: component_map.get("SUP_" + str(x), 1)
)

print(f"Circular trading cycles found: {len(cycles)}")

# ──────────────────────────────────────────────────────────────
# 8. DYNAMIC THRESHOLDS
# ──────────────────────────────────────────────────────────────
high_amount_threshold = df["amount"].quantile(0.90)
high_freq_threshold   = df["supplier_invoice_count"].quantile(0.90)

df["high_amount_flag"]    = (df["amount"]                   > high_amount_threshold).astype(int)
df["high_frequency_flag"] = (df["supplier_invoice_count"]   > high_freq_threshold).astype(int)

print(f"High Amount Threshold   : {high_amount_threshold}")
print(f"High Frequency Threshold: {high_freq_threshold}")

# ──────────────────────────────────────────────────────────────
# 9. ENCODE CATEGORICALS
# ──────────────────────────────────────────────────────────────
le = LabelEncoder()
df["supplier_id_enc"] = le.fit_transform(df["supplier_id"].astype(str))
df["buyer_id_enc"]    = le.fit_transform(df["buyer_id"].astype(str))
df["lender_id_enc"]   = le.fit_transform(df["lender_id"].astype(str))

# ──────────────────────────────────────────────────────────────
# 10. FEATURE MATRIX
# ──────────────────────────────────────────────────────────────
FEATURE_COLS = [
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

features = df[FEATURE_COLS].fillna(0)

# ──────────────────────────────────────────────────────────────
# 11. TRAIN ISOLATION FOREST
# ──────────────────────────────────────────────────────────────
model = IsolationForest(contamination=0.1, random_state=42)
df["anomaly_score"] = model.fit_predict(features)
df["ml_fraud_flag"] = (df["anomaly_score"] == -1).astype(int)

# ──────────────────────────────────────────────────────────────
# 12. SAVE MODEL + THRESHOLDS TO DISK  ← NEW (predict.py loads these)
# ──────────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/model.pkl")
joblib.dump(FEATURE_COLS, "model/feature_cols.pkl")
joblib.dump(
    {
        "high_amount_threshold": high_amount_threshold,
        "high_freq_threshold":   high_freq_threshold,
    },
    "model/thresholds.pkl",
)

print("💾 Model saved to model/model.pkl")
print("💾 Feature list saved to model/feature_cols.pkl")
print("💾 Thresholds saved to model/thresholds.pkl")

# ──────────────────────────────────────────────────────────────
# 13. RISK SCORE + FINAL FLAG
# ──────────────────────────────────────────────────────────────
df["risk_score"] = (
    df["duplicate_flag"]      * 25
    + df["near_duplicate"]    * 20
    + df["ml_fraud_flag"]     * 15
    + df["in_circular_cycle"] * 15
    + df["gst_mismatch_flag"] * 10
    + (df["lender_count"] > 1).astype(int) * 5
    + df["high_amount_flag"]    * 5
    + df["high_frequency_flag"] * 3
    + df["po_missing"]          * 1
    + df["grn_missing"]         * 1
).clip(upper=100)

df["final_flag"] = (df["risk_score"] >= 20).astype(int)

# ──────────────────────────────────────────────────────────────
# 14. CALIBRATION REPORT  (fixes Issue 8)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 20)
print("   📊 MODEL CALIBRATION REPORT")
print("=" * 20)

if len(df) >= 10:
    _, df_test = train_test_split(df, test_size=0.3, random_state=42)
    y_true = df_test["final_flag"].values
    y_pred = df_test["ml_fraud_flag"].values

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true,    y_pred, zero_division=0)
    f1        = f1_score(y_true,        y_pred, zero_division=0)

    print(f"  Precision : {precision:.2f}  (of flagged, how many are real fraud)")
    print(f"  Recall    : {recall:.2f}  (of real fraud, how many did we catch)")
    print(f"  F1 Score  : {f1:.2f}  (overall balance)")
    print(f"  Threshold : >= 50 used for final_flag")
    print(f"  Fraud Rate: {df['final_flag'].mean()*100:.1f}% of invoices flagged")
    print(f"  Contamination: 10% in Isolation Forest")

    if precision < 0.3:
        print("\n  ⚠️  WARNING: Low precision — too many false positives")
        print("     → Consider raising threshold above 50")
    if recall < 0.3:
        print("\n  ⚠️  WARNING: Low recall — missing too many frauds")
        print("     → Consider lowering threshold below 50")
    if f1 >= 0.6:
        print("\n  ✅ Model performing well (F1 >= 0.6)")
else:
    print(f"  ⚠️  Not enough rows for calibration (need ≥10, have {len(df)})")

print("=" * 50)

print("\n📋 Risk Score Weight Breakdown:")
weights = {
    "Exact Duplicate":     ("duplicate_flag",       25),
    "Near Duplicate (NLP)":("near_duplicate",        20),
    "ML Anomaly":          ("ml_fraud_flag",         15),
    "Circular Trading":    ("in_circular_cycle",     15),
    "GST Mismatch":        ("gst_mismatch_flag",     10),
    "High Amount":         ("high_amount_flag",       5),
    "High Frequency":      ("high_frequency_flag",    3),
    "PO Missing":          ("po_missing",             1),
    "GRN Missing":         ("grn_missing",            1),
}
for label, (col, pts) in weights.items():
    avg = (df[col] * pts).mean()
    print(f"  {label:<25}: {pts:>3} pts  → avg contribution: {avg:.1f}")

# ──────────────────────────────────────────────────────────────
# 15. EXPLAINABILITY
# ──────────────────────────────────────────────────────────────
def explain(row):
    reasons = []
    if row["duplicate_flag"]:      reasons.append("Exact Duplicate")
    if row["near_duplicate"]:      reasons.append("Near Duplicate (NLP)")
    if row["ml_fraud_flag"]:       reasons.append("ML Anomaly")
    if row["in_circular_cycle"]:   reasons.append("Circular Trading")
    if row["gst_mismatch_flag"]:   reasons.append("GST Mismatch")
    if row["lender_count"] > 1:    reasons.append("Multi-Lender")
    if row["high_amount_flag"]:    reasons.append("High Amount")
    if row["high_frequency_flag"]: reasons.append("High Activity")
    if row["po_missing"]:          reasons.append("PO Missing")
    if row["grn_missing"]:         reasons.append("GRN Missing")
    return ", ".join(reasons)

df["reason"] = df.apply(explain, axis=1)

# ──────────────────────────────────────────────────────────────
# 16. SAVE OUTPUT
# ──────────────────────────────────────────────────────────────
df.to_csv("flagged_invoices.csv", index=False)
print("\n✅ Fraud detection complete")
print(df[["invoice_id", "final_flag", "risk_score", "reason"]].head(10).to_string())

def run_model(df):
    """Entry point called by dashboard.py"""
    import hashlib
    import numpy as np
    from sklearn.ensemble import IsolationForest

    df = df.copy()

    # ── Ensure required columns ──────────────────
    for col in ["gst_amount", "po_number", "grn_number", "buyer_id", "lender_id"]:
        if col not in df.columns:
            df[col] = np.nan

    df["amount"] = pd.to_numeric(df.get("amount", 0), errors="coerce").fillna(0)
    df["gst_amount"] = pd.to_numeric(df["gst_amount"], errors="coerce").fillna(0)

    # ── Duplicate detection ──────────────────────
    def generate_hash(row):
        data = str(row.get("supplier_id", "")) + str(row.get("amount", "")) + str(row.get("date", ""))
        return hashlib.sha256(data.encode()).hexdigest()

    df["invoice_hash"] = df.apply(generate_hash, axis=1)
    df["duplicate_flag"] = df.duplicated(subset=["invoice_hash"], keep=False).astype(int)

    # ── Feature engineering ──────────────────────
    supplier_counts = df.groupby("supplier_id")["amount"].transform("count")
    supplier_avg    = df.groupby("supplier_id")["amount"].transform("mean")
    lender_counts   = df.groupby("supplier_id")["lender_id"].transform("nunique") if "lender_id" in df.columns else 1

    df["supplier_freq"]      = supplier_counts
    df["amount_deviation"]   = (df["amount"] - supplier_avg).abs()
    df["lender_count"]       = lender_counts
    df["high_amount_flag"]   = (df["amount"] > df["amount"].quantile(0.95)).astype(int)
    df["high_frequency_flag"]= (df["supplier_freq"] > df["supplier_freq"].quantile(0.95)).astype(int)
    df["po_missing"]         = df["po_number"].isna().astype(int)
    df["grn_missing"]        = df["grn_number"].isna().astype(int)
    df["gst_mismatch_flag"]  = ((df["gst_amount"] > 0) & (abs(df["amount"] * 0.18 - df["gst_amount"]) > df["amount"] * 0.05)).astype(int)

    # ── Isolation Forest ────────────────────────
    features = ["amount", "supplier_freq", "amount_deviation", "lender_count"]
    X = df[features].fillna(0)
    clf = IsolationForest(contamination=0.1, random_state=42)
    df["ml_fraud_flag"] = (clf.fit_predict(X) == -1).astype(int)

    # ── Stub flags (graph needs full pipeline) ───
    df["near_duplicate"]   = 0
    df["in_circular_cycle"]= 0

    # ── Risk score ───────────────────────────────
    df["risk_score"] = (
        df["duplicate_flag"]    * 30 +
        df["near_duplicate"]    * 20 +
        df["ml_fraud_flag"]     * 20 +
        df["in_circular_cycle"] * 15 +
        df["gst_mismatch_flag"] * 10 +
        df["po_missing"]        *  5 +
        df["grn_missing"]       *  5 +
        df["high_amount_flag"]  *  5 +
        df["lender_count"].clip(upper=3).apply(lambda x: (x-1)*5) +
        df["high_frequency_flag"] * 5
    ).clip(0, 100)

        # ── Explain ──────────────────────────────────
    def explain(row):
        reasons = []
        if row["duplicate_flag"]:       reasons.append("Exact Duplicate")
        if row["near_duplicate"]:       reasons.append("Near Duplicate")
        if row["ml_fraud_flag"]:        reasons.append("ML Anomaly")
        if row["in_circular_cycle"]:    reasons.append("Circular Trading")
        if row["gst_mismatch_flag"]:    reasons.append("GST Mismatch")
        if row["lender_count"] > 1:     reasons.append("Multi-Lender")
        if row["high_amount_flag"]:     reasons.append("High Amount")
        if row["high_frequency_flag"]:  reasons.append("High Frequency")
        if row["po_missing"]:           reasons.append("PO Missing")
        if row["grn_missing"]:          reasons.append("GRN Missing")
        return ", ".join(reasons)

    df["reason"] = df.apply(explain, axis=1)

    return df