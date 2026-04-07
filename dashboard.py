import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import pdfplumber
import re
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from ml import run_model

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Vigilant.AI – Supply Chain Fraud Detection",
    page_icon="🚨",
    layout="wide",
)

st.title("🚨 Supply Chain Fraud Detection System")
st.caption("Detecting invoice fraud using ML + Rules + Graph Intelligence")

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
if "all_results" not in st.session_state:
    st.session_state.all_results = pd.DataFrame()
if "processing_log" not in st.session_state:
    st.session_state.processing_log = []

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
DATE_FORMAT = "%d-%m-%Y"

def extract_value(text: str, patterns: list, default=None):
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return default

def parse_date(raw: str) -> str:
    if not raw:
        return datetime.today().strftime(DATE_FORMAT)
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).strftime(DATE_FORMAT)
        except ValueError:
            continue
    return datetime.today().strftime(DATE_FORMAT)

def pdf_to_row(file_obj, filename: str):
    warnings = []
    try:
        with pdfplumber.open(file_obj) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception as e:
        warnings.append(f"Could not read PDF: {e}")
        text = ""

    raw_amt = extract_value(text, [
        r"(?:total|amount|invoice value|grand total)[^\d]*?([\d,]+(?:\.\d{1,2})?)",
        r"₹\s*([\d,]+(?:\.\d{1,2})?)",
        r"INR\s*([\d,]+(?:\.\d{1,2})?)",
    ])
    try:
        amount = float(raw_amt.replace(",", "")) if raw_amt else 0.0
    except (ValueError, AttributeError):
        amount = 0.0

    raw_date = extract_value(text, [
        r"(?:invoice date|date of invoice|date)[:\s]+(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
        r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})",
        r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})",
    ])
    if not raw_date:
        warnings.append("Date not found in PDF — using today's date as fallback")
    date_str = parse_date(raw_date)

    gstins = re.findall(r'\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}Z[A-Z\d]{1}\b', text)
    supplier_id = gstins[0] if len(gstins) > 0 else f"SUP_{filename[:8].upper()}"
    buyer_id    = gstins[1] if len(gstins) > 1 else f"BUY_{filename[:8].upper()}"
    lender_id   = gstins[2] if len(gstins) > 2 else f"LEN_{filename[:8].upper()}"

    invoice_no = extract_value(text, [
        r"(?:invoice\s*(?:no|number|#))[:\s]+([A-Z0-9\/\-]+)",
        r"(?:inv)[:\s#]+([A-Z0-9\/\-]+)",
    ], default=f"INV_{filename}")

    row = {
        "invoice_id":   invoice_no,
        "supplier_id":  supplier_id,
        "buyer_id":     buyer_id,
        "lender_id":    lender_id,
        "amount":       amount,
        "date":         date_str,
        "_source_file": filename,
        "_warnings":    "; ".join(warnings) if warnings else "",
    }
    return row, warnings

def compute_hash(row: pd.Series) -> str:
    key = f"{row['supplier_id']}|{row['amount']}|{row['date']}"
    return hashlib.sha256(key.encode()).hexdigest()

def draw_graph(df: pd.DataFrame, results: pd.DataFrame = None):
    G = nx.DiGraph()
    for _, r in df.iterrows():
        if pd.notna(r["supplier_id"]) and pd.notna(r["buyer_id"]):
            G.add_edge(r["supplier_id"], r["buyer_id"],
               weight=r["amount"], label=r["invoice_id"])

    cycle_nodes = set()
    if results is not None and "in_circular_cycle" in results.columns:
        cycle_nodes = set(results[results["in_circular_cycle"] == 1]["supplier_id"].tolist())

    node_colors = ["#e74c3c" if n in cycle_nodes else "#4F8EF7" for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(10, 5))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx(G, pos, ax=ax, node_size=800,
                     node_color=node_colors, font_size=8,
                     arrows=True, arrowsize=15)
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=6)
    ax.set_title("Transaction Graph  (🔴 = Circular Trading Node)", fontsize=12)
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

# ─────────────────────────────────────────────
# Upload section
# ─────────────────────────────────────────────
st.subheader("Upload Invoices (CSV or PDF)")
uploaded_files = st.file_uploader(
    "Drag and drop files here",
    type=["csv", "pdf"],
    accept_multiple_files=True,
    help="You can select multiple CSV/PDF files at once.",
)

if uploaded_files:
    rows = []
    all_warnings = {}

    progress = st.progress(0, text="Parsing files…")

    for i, f in enumerate(uploaded_files):
        progress.progress((i + 1) / len(uploaded_files), text=f"Parsing: {f.name}")

        if f.name.lower().endswith(".pdf"):
            row, warns = pdf_to_row(f, f.name)
            rows.append(row)
            if warns:
                all_warnings[f.name] = warns

        elif f.name.lower().endswith(".csv"):
            try:
                tmp = pd.read_csv(f)
                tmp["_source_file"] = f.name
                tmp["_warnings"] = ""
                rows.extend(tmp.to_dict("records"))
            except Exception as e:
                st.error(f"Could not read CSV '{f.name}': {e}")

    progress.progress(1.0, text="Parsing complete ✓")

    if not rows:
        st.warning("No valid invoice data could be extracted from the uploaded files.")
        st.stop()

    df = pd.DataFrame(rows)

    for fname, warns in all_warnings.items():
        for w in warns:
            st.warning(f"⚠️ [{fname}] {w}")

    required_cols = ["invoice_id", "supplier_id", "buyer_id", "lender_id", "amount", "date"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns in uploaded data: {missing}")
        st.stop()

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    df["_hash"] = df.apply(compute_hash, axis=1)
    dup_mask = df.duplicated(subset=["_hash"], keep="first")
    n_dups = dup_mask.sum()
    if n_dups > 0:
        st.error(f"🔴 {n_dups} duplicate invoice(s) detected and removed before scoring.")
        df = df[~dup_mask].reset_index(drop=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Invoices", len(df))
    col2.metric("Files Uploaded", len(uploaded_files))
    col3.metric("Duplicates Removed", n_dups)

    with st.expander("📋 Raw Parsed Data", expanded=False):
        st.dataframe(
            df.drop(columns=["_hash", "_warnings", "_source_file"], errors="ignore"),
            use_container_width=True,
        )

    # ─────────────────────────────────────────────
    # Run ML model
    # ─────────────────────────────────────────────
    st.subheader("🔍 Running Fraud Detection…")
    with st.spinner("Scoring invoices…"):
        try:
            results = run_model(df)
        except Exception as e:
            st.error(f"Model error: {e}")
            st.stop()

    if results is None:                          # ← BUG 1 FIXED: correct indentation
        st.error("run_model() returned None — make sure ml.py ends with 'return df'.")
        st.stop()

    st.session_state.all_results = pd.concat(
        [st.session_state.all_results, results], ignore_index=True
    ).drop_duplicates(subset=["invoice_id"])

    # ─────────────────────────────────────────────
    # Circular trading alert
    # ─────────────────────────────────────────────
    if "in_circular_cycle" in results.columns:
        circular = results[results["in_circular_cycle"] == 1]
        if not circular.empty:
            st.error(
                f"🔄 **Circular Trading Detected!** "
                f"{len(circular)} invoice(s) form a suspicious transaction loop."
            )

    # ─────────────────────────────────────────────
    # Results display
    # ─────────────────────────────────────────────
    st.subheader("📊 Risk Scores")

    if "risk_score" in results.columns:
        flagged = results[results["risk_score"] >= 50]
        safe    = results[results["risk_score"] <  50]
    else:
        flagged = pd.DataFrame()
        safe    = results

    tab1, tab2, tab3 = st.tabs([
        f"🔴 Flagged ({len(flagged)})",
        f"🟢 Safe ({len(safe)})",
        "📈 All Results",
    ])

    with tab1:
        if flagged.empty:
            st.success("No high-risk invoices detected.")
        else:
            st.dataframe(
                flagged.style.background_gradient(subset=["risk_score"], cmap="Reds"),
                use_container_width=True,
            )

    with tab2:
        if safe.empty:
            st.warning("All invoices were flagged as high-risk.")
        else:
            st.dataframe(
                safe.style.background_gradient(subset=["risk_score"], cmap="Greens"),
                use_container_width=True,
            )

    with tab3:
        st.dataframe(results, use_container_width=True)

    # ─────────────────────────────────────────────
    # Graph  ← BUG 2+3 FIXED: removed from here, graph logic is in ml.py
    # ─────────────────────────────────────────────
    with st.expander("🕸️ Transaction Graph", expanded=False):
        draw_graph(df, results)

    csv_out = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Results CSV",
        data=csv_out,
        file_name=f"fraud_results_{datetime.today().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "files": [f.name for f in uploaded_files],
        "invoices_processed": len(df),
        "flagged": len(flagged),
    }
    st.session_state.processing_log.append(log_entry)

    with st.expander("📜 Processing Log", expanded=False):
        for entry in reversed(st.session_state.processing_log):
            st.write(entry)

           