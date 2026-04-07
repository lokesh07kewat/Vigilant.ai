import io
import re
import os
from typing import List

import pandas as pd
import numpy as np
import pdfplumber
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from predict import run_model

# ──────────────────────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Vigilant.AI API",
    description="Pre-disbursement supply chain fraud detection",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────
# PDF PARSER  (same logic as dashboard.py)
# ──────────────────────────────────────────────────────────────
def parse_invoice_text(text: str, filename: str = "") -> dict:
    data = {}

    # GSTIN — first = supplier, second = buyer
    gstin_pattern = r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b"
    gstins = re.findall(gstin_pattern, text)
    data["GSTIN"]       = gstins[0] if gstins else "NULL"
    data["supplier_id"] = gstins[0] if len(gstins) > 0 else "NULL"
    data["buyer_id"]    = gstins[1] if len(gstins) > 1 else "NULL"

    # Invoice ID
    inv = re.search(r"Invoice\s*No\.?\s*[:\-]?\s*([A-Z0-9\-/]+)", text, re.IGNORECASE)
    data["invoice_id"] = inv.group(1) if inv else f"INV_{filename.replace('.pdf','')}"

    # Date
    date_match = re.search(r"(\d{2}[\/\-]\d{2}[\/\-]\d{4})", text)
    if date_match:
        data["date"] = date_match.group(1).replace("/", "-")
    else:
        data["date"] = pd.Timestamp.today().strftime("%d-%m-%Y")

    # Amount
    amt = re.search(r"(?:₹|Rs\.?|INR)\s*([\d,]+\.?\d*)", text, re.IGNORECASE)
    if not amt:
        amt = re.search(
            r"(?:Total|Amount Due|Grand Total)[^\d]*([\d,]+\.?\d*)", text, re.IGNORECASE
        )
    data["amount"] = float(amt.group(1).replace(",", "")) if amt else 0.0

    # Lender
    lender = re.search(r"(?:Lender|Bank|Financer)[^\w]*([A-Z][a-zA-Z\s]+)", text)
    data["lender_id"] = lender.group(1).strip() if lender else "UNKNOWN_LEN"

    # PO / GRN
    po  = re.search(r"(?:PO|Purchase Order)\s*(?:No\.?|#)?\s*([A-Z0-9\-]+)",  text, re.IGNORECASE)
    grn = re.search(r"(?:GRN|Goods Receipt)\s*(?:No\.?|#)?\s*([A-Z0-9\-]+)", text, re.IGNORECASE)
    data["po_number"]  = po.group(1)  if po  else None
    data["grn_number"] = grn.group(1) if grn else None

    # GST amount
    gst = re.search(
        r"(?:GST|Tax Amount|IGST|CGST\s*\+\s*SGST)[^\d]*([\d,]+\.?\d*)", text, re.IGNORECASE
    )
    data["gst_amount"]     = float(gst.group(1).replace(",", "")) if gst else 0.0
    data["invoice_amount"] = data["amount"]
    data["_source"]        = filename

    return data


def parse_uploaded_file(f: UploadFile) -> pd.DataFrame:
    """Parse one UploadFile into a DataFrame regardless of format."""
    name = f.filename or ""
    low  = name.lower()
    contents = f.file.read()

    if low.endswith(".csv"):
        return pd.read_csv(io.StringIO(contents.decode("utf-8")))

    elif low.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(contents))

    elif low.endswith(".pdf"):
        text = ""
        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t
        if not text.strip():
            raise ValueError(f"No readable text found in PDF: {name}")
        parsed = parse_invoice_text(text, filename=name)
        return pd.DataFrame([parsed])

    else:
        raise ValueError(f"Unsupported file type: {name}. Use CSV, Excel, or PDF.")


# ──────────────────────────────────────────────────────────────
# HEALTH CHECK
# ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    model_loaded = os.path.exists("model/model.pkl")
    return {
        "status":       "ok",
        "model_loaded": model_loaded,
        "version":      "2.0.0",
    }


# ──────────────────────────────────────────────────────────────
# SCORE ENDPOINT  ← accepts multiple files (CSV, Excel, PDF)
# ──────────────────────────────────────────────────────────────
@app.post("/score")
async def score_invoices(files: List[UploadFile] = File(...)):
    """
    Upload one or more invoice files (CSV, Excel, PDF).
    Returns fraud scores for all invoices merged across all files.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    frames = []
    errors = []

    for f in files:
        try:
            df_part = parse_uploaded_file(f)
            df_part["_source"] = f.filename
            frames.append(df_part)
        except ValueError as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(f"{f.filename}: {e}")

    if not frames:
        raise HTTPException(
            status_code=422,
            detail={"message": "No valid invoice data extracted.", "errors": errors},
        )

    combined = pd.concat(frames, ignore_index=True)

    try:
        result = run_model(combined)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    # Convert NaN/NaT to None for clean JSON
    result = result.replace({np.nan: None, pd.NaT: None})

    response = {
        "total":          len(result),
        "fraud_detected": int(result["final_flag"].sum()),
        "files_processed": len(frames),
        "parse_errors":   errors,
        "results":        result.to_dict(orient="records"),
    }
    return JSONResponse(content=response)


# ──────────────────────────────────────────────────────────────
# FLAGGED-ONLY ENDPOINT  ← convenience endpoint
# ──────────────────────────────────────────────────────────────
@app.post("/score/flagged")
async def score_flagged(files: List[UploadFile] = File(...)):
    """
    Same as /score but returns ONLY the flagged (fraud) invoices.
    Useful for alert dashboards that only care about high-risk rows.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    frames = []
    errors = []

    for f in files:
        try:
            df_part = parse_uploaded_file(f)
            df_part["_source"] = f.filename
            frames.append(df_part)
        except ValueError as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(f"{f.filename}: {e}")

    if not frames:
        raise HTTPException(
            status_code=422,
            detail={"message": "No valid invoice data extracted.", "errors": errors},
        )

    combined = pd.concat(frames, ignore_index=True)

    try:
        result = run_model(combined)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    flagged = result[result["final_flag"] == 1].copy()
    flagged = flagged.replace({np.nan: None, pd.NaT: None})

    response = {
        "total":           len(result),
        "fraud_detected":  len(flagged),
        "files_processed": len(frames),
        "parse_errors":    errors,
        "flagged_invoices": flagged.to_dict(orient="records"),
    }
    return JSONResponse(content=response)


# ──────────────────────────────────────────────────────────────
# SUMMARY ENDPOINT  ← per-file breakdown
# ──────────────────────────────────────────────────────────────
@app.post("/score/summary")
async def score_summary(files: List[UploadFile] = File(...)):
    """
    Returns a per-file summary: invoice count, flagged count,
    average risk score for each uploaded file.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    frames = []
    errors = []

    for f in files:
        try:
            df_part = parse_uploaded_file(f)
            df_part["_source"] = f.filename
            frames.append(df_part)
        except Exception as e:
            errors.append(f"{f.filename}: {e}")

    if not frames:
        raise HTTPException(
            status_code=422,
            detail={"message": "No valid data extracted.", "errors": errors},
        )

    combined = pd.concat(frames, ignore_index=True)

    try:
        result = run_model(combined)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    if "_source" in result.columns:
        summary = (
            result.groupby("_source")
            .agg(
                invoices      = ("invoice_id",  "count"),
                flagged       = ("final_flag",  "sum"),
                avg_risk_score= ("risk_score",  "mean"),
            )
            .reset_index()
            .rename(columns={"_source": "file"})
        )
        summary["avg_risk_score"] = summary["avg_risk_score"].round(1)
        summary_list = summary.to_dict(orient="records")
    else:
        summary_list = []

    return JSONResponse(content={
        "total":           len(result),
        "fraud_detected":  int(result["final_flag"].sum()),
        "files_processed": len(frames),
        "parse_errors":    errors,
        "per_file_summary": summary_list,
    })


# ──────────────────────────────────────────────────────────────
# RUN  (python api.py)
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)