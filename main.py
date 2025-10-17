import io, os, shutil, zipfile, re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pdfplumber
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

APP = FastAPI(title="Recon Backend v16.22", version="1.0")

ALLOWED_ORIGINS = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:3000,https://recondb.vercel.app"
).split(",")

APP.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path.cwd()
RUN_ROOT = BASE_DIR / "runs"
RUN_ROOT.mkdir(parents=True, exist_ok=True)

CURRENT_RUN_DIR: Optional[Path] = None
CURRENT_BANK_TYPE: Optional[str] = None

# ---------- TPA Registry ----------
TPA_CHOICES = [
    "IHX (Original MIS)",
    "CARE HEALTH INSURANCE LIMITED",
]

CANON_COLS = ["Cheque/ NEFT/ UTR No.", "Patient Name", "In Patient Number", "Settled Amount"]

TPA_MIS_MAPS = {
    "IHX (Original MIS)": {
        "Cheque/ NEFT/ UTR No.": "Cheque/ NEFT/ UTR No.",
        "Patient Name": "Patient Name",
        "In Patient Number": "In Patient Number",
        "Settled Amount": "Settled Amount",
    },
    "CARE HEALTH INSURANCE LIMITED": {
        "Cheque/ NEFT/ UTR No.": "NEFT Number",
        "Patient Name": "Patient Name",
        "In Patient Number": "AL Number",
        "Settled Amount": "Settled Amount",
    },
}

# ---------- Helpers ----------
def _norm(s):
    s = "" if pd.isna(s) else str(s)
    return "".join(s.upper().split())

def _to_amt(x):
    try:
        s = str(x).replace(",", "").strip()
        if s in ["", "-", "—"]: return np.nan
        if s.startswith("(") and s.endswith(")"): return -round(float(s[1:-1]), 2)
        return round(float(s), 2)
    except: return np.nan

def save_xlsx(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)
    return path

def new_run_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = RUN_ROOT / f"reco_outputs_{stamp}"
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def zip_outputs(paths, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, p.name)

def ensure_xlsx(path: Path) -> Path:
    path = Path(path)
    if path.suffix.lower() == ".xls":
        df = pd.read_excel(path, dtype=str)
        new_path = path.with_suffix(".xlsx")
        df.to_excel(new_path, index=False)
        return new_path
    return path

# ---------- ICICI Bank (Excel) ----------
def clean_raw_bank_statement_icici(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, dtype=str, header=None)
    header_row_idx = None
    for i, row in raw.iterrows():
        vals = [str(v).strip() for v in row.tolist() if pd.notna(v)]
        if "Transaction ID" in vals and "Description" in vals:
            header_row_idx = i
            break
    if header_row_idx is None:
        raise ValueError("Could not detect header row in bank statement.")
    df = pd.read_excel(path, dtype=str, header=header_row_idx)
    keep_cols = ["No.", "Transaction ID", "Txn Posted Date",
                 "Description", "Cr/Dr", "Transaction Amount(INR)"]
    df = df[[c for c in keep_cols if c in df.columns]]
    return df.dropna(how="all").reset_index(drop=True)

# ---------- AXIS Bank (PDF) ----------
def _row_has_banner(row_vals) -> bool:
    joined = " ".join([("" if v is None else str(v)) for v in row_vals])
    flat = re.sub(r"\s+", "", joined).upper()
    tokens = [
        "OPENINGBALANCE","CARRYFORWARDBALANCE","TRANSACTIONTOTAL","CLOSINGBALANCE",
        "TRANDATEVALUEDATE",
    ]
    return any(tok in flat for tok in tokens)

def _prefix_from_col0(c0: str) -> str:
    s = str(c0 or "").strip()
    mdate = re.search(r"\d{2}/\d{2}/\d{4}", s)
    if mdate: s = s[:mdate.start()]
    s = re.sub(r"[^A-Za-z]", "", s)
    return s[:3]

def clean_raw_bank_statement_axis_pdf(pdf_path: Path):
    frames = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            for t in (page.extract_tables() or []):
                if not t: continue
                rows = [[("" if v is None else str(v)).replace("\n"," ").strip() for v in r] for r in t]
                frames.append(pd.DataFrame(rows))
    if not frames:
        raise ValueError("No tables found in Axis Bank PDF.")
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[~raw.apply(lambda r: _row_has_banner(list(r.values)), axis=1)].reset_index(drop=True)

    col0 = raw.iloc[:,0].astype(str) if raw.shape[1] > 0 else pd.Series([""]*len(raw))
    col1 = raw.iloc[:,1].astype(str) if raw.shape[1] > 1 else pd.Series([""]*len(raw))
    col3 = raw.iloc[:,3].astype(str) if raw.shape[1] > 3 else pd.Series([""]*len(raw))
    col4 = raw.iloc[:,4].astype(str) if raw.shape[1] > 4 else pd.Series([""]*len(raw))

    tran_dates, value_dates = [], []
    date_re = re.compile(r"\d{2}/\d{2}/\d{4}")
    for c in col0:
        ds = date_re.findall(c)
        tran_dates.append(ds[0] if len(ds) > 0 else "")
        value_dates.append(ds[1] if len(ds) > 1 else "")

    def make_desc(c0, c1):
        prefix = _prefix_from_col0(c0)
        d = str(c1 or "").strip()
        if prefix and not d[:len(prefix)].lower() == prefix.lower():
            return (prefix + d).strip()
        return d
    description = [make_desc(c0, c1) for c0, c1 in zip(col0, col1)]

    debit  = col3.apply(_to_amt)
    credit = col4.apply(_to_amt)
    crdr, amt = [], []
    for d,c in zip(debit,credit):
        if not np.isnan(c): crdr.append("Cr"); amt.append(c)
        elif not np.isnan(d): crdr.append("Dr"); amt.append(d)
        else: crdr.append(""); amt.append(np.nan)

    clean = pd.DataFrame({
        "No.": range(1, len(raw)+1),
        "Transaction ID": "",
        "Tran Date": tran_dates,
        "Value Date": value_dates,
        "Description": description,
        "Cr/Dr": crdr,
        "Transaction Amount(INR)": amt,
    }).dropna(how="all").reset_index(drop=True)

    return raw, clean

# ---------- ICICI Advance (PDF) ----------
def read_clean_pdf_to_df_icici_advance(pdf_path: Path) -> pd.DataFrame:
    tables = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            for t in (page.extract_tables() or []):
                if t: tables.append(pd.DataFrame(t))
    if not tables:
        raise ValueError("No tables found in Advance Account Statement PDF.")
    hdr = tables[0].iloc[0]
    proc=[]
    for d in tables:
        d = d.copy()
        d = d[~d.apply(lambda r: (r == hdr).all(), axis=1)]
        d.columns = hdr
        proc.append(d.reset_index(drop=True))
    df = pd.concat(proc, ignore_index=True).dropna(how="all").reset_index(drop=True)
    if "S.No." not in df.columns:
        raise ValueError("Expected 'S.No.' in Advance PDF.")
    df["S.No."] = df["S.No."].replace("", np.nan).fillna(method="ffill")
    df.columns = [str(c).strip().replace(".","_").replace(" ","_") for c in df.columns]
    if "Refer_No" not in df.columns:
        raise ValueError("Expected 'Refer_No' after normalization.")
    df["Refer_No_UTR"] = df["Refer_No"].apply(lambda x: str(x).replace("/XUTR/","").strip() if pd.notna(x) else "")
    return df

# ---------- AXIS Advance (Excel) ----------
def read_clean_axis_advance_xlsx(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, dtype=str).dropna(how="all").reset_index(drop=True)
    if "TRANID" not in df.columns:
        raise ValueError("Axis Advance Excel missing TRANID")
    if "UTR" not in df.columns:
        raise ValueError("Axis Advance Excel missing UTR")
    df["Msg_Refer_No"] = df["TRANID"].astype(str).str.strip()
    df["Refer_No_UTR"] = df["UTR"].astype(str).str.strip().str.replace("^/XUTR/","",regex=True)
    return df

# ---------- Step 2 (Bank × Advance) ----------
def step2_match_bank_advance(bank_df: pd.DataFrame, adv_df: pd.DataFrame):
    if bank_df.empty:
        return pd.DataFrame(), bank_df
    parts=[]
    if "Msg_Refer_No" not in adv_df.columns and "Refer_No_UTR" in adv_df.columns:
        adv_df = adv_df.copy()
        adv_df["Msg_Refer_No"] = adv_df["Refer_No_UTR"]
    for msg in adv_df["Msg_Refer_No"].dropna().astype(str).unique():
        s = msg.strip()
        if not s: continue
        m = bank_df["Description"].str.contains(s, regex=False, na=False)
        if m.any():
            t = bank_df.loc[m].copy()
            t["Matched_Key"] = s
            parts.append(t)
    if not parts:
        return pd.DataFrame(), bank_df
    matched = pd.concat(parts, ignore_index=True).merge(
        adv_df, left_on="Matched_Key", right_on="Msg_Refer_No", how="inner", suffixes=("_bank","_adv")
    )
    matched = matched.drop_duplicates()
    if "Transaction ID" in bank_df.columns and "Transaction ID" in matched.columns:
        not_in = bank_df.loc[~bank_df["Transaction ID"].isin(matched["Transaction ID"])]
    else:
        not_in = bank_df.loc[~bank_df["Description"].isin(matched["Description"])]
    return matched, not_in

# ---------- Step 3 (to MIS) — TPA-aware ----------
def step3_map_to_mis(step2_df: pd.DataFrame, mis_path: Path, tpa_name: str) -> pd.DataFrame:
    if step2_df.empty:
        return pd.DataFrame()
    if "Refer_No_UTR" not in step2_df.columns:
        raise ValueError("Step-2 data missing 'Refer_No_UTR'.")

    if tpa_name not in TPA_MIS_MAPS:
        raise ValueError(f"No MIS mapping registered for TPA: {tpa_name}")
    mapping = TPA_MIS_MAPS[tpa_name]
    missing_map = [c for c in CANON_COLS if c not in mapping]
    if missing_map:
        raise ValueError(f"Mapping incomplete for {tpa_name}. Needs: {missing_map}")

    mis = pd.read_excel(mis_path, dtype=str)

    missing_src = [src for src in mapping.values() if src not in mis.columns]
    if missing_src:
        raise ValueError(f"MIS file for {tpa_name} missing headers: {missing_src}")

    mis_std = mis.rename(columns={v: k for k, v in mapping.items()})[CANON_COLS].copy()
    for c in CANON_COLS:
        mis_std[c] = mis_std[c].astype(str).str.strip()

    merged = step2_df.merge(
        mis_std,
        left_on="Refer_No_UTR",
        right_on="Cheque/ NEFT/ UTR No.",
        how="inner",
        suffixes=("", "_mis"),
    )
    return merged.drop_duplicates()

# ---------- Step 4 (Outstanding strict) ----------
def step4_strict_matches(step3_df: pd.DataFrame, outstanding_path: Path) -> pd.DataFrame:
    if step3_df.empty:
        return pd.DataFrame()
    out = pd.read_excel(outstanding_path, dtype=str).copy()
    if "Claim No" in out.columns:
        out = out.rename(columns={"Claim No": "Claim No_out"})
    for col in ["Patient Name","CR No","Balance"]:
        if col not in out.columns:
            raise ValueError(f"Outstanding missing '{col}'.")
    for col in ["Patient Name","In Patient Number","Settled Amount"]:
        if col not in step3_df.columns:
            raise ValueError(f"Step-3 missing '{col}'.")
    L = out.copy(); R = step3_df.copy()
    L["_PNORM"] = L["Patient Name"].apply(_norm)
    R["_PNORM"] = R["Patient Name"].apply(_norm)
    L["_CRNORM"] = L["CR No"].apply(_norm)
    R["_CRNORM"] = R["In Patient Number"].apply(_norm)
    merged = L.merge(R, on=["_PNORM","_CRNORM"], how="inner", suffixes=("_out","_m3"))
    bal  = merged["Balance"].apply(_to_amt)
    sett = merged["Settled Amount"].apply(_to_amt)
    ok_mask = (bal.notna() & sett.notna() & (bal == sett))
    return merged.loc[ok_mask].drop(columns=["_PNORM","_CRNORM"]).drop_duplicates()

# ---------- API Endpoints ----------

@APP.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "version": "16.22"}

@APP.get("/download/{run_id}/{filename}")
async def download_file(run_id: str, filename: str):
    """Download a specific file from a run"""
    file_path = RUN_ROOT / run_id / filename
    
    print(f"[Download] Requested: {filename}")
    print(f"[Download] Looking in: {file_path}")
    print(f"[Download] File exists: {file_path.exists()}")
    
    if not file_path.exists():
        run_dir = RUN_ROOT / run_id
        if run_dir.exists():
            available = [f.name for f in run_dir.iterdir()]
            print(f"[Download] Available files: {available}")
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    if filename.endswith('.zip'):
        media_type = "application/zip"
    elif filename.endswith('.pdf'):
        media_type = "application/pdf"
    else:
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    
    print(f"[Download] Streaming file: {filename} ({media_type})")
    
    def file_iterator():
        with open(file_path, "rb") as f:
            yield f.read()
    
    return StreamingResponse(
        file_iterator(),
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Type": media_type
        }
    )

@APP.get("/tpa-choices")
async def get_tpa_choices():
    """Returns list of available TPA choices"""
    return {"tpa_choices": TPA_CHOICES}

@APP.post("/reconcile/step1")
async def reconcile_step1(
    bank_type: str = Form(...),
    bank_file: UploadFile = File(...),
    advance_file: UploadFile = File(...)
):
    """Step-1: Upload Bank + Advance files based on bank type"""
    global CURRENT_RUN_DIR, CURRENT_BANK_TYPE
    try:
        CURRENT_RUN_DIR = new_run_dir()
        CURRENT_BANK_TYPE = bank_type
        run_id = CURRENT_RUN_DIR.name

        print(f"[Step1] Starting with bank_type={bank_type}, run_id={run_id}")

        if bank_type not in ["ICICI", "AXIS"]:
            raise HTTPException(status_code=400, detail="Invalid bank_type. Must be 'ICICI' or 'AXIS'.")

        if bank_type == "ICICI":
            bank_path = CURRENT_RUN_DIR / "bank.xlsx"
            with open(bank_path, "wb") as f:
                f.write(await bank_file.read())
            bank_path = ensure_xlsx(bank_path)
            bank_df = clean_raw_bank_statement_icici(bank_path)
            
            advance_path = CURRENT_RUN_DIR / "advance.pdf"
            with open(advance_path, "wb") as f:
                f.write(await advance_file.read())
            adv_df = read_clean_pdf_to_df_icici_advance(advance_path)
            if "Msg_Refer_No" not in adv_df.columns:
                adv_df["Msg_Refer_No"] = adv_df["Refer_No_UTR"]
            
            out_bank = CURRENT_RUN_DIR / "01a_icici_bank_clean.xlsx"
            out_adv = CURRENT_RUN_DIR / "01b_icici_advance_clean.xlsx"
        else:
            bank_path = CURRENT_RUN_DIR / "bank.pdf"
            with open(bank_path, "wb") as f:
                f.write(await bank_file.read())
            raw_axis_df, bank_df = clean_raw_bank_statement_axis_pdf(bank_path)
            
            advance_path = CURRENT_RUN_DIR / "advance.xlsx"
            with open(advance_path, "wb") as f:
                f.write(await advance_file.read())
            advance_path = ensure_xlsx(advance_path)
            adv_df = read_clean_axis_advance_xlsx(advance_path)
            
            out_bank = CURRENT_RUN_DIR / "01a_axis_bank_clean.xlsx"
            out_adv = CURRENT_RUN_DIR / "01b_axis_advance_clean.xlsx"

        save_xlsx(bank_df, out_bank)
        save_xlsx(adv_df, out_adv)
        
        print(f"[Step1] Files saved: {out_bank.name}, {out_adv.name}")
        print(f"[Step1] Files exist: bank={out_bank.exists()}, advance={out_adv.exists()}")
        
        return {
            "status": "success",
            "run_id": run_id,
            "bank_type": bank_type,
            "counts": {
                "bank_rows": len(bank_df),
                "advance_rows": len(adv_df)
            },
            "files": {
                "bank": f"/download/{run_id}/{out_bank.name}",
                "advance": f"/download/{run_id}/{out_adv.name}"
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@APP.post("/reconcile/step2")
async def reconcile_step2():
    """Step-2: Match Bank with Advance (no upload needed)"""
    global CURRENT_RUN_DIR, CURRENT_BANK_TYPE
    if CURRENT_RUN_DIR is None:
        raise HTTPException(status_code=400, detail="Run not initialized. Complete Step-1 first.")
    try:
        run_id = CURRENT_RUN_DIR.name
        
        if CURRENT_BANK_TYPE == "ICICI":
            bank_path = CURRENT_RUN_DIR / "01a_icici_bank_clean.xlsx"
            adv_path = CURRENT_RUN_DIR / "01b_icici_advance_clean.xlsx"
        else:
            bank_path = CURRENT_RUN_DIR / "01a_axis_bank_clean.xlsx"
            adv_path = CURRENT_RUN_DIR / "01b_axis_advance_clean.xlsx"
        
        if not bank_path.exists() or not adv_path.exists():
            raise HTTPException(status_code=400, detail="Step-1 outputs missing.")
        
        bank_df = pd.read_excel(bank_path, dtype=str)
        adv_df = pd.read_excel(adv_path, dtype=str)
        
        s2_matched, s2_notin = step2_match_bank_advance(bank_df, adv_df)
        
        out2a = CURRENT_RUN_DIR / "02a_bank_x_advance_matches.xlsx"
        out2b = CURRENT_RUN_DIR / "02b_bank_not_in_advance.xlsx"
        save_xlsx(s2_matched, out2a)
        save_xlsx(s2_notin, out2b)
        
        return {
            "status": "success",
            "run_id": run_id,
            "counts": {
                "matches": len(s2_matched),
                "not_in": len(s2_notin)
            },
            "files": {
                "matches": f"/download/{run_id}/{out2a.name}",
                "not_in": f"/download/{run_id}/{out2b.name}"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@APP.post("/reconcile/step3")
async def reconcile_step3(
    tpa_name: str = Form(...),
    mis_file: UploadFile = File(...)
):
    """Step-3: Map to MIS with TPA selection"""
    global CURRENT_RUN_DIR
    if CURRENT_RUN_DIR is None:
        raise HTTPException(status_code=400, detail="Run not initialized. Complete Step-1 & Step-2 first.")
    try:
        run_id = CURRENT_RUN_DIR.name
        
        if tpa_name not in TPA_CHOICES:
            raise HTTPException(status_code=400, detail=f"Invalid TPA. Choose from: {TPA_CHOICES}")
        
        mis_path = CURRENT_RUN_DIR / "mis.xlsx"
        with open(mis_path, "wb") as f:
            f.write(await mis_file.read())
        mis_path = ensure_xlsx(mis_path)
        
        s2_path = CURRENT_RUN_DIR / "02a_bank_x_advance_matches.xlsx"
        if not s2_path.exists():
            raise HTTPException(status_code=400, detail="Step-2 output missing.")
        
        s2_matched = pd.read_excel(s2_path, dtype=str)
        s3 = step3_map_to_mis(s2_matched, mis_path, tpa_name)
        
        out3 = CURRENT_RUN_DIR / "03_matches_mapped_to_mis.xlsx"
        save_xlsx(s3, out3)
        
        return {
            "status": "success",
            "run_id": run_id,
            "tpa_name": tpa_name,
            "counts": {
                "rows": len(s3)
            },
            "files": {
                "mis": f"/download/{run_id}/{out3.name}"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@APP.post("/reconcile/step4")
async def reconcile_step4(outstanding_file: UploadFile = File(...)):
    """Step-4: Outstanding match and create ZIP"""
    global CURRENT_RUN_DIR
    if CURRENT_RUN_DIR is None:
        raise HTTPException(status_code=400, detail="Run not initialized. Complete previous steps first.")
    try:
        run_id = CURRENT_RUN_DIR.name
        
        out_path = CURRENT_RUN_DIR / "outstanding.xlsx"
        with open(out_path, "wb") as f:
            f.write(await outstanding_file.read())
        out_path = ensure_xlsx(out_path)
        
        s3_path = CURRENT_RUN_DIR / "03_matches_mapped_to_mis.xlsx"
        if not s3_path.exists():
            raise HTTPException(status_code=400, detail="Step-3 output missing.")
        
        s3 = pd.read_excel(s3_path, dtype=str)
        s4 = step4_strict_matches(s3, out_path)
        
        out4 = CURRENT_RUN_DIR / "04_outstanding_matches.xlsx"
        save_xlsx(s4, out4)
        
        all_outputs = list(CURRENT_RUN_DIR.glob("0*.xlsx"))
        zip_path = CURRENT_RUN_DIR / f"{run_id}.zip"
        zip_outputs(all_outputs, zip_path)
        
        return {
            "status": "success",
            "run_id": run_id,
            "counts": {
                "rows": len(s4)
            },
            "files": {
                "outstanding": f"/download/{run_id}/{out4.name}",
                "zip": f"/download/{run_id}/{zip_path.name}"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))