import io, os, shutil, zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pdfplumber
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

APP = FastAPI(title="Recon Backend v16.1", version="1.0")

# CORS for local Next.js by default; override with CORS_ALLOW_ORIGINS="http://localhost:3000,https://your.app"
# ALLOWED_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:3000",).split(",")
ALLOWED_ORIGINS = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:3000,https://recondb.vercel.app"
).split(",")



APP.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS if o.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path.cwd()
RUN_ROOT = BASE_DIR / "runs"
RUN_ROOT.mkdir(parents=True, exist_ok=True)

PUBLIC_DIR = BASE_DIR / "public"
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
APP.mount("/files", StaticFiles(directory=str(PUBLIC_DIR)), name="files")

CURRENT_RUN_DIR: Optional[Path] = None  # single-user local dev

# ---------- Helpers (same behavior as Colab) ----------
MDINDIA_SUBSTRINGS = [
    "M.D.INDIA","MDI","MDINDIA","MDAS","MEDIAS","MEDI ASSIST INDIA TPA PVT.LTD."
]

def _norm(s): 
    s = "" if pd.isna(s) else str(s)
    return "".join(s.upper().split())

def _to_amt(x):
    try: return round(float(str(x).replace(",","").strip()), 2)
    except: return np.nan

def save_xlsx(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)
    return path

def df_to_xlsx_response(df: pd.DataFrame, download_name: str, extra_headers: dict = None) -> StreamingResponse:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False)
    buf.seek(0)
    headers = {
        "Content-Disposition": f"attachment; filename*=UTF-8''{download_name}",
        "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
    if extra_headers:
        headers.update(extra_headers)
    return StreamingResponse(buf, headers=headers, media_type=headers["Content-Type"])

def new_run_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = RUN_ROOT / f"reco_outputs_{stamp}"
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def publish(path: Path, run_dir: Path) -> str:
    dst_dir = PUBLIC_DIR / run_dir.name
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / path.name
    shutil.copy2(path, dst)
    return f"/files/{run_dir.name}/{path.name}"

def zip_outputs(paths, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, p.name)

# ---------- Core steps (unchanged logic) ----------
def step1_filter_bank(bank_path: Path) -> pd.DataFrame:
    bank = pd.read_excel(bank_path, dtype=str)
    if "Description" not in bank.columns or "Cr/Dr" not in bank.columns:
        raise ValueError("Bank must contain 'Description' and 'Cr/Dr'.")
    def hit(d): return any(s in str(d).upper() for s in MDINDIA_SUBSTRINGS)
    m = (bank["Cr/Dr"].astype(str).str.upper()=="CR") & (bank["Description"].apply(hit))
    df = bank.loc[m].copy()
    if "Transaction ID" in df.columns: df = df.drop_duplicates(subset=["Transaction ID"])
    return df.drop_duplicates()

def read_clean_pdf_to_df(pdf_path: Path) -> pd.DataFrame:
    tabs=[]
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            for t in (page.extract_tables() or []):
                if t: tabs.append(pd.DataFrame(t))
    if not tabs: raise ValueError("No tables found in Advance Account Statement PDF.")
    hdr = tabs[0].iloc[0]
    proc=[]
    for d in tabs:
        d=d.copy()
        d = d[~d.apply(lambda r:(r==hdr).all(), axis=1)]
        d.columns = hdr
        proc.append(d.reset_index(drop=True))
    df = pd.concat(proc, ignore_index=True).dropna(how="all").reset_index(drop=True)

    if "S.No." not in df.columns: raise ValueError("Expected 'S.No.' in PDF tables.")
    df["S.No."] = df["S.No."].replace("", np.nan).fillna(method="ffill")

    merged=[]; prev=None
    for i,row in df.iterrows():
        new = (i==0) or (str(row["S.No."]).strip()!="" and pd.notna(row["S.No."]))
        if new:
            if prev is not None: merged.append(prev)
            prev = list(row)
        else:
            prev = [(f"{str(a)} {str(b)}".strip() if str(b).strip() else str(a)) for a,b in zip(prev,row)]
    if prev is not None: merged.append(prev)

    df = pd.DataFrame(merged, columns=df.columns)
    df.columns = [str(c).strip().replace(".","_").replace(" ","_") for c in df.columns]

    if "Refer_No" not in df.columns: raise ValueError("Expected 'Refer_No' after normalization.")
    def split(s):
        s=str(s).strip()
        return s[len("/XUTR/"):] if s.startswith("/XUTR/") else s
    df["Refer_No_UTR"] = df["Refer_No"].apply(lambda x: split(x) if pd.notna(x) else "")

    if "Msg_Refer_No" not in df.columns: raise ValueError("Expected 'Msg_Refer_No' after normalization.")
    return df

def step2_match_bank_advance(bank1: pd.DataFrame, adv: pd.DataFrame) -> pd.DataFrame:
    if bank1.empty: return pd.DataFrame()
    parts=[]
    for msg in adv["Msg_Refer_No"].dropna().astype(str).unique():
        s=msg.strip()
        if not s: continue
        m = bank1["Description"].str.contains(s, regex=False, na=False)
        if m.any():
            t = bank1.loc[m].copy(); t["Matched_Key"]=s; parts.append(t)
    if not parts: return pd.DataFrame()
    merged = pd.concat(parts, ignore_index=True).merge(
        adv, left_on="Matched_Key", right_on="Msg_Refer_No", how="inner", suffixes=("_bank","_adv")
    )
    keys=[k for k in ["Transaction ID","Msg_Refer_No"] if k in merged.columns]
    if keys: merged = merged.drop_duplicates(subset=keys)
    return merged.drop_duplicates()

def step3_map_to_mis(step2: pd.DataFrame, mis_path: Path) -> pd.DataFrame:
    if step2.empty: return pd.DataFrame()
    if "Refer_No_UTR" not in step2.columns: raise ValueError("Step-2 missing 'Refer_No_UTR'.")
    mis = pd.read_excel(mis_path, dtype=str)
    if "Cheque/ NEFT/ UTR No." not in mis.columns: raise ValueError("MIS missing 'Cheque/ NEFT/ UTR No.'")
    merged = step2.merge(mis, left_on="Refer_No_UTR", right_on="Cheque/ NEFT/ UTR No.", how="inner", suffixes=("", "_mis"))
    keys=[k for k in ["Cheque/ NEFT/ UTR No.","Claim Number","Transaction ID"] if k in merged.columns]
    if keys: merged = merged.drop_duplicates(subset=keys)
    return merged.drop_duplicates()

def step4_strict_matches(step3_df: pd.DataFrame, outstanding_path: Path) -> pd.DataFrame:
    out = pd.read_excel(outstanding_path, dtype=str).copy()
    if "Claim No" in out.columns:
        out = out.rename(columns={"Claim No": "Claim No_out"})

    for col in ["Patient Name","CR No","Balance"]:
        if col not in out.columns: raise ValueError(f"Outstanding missing '{col}'.")
    for col in ["Patient Name","In Patient Number","Settled Amount"]:
        if col not in step3_df.columns: raise ValueError(f"Step-3 missing '{col}'.")

    L = out.copy(); R = step3_df.copy()
    L["_PNORM"] = L["Patient Name"].apply(_norm); R["_PNORM"] = R["Patient Name"].apply(_norm)
    L["_CRNORM"] = L["CR No"].apply(_norm);       R["_CRNORM"] = R["In Patient Number"].apply(_norm)

    merged = L.merge(R, on=["_PNORM","_CRNORM"], how="inner", suffixes=("_out","_m3"))

    bal  = merged["Balance"].apply(_to_amt)
    sett = merged["Settled Amount"].apply(_to_amt)
    ok_mask = (bal.notna() & sett.notna() & (bal == sett))

    matches_ok = merged.loc[ok_mask].drop(columns=["_PNORM","_CRNORM"]).drop_duplicates()
    return matches_ok

# ---------- API ----------
@APP.post("/reconcile/bank")
async def reconcile_bank(bank1: UploadFile = File(...)):
    """Step-1: returns 01_bank_mdindia_filtered.xlsx (binary). Starts a fresh run dir."""
    global CURRENT_RUN_DIR
    try:
        CURRENT_RUN_DIR = new_run_dir()
        run_id = CURRENT_RUN_DIR.name

        bank_path = CURRENT_RUN_DIR / "bank.xlsx"
        with open(bank_path, "wb") as f:
            f.write(await bank1.read())

        s1 = step1_filter_bank(bank_path)
        out1 = CURRENT_RUN_DIR / "01_bank_mdindia_filtered.xlsx"
        save_xlsx(s1, out1)

        # publish artifacts for convenience (no frontend change needed)
        pub1 = publish(out1, CURRENT_RUN_DIR)

        return df_to_xlsx_response(
            s1, out1.name,
            extra_headers={
                "X-Run-Id": run_id,
                "X-Artifact-01": pub1,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@APP.post("/reconcile/pdf")
async def reconcile_pdf(pdf: UploadFile = File(...)):
    """Step-2: returns 02_bank_x_advance_matches.xlsx (binary)."""
    global CURRENT_RUN_DIR
    if CURRENT_RUN_DIR is None:
        raise HTTPException(status_code=400, detail="Run not initialized. Upload bank file first.")
    try:
        run_id = CURRENT_RUN_DIR.name

        pdf_path = CURRENT_RUN_DIR / "advance.pdf"
        with open(pdf_path, "wb") as f:
            f.write(await pdf.read())

        bank1_path = CURRENT_RUN_DIR / "01_bank_mdindia_filtered.xlsx"
        if not bank1_path.exists():
            raise HTTPException(status_code=400, detail="Step-1 output missing.")

        bank1 = pd.read_excel(bank1_path, dtype=str)
        adv = read_clean_pdf_to_df(pdf_path)
        s2 = step2_match_bank_advance(bank1, adv)
        out2 = CURRENT_RUN_DIR / "02_bank_x_advance_matches.xlsx"
        save_xlsx(s2, out2)

        pub2 = publish(out2, CURRENT_RUN_DIR)

        return df_to_xlsx_response(
            s2, out2.name,
            extra_headers={
                "X-Run-Id": run_id,
                "X-Artifact-02": pub2,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@APP.post("/reconcile/mis")
async def reconcile_mis(mis: UploadFile = File(...)):
    """Step-3: returns 03_matches_mapped_to_mis.xlsx (binary)."""
    global CURRENT_RUN_DIR
    if CURRENT_RUN_DIR is None:
        raise HTTPException(status_code=400, detail="Run not initialized. Upload bank & pdf first.")
    try:
        run_id = CURRENT_RUN_DIR.name

        mis_path = CURRENT_RUN_DIR / "mis.xlsx"
        with open(mis_path, "wb") as f:
            f.write(await mis.read())

        s2_path = CURRENT_RUN_DIR / "02_bank_x_advance_matches.xlsx"
        if not s2_path.exists():
            raise HTTPException(status_code=400, detail="Step-2 output missing.")

        s2 = pd.read_excel(s2_path, dtype=str)
        s3 = step3_map_to_mis(s2, mis_path)
        out3 = CURRENT_RUN_DIR / "03_matches_mapped_to_mis.xlsx"
        save_xlsx(s3, out3)

        pub3 = publish(out3, CURRENT_RUN_DIR)

        return df_to_xlsx_response(
            s3, out3.name,
            extra_headers={
                "X-Run-Id": run_id,
                "X-Artifact-03": pub3,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@APP.post("/reconcile/outstanding")
async def reconcile_outstanding(outstanding: UploadFile = File(...)):
    """Step-4: returns 04_outstanding_matches.xlsx (binary) AND creates the ZIP like Colab."""
    global CURRENT_RUN_DIR
    if CURRENT_RUN_DIR is None:
        raise HTTPException(status_code=400, detail="Run not initialized. Upload previous three files first.")
    try:
        run_id = CURRENT_RUN_DIR.name

        out_path = CURRENT_RUN_DIR / "outstanding.xlsx"
        with open(out_path, "wb") as f:
            f.write(await outstanding.read())

        s3_path = CURRENT_RUN_DIR / "03_matches_mapped_to_mis.xlsx"
        if not s3_path.exists():
            raise HTTPException(status_code=400, detail="Step-3 output missing.")

        s3 = pd.read_excel(s3_path, dtype=str)
        s4 = step4_strict_matches(s3, out_path)
        out4 = CURRENT_RUN_DIR / "04_outstanding_matches.xlsx"
        save_xlsx(s4, out4)

        # Build ZIP with ONLY the four outputs (same as Colab)
        p1 = CURRENT_RUN_DIR / "01_bank_mdindia_filtered.xlsx"
        p2 = CURRENT_RUN_DIR / "02_bank_x_advance_matches.xlsx"
        p3 = CURRENT_RUN_DIR / "03_matches_mapped_to_mis.xlsx"
        p4 = CURRENT_RUN_DIR / "04_outstanding_matches.xlsx"
        zip_path = CURRENT_RUN_DIR.with_suffix(".zip")
        zip_outputs([p1, p2, p3, p4], zip_path)

        # publish artifacts
        pub1 = publish(p1, CURRENT_RUN_DIR)
        pub2 = publish(p2, CURRENT_RUN_DIR)
        pub3 = publish(p3, CURRENT_RUN_DIR)
        pub4 = publish(p4, CURRENT_RUN_DIR)
        pubzip = publish(zip_path, CURRENT_RUN_DIR)

        # Return the step-4 XLSX (binary) — EXACTLY like your “04_outstanding_matches.xlsx”.
        # Extra headers expose artifact URLs (optional to consume).
        return df_to_xlsx_response(
            s4, out4.name,
            extra_headers={
                "X-Run-Id": run_id,
                "X-Artifact-01": pub1,
                "X-Artifact-02": pub2,
                "X-Artifact-03": pub3,
                "X-Artifact-04": pub4,
                "X-Artifact-Zip": pubzip
            },
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
