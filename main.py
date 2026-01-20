import io, os, shutil, zipfile, re, secrets
from datetime import datetime, timedelta
import pytz
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ==========================================
#  AUTHENTICATION SETUP
# ==========================================
try:
    from auth import (
        get_current_user, 
        authenticate_user, 
        create_access_token, 
        get_password_hash,
        ACCESS_TOKEN_EXPIRE_MINUTES
    )
    from models import (
        UserRegister, UserLogin, Token, UserResponse, UserInDB,
        ActivityLog, UserStatsResponse, ActivityResponse, DailyActivityResponse,
        ReconciliationHistoryResponse, ReconciliationSummary,
        SendVerificationEmailRequest, VerifyEmailRequest
    )
    from database import users_collection, activity_logs_collection, reconciliation_results_collection, fs
    AUTH_ENABLED = True
except ImportError:
    print("[WARNING] Authentication modules not found. Running without authentication.")
    AUTH_ENABLED = False
    get_current_user = None

try:
    from email_utils import send_verification_email
    EMAIL_ENABLED = True
except ImportError:
    print("[Email] ⚠️ email_utils not found. Email verification disabled.")
    EMAIL_ENABLED = False

APP = FastAPI(title="Recon Backend v17.10 (Duplicate Header Fix)", version="1.10")

# ==========================================
#  CORS CONFIGURATION
# ==========================================
ALLOWED_ORIGINS_STR = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:3000,https://recondb.vercel.app,https://www.recowiz.in,http://www.recowiz.in,https://recowiz.in,http://recowiz.in,http://127.0.0.1:3000,http://127.0.0.1:8000"
)
ALLOWED_ORIGINS = [origin.strip().rstrip("/") for origin in ALLOWED_ORIGINS_STR.split(",") if origin.strip()]
print(f"[CORS] Configured Allowed Origins: {ALLOWED_ORIGINS}")

APP.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

BASE_DIR = Path.cwd()
RUN_ROOT = BASE_DIR / "runs"
RUN_ROOT.mkdir(parents=True, exist_ok=True)

CURRENT_RUN_DIR: Optional[Path] = None
CURRENT_BANK_TYPE: Optional[str] = None

# ==========================================
#  PART 1: CONFIG & REGISTRY
# ==========================================

CANON_COLS = ["Cheque/ NEFT/ UTR No.", "Claim No"]

TPA_MIS_MAPS = {
    "IHX (Original MIS)": {
        "Cheque/ NEFT/ UTR No.": "Cheque/ NEFT/ UTR No.",
        "Claim No": "Claim Number"
    },
    "CARE HEALTH INSURANCE LIMITED": {
        "Cheque/ NEFT/ UTR No.": "NEFT Number",
        "Claim No": "Claim Number"
    },
    "HEALTH INDIA INSURANCE TPA SERVICES PRIVATE LTD.": {
        "Cheque/ NEFT/ UTR No.": "utrnumber",
        "Claim No": "Claim Number"
    },
    "HERITAGE HEALTH INSURANCE TPA PRIVATE LIMITED": {
        "Cheque/ NEFT/ UTR No.": "UTR No",
        "Claim No": "REF_CCN"
    },
    "MEDSAVE HEALTHCARE TPA PVT LTD": {
        "Cheque/ NEFT/ UTR No.": "UTR/Chq No.",
        "Claim No": "Claim Number"
    },
    "PARAMOUNT HEALTHCARE": {
        "Cheque/ NEFT/ UTR No.": "UTR_NO",
        "Claim No": "Claim Number"
    },
    "STAR HEALTH & ALLIED HEALTH INSURANCE CO.LTD.": {
        "Cheque/ NEFT/ UTR No.": "UTR",
        "Claim No": "Claim ID"
    },
    "ADITYA BIRLA": {
        "Cheque/ NEFT/ UTR No.": "UTR Number",
        "Claim No": "Claim Number"
    },
    "FHPL": {
        "Cheque/ NEFT/ UTR No.": "Cheque/NEFT No",
        "Claim No": "Claim Id"
    },
    "GOOD HEALTH": {
        "Cheque/ NEFT/ UTR No.": "TRASACTION_NO",
        "Claim No": "CCN_NO"
    },
    "VOLO HEALTH INSURANCE TPA PVT.LTD (EWA) (Mail Extract)": {
        "Cheque/ NEFT/ UTR No.": "UTR Number",
        "Claim No": "Alternate Claim Id"
    }
}
TPA_CHOICES = list(TPA_MIS_MAPS.keys())

# ==========================================
#  V2 PIPELINE CONFIGURATION
# ==========================================

CANON_COLS_V2 = ["Cheque/ NEFT/ UTR No.", "Claim No"]

TPA_MIS_MAPS_V2 = {
    "IHX (Original MIS)": {
        "Cheque/ NEFT/ UTR No.": "Cheque/ NEFT/ UTR No.",
        "Claim No": "Claim Number"
    },
    "CARE HEALTH INSURANCE LIMITED": {
        "Cheque/ NEFT/ UTR No.": "Instrument/NEFT No",
        "Claim No": "Claim NO"
    },
    "HEALTH INDIA INSURANCE TPA SERVICES PRIVATE LTD.": {
        "Cheque/ NEFT/ UTR No.": "utrnumber",
        "Claim No": "CCN"
    },
    "HERITAGE HEALTH INSURANCE TPA PRIVATE LIMITED": {
        "Cheque/ NEFT/ UTR No.": "UTR No",
        "Claim No": "REF_CCN"
    },
    "MEDSAVE HEALTHCARE TPA PVT LTD": {
        "Cheque/ NEFT/ UTR No.": "UTR/Chq No.",
        "Claim No": "Claim Number"
    },
    "PARAMOUNT HEALTHCARE": {
        "Cheque/ NEFT/ UTR No.": "UTR_NO",
        "Claim No": "Claim Number"
    },
    "PARK MEDICLAIM INSURANCE TPA PRIVATE LIMITED": {
        "Cheque/ NEFT/ UTR No.": "Chq No",
        "Claim No": "Claim Number"
    },
    "SAFEWAY INSURANCE TPA PVT.LTD": {
        "Cheque/ NEFT/ UTR No.": "Chequeno",
        "Claim No": "ClaimNo"
    },
    "STAR HEALTH & ALLIED HEALTH INSURANCE CO.LTD.": {
        "Cheque/ NEFT/ UTR No.": "UTR",
        "Claim No": "Claim ID"
    },
    "ADITYA BIRLA": {
        "Cheque/ NEFT/ UTR No.": "UTR Number",
        "Claim No": "CLAIM NO."
    },
    "FHPL": {
        "Cheque/ NEFT/ UTR No.": "Cheque/NEFT No",
        "Claim No": "Claim Id"
    },
    "FUTURE GENERALI": {
        "Cheque/ NEFT/ UTR No.": "Cheque/Ref No. Insured",
        "Claim No": "Claim No."
    },
    "GOOD HEALTH": {
        "Cheque/ NEFT/ UTR No.": "TRASACTION_NO",
        "Claim No": "CCN_NO"
    },
    "VOLO HEALTH INSURANCE TPA PVT.LTD (EWA) (Mail Extract)": {
        "Cheque/ NEFT/ UTR No.": "UTR Number",
        "Claim No": "Alternate Claim Id"
    },

    # ---- NEW LOT (patched) ----
    "VIDAL": {
        "Cheque/ NEFT/ UTR No.": "Cheque Number",
        "Claim No": "Claim Number"
    },
    "SBI GENERAL": {
        "Cheque/ NEFT/ UTR No.": "Payment Transaction ID UTR",
        "Claim No": "Claim No"
    },
    "RELIANCE": {
        "Cheque/ NEFT/ UTR No.": "Map Cheque/Neft Number",
        "Claim No": "Claim Number"
    },
    "ICICI LOMBARD": {
        "Cheque/ NEFT/ UTR No.": "Claim-Cheque Number",
        "Claim No": "Claim Number"
    },
    "ERICSON": {
        "Cheque/ NEFT/ UTR No.": "UTRNo",
        "Claim No": "ClaimId"
    }
}


def _to_amt(x):
    try:
        s = str(x).replace(",", "").strip()
        if s in ["", "-", "ÃƒÆ’Ã†â€™️Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â "]: return np.nan
        if s.startswith("(") and s.endswith(")"): return -round(float(s[1:-1]), 2)
        return round(float(s), 2)
    except: return np.nan

_NULL_TOKENS = {"", "nan", "none", "null", "na", "n/a", "n\\a", "0", "0.0"}

def _clean_key_series(series: pd.Series) -> pd.Series:
    """
    Make sure stringified nulls do not become joinable keys.
    Returns a pandas Series where null-like tokens become real NaN.
    """
    s = series.copy()
    s = s.astype("string")  # pandas string dtype preserves NA
    s = s.str.replace("\xa0", " ", regex=False).str.strip()
    lower = s.str.lower()
    s = s.mask(lower.isin(_NULL_TOKENS), pd.NA)
    return s.astype(object).where(s.notna(), np.nan)

# ==========================================
#  PART 2: BANK & ADVANCE CLEANERS
# ==========================================

def clean_raw_bank_statement_icici(path):
    def _extract_table_from_raw(raw):
        if raw.dropna(how="all").empty: return None
        header_row_idx = None
        for i, row in raw.iterrows():
            vals = [str(v).strip() for v in row.tolist() if pd.notna(v)]
            if "Transaction ID" in vals and "Description" in vals:
                header_row_idx = i
                break
        if header_row_idx is None:
            if len(raw) > 6: header_row_idx = 6
            else: return None
        header_vals = [str(x).strip() for x in raw.iloc[header_row_idx].tolist()]
        df = raw.iloc[header_row_idx + 1:].copy()
        if df.shape[1] < len(header_vals):
            for _ in range(len(header_vals) - df.shape[1]): df[df.shape[1]] = np.nan
        df = df.iloc[:, :len(header_vals)]
        df.columns = header_vals
        # FIX: Drop duplicate columns immediately to prevent DataFrame/Series confusion
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    path = Path(path)
    tables = []
    try:
        xls = pd.ExcelFile(path)
        for sheet in xls.sheet_names:
            raw = xls.parse(sheet_name=sheet, header=None)
            tbl = _extract_table_from_raw(raw)
            if tbl is not None: tables.append(tbl)
    except:
        try:
            raw = pd.read_csv(path, header=None)
            tbl = _extract_table_from_raw(raw)
            if tbl is not None: tables.append(tbl)
        except: pass

    if not tables:
        raise ValueError("ICICI bank parser: No usable transaction table found.")
    df_all = pd.concat(tables, ignore_index=True)
    
    keep_cols = ["No.", "Transaction ID", "Txn Posted Date", "Description", "Cr/Dr", "Transaction Amount(INR)"]
    existing = [c for c in keep_cols if c in df_all.columns]
    df = df_all[existing].copy().dropna(how="all")
    
    if not df.empty:
        def _is_repeated_header(row):
            for c in existing:
                if str(row[c]).strip() != c: return False
            return True
        df = df[~df.apply(_is_repeated_header, axis=1)]

    if "Description" in df.columns:
        desc = df["Description"].astype(str).str.upper()
        pattern = r"TOTAL|OPENING BALANCE|CLOSING BALANCE|BALANCE BROUGHT FORWARD"
        df = df[~desc.str.contains(pattern, regex=True, na=False)]

    for col in ["No.", "Transaction Amount(INR)"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="ignore")
    return df.drop_duplicates().reset_index(drop=True)

def clean_raw_bank_statement_axis(path):
    def _extract_table_from_raw(raw: pd.DataFrame):
        if raw.dropna(how="all").empty: return None
        header_row_idx = None
        for i, row in raw.iterrows():
            vals = [str(v).strip() for v in row.tolist() if pd.notna(v)]
            upper_vals = [v.upper() for v in vals]
            if (any(v == "S.NO" for v in upper_vals) and 
                any("TRANSACTION DATE" in v for v in upper_vals) and 
                any("PARTICULARS" in v for v in upper_vals)):
                header_row_idx = i
                break
        if header_row_idx is None: return None
        header_vals = [str(x).strip() for x in raw.iloc[header_row_idx].tolist()]
        df = raw.iloc[header_row_idx + 1:].copy()
        if df.shape[1] < len(header_vals):
            for _ in range(len(header_vals) - df.shape[1]):
                df[df.shape[1]] = np.nan
        df = df.iloc[:, :len(header_vals)]
        df.columns = header_vals
        # FIX: Drop duplicate columns immediately to prevent DataFrame/Series confusion
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    path = Path(path)
    tables = []
    try:
        xls = pd.ExcelFile(path)
        for sheet in xls.sheet_names:
            raw = xls.parse(sheet_name=sheet, header=None)
            tbl = _extract_table_from_raw(raw)
            if tbl is not None: tables.append(tbl)
    except:
        try:
            raw = pd.read_csv(path, header=None)
            tbl = _extract_table_from_raw(raw)
            if tbl is not None: tables.append(tbl)
        except: pass

    if not tables:
        raise ValueError("Axis bank parser: No usable transaction table found.")
    df_all = pd.concat(tables, ignore_index=True)

    orig_to_canon = {
        "S.No": "No.", "Transaction Date (dd/mm/yyyy)": "Txn Posted Date",
        "Value Date (dd/mm/yyyy)": "Value Date", "Particulars": "Description",
        "Amount(INR)": "Transaction Amount(INR)", "Transaction Amount(INR)": "Transaction Amount(INR)",
        "Debit/Credit": "Cr/Dr", "Balance(INR)": "Balance(INR)",
        "Cheque Number": "Cheque Number", "Branch Name(SOL)": "Branch Name(SOL)"
    }
    canon_cols_order = ["No.", "Txn Posted Date", "Value Date", "Description", "Cr/Dr", 
                       "Transaction Amount(INR)", "Balance(INR)", "Cheque Number", "Branch Name(SOL)"]
    
    col_map = {c: orig_to_canon[c] for c in df_all.columns if c in orig_to_canon}
    canon_df = pd.DataFrame()
    for canon in canon_cols_order:
        sources = [src for src, tgt in col_map.items() if tgt == canon]
        canon_df[canon] = df_all[sources[0]] if sources else np.nan

    canon_df = canon_df.dropna(how="all")
    if "Description" in canon_df.columns:
        desc_upper = canon_df["Description"].astype(str).str.upper()
        pattern = r"OPENING BALANCE|CLOSING BALANCE|BALANCE BROUGHT FORWARD|TOTAL"
        canon_df = canon_df[~desc_upper.str.contains(pattern, regex=True, na=False)]

    for col in ["Transaction Amount(INR)", "Balance(INR)"]:
        if col in canon_df.columns:
            canon_df[col] = canon_df[col].astype(str).str.replace(",", "", regex=False).str.strip()
            canon_df[col] = pd.to_numeric(canon_df[col], errors="ignore")
            
    return canon_df.drop_duplicates().reset_index(drop=True)

def read_clean_icici_advance_excel(xlsx_path: Path, deduplicate: bool = True) -> pd.DataFrame:
    def _extract_table_from_raw(raw: pd.DataFrame):
        if raw.dropna(how="all").empty: return None
        header_row_idx = None
        for i, row in raw.iterrows():
            vals = [str(v).strip() for v in row.tolist() if pd.notna(v)]
            upper_vals = [v.upper() for v in vals]
            if (any(v == "MSG REFER.NO" for v in upper_vals) and any(v == "AMOUNT" for v in upper_vals)):
                header_row_idx = i
                break
        if header_row_idx is None: return None
        header_vals = [str(x).strip() for x in raw.iloc[header_row_idx].tolist()]
        df = raw.iloc[header_row_idx + 1:].copy()
        if df.shape[1] < len(header_vals):
            for _ in range(len(header_vals) - df.shape[1]): df[df.shape[1]] = np.nan
        df = df.iloc[:, :len(header_vals)]
        df.columns = header_vals
        # FIX: Drop duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    path = Path(xlsx_path)
    tables = []
    try:
        xls = pd.ExcelFile(path)
        for sheet in xls.sheet_names:
            raw = xls.parse(sheet_name=sheet, header=None, dtype=object)
            tbl = _extract_table_from_raw(raw)
            if tbl is not None: tables.append(tbl)
    except:
        try:
            raw = pd.read_csv(path, header=None)
            tbl = _extract_table_from_raw(raw)
            if tbl is not None: tables.append(tbl)
        except: pass

    if not tables:
        raise ValueError("ICICI Advance Excel parser: No usable data table found.")
    df = pd.concat(tables, ignore_index=True).dropna(how="all").reset_index(drop=True)
    
    df.columns = [str(c).strip().replace(".", "_").replace(" ", "_") for c in df.columns]
    
    def _cnorm(name): return re.sub(r"[^a-z0-9]", "", name.lower())
    refer_col, msg_col = None, None
    for c in df.columns:
        n = _cnorm(c)
        if refer_col is None and n in ("referno", "refno"): refer_col = c
        if msg_col is None and "msgreferno" in n: msg_col = c

    df["Refer_No"] = df[refer_col].astype(str) if refer_col else ""
    if msg_col: df["Msg_Refer_No"] = df[msg_col].astype(str).str.strip()
    elif "Msg_Refer_No" not in df.columns: df["Msg_Refer_No"] = df["Refer_No"]

    df["Refer_No_UTR"] = df["Refer_No"].apply(lambda x: str(x).replace("/XUTR/", "").replace("XUTR/", "").strip() if pd.notna(x) else "")
    df["Msg_Refer_No"] = _clean_key_series(df["Msg_Refer_No"])
    df["Refer_No_UTR"] = _clean_key_series(df["Refer_No_UTR"])
    
    if deduplicate and "Refer_No_UTR" in df.columns:
        df = df.drop_duplicates(subset=["Refer_No_UTR"], keep="last")
    return df.reset_index(drop=True)

def read_clean_axis_advance_excel(x_path, deduplicate: bool = True) -> pd.DataFrame:
    def _extract_table_from_raw(raw: pd.DataFrame):
        if raw.dropna(how="all").empty: return None
        header_row_idx = None
        for i, row in raw.iterrows():
            vals = [str(v).strip() for v in row.tolist() if pd.notna(v)]
            upper_vals = [v.upper() for v in vals]
            if (any(v == "TXN_AMOUNT_IN_RS" for v in upper_vals) and any(v == "TRANID" for v in upper_vals)):
                header_row_idx = i
                break
        if header_row_idx is None: return None
        header_vals = [str(x).strip() for x in raw.iloc[header_row_idx].tolist()]
        df = raw.iloc[header_row_idx + 1:].copy()
        if df.shape[1] < len(header_vals):
            for _ in range(len(header_vals) - df.shape[1]): df[df.shape[1]] = np.nan
        df = df.iloc[:, :len(header_vals)]
        df.columns = header_vals
        # FIX: Drop duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    path = Path(x_path)
    tables = []
    try:
        xls = pd.ExcelFile(path)
        for sheet in xls.sheet_names:
            raw = xls.parse(sheet_name=sheet, header=None, dtype=object)
            tbl = _extract_table_from_raw(raw)
            if tbl is not None: tables.append(tbl)
    except:
        try:
            raw = pd.read_csv(path, header=None)
            tbl = _extract_table_from_raw(raw)
            if tbl is not None: tables.append(tbl)
        except: pass

    if not tables:
        raise ValueError("Axis Advance Excel parser: No usable data table found.")
    df = pd.concat(tables, ignore_index=True).dropna(how="all").reset_index(drop=True)
    df.columns = [str(c).strip().replace(".", "_").replace(" ", "_") for c in df.columns]
    
    def _cnorm(name): return re.sub(r"[^a-z0-9]", "", name.lower())
    utr_col, tranid_col, primary_col = None, None, None
    for c in df.columns:
        n = _cnorm(c)
        if utr_col is None and n == "utr": utr_col = c
        if tranid_col is None and n == "tranid": tranid_col = c
        if primary_col is None and n == "primarykey": primary_col = c

    df["Msg_Refer_No"] = df[utr_col].astype(str).str.strip() if utr_col else ""
    df["Refer_No_UTR"] = df[tranid_col].astype(str).str.strip() if tranid_col else ""
    
    clean_u = lambda x: str(x).replace("/XUTR/", "").replace("XUTR/", "").strip() if pd.notna(x) else ""
    df["Msg_Refer_No"] = df["Msg_Refer_No"].apply(clean_u)
    df["Refer_No_UTR"] = df["Refer_No_UTR"].apply(clean_u)

    df["Msg_Refer_No"] = _clean_key_series(df["Msg_Refer_No"])
    df["Refer_No_UTR"] = _clean_key_series(df["Refer_No_UTR"])

    if deduplicate:
        if primary_col is not None:
            df[primary_col] = _clean_key_series(df[primary_col])
            df = df.drop_duplicates(subset=[primary_col], keep="last")
        elif "Refer_No_UTR" in df.columns:
            df = df.drop_duplicates(subset=["Refer_No_UTR"], keep="last")
    return df.reset_index(drop=True)

# ==========================================
#  PART 3: MATCHING & MIS PARSING
# ==========================================

def step2_match_bank_advance(bank_df: pd.DataFrame, adv_df: pd.DataFrame):
    if bank_df.empty: return pd.DataFrame(), bank_df
    parts = []
    if "Msg_Refer_No" not in adv_df.columns and "Refer_No_UTR" in adv_df.columns:
        adv_df = adv_df.copy()
        adv_df["Msg_Refer_No"] = adv_df["Refer_No_UTR"]
    
    adv_df = adv_df.copy()
    if "Msg_Refer_No" in adv_df.columns: adv_df["Msg_Refer_No"] = _clean_key_series(adv_df["Msg_Refer_No"])
    if "Refer_No_UTR" in adv_df.columns: adv_df["Refer_No_UTR"] = _clean_key_series(adv_df["Refer_No_UTR"])

    col_to_search = "Description" if "Description" in bank_df.columns else bank_df.columns[1]
    keys = pd.Series(adv_df["Msg_Refer_No"]).dropna().astype(str).map(str.strip).unique()

    for msg in keys:
        s = msg.strip()
        if not s: continue
        # This check confirms we are searching in a Series, not a DataFrame
        m = bank_df[col_to_search].astype(str).str.contains(s, regex=False, na=False)
        if m.any():
            t = bank_df.loc[m].copy()
            t["Matched_Key"] = s
            parts.append(t)
            
    if not parts: return pd.DataFrame(), bank_df
    
    # Enforce 1-to-1: Deduplicate matches on Bank side and Advance side before merging
    bank_matched = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["Matched_Key"])
    # ALLOW multiple claims per UTR: Deduplicate on UTR + Claim keys
    # Note: 'Msg_Refer_No' is used as UTR, 'Refer_No_UTR' is used as Claim/Reference
    cols = adv_df.columns
    dedupe_subset = ["Msg_Refer_No"]
    if "Refer_No_UTR" in cols: dedupe_subset.append("Refer_No_UTR")
    
    adv_dedup = adv_df.drop_duplicates(subset=dedupe_subset)

    matched = bank_matched.merge(
        adv_dedup, left_on="Matched_Key", right_on="Msg_Refer_No", how="inner", suffixes=("_bank", "_adv")
    )
    # .drop_duplicates() matched is now redundant but kept for safety if needed, though uniqueness is now guaranteed by keys
    matched = matched.drop_duplicates()

    if "Transaction ID" in bank_df.columns and "Transaction ID" in matched.columns:
        not_in = bank_df.loc[~bank_df["Transaction ID"].isin(matched["Transaction ID"])]
    else:
        not_in = bank_df.loc[~bank_df["Description"].isin(matched["Description"])]
    return matched, not_in

def parse_mis_universal(mis_path, tpa_name: str, empty_threshold: float = 0.5) -> pd.DataFrame:
    mis_path = Path(mis_path)
    if tpa_name not in TPA_MIS_MAPS: 
        raise ValueError(f"Unknown TPA '{tpa_name}'")
    mapping = TPA_MIS_MAPS[tpa_name]
    expected_sources = list(mapping.values())
    
    def _norm_local(s): return re.sub(r"[^A-Z0-9]", "", str(s).upper())
    expected_norm = {_norm_local(x) for x in expected_sources + list(mapping.keys())}
    
    def _hunt_header(raw):
        best_idx, best_score = None, -1
        for i in range(min(80, len(raw))):
            row_vals = [str(v).strip() for v in raw.iloc[i].tolist()]
            hits = sum(1 for v in row_vals if _norm_local(v) in expected_norm)
            if hits > best_score: best_idx, best_score = i, hits
        return best_idx if best_idx is not None and best_score >= 1 else None

    def _extract_from_sheet(raw):
        if raw.dropna(how="all").empty: return None
        header_idx = _hunt_header(raw)
        if header_idx is None: return None
        header_vals = [str(x).strip() for x in raw.iloc[header_idx].tolist()]
        df = raw.iloc[header_idx + 1:].copy()
        if df.shape[1] < len(header_vals):
            for _ in range(len(header_vals) - df.shape[1]): df[df.shape[1]] = np.nan
        df = df.iloc[:, :len(header_vals)]
        df.columns = header_vals
        # FIX: Drop duplicates
        df = df.loc[:, ~df.columns.duplicated()]
        return df.dropna(how="all").reset_index(drop=True)

    tables = []
    try:
        xls = pd.ExcelFile(mis_path)
        for sh in xls.sheet_names:
            tbl = _extract_from_sheet(xls.parse(sh, header=None, dtype=str))
            if tbl is not None: tables.append(tbl)
    except Exception as e:
        try:
            tbl = _extract_from_sheet(pd.read_csv(mis_path, header=None, dtype=str))
            if tbl is not None: tables.append(tbl)
        except: pass
        
    if not tables: 
        raise ValueError(f"Universal MIS parser: No table detected for '{tpa_name}'")
    return pd.concat(tables, ignore_index=True).reset_index(drop=True)

def step3_map_to_mis(step2_df: pd.DataFrame, mis_df: pd.DataFrame, tpa_name: str, deduplicate: bool = True):
    if step2_df.empty: return pd.DataFrame()
    mapping = TPA_MIS_MAPS.get(tpa_name)
    mis_std = mis_df.rename(columns={v: k for k, v in mapping.items()}).copy()
    
    if "Cheque/ NEFT/ UTR No." not in mis_std.columns:
        raise ValueError(f"MIS mapping failed: UTR column not found.")
    
    s2 = step2_df.copy()
    s2["Refer_No_UTR"] = _clean_key_series(s2["Refer_No_UTR"])
    mis_std["Cheque/ NEFT/ UTR No."] = _clean_key_series(mis_std["Cheque/ NEFT/ UTR No."])
    if "Claim No" in mis_std.columns: mis_std["Claim No"] = _clean_key_series(mis_std["Claim No"])
    
    s2 = s2.dropna(subset=["Refer_No_UTR"])
    mis_std = mis_std.dropna(subset=["Cheque/ NEFT/ UTR No."])
    
    # Enforce 1-to-1 logic but preserve multi-claim UTRs
    # s2 (Bank+Advance) might have multple rows per UTR (different claims)
    # merged against MIS (UTR).
    
    # For s2: Relax deduplication to include Claim ID if possible, or just drop perfect duplicates
    # s2 uses 'Refer_No_UTR' as join key (presumably UTR in this context per existing logic)
    # But usually 'Refer_No_UTR' is Claim. If logic treats it as UTR, we must respect it.
    # To be safe and preserve multiple rows, we drop exact duplicates only if we can't be sure of Claim col.
    # The user instruction is (UTR, Claim No).
    s2 = s2.drop_duplicates() 

    # For MIS: Deduplicate on UTR + Claim No
    mis_dedup_cols = ["Cheque/ NEFT/ UTR No."]
    if "Claim No" in mis_std.columns: mis_dedup_cols.append("Claim No")
    mis_std = mis_std.drop_duplicates(subset=mis_dedup_cols)

    merged = s2.merge(
        mis_std, left_on="Refer_No_UTR", right_on="Cheque/ NEFT/ UTR No.", how="inner", suffixes=("", "_mis")
    )
    return merged.drop_duplicates() if deduplicate else merged

# ==========================================
#  PART 4: OUTSTANDING & HELPERS
# ==========================================

_FOOTER_TOKENS = ["SUB TOTAL", "SUBTOTAL", "TOTAL", "GRAND TOTAL"]
_OUT_HEADER_KEYS = ["Sl No", "Bill No", "Date", "CR No", "Patient Name", "Net Amount", 
                   "Amount Paid", "TDS", "Write-Off", "Balance", "Location", "Consultant", "Claim No", "Insurance Company"]

def parse_outstanding_excel_to_clean(xlsx_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, header=None, dtype=str)
    
    def _hunt_header(df_raw):
        norm = lambda s: re.sub(r"[^A-Z]", "", str(s).upper())
        targets = [norm(k) for k in _OUT_HEADER_KEYS]
        best_idx, best_hits = None, -1
        for i in range(min(len(df_raw), 60)):
            row = df_raw.iloc[i].fillna("").astype(str).tolist()
            hits = sum(1 for c in row if norm(c) in targets or (norm(c).startswith("CONSUL") and "CONSULTANT" in targets))
            if hits > best_hits: best_idx, best_hits = i, hits
        if best_idx is None or best_hits <= 2: raise ValueError("Outstanding parser: Header not found.")
        return best_idx

    hdr_idx = _hunt_header(raw)
    header_vals = [str(x).strip() for x in raw.iloc[hdr_idx].tolist()]
    header_norm = []
    for h in header_vals:
        hn = str(h).strip()
        if re.sub(r"[^A-Za-z]", "", hn).lower().startswith("consul"): hn = "Consultant"
        if hn.strip().lower() == "insurance companies": hn = "Insurance Company"
        header_norm.append(hn)
        
    df = raw.iloc[hdr_idx + 1:].copy()
    if df.shape[1] < len(header_norm):
        for _ in range(len(header_norm) - df.shape[1]): df[df.shape[1]] = ""
    df = df.iloc[:, :len(header_norm)]
    df.columns = header_norm
    # FIX: Drop duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    ent_col, current_entity, header_rows = [], "", []
    for idx, row in df.iterrows():
        vals = row.fillna("").astype(str).tolist()
        if vals[0].strip() and all(not str(v).strip() for v in vals[1:]):
            current_entity = vals[0].replace("\xa0", " ").strip()
            header_rows.append(idx)
        ent_col.append(current_entity)
    df["Insurance Company Automated"] = ent_col
    if header_rows: df = df.drop(index=header_rows)
    
    df = df[~df.apply(lambda r: any(t in " ".join([str(x) for x in r]).upper() for t in _FOOTER_TOKENS), axis=1)]
    for c in df.columns: df[c] = df[c].astype(str).str.replace("\xa0", " ", regex=False).str.strip()
    return df.reset_index(drop=True)

def step4_strict_matches(step3_df: pd.DataFrame, outstanding_path: Path, deduplicate: bool = True):
    if step3_df.empty: return pd.DataFrame(), pd.DataFrame()
    if "Claim No" not in step3_df.columns: raise ValueError("Step 3 missing Claim No")
    
    out = parse_outstanding_excel_to_clean(outstanding_path).copy()
    if "Claim No" in out.columns: out = out.rename(columns={"Claim No": "Claim No_out"})
    
    def _clean_claim(val):
        if pd.isna(val): return ""
        s = str(val).strip()
        if s.lower() in ['nan', 'none', 'null', 'na', 'n/a', '']: return ""
        if s.endswith(".0"): s = s[:-2]
        return s

    L, R = out.copy(), step3_df.copy()
    L["_CLAIM_KEY"] = L.get("Claim No_out", "").apply(_clean_claim)
    R["_CLAIM_KEY"] = R["Claim No"].apply(_clean_claim)
    
    # Enforce 1-to-1
    L_in = L[L["_CLAIM_KEY"] != ""].drop_duplicates(subset=["_CLAIM_KEY"])
    R_in = R[R["_CLAIM_KEY"] != ""].drop_duplicates(subset=["_CLAIM_KEY"])

    matched_merged = L_in.merge(R_in, on="_CLAIM_KEY", how="inner", suffixes=("_out", "_m3"))
    matched = matched_merged.drop(columns=["_CLAIM_KEY"])
    if deduplicate: matched = matched.drop_duplicates()
    
    matched_keys = matched_merged["_CLAIM_KEY"].unique()
    unmatched_step3 = R[~R["_CLAIM_KEY"].isin(matched_keys)].copy()
    unmatched_step3 = unmatched_step3.drop(columns=["_CLAIM_KEY"])
    if deduplicate: unmatched_step3 = unmatched_step3.drop_duplicates()
    
    return matched, unmatched_step3

# ==========================================
#  PART 5: V2 PIPELINE HELPERS
# ==========================================

def _clean_key_value(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).replace("\xa0", " ").strip()
    if s.lower() in _NULL_TOKENS:
        return None
    return s

def clean_raw_bank_statement_icici_v2(path):
    """
    ICICI bank statement parser (Excel/CSV) with:
      - .xls, .xlsx, .xlsm, .csv support
      - Multi-sheet handling (skip empty sheets, append valid ones)
      - Remove repeated headers + TOTAL/OPENING/CLOSING rows
      - Keeps original ICICI schema for compatibility
    """

    def _extract_table_from_raw(raw):
        if raw.dropna(how="all").empty:
            return None

        header_row_idx = None
        for i, row in raw.iterrows():
            vals = [str(v).strip() for v in row.tolist() if pd.notna(v)]
            if "Transaction ID" in vals and "Description" in vals:
                header_row_idx = i
                break

        if header_row_idx is None:
            if len(raw) > 6:
                header_row_idx = 6
            else:
                return None

        header_vals = [str(x).strip() for x in raw.iloc[header_row_idx].tolist()]
        df = raw.iloc[header_row_idx + 1:].copy()

        if df.shape[1] < len(header_vals):
            for _ in range(len(header_vals) - df.shape[1]):
                df[df.shape[1]] = np.nan

        df = df.iloc[:, :len(header_vals)]
        df.columns = header_vals
        return df

    path = Path(path)
    ext = path.suffix.lower()
    tables = []

    if ext in [".xlsx", ".xls", ".xlsm"]:
        xls = pd.ExcelFile(path)
        for sheet in xls.sheet_names:
            raw = xls.parse(sheet_name=sheet, header=None)
            tbl = _extract_table_from_raw(raw)
            if tbl is not None:
                tables.append(tbl)

    elif ext == ".csv":
        raw = pd.read_csv(path, header=None)
        tbl = _extract_table_from_raw(raw)
        if tbl is not None:
            tables.append(tbl)

    else:
        try:
            raw = pd.read_excel(path, header=None)
        except:
            raw = pd.read_csv(path, header=None)
        tbl = _extract_table_from_raw(raw)
        if tbl is not None:
            tables.append(tbl)

    if not tables:
        raise ValueError("ICICI bank parser: No usable transaction table found.")

    df_all = pd.concat(tables, ignore_index=True)

    keep_cols = [
        "No.",
        "Transaction ID",
        "Txn Posted Date",
        "Description",
        "Cr/Dr",
        "Transaction Amount(INR)",
    ]
    existing = [c for c in keep_cols if c in df_all.columns]
    df = df_all[existing].copy()

    df = df.dropna(how="all")

    if not df.empty:
        def _is_repeated_header(row):
            for c in existing:
                if str(row[c]).strip() != c:
                    return False
            return True
        df = df[~df.apply(_is_repeated_header, axis=1)]

    if "Description" in df.columns:
        desc = df["Description"].astype(str).str.upper()
        pattern = (
            r"TOTAL|OPENING BALANCE|CLOSING BALANCE|"
            r"BALANCE BROUGHT FORWARD|BALANCE CARRIED FORWARD"
        )
        df = df[~desc.str.contains(pattern, regex=True, na=False)]

    for col in ["No.", "Transaction Amount(INR)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    return df.drop_duplicates().reset_index(drop=True)

    if not tables:
        raise ValueError("ICICI bank parser: No usable transaction table found.")

    df_all = pd.concat(tables, ignore_index=True)

    keep_cols = [
        "No.",
        "Transaction ID",
        "Txn Posted Date",
        "Description",
        "Cr/Dr",
        "Transaction Amount(INR)",
    ]
    existing = [c for c in keep_cols if c in df_all.columns]
    df = df_all[existing].copy()
    df = df.dropna(how="all")

    if not df.empty:
        def _is_repeated_header(row):
            for c in existing:
                if str(row[c]).strip() != c:
                    return False
            return True
        df = df[~df.apply(_is_repeated_header, axis=1)]

    if "Description" in df.columns:
        desc = df["Description"].astype(str).str.upper()
        pattern = (
            r"TOTAL|OPENING BALANCE|CLOSING BALANCE|"
            r"BALANCE BROUGHT FORWARD|BALANCE CARRIED FORWARD"
        )
        df = df[~desc.str.contains(pattern, regex=True, na=False)]

    for col in ["No.", "Transaction Amount(INR)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    return df.drop_duplicates().reset_index(drop=True)

def clean_raw_bank_statement_axis_v2(path):
    """
    AXIS bank statement parser (Excel/CSV) with:
      - .xls, .xlsx, .xlsm, .csv support
      - Multi-sheet handling (skip empty sheets, append valid ones)
      - Detects AXIS header row (S.No / Transaction Date / Particulars / Amount(INR) / Debit/Credit)
      - Drops 'OPENING BALANCE', 'CLOSING BALANCE', 'TRANSACTION TOTAL', etc.
      - Drops footer/disclaimer rows (no date, no description, no amount)
      - Returns canonical schema
    """

    def _extract_table_from_raw(raw: pd.DataFrame):
        if raw.dropna(how="all").empty:
            return None

        header_row_idx = None
        for i, row in raw.iterrows():
            vals = [str(v).strip() for v in row.tolist() if pd.notna(v)]
            upper_vals = [v.upper() for v in vals]

            if (
                any(v == "S.NO" for v in upper_vals)
                and any("TRANSACTION DATE" in v for v in upper_vals)
                and any("PARTICULARS" in v for v in upper_vals)
                and any("AMOUNT(INR)" in v or "TRANSACTION AMOUNT" in v for v in upper_vals)
                and any("DEBIT/CREDIT" in v for v in upper_vals)
            ):
                header_row_idx = i
                break

        if header_row_idx is None:
            return None

        header_vals = [str(x).strip() for x in raw.iloc[header_row_idx].tolist()]
        df = raw.iloc[header_row_idx + 1:].copy()

        if df.shape[1] < len(header_vals):
            for _ in range(len(header_vals) - df.shape[1]):
                df[df.shape[1]] = np.nan

        df = df.iloc[:, :len(header_vals)]
        df.columns = header_vals
        return df

    path = Path(path)
    ext = path.suffix.lower()
    tables = []

    if ext in [".xlsx", ".xls", ".xlsm"]:
        xls = pd.ExcelFile(path)
        for sheet in xls.sheet_names:
            raw = xls.parse(sheet_name=sheet, header=None)
            tbl = _extract_table_from_raw(raw)
            if tbl is not None:
                tables.append(tbl)

    elif ext == ".csv":
        raw = pd.read_csv(path, header=None)
        tbl = _extract_table_from_raw(raw)
        if tbl is not None:
            tables.append(tbl)

    else:
        try:
            raw = pd.read_excel(path, header=None)
        except Exception:
            raw = pd.read_csv(path, header=None)
        tbl = _extract_table_from_raw(raw)
        if tbl is not None:
            tables.append(tbl)

    if not tables:
        raise ValueError("Axis bank parser: No usable transaction table found.")

    df_all = pd.concat(tables, ignore_index=True)

    orig_to_canon = {
        "S.No": "No.",
        "Transaction Date (dd/mm/yyyy)": "Txn Posted Date",
        "Value Date (dd/mm/yyyy)": "Value Date",
        "Particulars": "Description",
        "Amount(INR)": "Transaction Amount(INR)",
        "Transaction Amount(INR)": "Transaction Amount(INR)",
        "Debit/Credit": "Cr/Dr",
        "Balance(INR)": "Balance(INR)",
        "Cheque Number": "Cheque Number",
        "Branch Name(SOL)": "Branch Name(SOL)",
    }

    canon_cols_order = [
        "No.",
        "Txn Posted Date",
        "Value Date",
        "Description",
        "Cr/Dr",
        "Transaction Amount(INR)",
        "Balance(INR)",
        "Cheque Number",
        "Branch Name(SOL)",
    ]

    col_map = {}
    for col in df_all.columns:
        if col in orig_to_canon:
            col_map[col] = orig_to_canon[col]

    canon_df = pd.DataFrame()
    for canon in canon_cols_order:
        sources = [src for src, tgt in col_map.items() if tgt == canon]
        if sources:
            canon_df[canon] = df_all[sources[0]]
        else:
            canon_df[canon] = np.nan

    canon_df = canon_df.dropna(how="all")

    def _is_repeated_header(row):
        for c in canon_cols_order:
            val = str(row[c]).strip()
            if val == "" and canon_df[c].isna().all():
                continue
            if val == c:
                continue
            else:
                return False
        return True

    if not canon_df.empty:
        mask_header = canon_df.apply(_is_repeated_header, axis=1)
        canon_df = canon_df[~mask_header]

    if "Description" in canon_df.columns:
        desc_upper = canon_df["Description"].astype(str).str.upper()
        pattern = (
            r"OPENING BALANCE|CLOSING BALANCE|"
            r"BALANCE BROUGHT FORWARD|BALANCE CARRIED FORWARD|TRANSACTION TOTAL|TOTAL"
        )
        canon_df = canon_df[~desc_upper.str.contains(pattern, regex=True, na=False)]

    def _empty(x):
        s = str(x).strip().lower()
        return s in ("", "nan", "none")

    def _is_noise(row):
        return (
            _empty(row.get("Txn Posted Date"))
            and _empty(row.get("Description"))
            and _empty(row.get("Transaction Amount(INR)"))
        )

    if not canon_df.empty:
        noise_mask = canon_df.apply(_is_noise, axis=1)
        canon_df = canon_df[~noise_mask]

    for col in ["Transaction Amount(INR)", "Balance(INR)"]:
        if col in canon_df.columns:
            canon_df[col] = (
                canon_df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            canon_df[col] = pd.to_numeric(canon_df[col], errors="ignore")

    if "No." in canon_df.columns:
        canon_df["No."] = pd.to_numeric(
            canon_df["No."].astype(str).str.strip(), errors="ignore"
        )

    canon_df = canon_df.drop_duplicates().reset_index(drop=True)
    return canon_df

def detect_bank_type(bank_path: Path) -> str:
    """
    Auto-detect ICICI vs AXIS BEFORE parsing.

    ICICI signature: 'Transaction ID' + 'Description'
    AXIS signature: 'S.No' + 'Particulars' + 'Debit/Credit' (+ Transaction Date/Amount variants)
    """
    bank_path = Path(bank_path)
    ext = bank_path.suffix.lower()

    def _row_tokens(row) -> set:
        toks = set()
        for v in row:
            if pd.isna(v):
                continue
            s = str(v).strip()
            if s:
                toks.add(s)
                toks.add(s.upper())
        return toks

    def _scan_raw(raw: pd.DataFrame) -> Optional[str]:
        n = min(len(raw), 120)
        for i in range(n):
            tokens = _row_tokens(raw.iloc[i].tolist())

            if ("Transaction ID" in tokens) and ("Description" in tokens):
                return "ICICI"

            up = {t.upper() for t in tokens}
            axis_hit = (
                ("S.NO" in up) and
                ("PARTICULARS" in up) and
                ("DEBIT/CREDIT" in up)
            )
            if axis_hit:
                return "AXIS"

            axis_hit2 = (
                ("S.NO" in up) and
                ("PARTICULARS" in up) and
                ("DEBIT/CREDIT" in up) and
                (any("TRANSACTION DATE" in t for t in up)) and
                (any("AMOUNT(INR)" in t or "TRANSACTION AMOUNT" in t for t in up))
            )
            if axis_hit2:
                return "AXIS"

        return None

    if ext in [".xlsx", ".xls", ".xlsm"]:
        xls = pd.ExcelFile(bank_path)
        for sh in xls.sheet_names:
            raw = xls.parse(sh, header=None)
            bt = _scan_raw(raw)
            if bt:
                return bt

    elif ext == ".csv":
        raw = pd.read_csv(bank_path, header=None)
        bt = _scan_raw(raw)
        if bt:
            return bt

    else:
        try:
            raw = pd.read_excel(bank_path, header=None)
        except Exception:
            raw = pd.read_csv(bank_path, header=None)
        bt = _scan_raw(raw)
        if bt:
            return bt

    raise ValueError(
        "Could not auto-detect bank type from file headers. "
        "Expected ICICI headers like 'Transaction ID' + 'Description' "
        "or AXIS headers like 'S.No' + 'Particulars' + 'Debit/Credit'."
    )

def detect_tpa_choice(mis_path: Path, scan_rows: int = 120) -> str:
    """
    PATCH #2:
    Detect using the UNIQUE pair of RHS columns (UTR RHS + Claim RHS).
    If the pair matches multiple TPAs -> FAIL FAST.
    If no match -> FAIL FAST.
    Also tolerates 2-row headers by considering row_i + row_{i+1}.
    """
    mis_path = Path(mis_path)
    ext = mis_path.suffix.lower()

    def _norm_hdr(x: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", str(x).upper()).strip()

    def _best_header_tokens_for_raw(raw: pd.DataFrame) -> Optional[set]:
        best_i, best_hits = None, -1
        n = min(len(raw), scan_rows)

        for i in range(n):
            row1 = raw.iloc[i].tolist()
            row2 = raw.iloc[i + 1].tolist() if i + 1 < len(raw) else []
            toks = []
            toks.extend(row1)
            toks.extend(row2)
            tokens = {_norm_hdr(v) for v in toks if pd.notna(v) and str(v).strip() != ""}

            hits = 0
            for t in tokens:
                if t:
                    hits += 1
            if hits > best_hits:
                best_hits = hits
                best_i = i

        if best_i is None:
            return None

        row1 = raw.iloc[best_i].tolist()
        row2 = raw.iloc[best_i + 1].tolist() if best_i + 1 < len(raw) else []
        toks = []
        toks.extend(row1)
        toks.extend(row2)
        tokens = {_norm_hdr(v) for v in toks if pd.notna(v) and str(v).strip() != ""}
        if not tokens:
            return None
        return tokens

    raws = []
    if ext in [".xlsx", ".xls", ".xlsm"]:
        xls = pd.ExcelFile(mis_path)
        for sh in xls.sheet_names:
            raws.append(xls.parse(sh, header=None, dtype=str))
    elif ext == ".csv":
        raws.append(pd.read_csv(mis_path, header=None, dtype=str))
    else:
        try:
            raws.append(pd.read_excel(mis_path, header=None, dtype=str))
        except Exception:
            raws.append(pd.read_csv(mis_path, header=None, dtype=str))

    header_tokens = None
    best_token_count = -1
    for raw in raws:
        tokens = _best_header_tokens_for_raw(raw)
        if tokens and len(tokens) > best_token_count:
            header_tokens = tokens
            best_token_count = len(tokens)

    if not header_tokens:
        raise ValueError("TPA auto-detect failed: could not locate any recognizable header row.")

    candidates = []
    for tpa_name, mp in TPA_MIS_MAPS_V2.items():
        rhs_utr = _norm_hdr(mp["Cheque/ NEFT/ UTR No."])
        rhs_clm = _norm_hdr(mp["Claim No"])
        if (rhs_utr in header_tokens) and (rhs_clm in header_tokens):
            candidates.append(tpa_name)

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:
        raise ValueError(
            "TPA auto-detect failed: no TPA matched the unique RHS pair "
            "(UTR RHS + Claim RHS) in the detected header."
        )

    raise ValueError(
        "TPA auto-detect failed: ambiguous. The same RHS pair matched multiple TPAs: "
        f"{candidates}"
    )

def parse_mis_universal_v2(mis_path, tpa_name: str, empty_threshold: float = 0.5) -> pd.DataFrame:
    """
    Universal MIS ingestion parser (TPA-aware):
    - Uses TPA_MIS_MAPS[tpa_name] to drive header detection
    - Returns MIS with ORIGINAL source column names preserved.

    PATCH #2 (minimal):
    - Tolerate 2-row headers by scoring row_i + row_{i+1}
    - Build header by filling blanks from the next row (only fill; no concatenation)
    """
    mis_path = Path(mis_path)
    ext = mis_path.suffix.lower()

    if tpa_name not in TPA_MIS_MAPS_V2:
        raise ValueError(f"Unknown TPA '{tpa_name}' for MIS parsing.")

    mapping: Dict[str, str] = TPA_MIS_MAPS_V2[tpa_name]
    expected_sources = list(mapping.values())
    canonical_names = list(mapping.keys())

    def _norm_local(s: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", str(s).upper())

    expected_norm = {_norm_local(x) for x in expected_sources + canonical_names}

    def _hunt_header(raw: pd.DataFrame) -> Optional[int]:
        best_idx, best_score = None, -999
        for i in range(min(80, len(raw))):
            row1 = raw.iloc[i].tolist()
            row2 = raw.iloc[i + 1].tolist() if i + 1 < len(raw) else []
            merged = []
            merged.extend(row1)
            merged.extend(row2)

            hits = 0
            for v in merged:
                if _norm_local(v) in expected_norm:
                    hits += 1

            if hits > best_score:
                best_idx, best_score = i, hits

        if best_idx is None or best_score < 1:
            return None
        return best_idx

    def _row_is_noise(row: pd.Series, important_idx, empty_threshold: float) -> bool:
        vals = [str(v).strip() for v in row.tolist()]
        n_cols = len(vals)
        empties = sum(v.lower() in ("", "nan", "none") for v in vals)
        empty_ratio = empties / max(1, n_cols)

        if empty_ratio < empty_threshold:
            return False

        if important_idx:
            if all(vals[i].lower() in ("", "nan", "none") for i in important_idx):
                return True
            return False

        return True

    def _extract_from_sheet(raw: pd.DataFrame) -> Optional[pd.DataFrame]:
        if raw.dropna(how="all").empty:
            return None

        header_idx = _hunt_header(raw)
        if header_idx is None:
            return None

        header_row_1 = raw.iloc[header_idx].tolist()
        header_row_2 = raw.iloc[header_idx + 1].tolist() if header_idx + 1 < len(raw) else []

        header_vals = []
        for j in range(max(len(header_row_1), len(header_row_2))):
            h1 = str(header_row_1[j]).strip() if j < len(header_row_1) and pd.notna(header_row_1[j]) else ""
            h2 = str(header_row_2[j]).strip() if j < len(header_row_2) and pd.notna(header_row_2[j]) else ""
            if h1 and h1.lower() not in ("nan", "none"):
                header_vals.append(h1)
            else:
                header_vals.append(h2)

        df_sheet = raw.iloc[header_idx + 1:].copy()

        if df_sheet.shape[1] < len(header_vals):
            for _ in range(len(header_vals) - df_sheet.shape[1]):
                df_sheet[df_sheet.shape[1]] = np.nan
        df_sheet = df_sheet.iloc[:, :len(header_vals)]
        df_sheet.columns = header_vals

        def _is_header_like(row: pd.Series) -> bool:
            vals = [str(x).strip() for x in row.tolist()]
            return vals[:len(header_vals)] == header_vals

        df_sheet = df_sheet[~df_sheet.apply(_is_header_like, axis=1)]
        df_sheet = df_sheet.dropna(how="all")

        for c in df_sheet.columns:
            df_sheet[c] = (
                df_sheet[c]
                .astype(str)
                .str.replace("\xa0", " ", regex=False)
                .str.strip()
            )

        important_idx = []
        for i, col in enumerate(df_sheet.columns):
            if col in expected_sources or col in canonical_names:
                important_idx.append(i)

        noise_mask = df_sheet.apply(
            lambda r: _row_is_noise(r, important_idx, empty_threshold),
            axis=1
        )
        df_sheet = df_sheet[~noise_mask]

        return df_sheet.reset_index(drop=True)

    tables = []

    if ext in [".xlsx", ".xls", ".xlsm"]:
        xls = pd.ExcelFile(mis_path)
        for sh in xls.sheet_names:
            raw = xls.parse(sh, header=None, dtype=str)
            tbl = _extract_from_sheet(raw)
            if tbl is not None:
                tables.append(tbl)
    
    elif ext == ".csv":
        raw = pd.read_csv(mis_path, header=None, dtype=str)
        tbl = _extract_from_sheet(raw)
        if tbl is not None:
            tables.append(tbl)

    else:
        try:
            raw = pd.read_excel(mis_path, header=None, dtype=str)
        except Exception:
            raw = pd.read_csv(mis_path, header=None, dtype=str)
        tbl = _extract_from_sheet(raw)
        if tbl is not None:
            tables.append(tbl)

    if not tables:
        raise ValueError("Universal MIS parser: No table detected for this TPA in any sheet.")

    final = pd.concat(tables, ignore_index=True)
    return final.reset_index(drop=True)

def step2_match_bank_mis_by_utr_v2(bank_df: pd.DataFrame, mis_df: pd.DataFrame, tpa_name: str, deduplicate: bool = True, min_key_len: int = 8):
    """
    PATCH #1:
    - Build UNIQUE PAIRS of (UTR, Claim No) from MIS (NOT just unique UTR list)

    PATCH #3:
    - Ignore null-like tokens incl 0/0.0 (done in cleaner)
    - Minimum UTR string length guard (default 8)

    NEW (as discussed):
    - If MIS UTR length > 16, use only first 16 chars for bank substring search + merge.
      Example: AXISCN1218291971UTIBN62026011379400525 -> AXISCN1218291971
    - MIS-side only: original UTR preserved in output; _UTR_SEARCH used for matching.
    """
    if bank_df.empty:
        return pd.DataFrame(), bank_df
    if mis_df.empty:
        return pd.DataFrame(), bank_df

    mapping = TPA_MIS_MAPS_V2.get(tpa_name)
    if mapping is None:
        raise ValueError(f"Unknown TPA '{tpa_name}' for MIS mapping.")

    mis_std = mis_df.rename(columns={v: k for k, v in mapping.items()}).copy()

    if "Cheque/ NEFT/ UTR No." not in mis_std.columns:
        raise ValueError(
            f"MIS mapping failed: could not find UTR column after rename for TPA '{tpa_name}'. "
            f"Expected source column '{mapping.get('Cheque/ NEFT/ UTR No.')}'."
        )

    mis_std["Cheque/ NEFT/ UTR No."] = _clean_key_series(mis_std["Cheque/ NEFT/ UTR No."])
    if "Claim No" in mis_std.columns:
        mis_std["Claim No"] = _clean_key_series(mis_std["Claim No"])

    # NEW: MIS-side search key (truncate ONLY if >16)
    mis_std["_UTR_SEARCH"] = mis_std["Cheque/ NEFT/ UTR No."].astype("string").str.strip()
    mis_std["_UTR_SEARCH"] = mis_std["_UTR_SEARCH"].apply(
        lambda s: (str(s)[:16] if (pd.notna(s) and len(str(s)) > 16) else (str(s) if pd.notna(s) else np.nan))
    )
    mis_std["_UTR_SEARCH"] = _clean_key_series(mis_std["_UTR_SEARCH"])

    # Build unique (UTR_SEARCH, Claim) pairs
    if "Claim No" in mis_std.columns:
        pairs = mis_std[["_UTR_SEARCH", "Claim No"]].copy()
        pairs = pairs.dropna(subset=["_UTR_SEARCH"])
        pairs["_UTR_SEARCH"] = pairs["_UTR_SEARCH"].astype(str).map(str.strip)
        pairs["Claim No"] = pairs["Claim No"].astype(str).map(str.strip)
        pairs = pairs.drop_duplicates()
    else:
        pairs = mis_std[["_UTR_SEARCH"]].copy()
        pairs = pairs.dropna(subset=["_UTR_SEARCH"])
        pairs["_UTR_SEARCH"] = pairs["_UTR_SEARCH"].astype(str).map(str.strip)
        pairs = pairs.drop_duplicates()

    # Min length guard
    pairs = pairs[pairs["_UTR_SEARCH"].map(lambda x: len(str(x)) >= int(min_key_len))]

    col_to_search = "Description" if "Description" in bank_df.columns else bank_df.columns[1]
    bank_text = bank_df[col_to_search].astype(str)

    # We only need to substring-search each key once; the merge will explode to all claims safely.
    utr_keys = pairs["_UTR_SEARCH"].dropna().astype(str).map(str.strip).unique()
    
    parts = []
    for utr in utr_keys:
        s = utr.strip()
        if not s:
            continue

        m = bank_text.str.contains(s, regex=False, na=False)
        if m.any():
            t = bank_df.loc[m].copy()
            t["Matched_Key"] = s
            parts.append(t)

    if not parts:
        return pd.DataFrame(), bank_df

    matched_bank = pd.concat(parts, ignore_index=True)

    merged = matched_bank.merge(
        mis_std,
        left_on="Matched_Key",
        right_on="_UTR_SEARCH",
        how="inner",
        suffixes=("_bank", "_mis"),
    )

    if deduplicate:
        merged = merged.drop_duplicates()

    if "Transaction ID" in bank_df.columns and "Transaction ID" in merged.columns:
        not_in = bank_df.loc[~bank_df["Transaction ID"].isin(merged["Transaction ID"])]
    else:
        not_in = bank_df.loc[~bank_df[col_to_search].isin(merged[col_to_search])]

    return merged.reset_index(drop=True), not_in

def parse_outstanding_excel_to_clean_v2(xlsx_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, header=None, dtype=str)

    def _hunt_header_v2(df_raw):
        norm = lambda s: re.sub(r"[^A-Z]", "", str(s).upper())
        targets = [norm(k) for k in _OUT_HEADER_KEYS]
        best_idx, best_hits = None, -1
        for i in range(min(len(df_raw), 60)):
            row = df_raw.iloc[i].fillna("").astype(str).tolist()
            hits = 0
            for cell in row:
                nc = norm(cell)
                if nc in targets or (nc.startswith("CONSUL") and "CONSULTANT" in targets):
                    hits += 1
            if best_hits is None or hits > best_hits:
                best_idx, best_hits = i, hits
        if best_idx is None or best_hits <= 2:
            raise ValueError("Outstanding parser V2: Header not found.")
        return best_idx

    def _row_is_section_header_like(row: pd.Series) -> bool:
        vals = row.fillna("").astype(str).tolist()
        if not vals:
            return False
        col0 = vals[0].strip()
        others_empty = all((str(v).strip() == "") for v in vals[1:])
        return bool(col0) and others_empty

    def _clean_entity_name(text: str) -> str:
        s = (text or "").replace("\xa0", " ").strip()
        s = re.sub(r"^\s*\d+\s*", "", s)
        return re.sub(r"\s+", " ", s)

    hdr_idx = _hunt_header_v2(raw)
    header_vals = [str(x).strip() for x in raw.iloc[hdr_idx].tolist()]
    header_norm = []
    for h in header_vals:
        hn = str(h).strip()
        if re.sub(r"[^A-Za-z]", "", hn).lower().startswith("consul"):
            hn = "Consultant"
        if hn.strip().lower() == "insurance companies":
            hn = "Insurance Company"
        header_norm.append(hn)

    df = raw.iloc[hdr_idx + 1:].copy()
    if df.shape[1] < len(header_norm):
        for _ in range(len(header_norm) - df.shape[1]):
            df[df.shape[1]] = ""
    df = df.iloc[:, :len(header_norm)]
    df.columns = header_norm

    ent_col, current_entity, header_rows = [], "", []
    for idx, row in df.iterrows():
        if _row_is_section_header_like(row):
            current_entity = _clean_entity_name(row.iloc[0])
            header_rows.append(idx)
        ent_col.append(current_entity)
    df["Bill Company Name"] = ent_col

    if header_rows:
        df = df.drop(index=header_rows)

    df = df[~df.apply(lambda r: any(t in " ".join([str(x) for x in r]).upper() for t in _FOOTER_TOKENS), axis=1)]
    for c in df.columns:
        df[c] = df[c].astype(str).str.replace("\xa0", " ", regex=False).str.strip()
    return df.reset_index(drop=True)

def step4_strict_matches_v2(step3_df: pd.DataFrame, outstanding_path: Optional[Path] = None, outstanding_df: Optional[pd.DataFrame] = None, deduplicate: bool = True):
    """
    ONLY NEW CHANGES (per your instruction):
    - Strip parentheses "(...)" if present before keying
    - Strip content after "." before keying
    - Restrict trailing suffix stripping to ONLY -0..-20 or _0.._20 (NOT large digit groups)

    FIX (this specific issue):
    - Numeric fallback must NOT merge digit groups by removing separators.
      Example: RC-HS25-15278172 must yield largest group = 15278172 (not 2515278172).
    """
    if step3_df.empty: return pd.DataFrame(), pd.DataFrame()
    if "Claim No" not in step3_df.columns: raise ValueError("Step 3 missing Claim No")

    # Load Outstanding
    if outstanding_df is not None:
        out = outstanding_df.copy()
    elif outstanding_path:
        out = parse_outstanding_excel_to_clean_v2(outstanding_path).copy()
    else:
        raise ValueError("Either outstanding_path or outstanding_df must be provided")

    if "Claim No" in out.columns:
        out = out.rename(columns={"Claim No": "Claim No_out"})

    for col in ["Patient Name", "CR No", "Balance"]:
        if col not in out.columns:
             # Just warn or skip if missing? The user script raises ValueError.
             # We'll allow it but logs might show issues.
             pass

    _SEP_RE = re.compile(r"[\s\-_]+")
    _NONALNUM_RE = re.compile(r"[^A-Z0-9]")
    _DIGITS_RE = re.compile(r"\d+")

    _PAREN_RE = re.compile(r"\([^)]*\)")
    _DOT_RE = re.compile(r"\..*$")
    _SUFFIX_RE = re.compile(r"([_-])(\d{1,2})$")  # candidate small suffix only

    def _strip_trailing_suffix(s: str) -> str:
        m = _SUFFIX_RE.search(s)
        if not m:
            return s
        n = int(m.group(2))
        if 0 <= n <= 20:
            return s[:m.start()]
        return s

    def _md_canon(s: str) -> str:
        if s.startswith("MDI1"):
            return "MD1" + s[4:]
        if s.startswith("MDI"):
            return "MD1" + s[3:]
        return s

    def _alpha_key(val) -> str:
        if pd.isna(val):
            return ""
        s = str(val).replace("\xa0", " ").strip().upper()
        if s.lower() in ['nan', 'none', 'null', 'na', 'n/a', '']:
            return ""
        if s.endswith(".0"):
            s = s[:-2]

        s = _PAREN_RE.sub("", s)
        s = _DOT_RE.sub("", s)
        s = _strip_trailing_suffix(s)

        s = _SEP_RE.sub("", s)
        s = _md_canon(s)
        s = _NONALNUM_RE.sub("", s)
        return s

    # FIXED: do NOT remove separators before digit extraction (prevents HS25 + 15278172 collapsing)
    def _largest_digit_group(val):
        """
        Returns (largest_digit_group, ambiguous_flag).
        ambiguous_flag=True when there are 2+ digit groups of the same max length.
        """
        if pd.isna(val):
            return "", True
        s = str(val).replace("\xa0", " ").strip().upper()
        if s.lower() in ['nan', 'none', 'null', 'na', 'n/a', '']:
            return "", True
        if s.endswith(".0"):
            s = s[:-2]

        s = _PAREN_RE.sub("", s)
        s = _DOT_RE.sub("", s)
        s = _strip_trailing_suffix(s)

        # IMPORTANT: keep separators so digit groups stay separate (HS25 vs 15278172)
        groups = _DIGITS_RE.findall(s)
        if not groups:
            return "", True

        lengths = [len(g) for g in groups]
        max_len = max(lengths)
        best = [g for g in groups if len(g) == max_len]

        if len(best) != 1:
            return best[0], True
        return best[0], False

    def _out_numeric_tokens(val) -> list:
        if pd.isna(val):
            return []
        s = str(val).replace("\xa0", " ").strip().upper()
        if s.lower() in ['nan', 'none', 'null', 'na', 'n/a', '']:
            return []

        if s.endswith(".0"):
            s = s[:-2]

        s = _PAREN_RE.sub("", s)
        s = _DOT_RE.sub("", s)
        s = _strip_trailing_suffix(s)

        parts = [p.strip() for p in s.split("/") if p.strip()]
        tokens = []
        for p in parts:
            p2 = _SEP_RE.sub("", p)
            p2 = _NONALNUM_RE.sub("", p2)
            if p2.isdigit():
                tokens.append(p2)
        return tokens

    L = out.copy().reset_index(drop=False).rename(columns={"index": "_OUT_IDX"})
    R = step3_df.copy().reset_index(drop=True)

    R["_RID"] = np.arange(len(R))
    L["_ALPHA_KEY"] = L.get("Claim No_out", "").apply(_alpha_key)
    R["_ALPHA_KEY"] = R["Claim No"].apply(_alpha_key)

    L_alpha = L[L["_ALPHA_KEY"] != ""].copy()
    R_alpha = R[R["_ALPHA_KEY"] != ""].copy()

    merged_alpha = L_alpha.merge(
        R_alpha,
        on="_ALPHA_KEY",
        how="inner",
        suffixes=("_out", "_m3")
    )
    merged_alpha["MATCH_MODE"] = "ALPHA_STRICT"

    matched_alpha_keys = set(merged_alpha["_ALPHA_KEY"].unique().tolist())
    L_unmatched = L[~L["_ALPHA_KEY"].isin(matched_alpha_keys)].copy()

    MIN_NUM_LEN = 6

    ldg = R["Claim No"].apply(_largest_digit_group)
    R["_MIS_LDG"] = [x[0] for x in ldg]
    R["_MIS_LDG_AMB"] = [x[1] for x in ldg]
    R["_MIS_LDG_LEN"] = R["_MIS_LDG"].astype(str).map(len)

    R_num = R[(~R["_MIS_LDG_AMB"]) & (R["_MIS_LDG"] != "") & (R["_MIS_LDG_LEN"] >= MIN_NUM_LEN)].copy()

    token_rows = []
    for _, row in L_unmatched.iterrows():
        tokens = _out_numeric_tokens(row.get("Claim No_out", ""))
        if not tokens:
            continue
        tokens = [t for t in tokens if len(t) >= MIN_NUM_LEN]
        if not tokens:
            continue

        mode = "NUM_TOKEN" if len(tokens) > 1 else "NUM_LARGEST"
        for t in tokens:
            token_rows.append({
                "_OUT_IDX": row["_OUT_IDX"],
                "_OUT_TOKEN": t,
                "_TOKEN_MODE": mode
            })

    merged_num = pd.DataFrame()
    if token_rows and not R_num.empty:
        T = pd.DataFrame(token_rows)

        cand = T.merge(
            R_num[["_RID", "_MIS_LDG"]],
            left_on="_OUT_TOKEN",
            right_on="_MIS_LDG",
            how="inner"
        ).drop_duplicates()

        grp = cand.groupby("_OUT_IDX")["_RID"].nunique().reset_index(name="_N")
        ok_out = set(grp.loc[grp["_N"] == 1, "_OUT_IDX"].tolist())

        cand_ok = cand[cand["_OUT_IDX"].isin(ok_out)].copy()
        cand_ok = cand_ok.sort_values(["_OUT_IDX", "_RID"]).drop_duplicates(subset=["_OUT_IDX"])

        L_pick = L.merge(cand_ok[["_OUT_IDX", "_RID"]], on="_OUT_IDX", how="inner")
        L_pick = L_pick.merge(cand_ok[["_OUT_IDX", "_TOKEN_MODE"]], on="_OUT_IDX", how="left")

        merged_num = L_pick.merge(
            R,
            on="_RID",
            how="inner",
            suffixes=("_out", "_m3")
        )
        merged_num["MATCH_MODE"] = merged_num["_TOKEN_MODE"].astype(str)

    matched = pd.concat([merged_alpha, merged_num], ignore_index=True)

    if deduplicate and not matched.empty:
        matched = matched.drop_duplicates()

    matched_rids = set(matched["_RID"].unique().tolist()) if (not matched.empty and "_RID" in matched.columns) else set()
    unmatched_step3 = R[~R["_RID"].isin(matched_rids)].copy()

    drop_cols_R = [c for c in ["_RID", "_ALPHA_KEY", "_MIS_LDG", "_MIS_LDG_AMB", "_MIS_LDG_LEN"] if c in unmatched_step3.columns]
    if drop_cols_R:
        unmatched_step3 = unmatched_step3.drop(columns=drop_cols_R)

    drop_cols_matched = [c for c in ["_OUT_IDX", "_ALPHA_KEY", "_RID", "_TOKEN_MODE", "_OUT_TOKEN", "_MIS_LDG", "_MIS_LDG_AMB", "_MIS_LDG_LEN"] if c in matched.columns]
    if drop_cols_matched:
        matched = matched.drop(columns=drop_cols_matched)

    return matched.reset_index(drop=True), unmatched_step3.reset_index(drop=True)


def save_xlsx(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)
    return path

def new_run_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = RUN_ROOT / f"reco_outputs_{stamp}"
    shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)
    return p

def zip_outputs(paths, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            if p.exists(): z.write(p, p.name)

def log_activity(username, bank_type, step, run_id, duration=None, counts=None, tpa=None, success=True, error=None):
    if activity_logs_collection is not None:
        try:
            ist = pytz.timezone('Asia/Kolkata')
            ts = datetime.now(ist).replace(tzinfo=None)
            activity_logs_collection.insert_one({
                "username": username, "bank_type": bank_type, "step_completed": step, "run_id": run_id,
                "timestamp": ts, "duration_seconds": duration, "row_counts": counts or {},
                "tpa_name": tpa, "success": success, "error_message": error
            })
        except: pass

# ==================== API ENDPOINTS ====================

@APP.get("/")
async def root():
    return {"status": "ok", "version": "17.10+ExcelCore+Auth+DupFix", "auth_enabled": AUTH_ENABLED}

@APP.post("/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user: UserRegister):
    if not AUTH_ENABLED: raise HTTPException(503, "Authentication is not configured")
    if users_collection is None: raise HTTPException(503, "Database not connected")
    
    if users_collection.find_one({"username": user.username}):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Username already registered")
    
    user_dict = {
        "username": user.username, "email": user.email, "full_name": user.full_name,
        "hashed_password": get_password_hash(user.password), "created_at": datetime.now(pytz.timezone('Asia/Kolkata')).replace(tzinfo=None),
        "is_active": True, "email_verified": False, "verification_token": None
    }
    try:
        users_collection.insert_one(user_dict)
        return UserResponse(
            username=user_dict["username"], email=user_dict.get("email"),
            full_name=user_dict.get("full_name"), created_at=user_dict["created_at"],
            email_verified=user_dict.get("email_verified", False)
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to create user: {str(e)}")

@APP.post("/auth/login", response_model=Token)
async def login(username: str = Form(...), password: str = Form(...)):
    if not AUTH_ENABLED: raise HTTPException(503, "Authentication is not configured")
    user = authenticate_user(username, password)
    if not user: raise HTTPException(401, "Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
    token = create_access_token({"sub": user.username}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": token, "token_type": "bearer"}

@APP.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: UserInDB = Depends(get_current_user)):
    if not AUTH_ENABLED: raise HTTPException(503, "Auth disabled")
    return UserResponse(
        username=current_user.username, email=current_user.email,
        full_name=current_user.full_name, created_at=current_user.created_at,
        email_verified=current_user.email_verified
    )

@APP.get("/auth/verification-status")
async def get_verification_status(current_user: UserInDB = Depends(get_current_user)):
    if not AUTH_ENABLED: raise HTTPException(503, "Authentication is not configured")
    return {
        "username": current_user.username,
        "email": current_user.email,
        "email_verified": current_user.email_verified,
        "has_email": bool(current_user.email)
    }

@APP.post("/auth/send-verification-email")
async def send_verification_email_endpoint(current_user: UserInDB = Depends(get_current_user)):
    if not AUTH_ENABLED or users_collection is None: raise HTTPException(503, "Auth disabled")
    if not current_user.email: raise HTTPException(400, "No email address")
    if current_user.email_verified: return {"status": "already_verified"}
    
    token = secrets.token_urlsafe(32)
    users_collection.update_one(
        {"username": current_user.username},
        {"$set": {"verification_token": token, "verification_token_expires": datetime.now(pytz.timezone('Asia/Kolkata')).replace(tzinfo=None) + timedelta(hours=24)}}
    )
    
    verification_url = f"{os.getenv('FRONTEND_URL', 'https://recondb.vercel.app')}/auth/verify-email?token={token}"
    if EMAIL_ENABLED:
        if send_verification_email(current_user.email, current_user.username, verification_url):
            return {"status": "success", "message": f"Sent to {current_user.email}"}
    return {"status": "partial_success", "verification_url": verification_url}

@APP.post("/auth/verify-email")
async def verify_email_with_token(token: str = Form(...)):
    if not AUTH_ENABLED or users_collection is None: raise HTTPException(503, "Auth disabled")
    user = users_collection.find_one({"verification_token": token})
    if not user: raise HTTPException(400, "Invalid token")
    if user.get("verification_token_expires") and datetime.now(pytz.timezone('Asia/Kolkata')).replace(tzinfo=None) > user["verification_token_expires"]:
        raise HTTPException(400, "Token expired")
    
    users_collection.update_one(
        {"username": user["username"]},
        {"$set": {"email_verified": True}, "$unset": {"verification_token": "", "verification_token_expires": ""}}
    )
    return {"status": "success", "username": user["username"]}

@APP.post("/auth/change-password")
async def change_password(current_password: str = Form(...), new_password: str = Form(...), current_user: UserInDB = Depends(get_current_user)):
    if not AUTH_ENABLED or users_collection is None: raise HTTPException(503, "Auth disabled")
    try:
        from auth import verify_password
    except ImportError:
        raise HTTPException(500, "Password verification module missing in auth.py")

    if not verify_password(current_password, current_user.hashed_password): raise HTTPException(401, "Wrong password")
    if len(new_password) < 6: raise HTTPException(400, "Password too short")
    users_collection.update_one({"username": current_user.username}, {"$set": {"hashed_password": get_password_hash(new_password)}})
    return {"status": "success"}

@APP.get("/profile/stats", response_model=UserStatsResponse)
async def get_user_stats(current_user: UserInDB = Depends(get_current_user)):
    if not AUTH_ENABLED or activity_logs_collection is None: raise HTTPException(503, "Stats unavailable")
    activities = list(activity_logs_collection.find({"username": current_user.username, "success": True}, {"_id": 0}).sort("timestamp", -1))
    
    completed = [a for a in activities if a.get("step_completed") == 4]
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    def to_ist(d):
        if d.tzinfo is None:
            # Assume UTC if naive (legacy data)
            return pytz.utc.localize(d).astimezone(ist)
        return d.astimezone(ist)

    this_week = len([a for a in completed if to_ist(a["timestamp"]) >= now - timedelta(days=7)])
    this_month = len([a for a in completed if to_ist(a["timestamp"]) >= now - timedelta(days=30)])
    
    streak = 0
    if completed:
        dates = sorted(set(a["timestamp"].date() for a in completed), reverse=True)
        today, yesterday = now.date(), now.date() - timedelta(days=1)
        if dates[0] in [today, yesterday]:
            streak = 1
            curr = dates[0] - timedelta(days=1)
            for d in dates[1:]:
                if d == curr: streak += 1; curr -= timedelta(days=1)
                else: break

    return UserStatsResponse(
        username=current_user.username, total_reconciliations=len(completed),
        this_week=this_week, this_month=this_month, current_streak=streak,
        last_activity=activities[0]["timestamp"] if activities else None
    )

@APP.get("/profile/activity", response_model=List[ActivityResponse])
async def get_user_activity(current_user: UserInDB = Depends(get_current_user), limit: int = 10):
    if not AUTH_ENABLED or activity_logs_collection is None: raise HTTPException(503, "Unavailable")
    acts = activity_logs_collection.find({"username": current_user.username, "step_completed": 4, "success": True}, {"_id": 0, "username": 0}).sort("timestamp", -1).limit(limit)
    return [ActivityResponse(**a) for a in acts]

@APP.get("/profile/daily", response_model=List[DailyActivityResponse])
async def get_daily_activity(current_user: UserInDB = Depends(get_current_user), days: int = 30):
    if not AUTH_ENABLED or activity_logs_collection is None: raise HTTPException(503, "Unavailable")
    ist = pytz.timezone('Asia/Kolkata')
    start = datetime.now(ist) - timedelta(days=days)
    pipeline = [
        {"$match": {"username": current_user.username, "timestamp": {"$gte": start}, "step_completed": 4, "success": True}},
        {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp", "timezone": "Asia/Kolkata"}}, "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    results = {r["_id"]: r["count"] for r in activity_logs_collection.aggregate(pipeline)}
    now = datetime.now(ist).replace(tzinfo=None)
    return [DailyActivityResponse(date=(now - timedelta(days=days-i-1)).strftime("%Y-%m-%d"), count=results.get((now - timedelta(days=days-i-1)).strftime("%Y-%m-%d"), 0)) for i in range(days)]

@APP.get("/reconciliations/history")
async def get_reconciliation_history(current_user: UserInDB = Depends(get_current_user), limit: int = 50, skip: int = 0):
    if reconciliation_results_collection is None: raise HTTPException(503, "DB unavailable")
    
    results = list(reconciliation_results_collection.find({"username": current_user.username}, {"_id": 0}).sort("timestamp", -1).skip(skip).limit(limit))
    
    for r in results:
        if "summary" not in r: r["summary"] = {}
        s = r["summary"]
        if "total_amount" not in s and "total_value" in s: s["total_amount"] = s["total_value"]
        s["total_amount"] = float(s.get("total_amount", 0.0) or 0.0)
        s["step4_outstanding"] = int(s.get("step4_outstanding", 0) or 0)
        s["unique_patients"] = int(s.get("unique_patients", 0) or 0)
        s["step2_matches"] = int(s.get("step2_matches", 0) or 0)

    return results

@APP.get("/reconciliations/{run_id}/details")
async def get_reconciliation_details(run_id: str, current_user: UserInDB = Depends(get_current_user)):
    if reconciliation_results_collection is None: raise HTTPException(503, "DB unavailable")
    res = reconciliation_results_collection.find_one({"username": current_user.username, "run_id": run_id}, {"_id": 0})
    if not res: raise HTTPException(404, "Reconciliation not found")
    return res

@APP.get("/reconciliations/{run_id}/download-zip")
async def download_reconciliation_zip(run_id: str, current_user: UserInDB = Depends(get_current_user)):
    if reconciliation_results_collection is None or fs is None: raise HTTPException(503, "DB unavailable")
    from bson import ObjectId
    
    result = reconciliation_results_collection.find_one({"username": current_user.username, "run_id": run_id})
    if not result: raise HTTPException(404, "Reconciliation not found")
    if "zip_file_id" not in result: raise HTTPException(404, "ZIP file ID not found")

    try:
        zip_file = fs.get(ObjectId(result["zip_file_id"]))
        return StreamingResponse(
            zip_file, media_type="application/zip", 
            headers={"Content-Disposition": f"attachment; filename={run_id}.zip"}
        )
    except Exception as e:
        print(f"[Download ZIP] Error: {e}")
        raise HTTPException(500, "Failed to retrieve file")

@APP.delete("/reconciliations/{run_id}")
async def delete_reconciliation(run_id: str, current_user: UserInDB = Depends(get_current_user)):
    if reconciliation_results_collection is None or fs is None: raise HTTPException(503, "DB unavailable")
    from bson import ObjectId
    res = reconciliation_results_collection.find_one({"username": current_user.username, "run_id": run_id})
    if not res: raise HTTPException(404, "Not found")
    try:
        if "zip_file_id" in res: fs.delete(ObjectId(res["zip_file_id"]))
        reconciliation_results_collection.delete_one({"username": current_user.username, "run_id": run_id})
        return {"status": "success"}
    except Exception as e: raise HTTPException(500, str(e))

@APP.get("/download/{run_id}/{filename:path}")
async def download_file(run_id: str, filename: str, user: UserInDB = Depends(get_current_user) if AUTH_ENABLED else None):
    file_path = RUN_ROOT / run_id / filename
    if not file_path.exists(): raise HTTPException(404, "File not found")
    media = "application/zip" if filename.endswith(".zip") else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return StreamingResponse(open(file_path, "rb"), media_type=media, headers={"Content-Disposition": f"attachment; filename={file_path.name}"})

@APP.get("/tpa-choices")
async def get_tpa_choices(user: UserInDB = Depends(get_current_user) if AUTH_ENABLED else None):
    return {"tpa_choices": TPA_CHOICES}

# ----------------- RECONCILIATION FLOW -----------------

@APP.post("/reconcile/step1")
async def reconcile_step1(
    bank_type: str = Form(...),
    bank_file: UploadFile = File(...),
    advance_file: UploadFile = File(...),
    current_user: UserInDB = Depends(get_current_user) if AUTH_ENABLED else None
):
    """Step-1: Ingest Bank & Advance (Excel/CSV) -> 01_bank, 02_advance"""
    global CURRENT_RUN_DIR, CURRENT_BANK_TYPE
    try:
        CURRENT_RUN_DIR = new_run_dir()
        CURRENT_BANK_TYPE = bank_type
        run_id = CURRENT_RUN_DIR.name

        bank_path = CURRENT_RUN_DIR / "bank.xlsx"
        with open(bank_path, "wb") as f: f.write(await bank_file.read())
        
        adv_path = CURRENT_RUN_DIR / "advance.xlsx"
        with open(adv_path, "wb") as f: f.write(await advance_file.read())

        if bank_type == "ICICI":
            bank_df = clean_raw_bank_statement_icici(bank_path)
            adv_df = read_clean_icici_advance_excel(adv_path, deduplicate=True)
            out_bank = save_xlsx(bank_df, CURRENT_RUN_DIR / "01a_icici_bank_clean.xlsx")
            out_adv = save_xlsx(adv_df, CURRENT_RUN_DIR / "01b_icici_advance_clean.xlsx")
        elif bank_type == "AXIS":
            bank_df = clean_raw_bank_statement_axis(bank_path)
            adv_df = read_clean_axis_advance_excel(adv_path, deduplicate=True)
            out_bank = save_xlsx(bank_df, CURRENT_RUN_DIR / "01a_axis_bank_clean.xlsx")
            out_adv = save_xlsx(adv_df, CURRENT_RUN_DIR / "01b_axis_advance_clean.xlsx")
        else:
            raise HTTPException(400, "Invalid Bank Type. Only ICICI and AXIS are currently supported.")
        
        if current_user:
            log_activity(current_user.username, bank_type, 1, run_id, counts={"bank": len(bank_df), "adv": len(adv_df)})
        
        return {
            "status": "success", "run_id": run_id,
            "counts": {"bank_rows": len(bank_df), "advance_rows": len(adv_df)},
            "files": {
                "bank": f"/download/{run_id}/{out_bank.name}",
                "advance": f"/download/{run_id}/{out_adv.name}"
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(400, detail=str(e))

@APP.post("/reconcile/step2")
async def reconcile_step2(current_user: UserInDB = Depends(get_current_user) if AUTH_ENABLED else None):
    """Step-2: Match Bank & Advance -> 03a_mapped, 03b_notin"""
    global CURRENT_RUN_DIR
    if not CURRENT_RUN_DIR: raise HTTPException(400, "Run not initialized")
    try:
        run_id = CURRENT_RUN_DIR.name
        
        if CURRENT_BANK_TYPE == "ICICI":
            bank_path = CURRENT_RUN_DIR / "01a_icici_bank_clean.xlsx"
            adv_path = CURRENT_RUN_DIR / "01b_icici_advance_clean.xlsx"
        else:
            bank_path = CURRENT_RUN_DIR / "01a_axis_bank_clean.xlsx"
            adv_path = CURRENT_RUN_DIR / "01b_axis_advance_clean.xlsx"
        
        if not bank_path.exists() or not adv_path.exists():
            raise HTTPException(400, "Step 1 outputs missing")

        bank_df = pd.read_excel(bank_path, dtype=str)
        adv_df = pd.read_excel(adv_path, dtype=str)
        
        matched, not_in = step2_match_bank_advance(bank_df, adv_df)
        
        out_match = save_xlsx(matched, CURRENT_RUN_DIR / "02a_bank_x_advance_matches.xlsx")
        out_notin = save_xlsx(not_in, CURRENT_RUN_DIR / "02b_bank_not_in_advance.xlsx")
        
        if current_user:
            log_activity(current_user.username, CURRENT_BANK_TYPE, 2, run_id, counts={"matches": len(matched), "not_in": len(not_in)})

        return {
            "status": "success", "run_id": run_id,
            "counts": {"matches": len(matched), "not_in": len(not_in)},
            "files": {
                "matches": f"/download/{run_id}/{out_match.name}",
                "not_in": f"/download/{run_id}/{out_notin.name}"
            }
        }
    except Exception as e:
        raise HTTPException(400, detail=str(e))

@APP.post("/reconcile/step3")
async def reconcile_step3(
    tpa_name: str = Form(...),
    mis_file: UploadFile = File(...),
    current_user: UserInDB = Depends(get_current_user) if AUTH_ENABLED else None
):
    """Step-3: Parse MIS & Map -> 04_mis_clean, 05_mapped"""
    global CURRENT_RUN_DIR
    if not CURRENT_RUN_DIR: raise HTTPException(400, "Run not initialized")
    try:
        run_id = CURRENT_RUN_DIR.name
        mis_path = CURRENT_RUN_DIR / "mis.xlsx"
        with open(mis_path, "wb") as f: f.write(await mis_file.read())
        
        # 1. Clean MIS (New Logic)
        mis_clean = parse_mis_universal(mis_path, tpa_name)
        out_mis_clean = save_xlsx(mis_clean, CURRENT_RUN_DIR / "04_mis_cleaned.xlsx")
        
        # 2. Map Step 2 Matches to MIS
        s2_path = CURRENT_RUN_DIR / "02a_bank_x_advance_matches.xlsx"
        if not s2_path.exists(): raise HTTPException(400, "Step 2 output missing")
        
        s2_df = pd.read_excel(s2_path, dtype=str)
        s3_mapped = step3_map_to_mis(s2_df, mis_clean, tpa_name, deduplicate=True)
        out3 = save_xlsx(s3_mapped, CURRENT_RUN_DIR / "03_matches_mapped_to_mis.xlsx")
        
        if current_user:
            log_activity(current_user.username, CURRENT_BANK_TYPE, 3, run_id, tpa=tpa_name, counts={"mapped": len(s3_mapped)})

        return {
            "status": "success", "run_id": run_id, "tpa_name": tpa_name,
            "counts": {"rows": len(s3_mapped)},
            "files": {
                "mis_cleaned": f"/download/{run_id}/{out_mis_clean.name}",
                "mapped": f"/download/{run_id}/{out3.name}"
            }
        }
    except Exception as e:
        raise HTTPException(400, detail=str(e))

@APP.post("/reconcile/step4")
async def reconcile_step4(
    outstanding_file: UploadFile = File(...),
    current_user: UserInDB = Depends(get_current_user) if AUTH_ENABLED else None
):
    """Step-4: Parse Outstanding & Final Match (Claim No) -> 06_out_clean, 07_final"""
    global CURRENT_RUN_DIR, CURRENT_BANK_TYPE
    if not CURRENT_RUN_DIR: raise HTTPException(400, "Run not initialized")
    try:
        run_id = CURRENT_RUN_DIR.name
        out_path = CURRENT_RUN_DIR / "outstanding.xlsx"
        with open(out_path, "wb") as f: f.write(await outstanding_file.read())
        
        # 1. Clean Outstanding
        out_clean = parse_outstanding_excel_to_clean(out_path)
        out_clean_path = save_xlsx(out_clean, CURRENT_RUN_DIR / "06_outstanding_cleaned.xlsx")
        
        # 2. Final Strict Match
        s3_path = CURRENT_RUN_DIR / "03_matches_mapped_to_mis.xlsx"
        if not s3_path.exists(): raise HTTPException(400, "Step 3 output not found")
        
        s3_df = pd.read_excel(s3_path, dtype=str)
        final_matched, final_unmatched = step4_strict_matches(s3_df, out_path, deduplicate=True)
        
        out_final = save_xlsx(final_matched, CURRENT_RUN_DIR / "04_outstanding_matches.xlsx")
        out_unmatched = save_xlsx(final_unmatched, CURRENT_RUN_DIR / "07b_final_unmatched_in_outstanding.xlsx")
        
        # Zip all outputs
        all_outputs = sorted(list(CURRENT_RUN_DIR.glob("0*.xlsx")))
        zip_path = CURRENT_RUN_DIR / f"{run_id}.zip"
        zip_outputs(all_outputs, zip_path)
        
        # --- DB STORAGE LOGIC ---
        if current_user and reconciliation_results_collection is not None and fs is not None:
            try:
                with open(zip_path, "rb") as zf:
                    fid = fs.put(zf, filename=f"{run_id}.zip", username=current_user.username)
                
                tpa = "Unknown"
                if activity_logs_collection is not None:
                    last_act = activity_logs_collection.find_one(
                        {"username": current_user.username, "run_id": run_id, "step_completed": 3},
                        sort=[("timestamp", -1)]
                    )
                    if last_act and "tpa_name" in last_act:
                        tpa = last_act["tpa_name"]

                total_amount = 0.0
                if "Settled Amount" in final_matched.columns:
                    total_amount = final_matched["Settled Amount"].apply(_to_amt).sum()
                elif "Amount" in final_matched.columns:
                    total_amount = final_matched["Amount"].apply(_to_amt).sum()
                
                comp_counts = {"step4_outstanding": len(final_matched)}
                try:
                    # Step 1
                    s1b = CURRENT_RUN_DIR / (f"01a_{CURRENT_BANK_TYPE.lower()}_bank_clean.xlsx")
                    s1a = CURRENT_RUN_DIR / (f"01b_{CURRENT_BANK_TYPE.lower()}_advance_clean.xlsx")
                    if s1b.exists(): comp_counts["step1_bank_rows"] = len(pd.read_excel(s1b))
                    if s1a.exists(): comp_counts["step1_advance_rows"] = len(pd.read_excel(s1a))
                    # Step 2
                    s2_match = CURRENT_RUN_DIR / "02a_bank_x_advance_matches.xlsx"
                    s2_notin = CURRENT_RUN_DIR / "02b_bank_not_in_advance.xlsx"
                    if s2_match.exists(): comp_counts["step2_matches"] = len(pd.read_excel(s2_match))
                    if s2_notin.exists(): comp_counts["step2_not_in"] = len(pd.read_excel(s2_notin))
                    # Step 3
                    s3_mis = CURRENT_RUN_DIR / "03_matches_mapped_to_mis.xlsx"
                    if s3_mis.exists(): comp_counts["step3_mis_mapped"] = len(pd.read_excel(s3_mis))
                except: pass

                unique_patients = 0
                if "Patient Name" in final_matched.columns:
                    unique_patients = len(final_matched["Patient Name"].unique())

                ist = pytz.timezone('Asia/Kolkata')
                timestamp = datetime.now(ist).replace(tzinfo=None)

                rec_doc = {
                    "username": current_user.username,
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "created_at": timestamp,
                    "bank_type": CURRENT_BANK_TYPE or "Unknown",
                    "tpa_name": tpa,
                    "status": "Completed",
                    "summary": {
                        "step4_outstanding": int(len(final_matched)),
                        "total_amount": float(total_amount),
                        "unique_patients": int(unique_patients),
                        "step2_matches": int(comp_counts.get("step2_matches", 0)),
                        "step2_not_in": int(comp_counts.get("step2_not_in", 0)),
                        "step1_bank_rows": int(comp_counts.get("step1_bank_rows", 0)),
                        "step1_advance_rows": int(comp_counts.get("step1_advance_rows", 0)),
                        "step3_mis_mapped": int(comp_counts.get("step3_mis_mapped", 0)),
                        "unmatched_count": int(len(final_unmatched))
                    },
                    "zip_file_id": str(fid)
                }
                
                reconciliation_results_collection.insert_one(rec_doc)
                print(f"[DB] Saved history for {run_id}")

            except Exception as e:
                print(f"[DB Save Error] {str(e)}")

        if current_user:
            log_activity(current_user.username, CURRENT_BANK_TYPE, 4, run_id, counts={"final": len(final_matched)})

        return {
            "status": "success", 
            "run_id": run_id,
            "counts": {
                "rows": len(final_matched),
                "unmatched": len(final_unmatched)
            },
            "files": {
                "outstanding_cleaned": f"/download/{run_id}/{out_clean_path.name}",
                "final_matches": f"/download/{run_id}/{out_final.name}",
                "final_unmatched": f"/download/{run_id}/{out_unmatched.name}",
                "zip": f"/download/{run_id}/{zip_path.name}"
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(400, detail=str(e))

# ==========================================
#  V2 ENDPOINTS (New Pipeline)
# ==========================================

@APP.post("/reconcile/v2/step1")
async def reconcile_v2_step1(
    bank_file: UploadFile = File(...),
    current_user: UserInDB = Depends(get_current_user) if AUTH_ENABLED else None
):
    """V2 Step-1: Upload Bank Only -> Detect Type -> Clean -> 01_bank_clean"""
    global CURRENT_RUN_DIR, CURRENT_BANK_TYPE
    try:
        CURRENT_RUN_DIR = new_run_dir()
        run_id = CURRENT_RUN_DIR.name
        
        bank_path = CURRENT_RUN_DIR / "bank_raw.xlsx" # use generic name initially
        with open(bank_path, "wb") as f: f.write(await bank_file.read())
        
        # Detect Bank
        CURRENT_BANK_TYPE = detect_bank_type(bank_path)
        
        # Clean
        if CURRENT_BANK_TYPE == "ICICI":
            bank_df = clean_raw_bank_statement_icici_v2(bank_path)
        else:
            bank_df = clean_raw_bank_statement_axis_v2(bank_path)
            
        out_bank = save_xlsx(bank_df, CURRENT_RUN_DIR / "01_bank_clean.xlsx")
        
        if current_user:
            log_activity(current_user.username, CURRENT_BANK_TYPE, 1, run_id, 
                        counts={"bank": len(bank_df)}, success=True)
            
        return {
            "status": "success", "run_id": run_id, 
            "detected_bank_type": CURRENT_BANK_TYPE,
            "counts": {"bank_rows": len(bank_df)},
            "files": {"bank": f"/download/{run_id}/{out_bank.name}"}
        }
    except Exception as e:
        if current_user and CURRENT_RUN_DIR:
             log_activity(current_user.username, "Unknown", 1, CURRENT_RUN_DIR.name, success=False, error=str(e))
        import traceback
        traceback.print_exc()
        raise HTTPException(400, detail=str(e))

@APP.post("/reconcile/v2/step2")
async def reconcile_v2_step2(
    mis_file: UploadFile = File(...),
    current_user: UserInDB = Depends(get_current_user) if AUTH_ENABLED else None
):
    """V2 Step-2: Upload MIS -> Detect TPA -> Clean -> Match against Bank (from Step 1)"""
    global CURRENT_RUN_DIR, CURRENT_BANK_TYPE
    if not CURRENT_RUN_DIR: raise HTTPException(400, "Run not initialized")
    try:
        run_id = CURRENT_RUN_DIR.name
        mis_path = CURRENT_RUN_DIR / "mis_raw.xlsx"
        with open(mis_path, "wb") as f: f.write(await mis_file.read())
        
        # Detect TPA
        tpa_name = detect_tpa_choice(mis_path)
        
        # Clean MIS
        mis_clean = parse_mis_universal_v2(mis_path, tpa_name)
        out_mis = save_xlsx(mis_clean, CURRENT_RUN_DIR / "04_mis_cleaned.xlsx")
        
        # Match Bank <-> MIS
        # Need to load Bank from Step 1
        bank_path = CURRENT_RUN_DIR / "01_bank_clean.xlsx"
        if not bank_path.exists(): raise HTTPException(400, "Step 1 bank output missing")
        bank_df = pd.read_excel(bank_path, dtype=str)
        
        matched, not_in_bank = step2_match_bank_mis_by_utr_v2(bank_df, mis_clean, tpa_name)
        
        out_match = save_xlsx(matched, CURRENT_RUN_DIR / "05_bank_mis_mapped.xlsx")
        # Optional: Save not in bank if needed? The requirement says "Saves 04... and 05...".
        # But saving not_in_bank is usually good. I'll save it too but only return 04/05 as per common pattern.
        # Actually V1 returns both matches and not_in. I will assume saving it is helpful.
        out_notin = save_xlsx(not_in_bank, CURRENT_RUN_DIR / "05b_bank_not_in_mis.xlsx")
        
        if current_user:
            log_activity(current_user.username, CURRENT_BANK_TYPE, 2, run_id, 
                        tpa=tpa_name, counts={"matches": len(matched)}, success=True)
            
        return {
            "status": "success", "run_id": run_id, "detected_tpa": tpa_name,
            "counts": {"matches": len(matched), "mis_rows": len(mis_clean)},
            "files": {
                "mis_cleaned": f"/download/{run_id}/{out_mis.name}",
                "mapped": f"/download/{run_id}/{out_match.name}"
            }
        }
    except Exception as e:
        raise HTTPException(400, detail=str(e))

@APP.post("/reconcile/v2/step3")
async def reconcile_v2_step3(
    outstanding_file: UploadFile = File(...),
    current_user: UserInDB = Depends(get_current_user) if AUTH_ENABLED else None
):
    """V2 Step-3: Upload Outstanding -> Clean -> Match against Step 2 Mapped"""
    global CURRENT_RUN_DIR, CURRENT_BANK_TYPE
    if not CURRENT_RUN_DIR: raise HTTPException(400, "Run not initialized")
    try:
        run_id = CURRENT_RUN_DIR.name
        out_path = CURRENT_RUN_DIR / "outstanding_raw.xlsx"
        with open(out_path, "wb") as f: f.write(await outstanding_file.read())
        
        # Clean Outstanding
        out_clean = parse_outstanding_excel_to_clean_v2(out_path)
        save_xlsx(out_clean, CURRENT_RUN_DIR / "06_outstanding_cleaned.xlsx")
        
        # Match Step 2 (Mapped) <-> Outstanding
        s2_path = CURRENT_RUN_DIR / "05_bank_mis_mapped.xlsx"
        if not s2_path.exists(): raise HTTPException(400, "Step 2 mapped output missing")
        s2_df = pd.read_excel(s2_path, dtype=str)
        
        final_matched, final_unmatched = step4_strict_matches_v2(s2_df, out_path, deduplicate=True)
        
        out_final = save_xlsx(final_matched, CURRENT_RUN_DIR / "07_final_posting_sheet.xlsx")
        out_unmatched = save_xlsx(final_unmatched, CURRENT_RUN_DIR / "07b_final_unmatched.xlsx")
        
        # ZIP
        all_outputs = sorted(list(CURRENT_RUN_DIR.glob("0*.xlsx")))
        zip_path = CURRENT_RUN_DIR / f"{run_id}_v2.zip"
        zip_outputs(all_outputs, zip_path)
        
        # DB Logic
        fid = None
        if current_user and reconciliation_results_collection is not None and fs is not None:
            try:
                with open(zip_path, "rb") as zf:
                    fid = fs.put(zf, filename=f"{run_id}_v2.zip", username=current_user.username)
                
                # Fetch Bank Type & TPA from previous logs
                # Ensure bank_type is not just "Unknown" if global is lost
                bank_type = CURRENT_BANK_TYPE
                tpa = "Unknown"
                
                if activity_logs_collection is not None:
                    # Try to find bank_type from Step 1 log if CURRENT_BANK_TYPE is unset
                    if not bank_type:
                        step1_log = activity_logs_collection.find_one({"username": current_user.username, "run_id": run_id, "step_completed": 1})
                        if step1_log and "bank_type" in step1_log:
                             bank_type = step1_log["bank_type"]

                    # Try to find TPA from Step 2 log
                    step2_log = activity_logs_collection.find_one({"username": current_user.username, "run_id": run_id, "step_completed": 2})
                    if step2_log: tpa = step2_log.get("tpa_name", "Unknown")

                if not bank_type: bank_type = "Unknown"

                # Calculate Summary Stats for V1 compatibility
                total_amount = 0.0
                if "Transaction Amount(INR)_bank" in final_matched.columns:
                     total_amount = final_matched["Transaction Amount(INR)_bank"].apply(_to_amt).sum()

                # Gather counts
                comp_counts = {}
                try:
                    s2_mapped = CURRENT_RUN_DIR / "05_bank_mis_mapped.xlsx"
                    if s2_mapped.exists(): comp_counts["step2_matches"] = len(pd.read_excel(s2_mapped))
                    # Note: V2 Step 2 matches are conceptually similar to V1 Step 2 matches (Bank x Advance ? No, Bank x MIS)
                    # V2 pipeline is shorter: Bank -> Match MIS -> Match Outstanding.
                    # Mapping V2 stats to V1 keys for frontend compatibility:
                    # step2_matches -> V2 Step 2 (Bank x MIS) matches
                    # step4_outstanding -> V2 Step 3 (Final Matches)
                    comp_counts["step3_mis_mapped"] = comp_counts.get("step2_matches", 0) # Reuse
                except: pass

                unique_patients = 0
                if "Patient Name" in final_matched.columns:
                    unique_patients = len(final_matched["Patient Name"].unique())
                
                # IST Time
                ist = pytz.timezone('Asia/Kolkata')
                timestamp = datetime.now(ist).replace(tzinfo=None)

                rec_doc = {
                    "username": current_user.username,
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "created_at": timestamp, # redundancy
                    "bank_type": bank_type,
                    "tpa_name": tpa,
                    "status": "Completed",
                    "pipeline_mode": "v2",
                    "summary": {
                        "step4_outstanding": int(len(final_matched)),
                        "total_amount": float(total_amount),
                        "unique_patients": int(unique_patients),
                        "step2_matches": int(comp_counts.get("step2_matches", 0)),
                        "step2_not_in": 0, # Not tracked same way in V2
                        "step3_mis_mapped": int(comp_counts.get("step3_mis_mapped", 0)),
                        "unmatched_count": int(len(final_unmatched))
                    },
                    "zip_file_id": str(fid)
                }
                reconciliation_results_collection.insert_one(rec_doc)
            except Exception as e: print(f"DB Error: {e}")

        if current_user:
            log_activity(current_user.username, CURRENT_BANK_TYPE or "Unknown", 3, run_id, 
                        counts={"final": len(final_matched)}, success=True)
            
        return {
            "status": "success", "run_id": run_id,
            "counts": {"final_matches": len(final_matched), "unmatched": len(final_unmatched)},
            "files": {
                "final_sheet": f"/download/{run_id}/{out_final.name}",
                "zip": f"/download/{run_id}/{zip_path.name}"
            },
            "zip_url": f"/download/{run_id}/{zip_path.name}"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(400, detail=str(e))

@APP.post("/reconcile/v2/bulk")
async def reconcile_v2_bulk(
    bank_files: List[UploadFile] = File(...),
    mis_files: List[UploadFile] = File(...),
    outstanding_file: UploadFile = File(...),
    current_user: UserInDB = Depends(get_current_user) if AUTH_ENABLED else None
):
    """V2 Bulk: Process multiple Bank files x multiple MIS files against one Outstanding file"""
    global CURRENT_RUN_DIR
    try:
        CURRENT_RUN_DIR = new_run_dir()
        run_id = CURRENT_RUN_DIR.name
        
        # 1. Outstanding Processing (Once)
        out_path = CURRENT_RUN_DIR / "outstanding.xlsx"
        with open(out_path, "wb") as f: f.write(await outstanding_file.read())
        
        out_clean_df = parse_outstanding_excel_to_clean_v2(out_path)
        save_xlsx(out_clean_df, CURRENT_RUN_DIR / "00_outstanding_cleaned.xlsx")
        
        summary_rows = []
        consolidated_frames = [] # List to hold all final match DFs
        files_resp = {}  # Map of "filename" -> "download_url"

        # Stats Aggregation
        agg_stats = {
            "step1_bank_rows": 0,
            "step2_matches": 0,
            "step3_mis_mapped": 0,
            "step4_outstanding": 0, # Final Matches
            "total_amount": 0.0,
            "unique_patients_set": set() # Use set to track uniqueness across all
        }
        
        detected_banks = set()
        detected_tpas = set()

        # Helper to register file
        def _reg_file(abs_path: Path):
            rel_path = abs_path.relative_to(CURRENT_RUN_DIR)
            files_resp[str(rel_path)] = f"/download/{run_id}/{str(rel_path)}"
            return str(rel_path)
        
        # Register global files
        _reg_file(CURRENT_RUN_DIR / "00_outstanding_cleaned.xlsx")

        # Save MIS files locally first
        mis_paths = []
        for mf in mis_files:
            m_path = CURRENT_RUN_DIR / f"raw_mis_{mf.filename}"
            with open(m_path, "wb") as f: 
                mf.file.seek(0)
                f.write(mf.file.read())
            mis_paths.append((mf.filename, m_path))

        # 2. Iterate Banks
        for b_idx, bank_file in enumerate(bank_files):
            b_name = Path(bank_file.filename).stem
            b_path = CURRENT_RUN_DIR / f"bank_{b_idx}_{b_name}.xlsx"
            with open(b_path, "wb") as f: 
                bank_file.file.seek(0)
                f.write(bank_file.file.read())
            
            try:
                # Detect & Clean Bank
                b_type = detect_bank_type(b_path)
                detected_banks.add(b_type)
                
                if b_type == "ICICI":
                    bank_clean = clean_raw_bank_statement_icici_v2(b_path)
                else:
                    bank_clean = clean_raw_bank_statement_axis_v2(b_path)
                
                agg_stats["step1_bank_rows"] += len(bank_clean)

                # 3. Iterate MIS
                for m_filename, m_path in mis_paths:
                    m_stem = Path(m_filename).stem
                    pair_id = f"{b_name}_vs_{m_stem}"
                    pair_dir = CURRENT_RUN_DIR / pair_id
                    pair_dir.mkdir(exist_ok=True)
                    
                    try:
                        # Detect TPA & Clean
                        tpa = detect_tpa_choice(m_path)
                        detected_tpas.add(tpa)
                        
                        mis_clean_df = parse_mis_universal_v2(m_path, tpa)
                        
                        # Step 2 Match
                        step2_match, _ = step2_match_bank_mis_by_utr_v2(bank_clean, mis_clean_df, tpa)
                        agg_stats["step2_matches"] += len(step2_match)
                        agg_stats["step3_mis_mapped"] += len(step2_match) # Reuse for compat
                        
                        # Step 4 Match (Strict)
                        final_match, final_unmatched = step4_strict_matches_v2(step2_match, outstanding_df=out_clean_df)
                        
                        # Aggregate Final Stats
                        agg_stats["step4_outstanding"] += len(final_match)
                        
                        if "Transaction Amount(INR)_bank" in final_match.columns:
                             amt = final_match["Transaction Amount(INR)_bank"].apply(_to_amt).sum()
                             agg_stats["total_amount"] += amt
                        
                        if "Patient Name" in final_match.columns:
                            pats = final_match["Patient Name"].dropna().unique()
                            agg_stats["unique_patients_set"].update(pats)

                        # Logic to add to consolidated list
                        try:
                            final_match_copy = final_match.copy()
                            # Insert at position 0
                            final_match_copy.insert(0, "Pipeline_Source", f"{bank_file.filename} vs {m_filename}")
                            final_match_copy.insert(1, "Detected Bank", b_type)
                            final_match_copy.insert(2, "Detected TPA", tpa)
                            consolidated_frames.append(final_match_copy)
                        except Exception as ce:
                            print(f"[Consolidation] Error adding frame: {ce}")

                        # Save Outputs
                        p_bank = save_xlsx(bank_clean, pair_dir / "01_bank_clean.xlsx")
                        p_mis = save_xlsx(mis_clean_df, pair_dir / "02_mis_clean.xlsx")
                        p_s2 = save_xlsx(step2_match, pair_dir / "03_step2_matches.xlsx")
                        p_final = save_xlsx(final_match, pair_dir / "04_final_matches.xlsx")
                        p_unmatched = save_xlsx(final_unmatched, pair_dir / "05_final_unmatched.xlsx")
                        
                        # Register files for this pair
                        prod_files = {
                            "Bank Cleaned": _reg_file(p_bank),
                            "MIS Cleaned": _reg_file(p_mis),
                            "Step 2 Match": _reg_file(p_s2),
                            "Final Match": _reg_file(p_final),
                            "Final Unmatched": _reg_file(p_unmatched)
                        }

                        summary_rows.append({
                            "Bank File": bank_file.filename,
                            "MIS File": m_filename,
                            "Bank Type": b_type,
                            "TPA": tpa,
                            "Bank Rows": len(bank_clean),
                            "MIS Rows": len(mis_clean_df),
                            "Step 2 Match": len(step2_match),
                            "Final Match": len(final_match),
                            "Status": "Success",
                            "produced_files": prod_files
                        })
                        
                    except Exception as e:
                        summary_rows.append({
                            "Bank File": bank_file.filename,
                            "MIS File": m_filename,
                            "Error": str(e),
                            "Status": "Failed",
                            "produced_files": {}
                        })
            
            except Exception as e:
                 # Bank failure
                 summary_rows.append({
                    "Bank File": bank_file.filename,
                    "Error": f"Bank Processing Failed: {str(e)}",
                    "Status": "Failed",
                    "produced_files": {}
                })

        # BUILD THE CONSOLIDATED MASTER SHEET
        try:
            if consolidated_frames:
                master_df = pd.concat(consolidated_frames, ignore_index=True)
                master_path = CURRENT_RUN_DIR / "Consolidated_Matches.xlsx"
                save_xlsx(master_df, master_path)
                _reg_file(master_path)
            else:
                print("[Consolidation] No frames to concatenate.")
        except Exception as me:
            print(f"[Consolidation] Failed to create master sheet: {me}")

        # Save Summary
        pd.DataFrame(summary_rows).to_excel(CURRENT_RUN_DIR / "bulk_summary.xlsx", index=False)
        _reg_file(CURRENT_RUN_DIR / "bulk_summary.xlsx")
        
        # Create Zip Archive recursively (safe from recursion)
        zip_temp = RUN_ROOT / f"temp_bulk_{run_id}"
        shutil.make_archive(str(zip_temp), 'zip', CURRENT_RUN_DIR)
        final_zip_name = f"bulk_{run_id}.zip"
        shutil.move(str(zip_temp) + ".zip", CURRENT_RUN_DIR / final_zip_name)
        
        # Register Zip
        files_resp[final_zip_name] = f"/download/{run_id}/{final_zip_name}"

        # --- DB SAVING LOGIC (BULK) ---
        if current_user and reconciliation_results_collection is not None and fs is not None:
             try:
                # 1. Upload Zip to GridFS
                zip_path = CURRENT_RUN_DIR / final_zip_name
                with open(zip_path, "rb") as zf:
                    fid = fs.put(zf, filename=final_zip_name, username=current_user.username)
                
                # 2. Determine Bank Type & TPA
                final_bank_type = list(detected_banks)[0] if len(detected_banks) == 1 else "Multiple"
                if len(detected_banks) == 0: final_bank_type = "Unknown"
                
                final_tpa_name = list(detected_tpas)[0] if len(detected_tpas) == 1 else "Multiple"
                if len(detected_tpas) == 0: final_tpa_name = "Unknown"

                # 3. IST Time
                ist = pytz.timezone('Asia/Kolkata')
                timestamp = datetime.now(ist).replace(tzinfo=None)

                # 4. Insert Record
                rec_doc = {
                    "username": current_user.username,
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "created_at": timestamp,
                    "bank_type": final_bank_type,
                    "tpa_name": final_tpa_name,
                    "status": "Completed", 
                    "pipeline_mode": "v2_bulk",
                    "files": files_resp, # Save map of files
                    "summary": {
                        "step4_outstanding": int(agg_stats["step4_outstanding"]),
                        "total_amount": float(agg_stats["total_amount"]),
                        "unique_patients": int(len(agg_stats["unique_patients_set"])),
                        "step2_matches": int(agg_stats["step2_matches"]),
                        "step1_bank_rows": int(agg_stats["step1_bank_rows"]),
                        "step3_mis_mapped": int(agg_stats["step3_mis_mapped"])
                    },
                    "zip_file_id": str(fid)
                }
                reconciliation_results_collection.insert_one(rec_doc)
                print(f"[DB] Saved Bulk history for {run_id}")

             except Exception as e:
                 print(f"[DB Save Error Bulk] {str(e)}")
        
        if current_user:
            # Force log activity for Bulk run so it shows up in Profile
            log_activity(
                username=current_user.username, 
                bank_type=final_bank_type, 
                step=4, 
                run_id=run_id, 
                counts={"final": agg_stats["step4_outstanding"]}, 
                success=True,
                tpa=final_tpa_name
            )
        
        return {
            "status": "success",
            "run_id": run_id,
            "summary": summary_rows,
            "files": files_resp,
            "zip_url": f"/download/{run_id}/{final_zip_name}"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(400, detail=str(e))