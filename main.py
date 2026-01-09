import io, os, shutil, zipfile, re, secrets
from datetime import datetime, timedelta
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
ALLOWED_ORIGINS = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:3000,https://recondb.vercel.app,https://www.recowiz.in,http://www.recowiz.in,https://recowiz.in,http://recowiz.in,http://127.0.0.1:3000,http://127.0.0.1:8000"
).split(",")

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
    "PARK MEDICLAIM INSURANCE TPA PRIVATE LIMITED": {
        "Cheque/ NEFT/ UTR No.": "Chq No",
        "Claim No": "Claim Number"
    },
    "SAFEWAY INSURANCE TPA PVT.LTD": {
        "Cheque/ NEFT/ UTR No.": "'Chequeno",
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
    }
}


def _to_amt(x):
    try:
        s = str(x).replace(",", "").strip()
        if s in ["", "-", "ÃƒÆ’Ã†â€™️Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â "]: return np.nan
        if s.startswith("(") and s.endswith(")"): return -round(float(s[1:-1]), 2)
        return round(float(s), 2)
    except: return np.nan

def _clean_key_series(series: pd.Series) -> pd.Series:
    _NULL_TOKENS = {"", "nan", "none", "null", "na", "n/a", "n\\a"}
    s = series.copy().astype("string") 
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
    
    matched = pd.concat(parts, ignore_index=True).merge(
        adv_df, left_on="Matched_Key", right_on="Msg_Refer_No", how="inner", suffixes=("_bank", "_adv")
    ).drop_duplicates()

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
    
    matched_merged = L[L["_CLAIM_KEY"] != ""].merge(R[R["_CLAIM_KEY"] != ""], on="_CLAIM_KEY", how="inner", suffixes=("_out", "_m3"))
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

_NULL_TOKENS = {"", "nan", "none", "null", "na", "n/a", "n\\a"}

def _clean_key_value(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).replace("\xa0", " ").strip()
    if s.lower() in _NULL_TOKENS:
        return None
    return s

def clean_raw_bank_statement_icici_v2(path):
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
        return df

    path = Path(path)
    tables = []
    try:
        ext = path.suffix.lower()
        if ext in [".xlsx", ".xls", ".xlsm"]:
            xls = pd.ExcelFile(path)
            for sheet in xls.sheet_names:
                raw = xls.parse(sheet_name=sheet, header=None)
                tbl = _extract_table_from_raw(raw)
                if tbl is not None: tables.append(tbl)
        else:
            try: raw = pd.read_excel(path, header=None)
            except: raw = pd.read_csv(path, header=None)
            tbl = _extract_table_from_raw(raw)
            if tbl is not None: tables.append(tbl)
    except: pass

    if not tables: raise ValueError("ICICI bank parser: No usable transaction table found.")
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
        pattern = r"TOTAL|OPENING BALANCE|CLOSING BALANCE|BALANCE BROUGHT FORWARD|BALANCE CARRIED FORWARD"
        df = df[~desc.str.contains(pattern, regex=True, na=False)]

    for col in ["No.", "Transaction Amount(INR)"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="ignore")
    return df.drop_duplicates().reset_index(drop=True)

def clean_raw_bank_statement_axis_v2(path):
    def _extract_table_from_raw(raw: pd.DataFrame):
        if raw.dropna(how="all").empty: return None
        header_row_idx = None
        for i, row in raw.iterrows():
            vals = [str(v).strip() for v in row.tolist() if pd.notna(v)]
            upper_vals = [v.upper() for v in vals]
            if (any(v == "S.NO" for v in upper_vals) and 
                any("TRANSACTION DATE" in v for v in upper_vals) and 
                any("PARTICULARS" in v for v in upper_vals) and
                any("DEBIT/CREDIT" in v for v in upper_vals)):
                header_row_idx = i
                break
        if header_row_idx is None: return None
        header_vals = [str(x).strip() for x in raw.iloc[header_row_idx].tolist()]
        df = raw.iloc[header_row_idx + 1:].copy()
        if df.shape[1] < len(header_vals):
            for _ in range(len(header_vals) - df.shape[1]): df[df.shape[1]] = np.nan
        df = df.iloc[:, :len(header_vals)]
        df.columns = header_vals
        return df

    path = Path(path)
    tables = []
    try:
        ext = path.suffix.lower()
        if ext in [".xlsx", ".xls", ".xlsm"]:
            xls = pd.ExcelFile(path)
            for sheet in xls.sheet_names:
                raw = xls.parse(sheet_name=sheet, header=None)
                tbl = _extract_table_from_raw(raw)
                if tbl is not None: tables.append(tbl)
        else:
            try: raw = pd.read_excel(path, header=None)
            except: raw = pd.read_csv(path, header=None)
            tbl = _extract_table_from_raw(raw)
            if tbl is not None: tables.append(tbl)
    except: pass
    
    if not tables: raise ValueError("Axis bank parser: No usable transaction table found.")
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
        pattern = r"OPENING BALANCE|CLOSING BALANCE|BALANCE BROUGHT FORWARD|TOTAL|TRANSACTION TOTAL"
        canon_df = canon_df[~desc_upper.str.contains(pattern, regex=True, na=False)]

    def _is_noise(row):
        def _empty(x): return str(x).strip().lower() in ("", "nan", "none")
        return (_empty(row.get("Txn Posted Date")) and _empty(row.get("Description")) and _empty(row.get("Transaction Amount(INR)")))
    
    if not canon_df.empty:
        canon_df = canon_df[~canon_df.apply(_is_noise, axis=1)]

    for col in ["Transaction Amount(INR)", "Balance(INR)"]:
        if col in canon_df.columns:
            canon_df[col] = canon_df[col].astype(str).str.replace(",", "", regex=False).str.strip()
            canon_df[col] = pd.to_numeric(canon_df[col], errors="ignore")
    if "No." in canon_df.columns:
        canon_df["No."] = pd.to_numeric(canon_df["No."].astype(str).str.strip(), errors="ignore")
            
    return canon_df.drop_duplicates().reset_index(drop=True)

def detect_bank_type(bank_path: Path) -> str:
    path = Path(bank_path)
    def _row_tokens(row) -> set:
        toks = set()
        for v in row:
            if pd.isna(v): continue
            s = str(v).strip()
            if s: toks.add(s); toks.add(s.upper())
        return toks

    def _scan_raw(raw: pd.DataFrame) -> Optional[str]:
        for i in range(min(len(raw), 120)):
            tokens = _row_tokens(raw.iloc[i].tolist())
            if "Transaction ID" in tokens and "Description" in tokens: return "ICICI"
            up = {t.upper() for t in tokens}
            if "S.NO" in up and "PARTICULARS" in up and "DEBIT/CREDIT" in up: return "AXIS"
        return None

    try:
        xls = pd.ExcelFile(path)
        for sh in xls.sheet_names:
            if res := _scan_raw(xls.parse(sh, header=None)): return res
    except:
        try:
            if res := _scan_raw(pd.read_csv(path, header=None)): return res
        except:
             try:
                if res := _scan_raw(pd.read_excel(path, header=None)): return res
             except: pass
             
    raise ValueError("Could not auto-detect bank type (ICICI/AXIS) from file headers.")

def detect_tpa_choice(mis_path: Path, scan_rows: int = 120) -> str:
    mis_path = Path(mis_path)
    def _norm_hdr(x): return re.sub(r"[^A-Z0-9]", "", str(x).upper()).strip()
    
    all_rhs = []
    for _, mp in TPA_MIS_MAPS_V2.items(): all_rhs.extend(list(mp.values()))
    all_rhs_norm = {_norm_hdr(h) for h in all_rhs}
    
    def _get_header_tokens(raw):
        best_i, best_hits = None, -1
        for i in range(min(len(raw), scan_rows)):
            row = raw.iloc[i].tolist()
            tokens = {_norm_hdr(v) for v in row if pd.notna(v) and str(v).strip() != ""}
            hits = sum(1 for t in tokens if t in all_rhs_norm)
            if hits > best_hits: best_hits = hits; best_i = i
        if best_i is None or best_hits <= 0: return None
        return {_norm_hdr(v) for v in raw.iloc[best_i].tolist() if pd.notna(v) and str(v).strip() != ""}

    raws = []
    try:
        xls = pd.ExcelFile(mis_path)
        for sh in xls.sheet_names: raws.append(xls.parse(sh, header=None, dtype=str))
    except:
        try: raws.append(pd.read_csv(mis_path, header=None, dtype=str))
        except: pass
    
    header_tokens, best_count = None, -1
    for raw in raws:
        t = _get_header_tokens(raw)
        if t and len(t) > best_count: header_tokens = t; best_count = len(t)
        
    if not header_tokens: raise ValueError("TPA auto-detect failed: No header row found.")
    
    scored = []
    for tpa_name, mp in TPA_MIS_MAPS_V2.items():
        rhs_norm = {_norm_hdr(v) for v in mp.values()}
        score = sum(1 for h in rhs_norm if h in header_tokens)
        scored.append((tpa_name, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    
    best_name, best_score = scored[0]
    if best_score < 2: raise ValueError(f"TPA auto-detect failed (weak signal). Top: {scored[:3]}")
    return best_name

def parse_mis_universal_v2(mis_path, tpa_name: str, empty_threshold: float = 0.5) -> pd.DataFrame:
    mis_path = Path(mis_path)
    if tpa_name not in TPA_MIS_MAPS_V2: raise ValueError(f"Unknown TPA '{tpa_name}'")
    mapping = TPA_MIS_MAPS_V2[tpa_name]
    
    expected_sources = list(mapping.values())
    canonical_names = list(mapping.keys())
    def _norm_local(s): return re.sub(r"[^A-Z0-9]", "", str(s).upper())
    expected_norm = {_norm_local(x) for x in expected_sources + canonical_names}

    def _hunt_header(raw):
        best_idx, best_score = None, -999
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
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Noise filter
        important_idx = [i for i, c in enumerate(df.columns) if c in expected_sources or c in canonical_names]
        def _is_noise(row):
            vals = [str(v).strip() for v in row]
            if sum(v.lower() in _NULL_TOKENS for v in vals) / max(1, len(vals)) >= empty_threshold:
                if important_idx and all(vals[i].lower() in _NULL_TOKENS for i in important_idx): return True
            return False
        return df[~df.apply(_is_noise, axis=1)].reset_index(drop=True)

    tables = []
    try:
        xls = pd.ExcelFile(mis_path)
        for sh in xls.sheet_names:
            tbl = _extract_from_sheet(xls.parse(sh, header=None, dtype=str))
            if tbl is not None: tables.append(tbl)
    except:
        try:
            tbl = _extract_from_sheet(pd.read_csv(mis_path, header=None, dtype=str))
            if tbl is not None: tables.append(tbl)
        except: pass

    if not tables: raise ValueError(f"Universal MIS parser V2: No table detected for '{tpa_name}'")
    return pd.concat(tables, ignore_index=True).reset_index(drop=True)

def step2_match_bank_mis_by_utr_v2(bank_df: pd.DataFrame, mis_df: pd.DataFrame, tpa_name: str, deduplicate: bool = True):
    if bank_df.empty or mis_df.empty: return pd.DataFrame(), bank_df
    mapping = TPA_MIS_MAPS_V2.get(tpa_name)
    if not mapping: raise ValueError(f"Unknown TPA '{tpa_name}'")

    mis_std = mis_df.rename(columns={v: k for k, v in mapping.items()}).copy()
    if "Cheque/ NEFT/ UTR No." not in mis_std.columns:
        raise ValueError(f"MIS mapping failed: UTR column not found for '{tpa_name}'")

    mis_std["Cheque/ NEFT/ UTR No."] = _clean_key_series(mis_std["Cheque/ NEFT/ UTR No."])
    if "Claim No" in mis_std.columns: mis_std["Claim No"] = _clean_key_series(mis_std["Claim No"])
    
    keys = pd.Series(mis_std["Cheque/ NEFT/ UTR No."]).dropna().astype(str).map(str.strip).unique()
    col_to_search = "Description" if "Description" in bank_df.columns else bank_df.columns[1]
    
    parts = []
    for utr in keys:
        s = utr.strip()
        if not s: continue
        m = bank_df[col_to_search].astype(str).str.contains(s, regex=False, na=False)
        if m.any():
            t = bank_df.loc[m].copy()
            t["Matched_Key"] = s
            parts.append(t)
            
    if not parts: return pd.DataFrame(), bank_df
    
    merged = pd.concat(parts, ignore_index=True).merge(
        mis_std, left_on="Matched_Key", right_on="Cheque/ NEFT/ UTR No.", how="inner", suffixes=("_bank", "_mis")
    )
    if deduplicate: merged = merged.drop_duplicates()
    
    if "Transaction ID" in bank_df.columns and "Transaction ID" in merged.columns:
        not_in = bank_df.loc[~bank_df["Transaction ID"].isin(merged["Transaction ID"])]
    else:
        not_in = bank_df.loc[~bank_df["Description"].isin(merged["Description"])]
        
    return merged.reset_index(drop=True), not_in

def parse_outstanding_excel_to_clean_v2(xlsx_path: Path) -> pd.DataFrame:
    # This is similar to V1 but using valid prompt logic to be safe
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
                if nc in targets or (nc.startswith("CONSUL") and "CONSULTANT" in targets): hits += 1
            if best_hits is None or hits > best_hits: best_idx, best_hits = i, hits
        if best_idx is None or best_hits <= 2: raise ValueError("Outstanding parser V2: Header not found.")
        return best_idx

    hdr_idx = _hunt_header_v2(raw)
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

    ent_col, current_entity, header_rows = [], "", []
    for idx, row in df.iterrows():
        vals = row.fillna("").astype(str).tolist()
        if vals[0].strip() and all(not str(v).strip() for v in vals[1:]):
            current_entity = vals[0].replace("\xa0", " ").strip()
            header_rows.append(idx)
        ent_col.append(current_entity)
    df["Bill Company Name"] = ent_col
    if header_rows: df = df.drop(index=header_rows)
    
    df = df[~df.apply(lambda r: any(t in " ".join([str(x) for x in r]).upper() for t in _FOOTER_TOKENS), axis=1)]
    for c in df.columns: df[c] = df[c].astype(str).str.replace("\xa0", " ", regex=False).str.strip()
    return df.reset_index(drop=True)

def step4_strict_matches_v2(step3_df: pd.DataFrame, outstanding_path: Path, deduplicate: bool = True):
    if step3_df.empty: return pd.DataFrame(), pd.DataFrame()
    if "Claim No" not in step3_df.columns: raise ValueError("Step 3 missing Claim No")
    
    out = parse_outstanding_excel_to_clean_v2(outstanding_path).copy()
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
    
    L = L[L["_CLAIM_KEY"] != ""]
    R = R[R["_CLAIM_KEY"] != ""]
    
    matched_merged = L.merge(R, on="_CLAIM_KEY", how="inner", suffixes=("_out", "_m3"))
    matched = matched_merged.drop(columns=["_CLAIM_KEY"])
    if deduplicate: matched = matched.drop_duplicates()
    
    matched_keys = matched_merged["_CLAIM_KEY"].unique()
    unmatched_step3 = R[~R["_CLAIM_KEY"].isin(matched_keys)].copy()
    unmatched_step3 = unmatched_step3.drop(columns=["_CLAIM_KEY"])
    if deduplicate: unmatched_step3 = unmatched_step3.drop_duplicates()
    
    return matched, unmatched_step3


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
            activity_logs_collection.insert_one({
                "username": username, "bank_type": bank_type, "step_completed": step, "run_id": run_id,
                "timestamp": datetime.utcnow(), "duration_seconds": duration, "row_counts": counts or {},
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
        "hashed_password": get_password_hash(user.password), "created_at": datetime.utcnow(),
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
        {"$set": {"verification_token": token, "verification_token_expires": datetime.utcnow() + timedelta(hours=24)}}
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
    if user.get("verification_token_expires") and datetime.utcnow() > user["verification_token_expires"]:
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
    now = datetime.utcnow()
    this_week = len([a for a in completed if a["timestamp"] >= now - timedelta(days=7)])
    this_month = len([a for a in completed if a["timestamp"] >= now - timedelta(days=30)])
    
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
    start = datetime.utcnow() - timedelta(days=days)
    pipeline = [
        {"$match": {"username": current_user.username, "timestamp": {"$gte": start}, "step_completed": 4, "success": True}},
        {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}}, "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    results = {r["_id"]: r["count"] for r in activity_logs_collection.aggregate(pipeline)}
    return [DailyActivityResponse(date=(datetime.utcnow() - timedelta(days=days-i-1)).strftime("%Y-%m-%d"), count=results.get((datetime.utcnow() - timedelta(days=days-i-1)).strftime("%Y-%m-%d"), 0)) for i in range(days)]

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

@APP.get("/download/{run_id}/{filename}")
async def download_file(run_id: str, filename: str, user: UserInDB = Depends(get_current_user) if AUTH_ENABLED else None):
    file_path = RUN_ROOT / run_id / filename
    if not file_path.exists(): raise HTTPException(404, "File not found")
    media = "application/zip" if filename.endswith(".zip") else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return StreamingResponse(open(file_path, "rb"), media_type=media, headers={"Content-Disposition": f"attachment; filename={filename}"})

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

                rec_doc = {
                    "username": current_user.username,
                    "run_id": run_id,
                    "timestamp": datetime.utcnow(),
                    "created_at": datetime.utcnow(),
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
        # Note: Requirement says "Saves 07_final_posting_sheet.xlsx". 
        # Usually step 4 in V1 produces "06_outstanding...". I'll save cleaned just in case.
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
                
                # Fetch TPA from previous step log
                tpa = "Unknown"
                if activity_logs_collection is not None:
                     last = activity_logs_collection.find_one({"username": current_user.username, "run_id": run_id, "step_completed": 2})
                     if last: tpa = last.get("tpa_name", "Unknown")

                total_amount = 0.0
                if "Transaction Amount(INR)_bank" in final_matched.columns:
                     total_amount = final_matched["Transaction Amount(INR)_bank"].apply(_to_amt).sum()
                
                rec_doc = {
                    "username": current_user.username, "run_id": run_id,
                    "timestamp": datetime.utcnow(), "created_at": datetime.utcnow(),
                    "bank_type": CURRENT_BANK_TYPE or "Unknown", "tpa_name": tpa,
                    "status": "Completed", "pipeline": "v2",
                    "summary": {
                        "step4_outstanding": len(final_matched),
                        "total_amount": float(total_amount),
                        "unmatched_count": len(final_unmatched)
                    },
                    "zip_file_id": str(fid)
                }
                reconciliation_results_collection.insert_one(rec_doc)
            except Exception as e: print(f"DB Error: {e}")

        if current_user:
            log_activity(current_user.username, CURRENT_BANK_TYPE, 3, run_id, 
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
        raise HTTPException(400, detail=str(e))