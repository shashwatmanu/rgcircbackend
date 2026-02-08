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
from bs4 import BeautifulSoup

from recon_service import (
    normalize_to_xlsx,
    detect_bank_type,
    clean_raw_bank_statement_icici,
    clean_raw_bank_statement_axis,
    detect_tpa_choice,
    parse_mis_universal,
    step2_match_bank_mis_by_utr,
    parse_outstanding_excel_to_clean,

    filter_outstanding_for_tpa,
    run_step4_plan,
    TPA_MIS_MAPS,
    _CONVERTED_DIR,

    _clean_key_series
)

TPA_CHOICES = sorted(list(TPA_MIS_MAPS.keys()))

# ==========================================
#  AUTHENTICATION SETUP
# ==========================================
try:
    from auth import (
        get_current_user, 
        authenticate_user, 
        create_access_token, 
        get_password_hash,
        ACCESS_TOKEN_EXPIRE_MINUTES,
        get_current_admin_user
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
    get_current_admin_user = None

    # Dummy models for API docs/validation if auth is disabled
    from pydantic import BaseModel
    class UserResponse(BaseModel): pass
    class UserRegister(BaseModel): pass
    class UserLogin(BaseModel): pass
    class Token(BaseModel): pass
    class UserInDB(BaseModel): pass
    class ActivityLog(BaseModel): pass
    class UserStatsResponse(BaseModel): pass
    class ActivityResponse(BaseModel): pass
    class DailyActivityResponse(BaseModel): pass
    class ReconciliationHistoryResponse(BaseModel): pass
    class ReconciliationSummary(BaseModel): pass
    class SendVerificationEmailRequest(BaseModel): pass
    class VerifyEmailRequest(BaseModel): pass

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

# FORCE LOCALHOST ORIGINS (to avoid env var issues)
if "http://localhost:3000" not in ALLOWED_ORIGINS: ALLOWED_ORIGINS.append("http://localhost:3000")
if "http://127.0.0.1:3000" not in ALLOWED_ORIGINS: ALLOWED_ORIGINS.append("http://127.0.0.1:3000")

print(f"[CORS] Configured Allowed Origins: {ALLOWED_ORIGINS}")

@APP.middleware("http")
async def log_origin_middleware(request, call_next):
    origin = request.headers.get("origin")
    if origin:
        print(f"[Middleware] Request from Origin: {origin}")
    response = await call_next(request)
    return response

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
        "is_active": True, "is_admin": False, "email_verified": False, "verification_token": None
    }
    try:
        users_collection.insert_one(user_dict)
        return UserResponse(
            username=user_dict["username"], email=user_dict.get("email"),
            full_name=user_dict.get("full_name"), created_at=user_dict["created_at"],
            is_admin=user_dict.get("is_admin", False),
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
        is_admin=current_user.is_admin,
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

# ----------------- ADMIN ENDPOINTS -----------------

@APP.get("/admin/reconciliations/history")
async def get_admin_reconciliation_history(
    current_user: UserInDB = Depends(get_current_admin_user), 
    limit: int = 50, 
    skip: int = 0
):
    """
    Admin only: Fetch ALL reconciliation history across all users.
    """
    if reconciliation_results_collection is None: raise HTTPException(503, "DB unavailable")
    
    # Fetch all records, sorted by timestamp desc
    results = list(reconciliation_results_collection.find({}, {"_id": 0}).sort("timestamp", -1).skip(skip).limit(limit))
    
    # Process results similar to user history but ensure username is present (which it is in DB)
    for r in results:
        if "summary" not in r: r["summary"] = {}
        s = r["summary"]
        if "total_amount" not in s and "total_value" in s: s["total_amount"] = s["total_value"]
        s["total_amount"] = float(s.get("total_amount", 0.0) or 0.0)
        s["step4_outstanding"] = int(s.get("step4_outstanding", 0) or 0)
        s["unique_patients"] = int(s.get("unique_patients", 0) or 0)
        s["step2_matches"] = int(s.get("step2_matches", 0) or 0)

    return results

@APP.get("/admin/reconciliations/{run_id}/details")
async def get_admin_reconciliation_details(run_id: str, current_user: UserInDB = Depends(get_current_admin_user)):
    """
    Admin only: Fetch details for ANY reconciliation run.
    """
    if reconciliation_results_collection is None: raise HTTPException(503, "DB unavailable")
    
    # Find by run_id only (ignore username)
    res = reconciliation_results_collection.find_one({"run_id": run_id}, {"_id": 0})
    if not res: raise HTTPException(404, "Reconciliation not found")
    return res

@APP.get("/admin/reconciliations/{run_id}/download-zip")
async def download_admin_reconciliation_zip(run_id: str, current_user: UserInDB = Depends(get_current_admin_user)):
    """
    Admin only: Download zip for ANY reconciliation run.
    """
    if reconciliation_results_collection is None or fs is None: raise HTTPException(503, "DB unavailable")
    from bson import ObjectId
    
    # Find by run_id only
    result = reconciliation_results_collection.find_one({"run_id": run_id})
    if not result: raise HTTPException(404, "Reconciliation not found")
    if "zip_file_id" not in result: raise HTTPException(404, "ZIP file ID not found")

    try:
        zip_file = fs.get(ObjectId(result["zip_file_id"]))
        return StreamingResponse(
            zip_file, media_type="application/zip", 
            headers={"Content-Disposition": f"attachment; filename={run_id}.zip"}
        )
    except Exception as e:
        raise HTTPException(500, f"Error retrieving file: {str(e)}")

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

        # NORMALIZE (NEW)
        norm_bank = None
        try:
            norm_bank = normalize_to_xlsx(bank_path)
            
            # Detect Bank
            CURRENT_BANK_TYPE = detect_bank_type(norm_bank)
            
            # Clean
            # Clean
            if CURRENT_BANK_TYPE == "ICICI":
                bank_df = clean_raw_bank_statement_icici(norm_bank)
            else:
                bank_df = clean_raw_bank_statement_axis(norm_bank)
                
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
        finally:
            if norm_bank and norm_bank.exists():
                try: os.remove(norm_bank)
                except: pass
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
        
        # NORMALIZE (NEW)
        norm_mis = None
        try:
            norm_mis = normalize_to_xlsx(mis_path)
            
            # Detect TPA
            tpa_name = detect_tpa_choice(norm_mis)
            
            # Save TPA choice for Step 3
            with open(CURRENT_RUN_DIR / "tpa_name.txt", "w") as f:
                f.write(tpa_name)
            
            # Clean MIS
            mis_clean = parse_mis_universal(norm_mis, tpa_name)
            out_mis = save_xlsx(mis_clean, CURRENT_RUN_DIR / "04_mis_cleaned.xlsx")
            
            # Match Bank <-> MIS
            # Need to load Bank from Step 1
            bank_path = CURRENT_RUN_DIR / "01_bank_clean.xlsx"
            if not bank_path.exists(): raise HTTPException(400, "Step 1 bank output missing")
            bank_df = pd.read_excel(bank_path, dtype=str)
            
            matched, not_in_bank = step2_match_bank_mis_by_utr(bank_df, mis_clean, tpa_name)
            
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
        finally:
            if norm_mis and norm_mis.exists():
                try: os.remove(norm_mis)
                except: pass
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
        
        # NORMALIZE (NEW)
        norm_out = None
        try:
            norm_out = normalize_to_xlsx(out_path)
            
            # Clean Outstanding
            out_clean = parse_outstanding_excel_to_clean(norm_out)
            out_clean_path = save_xlsx(out_clean, CURRENT_RUN_DIR / "06_outstanding_cleaned.xlsx")
            
            # Match Step 2 (Mapped) <-> Outstanding
            s2_path = CURRENT_RUN_DIR / "05_bank_mis_mapped.xlsx"
            if not s2_path.exists(): raise HTTPException(400, "Step 2 mapped output missing")
            s2_df = pd.read_excel(s2_path, dtype=str)
            
            # Retrieve TPA name
            tpa_name = "Unknown"
            tpa_file = CURRENT_RUN_DIR / "tpa_name.txt"
            if tpa_file.exists():
                tpa_name = tpa_file.read_text().strip()
            
            # Filter Outstanding (if TPA known)
            if tpa_name and tpa_name != "Unknown":
                out_clean = filter_outstanding_for_tpa(out_clean, tpa_name)
                # Save filtered as the input for Step 4
                out_clean_path = save_xlsx(out_clean, CURRENT_RUN_DIR / "06_outstanding_filtered.xlsx")
                
            final_matched, final_unmatched = run_step4_plan(s2_df, out_clean_path, tpa_name)
            
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
        finally:
            if norm_out and norm_out.exists():
                try: os.remove(norm_out)
                except: pass

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
        
        norm_out = None
        try:
            norm_out = normalize_to_xlsx(out_path)
            
            out_clean_df = parse_outstanding_excel_to_clean(norm_out)
            clean_out_path = save_xlsx(out_clean_df, CURRENT_RUN_DIR / "00_outstanding_cleaned.xlsx")
            
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
                
                b_norm = None
                try:
                    # Detect & Clean Bank
                    b_norm = normalize_to_xlsx(b_path)
                    b_type = detect_bank_type(b_norm)
                    detected_banks.add(b_type)
                    
                    if b_type == "ICICI":
                        bank_clean = clean_raw_bank_statement_icici(b_norm)
                    else:
                        bank_clean = clean_raw_bank_statement_axis(b_norm)
                    
                    agg_stats["step1_bank_rows"] += len(bank_clean)
    
                    # 3. Iterate MIS
                    for m_filename, m_path in mis_paths:
                        m_stem = Path(m_filename).stem
                        pair_id = f"{b_name}_vs_{m_stem}"
                        pair_dir = CURRENT_RUN_DIR / pair_id
                        pair_dir.mkdir(exist_ok=True)
                        
                        m_norm = None
                        try:
                            # Detect TPA & Clean
                            m_norm = normalize_to_xlsx(m_path)
                            tpa = detect_tpa_choice(m_norm)
                            detected_tpas.add(tpa)
                            
                            mis_clean_df = parse_mis_universal(m_norm, tpa)
                            
                            # Step 2 Match
                            step2_match, _ = step2_match_bank_mis_by_utr(bank_clean, mis_clean_df, tpa)
                            agg_stats["step2_matches"] += len(step2_match)
                            agg_stats["step3_mis_mapped"] += len(step2_match) # Reuse for compat
                            
                            
                            # Filter Outstanding for this TPA
                            # We must operate on a copy or parse fresh? 
                            # parse_outstanding_excel_to_clean is expensive? 
                            # Creating a filtered file per TPA
                            out_clean_tpa = filter_outstanding_for_tpa(out_clean_df.copy(), tpa)
                            tpa_out_path = save_xlsx(out_clean_tpa, CURRENT_RUN_DIR / f" outstanding_{tpa[:10]}_filtered.xlsx")
                            
                            # Step 4 Match (Strict)
                            final_match, final_unmatched = run_step4_plan(step2_match, tpa_out_path, tpa)
                            
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
                            
                        except Exception as inner_e:
                            print(f"[Bulk Pair Error] {pair_id}: {inner_e}")
                            summary_rows.append({
                                "Bank File": bank_file.filename,
                                "MIS File": m_filename,
                                "Error": str(inner_e),
                                "Status": "Failed",
                                "produced_files": {}
                            })
                        finally:
                            if m_norm and m_norm.exists():
                                try: os.remove(m_norm)
                                except: pass
                                
                except Exception as b_e:
                    print(f"[Bulk Bank Error] {b_name}: {b_e}")
                    summary_rows.append({
                        "Bank File": bank_file.filename,
                        "Error": f"Bank Processing Failed: {str(b_e)}",
                        "Status": "Failed",
                        "produced_files": {}
                    })
                finally:
                    if b_norm and b_norm.exists(): 
                        try: os.remove(b_norm)
                        except: pass

            # BUILD THE CONSOLIDATED MASTER SHEET
            try:
                if consolidated_frames:
                    master_df = pd.concat(consolidated_frames, ignore_index=True)
                    master_path = CURRENT_RUN_DIR / "Final posting sheet (Consolidated).xlsx"
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
        finally:
            if norm_out and norm_out.exists():
                try: os.remove(norm_out)
                except: pass

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(400, detail=str(e))