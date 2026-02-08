import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Set, Union
from bs4 import BeautifulSoup
import io

# ==========================================
#  PART 0: FILE NORMALISER (ALL -> .XLSX)
#  (NEW - ONLY ADDITION)
# ==========================================

_CONVERTED_DIR = Path("_converted")
_CONVERTED_DIR.mkdir(exist_ok=True)

def _is_html_disguised_xls(p: Path) -> bool:
    try:
        head = p.read_bytes()[:4096].lower()
        return (b"<html" in head) or (b"<table" in head) or (b"<!doctype html" in head)
    except Exception:
        return False

def _html_table_to_grid(html_bytes: bytes):
    """
    Parse FIRST <table> into a 2D grid of strings.
    Preserves all rows exactly (no header inference).
    Expands colspan so the grid shape stays consistent.
    """
    soup = BeautifulSoup(html_bytes, "lxml")
    table = soup.find("table")
    if table is None:
        raise ValueError("HTML-backed .xls detected, but no <table> found.")

    rows = table.find_all("tr")
    grid = []

    for tr in rows:
        row_cells = []
        tds = tr.find_all(["td", "th"])
        for cell in tds:
            text = cell.get_text(separator=" ", strip=True)
            colspan = int(cell.get("colspan", 1) or 1)

            row_cells.append(text)
            for _ in range(colspan - 1):
                row_cells.append("")
        grid.append(row_cells)

    max_cols = max((len(r) for r in grid), default=0)
    for r in grid:
        if len(r) < max_cols:
            r.extend([""] * (max_cols - len(r)))

    return grid

def normalize_to_xlsx(src_path: Path) -> Path:
    """
    NORMALISE EVERYTHING to a fresh .xlsx (CloudConvert/Excel Save-As feel):
    - .xls, .xlsx, .xlsm, .csv -> .xlsx
    - Preserves ALL rows (no header inference)
    - Reads as text (no numeric/date coercion)
    - .xlsm macros are dropped by design
    - Handles HTML-disguised .xls (very common exports)
    """
    src_path = Path(src_path)
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    out_path = _CONVERTED_DIR / f"{src_path.stem}__to_xlsx_exact.xlsx"
    ext = src_path.suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(src_path, dtype=str, header=None)
        df.to_excel(out_path, index=False, header=False)
        return out_path

    if ext == ".xls" and _is_html_disguised_xls(src_path):
        grid = _html_table_to_grid(src_path.read_bytes())
        df = pd.DataFrame(grid)
        df.to_excel(out_path, index=False, header=False)
        return out_path

    if ext in [".xls", ".xlsx", ".xlsm"]:
        xls = pd.ExcelFile(src_path)
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            for sheet in xls.sheet_names:
                df = xls.parse(sheet_name=sheet, dtype=str, header=None)
                df.to_excel(writer, sheet_name=str(sheet)[:31], index=False, header=False)
        return out_path

    raise ValueError(f"Unsupported file type for normalization: {ext}")

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
        "Cheque/ NEFT/ UTR No.": "Instrument/NEFT No",
        "Claim No": "AL Number"
    },
    "HEALTH INDIA INSURANCE TPA SERVICES PRIVATE LTD.": {
        "Cheque/ NEFT/ UTR No.": "utrnumber",
        "Claim No": "CCN"
    },
    "HERITAGE HEALTH INSURANCE TPA PRIVATE LIMITED": {
        "Cheque/ NEFT/ UTR No.": "UTR_NO",
        "Claim No": "HHCCN"
    },
    "MEDSAVE HEALTHCARE TPA PVT LTD": {
        "Cheque/ NEFT/ UTR No.": "UTR/Chq No.",
        "Claim No": "FILENO"
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
        "Claim No": "Insurer Claim Number"
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

# ==========================================
#  GLOBAL HARDENING: KEY CLEANING (CRITICAL)
# ==========================================

_NULL_TOKENS = {"", "nan", "none", "null", "na", "n/a", "n\\a", "0", "0.0"}

def _clean_key_series(series: pd.Series) -> pd.Series:
    s = series.copy()
    s = s.astype("string")
    s = s.str.replace("\xa0", " ", regex=False).str.strip()
    lower = s.str.lower()
    s = s.mask(lower.isin(_NULL_TOKENS), pd.NA)
    return s.astype(object).where(s.notna(), np.nan)

def _clean_key_value(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).replace("\xa0", " ").strip()
    if s.lower() in _NULL_TOKENS:
        return None
    return s

def _norm_colname(x: str) -> str:
    if x is None:
        return ""
    s = str(x)
    s = s.replace("\xa0", " ")
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    return s.upper()

def _make_unique_columns(cols) -> list:
    seen = {}
    out = []
    for c in list(cols):
        base = str(c).replace("\xa0", " ").strip()
        if base == "" or base.lower() in ("nan", "none", "null"):
            base = "UNNAMED"
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}__DUP{seen[base]}")
    return out

def _force_series(x):
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x

# ==========================================
#  PART 2: CLEANERS (BANK)
# ==========================================

def clean_raw_bank_statement_icici(path):
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

    df = df.loc[:, ~df.columns.duplicated()].copy()
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

def clean_raw_bank_statement_axis(path):
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
    canon_df = canon_df.loc[:, ~canon_df.columns.duplicated()].copy()

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

# ==========================================
#  BANK TYPE AUTO-DETECTION (WORKING)
# ==========================================

def detect_bank_type(bank_path: Path) -> str:
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

# ==========================================
#  TPA TYPE AUTO-DETECTION (RHS-ONLY, UNIQUE PAIR, FAIL-FAST)
# ==========================================

def detect_tpa_choice(mis_path: Path, scan_rows: int = 120) -> str:
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
    for tpa_name, mp in TPA_MIS_MAPS.items():
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

# ==========================================
#  PART 3: MIS PARSING + BANK↔MIS MATCHING
# ==========================================

def parse_mis_universal(
    mis_path,
    tpa_name: str = "IHX (Original MIS)",
    empty_threshold: float = 0.5
) -> pd.DataFrame:

    mis_path = Path(mis_path)
    ext = mis_path.suffix.lower()

    if tpa_name not in TPA_MIS_MAPS:
        raise ValueError(f"Unknown TPA '{tpa_name}' for MIS parsing.")

    mapping: Dict[str, str] = TPA_MIS_MAPS[tpa_name]
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

        df_sheet.columns = _make_unique_columns(df_sheet.columns)
        df_sheet = df_sheet.dropna(how="all")

        for c in df_sheet.columns:
            df_sheet[c] = (
                _force_series(df_sheet[c])
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

def step2_match_bank_mis_by_utr(bank_df: pd.DataFrame,
                               mis_df: pd.DataFrame,
                               tpa_name: str,
                               deduplicate: bool = True,
                               min_key_len: int = 8):

    if bank_df.empty:
        return pd.DataFrame(), bank_df
    if mis_df.empty:
        return pd.DataFrame(), bank_df

    bank_df = bank_df.loc[:, ~bank_df.columns.duplicated()].copy()

    mapping = TPA_MIS_MAPS.get(tpa_name)
    if mapping is None:
        raise ValueError(f"Unknown TPA '{tpa_name}' for MIS mapping.")

    actual_lookup = {}
    for c in mis_df.columns:
        k = _norm_colname(c)
        if k and (k not in actual_lookup):
            actual_lookup[k] = c

    rename_map = {}
    for canon_col, src_col in mapping.items():
        key = _norm_colname(src_col)
        if key in actual_lookup:
            rename_map[actual_lookup[key]] = canon_col

    mis_std = mis_df.rename(columns=rename_map).copy()

    if "Cheque/ NEFT/ UTR No." not in mis_std.columns:
        raise ValueError(
            f"MIS mapping failed: could not find UTR column after rename for TPA '{tpa_name}'. "
            f"Expected source column '{mapping.get('Cheque/ NEFT/ UTR No.')}'."
        )

    mis_std["Cheque/ NEFT/ UTR No."] = _clean_key_series(mis_std["Cheque/ NEFT/ UTR No."])
    if "Claim No" in mis_std.columns:
        mis_std["Claim No"] = _clean_key_series(mis_std["Claim No"])

    mis_std["_UTR_SEARCH"] = mis_std["Cheque/ NEFT/ UTR No."].astype("string").str.strip()
    mis_std["_UTR_SEARCH"] = mis_std["_UTR_SEARCH"].apply(
        lambda s: (str(s)[:16] if (pd.notna(s) and len(str(s)) > 16) else (str(s) if pd.notna(s) else np.nan))
    )
    mis_std["_UTR_SEARCH"] = _clean_key_series(mis_std["_UTR_SEARCH"])

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

    pairs = pairs[pairs["_UTR_SEARCH"].map(lambda x: len(str(x)) >= int(min_key_len))]

    col_to_search = "Description" if "Description" in bank_df.columns else bank_df.columns[1]
    parts = []

    bank_text = _force_series(bank_df[col_to_search]).astype(str)
    utr_keys = pairs["_UTR_SEARCH"].dropna().astype(str).map(str.strip).unique()

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

    merged_text_col = col_to_search
    if merged_text_col not in merged.columns and f"{merged_text_col}_bank" in merged.columns:
        merged_text_col = f"{merged_text_col}_bank"

    if "Transaction ID" in bank_df.columns:
        merged_tid_col = "Transaction ID"
        if merged_tid_col not in merged.columns and f"{merged_tid_col}_bank" in merged.columns:
            merged_tid_col = f"{merged_tid_col}_bank"

        if merged_tid_col in merged.columns:
            not_in = bank_df.loc[~bank_df["Transaction ID"].isin(merged[merged_tid_col])]
        else:
            not_in = bank_df.loc[~bank_df[col_to_search].isin(merged[merged_text_col])]
    else:
        not_in = bank_df.loc[~bank_df[col_to_search].isin(merged[merged_text_col])]

    return merged.reset_index(drop=True), not_in

# ==========================================
#  PART 3B: OUTSTANDING PARSER + INSURANCE COMPANY
# ==========================================

_FOOTER_TOKENS = ["SUB TOTAL", "SUBTOTAL", "TOTAL", "GRAND TOTAL"]

_OUT_HEADER_KEYS = [
    "Sl No", "Bill No", "Date", "CR No", "Patient Name",
    "Net Amount", "Amount Paid", "TDS", "Write-Off",
    "Balance", "Location", "Consultant", "Claim No", "Insurance Company"
]

def _hunt_outstanding_header_row(df_raw: pd.DataFrame) -> int:
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
        raise ValueError("Outstanding parser: could not confidently locate header row.")
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

def _row_is_footer_like(row: pd.Series) -> bool:
    flat = " ".join([str(x) for x in row.fillna("").tolist()]).upper()
    return any(tok in flat for tok in _FOOTER_TOKENS)

def parse_outstanding_excel_to_clean(xlsx_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, header=None, dtype=str)
    hdr_idx = _hunt_outstanding_header_row(raw)
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

    ent_col = []
    current_entity = ""
    header_row_indices = []
    for idx, row in df.iterrows():
        if _row_is_section_header_like(row):
            current_entity = _clean_entity_name(row.iloc[0])
            header_row_indices.append(idx)
        ent_col.append(current_entity)
    
    # Fix for Double-Match (already cleaned file):
    # If "Bill Company Name" exists and we found NO new section headers, keep original.
    if "Bill Company Name" in df.columns and not any(ent_col) and not header_row_indices:
        # likely already cleaned, preserve
        pass
    else:
        df["Bill Company Name"] = ent_col

    if header_row_indices:
        df = df.drop(index=header_row_indices)

    df = df[~df.apply(_row_is_footer_like, axis=1)]

    for c in df.columns:
        df[c] = df[c].astype(str).str.replace("\xa0", " ", regex=False).str.strip()

    for col in ["Patient Name", "CR No", "Balance"]:
        if col not in df.columns:
            raise ValueError(f"Outstanding parser: missing required column '{col}' after parsing.")

    return df.reset_index(drop=True)

# ==========================================
#  PART 4: CLAIM MATCH HELPERS (STRICT KEYS)
# ==========================================

_SEP_RE = re.compile(r"[\s\-_]+")
_NONALNUM_RE = re.compile(r"[^A-Z0-9]")

_PAREN_RE = re.compile(r"\([^)]*\)")
_DOT_RE = re.compile(r"\..*$")
_SUFFIX_RE = re.compile(r"([_-])(\d{1,2})$")

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

def _split_tokens_any(s: str) -> List[str]:
    if s is None:
        return []
    x = str(s).replace("\xa0", " ").strip()
    if x.strip().lower() in ['nan', 'none', 'null', 'na', 'n/a', '']:
        return []
    # handle AND / and as separator too
    x = re.sub(r"\bAND\b", "/", x, flags=re.IGNORECASE)
    parts = [p.strip() for p in re.split(r"[/,&]", x) if p.strip()]
    return parts

def _numeric_tokens_from_parts(parts: List[str], min_len: int = 6) -> List[str]:
    toks = []
    for p in parts:
        p = _strip_trailing_suffix(p.strip())
        nums = re.findall(rf"\d{{{min_len},}}", p)
        toks.extend(nums)
    return toks

def _alpha_tokens_from_parts(parts: List[str]) -> List[str]:
    toks = []
    for p in parts:
        k = _alpha_key(p)
        if k:
            toks.append(k)
    return toks

# ==========================================
#  PART 4A: IHX STRICT MATCH (UNCHANGED BEHAVIOR)
#  - This preserves your existing IHX pipeline behavior.
# ==========================================

def step4_strict_matches(step3_df: pd.DataFrame,
                         outstanding_path: Path,
                         deduplicate: bool = True):

    if step3_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    if "Claim No" not in step3_df.columns:
        raise ValueError("Step 3 missing Claim No")

    out = parse_outstanding_excel_to_clean(outstanding_path).copy()

    if "Claim No" in out.columns:
        out = out.rename(columns={"Claim No": "Claim No_out"})

    for col in ["Patient Name", "CR No", "Balance"]:
        if col not in out.columns:
            raise ValueError(f"Outstanding missing '{col}'.")

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

    matched = merged_alpha.copy()

    if deduplicate and not matched.empty:
        matched = matched.drop_duplicates()

    matched_rids = set(matched["_RID"].unique().tolist()) if (not matched.empty and "_RID" in matched.columns) else set()
    unmatched_step3 = R[~R["_RID"].isin(matched_rids)].copy()

    drop_cols_R = [c for c in ["_RID", "_ALPHA_KEY"] if c in unmatched_step3.columns]
    if drop_cols_R:
        unmatched_step3 = unmatched_step3.drop(columns=drop_cols_R)

    drop_cols_matched = [c for c in ["_OUT_IDX", "_ALPHA_KEY", "_RID"] if c in matched.columns]
    if drop_cols_matched:
        matched = matched.drop(columns=drop_cols_matched)

    return matched.reset_index(drop=True), unmatched_step3.reset_index(drop=True)

# ==========================================
#  PART 4B: NON-IHX ALPHA STRICT UNIQUE (TPA-GATED)
#  - key must be unique on BOTH sides (OUT + STEP3)
# ==========================================

def step4_alpha_strict_unique(step3_df: pd.DataFrame,
                              outstanding_path: Path,
                              deduplicate: bool = True,
                              match_mode: str = "ALPHA_STRICT_UNIQUE"):

    if step3_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    if "Claim No" not in step3_df.columns:
        raise ValueError("Step 3 missing Claim No")

    out = parse_outstanding_excel_to_clean(outstanding_path).copy()
    if "Claim No" in out.columns:
        out = out.rename(columns={"Claim No": "Claim No_out"})

    L = out.copy().reset_index(drop=False).rename(columns={"index": "_OUT_IDX"})
    R = step3_df.copy().reset_index(drop=True)
    R["_RID"] = np.arange(len(R))

    L["_ALPHA_KEY"] = L.get("Claim No_out", "").apply(_alpha_key)
    R["_ALPHA_KEY"] = R["Claim No"].apply(_alpha_key)

    L = L[L["_ALPHA_KEY"] != ""].copy()
    R = R[R["_ALPHA_KEY"] != ""].copy()

    if L.empty or R.empty:
        return pd.DataFrame(), step3_df

    l_counts = L["_ALPHA_KEY"].value_counts()
    r_counts = R["_ALPHA_KEY"].value_counts()

    ok_keys = set(l_counts[l_counts == 1].index).intersection(set(r_counts[r_counts == 1].index))
    if not ok_keys:
        return pd.DataFrame(), step3_df

    L1 = L[L["_ALPHA_KEY"].isin(ok_keys)].copy()
    R1 = R[R["_ALPHA_KEY"].isin(ok_keys)].copy()

    merged = L1.merge(R1, on="_ALPHA_KEY", how="inner", suffixes=("_out", "_m3"))
    merged["MATCH_MODE"] = match_mode

    if deduplicate and not merged.empty:
        merged = merged.drop_duplicates()

    matched_rids = set(merged["_RID"].unique().tolist()) if (not merged.empty and "_RID" in merged.columns) else set()
    unmatched = R[~R["_RID"].isin(matched_rids)].copy()

    unmatched = unmatched.drop(columns=["_RID", "_ALPHA_KEY"], errors="ignore")
    merged = merged.drop(columns=["_OUT_IDX", "_RID", "_ALPHA_KEY"], errors="ignore")

    return merged.reset_index(drop=True), unmatched.reset_index(drop=True)

# ==========================================
#  PART 4C: NON-IHX SPLIT-ALPHA UNIQUE (VIDAL)
#  - OUT splits on / , & AND and -> alpha key -> unique match
# ==========================================

def step4_split_alpha_unique(step3_df: pd.DataFrame,
                             outstanding_path: Path,
                             deduplicate: bool = True,
                             match_mode: str = "SPLIT_ALPHA_UNIQUE"):

    if step3_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    if "Claim No" not in step3_df.columns:
        raise ValueError("Step 3 missing Claim No")

    out = parse_outstanding_excel_to_clean(outstanding_path).copy()
    if "Claim No" in out.columns:
        out = out.rename(columns={"Claim No": "Claim No_out"})

    L = out.copy().reset_index(drop=False).rename(columns={"index": "_OUT_IDX"}).copy()
    R = step3_df.copy().reset_index(drop=True).copy()
    R["_RID"] = np.arange(len(R))

    R["_ALPHA_KEY"] = R["Claim No"].apply(_alpha_key)
    R = R[R["_ALPHA_KEY"] != ""].copy()
    if R.empty:
        return pd.DataFrame(), step3_df

    key_to_rids = R.groupby("_ALPHA_KEY")["_RID"].apply(lambda x: set(x.tolist())).to_dict()

    picks = []
    for _, row in L.iterrows():
        parts = _split_tokens_any(row.get("Claim No_out", ""))
        toks = _alpha_tokens_from_parts(parts)
        if not toks:
            continue

        cand = set()
        ambiguous = False
        for t in toks:
            rids = key_to_rids.get(str(t).strip())
            if not rids:
                continue
            cand |= rids
            if len(rids) > 1:
                ambiguous = True

        if ambiguous:
            continue
        if len(cand) == 1:
            picks.append({"_OUT_IDX": row["_OUT_IDX"], "_RID": list(cand)[0]})

    if not picks:
        return pd.DataFrame(), step3_df

    P = pd.DataFrame(picks).drop_duplicates(subset=["_OUT_IDX"])
    grp = P.groupby("_OUT_IDX")["_RID"].nunique().reset_index(name="_N")
    ok_out = set(grp.loc[grp["_N"] == 1, "_OUT_IDX"].tolist())
    P = P[P["_OUT_IDX"].isin(ok_out)].copy()

    L_pick = L.merge(P, on="_OUT_IDX", how="inner")
    merged = L_pick.merge(R, on="_RID", how="inner", suffixes=("_out", "_m3"))
    merged["MATCH_MODE"] = match_mode

    if deduplicate and not merged.empty:
        merged = merged.drop_duplicates()

    matched_rids = set(merged["_RID"].unique().tolist()) if (not merged.empty and "_RID" in merged.columns) else set()
    remaining = R[~R["_RID"].isin(matched_rids)].copy()

    remaining = remaining.drop(columns=["_RID", "_ALPHA_KEY"], errors="ignore")
    merged = merged.drop(columns=["_OUT_IDX", "_RID", "_ALPHA_KEY"], errors="ignore")

    return merged.reset_index(drop=True), remaining.reset_index(drop=True)

# ==========================================
#  PART 4D: NON-IHX NUMERIC SPLIT UNIQUE (GOOD HEALTH / VOLO / SBI)
#  - both sides treated numeric; split on / , & AND and; unique token match
# ==========================================

def step4_numeric_split_unique(step3_df: pd.DataFrame,
                               outstanding_path: Path,
                               deduplicate: bool = True,
                               match_mode: str = "NUMERIC_SPLIT_UNIQUE",
                               min_len: int = 6):

    if step3_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    if "Claim No" not in step3_df.columns:
        raise ValueError("Step 3 missing Claim No")

    out = parse_outstanding_excel_to_clean(outstanding_path).copy()
    if "Claim No" in out.columns:
        out = out.rename(columns={"Claim No": "Claim No_out"})

    L = out.copy().reset_index(drop=False).rename(columns={"index": "_OUT_IDX"}).copy()
    R = step3_df.copy().reset_index(drop=True).copy()
    R["_RID"] = np.arange(len(R))

    def _mis_nums(val):
        parts = _split_tokens_any(val)
        nums = _numeric_tokens_from_parts(parts, min_len=min_len)
        return nums

    R["_NUMS"] = R["Claim No"].apply(_mis_nums)
    R_exp = R[["_RID", "_NUMS"]].explode("_NUMS").dropna()
    R_exp["_NUMS"] = R_exp["_NUMS"].astype(str).str.strip()
    R_exp = R_exp[R_exp["_NUMS"] != ""].drop_duplicates()

    if R_exp.empty:
        R = R.drop(columns=["_RID", "_NUMS"], errors="ignore")
        return pd.DataFrame(), step3_df

    tok_to_rids = R_exp.groupby("_NUMS")["_RID"].apply(lambda x: set(x.tolist())).to_dict()

    picks = []
    for _, row in L.iterrows():
        parts = _split_tokens_any(row.get("Claim No_out", ""))
        toks = _numeric_tokens_from_parts(parts, min_len=min_len)
        if not toks:
            continue

        cand = set()
        ambiguous = False
        for t in toks:
            rids = tok_to_rids.get(str(t).strip())
            if not rids:
                continue
            cand |= rids
            if len(rids) > 1:
                ambiguous = True

        if ambiguous:
            continue
        if len(cand) == 1:
            picks.append({"_OUT_IDX": row["_OUT_IDX"], "_RID": list(cand)[0]})

    if not picks:
        return pd.DataFrame(), step3_df

    P = pd.DataFrame(picks).drop_duplicates(subset=["_OUT_IDX"])
    grp = P.groupby("_OUT_IDX")["_RID"].nunique().reset_index(name="_N")
    ok_out = set(grp.loc[grp["_N"] == 1, "_OUT_IDX"].tolist())
    P = P[P["_OUT_IDX"].isin(ok_out)].copy()

    L_pick = L.merge(P, on="_OUT_IDX", how="inner")
    merged = L_pick.merge(R, on="_RID", how="inner", suffixes=("_out", "_m3"))
    merged["MATCH_MODE"] = match_mode

    if deduplicate and not merged.empty:
        merged = merged.drop_duplicates()

    matched_rids = set(merged["_RID"].unique().tolist()) if (not merged.empty and "_RID" in merged.columns) else set()
    remaining = R[~R["_RID"].isin(matched_rids)].copy()

    remaining = remaining.drop(columns=["_RID", "_NUMS"], errors="ignore")
    merged = merged.drop(columns=["_OUT_IDX", "_RID", "_NUMS"], errors="ignore")

    return merged.reset_index(drop=True), remaining.reset_index(drop=True)

# ==========================================
#  PART 4E: CARE-SPECIFIC NUMERIC FALLBACK (AFTER OUT FILTER)
#  (UNCHANGED)
# ==========================================

def step4_care_numeric_fallback(unmatched_step3: pd.DataFrame,
                               outstanding_path: Path,
                               deduplicate: bool = True):

    if unmatched_step3.empty:
        return pd.DataFrame(), unmatched_step3

    out = parse_outstanding_excel_to_clean(outstanding_path).copy()

    if "Claim No" in out.columns:
        out = out.rename(columns={"Claim No": "Claim No_out"})

    _NULL_L = {'nan', 'none', 'null', 'na', 'n/a', ''}

    _PAREN_RE2 = re.compile(r"\([^)]*\)")
    _DOT_RE2 = re.compile(r"\..*$")
    _SUFFIX_RE2 = re.compile(r"([_-])(\d{1,2})$")

    def _strip_trailing_suffix2(s: str) -> str:
        m = _SUFFIX_RE2.search(s)
        if not m:
            return s
        n = int(m.group(2))
        if 0 <= n <= 20:
            return s[:m.start()]
        return s

    def _care_clean_base(val: str) -> str:
        s = str(val).replace("\xa0", " ").strip().upper()
        if s.lower() in _NULL_L:
            return ""
        if s.endswith(".0"):
            s = s[:-2]

        s = _PAREN_RE2.sub("", s)
        s = _DOT_RE2.sub("", s)
        s = _strip_trailing_suffix2(s)

        s = re.sub(r"^\s*CLAIM\s*NO[\s\-:]*", "", s)
        s = re.sub(r"^\s*CLAIM\s*NO\.\s*", "", s)
        s = re.sub(r"^\s*AL[\s\.]*", "", s)

        if re.search(r"\d", s):
            s = re.sub(r"^\s*[A-Z]+\s*", "", s)

        return s.strip()

    def _care_numeric_tokens(val) -> list:
        if pd.isna(val):
            return []
        s = _care_clean_base(val)
        if not s:
            return []

        parts = [p.strip() for p in re.split(r"[/,&]", s) if p.strip()]
        toks = []
        for p in parts:
            p = _strip_trailing_suffix2(p)
            nums = re.findall(r"\d{6,}", p)
            for n in nums:
                toks.append(n)
        return toks

    L = out.copy().reset_index(drop=False).rename(columns={"index": "_OUT_IDX"}).copy()
    R = unmatched_step3.copy().reset_index(drop=True).copy()

    R["_RID"] = np.arange(len(R))
    R["_CARE_TOKS"] = R["Claim No"].apply(_care_numeric_tokens)

    R_exp = R[["_RID", "_CARE_TOKS"]].explode("_CARE_TOKS").dropna()
    R_exp["_CARE_TOKS"] = R_exp["_CARE_TOKS"].astype(str).str.strip()
    R_exp = R_exp[R_exp["_CARE_TOKS"] != ""].drop_duplicates()

    if R_exp.empty:
        R = R.drop(columns=["_RID", "_CARE_TOKS"], errors="ignore")
        return pd.DataFrame(), R.drop(columns=["_RID", "_CARE_TOKS"], errors="ignore")

    tok_to_rids = R_exp.groupby("_CARE_TOKS")["_RID"].apply(lambda x: set(x.tolist())).to_dict()

    picks = []
    for _, row in L.iterrows():
        toks = _care_numeric_tokens(row.get("Claim No_out", ""))
        if not toks:
            continue

        cand = set()
        ambiguous = False
        for t in toks:
            rids = tok_to_rids.get(str(t).strip())
            if not rids:
                continue
            cand |= rids
            if len(rids) > 1:
                ambiguous = True

        if ambiguous:
            continue
        if len(cand) == 1:
            picks.append({"_OUT_IDX": row["_OUT_IDX"], "_RID": list(cand)[0]})

    if not picks:
        R = R.drop(columns=["_RID", "_CARE_TOKS"], errors="ignore")
        return pd.DataFrame(), R.drop(columns=["_RID", "_CARE_TOKS"], errors="ignore")

    P = pd.DataFrame(picks).drop_duplicates(subset=["_OUT_IDX"])

    grp = P.groupby("_OUT_IDX")["_RID"].nunique().reset_index(name="_N")
    ok_out = set(grp.loc[grp["_N"] == 1, "_OUT_IDX"].tolist())
    P = P[P["_OUT_IDX"].isin(ok_out)].copy()

    L_pick = L.merge(P, on="_OUT_IDX", how="inner")
    merged = L_pick.merge(
        R,
        on="_RID",
        how="inner",
        suffixes=("_out", "_m3")
    )
    merged["MATCH_MODE"] = "CARE_NUMERIC"

    if deduplicate and not merged.empty:
        merged = merged.drop_duplicates()

    matched_rids = set(merged["_RID"].unique().tolist()) if (not merged.empty and "_RID" in merged.columns) else set()
    remaining = R[~R["_RID"].isin(matched_rids)].copy()

    remaining = remaining.drop(columns=["_RID", "_CARE_TOKS"], errors="ignore")
    merged = merged.drop(columns=["_OUT_IDX", "_RID", "_CARE_TOKS"], errors="ignore")

    return merged.reset_index(drop=True), remaining.reset_index(drop=True)

# ==========================================
#  PART 4F: FHPL-SPECIFIC NUMERIC FALLBACK (UNCHANGED)
# ==========================================

def step4_fhpl_numeric_fallback(unmatched_step3: pd.DataFrame,
                                outstanding_path: Path,
                                deduplicate: bool = True):

    if unmatched_step3.empty:
        return pd.DataFrame(), unmatched_step3

    out = parse_outstanding_excel_to_clean(outstanding_path).copy()
    if "Claim No" in out.columns:
        out = out.rename(columns={"Claim No": "Claim No_out"})

    _NULL_L = {'nan', 'none', 'null', 'na', 'n/a', ''}

    def _mis_num(val) -> str:
        if pd.isna(val):
            return ""
        s = str(val).replace("\xa0", " ").strip()
        if s.lower() in _NULL_L:
            return ""
        if "/" in s:
            s = s.split("/")[0].strip()
        s = re.sub(r"[^0-9]", "", s)
        return s.strip()

    def _out_nums(val) -> list:
        if pd.isna(val):
            return []
        s = str(val).replace("\xa0", " ").strip()
        if s.strip().lower() in _NULL_L:
            return []
        parts = [p.strip() for p in re.split(r"[/,&]", s) if p.strip()]
        toks = []
        for p in parts:
            nums = re.findall(r"\d{6,}", p)
            toks.extend(nums)
        return toks

    L = out.copy().reset_index(drop=False).rename(columns={"index": "_OUT_IDX"})
    R = unmatched_step3.copy().reset_index(drop=True)

    R["_RID"] = np.arange(len(R))
    R["_MIS_NUM"] = R["Claim No"].apply(_mis_num)
    R = R[R["_MIS_NUM"] != ""].copy()
    if R.empty:
        return pd.DataFrame(), unmatched_step3

    num_to_rids = R.groupby("_MIS_NUM")["_RID"].apply(lambda x: set(x.tolist())).to_dict()

    picks = []
    for _, row in L.iterrows():
        toks = _out_nums(row.get("Claim No_out", ""))
        if not toks:
            continue

        cand = set()
        ambiguous = False
        for t in toks:
            rids = num_to_rids.get(str(t).strip())
            if not rids:
                continue
            cand |= rids
            if len(rids) > 1:
                ambiguous = True

        if ambiguous:
            continue
        if len(cand) == 1:
            picks.append({"_OUT_IDX": row["_OUT_IDX"], "_RID": list(cand)[0]})

    if not picks:
        R = R.drop(columns=["_RID", "_MIS_NUM"], errors="ignore")
        return pd.DataFrame(), unmatched_step3

    P = pd.DataFrame(picks).drop_duplicates(subset=["_OUT_IDX"])
    grp = P.groupby("_OUT_IDX")["_RID"].nunique().reset_index(name="_N")
    ok_out = set(grp.loc[grp["_N"] == 1, "_OUT_IDX"].tolist())
    P = P[P["_OUT_IDX"].isin(ok_out)].copy()

    L_pick = L.merge(P, on="_OUT_IDX", how="inner")
    merged = L_pick.merge(R, on="_RID", how="inner", suffixes=("_out", "_m3"))
    merged["MATCH_MODE"] = "FHPL_NUMERIC"

    if deduplicate and not merged.empty:
        merged = merged.drop_duplicates()

    matched_rids = set(merged["_RID"].unique().tolist()) if (not merged.empty and "_RID" in merged.columns) else set()
    remaining = R[~R["_RID"].isin(matched_rids)].copy()

    remaining = remaining.drop(columns=["_RID", "_MIS_NUM"], errors="ignore")
    merged = merged.drop(columns=["_OUT_IDX", "_RID", "_MIS_NUM"], errors="ignore")

    return merged.reset_index(drop=True), remaining.reset_index(drop=True)

# ==========================================
#  PART 4G: ICICI LOMBARD OUTSTANDING SPLIT FALLBACK (UNCHANGED)
# ==========================================

def step4_icici_lombard_split_fallback(unmatched_step3: pd.DataFrame,
                                       outstanding_path: Path,
                                       deduplicate: bool = True):

    if unmatched_step3.empty:
        return pd.DataFrame(), unmatched_step3

    out = parse_outstanding_excel_to_clean(outstanding_path).copy()
    if "Claim No" in out.columns:
        out = out.rename(columns={"Claim No": "Claim No_out"})

    L = out.copy().reset_index(drop=False).rename(columns={"index": "_OUT_IDX"}).copy()
    R = unmatched_step3.copy().reset_index(drop=True).copy()

    R["_RID"] = np.arange(len(R))
    R["_ALPHA_KEY"] = R["Claim No"].apply(_alpha_key)
    R = R[R["_ALPHA_KEY"] != ""].copy()
    if R.empty:
        return pd.DataFrame(), unmatched_step3

    key_to_rids = R.groupby("_ALPHA_KEY")["_RID"].apply(lambda x: set(x.tolist())).to_dict()

    def _out_split_alpha_tokens(val) -> list:
        parts = _split_tokens_any(val)
        return _alpha_tokens_from_parts(parts)

    picks = []
    for _, row in L.iterrows():
        toks = _out_split_alpha_tokens(row.get("Claim No_out", ""))
        if not toks:
            continue

        cand = set()
        ambiguous = False
        for t in toks:
            rids = key_to_rids.get(str(t).strip())
            if not rids:
                continue
            cand |= rids
            if len(rids) > 1:
                ambiguous = True

        if ambiguous:
            continue
        if len(cand) == 1:
            picks.append({"_OUT_IDX": row["_OUT_IDX"], "_RID": list(cand)[0]})

    if not picks:
        R = R.drop(columns=["_RID", "_ALPHA_KEY"], errors="ignore")
        return pd.DataFrame(), unmatched_step3

    P = pd.DataFrame(picks).drop_duplicates(subset=["_OUT_IDX"])
    grp = P.groupby("_OUT_IDX")["_RID"].nunique().reset_index(name="_N")
    ok_out = set(grp.loc[grp["_N"] == 1, "_OUT_IDX"].tolist())
    P = P[P["_OUT_IDX"].isin(ok_out)].copy()

    L_pick = L.merge(P, on="_OUT_IDX", how="inner")
    merged = L_pick.merge(R, on="_RID", how="inner", suffixes=("_out", "_m3"))
    merged["MATCH_MODE"] = "ICICI_LOMBARD_SPLIT"

    if deduplicate and not merged.empty:
        merged = merged.drop_duplicates()

    matched_rids = set(merged["_RID"].unique().tolist()) if (not merged.empty and "_RID" in merged.columns) else set()
    remaining = R[~R["_RID"].isin(matched_rids)].copy()

    remaining = remaining.drop(columns=["_RID", "_ALPHA_KEY"], errors="ignore")
    merged = merged.drop(columns=["_OUT_IDX", "_RID", "_ALPHA_KEY"], errors="ignore")

    return merged.reset_index(drop=True), remaining.reset_index(drop=True)

# ==========================================
#  OUTSTANDING FILTER REGISTRY (ALL 17 NON-IHX)
#  - explicit rules only (fail-fast if missing or zero rows)
# ==========================================

# Each entry is list of tokens; any token match => row kept
_OUT_FILTER_RULES: Dict[str, List[str]] = {
    "PARK MEDICLAIM INSURANCE TPA PRIVATE LIMITED": ["PARK"],
    "CARE HEALTH INSURANCE LIMITED": ["CARE"],
    "HEALTH INDIA INSURANCE TPA SERVICES PRIVATE LTD.": ["HEALTH INDIA", "HEALTH"],
    "FHPL": ["FAMILY"],
    "ICICI LOMBARD": ["ICICI", "LOMBARD"],
    "FUTURE GENERALI": ["FUTURE", "GENERALI"],
    "RELIANCE": ["RELIANCE"],
    "ERICSON": ["ERICSON", "ERIC"],
    "ADITYA BIRLA": ["ADITYA", "BIRLA"],
    "SAFEWAY INSURANCE TPA PVT.LTD": ["SAFEWAY"],
    "STAR HEALTH & ALLIED HEALTH INSURANCE CO.LTD.": ["STAR"],
    "VIDAL": ["VIDAL"],
    "HERITAGE HEALTH INSURANCE TPA PRIVATE LIMITED": ["HERITAGE"],
    "MEDSAVE HEALTHCARE TPA PVT LTD": ["MEDSAVE", "MED SAVE"],
    "GOOD HEALTH": ["GOOD HEALTH", "GOODHEALTH", "GOOD"],
    "VOLO HEALTH INSURANCE TPA PVT.LTD (EWA) (Mail Extract)": ["VOLO"],
    "SBI GENERAL": ["SBI"],
}

def filter_outstanding_for_tpa(out_clean: pd.DataFrame, tpa_choice: str) -> pd.DataFrame:
    if "Bill Company Name" not in out_clean.columns:
        raise ValueError("Outstanding cleaned is missing 'Bill Company Name' - cannot filter by TPA.")

    if tpa_choice == "IHX (Original MIS)":
        return out_clean.copy()

    if tpa_choice not in _OUT_FILTER_RULES:
        raise ValueError(
            f"Non-IHX TPA detected ('{tpa_choice}') but no TPA-wise outstanding filter rule is defined yet. "
            f"Add it to _OUT_FILTER_RULES before running."
        )

    tokens = [t.upper() for t in _OUT_FILTER_RULES[tpa_choice] if str(t).strip()]
    bn = out_clean["Bill Company Name"].astype(str).str.upper()

    mask = False
    for tok in tokens:
        mask = mask | bn.str.contains(tok, na=False)

    out_filt = out_clean.loc[mask].copy()

    if out_filt.empty:
        raise ValueError(
            f"TPA-wise outstanding filter produced 0 rows for TPA '{tpa_choice}'. "
            f"This is fail-fast to prevent cross-TPA matching chaos. "
            f"Check Bill Company Name values and update _OUT_FILTER_RULES tokens."
        )
    return out_filt.reset_index(drop=True)

# ==========================================
#  NON-IHX MATCH PLAN REGISTRY (TPA-GATED)
# ==========================================

def run_step4_plan(step3_df: pd.DataFrame, out_path: Path, tpa_choice: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns: (matched_df, remaining_unmatched_step3_df)
    """
    # Defaults: conservative alpha unique
    if tpa_choice == "IHX (Original MIS)":
        return step4_strict_matches(step3_df, out_path, deduplicate=True)

    if tpa_choice in [
        "PARK MEDICLAIM INSURANCE TPA PRIVATE LIMITED",
        "HEALTH INDIA INSURANCE TPA SERVICES PRIVATE LTD.",
        "FUTURE GENERALI",
        "RELIANCE",
        "ERICSON",
        "ADITYA BIRLA",
        "SAFEWAY INSURANCE TPA PVT.LTD",
        "STAR HEALTH & ALLIED HEALTH INSURANCE CO.LTD.",
        "HERITAGE HEALTH INSURANCE TPA PRIVATE LIMITED",
        "MEDSAVE HEALTHCARE TPA PVT LTD",
    ]:
        return step4_alpha_strict_unique(step3_df, out_path, deduplicate=True, match_mode="ALPHA_STRICT_UNIQUE")

    if tpa_choice == "VIDAL":
        # first split alpha unique (more permissive than strict alpha unique when OUT has compound claims)
        m1, rem = step4_split_alpha_unique(step3_df, out_path, deduplicate=True, match_mode="VIDAL_SPLIT_ALPHA_UNIQUE")
        if m1.empty:
            # fallback: still allow straight unique alpha
            return step4_alpha_strict_unique(step3_df, out_path, deduplicate=True, match_mode="VIDAL_ALPHA_UNIQUE")
        return m1, rem

    if tpa_choice in [
        "GOOD HEALTH",
        "VOLO HEALTH INSURANCE TPA PVT.LTD (EWA) (Mail Extract)",
        "SBI GENERAL",
    ]:
        return step4_numeric_split_unique(step3_df, out_path, deduplicate=True, match_mode="NUMERIC_SPLIT_UNIQUE", min_len=6)

    if tpa_choice == "CARE HEALTH INSURANCE LIMITED":
        # alpha unique first, then CARE numeric fallback on remaining
        m1, rem = step4_alpha_strict_unique(step3_df, out_path, deduplicate=True, match_mode="ALPHA_STRICT_UNIQUE")
        m2, rem2 = step4_care_numeric_fallback(rem, out_path, deduplicate=True)
        if not m2.empty:
            m1 = pd.concat([m1, m2], ignore_index=True)
        return m1, rem2

    if tpa_choice == "FHPL":
        # alpha unique first, then FHPL numeric fallback
        m1, rem = step4_alpha_strict_unique(step3_df, out_path, deduplicate=True, match_mode="ALPHA_STRICT_UNIQUE")
        m2, rem2 = step4_fhpl_numeric_fallback(rem, out_path, deduplicate=True)
        if not m2.empty:
            m1 = pd.concat([m1, m2], ignore_index=True)
        return m1, rem2

    if tpa_choice == "ICICI LOMBARD":
        # alpha unique first, then ICICI lombard split fallback
        m1, rem = step4_alpha_strict_unique(step3_df, out_path, deduplicate=True, match_mode="ALPHA_STRICT_UNIQUE")
        m2, rem2 = step4_icici_lombard_split_fallback(rem, out_path, deduplicate=True)
        if not m2.empty:
            m1 = pd.concat([m1, m2], ignore_index=True)
        return m1, rem2

    # If we ever reach here for non-IHX: block
    raise ValueError(
        f"Non-IHX TPA detected ('{tpa_choice}') but no Step-4 match plan is defined in run_step4_plan()."
    )
