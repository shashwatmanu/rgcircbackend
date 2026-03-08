import sys
sys.path.append("/Users/apple/Desktop/RGCIRC/rgcircbackend")
import pandas as pd
from pathlib import Path
from recon_service import (
    clean_raw_bank_statement_axis,
    parse_mis_universal,
    step2_match_bank_mis_by_utr,
    parse_outstanding_excel_to_clean,
    classify_step3_matches,
    filter_outstanding_for_tpa,
    run_step4_plan,
    classify_step4_matches
)
b_path = Path("/Users/apple/Downloads/bank_0_Account_Statement_Report_28-01-2026_1053hrs (1) (1).xlsx")
m_path = Path("/Users/apple/Downloads/raw_mis_PaymentStatusPortalExportToExcel (6).xlsx")
out_path = Path("/Users/apple/Downloads/Outstanding_300126.xlsx")

b = clean_raw_bank_statement_axis(b_path)
m = parse_mis_universal(m_path, tpa_name="CARE HEALTH INSURANCE LIMITED")
s3_cand, s3_un = step2_match_bank_mis_by_utr(b, m, "CARE HEALTH INSURANCE LIMITED", True, 8)
print("Step 2 Match length:", len(s3_cand))
print("Step 3 Unmatched Bank:", len(s3_un))

s3_auto, s3_rev, s3_all, m3 = classify_step3_matches(s3_cand, "AXIS", "CARE", "run1")
print("Step 3 Auto:", len(s3_auto))

out_clean = parse_outstanding_excel_to_clean(out_path)
out_filt = filter_outstanding_for_tpa(out_clean, "CARE HEALTH INSURANCE LIMITED")
s4_cand, s4_un = run_step4_plan(s3_auto, out_filt, "CARE HEALTH INSURANCE LIMITED", keep_internal=True)
print("Step 4 CAND length:", len(s4_cand))

s4_auto, s4_rev, s4_all, m4 = classify_step4_matches(s4_cand, "AXIS", "CARE", "run1")
print("Step 4 AUTO:", len(s4_auto))
