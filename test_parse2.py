import sys
import pandas as pd
from pathlib import Path

# Load bank and MIS
sys.path.append('/Users/apple/Desktop/RGCIRC/rgcircbackend')
import recon_service
b_path = Path('/Users/apple/Downloads/bank_0_Account_Statement_Report_28-01-2026_1053hrs (1) (1).xlsx')
m_path = Path('/Users/apple/Downloads/raw_mis_PaymentStatusPortalExportToExcel (6).xlsx')

b1 = recon_service.clean_raw_bank_statement_axis(b_path)
m1 = recon_service.parse_mis_universal(m_path, tpa_name="CARE HEALTH INSURANCE LIMITED")

m_auto_1, m_un_1 = recon_service.step2_match_bank_mis_by_utr(b1, m1, "CARE HEALTH INSURANCE LIMITED", True, 8)
print("recon_service match:", len(m_auto_1), "unmatched_bank:", len(m_un_1))

sys.path.append('/Users/apple/Downloads')
import untitled41
b2 = untitled41.clean_raw_bank_statement_axis(b_path)
m2 = untitled41.parse_mis_universal(m_path, tpa_name="CARE HEALTH INSURANCE LIMITED")
m_auto_2, m_un_2 = untitled41.step2_match_bank_mis_by_utr(b2, m2, "CARE HEALTH INSURANCE LIMITED", True, 8)
print("untitled41 match:", len(m_auto_2), "unmatched_bank:", len(m_un_2))
