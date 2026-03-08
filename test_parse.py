import sys
sys.path.append('/Users/apple/Desktop/RGCIRC/rgcircbackend')
from recon_service import parse_mis_universal
from pathlib import Path
m_path = Path('/Users/apple/Downloads/raw_mis_PaymentStatusPortalExportToExcel (6).xlsx')
try:
    m = parse_mis_universal(m_path, tpa_name="CARE HEALTH INSURANCE LIMITED")
    print("recon_service.py mis_clean rows:", len(m))
except Exception as e:
    print("recon_service error:", e)

sys.path.pop()
sys.path.append('/Users/apple/Downloads')
import untitled41
try:
    m2 = untitled41.parse_mis_universal(m_path, tpa_name="CARE HEALTH INSURANCE LIMITED")
    print("untitled41.py mis_clean rows:", len(m2))
except Exception as e:
    print("untitled41 error:", e)
