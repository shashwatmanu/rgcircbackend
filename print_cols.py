import sys
sys.path.append('/Users/apple/Desktop/RGCIRC/rgcircbackend')
from recon_service import parse_mis_universal
from pathlib import Path
m_path = Path('/Users/apple/Downloads/raw_mis_PaymentStatusPortalExportToExcel (6).xlsx')
m = parse_mis_universal(m_path, tpa_name="CARE HEALTH INSURANCE LIMITED")
print("parse_mis_universal COLUMNS:", m.columns.tolist())
