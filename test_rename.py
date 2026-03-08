import sys
from pathlib import Path

sys.path.append('/Users/apple/Desktop/RGCIRC/rgcircbackend')
import recon_service
m_path = Path('/Users/apple/Downloads/raw_mis_PaymentStatusPortalExportToExcel (6).xlsx')
m1 = recon_service.parse_mis_universal(m_path, tpa_name="CARE HEALTH INSURANCE LIMITED")

tpa_mis_maps = recon_service.get_tpa_mis_maps()
mapping = tpa_mis_maps.get("CARE HEALTH INSURANCE LIMITED")

actual_lookup = {}
for c in m1.columns:
    k = recon_service._norm_colname(c)
    if k and (k not in actual_lookup):
        actual_lookup[k] = c

rename_map = {}
for canon_col, src_col in mapping.items():
    key = recon_service._norm_colname(src_col)
    if key in actual_lookup:
        rename_map[actual_lookup[key]] = canon_col

print("RECON_SERVICE RENAME MAP:", rename_map)

sys.path.pop()
sys.path.append('/Users/apple/Downloads')
import untitled41

mapping2 = untitled41.TPA_MIS_MAPS.get("CARE HEALTH INSURANCE LIMITED")
rename_map2 = {}
for canon_col, src_col in mapping2.items():
    key = untitled41._norm_colname(src_col)
    if key in actual_lookup:
        rename_map2[actual_lookup[key]] = canon_col

print("UNTITLED41 RENAME MAP:", rename_map2)
