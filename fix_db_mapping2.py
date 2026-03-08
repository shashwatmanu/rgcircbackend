import sys
sys.path.append('/Users/apple/Desktop/RGCIRC/rgcircbackend')
from database import get_db

db = get_db()
tpa_col = db.get_collection("tpa_mappings")
doc = tpa_col.find_one({"tpa_name": "CARE HEALTH INSURANCE LIMITED"})
print("BEFORE:", doc.get("mapping"))

tpa_col.update_one(
    {"tpa_name": "CARE HEALTH INSURANCE LIMITED"},
    {"$set": {"mapping.Cheque/ NEFT/ UTR No.": "Instrument/NEFT No"}}
)

new_doc = tpa_col.find_one({"tpa_name": "CARE HEALTH INSURANCE LIMITED"})
print("AFTER:", new_doc.get("mapping"))

