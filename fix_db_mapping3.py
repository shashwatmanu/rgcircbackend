import sys
sys.path.append('/Users/apple/Desktop/RGCIRC/rgcircbackend')
from database import tpa_mappings_collection

doc = tpa_mappings_collection.find_one({"tpa_name": "CARE HEALTH INSURANCE LIMITED"})
print("BEFORE:", doc.get("mapping"))

tpa_mappings_collection.update_one(
    {"tpa_name": "CARE HEALTH INSURANCE LIMITED"},
    {"$set": {"mapping.Cheque/ NEFT/ UTR No.": "Instrument/NEFT No"}}
)

new_doc = tpa_mappings_collection.find_one({"tpa_name": "CARE HEALTH INSURANCE LIMITED"})
print("AFTER:", new_doc.get("mapping"))
