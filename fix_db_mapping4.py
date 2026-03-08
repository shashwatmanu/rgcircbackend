import sys
sys.path.append('/Users/apple/Desktop/RGCIRC/rgcircbackend')
from database import tpa_mappings_collection

docs = tpa_mappings_collection.find()
for d in docs:
    print(d.get("tpa_name"), d.get("mapping"))
