import sys
from database import tpa_mappings_collection

docs = tpa_mappings_collection.find()
for d in docs:
    print(d.get("tpa_name"), "->", d.get("cheque_utr_column"), "|", d.get("claim_no_column"))
