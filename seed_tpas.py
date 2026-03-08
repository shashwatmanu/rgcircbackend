import os
import pytz
from datetime import datetime
from database import tpa_mappings_collection
from recon_service import FALLBACK_TPA_MIS_MAPS

if __name__ == "__main__":
    if tpa_mappings_collection is None:
        print("Database not configured. Exiting.")
        exit(1)

    print(f"Found {len(FALLBACK_TPA_MIS_MAPS)} fallback TPAs. Migrating to database...")

    inserted_count = 0
    skipped_count = 0
    
    for tpa_name, mapping in FALLBACK_TPA_MIS_MAPS.items():
        existing = tpa_mappings_collection.find_one({"tpa_name": tpa_name})
        if not existing:
            ist = pytz.timezone('Asia/Kolkata')
            now = datetime.now(ist).replace(tzinfo=None)
            tpa_mappings_collection.insert_one({
                "tpa_name": tpa_name,
                "cheque_utr_column": mapping.get("Cheque/ NEFT/ UTR No.", ""),
                "claim_no_column": mapping.get("Claim No", ""),
                "created_at": now,
                "updated_at": now
            })
            inserted_count += 1
            print(f"Inserted: {tpa_name}")
        else:
            skipped_count += 1
            print(f"Skipped (already exists): {tpa_name}")

    print(f"Seeding complete. Inserted: {inserted_count}, Skipped: {skipped_count}.")
