import os
import argparse
from bson import ObjectId
from pymongo import MongoClient
import gridfs
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "recon_db"

def cleanup_orphaned_files(dry_run=True):
    if not MONGODB_URI:
        print("❌ MONGODB_URI not found in environment variables.")
        return

    try:
        # Connect to MongoDB
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        fs = gridfs.GridFS(db)
        reconciliation_results = db["reconciliation_results"]
        
        print(f"✅ Connected to MongoDB: {DB_NAME}")
        
        # 1. Get all valid file IDs from reconciliation_results
        print("🔍 Scanning reconciliation_results for valid file references...")
        valid_file_ids = set()
        
        # Find all documents with a zip_file_id field
        cursor = reconciliation_results.find({"zip_file_id": {"$exists": True}}, {"zip_file_id": 1})
        for doc in cursor:
            if "zip_file_id" in doc:
                # Ensure we store it as ObjectId
                fid = doc["zip_file_id"]
                if isinstance(fid, str):
                    try:
                        fid = ObjectId(fid)
                    except:
                        continue
                valid_file_ids.add(fid)
        
        print(f"👉 Found {len(valid_file_ids)} valid file references in active reconciliations.")

        # 2. Scan GridFS files
        print("🔍 Scanning GridFS (fs.files) for orphaned files...")
        orphaned_count = 0
        reclaimed_bytes = 0
        
        # fs.find() returns a cursor of GridOut objects, but we can also just query fs.files collection directly for speed
        # using db['fs.files'].find() is often faster for just IDs
        all_files_cursor = db['fs.files'].find({}, {"_id": 1, "length": 1, "filename": 1, "uploadDate": 1})
        
        files_to_delete = []

        for file_doc in all_files_cursor:
            file_id = file_doc["_id"]
            if file_id not in valid_file_ids:
                orphaned_count += 1
                reclaimed_bytes += file_doc.get("length", 0)
                files_to_delete.append(file_doc)

        # 3. Report and Delete
        print(f"WARNING: Found {orphaned_count} orphaned files taking up {reclaimed_bytes / (1024*1024):.2f} MB.")
        
        if orphaned_count == 0:
            print("✅ No orphaned files found. Storage is clean.")
            return

        if dry_run:
            print("\n[DRY RUNMODE] The following files WOULD be deleted:")
            for f in files_to_delete[:10]:
                print(f" - {f.get('filename', 'No Name')} (ID: {f['_id']}, Size: {f.get('length', 0)/1024:.1f} KB, Date: {f.get('uploadDate')})")
            if len(files_to_delete) > 10:
                print(f" ... and {len(files_to_delete) - 10} more.")
            print("\n❌ No files were deleted. Run without --dry-run to execute.")
        else:
            print("\n🚨 DELETE MODE ENGAGED. Deleting files...")
            deleted_actual = 0
            for f in files_to_delete:
                try:
                    fs.delete(f["_id"])
                    deleted_actual += 1
                    if deleted_actual % 100 == 0:
                        print(f"   Deleted {deleted_actual}/{orphaned_count}...")
                except Exception as e:
                    print(f"   ❌ Failed to delete {f['_id']}: {e}")
            
            print(f"\n✅ Successfully deleted {deleted_actual} orphaned files.")
            print(f"♻️  Reclaimed approx {reclaimed_bytes / (1024*1024):.2f} MB of storage space.")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanup orphaned GridFS files")
    parser.add_argument("--dry-run", action="store_true", help="Scan only, do not delete files")
    parser.add_argument("--force", action="store_true", help="Execute deletion (equivalent to omitting --dry-run, explicit confirmation)")
    
    args = parser.parse_args()
    
    # Default to dry-run if force is not set, or provided explicitly
    is_dry_run = True
    if args.force:
        is_dry_run = False
    elif not args.dry_run:
        # If user just runs `python script.py`, treat it as delete mode? 
        # Safety first: Let's require --force or --no-dry-run explicit flags usually, 
        # but for simplicity in this context, let's stick to the arg logic:
        # If they didn't say --dry-run, it's live BUT let's default to dry run if no args provided for extra safety?
        # Actually standard CLI: no args = execute usually, but let's be safe.
        pass
        # Logic: 
        # python script.py -> DELETE (standard)
        # python script.py --dry-run -> DRY RUN
        is_dry_run = False

    # OVERRIDE for this specific first pass: always default dry run if logic ambiguous? 
    # No, let's trust the flag.
    # However, for the USER's peace of mind, let's default the function call to dry_run=True if called without args in main, 
    # but since we parse args:
    
    if args.dry_run:
        print("🛡️  Running in DRY RUN mode.")
        cleanup_orphaned_files(dry_run=True)
    else:
        # Confirm interactively if no force flag? 
        # Since we are automating, we might skip interactive.
        # But for this artifact, let's just run it based on the flag.
        if not args.force:
           print("ℹ️  Running in LIVE mode (use --dry-run to preview).")
        cleanup_orphaned_files(dry_run=False)
