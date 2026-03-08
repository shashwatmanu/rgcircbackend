import os
import argparse
from datetime import datetime, timedelta
from bson import ObjectId
from pymongo import MongoClient
import gridfs
from dotenv import load_dotenv
import pytz

# Load environment variables
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "recon_db"

def list_and_delete_old_reconciliations(days_old=7, dry_run=True, limit=None):
    """
    List and optionally delete reconciliations older than specified days.
    
    Args:
        days_old: Delete reconciliations older than this many days
        dry_run: If True, only show what would be deleted
        limit: Maximum number of reconciliations to delete (None = all matching)
    """
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
        
        # Calculate cutoff date
        ist = pytz.timezone('Asia/Kolkata')
        cutoff_date = datetime.now(ist).replace(tzinfo=None) - timedelta(days=days_old)
        
        print(f"🔍 Finding reconciliations older than {days_old} days (before {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')})")
        
        # Find old reconciliations
        query = {"timestamp": {"$lt": cutoff_date}}
        old_recons = list(reconciliation_results.find(query).sort("timestamp", 1))
        
        if limit:
            old_recons = old_recons[:limit]
        
        if not old_recons:
            print("✅ No old reconciliations found.")
            return
        
        print(f"\n📋 Found {len(old_recons)} reconciliation(s) to delete:\n")
        
        total_size = 0
        for i, recon in enumerate(old_recons, 1):
            run_id = recon.get("run_id", "Unknown")
            username = recon.get("username", "Unknown")
            timestamp = recon.get("timestamp", datetime.now())
            bank_type = recon.get("bank_type", "Unknown")
            tpa_name = recon.get("tpa_name", "N/A")
            
            # Get file size if available
            file_size = 0
            if "zip_file_id" in recon:
                try:
                    file_id = ObjectId(recon["zip_file_id"]) if isinstance(recon["zip_file_id"], str) else recon["zip_file_id"]
                    file_info = db['fs.files'].find_one({"_id": file_id})
                    if file_info:
                        file_size = file_info.get("length", 0)
                        total_size += file_size
                except:
                    pass
            
            print(f"{i:3d}. {run_id}")
            print(f"     User: {username} | Date: {timestamp.strftime('%Y-%m-%d %H:%M')}")
            print(f"     Bank: {bank_type} | TPA: {tpa_name} | Size: {file_size / (1024*1024):.2f} MB")
            print()
        
        print(f"💾 Total storage to reclaim: {total_size / (1024*1024):.2f} MB")
        print(f"{'='*70}")
        
        if dry_run:
            print("\n[DRY RUN MODE] No files were deleted.")
            print("Run with --force to actually delete these reconciliations.")
        else:
            print("\n🚨 DELETE MODE ENGAGED. Deleting reconciliations...")
            deleted_count = 0
            deleted_size = 0
            
            for recon in old_recons:
                try:
                    # Delete GridFS file if exists
                    if "zip_file_id" in recon:
                        file_id = ObjectId(recon["zip_file_id"]) if isinstance(recon["zip_file_id"], str) else recon["zip_file_id"]
                        file_info = db['fs.files'].find_one({"_id": file_id})
                        if file_info:
                            deleted_size += file_info.get("length", 0)
                        fs.delete(file_id)
                    
                    # Delete reconciliation record
                    reconciliation_results.delete_one({"_id": recon["_id"]})
                    deleted_count += 1
                    
                    if deleted_count % 10 == 0:
                        print(f"   Deleted {deleted_count}/{len(old_recons)}...")
                        
                except Exception as e:
                    print(f"   ❌ Failed to delete {recon.get('run_id', 'Unknown')}: {e}")
            
            print(f"\n✅ Successfully deleted {deleted_count} reconciliation(s).")
            print(f"♻️  Reclaimed {deleted_size / (1024*1024):.2f} MB of storage space.")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete old reconciliations to free up storage")
    parser.add_argument("--days", type=int, default=7, help="Delete reconciliations older than this many days (default: 7)")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of reconciliations to delete (default: all matching)")
    parser.add_argument("--dry-run", action="store_true", help="Preview what would be deleted without actually deleting")
    parser.add_argument("--force", action="store_true", help="Actually delete the reconciliations")
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.force:
        print("⚠️  Please specify either --dry-run (to preview) or --force (to delete)")
        exit(1)
    
    is_dry_run = args.dry_run or not args.force
    
    if is_dry_run:
        print("🛡️  Running in DRY RUN mode.\n")
    else:
        print("⚠️  Running in DELETE mode.\n")
    
    list_and_delete_old_reconciliations(
        days_old=args.days,
        dry_run=is_dry_run,
        limit=args.limit
    )
