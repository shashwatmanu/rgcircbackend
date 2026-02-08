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

def analyze_storage():
    if not MONGODB_URI:
        print("❌ MONGODB_URI not found in environment variables.")
        return

    try:
        # Connect to MongoDB
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        reconciliation_results = db["reconciliation_results"]
        
        print(f"✅ Connected to MongoDB: {DB_NAME}")
        
        # 1. Get all valid file IDs from reconciliation_results
        print("🔍 Scanning reconciliation_results for valid file references...")
        valid_file_ids = set()
        
        cursor = reconciliation_results.find({"zip_file_id": {"$exists": True}}, {"zip_file_id": 1})
        for doc in cursor:
            if "zip_file_id" in doc:
                fid = doc["zip_file_id"]
                if isinstance(fid, str):
                    try:
                         fid = ObjectId(fid)
                    except:
                        continue
                valid_file_ids.add(fid)
        
        print(f"👉 Found {len(valid_file_ids)} valid file references in active reconciliations.")

        # 2. Scan GridFS files
        print("🔍 Scanning GridFS (fs.files) ...")
        
        total_files = 0
        total_size = 0
        valid_files_count = 0
        valid_files_size = 0
        orphaned_files_count = 0
        orphaned_files_size = 0

        # fs.find() returns a cursor of GridOut objects, but we can also just query fs.files collection directly for speed
        all_files_cursor = db['fs.files'].find({}, {"_id": 1, "length": 1, "filename": 1, "uploadDate": 1})
        
        for file_doc in all_files_cursor:
            total_files += 1
            f_size = file_doc.get("length", 0)
            total_size += f_size
            
            file_id = file_doc["_id"]
            if file_id in valid_file_ids:
                valid_files_count += 1
                valid_files_size += f_size
            else:
                orphaned_files_count += 1
                orphaned_files_size += f_size

        # 3. Report
        def fmt_mb(bytes_val):
            return f"{bytes_val / (1024 * 1024):.2f} MB"

        print("\n📊 --- STORAGE ANALYSIS REPORT --- 📊")
        print(f"TOTAL Files:    {total_files} ({fmt_mb(total_size)})")
        print(f"-----------------------------------")
        print(f"✅ VALID Files:    {valid_files_count} ({fmt_mb(valid_files_size)})")
        print(f"⚠️  ORPHANED Files: {orphaned_files_count} ({fmt_mb(orphaned_files_size)})")
        print(f"-----------------------------------")
        
        if orphaned_files_count == 0:
            print("✅ System is CLEAN. No orphaned files found.")
        else:
            print(f"❌ Found {orphaned_files_count} orphaned files! Run cleanup script to remove them.")

        if valid_files_size > 400 * 1024 * 1024: # > 400MB
            print("\n💡 NOTE: Your VALID data is taking up significant space.")
            print("   This means your 40-day retention period might be keeping too many large files.")
            print("   Consider reducing TTL to 20 or 30 days if you need more space.")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    analyze_storage()
