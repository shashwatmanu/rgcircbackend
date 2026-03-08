from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
URI = os.getenv("MONGODB_URI")
client = MongoClient(URI)
db = client["recon_db"]

print("Collections in recon_db:")
for coll_name in db.list_collection_names():
    count = db[coll_name].count_documents({})
    # Approximate size
    stats = db.command("collstats", coll_name)
    size_mb = stats.get("size", 0) / (1024 * 1024)
    print(f" - {coll_name}: {count} documents, ~{size_mb:.2f} MB")

print("\nAttempting to delete all activity_logs...")
print("activity_logs deleted:", db.activity_logs.delete_many({}).deleted_count)

# Check other databases in case they are consuming the quota
print("\nOther databases:")
for db_name in client.list_database_names():
    if db_name not in ["admin", "local", "recon_db"]:
        print(f" - WARNING: DB {db_name} exists, you might want to drop it.")
