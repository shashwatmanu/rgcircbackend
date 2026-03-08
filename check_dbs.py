from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
URI = os.getenv("MONGODB_URI")
client = MongoClient(URI)

print("All databases and sizes:")
total_size = 0
for db_name in client.list_database_names():
    if db_name in ["admin", "local"]:
        continue
    db = client[db_name]
    stats = db.command("dbstats")
    size_mb = stats.get("dataSize", 0) / (1024 * 1024)
    storage_mb = stats.get("storageSize", 0) / (1024 * 1024)
    total_size += storage_mb
    print(f" - {db_name}: dataSize ~{size_mb:.2f} MB, storageSize ~{storage_mb:.2f} MB")

print(f"\nTotal storageSize ~{total_size:.2f} MB")
