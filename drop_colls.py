from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
URI = os.getenv("MONGODB_URI")
client = MongoClient(URI)
db = client["recon_db"]

print("Dropping heavy collections to reclaim storage...")
# Drop collections to instantly reclaim WiredTiger storage
db.drop_collection("activity_logs")
db.drop_collection("reconciliation_results")
db.drop_collection("fs.chunks")
db.drop_collection("fs.files")

print("Re-creating necessary indexes...")
db.activity_logs.create_index("username")
db.activity_logs.create_index("timestamp")
db.activity_logs.create_index([("username", 1), ("timestamp", -1)])

db.reconciliation_results.create_index("username")
db.reconciliation_results.create_index("run_id", unique=True)
db.reconciliation_results.create_index([("username", 1), ("timestamp", -1)])

TTL_SECONDS = 14 * 24 * 60 * 60
db.reconciliation_results.create_index(
    "timestamp",
    expireAfterSeconds=TTL_SECONDS
)

print("Drop and index recreation complete.")
