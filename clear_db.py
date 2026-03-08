from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
URI = os.getenv("MONGODB_URI")
client = MongoClient(URI)
db = client["recon_db"]

print("Clearing data...")
# Clear reconciliation results which take up text/metadata space
print("reconciliation_results deleted:", db.reconciliation_results.delete_many({}).deleted_count)

# Clear large files from GridFS
print("fs.chunks deleted:", db["fs.chunks"].delete_many({}).deleted_count)
print("fs.files deleted:", db["fs.files"].delete_many({}).deleted_count)

print("DB clear complete.")
