import sys
import certifi
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv("/Users/apple/Desktop/RGCIRC/rgcircbackend/.env")
db_url = os.getenv("MONGO_URI")
client = MongoClient(db_url, tlsCAFile=certifi.where())
if client:
    print("Connected to Mongo")
db = client.get_database("recon_db")
tpa_col = db.get_collection("tpa_mappings")

care_doc = tpa_col.find_one({"tpa_name": "CARE HEALTH INSURANCE LIMITED"})
print("BEFORE UPDATE:", care_doc.get("mapping") if care_doc else "NOT FOUND")

if care_doc and "mapping" in care_doc:
    tpa_col.update_one(
        {"tpa_name": "CARE HEALTH INSURANCE LIMITED"},
        {"$set": {"mapping.Cheque/ NEFT/ UTR No.": "Instrument/NEFT No"}}
    )

new_doc = tpa_col.find_one({"tpa_name": "CARE HEALTH INSURANCE LIMITED"})
print("AFTER UPDATE:", new_doc.get("mapping") if new_doc else "NOT FOUND")
