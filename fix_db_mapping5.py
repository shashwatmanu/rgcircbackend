import sys
from pymongo import MongoClient
import certifi
import os
from dotenv import load_dotenv

load_dotenv("/Users/apple/Desktop/RGCIRC/rgcircbackend/.env")
db_url = os.getenv("MONGO_URI")
client = MongoClient(db_url, tlsCAFile=certifi.where())
db = client.get_database("recon_db")
tpa_col = db.get_collection("tpa_mappings")

care_doc = tpa_col.find_one({"tpa_name": "CARE HEALTH INSURANCE LIMITED"})
print("BEFORE UPDATE:", care_doc)

tpa_col.update_one(
    {"tpa_name": "CARE HEALTH INSURANCE LIMITED"},
    {"$set": {"cheque_utr_column": "Instrument/NEFT No"}}
)

new_doc = tpa_col.find_one({"tpa_name": "CARE HEALTH INSURANCE LIMITED"})
print("AFTER UPDATE:", new_doc)
