import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
load_dotenv()

# Get MongoDB URI from environment variable
MONGODB_URI = os.getenv("MONGODB_URI")

if not MONGODB_URI:
    raise ValueError(
        "MONGODB_URI not found in environment variables. "
        "Please create a .env file with your MongoDB connection string."
    )

# Database name
DB_NAME = "recon_db"

# Initialize MongoDB client
try:
    print(f"[MongoDB] Attempting to connect to: {MONGODB_URI[:50]}...")
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    # Test connection
    client.admin.command('ping')
    print("[MongoDB] ✅ Connected successfully!")
    db = client[DB_NAME]
    users_collection = db["users"]
    
    # Create unique index on username
    users_collection.create_index("username", unique=True)
    print(f"[MongoDB] Database '{DB_NAME}' initialized")
    
except ConnectionFailure as e:
    print(f"[MongoDB] ❌ Connection failed: {e}")
    db = None
    users_collection = None
except Exception as e:
    print(f"[MongoDB] ❌ Unexpected error: {e}")
    db = None
    users_collection = None