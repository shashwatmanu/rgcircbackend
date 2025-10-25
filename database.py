import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# Try to load .env for local development, but don't fail if it doesn't exist
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[Database] Loaded .env file for local development")
except ImportError:
    print("[Database] python-dotenv not installed, using system environment variables")
except Exception:
    print("[Database] No .env file found, using system environment variables")

# Get MongoDB URI from environment variable
MONGODB_URI = os.getenv("MONGODB_URI")

if not MONGODB_URI:
    print("[Database] ⚠️ MONGODB_URI not found. Authentication will be disabled.")
    client = None
    db = None
    users_collection = None
else:
    # Database name
    DB_NAME = "recon_db"

    # Initialize MongoDB client
    try:
        # Hide password in logs
        safe_uri = MONGODB_URI.split(':')[2].split('@')[1] if '@' in MONGODB_URI else "***"
        print(f"[MongoDB] Attempting to connect to: mongodb+srv://{safe_uri[:30]}...")
        
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
        client = None
        db = None
        users_collection = None
    except Exception as e:
        print(f"[MongoDB] ❌ Unexpected error: {e}")
        client = None
        db = None
        users_collection = None