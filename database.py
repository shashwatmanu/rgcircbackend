import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import gridfs
from datetime import datetime, timedelta

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
    activity_logs_collection = None
    reconciliation_results_collection = None
    fs = None
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
        activity_logs_collection = db["activity_logs"]
        reconciliation_results_collection = db["reconciliation_results"]  # NEW
        
        # Initialize GridFS for file storage
        fs = gridfs.GridFS(db)  # NEW
        
        # Create unique index on username
        users_collection.create_index("username", unique=True)
        
        # Create indexes for activity logs (for fast queries)
        activity_logs_collection.create_index("username")
        activity_logs_collection.create_index("timestamp")
        activity_logs_collection.create_index([("username", 1), ("timestamp", -1)])
        
        # Create indexes for reconciliation results (NEW)
        reconciliation_results_collection.create_index("username")
        reconciliation_results_collection.create_index("run_id", unique=True)
        reconciliation_results_collection.create_index([("username", 1), ("timestamp", -1)])
        
        # Ensure TTL index reflects desired retention policy (40 days)
        TTL_SECONDS = 40 * 24 * 60 * 60
        existing_indexes = reconciliation_results_collection.index_information()
        timestamp_index = existing_indexes.get("timestamp_1")
        if timestamp_index:
            existing_ttl = timestamp_index.get("expireAfterSeconds")
            if existing_ttl != TTL_SECONDS:
                reconciliation_results_collection.drop_index("timestamp_1")
                print(f"[MongoDB] Dropped stale TTL index (had {existing_ttl}s, want {TTL_SECONDS}s)")
        reconciliation_results_collection.create_index(
            "timestamp",
            expireAfterSeconds=TTL_SECONDS
        )
        print(f"[MongoDB] ✅ TTL index set to auto-delete after {TTL_SECONDS // (24 * 60 * 60)} days")
        
        print(f"[MongoDB] Database '{DB_NAME}' initialized with collections:")
        print(f"  - users")
        print(f"  - activity_logs")
        print(f"  - reconciliation_results (TTL: 40 days)")
        print(f"  - fs.files (GridFS)")
        print(f"  - fs.chunks (GridFS)")
        
    except ConnectionFailure as e:
        print(f"[MongoDB] ❌ Connection failed: {e}")
        client = None
        db = None
        users_collection = None
        activity_logs_collection = None
        reconciliation_results_collection = None
        fs = None
    except Exception as e:
        print(f"[MongoDB] ❌ Unexpected error: {e}")
        client = None
        db = None
        users_collection = None
        activity_logs_collection = None
        reconciliation_results_collection = None
        fs = None