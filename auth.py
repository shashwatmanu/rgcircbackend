import os
import warnings
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Suppress bcrypt version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="passlib")

# Try to load .env for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from database import users_collection
from models import TokenData, UserInDB

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    print("[Auth] ⚠️ WARNING: SECRET_KEY not set! Using insecure default.")
    SECRET_KEY = "insecure-default-key-change-this-in-production"

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# Password hashing with improved bcrypt configuration
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,  # Explicitly set rounds
    bcrypt__ident="2b"   # Use bcrypt 2b variant
)

# Bearer token scheme
security = HTTPBearer()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its hash"""
    try:
        # Bcrypt has a 72 byte limit - truncate if necessary
        if len(plain_password.encode('utf-8')) > 72:
            print(f"[Auth] Warning: Password truncated to 72 bytes for bcrypt")
            plain_password = plain_password[:72]
        
        result = pwd_context.verify(plain_password, hashed_password)
        print(f"[Auth] Password verification: {'✓ Success' if result else '✗ Failed'}")
        return result
    except Exception as e:
        print(f"[Auth] Password verification error: {e}")
        return False

def get_password_hash(password: str) -> str:
    """Hash a password"""
    try:
        # Bcrypt has a 72 byte limit - truncate if necessary
        if len(password.encode('utf-8')) > 72:
            print(f"[Auth] Warning: Password truncated to 72 bytes for bcrypt")
            password = password[:72]
        
        hashed = pwd_context.hash(password)
        print(f"[Auth] Password hashed successfully")
        return hashed
    except Exception as e:
        print(f"[Auth] Password hashing error: {e}")
        raise

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user_from_db(username: str) -> Optional[UserInDB]:
    """Get user from database by username"""
    if users_collection is None:
        return None
    
    user_doc = users_collection.find_one({"username": username})
    if user_doc:
        return UserInDB(**user_doc)
    return None

def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate user with username and password"""
    print(f"[Auth] Attempting to authenticate user: {username}")
    
    user = get_user_from_db(username)
    if not user:
        print(f"[Auth] User not found: {username}")
        return None
    
    print(f"[Auth] User found, verifying password...")
    if not verify_password(password, user.hashed_password):
        print(f"[Auth] Password verification failed for user: {username}")
        return None
    
    print(f"[Auth] ✓ Authentication successful for user: {username}")
    return user

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserInDB:
    """
    Dependency to get current authenticated user from JWT token.
    This is FastAPI's equivalent to Express middleware.
    
    Usage in route:
    @APP.post("/protected-route")
    async def protected_route(current_user: UserInDB = Depends(get_current_user)):
        # Only authenticated users can access this
        return {"user": current_user.username}
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError as e:
        print(f"[Auth] JWT decode error: {e}")
        raise credentials_exception
    
    user = get_user_from_db(username=token_data.username)
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    return user

async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
    """
    Additional dependency to ensure user is active.
    Can be chained: Depends(get_current_active_user)
    """
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user