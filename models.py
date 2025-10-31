from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List
from datetime import datetime

class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    email: Optional[str] = None
    full_name: Optional[str] = None
    
    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username must be alphanumeric (can include _ and -)')
        return v.lower()

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    username: Optional[str] = None

class UserInDB(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

class UserResponse(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    created_at: datetime

# NEW: Activity Log Models
class ActivityLog(BaseModel):
    """Model for tracking reconciliation activity"""
    username: str
    bank_type: str  # "ICICI" or "AXIS"
    step_completed: int  # 1, 2, 3, or 4
    run_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration_seconds: Optional[int] = None
    row_counts: Optional[Dict[str, int]] = None  # e.g., {"bank_rows": 247, "matches": 189}
    tpa_name: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None

class UserStatsResponse(BaseModel):
    """Response model for user statistics"""
    username: str
    total_reconciliations: int
    this_week: int
    this_month: int
    current_streak: int
    avg_duration_seconds: Optional[float] = None
    last_activity: Optional[datetime] = None
    
class ActivityResponse(BaseModel):
    """Response model for single activity"""
    bank_type: str
    step_completed: int
    run_id: str
    timestamp: datetime
    duration_seconds: Optional[int] = None
    row_counts: Optional[Dict[str, int]] = None
    tpa_name: Optional[str] = None
    success: bool

class DailyActivityResponse(BaseModel):
    """Response model for daily activity aggregation"""
    date: str  # YYYY-MM-DD format
    count: int