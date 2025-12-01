from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.db import get_supabase
from loguru import logger
import hashlib

router = APIRouter()

class UserSignup(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

@router.post("/signup")
async def signup(user: UserSignup):
    supa = get_supabase()
    if not supa:
        raise HTTPException(status_code=500, detail="Database not available")
    
    try:
        # Check if user exists
        existing = supa.table("users").select("email").eq("email", user.email).execute()
        if existing.data:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create user
        hashed_password = hash_password(user.password)
        result = supa.table("users").insert({
            "email": user.email,
            "password": hashed_password
        }).execute()
        
        logger.info(f"User registered: {user.email}")
        return {"message": "User registered successfully"}
    
    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@router.post("/login")
async def login(user: UserLogin):
    supa = get_supabase()
    if not supa:
        raise HTTPException(status_code=500, detail="Database not available")
    
    try:
        hashed_password = hash_password(user.password)
        result = supa.table("users").select("*").eq("email", user.email).eq("password", hashed_password).execute()
        
        if not result.data:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        logger.info(f"User logged in: {user.email}")
        return {"message": "Login successful", "email": user.email}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")
