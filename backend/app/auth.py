import os
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from bson import ObjectId
from email_validator import EmailNotValidError, validate_email
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from pymongo.errors import DuplicateKeyError

from .db import users

JWT_SECRET = os.environ["JWT_SECRET"]
JWT_EXPIRE_HOURS = int(os.environ.get("JWT_EXPIRE_HOURS", "24"))

router = APIRouter(prefix="/api/auth", tags=["auth"])
bearer = HTTPBearer(auto_error=False)


class SignupIn(BaseModel):
    name: str = ""
    email: str = ""
    password: str = ""
    confirm_password: str = ""


class LoginIn(BaseModel):
    email: str = ""
    password: str = ""


def public_user(u: dict) -> dict:
    """What the frontend is allowed to see (never the password hash)."""
    return {
        "id": str(u["_id"]),
        "name": u["name"],
        "email": u["email"],
        "register": u.get("register"),
        "depth": u.get("depth"),
        "onboarding_completed": u.get("onboarding_completed", False),
    }


def make_token(user_id) -> str:
    payload = {
        "sub": str(user_id),
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def bad_request(msg: str, code: int = 400):
    raise HTTPException(status_code=code, detail=msg)


@router.post("/signup", status_code=201)
def signup(body: SignupIn):
    name = body.name.strip()
    email = body.email.strip().lower()

    if not name or not email or not body.password:
        bad_request("Please fill in all fields with valid values.")
    try:
        email = validate_email(email, check_deliverability=False).normalized
    except EmailNotValidError:
        bad_request("Please fill in all fields with valid values.")
    if len(body.password) < 8 or body.password != body.confirm_password:
        bad_request("The password must be at least eight characters long and match the confirmation.")
    if len(body.password.encode()) > 72:  # bcrypt only uses the first 72 bytes
        bad_request("The password is too long.")

    # bcrypt generates a random salt and stores it inside the hash string.
    password_hash = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode()

    doc = {
        "name": name,
        "email": email,
        "password_hash": password_hash,
        "register": None,  # set by the onboarding quiz
        "depth": None,     # set by the onboarding quiz
        "onboarding_completed": False,
        "created_at": datetime.now(timezone.utc),
        "last_login": None,
    }
    try:
        result = users.insert_one(doc)
    except DuplicateKeyError:
        bad_request("An account with this email address already exists.", 409)

    doc["_id"] = result.inserted_id
    return {"token": make_token(doc["_id"]), "user": public_user(doc)}


@router.post("/login")
def login(body: LoginIn):
    email = body.email.strip().lower()
    if not email or not body.password:
        bad_request("Please fill in all fields.")

    user = users.find_one({"email": email})
    pw = body.password.encode()
    if not user or len(pw) > 72 or not bcrypt.checkpw(pw, user["password_hash"].encode()):
        bad_request("Incorrect email or password entered.", 401)

    users.update_one({"_id": user["_id"]}, {"$set": {"last_login": datetime.now(timezone.utc)}})
    return {"token": make_token(user["_id"]), "user": public_user(user)}


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict:
    """Use as a dependency on any route that needs a logged-in user."""
    if creds is None:
        bad_request("Not logged in.", 401)
    try:
        payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=["HS256"])
        user = users.find_one({"_id": ObjectId(payload["sub"])})
    except Exception:
        user = None
    if not user:
        bad_request("Session expired, please log in again.", 401)
    return user


@router.get("/me")
def me(user: dict = Depends(current_user)):
    return public_user(user)