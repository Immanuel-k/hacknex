import os
import logging
import hashlib
import time
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    import pymongo
    from pymongo import MongoClient
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://Hacknex:hacknex123@cluster0.x3xpr8k.mongodb.net/?retryWrites=true&w=majority"
)
DB_NAME = "deepfake_forensics_db"

class MongoDatabase:
    def __init__(self):
        self.client = None
        self.db = None
        self.reports_col = None
        self.users_col = None
        self._connect()

    def _connect(self):
        if not PYMONGO_AVAILABLE:
            logger.warning("PyMongo not installed. Database running in in-memory fallback mode.")
            return

        try:
            logger.info("Connecting to MongoDB Atlas Cluster0...")
            self.client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=4000)
            self.client.admin.command('ping')
            self.db = self.client[DB_NAME]
            self.reports_col = self.db["analysis_reports"]
            self.users_col = self.db["users"]
            logger.info("✅ SUCCESS: Connected to MongoDB Atlas Cloud Cluster0!")
        except Exception as e:
            logger.warning(f"Could not connect to MongoDB Atlas ({e}). Running in in-memory fallback mode.")
            self.client = None

    def _hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

    def register_user(self, email: str, password: str, name: str = "") -> Dict[str, Any]:
        """Registers a new user in MongoDB Atlas."""
        if self.users_col is not None:
            existing = self.users_col.find_one({"email": email.lower()})
            if existing:
                return {"success": False, "message": "User with this email already exists."}
            
            user_doc = {
                "email": email.lower(),
                "name": name or email.split("@")[0],
                "password_hash": self._hash_password(password),
                "auth_provider": "email",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            self.users_col.insert_one(user_doc)
            return {"success": True, "user": {"email": email, "name": user_doc["name"]}}
        return {"success": True, "user": {"email": email, "name": email.split("@")[0]}}

    def authenticate_user(self, email: str, password: str) -> Dict[str, Any]:
        """Authenticates user with email and password in MongoDB."""
        if self.users_col is not None:
            user = self.users_col.find_one({"email": email.lower()})
            if not user:
                return {"success": False, "message": "User not found."}
            if user.get("password_hash") == self._hash_password(password):
                return {"success": True, "user": {"email": user["email"], "name": user["name"]}}
            return {"success": False, "message": "Incorrect password."}
        return {"success": True, "user": {"email": email, "name": email.split("@")[0]}}

    def google_auth_login(self, email: str, name: str, google_id: str) -> Dict[str, Any]:
        """Logs in or registers user via Google OAuth / Gmail."""
        if self.users_col is not None:
            user = self.users_col.find_one({"email": email.lower()})
            if not user:
                user_doc = {
                    "email": email.lower(),
                    "name": name,
                    "google_id": google_id,
                    "auth_provider": "google",
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                self.users_col.insert_one(user_doc)
            return {"success": True, "user": {"email": email.lower(), "name": name}}
        return {"success": True, "user": {"email": email, "name": name}}

    def save_report(self, report_data: Dict[str, Any]) -> bool:
        if self.reports_col is not None:
            try:
                self.reports_col.insert_one(report_data.copy())
                return True
            except Exception as e:
                logger.error(f"Error saving to MongoDB: {e}")
                return False
        return False

    def get_recent_reports(self, limit: int = 10) -> list:
        if self.reports_col is not None:
            try:
                reports = list(self.reports_col.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
                return reports
            except Exception as e:
                logger.error(f"Error fetching from MongoDB: {e}")
                return []
        return []

mongo_db = MongoDatabase()
