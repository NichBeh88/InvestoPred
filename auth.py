import os
import requests
import json
import firebase_admin
from firebase_admin import auth as firebase_auth, credentials, firestore
from flask import session
from datetime import datetime

def init_firebase():
    if not firebase_admin._apps:
        raw_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
        if not raw_json:
            raise ValueError("FIREBASE_CREDENTIALS_JSON not found in environment.")
        service_account_info = json.loads(raw_json)
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred)

# ✅ Create user with verification email
def create_user(email, password):
    try:
        user = firebase_auth.create_user(email=email, password=password)
        firebase_auth.send_email_verification(user.uid)
        return True, None
    except Exception as e:
        return False, str(e)

# ✅ Login only if verified
def verify_password_and_login(email, password):
    try:
        FIREBASE_API_KEY = os.environ.get("FIREBASE_API_KEY")
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
        data = {
            "email": email,
            "password": password,
            "returnSecureToken": True
        }
        response = requests.post(url, json=data).json()

        id_token = response.get("idToken")
        if not id_token:
            return False

        user = firebase_auth.get_user_by_email(email)
        if not user.email_verified:
            return "unverified"

        # ✅ Store user session
        session["user"] = {"email": email}
        session["last_activity"] = datetime.utcnow().isoformat()
        return True

    except Exception as e:
        print("Login error:", e)
        return False

# ✅ Get user from session cookie
def get_user_from_session_cookie():
    return session.get("user")

# ✅ Clear session on logout
def track_session_activity():
    session["last_activity"] = datetime.utcnow().isoformat()
