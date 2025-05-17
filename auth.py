import firebase_admin
from firebase_admin import credentials, auth
from flask import request, session
import json
import requests
import os

firebase_app = None

def init_firebase():
    if not firebase_admin._apps:
        raw_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
        if not raw_json:
            raise ValueError("FIREBASE_CREDENTIALS_JSON not found in environment.")
        service_account_info = json.loads(raw_json)
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred)

# 🔐 Helper to get user from session cookie
def get_user_from_session_cookie():
    user_email = session.get("user_email")
    if user_email:
        return {"email": user_email}  # Later can include display name, uid, etc.
    return None

# 🔐 Verify Firebase ID token and store in session (called from your login POST route)
def verify_token_and_login(id_token):
    try:
        decoded_token = auth.verify_id_token(id_token)
        user_email = decoded_token["email"]
        session["user_email"] = user_email
        return True
    except Exception as e:
        print("Login failed:", e)
        return False


def verify_password_and_login(email, password):
    try:
        api_key = os.environ.get("FIREBASE_API_KEY")
        if not api_key:
            print("❌ Firebase API key is missing!")
            return False

        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
        payload = {
            "email": email,
            "password": password,
            "returnSecureToken": True
        }
        response = requests.post(url, json=payload)

        print("📡 Firebase Response:", response.status_code, response.json())

        if response.status_code == 200:
            session["user_email"] = email
            return True
        else:
            return False
    except Exception as e:
        print("❌ Exception in login:", e)
        return False


def create_user(email, password):
    try:
        user = auth.create_user(email=email, password=password)
        print("✅ Created new user:", user.uid)
        return True, None
    except Exception as e:
        print("❌ Signup error:", e)
        return False, str(e)



