import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import requests
import json
import time
import re
from datetime import datetime, timedelta, timezone

# Initialize Firebase
if not firebase_admin._apps:
    firebase_creds = st.secrets["FIREBASE"]
    cred = credentials.Certificate(firebase_creds)
    firebase_admin.initialize_app(cred)

db = firestore.client()
FIREBASE_API_KEY = st.secrets["FIREBASE"]["api_key"]

# 🔹 Enforce Strong Passwords
def is_strong_password(password):
    return (
        len(password) >= 8 and  
        re.search(r"[A-Z]", password) and  
        re.search(r"[a-z]", password) and  
        re.search(r"\d", password) and  
        re.search(r"[!@#$%^&*(),.?\":{}|<>]", password)
    )

# 🔹 Validate general email format (NO domain restrictions)
def is_valid_email(email):
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email))

# 🔹 Send Verification Email
def send_verification_email(user_id_token):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={FIREBASE_API_KEY}"
    
    payload = {
        "requestType": "VERIFY_EMAIL",
        "idToken": user_id_token
    }
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, data=json.dumps(payload), headers=headers)
    result = response.json()

    if "error" in result:
        return f"❌ Error: {result['error']['message']}"
    
    return "✅ Verification email sent! Please check your inbox."

# 🔹 Signup Function
def signup(email, password, confirm_password):
    if not email or not password or not confirm_password:
        return "All fields are required."

    if not is_valid_email(email):
        return "❌ Invalid email format."

    if password != confirm_password:
        return "Passwords do not match."

    if not is_strong_password(password):
        return "Password must be at least 8 characters long and include an uppercase letter, lowercase letter, number, and special character."

    try:
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_API_KEY}"
        payload = {"email": email, "password": password, "returnSecureToken": True}
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        result = response.json()

        if "idToken" in result:
            # Send verification email
            verification_message = send_verification_email(result["idToken"])

            db.collection("users").document(result["localId"]).set({"email": email, "emailVerified": False})
            st.success(f"✅ Signup successful! {verification_message}")
            time.sleep(1)  # Pause before redirection
            st.switch_page("pages/loginsignup.py")

        else:
            return f"❌ Error: {result['error']['message']}"

    except Exception as e:
        return str(e)

# 🔹 Login Function
def login(email, password):
    try:
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
        payload = {"email": email, "password": password, "returnSecureToken": True}
        headers = {"Content-Type": "application/json"}

        response = requests.post(url, data=json.dumps(payload), headers=headers)
        result = response.json()

        if "idToken" in result:
            # 🔹 Check email verification status
            user_info_url = f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={FIREBASE_API_KEY}"
            user_info_payload = {"idToken": result["idToken"]}
            user_info_response = requests.post(user_info_url, data=json.dumps(user_info_payload), headers=headers)
            user_info = user_info_response.json()

            if user_info.get("users", [])[0].get("emailVerified", False):
                # Store session in Firestore
                db.collection("sessions").document(email).set({"authenticated": True,
                                                               "last_active": datetime.now(timezone.utc).isoformat()})

                st.session_state["user"] = {"email": email}
                st.session_state["authenticated"] = True

                st.success("✅ Login successful! Redirecting...")
                time.sleep(1)
                st.switch_page("pages/Home-Page.py")
            else:
                st.error("❌ Please verify your email before logging in.")
        else:
            st.error("❌ Invalid email or password.")

    except Exception as e:
        st.error(f"⚠ Error: {str(e)}")


# 🔹 Logout Function (clear session in both Firestore and Streamlit)
def logout():
    # Clear the session in Firestore
    if "user" in st.session_state:
        db.collection("sessions").document(st.session_state["user"]["email"]).delete()

    # Clear session state in Streamlit
    st.session_state.clear()
    st.session_state["authenticated"] = False
    st.session_state["logout_triggered"] = True
    st.rerun()


# 🔹 Reset Password Function
def reset_password(email):
    try:
        # 🔹 Firebase REST API URL for password reset
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={FIREBASE_API_KEY}"
        
        # 🔹 Payload for sending password reset email
        payload = {
            "requestType": "PASSWORD_RESET",
            "email": email
        }
        headers = {"Content-Type": "application/json"}
        
        # 🔹 Send request to Firebase
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        result = response.json()

        # 🔹 Check for errors
        if "error" in result:
            st.error(f"❌ Error: {result['error']['message']}")
        else:
            st.success("✅ Password reset email sent! Please check your inbox.")

    except Exception as e:
        st.error(f"⚠ Error: {str(e)}")


INACTIVITY_TIMEOUT_MINUTES = 15

def track_session_activity():
    if "authenticated" in st.session_state and st.session_state["authenticated"]:
        from firebase_admin import firestore
        db = firestore.client()

        email = st.session_state.get("user", {}).get("email")
        if not email:
            logout_user()
            return

        doc_ref = db.collection("sessions").document(email)
        doc = doc_ref.get()

        now = datetime.now(timezone.utc)

        if doc.exists:
            session_data = doc.to_dict()
            last_active_str = session_data.get("last_active")
            if last_active_str:
                last_active = datetime.fromisoformat(last_active_str)
                if now - last_active > timedelta(minutes=INACTIVITY_TIMEOUT_MINUTES):
                    logout_user(message="⏳ Session expired due to inactivity.")
                    return

        # Update Firestore timestamp to now
        doc_ref.set({
            "authenticated": True,
            "last_active": now.isoformat()
        }, merge=True)

def logout_user(message="🚪 You have been logged out due to inactivity."):
    from firebase_admin import firestore
    db = firestore.client()
    email = st.session_state.get("user", {}).get("email")
    if email:
        db.collection("sessions").document(email).delete()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.warning(message)
    time.sleep(2)
    st.switch_page("pages/Home-Page.py")
