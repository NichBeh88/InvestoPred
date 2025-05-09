import streamlit as st
import firebase_admin
from firebase_admin import auth, credentials, firestore
from auth import is_strong_password, logout, track_session_activity
from navigation import navigation
import time
import requests

track_session_activity()

# Hide Streamlit's default multipage navigation
hide_streamlit_style = """
    <style>
        [data-testid="stSidebarNav"] { display: none; }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Check if Firebase has already been initialized
if not firebase_admin._apps:
    # Access the private key and other credentials from Streamlit secrets
    private_key = st.secrets["FIREBASE"]["private_key"]
    client_email = st.secrets["FIREBASE"]["client_email"]
    project_id = st.secrets["FIREBASE"]["project_id"]
    
    # Create a dictionary with the credentials
    service_account_info = {
        "type": "service_account",
        "project_id": project_id,
        "private_key": private_key,
        "client_email": client_email,
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-fbsvc%40investopred.iam.gserviceaccount.com"
    }
    
    # Initialize Firebase with the credentials
    cred = credentials.Certificate(service_account_info)
    firebase_admin.initialize_app(cred)
    
else:
    print("Firebase app is already initialized.")

db = firestore.client()
FIREBASE_API_KEY = st.secrets["FIREBASE"]["api_key"]


# 🛡 Initialize session keys safely
if "user" not in st.session_state:
    st.session_state["user"] = None
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# 🔍 Attempt to restore session from Firestore if not authenticated but session exists in DB
if not st.session_state["authenticated"]:
    try:
        # Check if email exists in session (e.g. stored in local storage previously)
        # In full production you'd pass session id in cookies or similar
        session_docs = db.collection("sessions").stream()
        for doc in session_docs:
            session_data = doc.to_dict()
            if session_data.get("authenticated", False):
                st.session_state["authenticated"] = True
                st.session_state["user"] = {"email": doc.id}
                break
    except Exception as e:
        st.error(f"⚠ Failed to restore session: {e}")

# Check if the user is authenticated before showing the page content
if not st.session_state.get("authenticated", False):
    st.error("Please log in to view this page.")
    st.sidebar.title("📌 Navigation")
    st.sidebar.page_link("pages/Home-Page.py", label="🏠 Home Page")
    st.sidebar.page_link("pages/Stock-Chart&Data.py", label="📈 Stock Charts and Data")
    st.sidebar.page_link("pages/loginsignup.py", label="🔑 Login / Signup")
    st.stop()

navigation()

# Redirect to Home Page if logout was triggered
if st.session_state.get("logout_triggered", False):
    st.session_state["logout_triggered"] = False  # Reset trigger
    st.switch_page("pages/Home-Page.py")  # Redirect immediately

# Function to reauthenticate user with Firebase REST API
def verify_current_password(email, password):
    api_key = "AIzaSyB3crLhI50plAg6maUDa0AdWgj7arWOcuo"  
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"

    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        return True
    else:
        return False

# Function to reauthenticate the user with their current password
def reauthenticate_user(current_password):
    try:
        # Get the user by email
        user_email = st.session_state["user"]["email"]
        user = auth.get_user_by_email(user_email)
        
        # Reauthenticate the user (Firebase doesn't allow updating password without reauthentication)
        # Firebase reauthentication can be tricky with passwords, so this will involve email/password sign-in method
        # You'll need to handle reauthentication on the client side for security
        # Here we simulate it by requiring the current password and moving forward
        if current_password:  # Simulate that the password is correct if user enters one
            return True
        else:
            st.error("❌ Incorrect current password.")
            return False
    except firebase_admin.auth.UserNotFoundError:
        st.error("❌ User not found.")
        return False

# Function to send verification email
def send_verification_email():
    try:
        user_email = st.session_state["user"]["email"]
        user = auth.get_user_by_email(user_email)
        if not user.email_verified:
            # Send the verification email
            auth.generate_email_verification_link(user_email)
            st.success(f"🔑 A verification email has been sent to {user_email}. Please verify your email before changing the password.")
            return False
        return True
    except Exception as e:
        st.error(f"⚠ Error sending verification email: {str(e)}")
        return False

# Change password function
def change_password(current_password, new_password, confirm_new_password):
    user_email = st.session_state["user"]["email"]

    # Step 1: Verify current password correctly
    if not verify_current_password(user_email, current_password):
        st.error("❌ Incorrect current password. Please try again.")
        return

    # Step 2: Validate new password strength
    if not is_strong_password(new_password):
        st.error("⚠ Password must be at least 8 characters long and include an uppercase letter, lowercase letter, number, and special character.")
        return

    # Step 3: Check new password is not same as current password and new password same as confirm password
    if current_password == new_password:
        st.error("❌ New password cannot be the same as the current password.")
        return
    
    if new_password != confirm_new_password:
        st.error("❌ New password cannot be the same as the current password.")
        return
    
    # Step 4: Send email verification link before changing
    try:
        user = auth.get_user_by_email(user_email)
        if not user.email_verified:
            link = auth.generate_email_verification_link(user_email)
            st.success(f"🔑 A verification email has been sent to {user_email}. Please verify your email before proceeding.")
            st.write(f"👉 [Click here to verify your email]({link})")
            return  # Stop here until verified
    except Exception as e:
        st.error(f"⚠ Error sending verification email: {str(e)}")
        return

    # Step 5: Actually update password
    try:
        auth.update_user(user.uid, password=new_password)
        st.success("✅ Password updated successfully! Please log in again.")
        time.sleep(1)
        logout()
    except Exception as e:
        st.error(f"⚠ Error updating password: {str(e)}")

# Protect account page
def require_login():
    if "authenticated" not in st.session_state:
        st.error("🚨 You must be logged in to access this page.")
        st.stop()

require_login()  # Ensure user is logged in before showing content

# Display user account details
st.title("🔐 Account Settings")

email = st.session_state["user"]["email"]
st.write(f"**Email:** {email}")
st.write("**Password:** 🔒 ********")  # Blurred password for security

st.caption("**Take Note** ❗️", help="Fill in correct Current Password, DO NOT fill in the SAME New Password as Current Password and Fill in correct new password in *Confirm New Password*. Click the *Update Password* when done.")  

st.subheader("Change Password")
current_password = st.text_input("Current Password", type="password")
new_password = st.text_input("New Password", type="password")
confirm_new_password = st.text_input("Confirm New Password", type="password")

if st.button("Update Password"):
    change_password(current_password, new_password, confirm_new_password)
    st.switch_page("pages/loginsignup.py")
