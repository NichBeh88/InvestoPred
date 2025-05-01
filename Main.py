import streamlit as st
from navigation import navigation
import firebase_admin
from firebase_admin import firestore, credentials 

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

# 🔒 Hide default Streamlit sidebar navigation
hide_streamlit_style = """
    <style>
        [data-testid="stSidebarNav"] { display: none; }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# 🧠 Initialize session state variables
st.session_state.setdefault("authenticated", False)
st.session_state.setdefault("logout_triggered", False)
st.session_state.setdefault("current_page", "Home-Page")
st.session_state.setdefault("redirected_once", False)

# 🔄 Restore session from Firestore if user info exists
if not st.session_state.get("authenticated") and st.session_state.get("user"):
    user_email = st.session_state["user"].get("email")
    if user_email:
        try:
            session_ref = db.collection("sessions").document(user_email)
            session_doc = session_ref.get()

            if session_doc.exists and session_doc.to_dict().get("authenticated"):
                st.session_state["authenticated"] = True
                st.session_state["current_page"] = "Home-Page"
        except Exception as e:
            st.error(f"Session restore failed: {e}")

# 🌐 Load navigation bar
navigation()

# 🚀 Redirect only once to Home-Page
if not st.session_state["redirected_once"]:
    st.session_state["redirected_once"] = True
    st.switch_page("pages/Home-Page.py")
