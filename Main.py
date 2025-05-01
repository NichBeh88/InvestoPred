import streamlit as st
from navigation import navigation
import firebase_admin
from firebase_admin import firestore, credentials 

# Initialize Firebase
if not firebase_admin._apps:
    firebase_creds = st.secrets["FIREBASE"]
    cred = credentials.Certificate(firebase_creds)
    firebase_admin.initialize_app(cred)

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
