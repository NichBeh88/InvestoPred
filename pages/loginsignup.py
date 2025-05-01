import streamlit as st
from auth import login, signup, reset_password, logout
import time
from navigation import navigation

# ✅ Page-level session safety
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# 🔹 Hide Streamlit's default sidebar navigation
hide_streamlit_style = """
    <style>
        [data-testid="stSidebarNav"] { display: none; }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

navigation()

# Redirect logged-in users away from the auth page
if st.session_state.get("authenticated", False):
    st.success("✅ You are already logged in! Redirecting to Home Page...")
    time.sleep(1)
    st.switch_page("pages/Home-Page.py")
    st.stop()

st.title("🔐 Login / Signup")

# Tabs for Login, Signup, and Forgot Password
tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Forgot Password"])

with tab1:
    st.subheader("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        login(email, password)

with tab2:
    st.subheader("Sign Up")
    email = st.text_input("Email", key="signup_email")
    password = st.text_input("Password", type="password", key="signup_pass")
    confirm_password = st.text_input("Confirm Password", type="password")
    if st.button("Sign Up"):
        message = signup(email, password, confirm_password)
        if "successful" in message:
            st.success(message)
            time.sleep(1)
            st.switch_page("pages/Home-Page.py")
        else:
            st.error(message)

with tab3:
    st.subheader("Forgot Password")
    email = st.text_input("Enter your email")
    if st.button("Reset Password"):
        reset_password(email)
