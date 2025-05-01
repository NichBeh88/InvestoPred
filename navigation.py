import streamlit as st
from auth import logout


def navigation():
    st.sidebar.title("📌 Navigation")
    st.sidebar.page_link("pages/Home-Page.py", label="🏠 Home Page")
    st.sidebar.page_link("pages/Stock-Chart&Data.py", label="📈 Stock Charts and Data")

    if st.session_state["authenticated"]:
        st.sidebar.page_link("pages/My-Account.py", label="🆔 My Account")
        st.sidebar.page_link("pages/Screen-and-Predict-FTSE100.py", label="🇬🇧 FTSE100 Screener and Predict")
        st.sidebar.page_link("pages/Screen-and-Predict-S&P500.py", label="🇺🇸 S&P500 Screener and Predict")
        st.sidebar.page_link("pages/Stocks-Comparison.py", label="📊 Stocks Comparison")
        st.sidebar.page_link("pages/Watchlist.py", label="📋 Watchlist")

        if st.sidebar.button("🚪 Logout"):
            logout()
    else:
        st.sidebar.page_link("pages/loginsignup.py", label="🔑 Login / Signup")
