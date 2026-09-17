import streamlit as st
from app import rag_agentic, load_data
import app as app_module

st.set_page_config(page_title="Clinical Trials Eligibility Assistant", page_icon="🩺", layout="centered")

if "data_loaded" not in st.session_state:
    with st.spinner("Loading trial data..."):
        load_data()
    st.session_state.data_loaded = True

st.title("🩺 Clinical Trials Eligibility Assistant")
st.caption("Ask about clinical trials, or check age eligibility for a specific trial (mention its NCT ID, e.g. NCT07045896).")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = rag_agentic(query)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})