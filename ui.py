import streamlit as st
import requests

st.set_page_config(page_title="Clinical Trials Eligibility Assistant", page_icon="🩺", layout="centered")

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
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/ask",
                    json={"query": query},
                    timeout=30,
                )
                if response.status_code == 200:
                    answer = response.json()["answer"]
                else:
                    answer = f"Something went wrong (status {response.status_code})."
            except requests.exceptions.ConnectionError:
                answer = "Can't reach the backend. Is the FastAPI server running on port 8000?"
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})