import ollama
import streamlit as st

st.set_page_config(
    page_title="Tax RAG Chat",
    page_icon="✨",
    initial_sidebar_state="expanded",
)

if "msgs" not in st.session_state:
    st.session_state["msgs"] = []

with st.sidebar:
    model: str = st.selectbox(
        label="Select model:",
        options=sorted(
            [i["model"] for i in ollama.list()["models"] if "embed" not in i["model"]],
        ),
    )

st.title(body="✨ Tax RAG Chat")

for msg in st.session_state["msgs"]:
    st.chat_message(name=msg["role"]).markdown(body=msg["content"])

if prompt := st.chat_input():
    st.session_state["msgs"].append({"role": "user", "content": prompt})

    st.chat_message(name="user").markdown(body=prompt)

    with st.chat_message(name="assistant"):
        response: str = st.write_stream(
            stream=(
                chunk["message"]["content"]
                for chunk in ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                )
            ),
        )

    st.session_state["msgs"].append({"role": "assistant", "content": response})
