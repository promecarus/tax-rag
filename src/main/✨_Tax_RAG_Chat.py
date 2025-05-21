import chromadb
import ollama
import streamlit as st
from utils import get_augmented_prompt, get_timestamp, profile_card

st.set_page_config(
    page_title="Tax RAG Chat",
    page_icon="✨",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    if st.button(label="Chat baru", use_container_width=True):
        st.session_state["msgs"] = []

        if (user := st.user)["is_logged_in"]:
            st.query_params["ts"] = get_timestamp()

if (user := st.user)["is_logged_in"]:
    st.query_params["ts"] = st.query_params.get("ts", default=get_timestamp())

    # st.write(user["sub"])
    # st.write(st.query_params["ts"])

if "msgs" not in st.session_state:
    st.session_state["msgs"] = []

with st.sidebar:
    profile_card()

    with st.expander(label="Konfigurasi Chat"):
        if st.button(label="Bersihkan chat", use_container_width=True):
            st.session_state["msgs"] = []

        st.session_state["model"] = st.selectbox(
            label="Pilih model:",
            options=(
                models := sorted(
                    [
                        i["model"]
                        for i in ollama.list()["models"]
                        if "embed" not in i["model"]
                    ],
                )
            ),
            index=models.index("granite3.3:2b"),
            help="Pilih model untuk digunakan pada chat.",
        )

        st.session_state["n_results"] = st.number_input(
            label="Jumlah dokumen:",
            min_value=1,
            max_value=10,
            value=1,
            help="Dokumen teratas yang digunakan untuk menjawab (1-10)",
        )

        collection: chromadb.Collection = chromadb.PersistentClient(
            path=".chroma",
        ).get_collection(name="tax-rag")

        st.session_state["include"] = st.multiselect(
            label="Status peraturan yang disertakan",
            options=sorted(
                {x["status_dokumen"] for x in collection.get()["metadatas"]},
            ),
            default=["Berlaku"],
            help="Status peraturan yang disertakan dalam pencarian.",
        )

        st.session_state["show_retrieved"] = st.toggle(
            label="Tampilkan pencarian",
            value=True,
            help="Tampilkan dokumen yang diambil pada proses pencarian.",
        )

        st.session_state["show_augmented"] = st.toggle(
            label="Tampilkan prompt",
            value=True,
            help="Tampilkan prompt yang dihasilkan dari dokumen yang diambil.",
        )

st.title(body="✨ Tax RAG Chat")

for i, msg in enumerate(iterable=st.session_state["msgs"]):
    with st.chat_message(name=msg["role"]):
        if msg["role"] == "assistant":
            get_augmented_prompt(
                prompt=st.session_state["msgs"][i - 1]["content"],
                query_result=collection.query(
                    query_texts=st.session_state["msgs"][i - 1]["content"],
                    n_results=st.session_state["n_results"],
                    where={"status_dokumen": {"$in": st.session_state["include"]}},
                ),
            )

        st.markdown(body=msg["content"])

if prompt := st.chat_input():
    st.session_state["msgs"].append({"role": "user", "content": prompt})

    st.chat_message(name="user").markdown(body=prompt)

    with st.chat_message(name="assistant"):
        augmented_prompt: str = get_augmented_prompt(
            prompt=prompt,
            query_result=collection.query(
                query_texts=prompt,
                n_results=st.session_state["n_results"],
                where={"status_dokumen": {"$in": st.session_state["include"]}},
            ),
        )

        response: str = st.write_stream(
            stream=(
                chunk["message"]["content"]
                for chunk in ollama.chat(
                    model=st.session_state["model"],
                    messages=[{"role": "user", "content": augmented_prompt}],
                    stream=True,
                )
            ),
        )

    st.session_state["msgs"].append({"role": "assistant", "content": response})
