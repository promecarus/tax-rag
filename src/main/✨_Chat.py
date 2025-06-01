import json
import random

import chromadb
import streamlit as st
from google import genai
from google.genai import types
from sqlmodel import Field, Session, SQLModel, create_engine, select
from utils import get_augmented_prompt, get_timestamp, profile_card

SQLModel.__table_args__ = {"extend_existing": True}


class History(SQLModel, table=True):
    user_id: str = Field(primary_key=True)
    timestamp: str = Field(primary_key=True)
    messages: str


SQLModel.metadata.create_all(
    bind=(engine := create_engine(url="sqlite:///.history.db")),
)

st.set_page_config(
    page_title="Chat",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    if st.button(label="Chat Baru", use_container_width=True):
        st.session_state["msgs"] = []

        if st.user["is_logged_in"]:
            st.query_params["ts"] = get_timestamp()

if st.user["is_logged_in"]:
    st.query_params["ts"] = st.query_params.get("ts", default=get_timestamp())

    if "msgs" not in st.session_state:
        with Session(bind=engine) as session:
            st.session_state["msgs"] = (
                json.loads(s=session_data.messages)
                if (
                    session_data := session.exec(
                        statement=select(History).where(
                            (History.user_id == st.user["sub"])
                            & (History.timestamp == st.query_params["ts"]),
                        ),
                    ).first()
                )
                else []
            )


if "msgs" not in st.session_state:
    st.session_state["msgs"] = []

with st.sidebar:
    profile_card()

    if st.user["is_logged_in"]:
        with st.expander(label="Riwayat Chat"), Session(bind=engine) as session:
            for i in session.exec(
                statement=select(History)
                .where(History.user_id == st.user["sub"])
                .order_by(History.timestamp.desc()),
            ).all():
                select_history, delete_history = st.columns(spec=[6, 1])
                if select_history.button(
                    label=f":small[{json.loads(i.messages)[0]['content']}]",
                    help=f":small[{i.timestamp}]",
                    use_container_width=True,
                ):
                    st.query_params["ts"] = i.timestamp
                    del st.session_state["msgs"]
                    st.rerun()

                if delete_history.button(
                    label="🗑️",
                    key=f"delete_{i.timestamp}",
                    help="Hapus riwayat chat ini",
                    use_container_width=True,
                ):
                    session.delete(
                        instance=session.exec(
                            statement=select(History).where(
                                (History.user_id == st.user["sub"])
                                & (History.timestamp == i.timestamp),
                            ),
                        ).one(),
                    )
                    session.commit()
                    if st.query_params["ts"] == i.timestamp:
                        st.query_params["ts"] = get_timestamp()
                        del st.session_state["msgs"]
                    st.rerun()

    with st.expander(label="Konfigurasi Chat"):
        st.session_state["model"] = st.selectbox(
            label="Pilih model:",
            options=(
                models := [
                    "gemini-1.5-flash",
                    "gemini-1.5-flash-8b",
                    "gemini-1.5-pro",
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-lite",
                    "gemini-2.5-flash-preview-04-17",
                    "gemini-2.5-flash-preview-05-20",
                    "gemma-3-12b-it",
                    "gemma-3-1b-it",
                    "gemma-3-27b-it",
                    "gemma-3-4b-it",
                    "gemma-3n-e4b-it",
                ]
            ),
            index=models.index("gemini-2.0-flash"),
            help="Pilih model untuk digunakan pada chat.",
        )

        st.session_state["n_results"] = st.number_input(
            label="Jumlah dokumen:",
            min_value=1,
            max_value=10,
            value=2,
            help="Dokumen teratas yang digunakan untuk menjawab (1-10)",
        )

        collection: chromadb.Collection = chromadb.PersistentClient(
            path=".chroma",
            settings=chromadb.config.Settings(anonymized_telemetry=False),
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
            value=False,
            help="Tampilkan dokumen yang diambil pada proses pencarian.",
        )

        st.session_state["show_augmented"] = st.toggle(
            label="Tampilkan prompt",
            value=True,
            help="Tampilkan prompt yang dihasilkan dari dokumen yang diambil.",
        )

st.title(body="✨ Chat")

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
                chunk.text
                for chunk in genai.Client(
                    api_key=random.choice(seq=st.secrets["api_keys"]),
                ).models.generate_content_stream(
                    model=st.session_state["model"],
                    contents=[
                        *(
                            f"{x['role']}({x['content']})"
                            for x in st.session_state["msgs"][-5:-1]
                        ),
                        augmented_prompt,
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        system_instruction="""
Instruksi Generasi Jawaban:
1. Role: Anda adalah petugas sosialisasi pajak yang ahli dalam menjawab pertanyaan
perpajakan. Anda informatif dan membantu. Nantinya akan disertakan riwayat chat
sebelumnya, jika pengguna tidak menanyakan hal yang berkaitan dengan riwayat chat
sebelumnya, maka abaikan riwayat chat tersebut.
2. Bahasa:
- Jawablah hanya dalam Bahasa Indonesia, meskipun pertanyaan dalam bahasa lain.
- Sertakan sumber.
3. Format Jawaban:
- Jika pertanyaan di luar konteks perpajakan, respon dengan: "Pertanyaan tidak relevan
dengan perpajakan. Silakan ajukan pertanyaan lain yang berkaitan dengan perpajakan."
""",
                    ),
                )
            ),
        )

    st.session_state["msgs"].append({"role": "assistant", "content": response})

    if st.user["is_logged_in"]:
        with Session(bind=engine) as session:
            if existing := session.exec(
                statement=select(History).where(
                    (History.user_id == st.user["sub"])
                    & (History.timestamp == st.query_params["ts"]),
                ),
            ).first():
                existing.messages = json.dumps(st.session_state["msgs"])
            else:
                session.add(
                    instance=History(
                        user_id=st.user["sub"],
                        timestamp=st.query_params["ts"],
                        messages=json.dumps(st.session_state["msgs"]),
                    ),
                )

            session.commit()
