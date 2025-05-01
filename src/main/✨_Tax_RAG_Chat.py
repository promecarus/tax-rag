import datetime

import chromadb
import ollama
import polars as pl
import streamlit as st
from utils import get_df, profile_card

st.set_page_config(
    page_title="Tax RAG Chat",
    page_icon="✨",
    initial_sidebar_state="expanded",
)

if "msgs" not in st.session_state:
    st.session_state["msgs"] = []

with st.sidebar:
    profile_card()

    if st.button(label="Clear chat", use_container_width=True):
        st.session_state["msgs"] = []

    model: str = st.selectbox(
        label="Select model:",
        options=sorted(
            [i["model"] for i in ollama.list()["models"] if "embed" not in i["model"]],
        ),
        help="Select the model to use for the chat.",
    )

    n_results: int = st.number_input(
        label="Documents to retrieve:",
        min_value=1,
        max_value=10,
        value=3,
        help="Top documents used for answer (1-10)",
    )

    collection: chromadb.Collection = chromadb.PersistentClient(
        path=".chroma",
    ).get_collection(name="tax-rag")

    exlude: list[str] = st.multiselect(
        label="Exclude regulation types",
        options=sorted(
            {x["jenis_peraturan"] for x in collection.get()["metadatas"]},
        ),
        default=[
            "Pengumuman",
            "Surat Edaran Direktur Jenderal Pajak",
            "Surat Edaran Direktur Jenderal Anggaran",
        ],
        help="Regulation types to exclude from the search.",
    )

st.title(body="✨ Tax RAG Chat")

for msg in st.session_state["msgs"]:
    st.chat_message(name=msg["role"]).markdown(body=msg["content"])

if prompt := st.chat_input():
    st.session_state["msgs"].append({"role": "user", "content": prompt})

    st.chat_message(name="user").markdown(body=prompt)

    query_result: chromadb.QueryResult = collection.query(
        query_texts=prompt,
        n_results=n_results,
        where={"jenis_peraturan": {"$nin": exlude}},
    )

    with st.chat_message(name="assistant"):
        for tab, info, document, metadata in zip(
            st.tabs(tabs=[f"Dokumen {x}" for x in range(1, n_results + 1)]),
            query_result["ids"][0],
            query_result["documents"][0],
            query_result["metadatas"][0],
            strict=True,
        ):
            with tab:
                st.write(
                    "**{} Nomor: {}**\n\n**ID**: `{}`\n\n**Topik**: {}".format(
                        metadata["jenis_peraturan"],
                        metadata["nomor_peraturan"],
                        info.split(sep="#")[0],
                        ", ".join(
                            result[0]
                            if (
                                result := get_df(
                                    source="var/03_final/info_topik.csv",
                                ).filter(
                                    pl.col(name="uuid") == int(uuid),
                                )["keterangan"]
                            ).shape
                            else None
                            for uuid in metadata["topik"].split(" ")
                        ),
                    ),
                )

                st.code(
                    body=document,
                    wrap_lines=True,
                )

        prompt: str = (
            """
            **Pertanyaan Pengguna**:
            {0}

            ---

            **Konteks yang Tersedia**:
            {1}

            ---

            **Instruksi Generasi Jawaban**
            1. **Role**:
             Anda adalah petugas sosialisasi pajak yang ahli menjelaskan regulasi.
            2. **Analisis**:
             - Bandingkan semua dokumen konteks
             - Identifikasi 3 poin kunci paling relevan
             - Prioritaskan sumber resmi (UU/Peraturan Dirjen)
            3. **Struktur Jawaban Wajib**:
             **a. Ringkasan Eksekutif** (maks 2 kalimat)
             **b. Dasar Hukum** (format: [Pasal XX UU PPh])
            4. **Format Referensi**:
             - Untuk setiap pernyataan faktual, cantumkan [Dokumen X]
             - Contoh: "Berdasarkan [Dokumen 2]..."
            5. **Bahasa**:
             - Semi-formal (sesuai gaya sosialisasi publik)
             - Gunakan analogi sederhana untuk konsep kompleks
             - Bold untuk terminologi teknis: **NPWP**, **SPT**
            6. **Validasi**:
             - Jika informasi tidak lengkap:
             "Informasi terbatas pada {2}.
             Untuk detail lengkap, kunjungi [https://www.pajak.go.id]"
            """.replace("  ", "")
            .strip()
            .format(
                prompt,
                "\n".join(
                    [
                        "Dokumen {}: [{} Nomor: {}] {}".format(
                            i,
                            meta_data["jenis_peraturan"],
                            meta_data["nomor_peraturan"],
                            doc,
                        )
                        for i, doc, meta_data in zip(
                            list(range(1, n_results + 1)),
                            query_result["documents"][0],
                            query_result["metadatas"][0],
                            strict=True,
                        )
                    ],
                ),
                datetime.datetime.now(tz=datetime.UTC),
            )
        )

        st.code(
            body=prompt,
            wrap_lines=True,
        )

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
