import datetime

import chromadb
import ollama
import polars as pl
import streamlit as st
from auth0.authentication import GetToken
from auth0.management import Auth0
from utils import get_df


@st.dialog(title="Kelola Peran Pengguna", width="large")
def manage_user_roles(auth0: Auth0, current_user_id: str) -> None:
    users = auth0.users.list(
        q=st.text_input(
            label="Cari pengguna",
            placeholder="Masukkan ID, nama, atau email pengguna",
            help="Cari pengguna berdasarkan ID, nama, atau email",
        ),
    )["users"]

    for user in [user for user in users if user["user_id"] != current_user_id]:
        role = next(
            (r["name"] for r in auth0.users.list_roles(id=user["user_id"])["roles"]),
            "Pengguna",
        )

        with st.container(border=True):
            left, right = st.columns(spec=[2, 8])

            left.image(image=user["picture"], use_container_width=True)

            with right:
                st.write(
                    ":{}-badge[{}] `ID: {}`".format(
                        "blue" if role == "Admin" else "red",
                        role,
                        user["user_id"],
                    ),
                )
                st.write(user["name"])
                st.write(user["email"])

            if st.button(
                label="Ubah Peran Menjadi :{}-badge[{}]".format(
                    "blue" if role == "Pengguna" else "red",
                    "Admin" if role == "Pengguna" else "Pengguna",
                ),
                key=user["user_id"] + "roles",
                use_container_width=True,
            ):
                if role == "Admin":
                    auth0.users.remove_roles(
                        id=user["user_id"],
                        roles=[st.secrets["auth"]["auth0"]["role_id_admin"]],
                    )
                else:
                    auth0.users.add_roles(
                        id=user["user_id"],
                        roles=[st.secrets["auth"]["auth0"]["role_id_admin"]],
                    )


st.set_page_config(
    page_title="Tax RAG Chat",
    page_icon="✨",
    initial_sidebar_state="expanded",
)

if "msgs" not in st.session_state:
    st.session_state["msgs"] = []

with st.sidebar:
    with st.expander(label="Profil", expanded=True):
        if (user := st.experimental_user)["is_logged_in"]:
            auth0 = Auth0(
                domain=(domain := st.secrets["auth"]["auth0"]["domain"]),
                token=GetToken(
                    domain=domain,
                    client_id=st.secrets["auth"]["auth0"]["client_id"],
                    client_secret=st.secrets["auth"]["auth0"]["client_secret"],
                ).client_credentials(audience=f"https://{domain}/api/v2/")[
                    "access_token"
                ],
            )

            role_badge = next(
                (
                    f":blue-badge[{r['name']}]"
                    for r in auth0.users.list_roles(id=user["sub"])["roles"]
                ),
                ":red-badge[Pengguna]",
            )

            st.image(
                image=user["picture"],
                caption=f"{role_badge}\n\n{user['name']}",
                use_container_width=True,
            )

            if st.button(label="Keluar", use_container_width=True):
                st.logout()

            if role_badge != ":red-badge[Pengguna]" and st.button(
                label="Kelola Peran Pengguna",
                use_container_width=True,
            ):
                manage_user_roles(auth0=auth0, current_user_id=user["sub"])

        else:
            if st.button(label="Masuk", use_container_width=True):
                st.login(provider="auth0")

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
