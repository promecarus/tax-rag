# ruff: noqa: E501

import asyncio
import random
import time
import typing
from datetime import datetime
from pathlib import Path

import chromadb
import httpx
import polars as pl
import pydantic
import pytz
import streamlit as st
import toml
from auth0.authentication import GetToken
from auth0.management import Auth0
from google import genai
from google.genai import types
from lxml import html


def get_timestamp() -> str:
    return datetime.now(tz=pytz.timezone(zone="Asia/Jakarta")).strftime(
        format="%Y%m%d%H%M%S",
    )


def profile_card():
    with st.expander(label="Profil", expanded=True):
        if (user := st.user)["is_logged_in"]:
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


def get_augmented_prompt(
    prompt: str,
    query_result: chromadb.QueryResult,
) -> str:
    if st.session_state["show_retrieved"]:
        for tab, document, metadata in zip(
            st.tabs(
                tabs=[
                    f"Dokumen {x}" for x in range(1, st.session_state["n_results"] + 1)
                ],
            ),
            query_result["documents"][0],
            query_result["metadatas"][0],
            strict=True,
        ):
            with tab:
                st.write(
                    "**{} Nomor: {}**\n\n**ID**: `{}`\n\n**Topik**: {}".format(
                        metadata["jenis_peraturan"],
                        metadata["nomor_peraturan"],
                        metadata["permalink"],
                        ", ".join(
                            result[0]
                            if (
                                result := get_df(
                                    source="var/03_final/topic.csv",
                                ).filter(pl.col(name="uuid") == int(uuid))["keterangan"]
                            ).shape
                            else None
                            for uuid in metadata["topik"].split(" ")
                        ),
                    ),
                )

                st.write(
                    ":blue-badge[Tanya:] {}\n\n:green-badge[Jawab:] {}".format(
                        document,
                        metadata["answer"],
                    ),
                )

    augmented_prompt: str = (
        """
Konteks yang Tersedia:
{}

Pertanyaan Pengguna:
{}
""".replace("  ", "")
        .strip()
        .format(
            "\n".join(
                [
                    "- {} {} [Sumber: {} Nomor: {}] ".format(
                        question,
                        meta_data["answer"],
                        meta_data["jenis_peraturan"],
                        meta_data["nomor_peraturan"],
                    )
                    for question, meta_data in zip(
                        query_result["documents"][0],
                        query_result["metadatas"][0],
                        strict=True,
                    )
                ],
            ),
            prompt,
        )
    )

    if st.session_state["show_augmented"]:
        st.code(body=augmented_prompt, wrap_lines=True)

    return augmented_prompt


@st.cache_data
def get_df(source: str) -> pl.DataFrame:
    match Path(source).suffix:
        case ".csv":
            return pl.read_csv(source=source)
        case ".json":
            return pl.read_json(source=source)
        case _:
            raise ValueError(f"Unsupported file extension: {source}")


@st.dialog(title="Kelola Peran Pengguna", width="large")
def manage_user_roles(auth0: Auth0, current_user_id: str) -> None:
    users = auth0.users.list(
        q=st.text_input(
            label="Cari pengguna",
            placeholder="Masukkan ID, nama, atau email pengguna",
            help="Cari pengguna berdasarkan ID, nama, atau email",
        ),
    )["users"]

    for user in [
        user
        for user in users
        if (
            user["user_id"] != current_user_id
            and user["user_id"] != "auth0|683cbd4e405794935a6b3ce4"
        )
    ]:
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

                st.rerun()


async def req_list_regs(page: int, limit: int) -> httpx.Response:
    async with httpx.AsyncClient(timeout=60) as client:
        return await client.post(
            url=f"{toml.load(f='.env.toml')['url']['base']}/api/req-be",
            json={
                "method": "post",
                "url": f"{toml.load(f='.env.toml')['url']['index']}",
                "data": {
                    "sorted_by": "tanggal_efektif[desc]",
                    "pagination": {"page": page, "limit": limit},
                },
            },
        )


async def get_all_list_regs(limit: int) -> dict[str, typing.Any]:
    first: httpx.Response = await req_list_regs(page=1, limit=limit)

    data: dict[str, typing.Any] = first.json()["data"]["search_data"]

    tasks: list = []

    async with asyncio.TaskGroup() as tg:
        tasks.extend(
            [
                tg.create_task(coro=req_list_regs(page=page, limit=limit))
                for page in range(2, first.json()["pagination"]["total_page"] + 1)
            ],
        )

    for task in tasks:
        data.extend(task.result().json()["data"]["search_data"])

    return data


def get_detail_reg(permalink: str) -> dict[str, typing.Any]:
    with httpx.Client(timeout=60) as client:
        while True:
            try:
                response: httpx.Response = client.post(
                    url=f"{toml.load(f='.env.toml')['url']['base']}/api/req-be",
                    json={
                        "method": "post",
                        "url": f"{toml.load(f='.env.toml')['url']['detail']}",
                        "data": {"permalink": permalink},
                    },
                )
                if response.status_code == 200:
                    return response.json()["data"][0]
            except Exception as e:
                print(permalink, e)  # noqa: T201
                time.sleep(0.1)


def strip_html_tags(data: str) -> str:
    return html.fromstring(html=data.replace(">", "> ")).text_content()


class QAItem(pydantic.BaseModel):
    question: str
    answer: str


class QAList(pydantic.BaseModel):
    qa_list: list[QAItem]


counter = 0


def generate_qa_list(regulation: str) -> list[QAItem]:
    global counter
    counter += 1

    while True:
        try:
            response: types.GenerateContentResponse = genai.Client(
                api_key=random.choice(seq=toml.load(f=".env.toml")["api_keys"]),
            ).models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=regulation,
                config=types.GenerateContentConfig(
                    system_instruction="""
Buatlah daftar pertanyaan dan jawaban yang menyeluruh berdasarkan isi peraturan yang
diberikan, dalam bahasa Indonesia yang jelas dan mudah dipahami.

Keluaran yang diharapkan adalah dalam bentuk JSON array, di mana setiap item adalah
objek yang memiliki dua field:
- "question": pertanyaan yang relevan dan penting terkait dengan isi peraturan, yang
dapat mencakup aspek-aspek seperti tujuan peraturan, definisi istilah penting, kewajiban
atau hak yang diatur, sanksi atau konsekuensi pelanggaran, dan lain-lain.
- "answer": jawaban lengkap dan informatif yang menjawab pertanyaan tersebut,
berdasarkan isi peraturan yang diberikan. Jawaban harus mencakup informasi penting dari
peraturan, dan harus ditulis dalam bahasa Indonesia yang formal dan mudah dipahami.
Pastikan untuk tidak mengulangi pertanyaan dalam jawaban, dan fokus pada memberikan
jawaban yang tepat dan relevan.
""",
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=QAList,
                ),
            )

            qa_list: list[QAItem] = QAList.model_validate_json(
                json_data=response.text,
            ).qa_list

            print(f"{counter:> 5}, {len(qa_list):> 5} pasang pertanyaan-jawaban.")  # noqa: T201

            return qa_list

        except Exception as e:
            print(e)  # noqa: T201
