# ruff: noqa: E501

import asyncio
import time
import typing

import httpx
import ollama
import pydantic
import toml
from lxml import html

CONFIG: dict[str, typing.Any] = toml.load(f=".env.toml")


async def req_list_regs(page: int, limit: int) -> httpx.Response:
    async with httpx.AsyncClient(timeout=60) as client:
        return await client.post(
            url=f"{CONFIG['url']['base']}/api/req-be",
            json={
                "method": "post",
                "url": f"{CONFIG['url']['api']}/peraturan/list",
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
                    url=f"{CONFIG['url']['base']}/api/req-be",
                    json={
                        "method": "post",
                        "url": f"{CONFIG['url']['api']}/peraturan/detail",
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


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[dict]:
    chunks: list[dict] = []
    start = 0
    chunk_num = 1

    while start < len(text):
        end: int = start + chunk_size
        chunks.append(
            {
                "chunk": text[start:end],
                "chunk_num": f"{chunk_size}-{chunk_num}",
            },
        )
        start += chunk_size - overlap
        chunk_num += 1

    return chunks


class QAItem(pydantic.BaseModel):
    question: str
    answer: str


class QAList(pydantic.BaseModel):
    qa_list: list[QAItem]


PROMPT_QA_CREATION = """
Buatlah daftar pertanyaan dan jawaban berdasarkan isi peraturan tersebut, dalam bahasa Indonesia yang jelas dan mudah dipahami.

Keluaran yang diharapkan adalah dalam bentuk JSON array, di mana setiap item adalah objek yang memiliki dua field:
- "question": pertanyaan yang relevan berdasarkan isi peraturan
- "answer": jawaban lengkap dan benar atas pertanyaan tersebut, berdasarkan isi peraturan, jangan gunakan kata "itu" atau "ini" dalam jawaban, gunakan kalimat yang jelas dan lengkap

Gunakan bahasa yang natural namun tetap formal. Minimal 2 pasang pertanyaan dan jawaban, maksimal 10, jangan kurang atau lebih dari itu.
"""

counter = 0


def generate_qa_list(regulation: str) -> list[QAItem]:
    global counter
    counter += 1

    while True:
        try:
            response: ollama.ChatResponse = ollama.chat(
                model="granite3.3:2b",
                messages=[
                    {"role": "user", "content": regulation},
                    {"role": "user", "content": PROMPT_QA_CREATION},
                ],
                format=QAList.model_json_schema(),
                options={"temperature": 0},
            )
            print(counter)  # noqa: T201

            return QAList.model_validate_json(
                json_data=response.message.content,
            ).qa_list

        except Exception as e:
            print(e)  # noqa: T201
