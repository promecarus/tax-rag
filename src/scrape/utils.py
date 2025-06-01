import asyncio
import random
import time
import typing

import httpx
import pydantic
import toml
from google import genai
from google.genai import types
from lxml import html


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
