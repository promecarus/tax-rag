import asyncio
import time
import typing

import httpx
import toml

cfg: dict[str, typing.Any] = toml.load(f=".env.toml")


async def req_list_regs(page: int, limit: int) -> httpx.Response:
    async with httpx.AsyncClient(timeout=60) as client:
        return await client.post(
            url=f"{cfg['URL']['base']}/api/req-be",
            json={
                "method": "post",
                "url": f"{cfg['URL']['api']}/peraturan/list",
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
                    url=f"{cfg['URL']['base']}/api/req-be",
                    json={
                        "method": "post",
                        "url": f"{cfg['URL']['api']}/peraturan/detail",
                        "data": {"permalink": permalink},
                    },
                )
                if response.status_code == 200:
                    return response.json()["data"][0]
            except Exception as e:  # noqa: PERF203
                print(permalink, e)  # noqa: T201
                time.sleep(0.1)
