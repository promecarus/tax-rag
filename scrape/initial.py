import asyncio
import pathlib
import time

import polars as pl
import utils

path_raw = pathlib.Path("var/raw")
pathlib.Path(path_raw).mkdir(parents=True, exist_ok=True)

start: float = time.time()
accumulate_time: float = 0.0

path_01: pathlib.Path = path_raw / "01.json"
if not pathlib.Path(path_01).exists():
    (
        pl.DataFrame(data=asyncio.run(main=utils.get_all_list_regs(limit=4000)))
        .unique(subset="permalink")
        .write_json(file=path_01)
    )
path_01_time: float = time.time() - start - accumulate_time
accumulate_time += path_01_time
print(path_01, f"created in {path_01_time:.2f} seconds.")  # noqa: T201

path_02: pathlib.Path = path_raw / "02.json"
if not pathlib.Path(path_02).exists():
    (
        pl.read_json(source=path_01)
        .with_columns(
            pl.col(name="permalink")
            .map_elements(
                function=utils.get_detail_reg,
                return_dtype=pl.Struct,
            )
            .alias(name="detail"),
        )
        .write_json(file=path_02)
    )
path_02_time: float = time.time() - start - accumulate_time
accumulate_time += path_02_time
print(path_02, f"created in {path_02_time:.2f} seconds.")  # noqa: T201
