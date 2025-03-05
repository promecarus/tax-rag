import asyncio
import pathlib
import time

import polars as pl
import utils

path_raw = pathlib.Path("var/01_raw")
path_cleaned = pathlib.Path("var/02_cleaned")

for path in [path_raw, path_cleaned]:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

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

path_03: pathlib.Path = path_cleaned / "01.csv"
if not pathlib.Path(path_03).exists():
    (
        pl.read_json(source=path_02)
        .select(
            [
                pl.col(name="permalink"),
                pl.col(name="perihal"),
                pl.col(name="tanggal_efektif").str.to_date(format="%d-%m-%Y"),
                pl.col(name="status_dokumen"),
                pl.col(name="topik")
                .list.eval(
                    expr=pl.element().struct.field(name="uuid").cast(dtype=pl.Utf8),
                )
                .list.sort()
                .list.join(separator=" "),
                pl.col(name="detail").struct.field(
                    name=[
                        "jenis_peraturan",
                        "nomor_peraturan",
                        "body_final",
                        "peraturan_terbaru",
                        "peraturan_sebelumnya",
                        "peraturan_relevan",
                    ],
                ),
                pl.col(name="detail")
                .struct.field(name="meta")
                .struct.field(name="keywords"),
            ],
        )
        .with_columns(
            [
                pl.col(name="body_final")
                .str.replace_all(pattern=r"\r+|\n+|\t+", value="")
                .str.replace_all(pattern=r"\"", value="'"),
                pl.col(name="body_final")
                .map_elements(function=utils.strip_html_tags, return_dtype=pl.Utf8)
                .str.strip_chars()
                .str.replace_all(pattern=r"\s+", value=" ")
                .alias(name="body_final_text_only"),
                pl.col(name="peraturan_terbaru")
                .list.eval(
                    expr=pl.element().struct.field(name="permalink"),
                )
                .list.sort()
                .list.join(separator=" "),
                pl.col(name="peraturan_sebelumnya")
                .list.eval(
                    expr=pl.element().struct.field(name="permalink"),
                )
                .list.sort()
                .list.join(separator=" "),
                pl.col(name="peraturan_relevan")
                .list.eval(
                    expr=pl.element().struct.field(name="permalink"),
                )
                .list.sort()
                .list.join(separator=" "),
            ],
        )
        .write_csv(file=path_03)
    )
path_03_time: float = time.time() - start - accumulate_time
accumulate_time += path_03_time
print(path_03, f"created in {path_03_time:.2f} seconds.")  # noqa: T201
