import asyncio
import time
from pathlib import Path

import chromadb
import polars as pl
import utils

path_raw = Path("var/01_raw")
path_clean = Path("var/02_clean")
path_final = Path("var/03_final")

for path in [path_raw, path_clean, path_final]:
    path.mkdir(parents=True, exist_ok=True)

start: float = time.time()
accumulate_time: float = 0.0

if not (path_01 := path_raw / "index.json").exists():
    (
        pl.DataFrame(
            data=asyncio.run(main=utils.get_all_list_regs(limit=4000)),
        )
        .unique(subset="permalink")
        .with_columns(
            pl.col(name="topik")
            .list.eval(expr=pl.element().struct.field(name="uuid").cast(dtype=pl.Utf8))
            .list.sort()
            .list.join(separator=" ")
            .alias("flattened_topik"),
        )
        .filter(pl.col(name="flattened_topik").str.contains(pattern=r"2|3"))
        .write_json(file=path_01)
    )
path_01_time: float = time.time() - start - accumulate_time
accumulate_time += path_01_time
print(path_01, f"created in {path_01_time:.2f} seconds.")  # noqa: T201

if not (path_02 := path_raw / "detail.json").exists():
    (
        pl.read_json(source=path_01)
        .with_columns(
            pl.col(name="permalink")
            .map_elements(function=utils.get_detail_reg, return_dtype=pl.Struct)
            .alias(name="detail"),
        )
        .with_columns(pl.col(name="flattened_topik").alias("topik"))
        .drop("flattened_topik")
        .write_json(file=path_02)
    )
path_02_time: float = time.time() - start - accumulate_time
accumulate_time += path_02_time
print(path_02, f"created in {path_02_time:.2f} seconds.")  # noqa: T201

if not (path_03 := path_clean / "processed.csv").exists():
    (
        pl.read_json(source=path_02, infer_schema_length=284)
        .select(
            [
                pl.col(name="permalink"),
                pl.col(name="perihal"),
                pl.col(name="tanggal_efektif").str.to_date(format="%d-%m-%Y"),
                pl.col(name="status_dokumen"),
                pl.col(name="topik"),
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
                pl.col(name="peraturan_terbaru")
                .list.eval(expr=pl.element().struct.field(name="permalink"))
                .list.sort()
                .list.join(separator=" "),
                pl.col(name="peraturan_sebelumnya")
                .list.eval(expr=pl.element().struct.field(name="permalink"))
                .list.sort()
                .list.join(separator=" "),
                pl.col(name="peraturan_relevan")
                .list.eval(expr=pl.element().struct.field(name="permalink"))
                .list.sort()
                .list.join(separator=" "),
            ],
        )
        .write_csv(file=path_03)
    )
path_03_time: float = time.time() - start - accumulate_time
accumulate_time += path_03_time
print(path_03, f"created in {path_03_time:.2f} seconds.")  # noqa: T201

if not (path_04 := path_final / "topic.csv").exists():
    (
        pl.read_json(source=path_01)
        .select(["topik"])
        .explode(columns="topik")
        .unnest(columns="topik")
        .unique()
        .sort(by="uuid")
        .write_csv(file=path_04)
    )
path_04_time: float = time.time() - start - accumulate_time
accumulate_time += path_04_time
print(path_04, f"created in {path_04_time:.2f} seconds.")  # noqa: T201

if not (path_05 := path_final / "regulation.csv").exists():
    (pl.read_csv(source=path_03).write_csv(file=path_05))
path_05_time: float = time.time() - start - accumulate_time
accumulate_time += path_05_time
print(path_05, f"created in {path_05_time:.2f} seconds.")  # noqa: T201

if not (path_06 := path_final / "embed.csv").exists():
    (
        pl.read_csv(source=path_03)
        .select(["permalink", "body_final"])
        .with_columns(
            pl.col(name="body_final")
            .map_elements(function=utils.strip_html_tags, return_dtype=pl.Utf8)
            .str.strip_chars()
            .str.replace_all(pattern=r"\s+", value=" "),
        )
        .with_columns(
            pl.col(name="body_final").map_elements(
                function=utils.generate_qa_list,
                return_dtype=pl.List(
                    inner=pl.Struct(
                        fields=[
                            pl.Field(name="question", dtype=pl.Utf8),
                            pl.Field(name="answer", dtype=pl.Utf8),
                        ],
                    ),
                ),
            ),
        )
        .explode(columns="body_final")
        .unnest(columns="body_final")
        .with_columns(
            pl.col(name="permalink")
            .cum_count()
            .over(partition_by="permalink")
            .cast(dtype=pl.Utf8)
            .alias(name="id"),
        )
        .select(
            [
                pl.col(name="id"),
                pl.col(name="permalink"),
                pl.col(name="question"),
                pl.col(name="answer"),
            ],
        )
        .write_csv(file=path_06)
    )
path_06_time: float = time.time() - start - accumulate_time
accumulate_time += path_06_time
print(path_06, f"created in {path_06_time:.2f} seconds.")  # noqa: T201

if not (path_07 := Path(".chroma/chroma.sqlite3")).exists():
    chroma_client: chromadb.ClientAPI = chromadb.PersistentClient(
        path=".chroma",
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )
    collection: chromadb.Collection = chroma_client.create_collection(name="tax-rag")

    data: pl.DataFrame = (
        pl.read_csv(source=path_06)
        .join(other=pl.read_csv(source=path_03), on="permalink")
        .select(
            [
                pl.concat_str(
                    exprs=[pl.col(name="permalink"), pl.col(name="id")],
                    separator="#",
                ).alias(name="id"),
                pl.struct(
                    [
                        pl.col(name="answer"),
                        pl.col(name="permalink"),
                        pl.col(name="status_dokumen"),
                        pl.col(name="topik"),
                        pl.col(name="jenis_peraturan"),
                        pl.col(name="nomor_peraturan"),
                    ],
                ).alias(name="metadata"),
                pl.col(name="question").alias(name="document"),
            ],
        )
    )

    max_batch: int = chroma_client.get_max_batch_size()

    for i in range(0, len(data), max_batch):
        batch: pl.DataFrame = data[i : i + max_batch]
        collection.add(
            ids=batch["id"].to_list(),
            metadatas=batch["metadata"].to_list(),
            documents=batch["document"].to_list(),
        )
path_07_time: float = time.time() - start - accumulate_time
accumulate_time += path_07_time
print(path_07, f"created in {path_07_time:.2f} seconds.")  # noqa: T201

if (
    df_must_remove := (df_embed := pl.read_csv(source="var/03_final/embed.csv")).filter(
        pl.col(name="question") == "Failed to generate.",
    )
).height:
    pl.concat(
        items=[
            df_embed.filter(pl.col(name="question") != "Failed to generate."),
            df_must_remove.join(
                other=pl.read_csv(source="var/03_final/regulation.csv"),
                on="permalink",
                how="inner",
            )
            .select(["permalink", "body_final"])
            .with_columns(
                pl.col(name="body_final")
                .map_elements(function=utils.strip_html_tags, return_dtype=pl.Utf8)
                .str.strip_chars()
                .str.replace_all(pattern=r"\s+", value=" "),
            )
            .with_columns(
                pl.col(name="body_final").map_elements(
                    function=utils.generate_qa_list,
                    return_dtype=pl.List(
                        inner=pl.Struct(
                            fields=[
                                pl.Field(name="question", dtype=pl.Utf8),
                                pl.Field(name="answer", dtype=pl.Utf8),
                            ],
                        ),
                    ),
                ),
            )
            .explode(columns="body_final")
            .unnest(columns="body_final")
            .with_columns(
                pl.col(name="permalink")
                .cum_count()
                .over(partition_by="permalink")
                .cast(dtype=pl.Utf8)
                .alias(name="id"),
            )
            .select(
                [
                    pl.col(name="id").cast(dtype=pl.Int64),
                    pl.col(name="permalink"),
                    pl.col(name="question"),
                    pl.col(name="answer"),
                ],
            ),
        ],
    ).write_csv(file=path_06)
    failed_question_update_time: float = time.time() - start - accumulate_time
    accumulate_time += failed_question_update_time
    print(path_06, f"updated in {failed_question_update_time:.2f} seconds.")  # noqa: T201

print(f"\nTotal time: {accumulate_time:.2f} seconds.")  # noqa: T201
