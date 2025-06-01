# ruff: noqa: ERA001

import asyncio
import datetime
import time
from pathlib import Path

import chromadb
import polars as pl
import pytz
import schedule
import utils


def job() -> None:
    print(  # noqa: T201
        "=" * 51
        + "\nMulai proses pada: {}".format(
            datetime.datetime.now(tz=pytz.timezone(zone="Asia/Jakarta")),
        ),
    )

    start: float = time.time()

    chroma_client: chromadb.ClientAPI = chromadb.PersistentClient(
        path=".chroma",
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )

    collection: chromadb.Collection = chroma_client.get_collection(name="tax-rag")

    regulation_old: pl.DataFrame = pl.read_csv(
        source=Path("var/03_final") / "regulation.csv",
        infer_schema_length=10000,
    ).with_columns(pl.col(name="tanggal_efektif").str.to_date())

    regulation_new: pl.DataFrame = (
        pl.DataFrame(data=asyncio.run(main=utils.get_all_list_regs(limit=4000)))
        .unique(subset="permalink")
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
            ],
        )
        .filter(pl.col(name="topik").str.contains(pattern=r"2|3"))
    )

    # regulation_new: pl.DataFrame = pl.read_json(
    #     source=Path("var/03_final/regulation_new.json"),
    # )

    if (
        new := regulation_new.join(
            other=regulation_old.select("permalink"),
            on="permalink",
            how="anti",
        ).with_columns(
            pl.lit(value="").alias(name="jenis_peraturan"),
            pl.lit(value="").alias(name="nomor_peraturan"),
            pl.lit(value="").alias(name="body_final"),
            pl.lit(value="").alias(name="peraturan_terbaru"),
            pl.lit(value="").alias(name="peraturan_sebelumnya"),
            pl.lit(value="").alias(name="peraturan_relevan"),
            pl.lit(value="").alias(name="keywords"),
        )
    ).height:
        new: pl.DataFrame = (
            new.with_columns(
                pl.col(name="permalink")
                .map_elements(function=utils.get_detail_reg, return_dtype=pl.Struct)
                .alias(name="detail"),
            )
            .select(
                [
                    pl.col(name="permalink"),
                    pl.col(name="perihal"),
                    pl.col(name="tanggal_efektif"),
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
                    pl.when(pl.col(name="peraturan_terbaru").list.len() == 0)
                    .then(statement=pl.lit(value=[{"permalink": ""}]))
                    .otherwise(statement=pl.col(name="peraturan_terbaru"))
                    .alias(name="peraturan_terbaru"),
                    pl.when(pl.col(name="peraturan_sebelumnya").list.len() == 0)
                    .then(statement=pl.lit(value=[{"permalink": ""}]))
                    .otherwise(statement=pl.col(name="peraturan_sebelumnya"))
                    .alias(name="peraturan_sebelumnya"),
                    pl.when(pl.col(name="peraturan_relevan").list.len() == 0)
                    .then(statement=pl.lit(value=[{"permalink": ""}]))
                    .otherwise(statement=pl.col(name="peraturan_relevan"))
                    .alias(name="peraturan_relevan"),
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
        )

        embed: pl.DataFrame = (
            new.with_columns(
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

        for i in range(0, len(embed), max_batch):
            batch: pl.DataFrame = embed[i : i + max_batch]
            collection.upsert(
                ids=batch["id"].to_list(),
                metadatas=batch["metadata"].to_list(),
                documents=batch["document"].to_list(),
            )

        new.write_json(file=Path("var/03_final") / "_new.json")

    if (
        update := (
            regulation_new.join(
                other=regulation_old,
                on="permalink",
                how="left",
                suffix="_old",
            )
            .filter(
                (pl.col(name="perihal") != pl.col(name="perihal_old"))
                | (pl.col(name="tanggal_efektif") != pl.col(name="tanggal_efektif_old"))
                | (pl.col(name="status_dokumen") != pl.col(name="status_dokumen_old"))
                | (pl.col(name="topik") != pl.col(name="topik_old")),
            )
            .drop(
                [
                    pl.col(name="perihal_old"),
                    pl.col(name="tanggal_efektif_old"),
                    pl.col(name="status_dokumen_old"),
                    pl.col(name="topik_old"),
                ],
            )
        )
    ).height:
        for row in update.iter_rows():
            (
                permalink,
                _,
                _,
                status_dokumen,
                topik,
                jenis_peraturan,
                nomor_peraturan,
                _,
                _,
                _,
                _,
                _,
            ) = row

            get_result: chromadb.GetResult = collection.get(
                where={"permalink": permalink},
            )

            for gr_id, gr_metadata in zip(
                get_result["ids"],
                get_result["metadatas"],
                strict=True,
            ):
                gr_metadata["status_dokumen"] = status_dokumen
                gr_metadata["topik"] = topik
                gr_metadata["jenis_peraturan"] = jenis_peraturan
                gr_metadata["nomor_peraturan"] = nomor_peraturan

                collection.update(ids=[gr_id], metadatas=[gr_metadata])

        update.write_json(file=Path("var/03_final") / "_update.json")

    if (
        delete := (
            regulation_old.join(
                other=regulation_new.select("permalink"),
                on="permalink",
                how="anti",
            ).select(pl.col(name="permalink"))
        )
    ).height:
        collection.delete(where={"permalink": {"$in": delete["permalink"].to_list()}})

        delete.write_json(file=Path("var/03_final") / "_delete.json")

    (
        pl.concat(items=[regulation_old, new, update])
        .unique(subset="permalink", keep="last")
        .join(other=delete, on="permalink", how="anti")
        .write_csv(file=Path("var/03_final") / "regulation.csv")
    )

    print(f"Baru: {new.height}, Diperbarui: {update.height}, Dihapus: {delete.height}")  # noqa: T201

    print(f"Total waktu eksekusi: {time.time() - start:.2f} detik\n")  # noqa: T201


schedule.every().day.at(time_str="00:00", tz="Asia/Jakarta").do(job_func=job)
# schedule.every(interval=0.01).seconds.do(job_func=job)

while True:
    schedule.run_pending()
    time.sleep(1)
