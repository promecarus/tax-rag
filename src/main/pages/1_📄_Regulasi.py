import asyncio
import contextlib
import datetime
import locale
import time
from pathlib import Path

import chromadb
import polars as pl
import pytz
import streamlit as st
import utils
from auth0.authentication import GetToken
from auth0.management import Auth0
from streamlit.delta_generator import DeltaGenerator
from utils import get_df, profile_card

try:
    locale.setlocale(category=locale.LC_TIME, locale="id_ID.UTF-8")
except locale.Error:
    with contextlib.suppress(locale.Error):
        locale.setlocale(category=locale.LC_TIME, locale="Indonesian_Indonesia.1252")

st.set_page_config(
    page_title="Regulasi",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    if (user := st.user)["is_logged_in"]:
        auth0 = Auth0(
            domain=(domain := st.secrets["auth"]["auth0"]["domain"]),
            token=GetToken(
                domain=domain,
                client_id=st.secrets["auth"]["auth0"]["client_id"],
                client_secret=st.secrets["auth"]["auth0"]["client_secret"],
            ).client_credentials(audience=f"https://{domain}/api/v2/")["access_token"],
        )

        if (
            next(
                (
                    f":blue-badge[{r['name']}]"
                    for r in auth0.users.list_roles(id=user["sub"])["roles"]
                ),
                ":red-badge[Pengguna]",
            )
            == ":blue-badge[Admin]"
        ) and st.button(label="Sync Regulasi", use_container_width=True):
            st.toast(
                body="\nMulai proses pada: {}".format(
                    datetime.datetime.now(tz=pytz.timezone(zone="Asia/Jakarta")),
                ),
                icon="🔄",
            )

            start: float = time.time()

            chroma_client: chromadb.ClientAPI = chromadb.PersistentClient(
                path=".chroma",
                settings=chromadb.config.Settings(anonymized_telemetry=False),
            )

            collection: chromadb.Collection = chroma_client.get_collection(
                name="tax-rag",
            )

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
                            expr=pl.element()
                            .struct.field(name="uuid")
                            .cast(dtype=pl.Utf8),
                        )
                        .list.sort()
                        .list.join(separator=" "),
                    ],
                )
                .filter(pl.col(name="topik").str.contains(pattern=r"2|3"))
            )

            st.toast(
                body=f"Jumlah sebelumnya sebanyak {regulation_old.height} data.",
                icon="📊",
            )

            st.toast(
                body=f"Jumlah terbaru sebanyak {regulation_new.height} data.",
                icon="📊",
            )

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
                        .map_elements(
                            function=utils.get_detail_reg,
                            return_dtype=pl.Struct,
                        )
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
                        .map_elements(
                            function=utils.strip_html_tags,
                            return_dtype=pl.Utf8,
                        )
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

            st.toast(
                body=f"Jumlah regulasi baru sebanyak {new.height} data.",
                icon="📊",
            )

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
                        | (
                            pl.col(name="tanggal_efektif")
                            != pl.col(name="tanggal_efektif_old")
                        )
                        | (
                            pl.col(name="status_dokumen")
                            != pl.col(name="status_dokumen_old")
                        )
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

            st.toast(
                body=f"Jumlah regulasi diperbarui sebanyak {update.height} data.",
                icon="📊",
            )

            if (
                delete := (
                    regulation_old.join(
                        other=regulation_new.select("permalink"),
                        on="permalink",
                        how="anti",
                    ).select(pl.col(name="permalink"))
                )
            ).height:
                collection.delete(
                    where={"permalink": {"$in": delete["permalink"].to_list()}},
                )

                delete.write_json(file=Path("var/03_final") / "_delete.json")

            st.toast(
                body=f"Jumlah regulasi dihapus sebanyak {delete.height} data.",
                icon="📊",
            )

            (
                pl.concat(items=[regulation_old, new, update])
                .unique(subset="permalink", keep="last")
                .join(other=delete, on="permalink", how="anti")
                .write_csv(file=Path("var/03_final") / "regulation.csv")
            )

            st.toast(
                body=f"Selesai proses pada: {time.time() - start:.2f} detik",
                icon="✅",
            )

    profile_card()

st.title(body="📄 Regulasi")

df_info: pl.DataFrame = get_df(source="var/03_final/regulation.csv").with_columns(
    pl.col(name="tanggal_efektif").str.strptime(dtype=pl.Date, format="%Y-%m-%d"),
)
df_topik: pl.DataFrame = get_df(source="var/03_final/topic.csv")

if filters := st.multiselect(
    label="Filter berdasarkan:",
    options=[
        "Jenis Peraturan",
        "Keywords",
        "Status Dokumen",
        "Tanggal Efektif",
        "Topik",
    ],
    help="Pilih filter yang ingin digunakan untuk menyaring data.",
    placeholder="Pilih filter...",
):
    cols: list[DeltaGenerator] = st.columns(spec=len(filters))

    for k, v in enumerate(iterable=filters):
        match column := v.lower().replace(" ", "_"):
            case "jenis_peraturan" | "status_dokumen":
                df_info: pl.DataFrame = df_info.filter(
                    pl.col(name=column).is_in(
                        other=cols[k].multiselect(
                            label=f"Values for column **{v}**:",
                            options=(options := df_info[column].unique().sort()),
                            default=[] if k == 0 else options,
                        ),
                    ),
                )

            case "keywords":
                df_info: pl.DataFrame = df_info.filter(
                    pl.col(name=column).str.contains(
                        pattern=cols[k].text_input(label=f"Values for column **{v}**:"),
                    ),
                )

            case "tanggal_efektif":
                if not df_info[column].is_empty():
                    df_info: pl.DataFrame = df_info.filter(
                        pl.col(name=column).is_between(
                            lower_bound=(
                                bound := cols[k].slider(
                                    label=f"Values for column **{v}**",
                                    value=(
                                        df_info[column].min(),
                                        df_info[column].max(),
                                    ),
                                    format="DD MMMM Y",
                                )
                            )[0],
                            upper_bound=bound[1],
                        ),
                    )

            case "topik":
                options: list[str] = sorted(
                    [
                        df_topik.filter(pl.col(name="uuid") == int(i))["keterangan"][0]
                        for i in df_info["topik"].str.split(by=" ").explode().unique()
                    ],
                )

                df_info: pl.DataFrame = df_info.filter(
                    pl.col(name="topik").str.contains(
                        pattern="|".join(
                            [
                                str(
                                    object=df_topik.filter(
                                        pl.col(name="keterangan") == result,
                                    )["uuid"][0],
                                )
                                for result in cols[k].multiselect(
                                    label=f"Values for column **{v}**:",
                                    options=options,
                                    default=options,
                                )
                            ],
                        ),
                    ),
                )

if rows := st.dataframe(
    data=df_info[
        [
            "jenis_peraturan",
            "nomor_peraturan",
        ]
    ],
    use_container_width=True,
    column_config={
        "jenis_peraturan": st.column_config.Column(
            label="Jenis Peraturan",
        ),
        "nomor_peraturan": st.column_config.Column(
            label="Nomor Peraturan",
        ),
    },
    on_select="rerun",
    selection_mode=["multi-row"],
)["selection"]["rows"][:6]:
    cols = [
        st.columns(spec=len(rows)),
        st.columns(spec=len(rows)),
        st.columns(spec=len(rows)),
    ]

    for k, v in enumerate(iterable=rows):
        row: pl.DataFrame = df_info[v]

        cols[0][k].subheader(
            body="{} Nomor: {}".format(
                row["jenis_peraturan"][0],
                row["nomor_peraturan"][0],
            ),
        )

        with cols[1][k].expander(
            label="Informasi Detail Dokumen",
            icon="🔍",
        ):
            st.write(f"**Perihal:** {row['perihal'][0]}")

            st.write(
                f"**Tanggal Efektif:** {
                    row['tanggal_efektif'][0].strftime(format='%d %B %Y').lstrip('0')
                }",
            )

            st.write(
                f"**Status Dokumen:** {
                    content
                    if len(content := row['status_dokumen'][0])
                    else 'Tidak diketahui'
                }",
            )

            st.segmented_control(
                label="**Topik**",
                options=(
                    result[0]
                    if (
                        result := df_topik.filter(
                            pl.col(name="uuid") == int(uuid),
                        )["keterangan"]
                    ).shape
                    else None
                    for uuid in row["topik"][0].split(" ")
                ),
                key=f"topik{row}",
            )

            st.segmented_control(
                label="**Peraturan Terbaru**",
                options=peraturan.split(sep=" ")
                if len(peraturan := row["peraturan_terbaru"][0])
                else ["Tidak ada"],
                key=f"peraturan_terbaru{row}",
            )

            st.segmented_control(
                label="**Peraturan Sebelumnya**",
                options=peraturan.split(sep=" ")
                if len(peraturan := row["peraturan_sebelumnya"][0])
                else ["Tidak ada"],
                key=f"peraturan_sebelumnya{row}",
            )

            st.segmented_control(
                label="**Peraturan Relevan**",
                options=peraturan.split(sep=" ")
                if len(peraturan := row["peraturan_relevan"][0])
                else ["Tidak ada"],
                key=f"peraturan_relevan{row}",
            )

            st.segmented_control(
                label="**Keywords**",
                options=keywords.split(sep=",")
                if len(
                    keywords := row["keywords"][0].lstrip(", ").rstrip(", "),
                )
                else ["Tidak ada"],
                key=f"keywords{row}",
            )

        cols[2][k].html(body=row["body_final"][0])
