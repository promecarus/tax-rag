from pathlib import Path

import polars as pl
import streamlit as st
from streamlit.delta_generator import DeltaGenerator


@st.cache_data
def get_df(source: str) -> pl.DataFrame:
    match Path(source).suffix:
        case ".csv":
            return pl.read_csv(source=source)
        case ".json":
            return pl.read_json(source=source)
        case _:
            raise ValueError(f"Unsupported file extension: {source}")


st.set_page_config(
    page_title="Regulation",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(body="📄 Regulation")

df_info: pl.DataFrame = (
    get_df(source="var/03_final/info.csv")
    .with_columns(
        pl.col(name="tanggal_efektif").str.strptime(
            dtype=pl.Date,
            format="%Y-%m-%d",
        ),
    )
    .join(
        other=get_df(source="var/03_final/regulation.csv"),
        on="permalink",
        how="inner",
    )
)
df_info_topik: pl.DataFrame = get_df(source="var/03_final/info_topik.csv")

if filters := st.multiselect(
    label="Filter based on:",
    options=[
        "Jenis Peraturan",
        "Keywords",
        "Status Dokumen",
        "Tanggal Efektif",
        "Topik",
    ],
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
                        df_info_topik.filter(pl.col(name="uuid") == int(i))[
                            "keterangan"
                        ][0]
                        for i in df_info["topik"].str.split(by=" ").explode().unique()
                    ],
                )

                df_info: pl.DataFrame = df_info.filter(
                    pl.col(name="topik").str.contains(
                        pattern="|".join(
                            [
                                str(
                                    object=df_info_topik.filter(
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
)["selection"]["rows"]:
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
                        result := df_info_topik.filter(
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
