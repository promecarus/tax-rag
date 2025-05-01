from pathlib import Path

import polars as pl
import streamlit as st


@st.cache_data
def get_df(source: str) -> pl.DataFrame:
    match Path(source).suffix:
        case ".csv":
            return pl.read_csv(source=source)
        case ".json":
            return pl.read_json(source=source)
        case _:
            raise ValueError(f"Unsupported file extension: {source}")
