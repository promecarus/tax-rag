from pathlib import Path

import polars as pl
import streamlit as st
from auth0.authentication import GetToken
from auth0.management import Auth0


def profile_card():
    with st.expander(label="Profil", expanded=True):
        if (user := st.user)["is_logged_in"]:
            auth0 = Auth0(
                domain=(domain := st.secrets["auth"]["auth0"]["domain"]),
                token=GetToken(
                    domain=domain,
                    client_id=st.secrets["auth"]["auth0"]["client_id"],
                    client_secret=st.secrets["auth"]["auth0"]["client_secret"],
                ).client_credentials(audience=f"https://{domain}/api/v2/")[
                    "access_token"
                ],
            )

            role_badge = next(
                (
                    f":blue-badge[{r['name']}]"
                    for r in auth0.users.list_roles(id=user["sub"])["roles"]
                ),
                ":red-badge[Pengguna]",
            )

            st.image(
                image=user["picture"],
                caption=f"{role_badge}\n\n{user['name']}",
                use_container_width=True,
            )

            if st.button(label="Keluar", use_container_width=True):
                st.logout()

            if role_badge != ":red-badge[Pengguna]" and st.button(
                label="Kelola Peran Pengguna",
                use_container_width=True,
            ):
                manage_user_roles(auth0=auth0, current_user_id=user["sub"])

        else:
            if st.button(label="Masuk", use_container_width=True):
                st.login(provider="auth0")


@st.cache_data
def get_df(source: str) -> pl.DataFrame:
    match Path(source).suffix:
        case ".csv":
            return pl.read_csv(source=source)
        case ".json":
            return pl.read_json(source=source)
        case _:
            raise ValueError(f"Unsupported file extension: {source}")


@st.dialog(title="Kelola Peran Pengguna", width="large")
def manage_user_roles(auth0: Auth0, current_user_id: str) -> None:
    users = auth0.users.list(
        q=st.text_input(
            label="Cari pengguna",
            placeholder="Masukkan ID, nama, atau email pengguna",
            help="Cari pengguna berdasarkan ID, nama, atau email",
        ),
    )["users"]

    for user in [user for user in users if user["user_id"] != current_user_id]:
        role = next(
            (r["name"] for r in auth0.users.list_roles(id=user["user_id"])["roles"]),
            "Pengguna",
        )

        with st.container(border=True):
            left, right = st.columns(spec=[2, 8])

            left.image(image=user["picture"], use_container_width=True)

            with right:
                st.write(
                    ":{}-badge[{}] `ID: {}`".format(
                        "blue" if role == "Admin" else "red",
                        role,
                        user["user_id"],
                    ),
                )
                st.write(user["name"])
                st.write(user["email"])

            if st.button(
                label="Ubah Peran Menjadi :{}-badge[{}]".format(
                    "blue" if role == "Pengguna" else "red",
                    "Admin" if role == "Pengguna" else "Pengguna",
                ),
                key=user["user_id"] + "roles",
                use_container_width=True,
            ):
                if role == "Admin":
                    auth0.users.remove_roles(
                        id=user["user_id"],
                        roles=[st.secrets["auth"]["auth0"]["role_id_admin"]],
                    )
                else:
                    auth0.users.add_roles(
                        id=user["user_id"],
                        roles=[st.secrets["auth"]["auth0"]["role_id_admin"]],
                    )
