import streamlit as st
from scripts.mysql_manager import DBManager


st.set_page_config(layout="wide")

st.cache()


def load_data():
    db_manager = DBManager()
    db_manager.setup()
    df = db_manager.get_tellco_users_data()
    return df


users_data = load_data()

st.title('TellCo Users Analysis')

st.write("User MSISDN with various scores")

st.dataframe(users_data)
