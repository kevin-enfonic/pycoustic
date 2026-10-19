# App Setup
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import pydeck as pdk

# Import components
from src.components.sidebar import file_upload, resampling_period

## Interface between streamlit and Pycoustic
from src.utils.pycoustic_interface import ingest_files

## Import pages
from src.page_modules import Home_Page, Log_Page, Summary_Page, Map_Page

st.set_page_config(page_title="Pycoustic Acoustic Survey Explorer", layout="wide")

if "active_log" not in st.session_state:
    st.session_state["active_log"] = None

if "period" not in st.session_state:
    st.session_state["period"] = None
if "apply_agg" not in st.session_state:
    st.session_state["apply_agg"] = None
if "period_last" not in st.session_state:
    st.session_state["period_last"] = None

#Sidebar
with st.sidebar:
    manufacturer, files = file_upload()
    period = resampling_period()

if "logs" not in st.session_state:
    st.session_state["logs"] = {}

if files and manufacturer:
    files_last = st.session_state.get("files_last")
    manufacturer_last = st.session_state.get("manufacturer_last")
    if files != files_last or manufacturer != manufacturer_last:
        st.session_state["logs"] = ingest_files(files, manufacturer)
        st.session_state["files_last"] = files
        st.session_state["manufacturer_last"] = manufacturer


tabs = ["Home", "Logs", "Summary", "Maps"]
tab1, tab2, tab3, tab4 = st.tabs(tabs)


with tab1:
    Home_Page.render()

with tab2:
    Log_Page.render()

with tab3:
    Summary_Page.render()

with tab4:
    Map_Page.render()


