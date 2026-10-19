import os
import tempfile
from typing import List, Dict
import streamlit as st

from pycoustic.log import *
from pycoustic.survey import *


##### Setup #####
# Check for already existing logs
if "logs" not in st.session_state:
    st.session_state["logs"] = {}

logs = st.session_state["logs"]

# Check for existing files
files = st.session_state.get("files")
# Check for selected manufacturer
manufacturer = st.session_state.get("manufacturer")

#### Upload Files as logs ####
def ingest_files(files, manufacturer) -> Dict[str, Log]:
    """Load each uploaded file into a pycoustic Log"""
    logs: Dict[str, Log] = {}

    if not files:
        st.warning("Please Upload Files")
        st.stop()
    if not manufacturer:
        st.warning("Please Select Manufacturer or Other")
        st.stop()

    with st.spinner("Processing Data...", show_time=True):
        for upload_file in files:
            _, ext = os.path.splitext(upload_file.name)
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(upload_file.getbuffer())
                path = tmp.name
            try:
                # Create a log for the current file
                logs[upload_file.name] = Log(path, manufacturer)
            except Exception as err:
                st.error(f"Failed to load `{upload_file.name}` into Pycoustic: {err}")
            else:
                try:
                    os.unlink(path)
                except PermissionError:
                    st.warning(f"Could not delete temporary file {path}")

    return logs


