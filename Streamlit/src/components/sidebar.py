import streamlit as st

def file_upload():
    with st.expander("File Upload", expanded=True):
        manufacturer  =  st.selectbox(
            "Please select manufacturer",
            ("B&K", "Other"),
            index=0
        )
        files = st.file_uploader(
            "Select one or more files",
            type=["csv", "xlsx", "xlsm"],
            accept_multiple_files=True
        )

        st.session_state["manufacturer"] = manufacturer
        st.session_state["files"] = files

    return manufacturer, files

def resampling_period():
    with st.expander("Resampling Period", expanded=True):
        re_period = st.number_input(
            "Insert resampling period (Must be larger than data logging period)",
            step=1,
            value=15
        )
        period_select = st.selectbox(
            "Please select time period",
            ("second(s)", "Minute(s)", "Hour(s)", "Day(s)"),
            index=1
        )

        suffix_map = {"second(s)": "s", "minute(s)": "min", "hour(s)": "h", "days": "d"}
        period = f"{re_period}{suffix_map.get(period_select, '')}"

        #reset apply_agg flag if period changes
        if st.session_state["period_last"] != period:
            st.session_state["apply_agg"] = False
            st.session_state["period_last"] = period

        #trigger resampling
        apply_resampling_button = st.button("Apply Resampling Period")
        if apply_resampling_button:
            st.session_state["apply_agg"] = True

        st.session_state["period"] = period

    return period