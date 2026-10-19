import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Graph colour palette config
COLOURS = {
    "Leq A": "#9e9e9e",   # light grey
    "L90 A": "#4d4d4d",   # dark grey
    "Lmax A": "#fc2c2c",  # red
}
TEMPLATE = "plotly"

def set_active_log(log_name):
    st.session_state["active_log"] = log_name

def render():
    st.header("Log Page")

    logs = st.session_state.get("logs", {})
    if not logs:
        st.info("No logs loaded yet")
        return
    
    left, right = st.columns([1, 3], gap="large")

    with left:
        st.subheader("Logs")

        for file_name in logs.keys():
            is_active = st.session_state.get("active_log") == file_name

            with st.container(border=True):
                st.markdown(
                    f"**▶ {file_name}**" if is_active else f"**{file_name}**"  
                )

                if st.button(
                    "View",
                    key=f"select_{file_name}",
                    on_click=set_active_log,
                    args=(file_name,),
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ) :
                    st.session_state["active_log"] = file_name

    
    with right:
        active_key = st.session_state.get("active_log")

        if not active_key:
            st.info("Select a log from the left")
            return

        log = logs[active_key]

        st.subheader(active_key)

        # Rename
        new_name = st.text_input(
            "Log Name",
            value=active_key,
            key="rename_active_log",
        )

        # Load data
        if st.session_state["apply_agg"]:
            try:
                period = st.session_state.get("period")
                df_used = log.as_interval(t=period)
                df_used = df_used.reset_index().rename(
                    columns={df_used.index.name or "index": "Timestamp"}
                )
                subheader = "Integrated Survey Data"
            except Exception as e:
                st.error(f"Failed to apply integration period for `{file_name}`: {e}")
        else:
            try:
                raw_master = log._master  # original DataFrame, indexed by Timestamp
                df_used = raw_master.reset_index().rename(columns={"Time": "Timestamp"})
                subheader = "Raw Survey Data"
            except Exception as e:
                st.error(f"Failed to load raw data for `{file_name}`: {e}")

        #measurement location
        with st.expander("Measurement Location", expanded=True):
            with st.form(key=f"location_{active_key}_form"):
                position_name = st.text_input("Position name", value=active_key)

                col1, col2 = st.columns([1, 1])
                latitude = col1.number_input("Latitude", format="%.6f")
                longitude = col2.number_input("Longitude", format="%.6f")

                # Submit button triggers update only when pressed
                submitted = st.form_submit_button("Save")
                if submitted:
                    if "locations" not in st.session_state:
                        st.session_state["locations"] = {}

                    st.session_state["locations"][position_name] = {
                        "lat": latitude,
                        "lon": longitude
                    }
                    st.success(f"Saved location for {position_name}")

        # Prepare a flattened‐column header copy JUST FOR PLOTTING
        df_plot = df_used.copy()
        if isinstance(df_plot.columns, pd.MultiIndex):
            flattened_cols = []
            for lvl0, lvl1 in df_plot.columns:
                lvl0_str = str(lvl0)
                lvl1_str = str(lvl1) if lvl1 is not None else ""
                flattened_cols.append(f"{lvl0_str} {lvl1_str}".strip())
            df_plot.columns = flattened_cols

        #  Time‐history Graph (Leq A, L90 A, Lmax A) using df_plot 
        required_cols = {"Leq A", "L90 A", "Lmax A"}
        if required_cols.issubset(set(df_plot.columns)):
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df_plot["Timestamp"],
                    y=df_plot["Leq A"],
                    name="Leq A",
                    mode="lines",
                    line=dict(color=COLOURS["Leq A"], width=1),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_plot["Timestamp"],
                    y=df_plot["L90 A"],
                    name="L90 A",
                    mode="lines",
                    line=dict(color=COLOURS["L90 A"], width=1),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_plot["Timestamp"],
                    y=df_plot["Lmax A"],
                    name="Lmax A",
                    mode="markers",
                    marker=dict(color=COLOURS["Lmax A"], size=3),
                )
            )
            fig.update_layout(
                template=TEMPLATE,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(
                    title="Time & Date (hh:mm & dd/mm/yyyy)",
                    type="date",
                    tickformat="%H:%M<br>%d/%m/%Y",
                    tickangle=0,
                ),
                yaxis_title="Measured Sound Pressure Level dB(A)",
                legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="left", x=0),
                height=600,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Required columns {required_cols} missing in {subheader}.")

        # --- Finally, display the TABLE with MultiIndex intact ---
        st.subheader(subheader)
        st.dataframe(df_used, hide_index=True)
