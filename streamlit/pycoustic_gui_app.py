import os
import tempfile
from typing import List, Dict

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import pydeck as pdk



from log import *
from survey import *

# Streamlit app config
st.set_page_config(page_title="Pycoustic Acoustic Survey Explorer", layout="wide")

# Graph colour palette config
COLOURS = {
    "Leq A": "#9e9e9e",   # light grey
    "L90 A": "#4d4d4d",   # dark grey
    "Lmax A": "#fc2c2c",  # red
}
# Graph template config
TEMPLATE = "plotly"

def sort_freqs(freq_list):
    def key(x):
        try:
            return float(x) 
        except ValueError:
            return float('inf') 
    return sorted(freq_list, key=key)

if "apply_agg" not in st.session_state:
    st.session_state["apply_agg"] = False
if "period_last" not in st.session_state:
    st.session_state["period_last"] = ""

with st.sidebar:
    # File Upload in expander container
    with st.expander("File Upload", expanded=True):
        manufacturer_select = st.selectbox(
            "Please select manufacturer",
            ("B&K", "Other"),
            index=0,
        )
        files = st.file_uploader(
            "Select one or more files",
            type=["csv", "xlsx", "xlsm"],
            accept_multiple_files=True,
        )
        if not files:
            st.stop()
    # Integration period entry in expander container
    with st.expander("Integration Period", expanded=True):
        int_period = st.number_input(
            "Insert new integration period (must be larger than data)",
            step=1,
            value=15,
        )
        period_select = st.selectbox(
            "Please select time period",
            ("second(s)", "minute(s)", "hour(s)"),
            index=1,
        )

        # Build the period string
        suffix_map = {"second(s)": "s", "minute(s)": "min", "hour(s)": "h"}
        period = f"{int_period}{suffix_map.get(period_select, '')}"

        # If the period changed since last time, reset the "apply_agg" flag
        if st.session_state["period_last"] != period:
            st.session_state["apply_agg"] = False
            st.session_state["period_last"] = period

        # Button to trigger aggregation for ALL positions
        apply_agg_btn = st.button("Apply Integration Period")
        if apply_agg_btn:
            st.session_state["apply_agg"] = True

# Main Window / Data Load
with st.spinner("Processing Data...", show_time=True):
    # Load each uploaded file into a pycoustic Log
    logs: Dict[str, Log] = {}
    for upload_file in files:
        _, ext = os.path.splitext(upload_file.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(upload_file.getbuffer())
            path = tmp.name
        try:
            logs[upload_file.name] = Log(path, manufacturer_select)
        except Exception as err:
            st.error(f"Failed to load `{upload_file.name}` into Pycoustic: {err}")
        else:
            try:
                os.unlink(path)
            except PermissionError:
             st.warning(f"Could not delete temporary file {path}")

    # Build Survey and pull summary + spectra
    summary_df = leq_spec_df = lmax_spec_df = None
    summary_error = ""
    if logs:
        try:
            survey = Survey()
            if callable(getattr(survey, "add_log", None)):
                for name, lg in logs.items():
                    survey.add_log(lg, name=name)
            elif hasattr(survey, "_logs"):
                survey._logs = logs

            summary_df = survey.resi_summary()
            leq_spec_df = getattr(survey, "typical_leq_spectra", lambda: None)()
            lmax_spec_df = getattr(survey, "lmax_spectra", lambda: None)()
        except Exception as err:
            summary_error = str(err)
    else:
        summary_error = "No valid logs loaded."

    # Helper list of “position” names (i.e. filenames)
    pos_list = list(logs.keys())

    #Create tabs
    ui_tabs = st.tabs(["Summary"] + pos_list)

    #Summary tab
    with ui_tabs[0]:
        st.subheader("Broadband Summary")
        if summary_df is not None:
            st.dataframe(summary_df)
        else:
            st.warning(f"Summary unavailable: {summary_error}")


        # Dynamic Plot
        tidy = pd.DataFrame()
        for name, log in logs.items():
            if st.session_state["apply_agg"]:
                log_df = log.as_interval(t=period)
            else:
                log_df = log.get_data()

            log_df["Position"] = name

            if "Excluded" not in log_df.columns:
                log_df["Excluded"] = False

            log_df = log_df[log_df.get("Excluded", False) == False]
            
            if log_df.index.name in (None, ""):
                log_df = log_df.reset_index()
            elif log_df.index.name not in log_df.columns:
                log_df.reset_index(inplace=True)

            tidy = pd.concat([tidy, log_df], ignore_index=True)


        tidy.columns = [
            col if isinstance(col, str) else "_".join([str(i) for i in col if i not in (None, "")])
            for col in tidy.columns
        ]

        datetime_cols = [c for c in tidy.columns if "date" in c.lower() or "time" in c.lower()]
        if datetime_cols:
            dt_col = datetime_cols[0]
        else:
            dt_col = None

        # --- Melt to long format ---
        id_vars = ["Position"]
        if dt_col:
            id_vars.append(dt_col)

        tidy_long = tidy.melt(
            id_vars=id_vars,  
            var_name="Metric_Freq",
            value_name="Value"
        )

        if dt_col:
            tidy_long.rename(columns={dt_col: "DateTime"}, inplace=True)

        tidy_long[["Metric", "Freq"]] = tidy_long["Metric_Freq"].str.rsplit("_", n=1, expand=True)
        tidy_long["Freq"] = tidy_long["Freq"].replace(
            {"A": "A", "C": "C", "Z": "Z"} 
        )
        tidy_long["Freq_num"] = pd.to_numeric(tidy_long["Freq"], errors="coerce")
        tidy_long.drop(columns="Metric_Freq", inplace=True)

        tidy_long["is_freq_specific"] = tidy_long["Freq_num"].notna()


        # Selectors
        col1, col2, col3 = st.columns([1, 1, 2]) 

        with col1:
            x_axis = st.selectbox("X-axis", ["Frequency", "Time"])

        with col2:
            if x_axis == "Frequency":
                # Only metrics with numeric frequency
                metric_options = tidy_long.loc[tidy_long["is_freq_specific"], "Metric"].unique()
            else:  # Time or Position
                metric_options = tidy_long["Metric"].unique()
            y_axis = st.selectbox("Y-axis", metric_options)

        with col3:
            positions = st.multiselect(
                "Positions", tidy_long["Position"].unique(), default=tidy_long["Position"].unique()
            )

        if x_axis == "Time":
            # Identify available frequencies for the chosen metric
            freq_options = tidy_long.loc[
                tidy_long["Metric"] == y_axis, "Freq"
            ].dropna().unique()
            freq_options = sort_freqs(freq_options)

            # Add frequency selector
            colf1, colf2, colf3 = st.columns([1.5, 1, 1])
            with colf1:
                selected_freqs = st.multiselect(
                    "Frequencies (bands / weightings)",
                    freq_options,
                    default=freq_options[:1]
                )
            with colf2:
                start_date = st.date_input("Start Date", tidy_long["DateTime"].min().date())
            with colf3:
                end_date = st.date_input("End Date", tidy_long["DateTime"].max().date())

        # --- Filter Data ---
        plot_df = tidy_long[
            (tidy_long["Position"].isin(positions)) &
            (tidy_long["Metric"] == y_axis)
        ]

        if x_axis == "Time":
            plot_df = plot_df[
                (plot_df["DateTime"].dt.date >= start_date) &
                (plot_df["DateTime"].dt.date <= end_date)
            ]

            plot_df["Freq_str"] = plot_df["Freq"].astype(str)
            plot_df = plot_df[plot_df["Freq_str"].isin(selected_freqs)]


        if x_axis == "Frequency":
            numeric_freqs = plot_df["Freq_num"].dropna().unique()
            letter_freqs = plot_df.loc[plot_df["Freq_num"].isna(), "Freq"].unique()

            numeric_freqs_sorted = sorted(numeric_freqs)
            letter_freqs_sorted = sorted(letter_freqs)  

            freq_order = list(map(str, numeric_freqs_sorted)) + list(letter_freqs_sorted)

            plot_df["Freq_str"] = plot_df["Freq"].astype(str)
            plot_df["Freq_str"] = pd.Categorical(plot_df["Freq_str"], categories=freq_order, ordered=True)

            plot_df = plot_df.groupby(["Position", "Freq_str"])["Value"].mean().reset_index()

            fig = px.line(
                plot_df,
                x="Freq_str",
                y="Value",
                color="Position",
                markers=True
            )
            fig.update_xaxes(title="Frequency (Hz)", type="log")

        elif x_axis == "Time":
            plot_df["ColourGroup"] = plot_df["Position"] + "," + plot_df["Metric"] + "_" + plot_df["Freq"].astype(str)

            fig = px.line(
                plot_df,
                x="DateTime",
                y="Value",
                color="ColourGroup",
                hover_data=["Metric", "Freq", "Position"]

            )
            fig.update_xaxes(title="Time")

        elif x_axis == "Position":
            fig = px.bar(
                plot_df,
                x="Position",
                y="Value",
                color="Metric",
            )
            fig.update_xaxes(title="Position")

        # --- Common Styling ---
        fig.update_layout(
            yaxis_title=f"{y_axis} (dB)",
            title=f"{y_axis} vs {x_axis}",
            template="plotly_white",
        )

        st.plotly_chart(fig, use_container_width=True)

        
        # Map
        st.title("Measurement Locations")

        if "locations" in st.session_state and st.session_state["locations"]:
            df = pd.DataFrame.from_dict(st.session_state["locations"], orient="index").reset_index()
            df.columns = ["Position", "lat", "lon"]

            # Compute center of map (mean latitude & longitude)
            center_lat = df["lat"].mean()
            center_lon = df["lon"].mean()

            # Pydeck scatter plot layer
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=50,
                pickable=True,
            )

            # Initial view with zoom control
            view_state = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=12,   # <-- Adjust zoom level (higher = closer)
                pitch=0
            )

            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                map_style='road',  # <-- Satellite view
                tooltip={"text": "{Position}"}
            )

            st.pydeck_chart(r)
        else:
            st.info("No locations saved yet. Add coordinates in the position pages.")




    # Position‐Specific Tabs
    for tab, uf in zip(ui_tabs[1:], files):
        with tab:
            log = logs.get(uf.name)
            if log is None:
                st.error(f"Log for `{uf.name}` not found.")
                continue

            # Decide whether to show raw or aggregated data
            if st.session_state["apply_agg"]:
                # 1) Re-aggregate / resample using the chosen period
                try:
                    df_used = log.as_interval(t=period)
                    df_used = df_used.reset_index().rename(
                        columns={df_used.index.name or "index": "Timestamp"}
                    )
                    subheader = "Integrated Survey Data"
                except Exception as e:
                    st.error(f"Failed to apply integration period for `{uf.name}`: {e}")
                    continue
            else:
                # 2) Show the raw data (from log._master) if available
                try:
                    raw_master = log._master  # original DataFrame, indexed by Timestamp
                    df_used = raw_master.reset_index().rename(columns={"Time": "Timestamp"})
                    subheader = "Raw Survey Data"
                except Exception as e:
                    st.error(f"Failed to load raw data for `{uf.name}`: {e}")
                    continue

            if "Excluded" not in df_used.columns:
                df_used["Excluded"] = False

            df_used = df_used[df_used.get("Excluded", False) == False]

            #measurement location
            with st.expander("Measurement Location", expanded=True):
                with st.form(key=f"location_{uf.name}_form"):
                    col1, col2, col3 = st.columns([2, 1, 1])

                    position_name = col1.text_input("Position name", value="")
                    latitude = col2.number_input("Latitude", format="%.6f")
                    longitude = col3.number_input("Longitude", format="%.6f")

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


            ### For excluding data in the future
            # df_display = df_used.copy()

            # # Reset index if you want a stable row identifier
            # if df_display.index.name is None or df_display.index.name == "":
            #     df_display.reset_index(inplace=True)
            #     df_display.rename(columns={"index": "RowID"}, inplace=True)

            # gb = GridOptionsBuilder.from_dataframe(df_display)
            # gb.configure_selection("multiple", use_checkbox=True)
            # gb.configure_default_column(
            #     editable=False,  # don't allow editing unless you want
            #     groupable=True,
            #     resizable=True,
            #     filter=True,
            #     sortable=True,
            # )
            # gb.configure_grid_options(domLayout='normal')  # can also try 'autoHeight'
            # grid_options = gb.build()

            # grid_response = AgGrid(
            #     df_display,
            #     gridOptions=grid_options,
            #     update_mode=GridUpdateMode.SELECTION_CHANGED,
            #     height=400,
            #     allow_unsafe_jscode=True,  # for better styling if needed
            # )
            
            # selected_rows = grid_response['selected_rows']

            # if st.button("Exclude Selected Rows"):
            #     indices_to_exclude = [row["index"] for row in selected_rows if "index" in row]
            #     df_used.loc[indices_to_exclude, "Excluded"] = True

            # log.set_data(df_used)
