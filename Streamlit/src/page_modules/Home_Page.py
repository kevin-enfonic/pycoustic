import streamlit as st

def render():
    st.title("Pycoustic Acoustic Survey Explorer")

    st.markdown("""
        Select your Sound Level Meter Manufacturer from the dropdown list in the sidebar.\\
        **"B&K"** maps the column headers from exporting via Measurement Partner Suite BZ-5503\\
        **"Other"** defaults to the Pycoustic standard header types.
        \\
        Upload Excel or CSV files in the sidebar to add them as logs.
        \\
        Choose the desired Resampling period (if applicable) and press **"Apply Resampling Period"** to resample the data. This can be done at any time.
        
        #### Logs
        Navigate to **"Logs"** when files have been uploaded to rename, adjust data in, exclude data, and input location information for each log.\\
        This page also gives a summary and graph for each log 
    
        #### Summary
        Navigate to **"Summary"** to view an overall summary of all logs
    
        #### Maps
        Navigate to **"Maps"** to view a map of all measurement locations. Coordinates must first be inputted in **"Logs"**
    """)
