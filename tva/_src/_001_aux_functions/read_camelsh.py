#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 14:02:50 2026

@author: vinhtran
"""
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import scipy.io

def read_camelsh(path_data: str, catch_id: str) -> pd.DataFrame:
    """
    Read a specific catchment timeseries into a dataframe.

    Returns a dataframe with:
    - DateTime as index
    - timestamp column (for pipeline)
    - Streamflow column
    - ID column
    """
    # -------------------
    # Load daily/camelsh data
    # -------------------
    path_timeseries = Path(path_data) / "timeseries" / f"{catch_id}.nc"
    ds = xr.open_dataset(path_timeseries, engine="h5netcdf")
    df = ds.to_dataframe().reset_index().set_index("DateTime")

   
    # Drop old Streamflow column if exists
    if "Streamflow" in df.columns:
        df = df.drop(columns="Streamflow")

    # -------------------
    # Load hourly streamflow and merge
    # -------------------
    path_hourly = Path(path_data) / "Hourly" / f"{catch_id}_hourly.nc"
    ds2 = xr.open_dataset(path_hourly, engine="h5netcdf")
    df2 = ds2.to_dataframe().reset_index().set_index("time",drop=False)
    df2["time"] = pd.to_datetime(df2["time"])

    df['timestamp'] = df2['time'].reindex(df.index)
    df['Streamflow'] = df2['streamflow'].reindex(df.index)

    # -------------------
    # Add ID column
    # -------------------
    df["ID"] = 1

    return df