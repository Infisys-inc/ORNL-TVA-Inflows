import requests
import pandas as pd
import xarray as xr
import os
from datetime import datetime
import pytz
import numpy as np

def download_usgs_hourly_data(filename, site, start_date, end_date, save_netcdf=False):
    """
    Download hourly streamflow and water level data from USGS NWIS, convert to UTC,
    and create a complete hourly time series from 1980 to 2024 with missing data as NaN.
    If water level data is unavailable, set gage_height_m to NaN.
    Save to NetCDF only if data is available.

    Parameters:
    - filename: str, path to save the NetCDF file (e.g., 'Hourly/03339000_hourly.nc')
    - site: str, USGS site ID (e.g., '03339000')
    - start_date: str, 'YYYY-MM-DD'
    - end_date: str, 'YYYY-MM-DD'
    - save_netcdf: bool, whether to save output to NetCDF if data is available

    Returns:
    - pd.DataFrame with datetime (UTC), gage height (m), and streamflow (cms)
    """
    url = "https://waterservices.usgs.gov/nwis/iv/"
    parameter_codes = {
        '00065': 'gage_height_ft',
        '00060': 'streamflow_cfs'
    }
    
    all_data = []
    data_available = False

    for code, label in parameter_codes.items():
        params = {
            "format": "json",
            "sites": site,
            "startDT": start_date,
            "endDT": end_date,
            "parameterCd": code,
            "siteStatus": "all"
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Check if timeSeries data exists and has values
            if (not data['value']['timeSeries'] or 
                not data['value']['timeSeries'][0]['values'] or 
                not data['value']['timeSeries'][0]['values'][0]['value']):
                print(f"No data found for parameter {code} at site {site}.")
                continue

            values = data['value']['timeSeries'][0]['values'][0]['value']
            series_data = []
            for item in values:
                if item['value'] and item['value'].lower() != 'nan':
                    local_time = pd.to_datetime(item['dateTime'])
                    utc_time = local_time.astimezone(pytz.UTC)
                    series_data.append({
                        'datetime_utc': utc_time,
                        label: float(item['value'])
                    })

            if series_data:
                df_temp = pd.DataFrame(series_data)
                all_data.append(df_temp)
                data_available = True

        except (KeyError, IndexError, requests.RequestException) as e:
            print(f"Error fetching data for parameter {code} at site {site}: {e}")

    # Create full index for the time series
    full_index = pd.date_range(start='1980-01-01', end='2024-12-31 23:00:00', freq='h', tz='UTC')
    columns = ['gage_height_m', 'streamflow_cms']
    
    if not data_available:
        print(f"No data downloaded for site {site}.")
        return pd.DataFrame(index=full_index, columns=columns, data=np.nan)

    # Merge dataframes on datetime_utc
    df = all_data[0]
    if len(all_data) > 1:
        for df_temp in all_data[1:]:
            df = pd.merge(df, df_temp, on='datetime_utc', how='outer')

    df.set_index('datetime_utc', inplace=True)

    # Unit conversions
    if 'gage_height_ft' in df.columns:
        df['gage_height_m'] = df['gage_height_ft'] * 0.3048
        df = df.drop(columns=['gage_height_ft'])
    else:
        df['gage_height_m'] = np.nan  # Set to NaN if no gage height data

    if 'streamflow_cfs' in df.columns:
        df['streamflow_cms'] = df['streamflow_cfs'] * 0.0283168
        df = df.drop(columns=['streamflow_cfs'])
    else:
        df['streamflow_cms'] = np.nan  # Set to NaN if no streamflow data

    # Reindex to full time series and ensure correct columns
    df = df.reindex(full_index)
    df = df[columns].astype(float)
    df = df.where(df.notna(), np.nan)

    # Save to NetCDF only if data is available (non-empty and has some non-NaN values)
    if save_netcdf and not df[columns].isna().all().all():
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        ds = xr.Dataset(
            {
                'gage_height_m': ('time', df['gage_height_m']),
                'streamflow_cms': ('time', df['streamflow_cms'])
            },
            coords={'time': df.index}
        )

        ds['gage_height_m'].attrs = {'units': 'meters', 'long_name': 'Gage height'}
        ds['streamflow_cms'].attrs = {'units': 'cubic meters per second', 'long_name': 'Streamflow'}
        ds.attrs = {'site_id': site, 'source': 'USGS NWIS', 'timezone': 'UTC'}
        ds.to_netcdf(filename, format='NETCDF4')
        print(f"Saved to {filename}")

    return df

if __name__ == "__main__":
    # Read CSV with STAID as string
    site_list = pd.read_csv('attributes/attributes_gageii_BasinID.csv', dtype={'STAID': str})
    ID_all = site_list['STAID']
    start = '1980-01-01'
    end = '2024-12-31'
    for site_id in ID_all:
        filename = f"Hourly/{site_id}_hourly.nc"
        
        if not os.path.exists(filename):
            try:
                df = download_usgs_hourly_data(filename, site_id, start, end, save_netcdf=True)
                if not df.empty:
                    print(f"Data for {site_id}:\n{df.head()}")
                    print(f"Data shape: {df.shape}")
                else:
                    print(f"No data retrieved for {site_id}")
            except Exception as e:
                print(f"Error processing {site_id}: {e}")