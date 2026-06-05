import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit, prange
from typing import List, Union, Dict
import glob
import torch
import xarray as xr

class FLUXNET_READ():
    
    @staticmethod
    def read_data(path_data: str, site_ID: str, input_variables: List[str]=None)-> pd.DataFrame:
        """Read the catchments` timeseries

        Parameters
        ----------
        path_data : str
            Path to the CAMELS GB directory.
        site_ID : str
            identifier of the basin.
        forcings : List[str]
            Not used, is just to have consistency with CAMELS-US. 

        Returns
        -------
        df: pd.DataFrame
            Dataframe with the catchments` timeseries
        """
        #path_timeseries = Path(path_data)
        #pattern = f'*{site_ID}*'
        #Folder = list(path_timeseries.glob(pattern))
        #path_timeseries = Folder[0]
        #print(path_timeseries)
        path_timeseries = Path(path_data) / 'FLUXNET'
        pattern = f'*{site_ID}*DD*'
        read_files = list(path_timeseries.glob(pattern))
        #print(read_files)
        df = pd.read_csv(read_files[0])
        df = df.set_index('TIMESTAMP')
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        if 'NETRAD' in df.columns:
            df.loc[df['NETRAD'] < 0, 'NETRAD'] = np.nan         # set negative net radiation values to NaN
            if 'LE_F_MDS' in df.columns:
                df.loc[df['LE_F_MDS'] < 0, 'LE_F_MDS'] = np.nan
                df.loc[df['NETRAD'] < 0, 'LE_F_MDS'] = np.nan
                df.loc[df['NETRAD'] <= df['LE_F_MDS'], 'LE_F_MDS'] = np.nan
                df.loc[df['NETRAD'] <= df['LE_F_MDS'], 'NETRAD'] = np.nan 
        if 'sm_rootzone' in input_variables:
            path_timeseries = Path(path_data) / 'SMAP'
            pattern = f'*{site_ID}*'
            smap_files = list(path_timeseries.glob(pattern))  # Search in the same folder
            if smap_files:
                path_smap = smap_files[0]
                # Read NDVI file
                df3 = pd.read_csv(path_smap)
                df3['dt'] = pd.to_datetime(df3['date'], format="%Y-%m-%d")
                df3 = df3.set_index('dt')    
                # Align  data with main dataframe
                df['sm_rootzone'] = df3['sm_rootzone'].reindex(df.index)
                df['sm_surface'] = df3['sm_surface'].reindex(df.index)
                
        # Read soil moisture from GLEAM        
        if 'sm_surface_gleam' in input_variables:
            path_timeseries = Path(path_data) / 'GLEAM'
            pattern = f'*{site_ID}*'
            smap_files = list(path_timeseries.glob(pattern))  # Search in the same folder
            if smap_files:
                path_smap = smap_files[0]
                # Read NDVI file
                df3 = pd.read_csv(path_smap)
                df3['dt'] = pd.to_datetime(df3['date'], format="%Y-%m-%d")
                df3 = df3.set_index('dt')    
                # Align  data with main dataframe
                df['sm_rootzone_gleam'] = df3['sm_rootzone_gleam'].reindex(df.index)
                df['sm_surface_gleam'] = df3['sm_surface_gleam'].reindex(df.index)
                
        if 'NDVI' in input_variables:
            path_timeseries = Path(path_data) / 'NDVI'
            ndvi_pattern = '*NDVI*'
            ndvi_files = list(path_timeseries.glob(ndvi_pattern))  # Search in the same folder
            #print(ndvi_files)
            if ndvi_files:
                path_ndvi = ndvi_files[0]
    
                # Read NDVI file
                df2 = pd.read_csv(path_ndvi)

                df2['dt'] = pd.to_datetime(df2[' dt'], format="%Y-%m-%d")
                df2 = df2.set_index('dt')
                df2['NDVI'] = df2[' value_mean']
    
                # Align NDVI data with main dataframe
                df['NDVI'] = df2['NDVI'].reindex(df.index)
                
                # Linearly interpolate missing NDVI values
                ndvi_series = pd.to_numeric(df['NDVI'], errors='coerce')
                df['NDVI'] = ndvi_series.interpolate()
                #df['NDVI'] = df['NDVI'].fillna(method='bfill').fillna(method='ffill')
        # NEED TO ADD A CODE TO READ SOIL DATA FROM SMAP HERE
        
        
        for var in input_variables:
            if var not in df.columns:
                df[var] = np.nan
        #Replace all values of -999, -9999, -99999 as NaN
        df.replace([-999, -9999, -99999], np.nan, inplace=True)
        #print(df['NDVI'])
        #num_nan = df['LE_F_MDS'].isna().sum()
        
        #print(f"Number of NaN values in LE_F_MDS: {num_nan}")
        return df
    
    
    @staticmethod
    def get_data(path_data: str, catch_id: str, input_variables: List[str], target_variables: List[str],time_period: Union[None, List[str], Dict[str, str]])-> pd.DataFrame:
        timeseries = {}
        df = FLUXNET_READ.read_data(path_data,catch_id,input_variables)
        if isinstance(time_period, list):
            df = df.loc[time_period[0]:time_period[1], :]
        
        # Filter for specific time period is there is any [list]. If custom time periods are used, what we do it
        # run the model for the whole period and then filter the training/testing subsets.
        df = df.loc[:, input_variables + target_variables]

        # save information
        timeseries['df'] = df
        timeseries['inputs']= df.loc[:, input_variables].to_numpy()
        timeseries['target'] = df.loc[:, target_variables].to_numpy().reshape((-1,1))
        
        # Need to convert the unit of data
        
        return df
    
    
    @staticmethod
    def convert_unit_fluxnet(data)-> np.array:
        """
        qs: Specific humidity                                    [kg/kg] (torch.Tensor) 
        Rn: Net radiation                                       [W/m^2]        -->   [W/m^2] (torch.Tensor)
        Ts: Surface temperature                                 [C]            -->   [K] (torch.Tensor)
        theta_root: Soil moisture content                     [%]            -->   [-]
        theta_surface: Soil moisture content                   [%]            -->   [-]
        Sd: snow depth
        NDVI:                                                   [-]         --> [-]
        OW:                                                     [-]         --> []
        SC:                                                     []
        """
        data[:,2] = data[:,2] + 274.15
        data[:,3] = data[:,3] + 274.15
        #data[:,4] = data[:,3]/100
        #data[:,5] = 1
        return data


class CAMELS_READ():
    
    @staticmethod
    def read_attributes(path_data: str) -> pd.DataFrame:
        """Read the catchments` attributes

        Parameters
        ----------
        path_data : str
            Path to the CAMELS GB directory.

        Returns
        -------
        df: pd.DataFrame
            Dataframe with the catchments` attributes
        """
        # files that contain the attributes
        path_attributes = Path(path_data)
        read_files = list(path_attributes.glob('*attributes*csv'))

        dfs = []
        # Read each CSV file into a DataFrame and store it in list
        for file in read_files:
            df = pd.read_csv(file, sep=',', header=0, dtype={'gauge_id': str})
            df.set_index('gauge_id', inplace=True)
            dfs.append(df)
        # Join all dataframes
        df_attributes= pd.concat(dfs, axis=1)

        return df_attributes
    
    @staticmethod
    def read_data(path_data: str, catch_id: str, forcings: List[str]=None)-> pd.DataFrame:
        """Read the catchments` timeseries

        Parameters
        ----------
        path_data : str
            Path to the CAMELS GB directory.
        catch_id : str
            identifier of the basin.
        forcings : List[str]
            Not used, is just to have consistency with CAMELS-US. 

        Returns
        -------
        df: pd.DataFrame
            Dataframe with the catchments` timeseries
        """
        path_timeseries = Path(path_data) / 'timeseries'
        patterns = [
            f'{catch_id}.csv', f'*_{catch_id}.csv', f'*_{catch_id}_*.csv',
            f'{catch_id}_*.csv', f'{catch_id}_*.txt', f'{catch_id}.nc'
        ]
        
        for pattern in patterns:
            read_files = list(path_timeseries.glob(pattern))
            if len(read_files) == 1:
                path_timeseries = read_files[0]
                break
            
        #path_timeseries = Path(path_data) / 'timeseries' / f'CAMELS_GB_hydromet_timeseries_{catch_id}_19701001-20150930.csv'
        # load time series
        df = pd.read_csv(path_timeseries)
        df = df.set_index('date')
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
        return df
    
    
    @staticmethod
    def get_data(path_data: str, catch_id: str, input_variables: List[str], target_variables: List[str],time_period: Union[None, List[str], Dict[str, str]],forcings: List[str]=None)-> pd.DataFrame:
        timeseries = {}
        df = CAMELS_READ.read_data(path_data,catch_id,forcings)
        if isinstance(time_period, list):
            df = df.loc[time_period[0]:time_period[1], :]
        
        # Filter for specific time period is there is any [list]. If custom time periods are used, what we do it
        # run the model for the whole period and then filter the training/testing subsets.
        df = df.loc[:, input_variables + target_variables]

        # save information
        timeseries['df'] = df
        timeseries['inputs']= df.loc[:, input_variables].to_numpy()
        timeseries['target'] = df.loc[:, target_variables].to_numpy().reshape((-1,1))
        return df


class CAMELSH_READ():
    
    @staticmethod
    def read_attributes(path_data: str) -> pd.DataFrame:
        """Read the catchments` attributes

        Parameters
        ----------
        path_data : str
            Path to the CAMELS GB directory.

        Returns
        -------
        df: pd.DataFrame
            Dataframe with the catchments` attributes
        """
        # files that contain the attributes
        path_attributes = Path(path_data) / "attributes"
        read_files = list(path_attributes.glob("attributes_*.csv"))

        dfs = []
        # Read each CSV file into a DataFrame and store it in list
        for file in read_files:
            print(file)
            df = pd.read_csv(file, sep=',', header=0, dtype={'STAID': str})
            df.set_index('STAID', inplace=True)
            dfs.append(df)
        
        # Join all dataframes
        df_attributes= pd.concat(dfs, axis=1)
        
        # Encode categorical attributes in case there are any
        for column in df_attributes.columns:
            if df_attributes[column].dtype not in ['float64', 'int64']:
                df_attributes[column], _ = pd.factorize(df_attributes[column], sort=True)
        

        return df_attributes
    
    @staticmethod
    def read_data(path_data: str, catch_id: str, forcings: List[str]=None)-> pd.DataFrame:
        
        """Read a specific catchment forcing timeseries"""
    
        path_timeseries = Path(path_data) / "timeseries" / f"{catch_id}.nc"
    
        ds = xr.open_dataset(path_timeseries)
    
        # Ensure the time dimension is datetime
        if "time" in ds.dims or "time" in ds.coords:
            time_var = "time"
        elif "DateTime" in ds.dims or "DateTime" in ds.coords:
            time_var = "DateTime"
        else:
            raise KeyError("No time dimension found in dataset")
    
        # Convert to DataFrame and enforce datetime index
        df = ds.to_dataframe().reset_index()
        df[time_var] = pd.to_datetime(df[time_var])
        df = df.set_index(time_var).sort_index()
    
        return df
        
    
    @staticmethod
    def get_data(path_data: str, catch_id: str, input_variables: List[str], target_variables: List[str],time_period: Union[None, List[str], Dict[str, str]],forcings: List[str]=None)-> pd.DataFrame:
        timeseries = {}
        df = CAMELSH_READ.read_data(path_data,catch_id,forcings)
        if isinstance(time_period, list):
            df = df.loc[time_period[0]:time_period[1], :]
        
        # Filter for specific time period is there is any [list]. If custom time periods are used, what we do it
        # run the model for the whole period and then filter the training/testing subsets.
        df = df.loc[:, input_variables + target_variables]

        # save information
        timeseries['df'] = df
        timeseries['inputs']= df.loc[:, input_variables].to_numpy()
        timeseries['target'] = df.loc[:, target_variables].to_numpy().reshape((-1,1))
        
        df_attributes = CAMELSH_READ.read_attributes(path_data)	
        return df,df_attributes
    



@njit()
def validate_samples(x: np.ndarray, y: np.ndarray, attributes: np.ndarray, seq_length: int, predict_last_n:int=1, 
                     check_NaN:bool=True,) -> np.ndarray:
    
    """Checks for invalid samples due to NaN or insufficient sequence length.

    This function was taken from Neural Hydrology [#]_ and adapted for our specific case. 
        
    Parameters
    ----------
    x : np.ndarray
        array of dynamic input;
    y : np.ndarray
        arry of target values;
    attributes : np.ndarray
        array containing the static attributes;
    seq_length : int
        Sequence lengths; one entry per frequency
    predict_last_n: int
        Number of values that want to be used to calculate the loss
    check_NaN : bool
        Boolean to specify if Nan should be checked or not


    Returns
    -------
    flag:np.ndarray 
        Array has a value of 1 for valid samples and a value of 0 for invalid samples.
    """
    # Initialize vector to store the flag. 1 means valid sample for training
    flag = np.ones(x.shape[0])

    for i in prange(x.shape[0]):  # iterate through all samples

        # too early, not enough information
        if i < seq_length - 1:
            flag[i] = 0  
            continue

        if check_NaN:
            # any NaN in the dynamic inputs makes the sample invalid
            x_sample = x[i-seq_length+1 : i+1, :]
            if np.any(np.isnan(x_sample)):
                flag[i] = 0
                continue

        if check_NaN:
            # all-NaN in the targets makes the sample invalid
            y_sample = y[i-predict_last_n+1 : i+1]
            if np.all(np.isnan(y_sample)):
                flag[i] = 0
                continue

        # any NaN in the static features makes the sample invalid
        if attributes is not None and check_NaN:
            if np.any(np.isnan(attributes)):
                flag[i] = 0

    return flag



