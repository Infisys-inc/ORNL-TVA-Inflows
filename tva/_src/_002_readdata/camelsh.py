#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 00:18:58 2025

@author: vinhtran
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from basedataset_h import BaseDataset_hourly


class camelsh(BaseDataset_hourly):
    """Class to process hourly data in similar format as the CAMELS US dataset.

    Adapted/lightly modified from existing project code. Only the reader
    methods were made more robust to missing files, flexible datetime
    column names, and missing attributes.
    """

    def __init__(
        self,
        dynamic_input: Union[List[str], Dict[str, List[str]]],
        forcing: List[str],
        target: List[str],
        sequence_length: int,
        time_period: List[str],
        path_data: str,
        path_entities: str = None,
        entity: str = None,
        check_NaN: bool = True,
        path_additional_features: Optional[str] = "",
        predict_last_n: Optional[int] = 1,
        static_input: Optional[List[str]] = None,
        conceptual_input: Optional[List[str]] = None,
        custom_freq_processing: Optional[Dict[str, int]] = None,
        dynamic_embedding: Optional[bool] = False,
        unique_prediction_blocks: Optional[bool] = False,
        lookback_window: Optional[int] = 0,
    ):
        # store forcing early because some methods reference it
        self.forcing = forcing
        super(camelsh, self).__init__(
            dynamic_input=dynamic_input,
            target=target,
            sequence_length=sequence_length,
            time_period=time_period,
            path_data=path_data,
            path_entities=path_entities,
            entity=entity,
            check_NaN=check_NaN,
            path_additional_features=path_additional_features,
            predict_last_n=predict_last_n,
            static_input=static_input,
            conceptual_input=conceptual_input,
            custom_freq_processing=custom_freq_processing,
            dynamic_embedding=dynamic_embedding,
            unique_prediction_blocks=unique_prediction_blocks,
            lookback_window=lookback_window,
        )

    def _read_attributes(self) -> pd.DataFrame:
        """Read the catchments' attributes and return a DataFrame indexed by STAID."""
        path_attributes = Path(self.path_data) / "attributes"
        read_files = sorted(path_attributes.glob("attributes_*.csv"))

        if not read_files:
            raise FileNotFoundError(f"No attributes files found in {path_attributes}")

        dfs = []
        for file in read_files:
            try:
                # read with low_memory=False to avoid mixed-type column warnings for large files
                df = pd.read_csv(file, sep=',', header=0, dtype={'STAID': str}, low_memory=False)
            except:
                df = pd.read_csv(file, low_memory=False)
            if 'STAID' in df.columns:
                df = df.set_index('STAID')
            dfs.append(df)

        # Concatenate attribute files; drop duplicated columns
        df_attributes = pd.concat(dfs, axis=1)
        df_attributes = df_attributes.loc[:, ~df_attributes.columns.duplicated()]

        # Factorize non-numeric columns
        from pandas.api.types import is_numeric_dtype
        for col in df_attributes.columns:
            if not is_numeric_dtype(df_attributes[col].dtype):
                df_attributes[col], _ = pd.factorize(df_attributes[col], sort=True)

        # Select requested static inputs and reindex by requested entity ids to avoid KeyError
        ids = list(getattr(self, 'entities_ids', []))
        cols = [c for c in (getattr(self, 'static_input', []) or []) if c in df_attributes.columns]
        if not cols:
            raise KeyError("None of the requested static_input columns were found in attributes files")

        df_attributes = df_attributes.reindex(ids)[cols]

        return df_attributes

    def _read_data(self, catch_id: str) -> pd.DataFrame:
        """Read a specific catchment timeseries into a dataframe."""
        dfs = []
        for forcing in self.forcing:
            df = self._load_hourly_data(catch_id=catch_id, forcing=forcing)
            if len(self.forcing) > 1:
                df = df.rename(columns={col: f"{col}_{forcing}" for col in df.columns})
            dfs.append(df)

        df = pd.concat(dfs, axis=1)

        # optional: join discharges if available
        # df = df.join(self._load_hourly_discharge(catch_id=catch_id))

        return df

    def _load_hourly_data(self, catch_id: str, forcing: str) -> pd.DataFrame:
        """Read a specific catchment forcing timeseries with flexible datetime handling."""
        df = None

        # try NetCDF/Timeseries first
        path_nc = Path(self.path_data) / "timeseries" / f"{catch_id}.nc"
        if path_nc.exists():
            try:
                ds = xr.open_dataset(path_nc)
                df = ds.to_dataframe().reset_index()
                # find datetime-like columns
                from pandas.api.types import is_datetime64_any_dtype
                dt_cols = [c for c in df.columns if is_datetime64_any_dtype(df[c])]
                if dt_cols:
                    df = df.set_index(dt_cols[0])
                elif 'DateTime' in df.columns:
                    df = df.set_index('DateTime')
                elif 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df = df.set_index('date')
                else:
                    try:
                        df.index = pd.to_datetime(df.index)
                    except Exception:
                        pass
            except Exception:
                df = None

        # fallback to CSV hourly file
        if df is None:
            path_csv = Path(self.path_data) / "hourly" / f"{forcing}" / f"{catch_id}_hourly_nldas.csv"
            if not path_csv.exists():
                raise FileNotFoundError(f"Time series file not found for {catch_id}: {path_nc} or {path_csv}")

            # attempt common date column names
            parse_candidates = ['date', 'DateTime', 'datetime']
            parsed = False
            for col in parse_candidates:
                try:
                    df = pd.read_csv(path_csv, index_col=[col], parse_dates=[col])
                    parsed = True
                    break
                except Exception:
                    continue
            if not parsed:
                df = pd.read_csv(path_csv)
                for col in df.columns:
                    if 'date' in col.lower() or 'time' in col.lower():
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            df = df.set_index(col)
                            break
                        except Exception:
                            continue

        # Now add optional external Q/H series if requested
        if 'Q_obs_api' in getattr(self, 'target', []):
            path_out = Path(self.path_data) / "Out" / f"{catch_id}.csv"
            if not path_out.exists():
                raise FileNotFoundError(f"Out file not found: {path_out}")
            df2 = pd.read_csv(path_out)
            # find datetime column
            dt_col = None
            for c in ['DateTime'] + [c for c in df2.columns if 'date' in c.lower() or 'time' in c.lower()]:
                if c in df2.columns:
                    dt_col = c
                    break
            if dt_col is None:
                raise KeyError(f"No datetime column found in {path_out}")
            df2[dt_col] = pd.to_datetime(df2[dt_col], errors='coerce')
            df2 = df2.set_index(dt_col)
            if 'Q_obs_api' in df2.columns:
                df['Q_obs_api'] = df2['Q_obs_api'].reindex(df.index)
                df['Q'] = df['Q_obs_api']

        if 'H_obs_api' in getattr(self, 'target', []):
            path_out = Path(self.path_data) / "Out" / f"{catch_id}.csv"
            if not path_out.exists():
                raise FileNotFoundError(f"Out file not found: {path_out}")
            df2 = pd.read_csv(path_out)
            dt_col = None
            for c in ['DateTime'] + [c for c in df2.columns if 'date' in c.lower() or 'time' in c.lower()]:
                if c in df2.columns:
                    dt_col = c
                    break
            if dt_col is None:
                raise KeyError(f"No datetime column found in {path_out}")
            df2[dt_col] = pd.to_datetime(df2[dt_col], errors='coerce')
            df2 = df2.set_index(dt_col)
            if 'H_obs_api' in df2.columns:
                df['H_obs_api'] = df2['H_obs_api'].reindex(df.index)
                df['Q'] = df['H_obs_api']

        if 'Q_obs_api_norm' in getattr(self, 'target', []):
            path_norm = Path(self.path_data) / "Out" / f"normQ_{catch_id}.csv"
            if not path_norm.exists():
                raise FileNotFoundError(f"Normalized Q file not found: {path_norm}")
            df2 = pd.read_csv(path_norm)
            dt_col = None
            for c in ['DateTime'] + [c for c in df2.columns if 'date' in c.lower() or 'time' in c.lower()]:
                if c in df2.columns:
                    dt_col = c
                    break
            if dt_col is None:
                raise KeyError(f"No datetime column found in {path_norm}")
            df2[dt_col] = pd.to_datetime(df2[dt_col], errors='coerce')
            df2 = df2.set_index(dt_col)
            if 'Q_obs_api_norm' in df2.columns:
                df['Q_obs_api_norm'] = df2['Q_obs_api_norm'].reindex(df.index)
                df['Q'] = df['Q_obs_api_norm']

        if 'Q_camelsh_obs_norm' in getattr(self, 'target', []):
            path_norm = Path(self.path_data) / "Out" / f"normQ_{catch_id}.csv"
            if path_norm.exists():
                df2 = pd.read_csv(path_norm)
                dt_col = None
                for c in ['DateTime'] + [c for c in df2.columns if 'date' in c.lower() or 'time' in c.lower()]:
                    if c in df2.columns:
                        dt_col = c
                        break
                if dt_col is None:
                    raise KeyError(f"No datetime column found in {path_norm}")
                df2[dt_col] = pd.to_datetime(df2[dt_col], errors='coerce')
                df2 = df2.set_index(dt_col)
                if 'Q_camelsh_obs_norm' in df2.columns:
                    df['Q_camelsh_obs_norm'] = df2['Q_camelsh_obs_norm'].reindex(df.index)
            elif 'Streamflow' in df.columns and 'DRAIN_SQKM' in getattr(self, 'df_attributes', pd.DataFrame()).columns:
                area_values = np.asarray(self.df_attributes.loc[catch_id, 'DRAIN_SQKM']).ravel()
                area_km2 = float(area_values[0])
                df['Q_camelsh_obs_norm'] = df['Streamflow'] * 86400 * 1000 / (area_km2 * 10**6)
            else:
                raise FileNotFoundError(
                    f"Missing normalized streamflow file {path_norm} and unable to compute it from Streamflow."
                )
            df['Q'] = df['Q_camelsh_obs_norm']

        if 'Q_lookback' in (self.dynamic_input.get('1h') if isinstance(self.dynamic_input, dict) else []):
            df['Q_lookback'] = np.nan
            if getattr(self, 'lookback_window', 0) > 0:
                df.loc[df.index[self.lookback_window:], 'Q_lookback'] = df['Q'].iloc[:-self.lookback_window].values

        return df
