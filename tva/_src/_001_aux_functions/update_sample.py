# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 12:30:17 2025

@author: vinhtn1
"""
import numpy as np
import pandas as pd
import torch

def update_streamflow_hourly(sample, Obs, past_state=None, Opt=False):
    """
    Updates a data sample for real-time streamflow forecasting.

    Parameters
    ----------
    sample : dict
        A dictionary containing input features and observations.
    Obs : np.ndarray
        The latest observed streamflow (shape: [1, 1, 1] or similar).
    past_state : np.ndarray, optional
        Past observation state to append to Obs if Opt is True.
    Opt : bool, optional
        If True, appends past_state to Obs and updates only 'x_s';
        otherwise, removes the first timestep from all relevant arrays.

    Returns
    -------
    updated_sample : dict
        A dictionary with updated features and observations.
    """
    updated_sample = dict(sample)  # create a shallow copy

    if Opt:
        if past_state is not None:
            #past_state = past_state.unsqueeze(0)  # from (1,1) → (1,1,1)
            past_state = past_state.view(1, 1, 1)  # explicitly reshape to (1,1,1)
            Obs = torch.cat([past_state, Obs], dim=0)
        updated_sample['x_s'][:, -1] = Obs[:-1, 0, 0]
    else:
        updated_sample['x_d_1D'] = sample['x_d_1D'][1:, :, :]
        updated_sample['x_d_1h'] = sample['x_d_1h'][1:, :, :]
        updated_sample['y_obs'] = sample['y_obs'][1:, :, :]
        try:
            updated_sample['basin_std'] = sample['basin_std'][1:, :, :]
        except:
            updated_sample['basin_std'] = None
            
        updated_sample['basin'] = sample['basin'][1:, ]
        updated_sample['date'] = sample['date'][1:, :]
        updated_sample['x_s'] = sample['x_s'][1:, :]
        updated_sample['x_s'][:, -1] = sample['y_obs'][:-1, 0, 0]

    return updated_sample

def update_streamflow_daily(sample,past_state = None, Opt = False):
    """
    This function is used to update data sample in case of real-time forecasting at different lead time.
    The data (streamfow or water level) at previous time step will be updated to same to predict next time step.
    Parameters
    ----------
    sample : TYPE
        DESCRIPTION.
    past_state : TYPE, optional
        DESCRIPTION. The default is None.
    Opt : TYPE, optional
        DESCRIPTION. The default is False: Remove the first array of data

    Returns
    -------
    updated_sample : TYPE
        DESCRIPTION.

    """
    
    updated_sample = sample
    return updated_sample
