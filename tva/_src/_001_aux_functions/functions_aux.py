import os
import random
import numpy as np
import torch


def create_folder(folder_path: str):
    """Create a folder to store the results.

    Checks if the folder where one will store the results exist. If it does not, it creates it.

    Parameters
    ----------
    folder_path : str
        Path to the location of the folder

    """
    if not os.path.exists(folder_path):
        # Create the folder
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")


def set_random_seed(seed: int=None):
    """Set a seed for various packages to be able to reproduce the results.

    Parameters
    ----------
    seed : int
        Number of the seed

    """

    if seed is None:
        seed = int(np.random.uniform(low=0, high=1e6))

    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


def write_report(file_path: str, text: str):
    """Write a given text into a text file.
    
    If the file where one wants to write does not exists, it creates a new one.

    Parameters
    ----------
    file_path : str
        Path to the file where 
    text : str
        Text that wants to be added

    """
    if os.path.exists(file_path):
        append_write = 'a' # append if already exists   
    else:
        append_write = 'w' # make a new file if not

    highscore = open(file_path , append_write)
    highscore.write(text + '\n')
    highscore.close()


def combine_scaler(data):
    """
    Combines data from multiple CAMELS entries into a single dictionary with concatenated tensors.
    
    Args:
        data (dict): Dictionary with CAMELS data, where each key is a station ID and value is a dict
                    containing 'x_d_mean', 'x_d_std', 'y_mean', 'y_std', 'x_s_mean', 'x_s_std' tensors.
    
    Returns:
        dict: Combined dictionary with concatenated tensors for each key.
    """
    # Initialize result dictionary
    combined = {
        'x_d_mean': [],
        'x_d_std': [],
        'y_mean': [],
        'y_std': [],
        'x_s_mean': [],
        'x_s_std': []
    }
    
    # Collect tensors from each entry
    for station_id, station_data in data.items():
        for key in combined.keys():
            try:
                combined[key].append(station_data[key])
            except:
                print('')
    
    # Concatenate tensors for each key
    for key in combined.keys():
        # Stack tensors along a new dimension (dim=0)
        try:
            combined[key] = torch.stack(combined[key], dim=0)
        except:
            print('')
    
    return combined

def compute_scalers(data):
    """
    Computes scalers (mean and std) for all basins based on combined CAMELS data.
    
    Args:
        combined_data (dict): Dictionary with combined tensors for 'x_d_mean', 'x_d_std',
                             'y_mean', 'y_std', 'x_s_mean', 'x_s_std'.
    
    Returns:
        dict: Dictionary containing mean and std scalers for each key, with shapes matching
              the feature dimensions (excluding the basin dimension).
    """
    combined_data = combine_scaler(data)
    scalers = {}
    
    for key, tensor in combined_data.items():
        # Compute mean along basin dimension (dim=0), ignoring NaNs
        try:
            if key == 'x_s_std':
                std_scaler = torch.std(combined_data['x_s_mean'], dim=0, unbiased=True, keepdim=False)
                # If std contains NaNs, replace with zeros to avoid issues in normalization
                std_scaler = torch.where(torch.isnan(std_scaler), torch.ones_like(std_scaler), std_scaler)
                std_scaler = torch.where(std_scaler == 0, torch.ones_like(std_scaler), std_scaler)
                scalers[f"{key}"] = std_scaler
            else:
                mean_scaler = torch.nanmean(tensor, dim=0)
                mean_scaler = torch.where(torch.isnan(mean_scaler), torch.zeros_like(mean_scaler), mean_scaler)
                scalers[f"{key}"] = mean_scaler
        except:
            print()
        
    return scalers

