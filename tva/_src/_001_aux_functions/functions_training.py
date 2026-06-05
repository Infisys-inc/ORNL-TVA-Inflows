import torch

def mean_error_basin_averaged(y_sim: torch.Tensor, y_obs: torch.Tensor, 
                              per_basin_target_std: torch.Tensor) -> torch.Tensor:
    """Basin-averaged mean error (bias).

    Computes the simple mean error between observed and simulated discharges, averaged across all valid values.

    Parameters
    ----------
    y_sim : torch.Tensor
        Simulated discharges.
    y_obs : torch.Tensor
        Observed discharges.
    per_basin_target_std : torch.Tensor
        Not used in this loss, included to match function signature.

    Returns
    -------
    loss : torch.Tensor
        Mean error (bias) between observed and simulated values.
    """

    # Flatten and mask out NaNs
    mask = ~torch.isnan(torch.flatten(y_obs)) & ~torch.isnan(torch.flatten(y_sim))
    y_sim_masked = torch.flatten(y_sim)[mask]
    y_obs_masked = torch.flatten(y_obs)[mask]

    # Compute mean error (bias)
    loss = torch.mean(y_obs_masked - y_sim_masked)

    return loss



def heteroscedastic_loss(y_true,mu, sigma):
        """
        mu: Predicted mean (LSTM output)
        sigma: Predicted standard deviation (LSTM output)
        y_true: Actual true values
        
        Returns:
        NLL Loss
        """
        # Flatten all tensors to 1D
        mu = mu.flatten()
        sigma = sigma.flatten()
        y_true = y_true.flatten()

        # Remove NaN values
        valid_indices = ~torch.isnan(mu) & ~torch.isnan(sigma) & ~torch.isnan(y_true)
        
        mu = mu[valid_indices]
        sigma = sigma[valid_indices]
        y_true = y_true[valid_indices]

        # Ensure that sigma is positive (standard deviation cannot be negative)
        sigma = torch.clamp(sigma, min=1e-6)  # Prevent division by zero or negative values

        # Ensure that torch.pi is treated as a tensor
        pi = torch.tensor(torch.pi, dtype=torch.float32)

        # NLL loss for Gaussian distribution
        nll_loss = 0.5 * (torch.log(2 * pi) + torch.log(sigma ** 2) + ((y_true - mu) ** 2) / (sigma ** 2))
        
        # Return the mean NLL loss
        return torch.mean(nll_loss)




def nse_basin_averaged(y_sim: torch.tensor, y_obs: torch.tensor, 
                       per_basin_target_std: torch.tensor) -> torch.Tensor:
    """Basin-averaged Nash--Sutcliffe Efficiency.

    Loss function where the squared errors are weighed by the std of each basin. A description of this function is 
    available at [#]_.

    Parameters
    ----------
    y_sim : torch.Tensor
        simulated discharges.
    y_obs : torch.Tensor
        observed discharges.
    per_basin_target_std : torch.Tensor
        Standard deviation of the discharge (during training period) for the respective basins. 
    
    Returns
    -------
    loss: torch.Tensor
        value of the basin-averaged NSE

    """
        
    # Create a combined mask to remove NaN values from both observation and simulation
    mask = ~torch.isnan(torch.flatten(y_obs)) & ~torch.isnan(torch.flatten(y_sim))
    
    # Apply the mask to both time series
    y_sim_masked = torch.flatten(y_sim)[mask]
    y_obs_masked = torch.flatten(y_obs)[mask]
    
    # Handle the standard deviation values
    if per_basin_target_std.numel() == 1:  # Check if it's a single scalar value
        basin_std_masked = per_basin_target_std.view(1).expand(y_obs_masked.shape)
    else:
        basin_std_masked = torch.flatten(per_basin_target_std)[mask]
    
    # Calculate the loss components
    squared_error = (y_sim_masked - y_obs_masked)**2
    weights = 1 / (basin_std_masked + 0.1)**2  # The 0.1 is a small constant for numerical stability
    loss = weights * squared_error
    
    return torch.mean(loss)

def nse_multiple_out(y_sim: torch.Tensor, y_obs: torch.Tensor, 
                     per_basin_target_std: torch.Tensor) -> torch.Tensor:
    """Multiple-output Basin-averaged Nash--Sutcliffe Efficiency.

    Parameters
    ----------
    y_sim : torch.Tensor
        Simulated discharges.
    y_obs : torch.Tensor
        Observed discharges.
    per_basin_target_std : torch.Tensor
        Standard deviation of discharge per basin.
    
    Returns
    -------
    loss: torch.Tensor
        Value of the basin-averaged NSE.
    """
    
    # Ensure per_basin_target_std has the shape [batch_size, 1] before expanding it to [batch_size, num_outputs]
    if per_basin_target_std.dim() == 3:  # Check if it's [batch_size, 1, 1]
        per_basin_target_std = per_basin_target_std.squeeze(2)  # Remove the third dimension, making it [batch_size, 1]
    
    # Expand to match the output dimension of y_sim
    per_basin_target_std = per_basin_target_std.expand(-1, y_sim.shape[1])  # Expand to [batch_size, num_outputs]
    
    # Flatten tensors
    y_sim_flat = y_sim.flatten()
    y_obs_flat = y_obs.flatten()
    basin_std_flat = per_basin_target_std.flatten()

    # Create mask where y_obs is not NaN
    mask = ~torch.isnan(y_obs_flat)

    # Apply mask
    y_sim_masked = y_sim_flat[mask]
    y_obs_masked = y_obs_flat[mask]
    basin_std_masked = basin_std_flat[mask]

    # Compute weighted squared error
    squared_error = (y_sim_masked - y_obs_masked) ** 2
    weights = 1 / (basin_std_masked + 0.1) ** 2  # Add small constant for stability
    loss = weights * squared_error

    return torch.mean(loss)



def weighted_rmse(y_sim: torch.tensor, y_obs: torch.tensor) -> torch.Tensor:
    """Weighted root mean squared error. 
    
    Weighted root mean squared error between measured and observed discharges. However it uses both the discharge and
    the logarithm of the discharge. The second part aims at improving low flows representation. The function was taken
    from [#]_. 
    
    Parameters
    ----------
    y_sim : torch.Tensor
        simulated discharges.
    y_obs : torch.Tensor
        observed discharges.

    Returns
    -------
    loss: torch.Tensor
        Weighted root mean squared error

    References
    ----------
    .. [#] Feng, D., Liu, J., Lawson, K., & Shen, C. (2022). Differentiable, learnable, regionalized process-based 
        models with multiphysical outputs can approach state-of-the-art hydrologic prediction accuracy. Water Resources 
        Research, 58, e2022WR032404. https://doi.org/10.1029/2022WR032404
    
    """
        
    # calculate mask to avoid nan in observation to affect the loss
    mask = ~torch.isnan(torch.flatten(y_obs))
    
    y_sim_masked = torch.flatten(y_sim)[mask]
    y_obs_masked = torch.flatten(y_obs)[mask]
    y_sim_transformed = torch.log10(torch.sqrt(y_sim_masked + 1e-6) + 0.1)
    y_obs_transformed = torch.log10(torch.sqrt(y_obs_masked + 1e-6) + 0.1)

    loss = 0.75 * torch.sqrt(torch.mean((y_sim_masked - y_obs_masked)**2)) +\
        0.25*torch.sqrt(torch.mean((y_sim_transformed - y_obs_transformed)**2))

    return loss