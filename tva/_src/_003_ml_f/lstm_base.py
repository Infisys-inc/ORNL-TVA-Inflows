from typing import Dict, Union
import torch
import torch.nn as nn

class LSTM_base(nn.Module):
    """LSTM network. 

    Parameters
    ----------
    hyperparameters : Dict[str, Union[int, float, str, dict]]
        Various hyperparameters of the model
    """
    def __init__(self, hyperparameters: Dict[str, Union[int, float, str, dict]]):
        super().__init__()
        self.input_size_lstm = hyperparameters['input_size_lstm']
        self.hidden_size = hyperparameters['hidden_size']
        self.num_layers = hyperparameters['no_of_layers']
        
        self.lstm = nn.LSTM(input_size = self.input_size_lstm, 
                            hidden_size = self.hidden_size, 
                            batch_first = True,
                            num_layers = self.num_layers)

        
        self.dropout = torch.nn.Dropout(hyperparameters['drop_out_rate'])
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=1)
           
    def forward(self, x: torch.Tensor):
        """Forward pass of lstm networj 

        Parameters
        ----------
        x_lstm: torch.Tensor
            Tensor of size [batch_size, time_steps, input_size_lstm].

        Returns
        -------
        pred: Dict[str, torch.Tensor]
        """
        # initialize hidden state with zeros
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=True, dtype=torch.float32, 
                         device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=True, dtype=torch.float32,
                         device=x.device)
        
        out, (hn_1, cn_1) = self.lstm(x, (h0, c0))
        out = out[:,-1,:] # sequence to one
        out = self.dropout(out)
        out = self.linear(out)

        return {'y_sim': out}




class LSTM_MO(nn.Module):
    """LSTM network for multiple outputs

    Parameters
    ----------
    hyperparameters : Dict[str, Union[int, float, str, dict]]
        Various hyperparameters of the model
    """
    def __init__(self, hyperparameters: Dict[str, Union[int, float, str, dict]]):
        super().__init__()
        self.input_size_lstm = hyperparameters['input_size_lstm']
        self.hidden_size = hyperparameters['hidden_size']
        self.num_layers = hyperparameters['no_of_layers']
        
        self.lstm = nn.LSTM(input_size = self.input_size_lstm, 
                            hidden_size = self.hidden_size, 
                            batch_first = True,
                            num_layers = self.num_layers)

        
        self.dropout = torch.nn.Dropout(hyperparameters['drop_out_rate'])
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=1)
        self.n_out = hyperparameters['n_out']
           
    def forward(self, x: torch.Tensor):
        """Forward pass of lstm networj 

        Parameters
        ----------
        x_lstm: torch.Tensor
            Tensor of size [batch_size, time_steps, input_size_lstm].

        Returns
        -------
        pred: Dict[str, torch.Tensor]
        """
        # initialize hidden state with zeros
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=True, dtype=torch.float32, 
                         device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=True, dtype=torch.float32,
                         device=x.device)
        
        out, (hn_1, cn_1) = self.lstm(x, (h0, c0))
        out = out[:,-self.n_out:,:] # sequence to one
        out = self.dropout(out)
        out = self.linear(out)

        return {'y_sim': out}
        
        

import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNN_base(nn.Module):
    """Graph Neural Network for streamflow prediction.
    
    Parameters
    ----------
    hyperparameters : Dict[str, Union[int, float, str, dict]]
        Various hyperparameters of the model
    """
    def __init__(self, hyperparameters: Dict[str, Union[int, float, str, dict]]):
        super().__init__()
        self.input_size_gnn = hyperparameters['input_size_gnn']
        self.hidden_size = hyperparameters['hidden_size']
        self.num_layers = hyperparameters['no_of_layers']
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(self.input_size_gnn, self.hidden_size))
        for _ in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.hidden_size, self.hidden_size))
        
        self.dropout = nn.Dropout(hyperparameters['drop_out_rate'])
        self.linear = nn.Linear(self.hidden_size, 1)

    def forward(self, x, edge_index):
        """
        Parameters
        ----------
        x : torch.Tensor
            Node features, shape [num_nodes, input_size_gnn].
        edge_index : torch.LongTensor
            Graph edges, shape [2, num_edges].
        
        Returns
        -------
        pred : Dict[str, torch.Tensor]
        """
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        out = self.linear(x)
        return {'y_sim': out}



class Transformer_base(nn.Module):
    """Transformer network. 
    Parameters
    ----------
    hyperparameters : Dict[str, Union[int, float, str, dict]]
        Various hyperparameters of the model
    """
    def __init__(self, hyperparameters: Dict[str, Union[int, float, str, dict]]):
        super().__init__()
        self.input_size = hyperparameters['input_size_lstm']  # model input size
        self.hidden_size = hyperparameters['hidden_size']  # embedding dimension / transformer model dimension
        self.num_layers = hyperparameters['no_of_layers']  # number of transformer encoder layers
        self.num_heads = hyperparameters.get('num_heads', 4)  # number of attention heads
        self.dropout_rate = hyperparameters['drop_out_rate']
        self.seq_length = hyperparameters.get('seq_length', 10)  # sequence length (time steps)
        
        # Input embedding to map input features to hidden_size embedding space
        self.embedding = nn.Linear(self.input_size, self.hidden_size)
        
        # Positional encoding parameter to add position signals to embeddings
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.seq_length, self.hidden_size))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.num_heads, dropout=self.dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Final linear layer to output a single value
        self.linear = nn.Linear(self.hidden_size, 1)
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass of transformer network.
        
        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape [batch_size, time_steps, input_size].
        
        Returns
        -------
        pred : Dict[str, torch.Tensor]
        """
        # Input embedding
        x = self.embedding(x)  # Shape: [batch_size, time_steps, hidden_size]
        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]
        # Rearrange for transformer: [seq_len, batch_size, hidden_size]
        x = x.permute(1, 0, 2)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # [seq_len, batch_size, hidden_size]
        # Take output of last sequence step
        x = x[-1, :, :]  # Shape: [batch_size, hidden_size]
        
        x = self.dropout(x)
        out = self.linear(x)  # Shape: [batch_size, 1]
        return {'y_sim': out}




from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

class xLSTM_base(nn.Module):
    """xLSTM network for sequence regression with matching context_length and input sequence length."""
    def __init__(self, hyperparameters: Dict[str, Union[int, float, str, dict]]):
        super().__init__()
        self.input_size = hyperparameters['input_size_lstm']
        self.hidden_size = hyperparameters['hidden_size']
        self.num_layers = hyperparameters['no_of_layers']

        # Validate hyperparameters for sequence_length
        sequence_length = hyperparameters.get('seq_length', None)
        assert sequence_length is not None, "sequence_length must be set in hyperparameters"

        # Validate num_heads for attention
        num_heads = hyperparameters.get('num_heads', 1)
        if num_heads > self.input_size or self.input_size % num_heads != 0:
            print(f"Adjusting num_heads from {num_heads} to 1 due to input_size_lstm = {self.input_size}")
            num_heads = 1

        self.cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=num_heads
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda" if torch.cuda.is_available() else "vanilla",
                    num_heads=num_heads,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent"
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu")
            ),
            context_length=sequence_length,   # Must equal input seq length
            num_blocks=self.num_layers,
            embedding_dim=self.input_size,
            slstm_at=[0],                    # Valid sLSTM block index
        )
        self.xlstm = xLSTMBlockStack(self.cfg)
        self.dropout = nn.Dropout(hyperparameters['drop_out_rate'])
        self.linear = nn.Linear(self.input_size, 1)

    def forward(self, x: torch.Tensor):
        # x shape: [batch_size, sequence_length, input_size]
        # Assert input sequence length matches config context_length
        assert x.shape[1] == self.cfg.context_length, (
            f"Input sequence length {x.shape[1]} must match context_length {self.cfg.context_length}"
        )

        out = self.xlstm(x)            # [batch, sequence_length, input_size]
        out = out[:, -1, :]            # Last time step output
        out = self.dropout(out)
        out = self.linear(out)
        return {'y_sim': out}

class xLSTM_MO(nn.Module):
    """xLSTM network for multiple outputs with valid slstm_at configuration."""
    def __init__(self, hyperparameters: Dict[str, Union[int, float, str, dict]]):
        super().__init__()
        self.input_size = hyperparameters['input_size_lstm']
        self.hidden_size = hyperparameters['hidden_size']
        self.num_layers = hyperparameters['no_of_layers']
        self.n_out = hyperparameters['n_out']

        self.cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4)
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda" if torch.cuda.is_available() else "native",
                    num_heads=4, conv1d_kernel_size=4, bias_init="powerlaw_blockdependent"
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu")
            ),
            context_length=hyperparameters.get('sequence_length', 256),
            num_blocks=self.num_layers,
            embedding_dim=self.input_size,
            slstm_at=[0],  # Adjusted to valid index
        )
        self.xlstm = xLSTMBlockStack(self.cfg)
        self.dropout = nn.Dropout(hyperparameters['drop_out_rate'])
        self.linear = nn.Linear(self.input_size, 1)

    def forward(self, x: torch.Tensor):
        out = self.xlstm(x)                # [batch, sequence_length, input_size]
        out = out[:, -self.n_out:, :]     # Last n_out timesteps
        out = self.dropout(out)
        out = self.linear(out)
        return {'y_sim': out}
