from typing import Dict, Union
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):

        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)

        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):

        out = self.net(x)

        return self.norm(x + out)


class TransformerBlock(nn.Module):

    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()

        self.attn = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.ff = PositionwiseFeedForward(hidden_size, dropout)

    def forward(self, x):

        x = self.attn(x)
        x = self.ff(x)

        return x


class MFTFT(nn.Module):
    """
    Multi-Frequency Temporal Fusion Transformer
    """

    def __init__(self, model_configuration: Dict[str, Union[int, float, str, dict]]):

        super().__init__()

        self.input_size = model_configuration["input_size_lstm"]
        self.hidden_size = model_configuration["hidden_size"]
        self.predict_last_n = model_configuration["predict_last_n"]
        self.custom_freq_processing = model_configuration["custom_freq_processing"]

        self.dropout_rate = model_configuration["dropout_rate"]

        self.num_heads = model_configuration.get("num_heads", 4)
        self.num_transformer_layers = model_configuration.get("num_transformer_layers", 2)

        # optional dynamic embeddings
        self.embedding_net = self._get_embedding_net(model_configuration)

        # input projection
        self.input_projection = nn.Linear(self.input_size, self.hidden_size)

        # static embedding
        self.static_embedding = None
        if model_configuration.get("static_input_size") is not None:

            self.static_embedding = nn.Linear(
                model_configuration["static_input_size"],
                self.hidden_size
            )

        # transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                self.hidden_size,
                self.num_heads,
                self.dropout_rate
            )
            for _ in range(self.num_transformer_layers)
        ])

        self.dropout = nn.Dropout(self.dropout_rate)

        self.output_layer = nn.Linear(
            self.hidden_size,
            model_configuration.get("out_features", 1)
        )

    @staticmethod
    def _get_embedding_net(model_configuration):

        if model_configuration.get("dynamic_embeddings"):

            embedding_net = nn.ModuleDict()

            for freq in model_configuration["custom_freq_processing"].keys():

                if isinstance(model_configuration["dynamic_input_size"], int):
                    input_size = model_configuration["dynamic_input_size"]
                else:
                    input_size = model_configuration["dynamic_input_size"][freq]

                embedding_net[freq] = nn.Linear(
                    input_size,
                    model_configuration["n_dynamic_channels_lstm"]
                )

        else:
            embedding_net = None

        return embedding_net

    def forward(self, sample: Dict[str, torch.Tensor]):

        process_tensor = []

        # process multi-frequency inputs
        for freq in self.custom_freq_processing.keys():

            x = sample["x_d_" + freq]

            if self.embedding_net:
                x = self.embedding_net[freq](x)

            process_tensor.append(x)

        # concatenate frequencies
        x = torch.cat(process_tensor, dim=1)

        # add static attributes
        if sample.get("x_s") is not None:

            static = sample["x_s"]

            if self.static_embedding is not None:
                static = self.static_embedding(static)

            static = static.unsqueeze(1).repeat(1, x.shape[1], 1)

            x = torch.cat((x, static), dim=2)

        # project to hidden size
        x = self.input_projection(x)

        # transformer encoder
        for layer in self.transformer_layers:
            x = layer(x)

        # select last N steps
        x = x[:, -self.predict_last_n:, :]

        x = self.dropout(x)

        out = self.output_layer(x)

        return {"y_sim": out}