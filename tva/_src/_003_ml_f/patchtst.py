from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchTST(nn.Module):
    """PatchTST-style encoder for the existing multi-frequency sample format.

    The dataset already converts the 365-day hourly window into a shorter sequence
    of daily and hourly blocks. This model embeds each frequency, concatenates the
    blocks along time, patches that token sequence, and applies a Transformer
    encoder before predicting the target.
    """

    def __init__(self, model_configuration: Dict[str, Union[int, float, str, dict]]):
        super().__init__()
        self.predict_last_n = model_configuration.get("predict_last_n", 1)
        self.custom_freq_processing = model_configuration["custom_freq_processing"]
        self.patch_len = model_configuration.get("patch_len", 7)
        self.patch_stride = model_configuration.get("patch_stride", self.patch_len)
        self.d_model = model_configuration.get("d_model", model_configuration.get("hidden_size", 128))
        self.n_dynamic_channels = model_configuration.get(
            "n_dynamic_channels_patchtst",
            model_configuration.get("n_dynamic_channels_lstm", 10),
        )
        self.dropout = nn.Dropout(model_configuration.get("dropout_rate", 0.1))

        self.embedding_net = self._get_embedding_net(model_configuration)
        self.patch_projection = nn.Linear(self.patch_len * self.n_dynamic_channels, self.d_model)

        n_static = model_configuration.get("n_static_features", 0)
        self.static_projection = nn.Linear(n_static, self.d_model) if n_static > 0 else None

        max_tokens = sum(freq_info["n_steps"] for freq_info in self.custom_freq_processing.values())
        max_patches = max(1, (max_tokens + self.patch_stride - 1) // self.patch_stride)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_patches, self.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=model_configuration.get("n_heads", 4),
            dim_feedforward=model_configuration.get("dim_feedforward", self.d_model * 4),
            dropout=model_configuration.get("dropout_rate", 0.1),
            activation=model_configuration.get("activation", "gelu"),
            batch_first=True,
            norm_first=model_configuration.get("norm_first", False),
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=model_configuration.get("n_transformer_layers", model_configuration.get("no_of_layers", 2)),
        )
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(
            self.d_model,
            self.predict_last_n * model_configuration.get("out_features", 1),
        )
        self.out_features = model_configuration.get("out_features", 1)

    @staticmethod
    def _get_embedding_net(model_configuration: Dict[str, Union[int, float, str, dict]]) -> nn.ModuleDict:
        embedding_net = nn.ModuleDict()
        for freq in model_configuration["custom_freq_processing"].keys():
            if isinstance(model_configuration["dynamic_input_size"], int):
                input_size = model_configuration["dynamic_input_size"]
            else:
                input_size = model_configuration["dynamic_input_size"][freq]
            embedding_net[freq] = nn.Linear(
                in_features=input_size,
                out_features=model_configuration.get(
                    "n_dynamic_channels_patchtst",
                    model_configuration.get("n_dynamic_channels_lstm", 10),
                ),
            )
        return embedding_net

    def _patch_sequence(self, x: torch.Tensor) -> torch.Tensor:
        remainder = (x.size(1) - self.patch_len) % self.patch_stride
        if x.size(1) < self.patch_len:
            pad_len = self.patch_len - x.size(1)
        elif remainder == 0:
            pad_len = 0
        else:
            pad_len = self.patch_stride - remainder

        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        patches = x.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        patches = patches.transpose(-1, -2).contiguous()
        return patches.view(patches.size(0), patches.size(1), -1)

    def forward(self, sample: Dict[str, torch.Tensor]):
        process_tensor = []
        for freq in self.custom_freq_processing.keys():
            x = sample["x_d_" + freq]
            process_tensor.append(self.embedding_net[freq](x))

        x = torch.cat(process_tensor, dim=1)
        x = self._patch_sequence(x)
        x = self.patch_projection(x)

        if self.static_projection is not None and sample.get("x_s") is not None:
            x = x + self.static_projection(sample["x_s"]).unsqueeze(1)

        x = x + self.position_embedding[:, : x.size(1), :]
        x = self.encoder(self.dropout(x))
        x = self.norm(x.mean(dim=1))
        y = self.head(self.dropout(x))
        y = y.view(y.size(0), self.predict_last_n, self.out_features)

        return {"y_sim": y}
