import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, enc_layers: list[nn.Module], norm_layer: nn.Module = None):
        super(Encoder, self).__init__()

        self.enc_layers = nn.ModuleList(enc_layers)
        self.norm_layer = norm_layer

    def forward(self, x: torch.Tensor):
        attns = []
        for enc_layer in self.enc_layers:
            x, attn = enc_layer(x)
            attns.append(attn)

        if self.norm_layer is not None:
            x = self.norm_layer(x)

        return x, attns


class EncoderLayer(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        d_ff: int = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super(EncoderLayer, self).__init__()

        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor):
        # 1. compute attention
        new_x, attn = self.attention(queries=x, keys=x, values=x)

        # 2. add and norm
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)

        # 3. positionwise feed forward
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn
