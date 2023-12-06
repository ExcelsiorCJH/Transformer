import torch
import torch.nn as nn

from math import sqrt


class Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_keys: int = None,
        d_values: int = None,
        scale: float = None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ):
        super(Attention, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.n_heads = n_heads
        self.scale = scale
        self.output_attention = output_attention

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        # Q, K, V projection
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # Scaled Dot-Product Attention
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        self.scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(self.scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        out = self.out_projection(V.reshape(B, L, -1))

        if self.output_attention:
            return out, A
        else:
            return out, None
