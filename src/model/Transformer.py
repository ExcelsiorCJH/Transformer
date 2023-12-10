import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import DataEmbedding
from .attention import Attention
from .encoder_decoder import Encoder, Decoder
from .encoder_decoder import EncoderLayer, DecoderLayer


class Transformer(nn.Module):
    def __init__(
        self,
        task_name: str = "long_term_forecast",
        pred_len: int = 96,
        seq_len: int = None,
        num_class: int = None,
        enc_in: int = 7,
        dec_in: int = 7,
        c_out: int = 7,
        d_model: int = 512,
        embed_type: str = "time_features",
        freq: str = "h",
        dropout: float = 0.1,
        n_heads: int = 8,
        d_keys: int = None,
        d_values: int = None,
        d_ff: int = 2048,
        scale: float = None,
        attention_dropout: float = 0.1,
        output_attention: bool = True,
        activation: str = "gelu",
        num_enc_layers: int = 2,
        num_dec_layers: int = 1,
    ):
        super(Transformer, self).__init__()

        self.task_name = task_name
        self.pred_len = pred_len
        self.output_attention = output_attention

        # 1. Encoder embedding layer
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed_type, freq, dropout)

        # 2. Encoder
        enc_layer = EncoderLayer(
            attention=Attention(
                d_model,
                n_heads,
                d_keys,
                d_values,
                scale,
                attention_dropout,
                output_attention,
            ),
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
        )

        self.encoder = Encoder(
            enc_layers=[enc_layer for _ in range(num_enc_layers)],
            norm_layer=nn.LayerNorm(d_model),
        )

        # 3. Decoder
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            # 3.1 Decoder embedding layer
            self.dec_embedding = DataEmbedding(
                dec_in, d_model, embed_type, freq, dropout
            )

            # 3.2 Decoder
            dec_layer = DecoderLayer(
                self_attention=Attention(
                    d_model,
                    n_heads,
                    d_keys,
                    d_values,
                    scale,
                    attention_dropout,
                    output_attention,
                ),
                cross_attention=Attention(
                    d_model,
                    n_heads,
                    d_keys,
                    d_values,
                    scale,
                    attention_dropout,
                    output_attention,
                ),
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
            )

            self.decoder = Decoder(
                dec_layers=[dec_layer for _ in range(num_dec_layers)],
                norm_layer=nn.LayerNorm(d_model),
                projection=nn.Linear(d_model, c_out),
            )
        elif self.task_name == "imputation" or self.task_name == "anomaly_detection":
            self.projection = nn.Linear(d_model, c_out)
        elif self.task_name == "classification":
            self.dropout = nn.Dropout(dropout)
            self.projection = nn.Linear(d_model * seq_len, num_class)

    def forecast(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        future_values: torch.Tensor,
        future_time_features: torch.Tensor,
    ):
        enc_emb = self.enc_embedding(x=past_values, x_features=past_time_features)
        enc_out, enc_attns = self.encoder(enc_emb)

        dec_emb = self.dec_embedding(x=future_values, x_features=future_time_features)
        dec_out, dec_attns, cross_attns = self.decoder(dec_emb, enc_out)

        return_dict = {"last_hidden_states": dec_out}
        if self.output_attention:
            return_dict["encoder_attentions"] = enc_attns
            return_dict["decoder_attentions"] = dec_attns
            return_dict["cross_attentions"] = cross_attns

        return return_dict

    def imputation(self, past_values: torch.Tensor, past_time_features: torch.Tensor):
        enc_out = self.enc_embedding(x=past_values, x_features=past_time_features)
        enc_out, enc_attns = self.encoder(enc_out)

        dec_out = self.projection(enc_out)

        return_dict = {"last_hidden_states": dec_out}
        if self.output_attention:
            return_dict["encoder_attentions"] = enc_attns

        return return_dict

    def anomaly_detection(self, past_values: torch.Tensor):
        enc_out = self.enc_embedding(past_values, None)
        enc_out, enc_attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)

        return_dict = {"last_hidden_states": dec_out}
        if self.output_attention:
            return_dict["encoder_attentions"] = enc_attns

        return return_dict

    def classification(
        self, past_values: torch.Tensor, past_time_features: torch.Tensor
    ):
        enc_out = self.enc_embedding(past_values, None)
        enc_out, enc_attns = self.encoder(enc_out, attn_mask=None)

        output = F.gelu(enc_out)
        output = self.dropout(output)
        output = output * past_time_features.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)

        return_dict = {"last_hidden_states": output}
        if self.output_attention:
            return_dict["encoder_attentions"] = enc_attns

        return return_dict

    def forward(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        future_values: torch.Tensor,
        future_time_features: torch.Tensor,
    ):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            output = self.forecast(
                past_values, past_time_features, future_values, future_time_features
            )

        elif self.task_name == "imputation":
            output = self.imputation(past_values, past_time_features)
        elif self.task_name == "classification":
            output = self.classification(past_values, past_time_features)
        elif self.task_name == "anomaly_detection":
            output = self.anomaly_detection(past_values, past_time_features)

        return output
