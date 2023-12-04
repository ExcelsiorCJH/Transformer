import torch
import pandas as pd

from torch.utils.data import Dataset
from .preprocess import get_stamp_data


class ETTDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        label_len: int,
        pred_len: int,
        freq: str = "h",
        use_time_enc: bool = True,
    ):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.freq = freq
        self.use_time_enc = use_time_enc

        self.input_data = df.values
        self.target_data = df.values
        self.stamp_data = get_stamp_data(df, use_time_enc, freq)

    def __len__(self):
        return len(self.input_data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        past_values = self.input_data[s_begin:s_end]
        past_time_features = self.stamp_data[s_begin:s_end]
        future_values = self.target_data[r_begin:r_end]
        future_time_features = self.stamp_data[r_begin:r_end]

        return {
            "past_values": torch.FloatTensor(past_values),
            "past_time_features": torch.FloatTensor(past_time_features),
            "future_values": torch.FloatTensor(future_values),
            "future_time_features": torch.FloatTensor(future_time_features),
        }
