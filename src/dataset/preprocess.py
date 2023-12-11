import numpy as np
import pandas as pd

from gluonts.time_feature import time_features_from_frequency_str


def load_data(data_path: str, task_type: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    if task_type == "S":
        df = df[[target]]

    return df


def trn_val_tst_split(
    df: pd.DataFrame, split_idx_dict: dict[str, list[int]], scaler=None
) -> dict[str, pd.DataFrame]:
    if scaler:
        s_idx, e_idx = split_idx_dict["train"]
        train_df = df[s_idx:e_idx]
        scaler.fit(train_df)
        df[df.columns] = scaler.transform(df[df.columns])

    data_dict = {}
    for stage in split_idx_dict:
        s_idx, e_idx = split_idx_dict[stage]
        data_dict[stage] = df[s_idx:e_idx]

    return data_dict


def get_stamp_data(df: pd.DataFrame, use_time_enc: bool = True, freq: str = "h"):
    stamp_df = pd.DataFrame()
    stamp_df["date"] = df.index
    if use_time_enc:
        dates = pd.to_datetime(stamp_df["date"].values)
        stamp_data = np.vstack(
            [feat(dates) for feat in time_features_from_frequency_str(freq)]
        )
        stamp_data = stamp_data.transpose(1, 0)
    else:
        if freq == "h":
            stamp_df["month"] = stamp_df["date"].apply(lambda row: row.month)
            stamp_df["day"] = stamp_df["date"].apply(lambda row: row.day)
            stamp_df["weekday"] = stamp_df["date"].apply(lambda row: row.weekday())
            stamp_df["hour"] = stamp_df["date"].apply(lambda row: row.hour)
            stamp_data = stamp_df.drop(["date"], axis=1).values
        elif freq == "t":
            stamp_df["month"] = stamp_df.date.apply(lambda row: row.month, 1)
            stamp_df["day"] = stamp_df.date.apply(lambda row: row.day, 1)
            stamp_df["weekday"] = stamp_df.date.apply(lambda row: row.weekday(), 1)
            stamp_df["hour"] = stamp_df.date.apply(lambda row: row.hour, 1)
            stamp_df["minute"] = stamp_df.date.apply(lambda row: row.minute, 1)
            stamp_df["minute"] = stamp_df.minute.map(lambda x: x // 15)
            stamp_df = stamp_df.drop(["date"], 1).values
    return stamp_data
