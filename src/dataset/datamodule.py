from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from .dataset import ETTDataset
from .preprocess import load_data, trn_val_tst_split


class ETTDataModule:
    def __init__(
        self,
        data_path: str,
        task_type: str = "M",
        freq: str = "h",
        target: str = "OT",
        seq_len: int = 96,
        label_len: int = 48,
        pred_len: int = 96,
        use_scaler: bool = True,
        use_time_enc: bool = True,
        batch_size: int = 32,
    ):
        self.data_path = data_path
        self.task_type = task_type
        self.freq = freq
        self.target = target
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.use_scaler = use_scaler
        self.use_time_enc = use_time_enc

        self.batch_size = batch_size

        self.scaler = None
        if self.use_scaler:
            self.scaler = StandardScaler()

        self.split_idx_dict = {
            "train": [0, 12 * 30 * 24],
            "val": [12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24],
            "test": [12 * 30 * 24 + 4 * 30 * 24 - seq_len, 12 * 30 * 24 + 8 * 30 * 24],
        }

        self.setup()

    def setup(self):
        df = load_data(self.data_path, task_type=self.task_type, target=self.target)
        data_dict = trn_val_tst_split(df, self.split_idx_dict, self.scaler)

        self.trainset = ETTDataset(
            data_dict["train"],
            self.seq_len,
            self.label_len,
            self.pred_len,
            self.freq,
            self.use_time_enc,
        )

        self.valset = ETTDataset(
            data_dict["val"],
            self.seq_len,
            self.label_len,
            self.pred_len,
            self.freq,
            self.use_time_enc,
        )

        self.testset = ETTDataset(
            data_dict["test"],
            self.seq_len,
            self.label_len,
            self.pred_len,
            self.freq,
            self.use_time_enc,
        )

    def train_dataloader(self):
        return DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.valset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )
