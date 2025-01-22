import pandas as pd
import numpy as np
import torch
from pycox.models import DeepHitSingle


class DataPreprocessor:
    def __init__(self, csv_file):
        self.csv_file = csv_file

    def load_data(self):
        self.df = pd.read_csv(self.csv_file)

    def split_data(self):
        self.train_df = self.df[self.df["split"] == "train"]
        self.test_df = self.df[self.df["split"] == "test"]

    @staticmethod
    def load_features_from_npz(file_path):
        data = np.load(file_path)
        return data["features"]

    def get_features(self, df):
        return np.array([self.load_features_from_npz(fp) for fp in df["file_path"]])

    def discretize_time_init(self, num_durations):
        self.labtrans = DeepHitSingle.label_transform(num_durations)

    def discretize_time(self, durations, events, num_durations=10):
        return self.labtrans.fit_transform(durations, events)

    def process_data(self, num_durations=10):
        self.load_data()
        self.split_data()
        train_features = self.get_features(self.train_df)
        test_features = self.get_features(self.test_df)

        # Discretize time
        self.discretize_time_init(num_durations)
        train_t, train_e = self.discretize_time(
            self.train_df["time_to_event"].values,
            self.train_df["event"].values,
            num_durations,
        )
        test_t, test_e = self.discretize_time(
            self.test_df["time_to_event"].values,
            self.test_df["event"].values,
            num_durations,
        )
        train_t = torch.from_numpy(train_t).long()
        test_t = torch.from_numpy(test_t).long()
        train_e = torch.from_numpy(train_e).float()
        test_e = torch.from_numpy(test_e).float()

        # Convert to PyTorch tensors
        train_x = (
            torch.from_numpy(train_features).float().view(train_features.shape[0], -1)
        )
        test_x = (
            torch.from_numpy(test_features).float().view(test_features.shape[0], -1)
        )

        # Convert event and time-to-event columns to tensors
        # train_e = torch.from_numpy(self.train_df['event'].values).float()
        # train_t = torch.from_numpy(self.train_df['time_to_event'].values).float()
        # test_e = torch.from_numpy(self.test_df['event'].values).float()
        # test_t = torch.from_numpy(self.test_df['time_to_event'].values).float()

        # Discretize time
        train_data = (train_x, (train_t, train_e))
        test_data = (test_x, (test_t, test_e))

        return train_data, test_data
