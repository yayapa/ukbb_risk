import os
from datetime import datetime

import torch
from pycox.models import DeepHitSingle
from torchtuples.callbacks import EarlyStopping


# Define the model


class ModelTrainer:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        # self._create_experiment_folder()

    def set_model(self, model):
        self.model = model

    def set_experiment_dir(self, experiment_dir):
        self.experiment_dir = experiment_dir

    def _create_experiment_folder(self):
        """
        Create store folder for storing experiment results
        """
        time = datetime.now()
        folder_name = time.strftime("%Y-%m-%d_%H-%M-%S/")
        path = os.path.join(self.STORE_DIR, "experiments/", folder_name)
        os.makedirs(path)
        # os.makedirs(path + "model")
        # os.makedirs(path + "images")
        # os.makedirs(path + "results")
        # self.store_dir = path
        self.experiment_dir = path

    def create_store_folder(self, store_name):
        self.store_dir = os.path.join(self.experiment_dir, store_name)
        os.makedirs(self.store_dir)
        os.makedirs(os.path.join(self.store_dir, "model"))
        os.makedirs(os.path.join(self.store_dir, "results"))

    def train(self, batch_size=32, epochs=1000, early_stopping_patience=None):
        if early_stopping_patience is not None:
            callbacks = [EarlyStopping(patience=early_stopping_patience)]
        else:
            callbacks = None
        log = self.model.fit(
            self.train_data[0],
            self.train_data[1],
            batch_size,
            epochs,
            callbacks,
            verbose=True,
        )  # , val_data=self.test_data.repeat(10).cat())
        return log
