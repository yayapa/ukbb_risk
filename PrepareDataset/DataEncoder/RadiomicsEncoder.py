from PrepareDataset.DataEncoder.BaseDataEncoder import BaseDataEncoder
import pandas as pd


class RadiomicsEncoder(BaseDataEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def encode(self):
        # replace in all columns pd.NaN with unknown
        self.df = self.df.fillna("unknown")
