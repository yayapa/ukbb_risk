import pandas as pd
from sklearn.preprocessing import StandardScaler


class BaseDataEncoder:
    def __init__(self, df_file_path, logger, eids_to_read=None):
        self.df = pd.read_csv(df_file_path)
        # extract all rows with eids from eids_to_read
        if eids_to_read is not None:
            self.df = self.df[self.df["eid"].isin(eids_to_read)]

        self._find_cols_with_many_array_instances()
        self.logger = logger

    def encode(self):
        pass

    def _find_cols_with_many_array_instances(self):
        self.cols_with_many_array_instances = []
        for col in self.df.columns:
            if "_array" in col:
                col_name = col.split("_array")[0]
                self.cols_with_many_array_instances.append(col_name)
        self.cols_with_many_array_instances = list(
            set(self.cols_with_many_array_instances)
        )

    def scale_features(self, cols=None):
        if cols is None:
            cols = list(set(self.df.columns) - set(["eid"]))
        df_scaled = StandardScaler().fit_transform(self.df[cols])
        df_scaled_df = pd.DataFrame(df_scaled, columns=cols, index=self.df.index)
        self.df = pd.concat(
            [self.df.drop(cols, axis=1), df_scaled_df], axis=1, join="inner"
        )

    def drop_unknown_columns(self, percentage_of_unknown=0.25, selected_cols=None):
        if selected_cols is None:
            selected_cols = self.df.columns

        cols_to_drop = []
        for col in selected_cols:
            if (
                "unknown" in self.df[col].value_counts()
                and self.df[col].value_counts()["unknown"] / self.df.shape[0]
                > percentage_of_unknown
            ) and col not in self.cols_with_many_array_instances:
                cols_to_drop.append(col)
        self.logger.info(
            f"Dropping columns that have more than {percentage_of_unknown}% unknown values: \n{cols_to_drop}"
        )
        self.df.drop(cols_to_drop, axis=1, inplace=True)
        return cols_to_drop
