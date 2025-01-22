from PrepareDataset.DataEncoder.BaseDataEncoder import BaseDataEncoder
from PrepareDataset.DataEncoder.ComorbidityEncoder import ComorbidityEncoder
from PrepareDataset.DataEncoder.TabularEncoder import TabularEncoder
from PrepareDataset.DataEncoder.RadiomicsEncoder import RadiomicsEncoder
import numpy as np
import re
import pandas as pd
from PrepareDataset.DataEncoder.PreprocessLogger import PreprocessLogger
import os


class FeatureCollector:
    def __init__(self, label_path, logger=None):
        self.features = {}
        base_data_encoder = BaseDataEncoder(df_file_path=label_path, logger=logger)
        self.features["label"] = base_data_encoder
        if logger is None:
            self.logger = PreprocessLogger("FeatureCollector")
        else:
            self.logger = logger

    def set_logger(self, logger):
        self.logger = logger

    def add_feature_set(self, feature_set_name, feature_set):
        self.features[feature_set_name] = feature_set
        # feature_set.set_logger(self.logger)

    def get_features(self, selected_features=None):
        if selected_features is None:
            selected_features = set(self.features.keys())
        features = self.features["label"].df
        for feature_set_name in selected_features - set(["label"]):
            if feature_set_name not in self.features:
                raise ValueError(f"Feature set {feature_set_name} not found")
            # merge on eid but remove duplicated columns
            intersected_cols = set(
                self.features[feature_set_name].df.columns
            ).intersection(set(features.columns))
            self.logger.info(f"Intersected columns: \n{intersected_cols}")
            if "eid" not in intersected_cols:
                raise ValueError(
                    f"Feature set {feature_set_name} does not contain 'eid' column"
                )
            features = features.merge(
                self.features[feature_set_name].df,
                on=list(intersected_cols),
                how="inner",
            )
        return features

    def encode_features(self, selected_features=None):
        if selected_features is None:
            selected_features = list(self.features.keys())
        for feature_set_name in selected_features:
            self.logger.info(f"Encoding {feature_set_name}")
            self.features[feature_set_name].encode()

    def _default_categorical_multiple_preprocess(
        self, tabular_encoder, drop_original=True
    ):
        # Pattern to identify columns and extract feature names
        pattern = re.compile(r"(.+)_array_(\d+)$")

        # Find all columns that match the pattern and group by feature name
        columns_to_encode = {}
        for col in tabular_encoder.df.columns:
            match = pattern.match(col)
            if match:
                feature_name = match.group(1)
                if feature_name not in columns_to_encode:
                    columns_to_encode[feature_name] = []
                columns_to_encode[feature_name].append(col)

        # Process each feature group
        for feature_name, cols in columns_to_encode.items():
            # One-hot encode these columns
            encoded_df = pd.get_dummies(tabular_encoder.df[cols], prefix=feature_name)

            # Aggregate one-hot encoded results if the same category appears in different columns
            final_columns = {
                col.split("_")[-1]: [
                    c for c in encoded_df.columns if f"_{col.split('_')[-1]}" in c
                ]
                for col in encoded_df.columns
            }
            for cat, cols in final_columns.items():
                tabular_encoder.df[feature_name + "_" + cat] = encoded_df[cols].sum(
                    axis=1
                )

            # Drop original columns if required
        if drop_original:
            for feature_name in columns_to_encode.keys():
                tabular_encoder.df.drop(
                    columns_to_encode[feature_name], axis=1, inplace=True
                )
        tabular_encoder.drop_unknown_features()

    def _default_categorical_preprocess(
        self,
        tabular_encoder,
        ordinal_encoding_columns,
        one_hot_encoding_columns,
        percentage_of_unknown=0.5,
        unite_sparse_categories_threshold=None,
    ):
        # check the instance of the TabularEncoder
        if not isinstance(tabular_encoder, TabularEncoder):
            self.logger.info("Skip: the input is not an instance of TabularEncoder")
            return

        selected_cols = (
            tabular_encoder.get_cols_by_type("Categorical single", by="description")
        ) + (tabular_encoder.get_cols_by_type("Categorical multiple", by="description"))
        dropped_cols = tabular_encoder.drop_unknown_columns(
            percentage_of_unknown=percentage_of_unknown, selected_cols=selected_cols
        )
        ordinal_encoding_columns = list(
            set(ordinal_encoding_columns) - set(dropped_cols)
        )
        one_hot_encoding_columns = list(
            set(one_hot_encoding_columns) - set(dropped_cols)
        )

        tabular_encoder.encode_ordinal(ordinal_encoding_columns)
        if unite_sparse_categories_threshold is not None:
            tabular_encoder.unite_sparse_categories(
                one_hot_encoding_columns, unite_sparse_categories_threshold
            )
        tabular_encoder.encode_one_hot(one_hot_encoding_columns)
        tabular_encoder.drop_unknown_features()

    def _impute_continuous(self, x, impute_method="mean"):
        # if the columns has inf values, replace them with "unknown"
        x = x.replace([np.inf, -np.inf], "unknown")
        if impute_method == "mean":
            x = x.replace("unknown", np.nan).astype(float)
            x = x.fillna(x.mean())
        else:
            raise ValueError(f"Unknown impute method {impute_method}")
        return x

    def _default_continuous_preprocess(
        self,
        tabular_encoder,
        percentage_of_unknown=0.5,
        impute_method="mean",
        scale_features=True,
    ):
        """
        If the number of unknown values in a column is more than percentage_of_unknown %, the column is dropped.
        Other continuous columns are converted to float and imputed.
        :param tabular_encoder:
        :param percentage_of_unknown:
        :return:
        """
        if not isinstance(tabular_encoder, TabularEncoder):
            self.logger.info("Skip: the input is not an instance of TabularEncoder")
            return

        selected_cols = tabular_encoder.get_cols_by_type(
            "Continuous", by="description"
        ) + tabular_encoder.get_cols_by_type("Integer", by="description")

        tabular_encoder.drop_unknown_columns(
            percentage_of_unknown=percentage_of_unknown, selected_cols=selected_cols
        )

        continuous_cols = tabular_encoder.get_cols_by_type(
            "Continuous", by="description"
        ) + tabular_encoder.get_cols_by_type("Integer", by="description")
        for col in continuous_cols:
            tabular_encoder.df[col] = self._impute_continuous(
                tabular_encoder.df[col], impute_method=impute_method
            )

        if scale_features and len(continuous_cols) > 0:
            tabular_encoder.scale_features(cols=continuous_cols)

    def _default_continuous_preprocess_leaving_unknown(
        self, tabular_encoder, scale_features=True
    ):
        if not isinstance(tabular_encoder, TabularEncoder):
            self.logger.info("Skip: the input is not an instance of TabularEncoder")
            return

        continuous_cols = tabular_encoder.get_cols_by_type(
            "Continuous", by="description"
        ) + tabular_encoder.get_cols_by_type("Integer", by="description")
        tabular_encoder.df[continuous_cols] = tabular_encoder.df[
            continuous_cols
        ].replace("unknown", np.nan)

        if scale_features and len(continuous_cols) > 0:
            tabular_encoder.scale_features(cols=continuous_cols)

    def _default_preprocess(self, tabular_encoder, settings):
        if isinstance(tabular_encoder, RadiomicsEncoder):
            self._default_radiomics_preprocess(
                tabular_encoder,
                settings["impute_method"],
                settings["percentage_of_unknown_continuous"],
            )
            return
        if not isinstance(tabular_encoder, TabularEncoder):
            self.logger.info("Skip: the input is not an instance of TabularEncoder")
            return
        # tabular_encoder.drop_unknown_columns(percentage_of_unknown=percentage_of_unknown)
        self._default_categorical_preprocess(
            tabular_encoder,
            settings["ordinal_encoding_columns"],
            settings["one_hot_encoding_columns"],
            settings["percentage_of_unknown_categorical"],
            settings["unite_sparse_categories_threshold"],
        )
        self._default_continuous_preprocess(
            tabular_encoder,
            settings["percentage_of_unknown_continuous"],
            settings["impute_method"],
            settings["scale_features"],
        )
        if settings["categorical_multiple"]:
            self._default_categorical_multiple_preprocess(tabular_encoder)

    def _default_preprocess_leaving_unknown(self, tabular_encoder, settings):
        if isinstance(tabular_encoder, RadiomicsEncoder):
            cols = list(set(tabular_encoder.df.columns) - set(["eid"]))
            tabular_encoder.df[cols] = tabular_encoder.df[cols].replace(
                "unknown", np.nan
            )
            tabular_encoder.scale_features(cols=cols)
            return

        if not isinstance(tabular_encoder, TabularEncoder):
            self.logger.info("Skip: the input is not an instance of TabularEncoder")
            return
        # tabular_encoder.drop_unknown_columns(percentage_of_unknown=percentage_of_unknown)
        self._default_categorical_preprocess(
            tabular_encoder,
            settings["ordinal_encoding_columns"],
            settings["one_hot_encoding_columns"],
            settings["percentage_of_unknown_categorical"],
            settings["unite_sparse_categories_threshold"],
        )
        self._default_continuous_preprocess_leaving_unknown(
            tabular_encoder, settings["scale_features"]
        )
        if settings["categorical_multiple"]:
            self._default_categorical_multiple_preprocess(tabular_encoder)

    def _default_radiomics_preprocess(
        self, radiomics_encoder, impute_method="mean", percentage_of_unknown=0.5
    ):
        if not isinstance(radiomics_encoder, RadiomicsEncoder):
            self.logger.info("Skip: the input is not an instance of RadiomicsEncoder")
            return
        radiomics_encoder.drop_unknown_columns(
            percentage_of_unknown=percentage_of_unknown
        )
        for col in radiomics_encoder.df.columns:
            if col == "eid":
                continue
            radiomics_encoder.df[col] = self._impute_continuous(
                radiomics_encoder.df[col], impute_method=impute_method
            )
        radiomics_encoder.scale_features()

    def preprocess_categorical(self, selected_features=None):
        if selected_features is None:
            selected_features = list(self.features.keys())
        for feature_set_name in selected_features:
            self._default_categorical_preprocess(self.features[feature_set_name])

    def preprocess_continuous(self, selected_features=None):
        if selected_features is None:
            selected_features = list(self.features.keys())
        for feature_set_name in selected_features:
            self._default_continuous_preprocess(self.features[feature_set_name])

    def preprocess_all(self, selected_features=None, settings=None):
        if selected_features is None:
            selected_features = list(self.features.keys())
        for feature_set_name in selected_features:
            self._default_preprocess(self.features[feature_set_name], settings)

    def preprocess_all_leaving_unknown(self, selected_features=None, settings=None):
        if selected_features is None:
            selected_features = list(self.features.keys())
        for feature_set_name in selected_features:
            self._default_preprocess_leaving_unknown(
                self.features[feature_set_name], settings
            )

    def _count_unknown_values(self, tabular_encoder):
        if not isinstance(tabular_encoder, TabularEncoder):
            self.logger.info("Skip: the input is not an instance of TabularEncoder")
            return
        for col in tabular_encoder.df.columns:
            if col == "eid":
                continue
            if "unknown" in tabular_encoder.df[col].value_counts():
                self.logger.info(col)
                self.logger.info(tabular_encoder.df[col].value_counts()["unknown"])
                # self.logger.info("-" * 15)

    def count_unknown_values(self, selected_features=None):
        if selected_features is None:
            selected_features = list(self.features.keys())
        for feature_set_name in selected_features:
            self._count_unknown_values(self.features[feature_set_name])

    def load_features(self, file_path_to_features, data_showcase_path):
        # list al csv files in the directory
        self.features = {}
        for file in os.listdir(file_path_to_features):
            if file.endswith(".csv"):
                feature_name = file.split(".")[0]
                if "radiomics" in feature_name:
                    self.features[feature_name] = RadiomicsEncoder(
                        df_file_path=os.path.join(file_path_to_features, file),
                        logger=self.logger,
                    )
                elif "comorbidit" in feature_name:
                    self.features[feature_name] = ComorbidityEncoder(
                        df_file_path=os.path.join(file_path_to_features, file),
                        logger=self.logger,
                    )
                else:
                    self.features[feature_name] = TabularEncoder(
                        data_showcase_path=data_showcase_path,
                        df_file_path=os.path.join(file_path_to_features, file),
                        logger=self.logger,
                    )
