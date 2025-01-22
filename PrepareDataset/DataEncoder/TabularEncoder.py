import os
from PrepareDataset.DataEncoder.BaseDataEncoder import BaseDataEncoder
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class TabularEncoder(BaseDataEncoder):
    """
    Encoder for the tabular data in the UK Biobank dataset
    """

    COLS_TO_READ = ["FieldID", "Field", "Coding", "ValueType"]
    CODING_FOLDER = (
        "/home/dmitrii/GitHub/ukbb_risk_assessment/PrepareDataset/resources/coding"
    )
    DOWNLOAD_LINK = "https://biobank.ndph.ox.ac.uk/ukb/codown.cgi?id="
    UNKNOWN_SET = ["unknown", "prefer not to answer", "do not know"]
    UNITED_CATEGORIES_SYMBOL = "other (united)"

    def __init__(self, data_showcase_path, **kwargs):
        super().__init__(**kwargs)
        self.data_showcase = pd.read_csv(data_showcase_path, usecols=self.COLS_TO_READ)

    def set_logger(self, logger):
        self.logger = logger

    def encode(self, with_instance_id=False):
        """
        Main method to encode the data
        """
        for col in self.df.columns:
            if "-" in col:
                self.df[col] = self.df[col].apply(
                    lambda x: self.process_field_value(x, col)
                )
        cols_with_many_array_instances = [
            col
            for col in self.df.columns
            if "-" in col and self.get_number_array_instances(col) > 1
        ]
        for col in self.df.columns:
            if "-" in col:
                with_array_id = col in cols_with_many_array_instances
                self.df.rename(
                    columns={
                        col: self.get_field_description(
                            col,
                            with_instance_id=with_instance_id,
                            with_array_id=with_array_id,
                        )
                    },
                    inplace=True,
                )
        self.cols_with_many_array_instances = [
            self.get_field_description(
                col, with_instance_id=with_instance_id, with_array_id=True
            )
            for col in cols_with_many_array_instances
        ]
        return self.df

    def process_field_value(self, value, column_name):
        field_id = self.extract_field_id(column_name)
        value_type = self.get_value_type(field_id)
        try:
            if value_type == "Categorical single":
                coding = self.get_coding(self.get_field_coding(field_id))
                return self.process_categorical_single(value, coding)
            elif value_type == "Categorical multiple":
                coding = self.get_coding(self.get_field_coding(field_id))
                return self.process_categorical_single(value, coding)
            elif value_type == "Integer":
                return self.process_integer(value)
            elif value_type == "Continuous":
                return self.process_continuous(value)
            else:
                self.logger.error(
                    f"Unknown value type {value_type} of the field {field_id} with value {value}"
                )
                return value
        except Exception as e:
            self.logger.error(
                f"Error processing field {field_id} and value {value} with type {type(value)} and value_type {value_type} and error {e}"
            )
            return value
            # raise ValueError(f"Unknown value type {self.get_value_type(field_id)}")

    def drop_unknown_features(self, unknown_set=None):
        if unknown_set is None:
            unknown_set = self.UNKNOWN_SET
        cols_to_drop = []
        for col in self.df.columns:
            for unknown in unknown_set:
                if unknown in col:
                    cols_to_drop.append(col)
        self.logger.info(
            f"Dropping columns that have {unknown_set} in columns: \n{cols_to_drop}"
        )
        self.df.drop(cols_to_drop, axis=1, inplace=True)

    def drop_unusual_columns(self):
        # drop columns with more than 1 "-" and more than one "."
        cols_to_drop = [
            col for col in self.df.columns if col.count("-") > 1 or col.count(".") > 1
        ]
        self.logger.info(
            f"Dropping columns that have unexpected format: \n{cols_to_drop}"
        )
        self.df.drop(cols_to_drop, axis=1, inplace=True)

    def drop_columns_by_instance(self, instances):
        cols_to_drop = [
            col
            for col in self.df.columns
            if "-" in col and self.extract_instance_id(col) in instances
        ]
        self.logger.info(
            f"Dropping columns with {instances} instances: \n{cols_to_drop}"
        )
        self.df.drop(cols_to_drop, axis=1, inplace=True)

    def extract_field_id(self, column_name):
        """
        Extract field id from the coded column name
        :param column_name: e.g. 53-0.1
        :return: e.g. 53
        """
        return int(column_name.split("-")[0])

    def extract_instance_id(self, column_name):
        """
        Extract instance id from the coded column name
        :param column_name: e.g. 53-0.1
        :return: e.g 0
        """
        return int(column_name.split("-")[1].split(".")[0])

    def extract_array_id(self, column_name):
        """
        Extract array id from the coded column name
        :param column_name: e.g. 53-0.1
        :return: e.g 1
        """
        return int(column_name.split("-")[1].split(".")[1])

    def get_field_description(
        self, column_name, with_instance_id=False, with_array_id=False
    ):
        """
        Get field description by the coded column name, e.g. 53-0.1 -> 53 -> "Date of attending assessment centre"
        :param with_array_id: if True add array id to the field description -> "Date of attending assessment centre_array_1"
        :param column_name: 53-0.1
        :param with_instance_id: if True add instance id to the field description -> "Date of attending assessment centre_instance_0"
        :return:
        """
        description = self.data_showcase[
            self.data_showcase["FieldID"] == self.extract_field_id(column_name)
        ]["Field"].values[0]

        if with_instance_id:
            description += "_instance_" + f"{self.extract_instance_id(column_name)}"
        if with_array_id:
            description += "_array_" + f"{self.extract_array_id(column_name)}"
        return description

    def get_number_array_instances(self, column_name):
        # find the number of instances in the array e.g. 53-0.0 and 53-0.1 are two instances of the same array
        field_id = self.extract_field_id(column_name)
        instance_id = self.extract_instance_id(column_name)
        pattern = f"{field_id}-{instance_id}."
        number_array_instances = 0
        for col in self.df.columns:
            if pattern in col:
                number_array_instances += 1
        return number_array_instances

    def get_field_coding(self, field_id):
        return int(
            self.data_showcase[self.data_showcase["FieldID"] == field_id][
                "Coding"
            ].values[0]
        )

    def get_value_type(self, field_id):
        """
        Look at https://biobank.ctsu.ox.ac.uk/crystal/help.cgi?cd=value_type
        :param field_id:
        :return:
        """
        values = self.data_showcase[self.data_showcase["FieldID"] == field_id][
            "ValueType"
        ].values
        if len(values) == 0:
            return "UnknownType"
        return values[0]

    def get_value_type_by_description(self, description):
        values = self.data_showcase[self.data_showcase["Field"] == description][
            "ValueType"
        ].values
        if len(values) == 0:
            return "UnknownType"
        return values[0]

    def get_coding_id_by_description(self, description):
        field = self.data_showcase[self.data_showcase["Field"] == description]["Coding"]
        if field.empty:
            self.logger.error(
                f"Field {description} not found in the data showcase; Field: {field}"
            )
            return None
        return int(
            self.data_showcase[self.data_showcase["Field"] == description][
                "Coding"
            ].values[0]
        )

    def get_coding(self, coding_id):
        if not os.path.exists(f"{self.CODING_FOLDER}/{coding_id}.tsv"):
            os.system(
                f"wget {self.DOWNLOAD_LINK}{coding_id} -O {self.CODING_FOLDER}/{coding_id}.tsv"
            )
        coding = pd.read_csv(
            f"{self.CODING_FOLDER}/{coding_id}.tsv",
            sep="\t",
            dtype={"coding": int, "meaning": str},
        )
        coding.fillna("None", inplace=True)
        return coding

    def reverse_categorical_single(self, value, coding_df):
        """
        Reverse the categorical single value to the original value from the coding dataframe.
        But if the value is unknown, prefer not to answer or do not know, return -1
        :param value:
        :param coding_df:
        :return:
        """
        coding_dict = dict(zip(coding_df["meaning"], coding_df["coding"]))
        # lower case for all keys in the dictionary
        coding_dict = {k.lower(): v for k, v in coding_dict.items()}
        if (
            value == "unknown"
            or value == "prefer not to answer"
            or value == "do not know"
        ):
            return -1
        else:
            return coding_dict[value]

    @staticmethod
    def log_dataframe(df, logger):
        df_str = df.to_string()
        logger.info(df_str)
        # for line in df_str.split('\n'):
        # logger.info(line)

    def show_value_counts(self):
        for col in self.df.columns:
            value_type = self.get_value_type_by_description(col)
            if (
                value_type == "Categorical single"
                or value_type == "Categorical multiple"
            ):
                self.logger.info(
                    f"Column {col} \n value type: {value_type} \n {self.df[col].value_counts()}"
                )
                # self.logger.info(f"Column {col}")
                # self.logger.info(f"value type: {value_type}")
                # self.log_dataframe(self.df[col].value_counts(), self.logger)
            elif value_type == "Integer" or value_type == "Continuous":
                if "unknown" in self.df[col].value_counts():
                    number_unknown = self.df[col].value_counts()["unknown"]
                else:
                    number_unknown = 0
                self.logger.info(
                    f"Column {col} \n value type: {value_type} \n unknown values: {number_unknown}"
                )
                # self.logger.info(f"Column {col}")
                # self.logger.info(f"value type: {value_type}")
                # self.logger.info(f"unknown values: {number_unknown}")
            # self.logger.info("-"*15)

    def process_categorical_single(self, value, coding_df):
        coding_dict = dict(zip(coding_df["coding"], coding_df["meaning"]))
        if pd.isna(value):
            return "unknown"
        else:
            return coding_dict[int(value)].lower()

    def process_integer(self, value):
        if pd.isna(value) or value < 0:
            return "unknown"
        else:
            return int(value)

    def process_continuous(self, value):
        if pd.isna(value):
            return "unknown"
        else:
            return float(value)

    def process_field_value(self, value, column_name):
        field_id = self.extract_field_id(column_name)
        value_type = self.get_value_type(field_id)
        try:
            if value_type == "Categorical single":
                coding = self.get_coding(self.get_field_coding(field_id))
                return self.process_categorical_single(value, coding)
            elif value_type == "Categorical multiple":
                coding = self.get_coding(self.get_field_coding(field_id))
                return self.process_categorical_single(value, coding)
            elif value_type == "Integer":
                return self.process_integer(value)
            elif value_type == "Continuous":
                return self.process_continuous(value)
            else:
                self.logger.error(
                    f"Unknown value type {value_type} of the field {field_id} with value {value}"
                )
                return value
        except Exception as e:
            self.logger.error(
                f"Error processing field {field_id} and value {value} with type {type(value)} and value_type {value_type} and error {e}"
            )
            return value
            # raise ValueError(f"Unknown value type {self.get_value_type(field_id)}")

    def _has_unknown_word(self, category):
        for unknown_word in self.UNKNOWN_SET:
            if unknown_word in str(category):
                return True
        return False

    def unite_sparse_categories(self, cols, threshold=0.01):
        for col in cols:
            united_categories = []
            for category in self.df[col].unique():
                if self._has_unknown_word(category):
                    continue
                if self.df[col].value_counts()[category] / self.df.shape[0] < threshold:
                    united_categories.append(category)
            if len(united_categories) > 1:
                self.df[col] = self.df[col].apply(
                    lambda x: self.UNITED_CATEGORIES_SYMBOL
                    if x in united_categories
                    else x
                )
                self.logger.info(
                    f"Column '{col}' united categories: {united_categories} to {self.UNITED_CATEGORIES_SYMBOL}"
                )

    def encode_one_hot(self, cols):
        encoder = OneHotEncoder(handle_unknown="ignore")
        encoded_columns = encoder.fit_transform(self.df[cols]).toarray()
        encoded_df = pd.DataFrame(
            encoded_columns,
            columns=encoder.get_feature_names_out(),
            index=self.df.index,
        )
        self.df = pd.concat(
            [self.df.drop(cols, axis=1), encoded_df], axis=1, join="inner"
        )

    def encode_ordinal(self, cols):
        for col in cols:
            coding_df = self.get_coding(self.get_coding_id_by_description(col))
            self.df[col] = self.df[col].apply(
                lambda x: self.reverse_categorical_single(x, coding_df)
            )
        self.df[cols] = self.df[cols].astype(int)

    def get_cols_by_type(self, value_type, by="field_id"):
        if by == "field_id":
            return [
                col
                for col in self.df.columns
                if self.get_value_type(self.extract_field_id(col)) == value_type
            ]
        elif by == "description":
            return [
                col
                for col in self.df.columns
                if self.get_value_type_by_description(col) == value_type
            ]
        else:
            raise ValueError(f"Unknown by {by}")
