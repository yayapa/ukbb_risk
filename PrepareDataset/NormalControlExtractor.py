import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class NormalControlExtractor:
    sex_field = "31"
    age_field = "21003"
    bmi_field = "21001"
    ethnicity = "21000"

    def __init__(
        self,
        source_file_path="../data/data/tabular/ukb668815_imaging.csv",
        interested_date="first_imaging_date",
        propensity_source_file_path=None,
    ):
        self.source_file_path = source_file_path
        self.propensity_source_file_path = propensity_source_file_path

        if interested_date == "first_imaging_date":
            self.interested_date_field = ["53-2.0"]
        elif interested_date == "first_visiting_date":
            self.interested_date_field = ["53-0.0"]
        elif interested_date == "second_imaging_date":
            self.interested_date_field = ["53-3.0"]
        elif interested_date == "second_visiting_date":
            self.interested_date_field = ["53-1.0"]
        else:
            raise ValueError("Invalid interested_date")

    def _find_death_register_eids(self):
        df = pd.read_csv(self.source_file_path, nrows=1)
        death_date_cols = [col for col in df.columns if col.startswith("40000")]
        death_icd_code_cols = [col for col in df.columns if col.startswith("40001")]
        death_cols = death_icd_code_cols + death_date_cols
        df_death = pd.read_csv(self.source_file_path, usecols=["eid"] + death_cols)
        eids_death = []
        for index, row in df_death.iterrows():
            for col in death_cols:
                if pd.notna(row[col]) or (
                    type(row[col]) == str and row[col].lower() != "nan"
                ):
                    eids_death.append(row["eid"])
        self.eids_death = eids_death
        print("Number of eids in death register: ", len(eids_death))

    def _find_eids_by_censoring(self):
        df_censoring = pd.read_csv(
            self.source_file_path, usecols=["eid"] + self.interested_date_field
        )
        eids_lessNyears = []
        for index, row in df_censoring.iterrows():
            first_imaging_date = pd.to_datetime(
                row[self.interested_date_field[0]], errors="coerce", format="%Y-%m-%d"
            )
            diff = self.censoring_date - first_imaging_date
            if diff < pd.Timedelta(365 * self.n_years, "D"):
                eids_lessNyears.append(row["eid"])
        self.eids_censoring = eids_lessNyears
        print(
            "Number of eids less than ", self.n_years, " years: ", len(eids_lessNyears)
        )

    def set_eids_positive(self, eids_positive):
        self.eids_positive = eids_positive

    def set_eids_positive_cohort(self, eids_positive_cohort):
        self.eids_positive_cohort = eids_positive_cohort

    def set_eids_pool(self, eids_pool):
        self.eids_pool = eids_pool

    def set_propensity_attributes(self, propensity_attributes):
        self.propensity_attributes = propensity_attributes

    def set_cencoring_date(self, n_years, censoring_date):
        self.censoring_date = censoring_date
        self.n_years = n_years

    def _get_df_by_eids(self, eids_selected):
        i = 0
        chunk_size = 10000
        filtered_chunks = []
        columns_to_read = ["eid"] + self.propensity_attributes
        for chunk in pd.read_csv(
            self.propensity_source_file_path,
            chunksize=chunk_size,
            usecols=columns_to_read,
        ):
            print("Chunk: ", chunk_size * i)
            i += 1
            # Filter the chunk
            filtered_chunk = chunk[chunk["eid"].isin(eids_selected)]

            # Option 1: Append the filtered chunk to the list (to concatenate later)
            filtered_chunks.append(filtered_chunk)

        df_control = pd.concat(filtered_chunks, ignore_index=True)
        return df_control

    def extract(self):
        self._find_death_register_eids()
        self._find_eids_by_censoring()

        eids_selected = list(
            set(self.eids_pool)
            - set(self.eids_death + self.eids_censoring + self.eids_positive)
        )

        pd.DataFrame(eids_selected).to_csv(
            "eids_selected_normal_control.csv", index=False
        )

        print("Number of eids selected for normal control: ", len(eids_selected))
        print("Number of eids in positive cohort: ", len(self.eids_positive_cohort))
        print(
            "Intersection of eids of normal and positive cohort",
            len(set(eids_selected).intersection(set(self.eids_positive_cohort))),
        )

        matched_dataset = self.extract_normal_control(eids_selected)
        return matched_dataset

    def extract_normal_control(self, eids_selected, knn_mode=False):
        control_group = self._get_df_by_eids(eids_selected)
        control_group["status"] = 0
        self._preprocess_propensity_attributes(control_group)

        positive_group = self._get_df_by_eids(self.eids_positive_cohort)
        positive_group["status"] = 1
        self._preprocess_propensity_attributes(positive_group)
        # positive_group.fillna(positive_group.mean(), inplace=True)

        df = pd.concat([positive_group, control_group], ignore_index=True)
        df = df.dropna()

        # Separate the features and the target variable
        X = df[self.propensity_attributes]  # Features

        y = df["status"]  # Target variable

        # Create a logistic regression model
        model = LogisticRegression()
        model.fit(X, y)

        # Calculate propensity scores (probability of being a cancer case)
        df["propensity_score"] = model.predict_proba(X)[:, 1]

        positive_group = df[df["status"] == 1]
        control_group = df[df["status"] == 0]

        if knn_mode:
            # Using NearestNeighbors to find matches
            nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(
                control_group[["propensity_score"]]
            )
            matched_controls = pd.DataFrame()

            for index, row in positive_group.iterrows():
                distances, indices = nbrs.kneighbors([row[["propensity_score"]]])
                # Get the index of the matched control
                control_index = control_group.iloc[indices[0]].index
                # Append the matched control to the matched_controls dataframe
                matched_controls = matched_controls.append(
                    control_group.loc[control_index]
                )
                # Drop the matched control to prevent re-use
                control_group = control_group.drop(control_index)
        else:
            # Calculate pairwise distances between cancer cases and controls based on propensity scores
            distances = pairwise_distances(
                positive_group[["propensity_score"]],
                control_group[["propensity_score"]],
                metric="euclidean",
            )

            # Initialize an empty list to store the indices of matched controls
            matched_indices = []

            for i in range(distances.shape[0]):
                # For each cancer case, find the index of the closest control
                closest_control_index = np.argmin(distances[i])

                # Check if this control has already been used
                while closest_control_index in matched_indices:
                    distances[i][closest_control_index] = np.inf
                    closest_control_index = np.argmin(distances[i])

                # Add the index to the list of matched indices
                matched_indices.append(closest_control_index)

            # Create the matched control group
            matched_controls = control_group.iloc[matched_indices]
        matched_dataset = pd.concat([positive_group, matched_controls])
        return matched_dataset

    def _preprocess_propensity_attributes(self, group):
        for attr in self.propensity_attributes:
            if attr.startswith(self.bmi_field):
                group[attr].fillna(group[attr].mean(), inplace=True)
                # scale the BMI
                group[attr] = StandardScaler().fit_transform(
                    group[attr].values.reshape(-1, 1)
                )
            elif attr.startswith(self.age_field):
                group[attr].fillna(group[attr].mean(), inplace=True)
            elif attr.startswith(self.ethnicity):
                # one-hot encoding of ethnicity field
                group = pd.get_dummies(group, columns=[attr])
            elif attr.startswith(self.sex_field):
                # most frequent value
                group[attr].fillna(group[attr].mode()[0], inplace=True)
        # drop nan values
        group.dropna(inplace=True)

    def visualize_sex_distribution(self, group):
        sex_field = ["31-0.0"]
        # ethnicity
        import seaborn as sns
        import matplotlib.pyplot as plt

        def decode_sex(lst):
            return list(map(lambda x: "female" if x == 0 else "male", lst))

        value_counts = group[sex_field].value_counts()
        keys = decode_sex(value_counts.keys().tolist())
        values = value_counts.values.tolist()

        sns.set_style("whitegrid")
        plt.figure(figsize=(6, 6))
        plt.pie(values, labels=keys, autopct="%1.1f%%")
        plt.title("Sex distribution in the dataset")
        plt.show()

    def visualize_hist(self, group, field, bins=10):
        group[field].hist(bins=bins)
        group[field].describe()
