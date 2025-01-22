import json

from .CustomEncoder import CustomEncoder
from .utils import *

# ignore warnings
import warnings

warnings.filterwarnings("ignore")


class ICDExtractor:
    def __init__(
        self,
        source_file_path="../data/data/tabular/ukb668815_imaging.csv",
        icd_code_dict_file_path="icd_code_dict_imaging_dementia.json",
    ):
        self.source_file_path = source_file_path
        self.icd_code_dict_file_path = icd_code_dict_file_path

    def _get_icd_columns(self):
        df = pd.read_csv(self.source_file_path, nrows=1)

        self.icd_diagnosis_main, self.icd_diagnosis_date_main = get_icd_infos("main")
        self.icd_diagnosis_secondary, self.icd_diagnosis_date_secondary = get_icd_infos(
            "secondary"
        )
        icd_diagnosis_main_cols = [
            col for col in df.columns if col.startswith(self.icd_diagnosis_main)
        ]
        icd_diagnosis_secondary_cols = [
            col for col in df.columns if col.startswith(self.icd_diagnosis_secondary)
        ]
        icd_diagnosis_date_main_cols = [
            col for col in df.columns if col.startswith(self.icd_diagnosis_date_main)
        ]
        icd_diagnosis_date_secondary_cols = [
            col
            for col in df.columns
            if col.startswith(self.icd_diagnosis_date_secondary)
        ]

        self.icd_diagnosis_cancer, self.icd_diagnosis_date_cancer = get_icd_infos(
            "cancer"
        )
        icd_diagnosis_cancer_cols = [
            col for col in df.columns if col.startswith(self.icd_diagnosis_cancer)
        ]
        icd_diagnosis_date_cancer_cols = [
            col for col in df.columns if col.startswith(self.icd_diagnosis_date_cancer)
        ]

        self.icd_cols = (
            icd_diagnosis_main_cols
            + icd_diagnosis_secondary_cols
            + icd_diagnosis_cancer_cols
        )
        self.icd_date_cols = (
            icd_diagnosis_date_main_cols
            + icd_diagnosis_date_secondary_cols
            + icd_diagnosis_date_cancer_cols
        )
        self.visit_date_cols = get_visit_dates_fields()
        self.ado_date = {
            "asthma": "42014-0.0",
            "copd": "42016-0.0",
            "endstage_renal": "42026-0.0",
            "motor_neurone": "42028-0.0",
            "myoinfraction": "42000-0.0",
            "stroke": "42006-0.0",
            "dementia_allcause": "42018-0.0",
        }

        self.df_icd = pd.read_csv(
            self.source_file_path,
            usecols=["eid"]
            + self.icd_cols
            + self.icd_date_cols
            + self.visit_date_cols
            + list(self.ado_date.values()),
        )

    def extract_icd_codes(self):
        self._get_icd_columns()
        # Assuming eID is an integer
        icd_code_dict = {}
        for index, row in self.df_icd.iterrows():
            first_imaging_date = pd.to_datetime(row["53-2.0"])
            second_imaging_date = pd.to_datetime(row["53-3.0"])
            first_visiting_date = pd.to_datetime(row["53-0.0"])
            second_visiting_date = pd.to_datetime(row["53-1.0"])
            for i, col in enumerate(self.icd_cols):
                icd_code = row[col]
                if pd.isna(icd_code):
                    continue
                if col.startswith(self.icd_diagnosis_secondary):
                    icd_origin = "secondary"
                elif col.startswith(self.icd_diagnosis_main):
                    icd_origin = "main"
                elif col.startswith(self.icd_diagnosis_cancer):
                    icd_origin = "cancer"
                else:
                    icd_origin = "unknown"

                icd_date = pd.to_datetime(
                    row[self.icd_date_cols[i]], errors="coerce", format="%Y-%m-%d"
                )
                if icd_code_dict.get(row["eid"]) is None:
                    icd_code_dict[row["eid"]] = {
                        "icd_codes": [icd_code],
                        "icd_dates": [icd_date],
                        "icd_origin": [icd_origin],
                        "first_imaging_date": first_imaging_date,
                        "first_visiting_date": first_visiting_date,
                        "second_imaging_date": second_imaging_date,
                        "second_visiting_date": second_visiting_date,
                    }
                else:
                    icd_code_dict[row["eid"]]["icd_codes"].append(icd_code)
                    icd_code_dict[row["eid"]]["icd_dates"].append(icd_date)
                    icd_code_dict[row["eid"]]["icd_origin"].append(icd_origin)
            for key, value in self.ado_date.items():
                ado_date_value = pd.to_datetime(
                    row[value], errors="coerce", format="%Y-%m-%d"
                )
                if not pd.isna(ado_date_value):
                    if icd_code_dict.get(row["eid"]) is None:
                        icd_code_dict[row["eid"]] = {
                            "icd_codes": [key],
                            "icd_dates": [ado_date_value],
                            "icd_origin": ["ado"],
                            "first_imaging_date": first_imaging_date,
                            "first_visiting_date": first_visiting_date,
                            "second_imaging_date": second_imaging_date,
                            "second_visiting_date": second_visiting_date,
                        }
                    else:
                        icd_code_dict[row["eid"]]["icd_codes"].append(key)
                        icd_code_dict[row["eid"]]["icd_dates"].append(ado_date_value)
                        icd_code_dict[row["eid"]]["icd_origin"].append("ado")
        self.icd_code_dict = icd_code_dict

    def save_icd_code_dict(self, icd_code_dict, icd_code_dict_file_path=None):
        # Serialize to JSON using the custom encoder
        json_data = json.dumps(icd_code_dict, cls=CustomEncoder)

        # Save to file
        with open(icd_code_dict_file_path, "w") as file:
            file.write(json_data)

    @staticmethod
    def safe_convert_to_timestamp(item):
        try:
            if item is None:
                return pd.NaT
            return pd.to_datetime(item, errors="raise", format="%Y-%m-%dT%H:%M:%S")
        except ValueError:
            if not item[0].isalpha():
                print("Everything okay with item?", item)
            return item

    def load_icd_code_dict(self):
        with open(self.icd_code_dict_file_path, "r") as file:
            data_loaded = json.load(file)

        # Convert keys back to integers and handle timestamps
        self.icd_code_dict = {
            int(k): {
                sub_k: self.safe_convert_to_timestamp(sub_v)
                if not isinstance(sub_v, list)
                else [
                    self.safe_convert_to_timestamp(sub_v_item) for sub_v_item in sub_v
                ]
                for sub_k, sub_v in v.items()
            }
            for k, v in data_loaded.items()
        }

        return self.icd_code_dict

    def load_self_reported_icd_code_dict(self, icd_code_dict_file_path):
        with open(icd_code_dict_file_path, "r") as file:
            data_loaded = json.load(file)

        self_reported_dict = {
            int(k): [self.safe_convert_to_timestamp(d) for d in v]
            for k, v in data_loaded.items()
        }
        return self_reported_dict

    def get_all_before_date(self, eid, date):
        icd_codes = self.icd_code_dict[eid]["icd_codes"]
        icd_dates = self.icd_code_dict[eid]["icd_dates"]
        icd_origin = self.icd_code_dict[eid]["icd_origin"]
        icd_before_date = {
            "icd_codes": [],
            "icd_origin": [],
            "icd_dates": [],
            "first_imaging_date": self.icd_code_dict[eid]["first_imaging_date"],
            "first_visiting_date": self.icd_code_dict[eid]["first_visiting_date"],
            "second_imaging_date": self.icd_code_dict[eid]["second_imaging_date"],
            "second_visiting_date": self.icd_code_dict[eid]["second_visiting_date"],
        }
        for i, icd_date in enumerate(icd_dates):
            if pd.isna(icd_date):
                continue
            if icd_date < date:
                icd_before_date["icd_codes"].append(icd_codes[i])
                icd_before_date["icd_origin"].append(icd_origin[i])
                icd_before_date["icd_dates"].append(icd_date)
        return icd_before_date

    def get_all_after_date(self, eid, date):
        icd_codes = self.icd_code_dict[eid]["icd_codes"]
        icd_dates = self.icd_code_dict[eid]["icd_dates"]
        icd_origin = self.icd_code_dict[eid]["icd_origin"]
        icd_after_date = {
            "icd_codes": [],
            "icd_origin": [],
            "icd_dates": [],
            "first_imaging_date": self.icd_code_dict[eid]["first_imaging_date"],
            "first_visiting_date": self.icd_code_dict[eid]["first_visiting_date"],
            "second_imaging_date": self.icd_code_dict[eid]["second_imaging_date"],
            "second_visiting_date": self.icd_code_dict[eid]["second_visiting_date"],
        }
        for i, icd_date in enumerate(icd_dates):
            if pd.isna(icd_date):
                continue
            if icd_date > date:
                icd_after_date["icd_codes"].append(icd_codes[i])
                icd_after_date["icd_origin"].append(icd_origin[i])
                icd_after_date["icd_dates"].append(icd_date)
        return icd_after_date

    def save_all_with_eids(self, eids, file_path):
        chunk_size = 10000
        filtered_chunks = []
        i = 0
        for chunk in pd.read_csv(self.source_file_path, chunksize=chunk_size):
            print("Chunk: ", chunk_size * i)
            i += 1
            filtered_chunk = chunk[chunk["eid"].isin(eids)]
            filtered_chunks.append(filtered_chunk)
        filtered_data = pd.concat(filtered_chunks, ignore_index=True)
        filtered_data.to_csv(file_path, index=False)

    def get_main_infos(self, after):
        deltas = []
        eids = []
        for eid in self.icd_code_dict.keys():
            if (
                len(self.icd_code_dict[eid]["icd_dates"]) == 0
                or self.icd_code_dict[eid][after] is pd.NaT
                or pd.NaT in self.icd_code_dict[eid]["icd_dates"]
            ):
                continue
            diff = (
                min(self.icd_code_dict[eid]["icd_dates"])
                - self.icd_code_dict[eid][after]
            )
            if diff > pd.Timedelta(0):
                deltas.append(diff.days)
                eids.append(eid)
        mean = sum(deltas) / len(deltas)
        std = (sum((x - mean) ** 2 for x in deltas) / len(deltas)) ** 0.5
        n = len(deltas)
        print(f"mean: {mean}, std: {std}, n: {n}")
        return eids, deltas

    def extract_self_reported_cancer(self):
        df = pd.read_csv(self.source_file_path, nrows=1)
        self_reported_cancer_year_fields = [
            col for col in df.columns if col.startswith("20006")
        ]
        df = pd.read_csv(
            self.source_file_path, usecols=["eid"] + self_reported_cancer_year_fields
        )
        self_reported_dict = {}
        import calendar

        def float_year_to_timestamp(year_float):
            # Split the year and the fraction
            year = int(year_float)
            fraction = year_float - year

            # Calculate the number of days in the year
            days_in_year = 366 if calendar.isleap(year) else 365

            # Calculate the day of the year
            day_of_year = int(fraction * days_in_year)

            # Convert to pandas timestamp
            return pd.to_datetime(f"{year}-01-01") + pd.Timedelta(days=day_of_year)

        for i, row in df.iterrows():
            for field in self_reported_cancer_year_fields:
                if pd.notna(row[field]):
                    ts = float_year_to_timestamp(row[field])
                    if row["eid"] in self_reported_dict.keys():
                        self_reported_dict[int(row["eid"])].append(ts)
                    else:
                        self_reported_dict[int(row["eid"])] = [ts]
        return self_reported_dict

    def extract_icd_codes_from_pattern(self, patterns):
        icd_selector = []
        all_icds = pd.read_csv("../tabular/coding/coding19.tsv", sep="\t")[
            "coding"
        ].tolist()
        for icd in all_icds:
            for pattern in patterns:
                if icd.startswith(pattern):
                    icd_selector.append(icd)
        icd_selector = list(set(icd_selector))
        return icd_selector
