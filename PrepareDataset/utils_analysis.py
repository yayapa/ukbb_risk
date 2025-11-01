import pandas as pd
import sys
import os
import numpy as np

from PrepareDataset.utils import *

current_dir = "put_yours"

age = ["21003-2.0"]
sex = ["31-0.0"]
bmi = ["21001-2.0"]
# ethnicity = ['21000-2.0']
ethnicity = ["21000-0.0"]
# assessment_center = ['54-2.' + str(a) for a in range(6)] # Not available
education = ["6138-2." + str(a) for a in range(6)]  # todo: make into 3 classes
# deprivation = ['22189-0.0'] # Not available
diabetes = ["2443-2.0"]  # self-reported diabetes

smoking = ["20116-2.0"]
alcohol = ["1558-2.0"]
physical_activity = ["22032-2.0"]

fresh_vegetable_intake = ["1299-2.0"]
fresh_fruit_intake = ["1309-2.0"]
# processed_meat_intake = ['1349-2.0'] # not available

# todo: unite cancer types
# illness_father = ['20107-2.0']  # not available
# illness_mother = ['20110-2.0']# not available
# illness_sibling = ['20111-2.0']# not available

time_spent_outdoor_summer = ["1050-2.0"]
sun_burn_childhood = ["1737-2.0"]
ease_of_skin_tanning = ["1727-2.0"]
skin_color = ["1717-2.0"]
hair_color = ["1747-2.0"]
use_of_sunscreen = ["2267-2.0"]
solarium_use = ["2277-2.0"]


def get_coding_dict():
    coding = pd.read_excel(
        os.path.join(current_dir, "coding/ICD10 Codes.xlsx"),
        sheet_name="description",
        header=None,
    )
    # add new columns for df from coding[1] and fill with 0
    coding_dict = dict(zip(coding[0].astype(str), coding[1]))
    return coding_dict


def get_codes_dict():
    codes = pd.read_excel(
        os.path.join(current_dir, "./coding/ICD10 Codes.xlsx"), sheet_name="ICD10 Codes"
    )
    # make code_dict with primary_score as key and coding as value
    codes_dict = (
        codes[["primary_score", "coding"]]
        .dropna()
        .groupby("primary_score")["coding"]
        .apply(list)
        .to_dict()
    )
    # make keys as string
    codes_dict = {str(k): v for k, v in codes_dict.items()}
    return codes_dict


def extract_disease_history(df_tabular, df):
    coding_dict = get_coding_dict()

    for i in coding_dict.values():
        df_tabular[i] = 0

    codes_dict = get_codes_dict()

    # in what fields to look for ICD codes
    icd_diagnosis_main, icd_diagnosis_date_main = get_icd_infos("main")
    icd_diagnosis_all, icd_diagnosis_date_all = get_icd_infos("secondary")
    icd_diagnosis_cancer, icd_diagnosis_date_cancer = get_icd_infos("cancer")
    visit_dates_fields = get_visit_dates_fields()

    icd_diagnosis_main_cols = [
        col for col in df.columns if col.startswith(icd_diagnosis_main)
    ]
    icd_diagnosis_all_cols = [
        col for col in df.columns if col.startswith(icd_diagnosis_all)
    ]
    icd_diagnosis_cancer_cols = [
        col for col in df.columns if col.startswith(icd_diagnosis_cancer)
    ]
    icd_cols = (
        icd_diagnosis_main_cols + icd_diagnosis_all_cols + icd_diagnosis_cancer_cols
    )
    icd_diagnosis_date_main_cols = [
        col for col in df.columns if col.startswith(icd_diagnosis_date_main)
    ]
    icd_diagnosis_date_all_cols = [
        col for col in df.columns if col.startswith(icd_diagnosis_date_all)
    ]
    icd_diagnosis_date_cancer_cols = [
        col for col in df.columns if col.startswith(icd_diagnosis_date_cancer)
    ]
    icd_date_cols = (
        icd_diagnosis_date_main_cols
        + icd_diagnosis_date_all_cols
        + icd_diagnosis_date_cancer_cols
    )
    df_icd = df[["eid"] + icd_cols + icd_date_cols + ["53-2.0"]]

    for index, row in df_icd.iterrows():
        first_imaging_date = pd.to_datetime(row["53-2.0"])
        for i, col in enumerate(icd_cols):
            for primary_score, icd_codes in codes_dict.items():
                if row[col] in icd_codes:
                    icd_date = pd.to_datetime(
                        row[icd_date_cols[i]], errors="coerce", format="%Y-%m-%d"
                    )
                    if icd_date is not pd.NaT and icd_date < first_imaging_date:
                        column_name = coding_dict[primary_score]
                        eid_index = df_tabular[df_tabular["eid"] == row["eid"]].index
                        df_tabular.loc[eid_index, column_name] = 1
                        # print(row)
    return df_tabular


def process_ethnicity(row):
    if str(row[ethnicity].iloc[0]).startswith("1"):
        return "white"
    elif str(row[ethnicity].iloc[0]).startswith("3"):
        return "asian"
    elif str(row[ethnicity].iloc[0]).startswith("4"):
        return "black"
    elif pd.isna(row[ethnicity].iloc[0]):
        return "unknown"
    else:
        return "others"


def process_education(row):
    # any row education has 1.0 value is considered as 'high'
    if any(row[education] == 1.0):
        return "high"
    elif any(
        np.logical_or.reduce(
            [
                row[education] == 2.0,
                row[education] == 3.0,
                row[education] == 4.0,
                row[education] == 5.0,
                row[education] == 6.0,
            ]
        )
    ):
        return "middle"
    elif any(row[education] == -7.0):
        return "low"
    else:
        return "unknown"
    # any row education has 2.0  value is considered as 'middle'


def process_diabetes(row):
    if row[diabetes].iloc[0] == 1.0:
        return "yes"
    elif row[diabetes].iloc[0] == 0.0:
        return "no"
    else:
        return "unknown"


def process_coding(row, column, coding_file_path):
    coding_df = pd.read_csv(coding_file_path, sep="\t")
    coding_dict = dict(zip(coding_df["coding"], coding_df["meaning"]))
    if pd.isna(row[column].iloc[0]) or row[column].iloc[0] < 0:
        return "unknown"
    else:
        return coding_dict[int(row[column].iloc[0])].lower()


def process_threshold(row, column, threshold):
    if row[column].iloc[0] >= threshold:
        return ">=" + str(threshold) + "times"
    elif row[column].iloc[0] < 0:
        return "unknown"
    else:
        return "<" + str(threshold) + "times"


def process_time_spent_outdoor_summer(row):
    if row[time_spent_outdoor_summer].iloc[0] > 5.0:
        return ">5h/day"
    elif 3.0 <= row[time_spent_outdoor_summer].iloc[0] < 5.0:
        return "3-5h/day"
    elif 1.0 <= row[time_spent_outdoor_summer].iloc[0] < 2.0:
        return "1-2h/day"
    elif row[time_spent_outdoor_summer].iloc[0] < 1.0:
        return "<1h/day"
    else:
        return "unknown"


def process_sun_burn_childhood(row):
    if row[sun_burn_childhood].iloc[0] > 0.0:
        return "yes"
    elif row[sun_burn_childhood].iloc[0] == 0.0:
        return "no"
    else:
        return "unknown"


def process_solarium_use(row):
    if row[solarium_use].iloc[0] > 0.0 or row[solarium_use].iloc[0] == -10.0:
        return "yes"
    elif row[solarium_use].iloc[0] == 0.0:
        return "no"
    else:
        return "unknown"
