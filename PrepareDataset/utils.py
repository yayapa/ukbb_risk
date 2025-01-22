import os
import pandas as pd
from PrepareDataset.utils_analysis import *


def get_patterns(MODE):
    # ICD codes for the diseases of interest
    PATTERN_SELECTION_MODE = "in"
    if MODE == "cvd_jessica":
        icd_jessica = get_codes_dict()
        patterns = icd_jessica["8"]
    elif MODE == "cvd":
        icd_coronary_heart_disease = [
            "I250",
            "I251",
            "I253",
            "I254",
            "I255",
            "I256",
            "I258",
            "I259",
        ]
        icd_myocardial_infraction = [
            "I21",
            "I210",
            "I211",
            "I212",
            "I213",
            "I214",
            "I219",
            "I22",
            "I220",
            "I221",
            "I228",
            "I229",
            "I23",
            "I230",
            "I231",
            "I232",
            "I233",
            "I234",
            "I235",
            "I236",
            "I238",
            "I241",
            "I252",
        ]
        icd_heart_failure = ["I500", "I501", "I509"]

        icd_valvular_heart_disease = [
            "I062",
            "I060",
            "I352",
            "I350",
            "I061",
            "I351",
            "I050",
            "I052",
            "I342",
            "I051",
            "I340",
        ]

        icd_other_cardio = ["I63", "I64", "G45"]
        # combine all patterns
        patterns = (
            icd_coronary_heart_disease
            + icd_myocardial_infraction
            + icd_heart_failure
            + icd_valvular_heart_disease
            + icd_other_cardio
        )
    elif MODE == "age":
        cerebral_stroke = ["I63", "I65", "I66"]
        abdominal_ortic_neurysm = ["I71"]
        peripheral_artery_disease = [
            "I70",
            "I71",
            "I72",
            "I73",
            "I74",
            "I75",
            "I76",
            "I77",
            "I78",
            "I79",
        ]
        liver_disease = [
            "B15",
            "B16",
            "B17",
            "B18",
            "B19",
            "C22",
            "E83",
            "E88",
            "I85",
            "K70",
            "K72",
            "K73",
            "K74",
            "K75",
            "K76",
            "R18",
            "Z94",
        ]
        atherosclerotic_cardiovascular_disease = ["I25"]
        t2dm = ["E11"]
        metabolic_syndrome = [
            "E70",
            "E71",
            "E72",
            "E73",
            "E74",
            "E75",
            "E76",
            "E77",
            "E78",
            "E79",
            "E80",
            "E83",
            "E84",
            "E85",
            "E86",
            "E87",
            "E88",
            "E89",
            "E90",
        ]
        patterns = (
            cerebral_stroke
            + abdominal_ortic_neurysm
            + peripheral_artery_disease
            + liver_disease
            + atherosclerotic_cardiovascular_disease
            + t2dm
            + metabolic_syndrome
        )
    elif MODE == "liver":
        PATTERN_SELECTION_MODE = "startswith"
        # https://www.icd-code.de/icd/code/K70-K77.html

        patterns = [
            "K7",  # Krankheiten der Leber
            "B15",
            "B16",
            "B17",
            "B18",
            "B19",  # Virushepatitis
            "E83",  # Störungen des Mineralstoffwechsels
            "C22",  # Malignant Neoplasm of Liver and Intrahepatic Bile Ducts
            "R170",  # Gelbsucht
            "G937",  # Reye-Syndrom
        ]
    elif MODE == "cancer":
        PATTERN_SELECTION_MODE = "startswith"
        patterns = ["C{0:02d}".format(x) for x in range(0, 98)]  # all C00-C97
        patterns.remove("C44")  # without C44
        patterns += ["D{0:02d}".format(x) for x in range(37, 49)]  # all D37-D48
        # why not D?
        # 3. In Situ Neoplasms (D00-D09): These are localized neoplasms that have not invaded neighboring tissues or metastasized but have potential for becoming cancerous.
        # Benign Neoplasms (D10-D36): Generally, benign tumors are not included in cancer registries, but some benign tumors of the brain and other parts of the central nervous system are often tracked due to their potential to cause significant health problems.
        # Neoplasms of Uncertain or Unknown Behavior (D37-D48): These codes are sometimes included, especially when the neoplasm behaves aggressively or has the potential to become malignant.
    return patterns, PATTERN_SELECTION_MODE


def get_mri_infos(mri_type):
    if mri_type == "cardiac":
        file_path_mri = r"./data/data/cardiac/shmolli_data"  # cardiac
        mri_field, mri_instance = "20214", "2"  # cardiac
    elif mri_type == "whole-body":
        file_path_mri = r"./data/data/whole_body/original_zip_files/"
        mri_field, mri_instance = "20201", "2"  # whole-body
    elif mri_type == "brain":
        file_path_mri = r"./data/data/brain/compressed_data/"
        mri_field, mri_instance = "20252", "2"  # brain
    elif mri_type == "liver":
        file_path_mri = r"./data/data/abdominal/liver_data/ShMoLLI"
        mri_field, mri_instance = "20204", "2"  # liver
    return file_path_mri, mri_field, mri_instance


def get_icd_infos(icd_type):
    # in what fields to look for ICD codes
    if icd_type == "main":
        icd_diagnosis_field = "41202"  # main icds
        icd_diagnosis_date_field = "41262"  # main icd dates
    elif icd_type == "secondary":
        icd_diagnosis_field = "41270"  # all icds
        icd_diagnosis_date_field = "41280"  # all icd dates
    elif icd_type == "cancer":
        icd_diagnosis_field = "40006"
        icd_diagnosis_date_field = "40005"
    return icd_diagnosis_field, icd_diagnosis_date_field


def get_visit_dates_fields():
    visit_dates_fields = ["53-0.0", "53-1.0", "53-2.0", "53-3.0"]
    return visit_dates_fields


def get_eids_from_zips(folder, searched_field, searched_instance):
    files = os.listdir(folder)
    files_new = []
    for i in range(len(files)):
        splits = files[i].split("_")
        if len(splits) != 4:
            print(f"file {files[i]} does not match the pattern")
            continue
        eid = splits[0]
        field = splits[1]
        instance = splits[2]
        array = splits[3]

        # check if eid is numeric
        if (
            eid.isnumeric()
            and field == searched_field
            and instance == searched_instance
            and array.endswith("0.zip")
        ):
            files_new.append(eid)
        else:
            print(f"file {files[i]} does not match the pattern")
    return files_new


def get_data_dict(file_path_to_read, MODE, icd_type):
    patterns, PATTERN_SELECTION_MODE = get_patterns(MODE)
    icd_diagnosis_field, icd_diagnosis_date_field = get_icd_infos(icd_type)
    visit_dates_fields = get_visit_dates_fields()

    filtered_df = pd.read_csv(file_path_to_read, nrows=100)
    icd_code_cols = [
        col for col in filtered_df.columns if col.startswith(icd_diagnosis_field)
    ]
    icd_date_cols = [
        col for col in filtered_df.columns if col.startswith(icd_diagnosis_date_field)
    ]

    columns_to_read = ["eid"] + visit_dates_fields + icd_code_cols + icd_date_cols
    filtered_df = pd.read_csv(file_path_to_read, usecols=columns_to_read)

    for col in visit_dates_fields + icd_date_cols:
        filtered_df[col] = pd.to_datetime(filtered_df[col], errors="coerce")
    data_dict = {}
    for eid in filtered_df["eid"]:
        data_dict[eid] = {
            "icd_codes": [],
            "icd_dates": [],
            "first_visit_date": None,
            "second_visit_date": None,
            "first_imaging_visit_date": None,
            "second_imaging_visit_date": None,
        }

    for index, row in filtered_df.iterrows():
        for col in icd_code_cols:
            if pd.notna(row[icd_date_cols[icd_code_cols.index(col)]]):
                add_data = False
                if MODE == "cancer":
                    add_data = True
                else:
                    for pattern in patterns:
                        if (PATTERN_SELECTION_MODE == "in" and row[col] == pattern) or (
                            PATTERN_SELECTION_MODE == "startswith"
                            and row[col].startswith(pattern)
                        ):
                            add_data = True
                            break
                if add_data:
                    data_dict[row["eid"]]["icd_codes"].append(row[col])
                    data_dict[row["eid"]]["icd_dates"].append(
                        row[icd_date_cols[icd_code_cols.index(col)]]
                    )
        data_dict[row["eid"]]["first_visit_date"] = row["53-0.0"]
        data_dict[row["eid"]]["second_visit_date"] = row["53-1.0"]
        data_dict[row["eid"]]["first_imaging_visit_date"] = row["53-2.0"]
        data_dict[row["eid"]]["second_imaging_visit_date"] = row["53-3.0"]

    return data_dict


def get_main_infos(data_dict, after):
    deltas = []
    eids = []
    for eid in data_dict:
        if (
            len(data_dict[eid]["icd_dates"]) == 0
            or data_dict[eid][after] is pd.NaT
            or pd.NaT in data_dict[eid]["icd_dates"]
        ):
            continue
        diff = min(data_dict[eid]["icd_dates"]) - data_dict[eid][after]
        if diff > pd.Timedelta(0):
            deltas.append(diff.days)
            eids.append(eid)
    mean = sum(deltas) / len(deltas)
    std = (sum((x - mean) ** 2 for x in deltas) / len(deltas)) ** 0.5
    n = len(deltas)
    print(f"mean: {mean}, std: {std}, n: {n}")
    return eids, deltas


def get_cancer_types():
    neck = [
        "C{0:02d}".format(x) for x in range(0, 15)
    ]  # Malignant neoplasms of lip, oral cavity and pharynx
    digestive = [
        "C{0:02d}".format(x) for x in range(15, 27)
    ]  # Malignant neoplasms of digestive organs
    respiratory = [
        "C{0:02d}".format(x) for x in range(27, 40)
    ]  # Malignant neoplasms of respiratory and intrathoracic organs
    bone = [
        "C{0:02d}".format(x) for x in range(40, 43)
    ]  # Malignant neoplasms of bone and articular cartilage
    skin_wo_c44 = ["C43"]  # Melanoma and other malignant neoplasms of skin
    c44 = ["C44"]  # Other malignant neoplasms of skin
    soft_tissues = [
        "C{0:02d}".format(x) for x in range(45, 50)
    ]  # Malignant neoplasms of mesothelial and soft tissue
    breast = ["C{0:02d}".format(x) for x in range(50, 51)]  # Malignant neoplasm of
    female_genital = [
        "C{0:02d}".format(x) for x in range(51, 59)
    ]  # Malignant neoplasms of female genital organs5785
    male_genital = [
        "C{0:02d}".format(x) for x in range(59, 64)
    ]  # Malignant neoplasms of male genital organs16766
    urinary = [
        "C{0:02d}".format(x) for x in range(64, 69)
    ]  # Malignant neoplasms of urinary tract
    central_nervous = [
        "C{0:02d}".format(x) for x in range(69, 73)
    ]  # Malignant neoplasms of urinary tract
    endocrine = [
        "C{0:02d}".format(x) for x in range(73, 76)
    ]  # Malignant neoplasms of thyroid and other endocrine glands
    ill_defined = [
        "C{0:02d}".format(x) for x in range(76, 81)
    ]  # Malignant neoplasms of ill-defined, secondary and unspecified sites
    secondary = [
        "C{0:02d}".format(x) for x in range(81, 97)
    ]  # Malignant neoplasms, stated or presumed to be primary, of lymphoid, haematopoietic and related tissue
    independent = [
        "C{0:02d}".format(x) for x in range(97, 98)
    ]  # Malignant neoplasms of independent (primary) multiple sites

    neoplasm_of_uncertain_or_unknown_behavior = [
        "D{0:02d}".format(x) for x in range(37, 49)
    ]  # Neoplasms of uncertain or unknown behavior

    # name of variables as keys and list of ICD10 codes as values
    """
    cancer_types = {
        "head_neck": neck,
        "digestive": digestive,
        "respiratory": respiratory,
        "bone": bone,
        "skin_wo_c44": skin_wo_c44,
        "c44": c44,
        "mesothelial_soft_tissue": soft_tissues,
        "breast": breast,
        "female_genital": female_genital,
        "male_genital": male_genital,
        "urinary": urinary,
        "central_nervous": central_nervous,
        "endocrine": endocrine,
        "ill_defined_secondary": ill_defined,
        "lymphoid_haematopoietic": secondary,
        "independent_multiple_sites": independent,
        "neoplasm_of_uncertain_or_unknown_behavior": neoplasm_of_uncertain_or_unknown_behavior,
    }
    """
    cancer_types = {
        "Malignant neoplasms of lip, oral cavity and pharynx": neck,
        "Malignant neoplasms of digestive organs": digestive,
        "Malignant neoplasms of respiratory and intrathoracic organs": respiratory,
        "Malignant neoplasms of bone and articular cartilage": bone,
        "C43 Malignant melanoma of skin": skin_wo_c44,
        "C44 Other malignant neoplasms of skin": c44,
        "Malignant neoplasms of mesothelial and soft tissue": soft_tissues,
        "Malignant neoplasm of breast": breast,
        "Malignant neoplasms of female genital organs": female_genital,
        "Malignant neoplasms of male genital organs": male_genital,
        "Malignant neoplasms of urinary tract": urinary,
        "Malignant neoplasms of eye, brain and other parts of central nervous system": central_nervous,
        "Malignant neoplasms of thyroid and other endocrine glands": endocrine,
        "Malignant neoplasms of ill-defined, secondary and unspecified sites": ill_defined,
        "Malignant neoplasms, stated or presumed to be primary, of \n lymphoid, haematopoietic and related tissue": secondary,
        "Malignant neoplasms of independent (primary) multiple sites": independent,
        "Neoplasms of uncertain or unknown behaviour": neoplasm_of_uncertain_or_unknown_behavior,
    }
    return cancer_types
