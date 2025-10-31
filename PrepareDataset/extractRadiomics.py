import datetime
import os
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import pandas as pd
from radiomics import featureextractor
from pathlib import Path
import argparse
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed

file_path_dir = "/path/to/total_segmentator"
file_path_original_dir = "/path/to/nifti"
file_path_save = "/path/to/radiomics"

num_processes = 127

labels = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "pancreas",
    8: "adrenal_gland_right",
    9: "adrenal_gland_left",
    10: "lung_upper_lobe_left",
    11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right",
    14: "lung_lower_lobe_right",
    15: "esophagus",
    16: "trachea",
    17: "thyroid_gland",
    18: "intestine",
    19: "duodenum",
    20: "unused",
    21: "urinary_bladder",
    22: "prostate",
    23: "sacrum",
    24: "heart",
    25: "aorta",
    26: "pulmonary_vein",
    27: "brachiocephalic_trunk",
    28: "subclavian_artery_right",
    29: "subclavian_artery_left",
    30: "common_carotid_artery_right",
    31: "common_carotid_artery_left",
    32: "brachiocephalic_vein_left",
    33: "brachiocephalic_vein_right",
    34: "atrial_appendage_left",
    35: "superior_vena_cava",
    36: "inferior_vena_cava",
    37: "portal_vein_and_splenic_vein",
    38: "iliac_artery_left",
    39: "iliac_artery_right",
    40: "iliac_vena_left",
    41: "iliac_vena_right",
    42: "humerus_left",
    43: "humerus_right",
    44: "scapula_left",
    45: "scapula_right",
    46: "clavicula_left",
    47: "clavicula_right",
    48: "femur_left",
    49: "femur_right",
    50: "hip_left",
    51: "hip_right",
    52: "spinal_cord",
    53: "gluteus_maximus_left",
    54: "gluteus_maximus_right",
    55: "gluteus_medius_left",
    56: "gluteus_medius_right",
    57: "gluteus_minimus_left",
    58: "gluteus_minimus_right",
    59: "autochthon_left",
    60: "autochthon_right",
    61: "iliopsoas_left",
    62: "iliopsoas_right",
    63: "sternum",
    64: "costal_cartilages",
    65: "subcutaneous_fat",
    66: "muscle",
    67: "inner_fat",
    68: "IVD",
    69: "vertebra_body",
    70: "vertebra_posterior_elements",
    71: "spinal_channel",
    72: "bone_other"
}

def find_files(dir, filename):
    matches = []
    for root, dirnames, filenames in os.walk(dir):
        for file in filenames:
            if file == filename:
                matches.append(os.path.join(root, file))
    return matches

def get_radiomics_features(img_file_path, seg_file_path, labels):
    img = nib.load(img_file_path)
    img_np = img.get_fdata()
    img_sitk = sitk.GetImageFromArray(img_np)
    #img_sitk.SetSpacing(img.header.get_zooms())  # Set original spacing from nibabel

    seg = nib.load(seg_file_path)
    seg_np = seg.get_fdata()
    seg_np = seg_np.astype(np.int16)
    #seg_sitk = sitk.GetImageFromArray(seg_np)
    #seg_sitk.SetSpacing(img.header.get_zooms())  # Set original spacing; not needed due to unified spacing

    features = {}
    settings = {
        #'binWidth': 5, mr example from pyradiomics, after discussion we decided to go with the default
        #'normalizeScale': 100,  # This allows you to use more or less the same bin width.
        "geometryTolerance": 1e-3,
        "normalize": True  # Enable intensity normalization
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("shape")
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("glcm")
    extractor.enableFeatureClassByName("gldm")
    extractor.enableFeatureClassByName("glrlm")
    extractor.enableFeatureClassByName("glszm")
    extractor.enableFeatureClassByName("ngtdm")

    for label_value, label_name in labels.items():
        try:
            binary_mask_np = (seg_np == label_value).astype(np.int16)
            binary_mask_sitk = sitk.GetImageFromArray(binary_mask_np)
            binary_mask_sitk.CopyInformation(img_sitk)

            print(f"Start extracting features for: {label_name}")
            lab_features = extractor.execute(img_sitk, binary_mask_sitk)

            # Add each feature to the features dictionary with descriptive naming
            for k, v in lab_features.items():
                if k.startswith("original_"):
                    feature_name = f"{label_name}_{k.replace('original_', '')}"
                    features[feature_name] = float(v)

        except Exception as e:
            features[f"{label_name}_exception"] = str(e)
            print(f"Error extracting features for {label_name}: {e}")

    try:
        del img, img_np, img_sitk, seg, seg_np, extractor
        del binary_mask_np, binary_mask_sitk, lab_features
    except:
        pass
    import gc
    gc.collect() # fixes parallelization problem

    return features


def process_eids(eids, labels):
    for eid in eids:
        wat_file = os.path.join(file_path_original_dir, str(eid), "wat.nii.gz")
        fat_file = os.path.join(file_path_original_dir, str(eid), "fat.nii.gz")
        seg_file = os.path.join(file_path_dir, str(eid), f"{str(eid)}_total_part-water_seg.nii.gz")

        if not (os.path.exists(seg_file) and os.path.exists(wat_file) and os.path.exists(fat_file)):
            print(f"Not enough files for eid: {eid}")
            print("Segmentation file: ", seg_file)
            print("Water file: ", wat_file)
            print("Fat file: ", fat_file)
            continue

        save_path = os.path.join(file_path_save, str(eid))
        save_file_wat = os.path.join(save_path, "radiomics_features_wat.csv")
        save_file_fat = os.path.join(save_path, "radiomics_features_fat.csv")

        Path(save_path).mkdir(parents=True, exist_ok=True)
        print(f"Processing eid: {eid}")

        try:
            # Calculate and save water features
            if os.path.exists(save_file_wat):
                print(f"Water features already calculated for eid: {eid}")
            else:
                features_wat = get_radiomics_features(wat_file, seg_file, labels)
                pd.DataFrame([features_wat]).to_csv(save_file_wat, index=False)
                print(f"Water features saved for eid: {eid}")

            if os.path.exists(save_file_fat):
                print(f"Fat features already calculated for eid: {eid}")
            else:
                # Calculate and save fat features
                features_fat = get_radiomics_features(fat_file, seg_file, labels)
                pd.DataFrame([features_fat]).to_csv(save_file_fat, index=False)
                print(f"Fat features saved for eid: {eid}")

        except Exception as e:
            print(f"Error processing eid {eid}: {e}")

def process_single_eid(eid):
    wat_file = os.path.join(file_path_original_dir, str(eid), "wat.nii.gz")
    fat_file = os.path.join(file_path_original_dir, str(eid), "fat.nii.gz")
    seg_file = os.path.join(file_path_dir, str(eid), f"{str(eid)}_total_part-water_seg.nii.gz")

    if not (os.path.exists(seg_file) and os.path.exists(wat_file) and os.path.exists(fat_file)):
        print(f"Not enough files for eid: {eid}")
        print("Segmentation file: ", seg_file)
        print("Water file: ", wat_file)
        print("Fat file: ", fat_file)
        return

    save_path = os.path.join(file_path_save, str(eid))
    save_file_wat = os.path.join(save_path, "radiomics_features_wat.csv")
    save_file_fat = os.path.join(save_path, "radiomics_features_fat.csv")

    Path(save_path).mkdir(parents=True, exist_ok=True)
    if os.path.exists(save_file_wat) and os.path.exists(save_file_fat):
        print(f"Features already calculated for eid: {eid}")
        return

    print(f"Processing eid: {eid}")

    try:
        # Calculate and save water features
        if os.path.exists(save_file_wat):
            print(f"Water features already calculated for eid: {eid}")
        else:
            # Calculate and save water features
            features_wat = get_radiomics_features(wat_file, seg_file, labels)
            pd.DataFrame([features_wat]).to_csv(save_file_wat, index=False)
            print(f"Water features saved for eid: {eid}")

        # Calculate and save fat features
        if os.path.exists(save_file_fat):
            print(f"Fat features already calculated for eid: {eid}")
        else:
            features_fat = get_radiomics_features(fat_file, seg_file, labels)
            pd.DataFrame([features_fat]).to_csv(save_file_fat, index=False)
            print(f"Fat features saved for eid: {eid}")

        print(f"Features saved for eid: {eid}")
        print("Current time: ", datetime.now())

    except Exception as e:
        print(f"Error processing eid {eid}: {e}")

def main():
    eids = pd.read_csv("/path/to/eids/eids_all.csv", usecols=["eid"])["eid"].tolist()
    process_eids(eids, labels)

def main_parallel():
    # make the args input for eids
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, default=0)
    args = parser.parse_args()

    eids = pd.read_csv(args.c, usecols=["eid"])["eid"].tolist()

    # Use multiprocessing to parallelize the task
    with Pool(num_processes) as pool:
        pool.map(process_single_eid, eids)


if __name__ == "__main__":
    #main()
    main_parallel()
