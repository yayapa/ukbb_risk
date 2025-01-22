import os
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from TwoDImage.image_classification.constants import Constants
import nibabel as nib
#from joblib import Memory
import torchio as tio
import torch.nn.functional as F


# Set up cache directory
#cache_dir = "/cache/"

#memory = Memory(cache_dir, verbose=0)


#@memory.cache
def load_image(img_path):
    return np.load(img_path)


#@memory.cache
def load_nifti_image(img_path):
    image_nifti = nib.load(img_path)
    return image_nifti.get_fdata()


#@memory.cache
def load_npz_image(img_path):
    npz = np.load(img_path)
    return npz["sax"], npz["lax"]


class UKBBDataset(Dataset):
    """
    Custom class for dataset.
    """

    def __init__(
        self,
        labels_csv_file,
        tabular_features_csv,
        img_dir,
        split="train",
        transform=None,
        input_modality="whole_body_projections",
        sax_num_slices=6,
        lax_num_slices=3,
    ):
        """
        Initialize the dataset.
        :param csv_file: path to csv file with eids and labels
        :param tabular_features_csv: path to csv file with tabular features
        :param img_dir: path to directory with images
        :param transform: transform to apply to images
        """
        df = pd.read_csv(labels_csv_file)
        print("len of df: ", len(df))
        if tabular_features_csv is not None:
            df_tabular_features = pd.read_csv(tabular_features_csv)
            print("len of df_tabular_features: ", len(df_tabular_features))
            # drop split, label, time_to_event columns if they exist in df_tabular_features
            if "split" in df_tabular_features.columns and "event" in df_tabular_features.columns and "time_to_event" in df_tabular_features.columns:
                df_tabular_features = df_tabular_features.drop(columns=["split", "event", "time_to_event"])

            df = df.merge(df_tabular_features, on="eid")
            print("len of df after merge: ", len(df))

        df = df[df["split"] == split].reset_index(drop=True)
        self.img_labels = df[["eid", "event", "time_to_event"]]
        print("len of img_labels: ", len(self.img_labels))

        # tabular features are all coumns except the ["event", "time_to_event", "split"]

        self.tabular_features = df.drop(
            columns=["eid", "event", "time_to_event", "split"]
        )
        self.img_dir = img_dir
        self.transform = transform
        self.classes = self.img_labels[Constants.TARGET_NAME.value].unique()
        self.classes.sort()
        self.input_modality = input_modality
        self.sax_num_slices = sax_num_slices
        self.lax_num_slices = lax_num_slices

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Get item from dataset.
        :param idx: index of item
        :return:  image, label, path and features
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Convert the NumPy array to a PyTorch tensor
        if self.input_modality == "whole_body_projections":
            image, img_path = self._get_image(idx)
        elif self.input_modality == "whole_body_3d":
            image, img_path = self._get_image_3d(idx)
        elif self.input_modality == "whole_body_3d_all":
            image, img_path = self._get_image_3d_all(idx)
        elif self.input_modality == "whole_body_3d_wat_fat":
            image, img_path = self._get_3d_image_wat_fat(idx)
        elif self.input_modality == "cardiac_4d" and not self.sax_num_slices:
            image, img_path = self._get_image_cardiac_4d(idx)
        elif self.input_modality == "liver_projections_all":
            image, img_path = self._get_image_liver_projections_all(idx)
        elif self.input_modality == "liver_t1_map" or self.input_modality == "pancreas_t1_map":
            image, img_path = self._get_liver_t1_map(idx)
        elif self.input_modality == "liver_pancreas_t1_map":
            image, img_path = self._get_liver_pancreas_t1_map(idx)
        elif self.input_modality == "tabular":
            image = torch.tensor(0)
            img_path = ""
        elif self.input_modality == "cardiac_4d" and self.sax_num_slices:
            image, img_path = self._get_image_cardiac_sliced(
                idx, self.sax_num_slices, self.lax_num_slices
            )
        else:
            raise ValueError("Invalid input_modality")

        event = self.img_labels.loc[idx, "event"]
        time_to_event = self.img_labels.loc[idx, "time_to_event"]
        tabular_vector = torch.tensor(
            self.tabular_features.loc[idx, self.tabular_features.columns != "eid"],
            dtype=torch.float32,
        )

        if self.transform and image is not None:
            if isinstance(self.img_dir, list):
                image = list(map(self.transform, image))
            else:
                image = self.transform(image)
        sample = {
            "image": image,
            "event": event,
            "time_to_event": time_to_event,
            "img_path": img_path,
            "tabular_vector": tabular_vector,
        }
        return sample

    def _get_image(self, idx):
        """
        Get image from path.
        :param img_path: path to image
        :return: image
        """
        img_name = str(self.img_labels.loc[idx, "eid"]) + ".npy"
        img_path = os.path.join(self.img_dir, img_name)
        image_array = load_image(img_path)

        # Convert the NumPy array to a PyTorch tensor
        image = torch.from_numpy(image_array)
        return image, img_path

    def _get_image_3d(self, idx):
        """
        Get image from path.
        :param idx:
        :return:
        """
        img_name = os.path.join(str(self.img_labels.loc[idx, "eid"]), "wat.nii.gz")
        img_path = os.path.join(self.img_dir, img_name)
        image_nifti = load_nifti_image(img_path)
        # image_nifti = image_nifti.get_fdata()
        image_nifti = np.expand_dims(image_nifti, axis=0)  # add channel dimension
        image = torch.from_numpy(image_nifti)
        return image, img_path

    def _get_image_3d_all(self, idx):
        """
        Get image from path.
        :param idx:
        :return:
        """
        img_name_wat = os.path.join(str(self.img_labels.loc[idx, "eid"]), "wat.nii.gz")
        img_path_wat = os.path.join(self.img_dir, img_name_wat)
        image_nifti_wat = load_nifti_image(img_path_wat)
        image_nifti_wat = np.expand_dims(image_nifti_wat, axis=0)  # add channel dimension

        img_name_fat = os.path.join(str(self.img_labels.loc[idx, "eid"]), "fat.nii.gz")
        img_path_fat = os.path.join(self.img_dir, img_name_fat)
        image_nifti_fat = load_nifti_image(img_path_fat)
        image_nifti_fat = np.expand_dims(image_nifti_fat, axis=0)  # add channel dimension

        img_name_inp = os.path.join(str(self.img_labels.loc[idx, "eid"]), "inp.nii.gz")
        img_path_inp = os.path.join(self.img_dir, img_name_inp)
        image_nifti_inp = load_nifti_image(img_path_inp)
        image_nifti_inp = np.expand_dims(image_nifti_inp, axis=0)

        img_name_opp = os.path.join(str(self.img_labels.loc[idx, "eid"]), "opp.nii.gz")
        img_path_opp = os.path.join(self.img_dir, img_name_opp)
        image_nifti_opp = load_nifti_image(img_path_opp)
        image_nifti_opp = np.expand_dims(image_nifti_opp, axis=0)

        #concactenate the images
        if image_nifti_inp.shape == image_nifti_opp.shape == image_nifti_fat.shape == image_nifti_wat.shape:
            image = np.concatenate((image_nifti_wat, image_nifti_fat, image_nifti_inp, image_nifti_opp), axis=0)
        else:
            transform = tio.transforms.CropOrPad((224, 168, 363))
            image_nifti_wat = transform(image_nifti_wat)
            image_nifti_fat = transform(image_nifti_fat)
            image_nifti_opp = transform(image_nifti_inp)
            image_nifti_inp = transform(image_nifti_inp)
            image = np.concatenate((image_nifti_wat, image_nifti_fat, image_nifti_inp, image_nifti_opp), axis=0)
            print("Image is cropped and padded", img_path_wat)
        image = torch.from_numpy(image).float()  # [4, 224, 168, 363]
        img_path = [img_path_wat, img_path_fat, img_path_inp, img_path_opp]

        # image_nifti = image_nifti.get_fdata()
        #image_nifti = np.expand_dims(image_nifti, axis=0)  # add channel dimension
        #image = torch.from_numpy(image_nifti)
        #print("image shape", image.shape)
        return image, img_path

    def _get_3d_image_wat_fat(self, idx):
        """
        Get image from path.
        :param idx:
        :return:
        """
        img_name_wat = os.path.join(str(self.img_labels.loc[idx, "eid"]), "wat.nii.gz")
        img_path_wat = os.path.join(self.img_dir, img_name_wat)
        image_nifti_wat = load_nifti_image(img_path_wat)
        image_nifti_wat = np.expand_dims(image_nifti_wat, axis=0)  # add channel dimension

        img_name_fat = os.path.join(str(self.img_labels.loc[idx, "eid"]), "fat.nii.gz")
        img_path_fat = os.path.join(self.img_dir, img_name_fat)
        image_nifti_fat = load_nifti_image(img_path_fat)
        image_nifti_fat = np.expand_dims(image_nifti_fat, axis=0)  # add channel dimension

        # concactenate the images
        if image_nifti_fat.shape == image_nifti_wat.shape:
            image = np.concatenate((image_nifti_wat, image_nifti_fat), axis=0)
        else:
            transform = tio.transforms.CropOrPad((224, 168, 363))
            image_nifti_wat = transform(image_nifti_wat)
            image_nifti_fat = transform(image_nifti_fat)
            image = np.concatenate((image_nifti_wat, image_nifti_fat), axis=0)
            print("Image is cropped and padded", img_path_wat)
        image = torch.from_numpy(image).float()  # [2, 224, 168, 363]
        img_path = [img_path_wat, img_path_fat]

        return image, img_path

    def _get_image_cardiac_4d(self, idx):
        """
        Get image from path.
        :param idx:
        :return:
        """
        img_path = os.path.join(
            self.img_dir,
            str(self.img_labels.loc[idx, "eid"]),
            "processed_seg_allax.npz",
        )
        # npz = np.load(img_path)
        # sax = npz["sax"]  # (128, 128, 11, 50)
        # lax = npz["lax"]  # (128, 128, 11, 50)
        sax, lax = load_npz_image(img_path)
        image = np.concatenate((sax, lax), axis=2)  # (128, 128, 14, 50)
        # Permute and add the channel dimension
        image = image.transpose(
            3, 2, 0, 1
        )  # Shape: (50, 14, 128, 128) -> time is a channel dimension
        image = torch.from_numpy(image).float()
        # print("image shape: ", image.shape)
        return image, img_path

    def _get_image_cardiac_sliced(self, idx, sax_num_slices=6, lax_num_slices=3):
        img_path = os.path.join(
            self.img_dir,
            str(self.img_labels.loc[idx, "eid"]),
            "processed_seg_allax.npz",
        )
        sax, lax = load_npz_image(img_path)
        sax = sax[:, :, :sax_num_slices, :]  # (128, 128, 6, 50)
        lax = lax[:, :, :lax_num_slices, :]  # (128, 128, 3, 50)
        sax = sax.transpose(2, 3, 0, 1)  # (128, 128, 6, 50) -> (6, 50, 128, 128)
        lax = lax.transpose(2, 3, 0, 1)  # (128, 128, 3, 50) -> (3, 50, 128, 128)
        sax = torch.from_numpy(sax).float()
        lax = torch.from_numpy(lax).float()

        if sax.shape[2] != 128 or sax.shape[3] != 128:
            print(img_path, "has shape: ", sax.shape)
            sax = F.interpolate(sax, size=(128, 128), mode="bilinear", align_corners=False)
        if lax.shape[2] != 128 or lax.shape[3] != 128:
            print(img_path, "has shape: ", lax.shape)
            lax = F.interpolate(lax, size=(128, 128), mode="bilinear", align_corners=False)

        # concatenate the slices
        image = torch.cat([sax, lax], dim=0)
        # print("cardiac sliced image shape: ", image.shape)
        image = image.permute(1, 0, 2, 3)  # (50, 9, 128, 128)

        return image, img_path

    def _get_image_liver_projections_all(self, idx):
        """
        Get image from path.
        :param img_path: path to image
        :return: image
        """
        img_name = str(self.img_labels.loc[idx, "eid"]) + ".npy"
        img_path = os.path.join(self.img_dir, img_name)
        image_array = np.load(img_path)

        # Convert the NumPy array to a PyTorch tensor
        image = torch.from_numpy(image_array)  # (384, 288, 1, 23)
        image = image.squeeze(2)  # (384, 288, 23)
        image = image.permute(2, 1, 0)  # (23, 288, 382)
        if image.size() != (23, 288, 384):
            print("image size: ", image.size())
            print("img_path: ", img_path)
        #assert image.size() == (23, 288, 384)
        return image, img_path

    def _get_liver_t1_map(self, idx, img_dir_idx=None):
        img_name = str(self.img_labels.loc[idx, "eid"]) + ".npy"
        if img_dir_idx is not None:
            img_path = os.path.join(self.img_dir[img_dir_idx], img_name)
        else:
            img_path = os.path.join(self.img_dir, img_name)
        image_array = np.load(img_path)

        # Convert the NumPy array to a PyTorch tensor
        image = torch.from_numpy(image_array)  # (384, 288, 1, 23)
        image = image.squeeze(2)  # (384, 288, 23)
        if image.size() == (384, 288, 23):
            image = image.permute(2, 1, 0)  # (23, 288, 384)
        elif image.size() == (288, 384, 23):
            image = image.permute(2, 0, 1)
        if image.size() != (23, 288, 384):
            print("image size if: ", image.size())
            print("img_path: ", img_path)
        image = image[14, :, :360] # 360 is an empirical value to remover scaler bar on the right
        image = image.unsqueeze(0)
        #assert image.size() == (1, 288, 384)
        assert image.size() == (1, 288, 360)
        return image, img_path

    def _get_liver_pancreas_t1_map(self, idx):
        image_1, img_path_1 = self._get_liver_t1_map(idx, img_dir_idx=0)
        image_2, img_path_2 = self._get_liver_t1_map(idx, img_dir_idx=1)

        return [image_1, image_2], [img_path_1, img_path_2]

