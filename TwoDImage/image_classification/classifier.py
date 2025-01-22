from __future__ import print_function, division

import shutil

import psutil
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.backends import cudnn
#from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchmetrics
import matplotlib.pyplot as plt
import time
import os
import copy
import numpy as np
import pandas as pd
from datetime import datetime
import json
from torch.autograd import Variable

from TwoDImage.image_classification.constants import Constants
from TwoDImage.image_classification.ukbb_dataset import UKBBDataset
from TwoDImage.image_classification.early_stop import EarlyStopping
import random
import gc
import wandb
import torchio as tio


import warnings

warnings.filterwarnings("ignore")

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("DEVICE", DEVICE)
#cudnn.benchmark = True




class ImagewiseNormalize(object):
    def __call__(self, tensor):
        # Calculate the mean and std for each image
        mean = tensor.mean([1, 2], keepdim=True)
        std = tensor.std([1, 2], keepdim=True)

        # Perform the normalization
        tensor = (tensor - mean) / (std + 1e-8)  # Adding epsilon to avoid division by zero
        return tensor



class CardiacAugmentation(object):
    def __init__(self, p=0.5, degrees=180, test=False):
        self.p = p
        self.degrees = degrees
        self.test = test
    def image_normalization(self, image, scale=1, mode="2D"):
        if isinstance(image, np.ndarray) and np.iscomplexobj(image):
            image = np.abs(image)
        low = image.min()
        high = image.max()
        im_ = (image - low) / (high - low)
        if scale is not None:
            im_ = im_ * scale
        return im_

    def image_normalization_torch(self, image, scale=1, mode="3D"):
        if mode == "3D":
            max_3d = image.abs().max()
            min_3d = image.abs().min()
            image = (image - min_3d) / (max_3d - min_3d) * scale
        else:
            raise NotImplementedError
        return image

    def __call__(self, im):
        im = self.image_normalization_torch(im)
        if self.test:
            return im
        #print("im before", im.shape)
        transforms_ = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=self.p),
                transforms.RandomRotation(degrees=self.degrees),
            ]
        )
        im = transforms_(im)
        #print("im before autocontrast", im.shape)
        contrast = transforms.RandomAutocontrast(p=0.5)
        im = contrast(im.unsqueeze(-3))  #[50, 9, 1, 128, 128] -> [50, 9, 128, 128]
        #print("im after autocontrast", im.shape)
        im = im.squeeze(-3)  # Remove channel dimension
        #print("im after squeeze", im.shape)
        return im



class WholeBodyAugmentation(object):
    def __init__(self, p=0.5, degrees=180):
        self.p = p

    def __call__(self, im):
        random_value = np.random.rand()
        transform = tio.transforms.Compose(
            [
                tio.ZNormalization(), 
                tio.transforms.CropOrPad((224, 168, 363)),
                tio.transforms.RandomFlip(axes=0, p=self.p),
                tio.transforms.RandomFlip(axes=1, p=self.p),
                tio.transforms.RandomFlip(axes=2, p=self.p),
                tio.transforms.RandomBlur(p=0.5, std=np.min([random_value, 0.5])),
                tio.transforms.RandomNoise(p=0.5, std=np.min([random_value, 0.5])),
            ]
        )
        im = transform(im)
        return im


class RandomMaskBoxes(tio.Transform):
    def __init__(self, range_box_size, nr_boxes, p=0.5):
        super().__init__()
        self.range_box_size = range_box_size
        self.nr_boxes = nr_boxes
        self.p = p

    def apply_transform(self, subject):
        if np.random.rand() < self.p:
            for image_name, image in subject.get_images_dict(intensity_only=True).items():
                # Access image tensor and convert to numpy array
                data = image.numpy()
                # Apply the mask_boxes function
                data = self.mask_boxes(data)
                # Replace image data with masked image
                image.set_data(torch.from_numpy(data))
        return subject

    def mask_boxes(self, image):
        for _ in range(self.nr_boxes):
            z = np.random.randint(0, image.shape[0])
            y = np.random.randint(0, image.shape[1])
            x = np.random.randint(0, image.shape[2])
            z_size = np.random.randint(self.range_box_size[0], self.range_box_size[1])
            y_size = np.random.randint(self.range_box_size[0], self.range_box_size[1])
            x_size = np.random.randint(self.range_box_size[0], self.range_box_size[1])

            z_end = min(z + z_size, image.shape[0])
            y_end = min(y + y_size, image.shape[1])
            x_end = min(x + x_size, image.shape[2])

            # Set the box area to zero (mask it)
            image[z:z_end, y:y_end, x:x_end] = 0

        return image

class EventClassifier:
    """
    Class for training classification model with transfer learning and make predictions.
    """

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.result = {}

        if self.config["save_results"]:
            if config:
                self.result.update(config)
        else:
            self.experiment_dir = Constants.STORE_DIR.value

        self._create_data_transforms()
        self._create_wholebody_image_datasets_val()
        self._create_dataloaders_val()

    def set_device_by_id(self, gpu_id):
        self.DEVICE = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
        cudnn.benchmark = True
        print("DEVICE", self.DEVICE)

    def set_device(self):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cudnn.benchmark = True
        print("DEVICE", self.DEVICE)

    def save_checkpoint(self, state, checkpoint_dir, filename="checkpoint.pth.tar"):
        if not self.config.get("save_results", True):
            return
        filepath = os.path.join(checkpoint_dir, filename)
        torch.save(state, filepath)

    def load_checkpoint(self, checkpoint_dir, model, optimizer):
        checkpoint_path = os.path.join(
            checkpoint_dir, "model", "last_checkpoint.pth.tar"
        )
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            epoch = checkpoint["epoch"]
            best_balanced_acc = checkpoint["best_balanced_acc"]
            best_acc = checkpoint["best_acc"]
            best_f1 = checkpoint["best_f1"]
            best_auroc = checkpoint["best_auroc"]
            y_loss = checkpoint.get("y_loss", {"train": [], "val": []})
            y_balanced_acc = checkpoint.get("y_balanced_acc", {"train": [], "val": []})
            y_acc = checkpoint.get("y_acc", {"train": [], "val": []})
            y_f1 = checkpoint.get("y_f1", {"train": [], "val": []})
            y_auroc = checkpoint.get("y_auroc", {"train": [], "val": []})
            best_model_wts = checkpoint["best_model_wts"]
            best_epoch_number = checkpoint["best_epoch_number"]
            self.logger.info(
                f"Loaded checkpoint '{checkpoint_path}' (epoch {epoch}) with best balanced acc {best_balanced_acc} best acc {best_acc:.4f} and best f1 {best_f1:.4f} and best auroc {best_auroc:.4f} and best epoch number {best_epoch_number}"
            )
            return (
                epoch,
                best_balanced_acc,
                best_acc,
                best_f1,
                best_auroc,
                y_loss,
                y_balanced_acc,
                y_acc,
                y_f1,
                y_auroc,
                best_model_wts,
                best_epoch_number,
            )
        else:
            self.logger.info(f"No checkpoint found at '{checkpoint_path}'")
            return (
                0,
                0,
                0,
                0,
                0,
                {"train": [], "val": []},
                {"train": [], "val": []},
                {"train": [], "val": []},
                {"train": [], "val": []},
                {"train": [], "val": []},
                None,
                0,
            )


    def _create_data_transforms(self):
        """
        Create default data transformation if no given in constructor
        """
        if self.config["input_modality"] == "tabular":
            self.data_transforms = {"train": None, "test": None, "val": None}
            return

        if self.config["input_modality"] == "cardiac_4d":
            random_value = np.random.rand()
            self.data_transforms = {
                "train": transforms.Compose([
                    tio.ZNormalization(),
                    tio.transforms.RandomFlip(axes=2, p=0.5),
                    #tio.transforms.RandomFlip(axes=3, p=0.5),
                    tio.transforms.RandomBlur(p=0.5, std=np.min([random_value, 0.5])),
                    tio.transforms.RandomNoise(p=0.5, std=np.min([random_value, 0.5])),
                ]),
                "test": transforms.Compose([
                    tio.ZNormalization()
                ]),
                "val": transforms.Compose([
                    tio.ZNormalization()
                ])
            }
            #self.data_transforms = {
            #    "train": transforms.Compose([CardiacAugmentation(test=False)]),
            #    "test": transforms.Compose([CardiacAugmentation(test=True)]),
            #    "val": transforms.Compose([CardiacAugmentation(test=True)]),
            #}
            return

        if self.config["input_modality"] == "whole_body_3d" or self.config["input_modality"] == "whole_body_3d_all" or self.config["input_modality"] == "whole_body_3d_wat_fat":
            random_value = np.random.rand()
            self.data_transforms = {
                "train": tio.transforms.Compose(
                [
                    tio.ZNormalization(), 
                    tio.transforms.CropOrPad((224, 168, 363)),
                    tio.transforms.RandomFlip(axes=0, p=0.5),
                    tio.transforms.RandomFlip(axes=1, p=0.5),
                    tio.transforms.RandomFlip(axes=2, p=0.5),
                    tio.transforms.RandomBlur(p=0.5, std=np.min([random_value, 0.5])),
                    tio.transforms.RandomNoise(p=0.5, std=np.min([random_value, 0.5])),
                    #RandomMaskBoxes(range_box_size=(20, 50), nr_boxes=10, p=0.5),
                ]),
                "test": tio.transforms.Compose([tio.transforms.Resize((224, 168, 363))]),
                "val": tio.transforms.Compose([tio.transforms.Resize((224, 168, 363))])
            }
            print("transform", self.data_transforms)
            return

        if (self.config["input_modality"] == "liver_projections" or
                self.config["input_modality"] == "liver_projections_all" or
                self.config["input_modality"] == "liver_t1_map" or
                self.config["input_modality"] == "pancreas_t1_map" or
                self.config["input_modality"] == "liver_pancreas_t1_map"):
            self.data_transforms = {
                "train": transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(degrees=90),
                    ImagewiseNormalize()
                            ]),
                "test": transforms.Compose([ImagewiseNormalize()]),
                "val": transforms.Compose([ImagewiseNormalize()])
            }
            return

        if self.config["input_modality"] == "whole_body_projections":
            self.data_transforms = {
                "train": transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(degrees=90),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ]),
                "test": transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
                "val": transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

            }
            return

    def _create_wholebody_image_datasets(self):
        """
        Create train and test datasets
        """
        if self.config["tabular_features_file"] is None:
            tabular_features_file = None
        else:
            tabular_features_file = Constants.PROJECT_DIR.value + self.config["tabular_features_file"]
        self.image_datasets = {
            x: UKBBDataset(
                Constants.PROJECT_DIR.value + self.config["labels_file"],
                tabular_features_file,
                self.config["data_dir"],
                x,
                self.data_transforms[x],
                self.config["input_modality"],
                self.config.get("sax_num_slices", None),
                self.config.get("lax_num_slices", None),
            )
            for i, x in enumerate(["train", "test"])
        }
        self.class_names = self.image_datasets["train"].classes

    def _create_wholebody_image_datasets_val(self):
        """
        Create train and test datasets
        """
        if self.config["tabular_features_file"] is None:
            tabular_features_file = None
        else:
            tabular_features_file = Constants.PROJECT_DIR.value + self.config["tabular_features_file"]
        self.image_datasets = {
            x: UKBBDataset(
                Constants.PROJECT_DIR.value + self.config["labels_file"],
                tabular_features_file,
                self.config["data_dir"],
                x,
                self.data_transforms[x],
                self.config["input_modality"],
                self.config.get("sax_num_slices", None),
                self.config.get("lax_num_slices", None),
            )
            for i, x in enumerate(["train", "test", "val"])
        }
        self.class_names = self.image_datasets["train"].classes

    def create_store_folder(self, store_name):
        if self.config.get("save_results", True):
            self.store_dir = os.path.join(self.experiment_dir, store_name)
            os.makedirs(self.store_dir, exist_ok=True)
            os.makedirs(os.path.join(self.store_dir, "model"), exist_ok=True)
            os.makedirs(os.path.join(self.store_dir, "results"), exist_ok=True)

    def evaluate(self, model, dataloader, criterion):
        model.eval()  # Set the model to evaluation mode
        total_loss = 0
        total_samples = 0

        # Initialize metrics
        get_f1 = torchmetrics.F1Score(task="binary").to(self.DEVICE)
        get_acc = torchmetrics.Accuracy(task="binary").to(self.DEVICE)
        get_auroc = torchmetrics.AUROC(task="binary").to(self.DEVICE)

        # for balanced accuracy
        all_labels = []
        all_preds = []
        all_probs = []

        with (torch.no_grad()):
            for idx, sample in enumerate(dataloader):
                #print("idx", idx)
                inputs = self._get_inputs(sample)
                inputs = self._move_inputs_to_device(inputs, self.DEVICE)
                labels = sample[Constants.TARGET_NAME.value].to(self.DEVICE).long() #.float()
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels)
                #loss = criterion(outputs, labels) #CHANGED

                total_loss += loss.item() * self._get_inputs_size(sample)
                total_samples += self._get_inputs_size(sample)

                #probs = torch.sigmoid(outputs)
                #preds = (outputs > 0).float()  # Binary predictions

                probs = torch.softmax(outputs, dim=1)[:, 1].float()  #CHANGED
                preds = torch.argmax(outputs, dim=1).float()  # CHANGED


                get_f1.update(preds, labels)
                get_acc.update(preds, labels)
                get_auroc.update(probs, labels)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        mean_loss = total_loss / total_samples
        f1 = get_f1.compute()
        acc = get_acc.compute()
        auroc = get_auroc.compute()
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)


        self.logger.info(
            f"Loss: {mean_loss:.4f} Balanced_Acc: {balanced_acc} Acc: {acc:.4f} F1: {f1:.4f} AUROC: {auroc:.4f}"
        )


        return mean_loss, acc, f1, balanced_acc, auroc, all_labels, all_preds, all_probs

    def evaluate_by_eid(self, model, dataloader, criterion):
        model.eval()  # Set the model to evaluation mode
        all_preds = []
        all_labels = []
        all_img_paths = []

        with torch.no_grad():
            for sample in dataloader:
                inputs = self._get_inputs(sample)
                inputs = self._move_inputs_to_device(inputs, self.DEVICE)
                labels = sample[Constants.TARGET_NAME.value].to(self.DEVICE)
                outputs = model(inputs).squeeze(1)
                img_paths = self._get_image_paths(sample)

                preds = (outputs > 0).float()  # Binary predictions
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_img_paths.extend(img_paths)

        return {"preds": all_preds, "labels": all_labels, "img_paths": all_img_paths}

    def _get_inputs(self, sample):
        if self.config["mode"] == "image":
            if isinstance(sample["image"], list):
                inputs = list(map(lambda x: x.float(), sample["image"]))
            else:
                inputs = sample["image"].float()
        elif self.config["mode"] == "tabular":
            inputs = sample["tabular_vector"].float()
        elif self.config["mode"] == "tabular+image":
            if isinstance(sample["image"], list):
                inputs = {
                    "image": list(map(lambda x: x.float(), sample["image"])),
                    "tabular_vector": sample["tabular_vector"].float(),
                }
            else:
                inputs = {
                    "image": sample["image"].float(),
                    "tabular_vector": sample["tabular_vector"].float(),
                }
        else:
            raise ValueError("Unknown mode")
        return inputs

    def _get_image_paths(self, sample):
        return sample["img_path"]

    def _move_inputs_to_device(self, inputs, device):
        if isinstance(inputs, dict):
            inputs_to_device = {}
            for key, value in inputs.items():
                if isinstance(value, list):
                    inputs_to_device[key] = list(map(lambda x: x.to(device), value))
                else:
                    inputs_to_device[key] = value.to(device)
            return inputs_to_device
        elif isinstance(inputs, list):
            return list(map(lambda x: x.to(device), inputs))
        else:
            return inputs.to(device)

    def _get_inputs_size(self, sample):
        if self.config["mode"] == "image":
            if isinstance(sample["image"], list):
                inputs_size = sample["image"][0].size(0)
            else:
                inputs_size = sample["image"].size(0)
        elif self.config["mode"] == "tabular":
            inputs_size = sample["tabular_vector"].size(0)
        elif self.config["mode"] == "tabular+image":
            inputs_size = sample["tabular_vector"].size(0)
        else:
            raise ValueError("Unknown mode")
        return inputs_size

    def train_model(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
        model_name=None,
    ):
        """
        Start model training
        :param model: model to train
        :param criterion: loss function
        :param optimizer: optimizer
        :param scheduler: scheduler
        :param model_name: name of the model to save
        :return: trained model
        """
        if not model_name:
            model_name = model.__class__.__name__
        model.to(self.DEVICE)

        dataset_sizes = {x: len(self.dataloaders[x].sampler) for x in ["train", "val"]}
        #writer = (
        #    SummaryWriter(self.store_dir + "tensorboard")
        #    if self.config.get("save_results", True)
        #    else None
        #)

        best_model_wts = copy.deepcopy(model.state_dict())
        best_balanced_acc, best_acc, best_f1, best_auroc, best_epoch_number = 0.0, 0.0, 0.0, 0.0, 0

        early_stop = EarlyStopping(
            patience=self.config["early_stopping_patience"],
            verbose=self.config["early_stopping"],
            path=self.store_dir + "model/"
            if self.config.get("save_results", True)
            else None,
            logger=self.logger,
            save_results=self.config.get("save_results", True),
        )

        since = time.time()

        num_epochs = self.config["num_epochs"]

        get_f1 = torchmetrics.F1Score(task="binary").to(self.DEVICE)
        get_acc = torchmetrics.Accuracy(task="binary").to(self.DEVICE)
        get_auroc = torchmetrics.AUROC(task="binary").to(self.DEVICE)

        start_epoch = 0
        y_loss = {"train": [], "val": []}  # loss history
        y_balanced_acc = {"train": [], "val": []}  # balanced accuracy history
        y_acc = {"train": [], "val": []}  # accuracy history
        y_f1 = {"train": [], "val": []}  # f1 history
        y_auroc = {"train": [], "val": []}  # auroc history

        # Load checkpoint if resume_dir is specified
        if self.config.get("resume_dir"):
            (
                start_epoch,
                best_balanced_acc,
                best_acc,
                best_f1,
                best_auroc,
                y_loss,
                y_balanced_acc,
                y_acc,
                y_f1,
                y_auroc,
                best_model_wts,
                best_epoch_number,
            ) = self.load_checkpoint(self.store_dir, model, optimizer)


        for epoch in range(start_epoch, num_epochs):
            epoch_time = time.time()
            self.logger.info(f"Epoch {epoch}/{num_epochs - 1}")
            self.logger.info("-" * 10)

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                    if self.config["optimizer"] == "schedulefree":
                        optimizer.train()
                else:
                    model.eval()
                    if self.config["optimizer"] == "schedulefree":
                        optimizer.eval()

                running_loss = 0.0
                get_f1.reset()
                get_acc.reset()
                get_auroc.reset()

                # for balanced accuracy
                all_preds = []
                all_labels = []

                # Iterate over data
                #print("Before loop")
                for idx, sample in enumerate(self.dataloaders[phase]):
                    batch_time = time.time()
                    inputs = self._get_inputs(sample)
                    labels = sample[Constants.TARGET_NAME.value]
                    inputs = self._move_inputs_to_device(inputs, self.DEVICE)
                    labels = Variable(labels.to(self.DEVICE)).long()#CHABGED from float()

                    # Zero the parameter gradients if in training phase
                    if phase == "train":
                        optimizer.zero_grad()

                    # forward
                    with torch.set_grad_enabled(phase == "train"):
                        # check if inputs is a dictionary
                        outputs = model(inputs).squeeze(1)
                        # outputs = model(inputs.float()).squeeze(1)  # Ensure outputs have the right shape for BCEWithLogitsLoss
                        #preds = (outputs > 0).float()  # Use a threshold of 0 to get binary predictions (0 or 1)
                        preds = torch.argmax(outputs, dim=1).float() #CHANGED
                        loss = criterion(
                            outputs, labels
                        )# Ensure labels are of type float for BCEWithLogitsLoss

                        #print("idx", idx)
                        #print("loss", loss)
                        #print("outputs", outputs)
                        #print("preds", preds)
                        #print("labels", labels)

                        get_f1.update(preds, labels)
                        get_acc.update(preds, labels)

                        get_auroc.update(torch.softmax(outputs, dim=1)[:, 1].float(), labels) #CHANGED
                        #get_auroc.update(torch.sigmoid(outputs).float(), labels)

                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * self._get_inputs_size(sample)
                    batch_time_elapsed = time.time() - batch_time
                    #self.logger.info(f"Batch time: {batch_time_elapsed:.2f}s")
                if phase == "train" and self.config["optimizer"] != "schedulefree":
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = get_acc.compute()
                epoch_f1 = get_f1.compute()
                epoch_auroc = get_auroc.compute()
                epoch_balanced_acc = balanced_accuracy_score(all_labels, all_preds)

                current_lr = optimizer.param_groups[0]["lr"]

                self.logger.info(
                    f"{phase} Loss: {epoch_loss:.4f} Balanced Acc {epoch_balanced_acc:.4f} Acc: {epoch_acc:.4f} "
                    f"F1: {epoch_f1:.4f} AUROC: {epoch_auroc:.4f} LR: {current_lr:.4f}"
                )
                if self.config["save_results"]:
                    wandb.log(
                        {
                            f"{phase}_loss": epoch_loss,
                            f"{phase}_balanced_accuracy": epoch_balanced_acc,
                            f"{phase}_accuracy": epoch_acc,
                            f"{phase}_f1": epoch_f1,
                            f"{phase}_auroc": epoch_auroc,
                            "epoch": epoch,
                            "learning_rate": current_lr,
                        },
                        step=epoch,
                    )

                y_loss[phase].append(epoch_loss)
                y_balanced_acc[phase].append(epoch_balanced_acc)
                y_acc[phase].append(epoch_acc)
                y_f1[phase].append(epoch_f1)
                y_auroc[phase].append(epoch_auroc)

                get_f1.reset()
                get_acc.reset()

                """
                # deep copy the model
                if phase == "val":
                    logger_losses = {
                        "val_loss": y_loss["val"][epoch],
                        "train_loss": y_loss["train"][epoch],
                    }
                    logger_balanced_acc = {
                        "val_balanced_acc": y_balanced_acc["val"][epoch],
                        "train_balanced_acc": y_balanced_acc["train"][epoch],
                    }
                    logger_acc = {
                        "val_acc": y_acc["val"][epoch],
                        "train_acc": y_acc["train"][epoch],
                    }
                    logger_f1 = {
                        "val_f1": y_f1["val"][epoch],
                        "train_f1": y_f1["train"][epoch],
                    }
                    logger_auroc = {
                        "val_auroc": y_auroc["val"][epoch],
                        "train_auroc": y_auroc["train"][epoch],
                    }
                    if writer:
                        writer.add_scalars("losses", logger_losses, global_step=epoch)
                        writer.add_scalars(
                            "balanced_accuracy", logger_balanced_acc, global_step=epoch
                        )
                        writer.add_scalars("accuracy", logger_acc, global_step=epoch)
                        writer.add_scalars("f1", logger_f1, global_step=epoch)
                        writer.add_scalars("auroc", logger_auroc, global_step=epoch)
                """
                # deep copy the model
                if phase == "val":
                    #if ((best_f1 < epoch_f1) or
                    #        (best_f1 == epoch_f1 and best_auroc < epoch_auroc) or
                    #        (best_f1 == epoch_f1 and best_auroc == epoch_auroc and best_balanced_acc < epoch_balanced_acc) or
                    #        (best_f1 == epoch_f1 and best_auroc == epoch_auroc and best_balanced_acc == epoch_balanced_acc and best_epoch_number < epoch)
                    #):
                    if ((best_balanced_acc < epoch_balanced_acc) or
                            (best_balanced_acc == epoch_balanced_acc and best_f1 < epoch_f1) or
                            (best_balanced_acc == epoch_balanced_acc and best_f1 == epoch_f1 and best_auroc < epoch_auroc) or
                            (best_balanced_acc == epoch_balanced_acc and best_f1 == epoch_f1 and best_auroc == epoch_auroc and best_epoch_number < epoch)):

                        best_balanced_acc = epoch_balanced_acc
                        best_acc = epoch_acc
                        best_f1 = epoch_f1
                        best_auroc = epoch_auroc
                        best_epoch_number = epoch
                        best_model_wts = copy.deepcopy(model.state_dict())
                        self.logger.info(
                            f"Best model saved at epoch {epoch} with balanced accuracy {best_balanced_acc:.4f}, accuracy {best_acc:.4f} and F1 {best_f1:.4f}, AUROC {best_auroc:.4f}"
                        )

                        # Save the best model
                        if self.config.get("save_results", True):
                            best_model_path = os.path.join(self.store_dir, "model", "model_best.pth")
                            torch.save(best_model_wts, best_model_path)

            #early_stop(epoch_acc, model, epoch)
            early_stop(epoch_f1, model, epoch)

            if self.config["early_stopping"] and early_stop.early_stop:
                self.logger.info("Early stopping")
                break

            if self.config.get("save_results", True):
                self.save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "best_balanced_acc": best_balanced_acc,
                        "best_acc": best_acc,
                        "best_f1": best_f1,
                        "best_auroc": best_auroc,
                        "best_epoch_number": best_epoch_number,
                        "best_model_wts": best_model_wts,
                        "optimizer": optimizer.state_dict(),
                        "y_loss": y_loss,
                        "y_balanced_acc": y_balanced_acc,
                        "y_acc": y_acc,
                        "y_f1": y_f1,
                    },
                    checkpoint_dir=self.store_dir + "model/",
                    filename=f"last_checkpoint.pth.tar",
                )
            epoch_time_elapsed = time.time() - epoch_time
            self.logger.info(
                f"Epoch time: {epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.0f}s"
            )
            if self.config["save_results"]:
                wandb.log(
                    {
                        "epoch_time": epoch_time_elapsed,
                    },
                    step=epoch,
                )

        time_elapsed = time.time() - since
        self.logger.info(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        self.logger.info(f"Best val balanced_acc: {best_balanced_acc:4f}")
        self.logger.info(f"Best val acc: {best_acc:4f}")
        self.logger.info(f"Best val F1: {best_f1:4f}")
        self.logger.info(f"Best val AUROC: {best_auroc:4f}")
        self.logger.info(f"Best epoch number: {best_epoch_number}")

        # After the training and validation loop in train_model
        model.load_state_dict(best_model_wts)  # Load the best model weights

        # Evaluation on the test set
        test_loss, test_accuracy, test_f1, test_balanced_acc, test_auroc, all_labels, all_preds, all_probs = self.evaluate(
            model, self.dataloaders["test"], criterion
        )
        self.logger.info(
            f"Final evaluation on test set - Loss: {test_loss:.4f}, Balanced Accuracy: {test_accuracy:.4f}, "
            f"Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}, AUROC: {test_auroc:.4f}"
        )
        if self.config["save_results"]:
            wandb.log(
                {
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                    "test_f1": test_f1,
                    "test_balanced_acc": test_balanced_acc,
                    "test_auroc": test_auroc,
                },
                step=num_epochs,
            )
            # wandb for roc curves
            try:
                wandb.log({"roc": wandb.plot.roc_curve(all_labels, all_probs)}, step=num_epochs)
            except Exception as e:
                self.logger.error(f"Error in plotting roc curve {e}")


        self.metrics = {
            "best_balanced_acc_val": best_balanced_acc,
            "best_acc_val": best_acc.item(),
            "best_f1_val": best_f1.item(),
            "best_auroc_val": best_auroc.item(),
            "best_epoch_number": best_epoch_number,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy.item(),
            "test_f1": test_f1.item(),
            "test_balanced_acc": test_balanced_acc,
            "test_auroc": test_auroc.item(),
        }
        self.result.update(self.metrics)

        if self.config["save_results"]:
            df_metrics = pd.DataFrame(self.metrics, index=[0])
            df_metrics.to_csv(self.store_dir + "results/metrics.csv", index=False)
            df_labels_preds = pd.DataFrame({"all_preds": all_preds, "all_labels": all_labels, "all_probs": all_probs})
            df_labels_preds.to_csv(self.store_dir + "results/labels_preds.csv", index=False)


        # save best model
        model.load_state_dict(best_model_wts)
        if self.config.get("save_results", True):
            path = self.store_dir + "model/" + model_name + "_weights.pth"
            torch.save(model.state_dict(), path)
        if self.config.get("save_results", True):
            path = self.store_dir + "model/" + model_name + "_model.pth"
            torch.save(model, path)
        #if writer:
        #    writer.close()

        return model

    def _create_dataloaders(self):
        """
        Create data loaders, including splitting the training dataset into training and validation subsets.
        """
        from sklearn.model_selection import train_test_split

        # Split training data into training and validation sets with stratification on the event label
        train_idx, val_idx = train_test_split(
            range(len(self.image_datasets["train"])),
            test_size=self.config["val_size"],  # 10% for validation
            random_state=self.config["seed"],
            stratify=self.image_datasets["train"]
            .img_labels[Constants.TARGET_NAME.value]
            .tolist(),
        )

        # Create samplers for training and validation subsets
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

        # shuffling is not necessary since we use SubsetRandomSampler
        # Update dataloaders dictionary to include validation
        self.dataloaders = {
            "train": torch.utils.data.DataLoader(
                self.image_datasets["train"],
                batch_size=self.config["batch_size"],
                sampler=train_sampler,
                num_workers=self.config["num_workers"],
                prefetch_factor=self.config["prefetch_factor"],
                pin_memory=True,
                persistent_workers=True,
            ),
            "val": torch.utils.data.DataLoader(
                self.image_datasets[
                    "train"
                ],  # Notice it still uses the 'train' dataset
                batch_size=self.config["batch_size"],
                sampler=val_sampler,
                num_workers=0,
                # num_workers=self.config["num_workers"],
                # prefetch_factor=self.config["prefetch_factor"],
                # pin_memory=True,
                # persistent_workers=True,
            ),
            "test": torch.utils.data.DataLoader(
                self.image_datasets["test"],
                batch_size=self.config["batch_size"],
                shuffle=self.config["shuffle"],
                num_workers=0,
                # num_workers=self.config["num_workers"],
                # prefetch_factor=self.config["prefetch_factor"],
                # pin_memory=True,
                # persistent_workers=True,
            ),
        }

    def _create_dataloaders_val(self):
        self.dataloaders = {
            "train": torch.utils.data.DataLoader(
                self.image_datasets["train"],
                batch_size=self.config["batch_size"],
                num_workers=self.config["num_workers"],
                shuffle=self.config["shuffle"],
                prefetch_factor=self.config["prefetch_factor"],
                pin_memory=True,
                # persistent_workers=True,
            ),
            "val": torch.utils.data.DataLoader(
                self.image_datasets["val"],
                batch_size=self.config["batch_size"],
                shuffle=False,
                num_workers=0,
                # num_workers=self.config["num_workers"],
                # prefetch_factor=self.config["prefetch_factor"],
                pin_memory=True,
                # persistent_workers=True,
            ),
            "test": torch.utils.data.DataLoader(
                self.image_datasets["test"],
                batch_size=self.config["batch_size"],
                shuffle=False,
                num_workers=0,
                # num_workers=self.config["num_workers"],
                # prefetch_factor=self.config["prefetch_factor"],
                pin_memory=True,
                # persistent_workers=True,
            ),
        }

    def get_classes_weight(self):
        """
        Get the classes weight for the loss function
        :return: classes weight
        """
        y_train = (
            self.image_datasets["train"]
            .img_labels[Constants.TARGET_NAME.value]
            .tolist()
        )
        class_sample_count = np.array(
            [len(np.where(y_train == t)[0]) for t in np.unique(y_train)]
        )
        weight = 1.0 - class_sample_count / sum(class_sample_count)
        weight = torch.tensor(weight, dtype=torch.float).to(self.DEVICE)
        return weight

    def update_results(self):
        """
        Update results.csv with current experiment results.
        """
        df_results = pd.read_csv(Constants.PROJECT_DIR.value + "data/results.csv")
        df_results = pd.concat(
            [df_results, pd.DataFrame.from_records([self.result])], ignore_index=True
        )
        df_results.to_csv(Constants.PROJECT_DIR.value + "data/results.csv", index=False)
