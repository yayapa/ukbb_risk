import os
import shutil
from datetime import datetime
import json

import numpy as np
import torch
import torchvision

import sys

from TwoDImage.image_classification.constants import Constants
from TwoDImage.image_classification.classifier import EventClassifier
from TwoDImage.image_classification.ResNet18Custom import ResNet18Custom
from TwoDImage.image_classification.ResNet18CustomNChannels import (
    ResNet18CustomNChannels,
    DualEncoderResNet18,
    JointDualEncoderResNet18
)
from TwoDImage.image_classification.Simple3DCNN import Simple3DCNN_Model
from TwoDImage.image_classification.SimpleCNN import SimpleCNN
from TwoDImage.image_classification.LinearClassifierModule import LinearClassifierModule
from TwoDImage.image_classification.JointModule import JointModule
from TwoDImage.image_classification.JointModuleResNet18NChannels import (
    JointModuleResNet18NChannels,
)
from TwoDImage.image_classification.JointCAModuleResnet18_3D import JointCAModuleResNet18_3D
from TwoDImage.image_classification.ResNet18_3D import (
    ResNet18_3D,
    ResNet18TwoEncoders_3D,
)

from TwoDImage.image_classification.JointModuleResNet18_3D import JointModuleResNet18_3D
from TwoDImage.image_classification.TwoDImageLogger import TwoDImageLogger
from TwoDImage.image_classification.ViT.VisionTransformer import VisionTransformer
from TwoDImage.image_classification.ViT.ViTWholeBodyCA import ViTWholeBodyCA
from TwoDImage.image_classification.ViT.WarmupConstantCosineDecayScheduler import (
    WarmupConstantCosineDecayScheduler,
)
import wandb
from json_minify import json_minify
import pandas as pd
import schedulefree
import torch.multiprocessing as mp

class Trainer:
    def __init__(self, config_file_path="config.json"):
        self.config = self._load_config(config_file_path)
        # self._init_wandb()

    def _load_config(self, config_file_path):
        with open(config_file_path, "r") as f:
            json_str = f.read()
        json_str = json_minify(json_str)
        config = json.loads(json_str)
        # self._print_config(config)
        return config

    def _init_wandb(self):
        os.environ["WANDB_DIR"] = Constants.WANDB_DIR.value
        wandb.init(
            project=Constants.WANDB_PROJECT.value,
            config=self.config,
            name=self.config["experiment_name"],
        )

    def _reinit_wandb(self, group_name, experiment_dir):
        os.environ["WANDB_DIR"] = Constants.WANDB_DIR.value
        wandb.init(
            project=Constants.WANDB_PROJECT.value,
            config=self.config,
            name=self.config["experiment_name"],
            group=group_name,
            reinit=True,
        )

        wandb.config.update({"experiment_dir": experiment_dir})

    def _load_logger(self, experiment_dir, logger_name="training.log"):
        save_results = self.config.get("save_results", True)
        if save_results and experiment_dir:
            logger = TwoDImageLogger(
                log_dir=experiment_dir + "logs/", save_results=save_results
            ).get_logger()
        else:
            logger = TwoDImageLogger(save_results=save_results).get_logger()
        return logger

    def _print_config(self, config):
        self.logger.info("Config:")
        self.logger.info("-" * 15)
        for key, value in config.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("-" * 15)

    def _print_config_logger(self, config, logger):
        logger.info("Config:")
        logger.info("-" * 15)
        for key, value in config.items():
            logger.info(f"{key}: {value}")
        logger.info("-" * 15)

    @staticmethod
    def set_seed(seed=42):
        """Set seed for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _get_tabular_model(self):
        if len(self.classifier.class_names) == 2:
            model = LinearClassifierModule(
                num_features=len(
                    self.classifier.image_datasets["train"].tabular_features.columns
                ),
                hidden_layers=self.config["tabular_hidden_layers"],
                num_classes=1,
                dropout_rate=self.config["dropout_rate"],
            )
        else:
            model = LinearClassifierModule(
                num_features=len(
                    self.classifier.image_datasets["train"].tabular_features.columns
                )
                - 1,
                hidden_layers=self.config["tabular_hidden_layers"],
                num_classes=len(self.classifier.class_names),
                dropout_rate=self.config["dropout_rate"],
            )
        return model

    def _get_tabular_image_model(self):
        if len(self.classifier.class_names) == 2:
            model = JointModule(
                num_tabular_features=len(
                    self.classifier.image_datasets["train"].tabular_features.columns
                ),
                tabular_hidden_layers=self.config["tabular_hidden_layers"],
                combine_hidden_layers=self.config["combine_hidden_layers"],
                num_classes=1,
                dropout_rate=self.config["dropout_rate"],
            )
        else:
            model = JointModule(
                num_tabular_features=len(
                    self.classifier.image_datasets["train"].tabular_features.columns
                ),
                tabular_hidden_layers=self.config["tabular_hidden_layers"],
                combine_hidden_layers=self.config["combine_hidden_layers"],
                num_classes=len(self.classifier.class_names),
                dropout_rate=self.config["dropout_rate"],
            )
        return model

    def _get_joint_camodule_resnet18_3D(self, dropout_rate=None, num_tabular_features=None):
        if self.config["input_modality"] == "whole_body_3d_wat_fat":
            num_channels = 2
        else:
            ValueError("Not implemented")

        if self.config["num_classes"] == 2:
            model = JointCAModuleResNet18_3D(
                num_tabular_features=num_tabular_features,
                tabular_hidden_layers=self.config["tabular_hidden_layers"],
                combine_hidden_layers=self.config["combine_hidden_layers"],
                #num_classes=1,
                num_classes=2,
                dropout_rate=self.config["dropout_rate"],
                n_channels=num_channels,
            )
        else:
            model = JointCAModuleResNet18_3D(
                num_tabular_features=num_tabular_features,
                tabular_hidden_layers=self.config["tabular_hidden_layers"],
                combine_hidden_layers=self.config["combine_hidden_layers"],
                num_classes=len(self.classifier.class_names),
                dropout_rate=self.config["dropout_rate"],
                n_channels=num_channels,
            )
        return model




    def _get_joint_module_resnet18NChannels(self, dropout_rate=None, num_tabular_features=None, seed=None):
        if self.config["input_modality"] == "liver_projections":
            num_channels = 7
        elif self.config["input_modality"] == "liver_projections_all":
            num_channels = 23
        elif self.config["input_modality"] == "liver_t1_map" or self.config["input_modality"] == "pancreas_t1_map":
            num_channels = 1
        else:
            ValueError("Not implemented")

        if self.config["num_classes"] == 2:
            model_name = "resnet18_NChannel"
            if self.config["restore_image_model_path"] is not None:
                restore_image_model_path = self.config["restore_image_model_path"] + f"{model_name}/{seed}/model/{model_name}_{seed}_weights.pth"
            else:
                restore_image_model_path = None
            model = JointModuleResNet18NChannels(
                num_tabular_features=num_tabular_features,
                tabular_hidden_layers=self.config["tabular_hidden_layers"],
                combine_hidden_layers=self.config["combine_hidden_layers"],
                #num_classes=1,
                num_classes=2,
                dropout_rate=self.config["dropout_rate"],
                tabular_input_dropout_rate=self.config["tabular_input_dropout_rate"],
                tabular_dropout_rate=self.config["tabular_dropout_rate"],
                combine_dropout_rate=self.config["combine_dropout_rate"],
                restore_image_model_path=restore_image_model_path,
                n_channels=num_channels,
            )
        else:
            model = JointModuleResNet18NChannels(
                num_tabular_features=len(
                    self.classifier.image_datasets["train"].tabular_features.columns
                ),
                tabular_hidden_layers=self.config["tabular_hidden_layers"],
                combine_hidden_layers=self.config["combine_hidden_layers"],
                num_classes=len(self.classifier.class_names),
                dropout_rate=self.config["dropout_rate"],
                n_channels=num_channels,
            )
        return model

    def _get_joint_module_resnet18_3D(self, dropout_rate=None, num_tabular_features=None, seed=None):
        if self.config["input_modality"] == "whole_body_3d_wat_fat":
            num_channels = 2
        else:
            ValueError("Not implemented")

        if self.config["num_classes"] == 2:
            model_name = "resnet18_3d"
            restore_image_model_path = self.config["restore_image_model_path"] + f"{model_name}/{seed}/model/{model_name}_{seed}_weights.pth"
            model = JointModuleResNet18_3D(
                num_tabular_features=num_tabular_features,
                tabular_hidden_layers=self.config["tabular_hidden_layers"],
                combine_hidden_layers=self.config["combine_hidden_layers"],
                #num_classes=1,
                num_classes=2,
                tabular_input_dropout_rate=self.config["tabular_input_dropout_rate"],
                tabular_dropout_rate=self.config["tabular_dropout_rate"],
                combine_dropout_rate=self.config["combine_dropout_rate"],
                restore_image_model_path=restore_image_model_path,
                dropout_rate=dropout_rate,
                n_channels=num_channels,
            )
        else:
            model = JointModuleResNet18_3D(
                num_tabular_features=num_tabular_features,
                tabular_hidden_layers=self.config["tabular_hidden_layers"],
                combine_hidden_layers=self.config["combine_hidden_layers"],
                num_classes=len(self.classifier.class_names),
                dropout_rate=dropout_rate,
                n_channels=num_channels,
            )
        return model



    def _get_resnet18(self, dropout_rate=None):
        if len(self.classifier.class_names) == 2:
            #model = ResNet18Custom(n_classes=1, dropout_rate=dropout_rate)
            model = ResNet18Custom(n_classes=2, dropout_rate=dropout_rate)
        else:
            model = ResNet18Custom(
                n_classes=len(self.classifier.class_names), dropout_rate=dropout_rate
            )
        return model

    def _get_resnet18_Nchannel(self, dropout_rate=None):
        if self.config["input_modality"] == "liver_projections":
            num_channels = 7
        elif self.config["input_modality"] == "liver_projections_all":
            num_channels = 23
        elif self.config["input_modality"] == "liver_t1_map" or self.config["input_modality"] == "pancreas_t1_map":
            num_channels = 1
        else:
            ValueError("Not implemented")

        if self.config["num_classes"] == 2:
            model = ResNet18CustomNChannels(
                #n_classes=1, dropout_rate=dropout_rate, n_channels=num_channels
                n_classes=2, dropout_rate=dropout_rate, n_channels=num_channels
            )
        else:
            model = ResNet18CustomNChannels(
                n_classes=len(self.classifier.class_names),
                dropout_rate=dropout_rate,
                n_channels=num_channels,
            )
        return model

    def _get_resnet18_Nchannel_dual(self, dropout_rate=None):
        if self.config["input_modality"] == "liver_pancreas_t1_map":
            num_channels = 1
        else:
            ValueError("Not implemented")

        if self.config["num_classes"] == 2:
            model = DualEncoderResNet18(
                n_classes=1,
                dropout_rate=dropout_rate,
                n_channels=1
            )
        else:
            model = DualEncoderResNet18(
                n_classes=len(self.classifier.class_names),
                dropout_rate=dropout_rate,
                n_channels=num_channels
            )
        return model

    def _get_joint_resnet18_Nchannel_dual(self, dropout_rate=None, num_tabular_features=None):
        if self.config["input_modality"] == "liver_pancreas_t1_map":
            num_channels = 1
        else:
            ValueError("Not implemented")

        if self.config["num_classes"] == 2:
            model = JointDualEncoderResNet18(
                num_tabular_features=num_tabular_features,
                tabular_hidden_layers=self.config["tabular_hidden_layers"],
                combine_hidden_layers=self.config["combine_hidden_layers"],
                num_classes=1,
                dropout_rate=dropout_rate,
                n_channels=1)
        else:
            model = JointDualEncoderResNet18(
                num_tabular_features=num_tabular_features,
                tabular_hidden_layers=self.config["tabular_hidden_layers"],
                combine_hidden_layers=self.config["combine_hidden_layers"],
                num_classes=len(self.classifier.class_names),
                dropout_rate=dropout_rate,
                n_channels=num_channels
            )
        return model

    def _get_resnet18_3d(self, dropout_rate=None):
        if self.config["input_modality"] == "whole_body_3d":
            num_channels = 1
        elif self.config["input_modality"] == "whole_body_3d_all":
            num_channels = 4
        elif self.config["input_modality"] == "whole_body_3d_wat_fat":
            num_channels = 2
        elif self.config["input_modality"] == "cardiac_4d":
            # num_channels = 6
            # num_channels = 9
            num_channels = 50
        else:
            raise ValueError

        if self.config["num_classes"] == 2:
            model = ResNet18_3D(
                #num_classes=1, dropout_prob=dropout_rate, num_channels=num_channels
                num_classes=2, dropout_prob=dropout_rate, num_channels=num_channels  #CHANGED
            )
        else:
            model = ResNet18_3D(num_classes=len(self.classifier.class_names))
        return model

    def _get_resnet18twoencoders_3d(self, dropout_rate=None):
        num_channels = 50

        if len(self.classifier.class_names) == 2:
            model = ResNet18TwoEncoders_3D(
                num_classes=1,
                dropout_prob=dropout_rate,
                num_channels=num_channels,
                sax_num_slices=self.config["sax_num_slices"],
                lax_num_slices=self.config["lax_num_slices"],
            )
        else:
            model = ResNet18TwoEncoders_3D(num_classes=len(self.classifier.class_names))
        return model

    def _get_simple3dcnn(self, dropout_rate=None):
        if self.config["input_modality"] == "whole_body_3d":
            num_channels = 1
        elif self.config["input_modality"] == "cardiac_4d":
            num_channels = 50
        else:
            raise ValueError
        if len(self.classifier.class_names) == 2:
            model = Simple3DCNN_Model(
                in_channels=num_channels,
                num_classes=1,
                dropout_prob=self.config["dropout_rate"],
            )
        else:
            model = Simple3DCNN_Model(
                in_channels=num_channels,
                num_classes=len(self.classifier.class_names),
                dropout_prob=self.config["dropout_rate"],
            )
        return model

    def _get_simplecnn(self, dropout_rate=None):
        if self.config["input_modality"] == "liver_projections":
            num_channels = 7
        elif self.config["input_modality"] == "liver_projections_all":
            num_channels = 23
        else:
            num_channels = 3
        if len(self.classifier.class_names) == 2:
            model = SimpleCNN(
                num_channels, num_classes=1, dropout=self.config["dropout_rate"]
            )
        else:
            model = SimpleCNN(
                num_channels,
                num_classes=len(self.classifier.class_names),
                dropout=self.config["dropout_rate"],
            )
        return model

    def _get_resnet50(self, dropout_rate=None):
        if len(self.classifier.class_names) == 2:
            num_classes = 1
        else:
            num_classes = len(self.classifier.class_names)
        model = torchvision.models.resnet50(pretrained=True)
        if dropout_rate:
            model.fc = torch.nn.Sequential(
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(model.fc.in_features, num_classes),
            )
        else:
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model

    def _get_resnet101(self, dropout_rate=None):
        model = torchvision.models.resnet101(pretrained=True)
        if dropout_rate:
            model.fc = torch.nn.Sequential(
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(model.fc.in_features, len(self.classifier.class_names)),
            )
        else:
            model.fc = torch.nn.Linear(
                model.fc.in_features, len(self.classifier.class_names)
            )
        return model

    def _get_google_net(self, dropout_rate=None):
        model = torchvision.models.googlenet(pretrained=True)
        if dropout_rate:
            model.fc = torch.nn.Sequential(
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(model.fc.in_features, len(self.classifier.class_names)),
            )
        else:
            model.fc = torch.nn.Linear(
                model.fc.in_features, len(self.classifier.class_names)
            )
        return model

    def _get_mobilenet_v3_small(self, dropout_rate=None):
        if len(self.classifier.class_names) == 2:
            num_classes = 1
        else:
            num_classes = len(self.classifier.class_names)
        model = torchvision.models.mobilenet_v3_small(pretrained=True)
        if dropout_rate:
            model.classifier[3] = torch.nn.Sequential(
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(model.classifier[3].in_features, num_classes),
            )
        else:
            model.classifier[3] = torch.nn.Linear(
                model.classifier[3].in_features, num_classes
            )
        return model

    def _getVit(self, dropout_rate=None):
        img_size = 128
        slice_num = 14
        time_frame = 50
        patch_size = tuple(self.config["patch_size"])
        embed_dim = self.config["embed_dim"]
        depth = self.config["depth"]
        num_heads = self.config["num_heads"]
        in_channels = 1
        num_classes = self.config["num_classes"]
        norm_layer = torch.nn.LayerNorm
        use_both_axes = True
        return VisionTransformer(
            img_size=img_size,
            slice_num=slice_num,
            time_frame=time_frame,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            in_channels=in_channels,
            num_classes=num_classes,
            norm_layer=norm_layer,
            use_both_axes=use_both_axes,
        )

    def _getViTWholeBodyCA(self, dropout_rate=None):
        in_channels = 1
        patch_size = tuple(self.config["patch_size"])
        embed_dim = self.config["embed_dim"]
        depth = self.config["depth"]
        num_heads = self.config["num_heads"]
        in_channels = 1
        num_classes = self.config["num_classes"]
        if num_classes == 2:
            num_classes = 1
        forward_expansion = self.config["forward_expansion"]

        return ViTWholeBodyCA(
            in_channels=in_channels,
            patch_size=patch_size,
            num_classes=num_classes,
            embed_size=embed_dim,
            depth=depth,
            num_heads=num_heads,
            forward_expansion=forward_expansion,
            dropout=dropout_rate,
        )

    def _create_experiment_folder(self):
        """
        Create store folder for storing experiment results
        """
        if self.config["resume_dir"] is not None:
            return self.config["resume_dir"]
        elif self.config.get("save_results", True):
            time = datetime.now()
            folder_name = time.strftime("%Y-%m-%d_%H-%M-%S/")
            path = Constants.STORE_DIR.value + "experiments/" + folder_name
            os.makedirs(path, exist_ok=True)
            # store config
            with open(path + "config.json", "w") as f:
                json.dump(self.config, f, indent=4)
            return path
        else:
            return None

    def _get_optimizer(self, model):
        if self.config["optimizer"].lower() == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config["weight_decay"],
            )
        elif self.config["optimizer"].lower() == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config["weight_decay"],
            )
        elif self.config["optimizer"].lower() == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config["weight_decay"],
            )
        elif self.config["optimizer"].lower() == "schedulefree":
            optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=self.config["lr"])
        else:
            raise ValueError("Unknown optimizer")
        return optimizer

    def _get_scheduler(self, optimizer):
        if self.config["scheduler"] == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config["scheduler_step_size"],
                gamma=self.config["scheduler_gamma"],
            )
        elif self.config["scheduler"] == "cosine_constant":
            scheduler = WarmupConstantCosineDecayScheduler(
                optimizer,
                warmup_epochs=self.config["warmup_epochs"],
                total_epochs=self.config["num_epochs"],
                min_lr=self.config["min_lr"],
            )
        else:
            raise NotImplementedError("Unknown scheduler")
        return scheduler

    def _get_model(self, model_name, classifier=None, seed=None):
        if model_name == "resnet18":
            return self._get_resnet18(dropout_rate=self.config["dropout_rate"])
        elif model_name == "resnet18_NChannel":
            return self._get_resnet18_Nchannel(dropout_rate=self.config["dropout_rate"])
        elif model_name == "resnet18_NChannel_dual":
            return self._get_resnet18_Nchannel_dual(dropout_rate=self.config["dropout_rate"])
        elif model_name == "joint_resnet18_NChannel_dual":
            num_tabular_features = len(classifier.image_datasets["train"].tabular_features.columns)
            return self._get_joint_resnet18_Nchannel_dual(
                num_tabular_features=num_tabular_features,
                dropout_rate=self.config["dropout_rate"])
        elif model_name == "simplecnn":
            return self._get_simplecnn(dropout_rate=self.config["dropout_rate"])
        elif model_name == "resnet18_3d":
            return self._get_resnet18_3d(dropout_rate=self.config["dropout_rate"])
        elif model_name == "resnet18twoencoders_3d":
            return self._get_resnet18twoencoders_3d(
                dropout_rate=self.config["dropout_rate"]
            )
        elif model_name == "simple3dcnn":
            return self._get_simple3dcnn(dropout_rate=self.config["dropout_rate"])
        elif model_name == "resnet50":
            return self._get_resnet50(dropout_rate=self.config["dropout_rate"])
        elif model_name == "resnet101":
            return self._get_resnet101(dropout_rate=self.config["dropout_rate"])
        elif model_name == "google_net":
            return self._get_google_net(dropout_rate=self.config["dropout_rate"])
        elif model_name == "mobilenet_v3_small":
            return self._get_mobilenet_v3_small(
                dropout_rate=self.config["dropout_rate"]
            )
        elif model_name == "tabular":
            return self._get_tabular_model()
        elif model_name == "tabular+image":
            return self._get_tabular_image_model()
        elif model_name == "JointModuleResNet18NChannels":
            num_tabular_features = len(classifier.image_datasets["train"].tabular_features.columns)
            return self._get_joint_module_resnet18NChannels(num_tabular_features=num_tabular_features, dropout_rate=self.config["dropout_rate"], seed=seed)
        elif model_name == "JointModuleResNet18_3D":
            num_tabular_features = len(classifier.image_datasets["train"].tabular_features.columns)
            return self._get_joint_module_resnet18_3D(num_tabular_features=num_tabular_features, dropout_rate=self.config["dropout_rate"], seed=seed)
        elif model_name == "JointCAModuleResNet18_3D":
            num_tabular_features = len(classifier.image_datasets["train"].tabular_features.columns)
            return self._get_joint_camodule_resnet18_3D(num_tabular_features=num_tabular_features, dropout_rate=self.config["dropout_rate"])
        elif model_name == "ViT":
            return self._getVit(dropout_rate=self.config["dropout_rate"])
        elif model_name == "ViTWholeBodyCA":
            return self._getViTWholeBodyCA(dropout_rate=self.config["dropout_rate"])
        else:
            raise ValueError("Unknown model name")

    def copy_file(self, src_file, dest_file, logger):
        #logger.info(f"Copying file from {src_file} to {dest_file}")
        if os.path.exists(src_file):
            shutil.copy(src_file, dest_file)
            #logger.info(f"Successfully copied: {src_file} to {dest_file}")
        else:
            logger.warning(f"Source file does not exist: {src_file}")

    def copy_to_tmp(self):
        self.logger.info("Copying data to tmp directory")

        # Reading the labels file and extracting 'eid' column
        try:
            eids = pd.read_csv(self.config["labels_file"])["eid"]
            self.logger.info(f"Reading labels file: {self.config['labels_file']}")
        except FileNotFoundError as e:
            self.logger.error(f"Labels file not found: {e}")
            raise
        except pd.errors.EmptyDataError as e:
            self.logger.error(f"Labels file is empty: {e}")
            raise
        except KeyError as e:
            self.logger.error(f"'eid' column not found in labels file: {e}")
            raise

        data_dir = self.config["data_dir"]

        for eid in eids:
            try:
                tmp_dir = os.path.join(
                    Constants.TMP_DIR.value, "ukbb_risk_assessment", str(eid)
                )
                os.makedirs(tmp_dir, exist_ok=True)
                #self.logger.info(f"Created directory: {tmp_dir}")

                if self.config["input_modality"] == "cardiac_4d":
                    src_file = os.path.join(data_dir, str(eid), "processed_seg_allax.npz")
                    dest_file = os.path.join(tmp_dir, "processed_seg_allax.npz")
                    self.copy_file(src_file, dest_file, self.logger)

                elif self.config["input_modality"] == "whole_body_3d":
                    src_file = os.path.join(data_dir, str(eid), "wat.nii.gz")
                    dest_file = os.path.join(tmp_dir, "wat.nii.gz")
                    self.copy_file(src_file, dest_file, self.logger)

                elif self.config["input_modality"] == "whole_body_3d_all":
                    files_to_copy = ["wat.nii.gz", "fat.nii.gz", "inp.nii.gz", "opp.nii.gz"]
                    for file_name in files_to_copy:
                        src_file = os.path.join(data_dir, str(eid), file_name)
                        dest_file = os.path.join(tmp_dir, file_name)
                        self.copy_file(src_file, dest_file, self.logger)
                elif self.config["input_modality"] == "whole_body_3d_wat_fat":
                    files_to_copy = ["wat.nii.gz", "fat.nii.gz"]
                    for file_name in files_to_copy:
                        src_file = os.path.join(data_dir, str(eid), file_name)
                        dest_file = os.path.join(tmp_dir, file_name)
                        self.copy_file(src_file, dest_file, self.logger)

                else:
                    raise ValueError("Not implemented")

            except Exception as e:
                self.logger.error(f"Error copying file for eid {eid}: {e}")
                raise

        self.config["data_dir"] = Constants.TMP_DIR.value + "/ukbb_risk_assessment"
        self.logger.info("Data copy to tmp directory completed")

    def calculate_metrics_over_seeds(self, experiment_dir):
        dirs = os.listdir(experiment_dir)
        for dir in dirs:
            # check if the directory is a directory
            if not os.path.isdir(os.path.join(experiment_dir, dir)) or dir == "logs":
                continue
            seed_dirs = os.listdir(os.path.join(experiment_dir, dir))
            test_acc_through_seeds = []
            test_f1_through_seeds = []
            test_auroc_through_seeds = []
            val_acc_through_seeds = []
            val_f1_through_seeds = []
            val_auroc_through_seeds = []
            for seed_dir in seed_dirs:
                metrics_dir = os.path.join(
                    experiment_dir, dir, seed_dir, "results", "metrics.csv"
                )
                metrics = pd.read_csv(metrics_dir)

                test_balanced_acc = metrics["test_balanced_acc"].values[0]
                val_acc = metrics["best_balanced_acc_val"].values[0]
                test_f1 = metrics["test_f1"].values[0]
                val_f1 = metrics["best_f1_val"].values[0]
                test_auroc = metrics["test_auroc"].values[0]
                val_auroc = metrics["best_auroc_val"].values[0]

                test_acc_through_seeds.append(test_balanced_acc)
                val_acc_through_seeds.append(val_acc)
                test_f1_through_seeds.append(test_f1)
                val_f1_through_seeds.append(val_f1)
                test_auroc_through_seeds.append(test_auroc)
                val_auroc_through_seeds.append(val_auroc)
            # print the mean and std of the accuracy and f1 with 3 decimal points
            self.logger.info(
                f"{dir}: test balanced acc: {np.mean(test_acc_through_seeds):.3f}+-{np.std(test_acc_through_seeds):.3f}, val acc: {np.mean(val_acc_through_seeds):.3f}+-{np.std(val_acc_through_seeds):.3f}"
            )
            self.logger.info(
                f"{dir}: test f1: {np.mean(test_f1_through_seeds):.3f}+-{np.std(test_f1_through_seeds):.3f}, val f1: {np.mean(val_f1_through_seeds):.3f}+-{np.std(val_f1_through_seeds):.3f}"
            )
            self.logger.info(
                f"{dir}: test auroc: {np.mean(test_auroc_through_seeds):.3f}+-{np.std(test_auroc_through_seeds):.3f}, val f1: {np.mean(val_auroc_through_seeds):.3f}+-{np.std(val_auroc_through_seeds):.3f}"
            )

    def train_models(self):
        seeds = self.config["random_seed_pool"][: self.config["num_seeds"]]
        assert (
            len(self.config["random_seed_pool"]) >= self.config["num_seeds"]
        ), "Not enough random seeds in the pool"
        experiment_dir = self._create_experiment_folder()
        self.logger = self._load_logger(experiment_dir)
        self._print_config(self.config)
        self.logger.info(f"Experiment directory: {experiment_dir}")
        self.copy_to_tmp()
        # wandb.config.update({"experiment_dir": experiment_dir})

        for seed in seeds:
            self.set_seed(seed)
            self.classifier = EventClassifier(self.config, self.logger)
            self.classifier.set_device()
            self.classifier.experiment_dir = experiment_dir

            for model_name in self.config["model_names"]:
                # wandb.run.group = f"{model_name}_seed_{seed}"
                experiment_name = self.config["experiment_name"]
                if self.config["save_results"]:
                    self._reinit_wandb(f"{experiment_name}_{model_name}", experiment_dir)
                model = self._get_model(model_name, self.classifier, seed)
                self.logger.info(f"Training model: {model_name} with seed: {seed}")
                if self.config["save_results"]:
                    wandb.log({"seed": seed, "model_name": model_name}, step=0)
                #print(model)
                # Set the random seed for reproducibility

                #criterion = torch.nn.BCEWithLogitsLoss()
                criterion = torch.nn.CrossEntropyLoss()  #CHANGED
                optimizer = self._get_optimizer(model)
                scheduler = self._get_scheduler(optimizer)

                self.classifier.create_store_folder(model_name + "/" + str(seed) + "/")

                # Train the model
                model_ft = self.classifier.train_model(
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    model_name=f"{model_name}_{seed}",
                )

                del model
                torch.cuda.empty_cache()

        wandb.finish()
        if experiment_dir is None:
            experiment_dir = ""
        self.logger.info("Experiment directory: " + experiment_dir)
        self.calculate_metrics_over_seeds(experiment_dir)


    def _train_model_single_seed(self, seed, gpu_id, experiment_dir):
        torch.cuda.set_device(gpu_id)  # Set the GPU for this process
        num_gpus = torch.cuda.device_count()
        print("Number of GPUs detected: ", num_gpus)
        logger = self._load_logger(experiment_dir=experiment_dir, logger_name=f"training_{seed}.log")
        self._print_config_logger(logger=logger, config=self.config)
        logger.info(f"Experiment directory: {experiment_dir}")
        self.set_seed(seed)
        classifier = EventClassifier(self.config, logger)
        classifier.set_device_by_id(gpu_id)
        classifier.experiment_dir = experiment_dir
        for model_name in self.config["model_names"]:
            experiment_name = self.config["experiment_name"]
            if self.config["save_results"]:
                self._reinit_wandb(f"{experiment_name}_{model_name}", experiment_dir)
            model = self._get_model(model_name, classifier)
            logger.info(f"Training model: {model_name} with seed: {seed}")
            if self.config["save_results"]:
                wandb.log({"seed": seed, "model_name": model_name}, step=0)
            # print(model)
            # Set the random seed for reproducibility

            criterion = torch.nn.BCEWithLogitsLoss()
            optimizer = self._get_optimizer(model)
            scheduler = self._get_scheduler(optimizer)

            classifier.create_store_folder(model_name + "/" + str(seed) + "/")

            # Train the model
            model_ft = classifier.train_model(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                model_name=f"{model_name}_{seed}",
            )

            del model
            torch.cuda.empty_cache()

        if self.config["save_results"]:
            wandb.finish()

    def train_models_parallel(self):
        mp.set_start_method('spawn', force=True)  # Set the start method to 'spawn'
        seeds = self.config["random_seed_pool"][:self.config["num_seeds"]]
        assert (
                len(self.config["random_seed_pool"]) >= self.config["num_seeds"]
        ), "Not enough random seeds in the pool"
        experiment_dir = self._create_experiment_folder()

        # Initialize main logger (for non-process-specific logging)
        self.logger = self._load_logger(experiment_dir)
        self._print_config(self.config)
        self.logger.info(f"Experiment directory: {experiment_dir}")
        self.copy_to_tmp()

        num_gpus = torch.cuda.device_count()
        print("Number of GPUs detected: ", num_gpus)
        processes = []

        for i, seed in enumerate(seeds):
            gpu_id = i % num_gpus  # Cycle through available GPUs
            p = mp.Process(target=self._train_model_single_seed, args=(seed, gpu_id, experiment_dir))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()  # Ensure all processes complete

        # Finalize main logger and wandb (if any overall logging is needed)
        self.logger.info("Experiment directory: " + experiment_dir)
        self.calculate_metrics_over_seeds(experiment_dir)
