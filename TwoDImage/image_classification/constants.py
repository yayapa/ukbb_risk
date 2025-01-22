from enum import Enum


class Constants(Enum):
    """
    Enumeration class to define project path variables
    """

    PROJECT_DIR = ""  # Directory with the current repository
    # PROJECT_DIR = "/u/home/sdm/GitHub/ukbb_risk_assessment/"  # Directory with the current repository
    # STORE_DIR = "/home/dmitrii/GitHub/ukbb_risk_assessment/saved/"  # Directory for the file storing
    STORE_DIR = "/vol/aimspace/users/sdm/Projects/ukbb_risk_assessment/saved/"
    # DATA_DIR = "/home/dmitrii/GitHub/ukbb_risk_assessment/data/data/whole_body/projections/"  # Directory for input data
    TARGET_NAME = "event"
    WANDB_DIR = "/vol/aimspace/users/sdm/Projects/ukbb_risk_assessment/wandb/"
    #WANDB_DIR = "/home/dmitrii/GitHub/ukbb_risk_assessment/wandb/"
    #WANDB_PROJECT = "ukbb_risk_assessment_tabular"
    WANDB_PROJECT = "ukbb_risk_assessment"
    TMP_DIR = "/tmp/"
