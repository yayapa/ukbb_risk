import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from TwoDImage.image_classification.trainer import Trainer


def run():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(
        description="Run the UKBB Risk Assessment training script with a specified config file."
    )

    # Add the -c / --config argument
    parser.add_argument(
        '-c', '--config',
        type=str,
        default="/u/home/sdm/GitHub/ukbb_risk_assessment/TwoDImage/configs/config_3d.json",
        help='Path to the configuration JSON file. Default is config_3d.json'
    )

    # Parse the arguments
    args = parser.parse_args()

    # Retrieve the config file path from the arguments
    config_file_path = args.config

    # Check if the config file exists
    if not os.path.isfile(config_file_path):
        print(f"Error: The configuration file '{config_file_path}' does not exist.")
        sys.exit(1)

    print(f"Using configuration file: {config_file_path}")  # Optional: For debugging/logging

    # Initialize the Trainer with the config file and start training
    trainer = Trainer(config_file_path)
    trainer.train_models()

if __name__ == "__main__":
    run()
