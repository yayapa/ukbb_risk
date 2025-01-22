import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from TwoDImage.image_classification.trainer import Trainer


def run():
    config_file_path = (
        "/u/home/sdm/GitHub/ukbb_risk_assessment/TwoDImage/configs/config_3d.json"
    )

    trainer = Trainer(config_file_path)
    trainer.train_models_parallel()


if __name__ == "__main__":
    run()
