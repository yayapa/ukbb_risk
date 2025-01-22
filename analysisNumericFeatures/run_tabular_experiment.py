import json
import sys

sys.path.append("../")  # Adjust the path accordingly
import warnings

warnings.filterwarnings("ignore")

from DataAnalysis.EventAnalyzer import EventAnalyzer
from DataAnalysis.EventAnalyzerSurvival import EventAnalyzerSurvival
from PrepareDataset.DataEncoder.FeatureCollector import FeatureCollector

from PrepareDataset.DataEncoder.PreprocessLogger import PreprocessLogger


class TabularRunner:
    def __init__(self, config):
        self.config = config
        self.logger = PreprocessLogger(
            PreprocessLogger.__name__,
            jupyter=False,
            file_name=config["file_name_config"],
        ).logger

    def _print_config(self, config):
        self.logger.info("Config:")
        self.logger.info("-" * 15)
        for key, value in config.items():
            self.logger.info(str(key) + ": " + str(value))
        self.logger.info("-" * 15)

    def run(self):
        self._print_config(self.config)
        feature_collector = FeatureCollector(
            label_path=(self.config["cohort_path"] + "labels.csv"), logger=self.logger
        )
        file_path_to_features = self.config["cohort_path"] + "/preprocessed_features/"
        feature_collector.load_features(
            data_showcase_path=self.config["data_showcase_path"],
            file_path_to_features=file_path_to_features,
        )

        for feature_set_el in self.config["feature_set"]:
            # features = feature_collector.get_features({"basic_features", feature_set_el})
            features = feature_collector.get_features({feature_set_el})
            self.logger.info(f"START: Feature set processing {feature_set_el}")

            ca = EventAnalyzer(features, self.logger)

            ca.run_default_pipeline(self.config["settings_analyzer"])

            self.logger.info(f"FINISH: Feature set processing  {feature_set_el}")

    def run_survival(self):
        self._print_config(self.config)
        feature_collector = FeatureCollector(
            label_path=(self.config["cohort_path"] + "labels.csv"), logger=self.logger
        )
        file_path_to_features = self.config["cohort_path"] + "/preprocessed_features/"
        feature_collector.load_features(
            data_showcase_path=self.config["data_showcase_path"],
            file_path_to_features=file_path_to_features,
        )

        for feature_set_el in self.config["feature_set"]:
            # features = feature_collector.get_features({"basic_features", feature_set_el})
            features = feature_collector.get_features({feature_set_el})
            self.logger.info(f"START: Feature set processing {feature_set_el}")

            ca = EventAnalyzerSurvival(features, self.logger)
            try:
                ca.run_default_pipeline(self.config["settings_analyzer"])
            except Exception as e:
                self.logger.error(
                    f"Error in feature set processing {feature_set_el}: {e}"
                )

            self.logger.info(f"FINISH: Feature set processing  {feature_set_el}")


if __name__ == "__main__":
    with open("./resources/configs/config_pancreas.json", "r") as f:
        config = json.load(f)

    tr = TabularRunner(config)
    tr.run()
    # tr.run_survival()
