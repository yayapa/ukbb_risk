import logging
import os


class TwoDImageLogger:
    def __init__(self, log_dir=None, log_level=logging.INFO, save_results=True, logger_name="training.log"):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Create handlers
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Create formatters and add them to handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(console_handler)

        # Only create a file handler if save_results is True
        if save_results and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, logger_name))
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger
