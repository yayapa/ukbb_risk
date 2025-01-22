from IPython.display import display, HTML
import logging


class JupyterHandler(logging.StreamHandler):
    def emit(self, record):
        # display(self.format(record))
        formatted_record = self.format(record)
        display(
            HTML(f"<pre>{formatted_record}</pre>")
        )  # Use HTML with <pre> tags for proper rendering


class PreprocessLogger:
    def __init__(self, name, jupyter=True, file_name="preprocess.log"):
        self.logger = logging.getLogger(name)
        if jupyter:
            self.configure_jupyter_logger()
        else:
            self.configure_file_logger(file_name)

    def configure_jupyter_logger(self):
        jupyter_handler = JupyterHandler()
        jupyter_handler.setLevel(logging.DEBUG)

        # format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        format = logging.Formatter("%(message)s")
        jupyter_handler.setFormatter(format)

        self.logger.addHandler(jupyter_handler)
        self.logger.setLevel(logging.DEBUG)

    def configure_file_logger(self, file_path):
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(format)
        console_handler.setFormatter(format)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.setLevel(logging.DEBUG)
