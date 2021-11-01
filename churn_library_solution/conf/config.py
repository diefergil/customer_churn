import logging
import logging.config
import pathlib
import sys

from rich.logging import RichHandler

# Directories
PACKAGE_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()
BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()
CONFIG_DIR = pathlib.Path(BASE_DIR, "conf")
LOGS_DIR = pathlib.Path(PACKAGE_ROOT, "logs")
DATA_DIR = pathlib.Path(PACKAGE_ROOT, "data")
MODEL_DIR = pathlib.Path(PACKAGE_ROOT, "models")
RESULTS_DIR = pathlib.Path(PACKAGE_ROOT, "results")
IMAGES_DIR = pathlib.Path(PACKAGE_ROOT, "images")


# Local stores
EDA_IMAGES = pathlib.Path(IMAGES_DIR, "eda")
RESULTS_IMAGES = pathlib.Path(IMAGES_DIR, "result")


# Create dirs
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
EDA_IMAGES.mkdir(parents=True, exist_ok=True)
RESULTS_IMAGES.mkdir(parents=True, exist_ok=True)


# Logger
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": """%(levelname)s %(asctime)s
 [%(filename)s:%(funcName)s:
 %(lineno)d]\n%(message)s\n"""
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": pathlib.Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": pathlib.Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "loggers": {
        "root": {
            "handlers": ["console", "info", "error"],
            "level": logging.INFO,
            "propagate": True,
        },
    },
}
logging.config.dictConfig(logging_config)
logger = logging.getLogger("root")
logger.handlers[0] = RichHandler(markup=True)
