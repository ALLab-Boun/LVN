import logging
import sys


def get_logger():
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler("training.log")
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.DEBUG)

    c_format = logging.Formatter(
        "%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    f_format = logging.Formatter(
        "%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


logger = get_logger()

__all__ = ["get_logger"]
