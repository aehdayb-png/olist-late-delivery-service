import logging
from logging.handlers import RotatingFileHandler

from src.config_loader import load_config, resolve_path

FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str = "olist_service") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured, avoid duplicate lines
        return logger

    cfg = load_config()["logging"]
    logger.setLevel(cfg["level"])
    formatter = logging.Formatter(FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_path = resolve_path(cfg["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
