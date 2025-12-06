# logger_setup.py
import logging
import logging.handlers
import pathlib
import sys


def setup_logger(name: str = "bot") -> logging.Logger:
    logs_dir = pathlib.Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_fmt = "%(asctime)s [%(levelname)s] %(message)s"

    file_handler = logging.handlers.TimedRotatingFileHandler(
        logs_dir / f"{name}.log",
        when="midnight",
        backupCount=14,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(log_fmt))
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_fmt))
    console_handler.setLevel(logging.INFO)

    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

    logger = logging.getLogger(name)
    return logger
