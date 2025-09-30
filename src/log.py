import logging
import logging.config
import os


def configure_logging() -> None:
    """
    Configures logging for the script.
    """
    debug = bool(os.environ.get("DEBUG"))
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "root": {
            "level": "DEBUG" if debug else "INFO",
            "handlers": ["console"],
        },
        "formatters": {
            "default": {"format": "%(asctime)s %(levelname)s - %(message)s"},
        },
        "handlers": {
            "console": {
                "level": "DEBUG",
                "class": "logging.StreamHandler",
                "formatter": "default",
            }
        },
        "loggers": {},
    }
    logging.config.dictConfig(logging_config)
