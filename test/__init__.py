import logging

LOGGING_DATE_FMT = "%Y-%m-%d %H:%M:%S"
LOGGING_FMT = "[%(asctime)s %(levelname)7s] %(message)s"

logging.basicConfig(format=LOGGING_FMT, datefmt=LOGGING_DATE_FMT, level=logging.INFO)
