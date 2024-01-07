import logging


sort_handler = logging.StreamHandler()
fmt = "%(asctime)-18s %(levelname)-8s: %(message)s"
fmt_date = '%Y-%m-%d %T'
formatter = logging.Formatter(fmt, fmt_date)
sort_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(sort_handler)

file_handler = logging.FileHandler("logging.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)