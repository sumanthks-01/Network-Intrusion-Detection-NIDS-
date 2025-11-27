from loguru import logger
import sys

def configure_logging():
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add("logs/app.log", rotation="10 MB", retention="14 days")
