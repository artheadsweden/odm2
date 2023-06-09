import logging

from pymongo import MongoClient
from pymongo.database import Database

from .exceptions import ConnectionExist

__connection = {}
logger = logging.getLogger(__name__)


def _get_connection_client(url: str) -> MongoClient:
    return MongoClient(url)


def connect(url: str) -> MongoClient:
    if "client" in __connection:
        logger.info("Already have an connection.")
        return __connection["client"]
    __connection["url"] = url
    __connection["client"] = _get_connection_client(url)
    logger.info("Connection established successfully")
    return __connection["client"]


def disconnect() -> bool:
    if "client" in __connection:
        __connection["client"].close()
        del __connection["client"]
    else:
        logger.warn("No client connection found")
    if "url" in __connection:
        del __connection["url"]
    else:
        logger.info("No connection URL found.")
    logger.info("Disconnect the db connection")
    return True


def _get_client() -> MongoClient:
    if not __connection or "client" not in __connection:
        if "url" in __connection:
            connect(__connection["url"])
        else:
            raise ConnectionExist("DB connection is not provided")
    return __connection["client"]


def get_db() -> Database:
    return _get_client().get_database()
