from svs_mongodb_odm.connection import get_db
from typing import List, Dict


def create_counters_collection(counters_collection_name: str, counters: List[Dict]) -> None:
    """
    Create counters collection in the database

    Note that this function will not create the collection if it already exists.

    The counters dictionary should be in the following format:
    {
        "_id": "name of the counter",
        "sequence_value": 4000  # initial value of the counter
    }

    :param counters_collection_name: Name of the counters collection
    :param counters: List of counters. Each counter is a dictionary

    :return: None
    """
    db = get_db()
    if counters_collection_name in db.list_collection_names():
        return
    db[counters_collection_name].insert_many(counters)