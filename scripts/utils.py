from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


def init_mongo_db(uri: str, db_name: str):
    """Initialize the mongodb database."""

    try:
        assert uri is not None, "No URI passed"
        client = MongoClient(uri, server_api=ServerApi("1"))
        database = client[db_name]
        print("Connected to MongoDB")
        return database
    except Exception as e:
        print("Something went wrong connecting to mongodb")
        return
