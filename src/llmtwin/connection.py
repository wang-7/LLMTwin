from pymongo import MongoClient
from qdrant_client import QdrantClient

mongo_uri = "mongodb://localhost:27017/"
qdrant_url = "http://localhost:6333"

class MongoDBConnection:
    _instance = None
    def __new__(cls,*args, **kwargs):
        if cls._instance is None:
            try:
                # print(kwargs)
                cls._instance = MongoClient(mongo_uri, connect=True, **kwargs)
            except Exception as e:
                print(f'Exception {e} occurred when connecting to MongoDB.')
        return cls._instance

class QdrantConnection:
    _instance=None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            try:
                cls._instance = QdrantClient(url=qdrant_url)
            except Exception as e:
                print(f'Exception {e} occurred when connecting to MongoDB.')
        return cls._instance