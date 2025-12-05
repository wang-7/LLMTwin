from pymongo import MongoClient

uri = "mongodb://localhost:27017/"

class MongoDBConnection:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            try:
                cls._instance = MongoClient(uri, connect=True)
            except Exception as e:
                print(f'Exception {e} occurred when connecting to MongoDB.')
        return cls._instance
