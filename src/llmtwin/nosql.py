from typing import TypeVar, Generic, Type
from pydantic import BaseModel
from pymongo.results import InsertOneResult, InsertManyResult
from abc import ABC, abstractmethod
from .connection import MongoDBConnection
from .settings import settings

_database = MongoDBConnection().get_database(settings.DATABASE_NAME)

T = TypeVar('T', bound="BaseDocument")

class BaseDocument(BaseModel, ABC, Generic[T]):

    @classmethod
    @abstractmethod
    def get_collection_name(cls: Type[T]) -> str:
        pass
    
    @classmethod
    def from_mongo(cls: Type[T], data: dict) -> T:
        return cls(**data)
    
    def to_mongo(self: T, **kwargs) -> dict:
        return self.model_dump(**kwargs)

    @classmethod
    def find_one(cls: Type[T], **filter_options) -> T:
        collection = _database[cls.get_collection_name()]
        result = collection.find_one(filter_options)
        return cls.from_mongo(result)
    
    def insert(self: T) -> InsertOneResult:
        collection = _database[self.get_collection_name()]
        result = collection.insert_one(self.to_mongo())
        return result
    
    @classmethod
    def bulk_find(cls: Type[T], **filter_options) -> list[T]:
        collection = _database[cls.get_collection_name()]
        instances = collection.find(filter_options)
        result = [cls.from_mongo(instance) for instance in instances]
        return result
    
    @classmethod
    def bulk_insert(cls: Type[T], doc_list: list[T], **kwargs) -> InsertManyResult:
        collection = _database[cls.get_collection_name()]
        result = collection.insert_many(doc.to_mongo(**kwargs) for doc in doc_list)
        return result