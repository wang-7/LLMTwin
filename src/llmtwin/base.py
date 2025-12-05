from typing import TypeVar, Generic, Type
from pydantic import BaseModel
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
    def find_one(cls: Type[T], **filter_options) -> T:
        collection = _database[cls.get_collection_name()]
        result = collection.find_one(filter_options)
        return result

    def insert():
        pass