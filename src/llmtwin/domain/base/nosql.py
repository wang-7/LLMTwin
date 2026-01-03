import uuid
from typing import TypeVar, Generic, Type
from pydantic import BaseModel, Field, UUID4
from pymongo.results import InsertOneResult, InsertManyResult
from bson.codec_options import CodecOptions
from bson.binary import UuidRepresentation
from abc import ABC, abstractmethod
from ...connection import MongoDBConnection
from ...settings import settings

_database = MongoDBConnection(uuidRepresentation='standard').get_database(settings.DATABASE_NAME)

T = TypeVar('T', bound="BaseDocument")

class BaseDocument(BaseModel, ABC, Generic[T]):
    id: UUID4= Field(default_factory=uuid.uuid4, alias='_id')

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, self.__class__):
            return False

        return self.id == value.id

    def __hash__(self) -> int:
        return hash(self.id)
    
    @classmethod
    @abstractmethod
    def get_collection_name(cls: Type[T]) -> str:
        pass
    
    @classmethod
    @abstractmethod
    def get_category(cls: Type[T]) -> str:
        pass

    @classmethod
    def from_mongo(cls: Type[T], data: dict, **kwargs) -> T:
        return cls.model_validate(data, **kwargs)
    
    def to_mongo(self: T, **kwargs) -> dict:
        return self.model_dump(**kwargs)

    @classmethod
    def find_one(cls: Type[T], filter_options: dict, **kwargs) -> T:
        collection = _database[cls.get_collection_name()]
        result = collection.find_one(filter_options)
        return cls.from_mongo(result, **kwargs)
    
    def insert(self: T, **kwargs) -> InsertOneResult:
        collection = _database[self.get_collection_name()]
        result = collection.insert_one(self.to_mongo(**kwargs))
        return result
    
    @classmethod
    def bulk_find(cls: Type[T], filter_options: dict, **kwargs) -> list[T]:
        collection = _database[cls.get_collection_name()]
        instances = collection.find(filter_options)
        result = [cls.from_mongo(instance, **kwargs) for instance in instances]
        return result
    
    @classmethod
    def bulk_insert(cls: Type[T], doc_list: list[T], **kwargs) -> InsertManyResult:
        collection = _database[cls.get_collection_name()]
        result = collection.insert_many(doc.to_mongo(**kwargs) for doc in doc_list)
        return result