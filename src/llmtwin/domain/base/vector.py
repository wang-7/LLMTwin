from xml.dom.minidom import Document
from ...connection import QdrantConnection
from pydantic import BaseModel, UUID4, Field
from abc import ABC, abstractmethod
from typing import Type, Generic, TypeVar
import numpy as np
import uuid
from uuid import UUID

from qdrant_client.http import exceptions
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import CollectionInfo, PointStruct, Record

_database = QdrantConnection()

T = TypeVar('T', bound='VectorBaseDocument')

class VectorBaseDocument(BaseModel, Generic[T], ABC):
    # 注意：由于qdrant不像mongodb那样自动提供id，因此必须手动设置id
    # 由于id是每次实例化时随机生成的，因此不能固定id（会在不同实例共享），而是需要提供一个生成方法。
    id: UUID4=Field(default_factory=uuid.uuid4)

    @classmethod
    @abstractmethod
    def get_collection_name(cls: Type[T]):
        pass

    @classmethod
    def from_record(cls: Type[T], record: Record) -> T:
        # Record 是一个包含 id, score, vector, payload等属性的结构。代表qdrant查询的返回结果。
        payload = record.payload or {}
        vector = record.vector
        # 读取id时从str转换为UUID
        _id = UUID(record.id, version=4)

        attributes = {
            "id": _id,
            **payload
        }

        if 'embedding' in cls.model_fields:
            attributes['embedding'] = vector or None

        return cls(**attributes)
    
    def to_point(self: T) -> PointStruct:
        attributes = self.model_dump()

        # 生成id时，从UUID转换为str
        _id = str(attributes.pop('id'))
        vector = attributes.pop('embedding', {})
        if isinstance(vector, np.ndarray):
            vector.tolist()

        return PointStruct(id=_id, vector=vector, payload=attributes)
    
    @classmethod
    def bulk_insert(cls: Type[T], documents: list[T]) -> bool:
        collection_name = cls.get_collection_name()
        points = [doc.to_point() for doc in documents]

        _database.upsert(
            collection_name=collection_name,
            points=points,
        )
        return True
    
    @classmethod
    def bulk_find(cls: Type[T], limit: int=10, **kwargs) -> tuple[list[T], UUID4 | None]:
        collection_name = cls.get_collection_name()
        # offset是用id表示，因此要类似id那样做转换
        offset = kwargs.pop('offset', None)
        offset = str(offset) if offset else None

        results, next_offset = _database.scroll(
            collection_name=collection_name,
            limit=limit,
            with_vectors=kwargs.pop('with_vectors', False),
            with_payload=kwargs.pop('with_payloads', True),
            offset=offset,
            **kwargs
        )

        documents = [cls.from_record(record) for record in results]
        if next_offset is not None:
            next_offset = UUID(next_offset, version=4)
        
        return documents, next_offset
    
    @classmethod
    def query(cls: Type[T], query_vector: list, limit: int=10, **kwargs) -> list[T]:
        collection_name = cls.get_collection_name()
        results = _database.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=kwargs.pop('with_payloads', True),
            with_vectors=kwargs.pop('with_vectors', False),
            **kwargs
        )

        documents = [cls.from_record(doc) for doc in results.points]
        return documents