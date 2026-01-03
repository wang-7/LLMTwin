from abc import ABC
from llmtwin.domain.base.vector import VectorBaseDocument
from llmtwin.domain.data_category import DataCategory
from pydantic import UUID4, Field


class ChunkDocument(VectorBaseDocument, ABC):
    content: str
    platform: str
    author_id: UUID4
    document_id: UUID4
    author_full_name: str
    metadata: dict = Field(default_factory=dict)

class ChunkArticle(ChunkDocument):
    link: str

    @classmethod
    def get_collection_name(cls) -> str:
        return "chunk_articles"
    
    @classmethod
    def get_category(cls) -> str:
        return DataCategory.Article