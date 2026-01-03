from llmtwin.domain.base.vector import VectorBaseDocument
from llmtwin.domain.data_category import DataCategory
from pydantic import UUID4
from abc import ABC

class CleanedDocument(VectorBaseDocument, ABC):
    content: str
    platform: str
    author_id: UUID4
    author_full_name: str

class CleanedArticle(CleanedDocument):
    link: str

    @classmethod
    def get_collection_name(cls) -> str:
        return "cleaned_articles"
    
    @classmethod
    def get_category(cls) -> str:
        return DataCategory.Article