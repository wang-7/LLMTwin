from llmtwin.domain.base.nosql import BaseDocument
from llmtwin.domain.data_category import DataCategory
from pydantic import UUID4

class ArticleDocument(BaseDocument):
    content: dict
    platform: str
    author_id: UUID4
    author_full_name: str
    link: str

    @classmethod
    def get_collection_name(cls) -> str:
        return "documents"
    
    @classmethod
    def get_category(cls) -> str:
        return DataCategory.Article
    
