from abc import ABC, abstractmethod

from llmtwin.domain.base.nosql import BaseDocument
from llmtwin.domain.cleaned_document import CleanedArticle, CleanedDocument
from llmtwin.domain.document import ArticleDocument
from llmtwin.preprocess.clean_func import clean_text

class CleanHandler(ABC):

    @abstractmethod
    def clean(self, data_model: BaseDocument) -> CleanedDocument:
        pass

class ArticleCleanHandler(CleanHandler):
    def clean(self, data_model: ArticleDocument) -> CleanedArticle:
        # Implement the cleaning logic for ArticleDocument here
        valid_content = [content for content in data_model.content.values() if content]
        return CleanedArticle(
            id=data_model.id,
            content=clean_text('#### '.join(valid_content)),
            platform=data_model.platform,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            link=data_model.link
        )