from abc import ABC, abstractmethod
from llmtwin.domain.chunk_document import ChunkDocument
from llmtwin.domain.cleaned_document import CleanedDocument
from llmtwin.domain.cleaned_document import CleanedArticle
from llmtwin.domain.chunk_document import ChunkArticle
from typing import Generic, TypeVar
import hashlib
from uuid import UUID
from llmtwin.preprocess.chunk_func import chunk_article


CleanedDocumentT = TypeVar("CleanedDocumentT", bound=CleanedDocument)
ChunkDocumentT = TypeVar("ChunkDocumentT", bound=ChunkDocument)

class ChunkHandler(ABC, Generic[CleanedDocumentT, ChunkDocumentT]):
    @property
    @abstractmethod
    def metadata(self) -> dict:
        pass

    @abstractmethod
    def chunk(self, data_model: CleanedDocumentT) -> list[ChunkDocumentT]:
        pass

class ArticleChunkHandler(ChunkHandler):
    @property
    def metadata(self) -> dict:
        return {
            "min_length": 1000,
            "max_length": 2000,
        }

    def chunk(self, data_model: CleanedArticle) -> list[ChunkArticle]:
        data_models_list = []

        cleaned_content = data_model.content
        chunks = chunk_article(
            cleaned_content, min_length=self.metadata["min_length"], max_length=self.metadata["max_length"]
        )

        for chunk in chunks:
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            model = ChunkArticle(
                id=UUID(chunk_id, version=4),
                content=chunk,
                platform=data_model.platform,
                link=data_model.link,
                document_id=data_model.id,
                author_id=data_model.author_id,
                author_full_name=data_model.author_full_name,
                metadata=self.metadata,
            )
            data_models_list.append(model)

        return data_models_list