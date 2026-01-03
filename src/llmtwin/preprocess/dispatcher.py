from llmtwin.domain.base.nosql import BaseDocument
from llmtwin.domain.cleaned_document import CleanedDocument
from llmtwin.domain.chunk_document import ChunkDocument
from llmtwin.domain.embed_document import EmbedDocument
from llmtwin.domain.data_category import DataCategory
from llmtwin.preprocess.clean_handler import ArticleCleanHandler, CleanHandler
from llmtwin.preprocess.chunk_handler import ArticleChunkHandler, ChunkHandler, CleanedDocumentT
from llmtwin.preprocess.embed_handler import ArticleEmbedHandler, EmbedHandler
from typing import Generic, TypeVar
from loguru import logger

DocumentT = TypeVar("DocumentT", bound=BaseDocument)
CleanedDocumentT = TypeVar("CleanedDocumentT", bound=CleanedDocument)
ChunkDocumentT = TypeVar("ChunkDocumentT", bound=ChunkDocument)
EmbedDocumentT = TypeVar("EmbedDocumentT", bound=EmbedDocument)

class CleanHandlerFactory:
    @staticmethod
    def create_handler(data_category: DataCategory) -> CleanHandler:
        if data_category == DataCategory.Article:
            return ArticleCleanHandler()
        else:
            raise ValueError("Unsupported data type")

class CleanDispatcher(Generic[DocumentT, CleanedDocumentT]):
    factory = CleanHandlerFactory()

    @ classmethod
    def dispatch(cls, data_model: DocumentT) -> CleanedDocumentT:
        data_category = data_model.get_category()
        handler = cls.factory.create_handler(data_category)
        clean_model = handler.clean(data_model)

        logger.info(
            "Document cleaned successfully.",
            data_category=data_category,
            cleaned_content_len=len(clean_model.content),
        )

        return clean_model
    
class ChunkHandlerFactory:
    @staticmethod
    def create_handler(data_category: DataCategory) -> ChunkHandler:
        if data_category == DataCategory.Article:
            return ArticleChunkHandler()
    
class ChunkDispatcher(Generic[CleanedDocumentT, ChunkDocumentT]):
    factory = ChunkHandlerFactory()

    @ classmethod
    def dispatch(cls, cleaned_document: CleanedDocumentT) -> list[ChunkDocumentT]:
        data_category = cleaned_document.get_category()
        handler = cls.factory.create_handler(data_category)
        chunked_documents = handler.chunk(cleaned_document)
        # logger.info(
        #     "Document chunked successfully.",
        #     num=len(chunked_documents),
        #     data_category=data_category,
        # )

        return chunked_documents
    
class EmbedHandlerFactory:
    @staticmethod
    def create_handler(data_category: DataCategory) -> EmbedHandler:
        if data_category == DataCategory.Article:
            return ArticleEmbedHandler()
    
class EmbedDispatcher(Generic[ChunkDocumentT, EmbedDocumentT]):
    factory = EmbedHandlerFactory()

    @ classmethod
    def dispatch(cls, chunked_document: list[ChunkDocumentT]) -> list[EmbedDocumentT]:
        is_list = isinstance(chunked_document, list)
        if not is_list:
            chunked_document = [chunked_document]

        if len(chunked_document) == 0:
            return []

        data_category = chunked_document[0].get_category()
        assert all(
            data_model.get_category() == data_category for data_model in chunked_document
        ), "Data models must be of the same category."
        handler = cls.factory.create_handler(data_category)

        embedded_chunk_model = handler.embed_batch(chunked_document)

        if not is_list:
            embedded_chunk_model = embedded_chunk_model[0]

        logger.info(
            "Data embedded successfully.",
            data_category=data_category,
        )

        return embedded_chunk_model
