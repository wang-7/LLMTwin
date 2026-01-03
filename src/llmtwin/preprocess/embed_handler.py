from abc import ABC, abstractmethod

from llmtwin.domain.chunk_document import ChunkArticle, ChunkDocument
from llmtwin.domain.embed_document import EmbedDocument, EmbedArticle

from typing import Generic, TypeVar, cast

from llmtwin.networks.embeddings import EmbeddingModelSingleton

ChunkT = TypeVar("ChunkT", bound=ChunkDocument)
EmbeddedChunkT = TypeVar("EmbeddedChunkT", bound=EmbedDocument)

embedding_model = EmbeddingModelSingleton()

class EmbedHandler(ABC, Generic[ChunkT, EmbeddedChunkT]):
    """
    Abstract class for all embedding data handlers.
    All data transformations logic for the embedding step is done here
    """

    def embed(self, data_model: ChunkT) -> EmbeddedChunkT:
        return self.embed_batch([data_model])[0]

    def embed_batch(self, data_model: list[ChunkT]) -> list[EmbeddedChunkT]:
        embedding_model_input = [data_model.content for data_model in data_model]
        embeddings = embedding_model(embedding_model_input, to_list=True)

        embedded_chunk = [
            self.map_model(data_model, cast(list[float], embedding))
            for data_model, embedding in zip(data_model, embeddings, strict=False)
        ]

        return embedded_chunk

    @abstractmethod
    def map_model(self, data_model: ChunkT, embedding: list[float]) -> EmbeddedChunkT:
        pass


class ArticleEmbedHandler(EmbedHandler):
    def map_model(self, data_model: ChunkArticle, embedding: list[float]) -> EmbedArticle:
        return EmbedArticle(
            id=data_model.id,
            content=data_model.content,
            embedding=embedding,
            platform=data_model.platform,
            link=data_model.link,
            document_id=data_model.document_id,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
            },
        )