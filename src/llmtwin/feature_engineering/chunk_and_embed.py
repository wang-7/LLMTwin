import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from llmtwin.networks.embeddings import EmbeddingModelSingleton
from typing import TypeVar
from llmtwin.domain.embed_document import EmbedArticle
from llmtwin.preprocess.dispatcher import ChunkDispatcher, EmbedDispatcher
from llmtwin.domain.cleaned_document import CleanedDocument, CleanedArticle
from tqdm import tqdm
from llmtwin.domain.embed_document import EmbedDocument

CleanedDocumentT = TypeVar("CleanedDocumentT", bound=CleanedDocument)
EmbedDocumentT = TypeVar("EmbedDocumentT", bound=EmbedDocument)

def chunk_and_embed(document: CleanedDocumentT | list[CleanedDocumentT]) -> list[EmbedDocumentT]:
    """Chunks and embeds the given document(s).

    Args:
        document (CleanedDocumentT | list[CleanedDocumentT]): The document or list of documents to chunk and embed.
        chunk_size (int): The size of each chunk.
        overlap (int): The overlap between chunks.

    """
    if not isinstance(document, list):
        document = [document]
    
    chunked_documents = []
    for doc in tqdm(document, desc="Chunking documents"):
        ChunkDispatcher.dispatch(doc)
        chunked_documents.extend(ChunkDispatcher.dispatch(doc))
    
    embedded_documents = EmbedDispatcher.dispatch(chunked_documents)
    
    return embedded_documents

if __name__ == "__main__":

    cleaned_doc, _ = CleanedArticle.bulk_find(limit=999, with_payloads=True)
    result = chunk_and_embed(cleaned_doc)
    print(f"Processed {len(result)} embedded documents.")
    EmbedArticle.get_or_create_collection(vector_size= EmbeddingModelSingleton().embedding_size, use_vector_index=True)
    # save result to qdrant
    inset_result = EmbedArticle.bulk_insert(result)
    print(inset_result)