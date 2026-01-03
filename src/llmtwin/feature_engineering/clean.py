from llmtwin.domain.base.nosql import BaseDocument
from llmtwin.domain.cleaned_document import CleanedArticle, CleanedDocument
from llmtwin.domain.document import ArticleDocument
from llmtwin.preprocess.dispatcher import CleanDispatcher

def clean_documents(
    raw_documents: list[BaseDocument],
) -> list[CleanedDocument]:
    cleaned_documents = []
    for document in raw_documents:
        cleaned_document = CleanDispatcher.dispatch(document)
        cleaned_documents.append(cleaned_document)
    return cleaned_documents


if __name__ == "__main__":
    raw_docs = ArticleDocument.bulk_find({}, by_alias=True)
    cleaned_docs = clean_documents(raw_docs)
    CleanedArticle.get_or_create_collection(use_vector_index=False)
    result = CleanedArticle.bulk_insert(cleaned_docs)
    print(result)