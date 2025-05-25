from .base import RetrievalChain
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Annotated


class PDFRetrievalChain(RetrievalChain):
    def __init__(self, source_uri: Annotated[list[str], "Source URI"]):

        # 부모 클래스 생성자 실행
        super().__init__()

        self.source_uri = source_uri
        self.k = 10

    def load_documents(self, source_uris: List[str]):
        docs: list[Document] = []
        for source_uri in source_uris:
            loader = PDFPlumberLoader(source_uri)
            docs.extend(loader.load())

        return docs

    def create_text_splitter(self):
        return RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
