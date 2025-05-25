from typing import Any
from langchain_core.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from abc import ABC, abstractmethod
from operator import itemgetter
from langchain import hub
from langchain_text_splitters import TextSplitter
import os


class RetrievalChain(ABC):
    def __init__(self):
        self.source_uri = None
        self.k = 10

        # 벡터 저장소 경로 설정
        self.vector_db_path = "./data/db/"

    @abstractmethod
    def load_documents(self, source_uris) -> list[Any]:
        """loader를 사용하여 문서를 로드합니다."""
        pass

    @abstractmethod
    def create_text_splitter(self) -> TextSplitter:
        """text splitter를 생성합니다."""
        pass

    def split_documents(self, docs, text_splitter):
        """text splitter를 사용하여 문서를 분할합니다."""
        return text_splitter.split_documents(docs)

    def create_embedding(self):
        return OllamaEmbeddings(model="nomic-embed-text")

    def create_vectorstore(self):
        docs = self.load_documents(self.source_uri)
        text_splitter = self.create_text_splitter()
        split_docs = self.split_documents(docs, text_splitter)

        # 디렉토리가 없으면 생성
        if not os.path.exists(self.vector_db_path):
            os.makedirs(self.vector_db_path)

        # FAISS 벡터 저장소 생성 및 로컬에 저장
        vectorstore = FAISS.from_documents(
            documents=split_docs, embedding=self.create_embedding()
        )
        vectorstore.save_local(self.vector_db_path)
        print("Vector store created and saved locally.")

    def load_vectorstore(self):
        # 경로가 없으면 vectorstore를 생성합니다.
        if not os.path.exists(self.vector_db_path):
            self.create_vectorstore()

        # 저장된 벡터 저장소 로드
        loaded_vectorstore = FAISS.load_local(
            self.vector_db_path,
            self.create_embedding(),
            allow_dangerous_deserialization=True,
        )
        return loaded_vectorstore

    def create_retriever(self, vectorstore):
        # MMR을 사용하여 검색을 수행하는 retriever를 생성합니다.
        dense_retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": self.k}
        )
        return dense_retriever

    def create_model(self):
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    def create_prompt(self):
        return hub.pull("teddynote/rag-prompt-chat-history")

    @staticmethod
    def format_docs(docs):
        return "\n".join(docs)

    def create_chain(self):
        self.vectorstore = self.load_vectorstore()
        self.retriever = self.create_retriever(self.vectorstore)
        model = self.create_model()
        prompt = self.create_prompt()
        self.chain = (
            {
                "question": itemgetter("question"),
                "context": itemgetter("context"),
                "chat_history": itemgetter("chat_history"),
            }
            | prompt
            | model
            | StrOutputParser()
        )
        return self
