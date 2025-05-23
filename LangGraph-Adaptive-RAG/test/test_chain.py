from dotenv import load_dotenv

load_dotenv()

from graph.chains.retriever import pdf_retriever
from graph.chains.generation import genaration_chain
from utils.messages import format_docs


def test_rag_chain():

    question = "삼성전자가 만든 생성형 AI 의 이름은?"

    # 질문에 대한 관련 문서 검색
    docs = pdf_retriever.invoke(question)

    formatted_docs = format_docs(docs)

    res = genaration_chain.invoke({"context": formatted_docs, "question": question})

    print(res)

    assert isinstance(res, str)
