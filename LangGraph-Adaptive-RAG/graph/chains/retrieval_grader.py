from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from graph.chains.retriever import pdf_retriever
from graph.consts import MODEL_NAME
from typing import cast


# 문서 평가를 위한 데이터 모델 정의
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# LLM 초기화 및 함수 호출을 통한 구조화된 출력 생성
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# 시스템 메시지와 사용자 질문을 포함한 프롬프트 템플릿 생성
system_prompt = """
You are a grader assessing relevance of a retrieved document to a user question. \n 
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# 문서 검색결과 평가기 생성
retrieval_grader = grade_prompt | structured_llm_grader

if __name__ == "__main__":

    # 사용자 질문 설정
    question = "삼성전자가 만든 생성형 AI 의 이름은?"

    # 질문에 대한 관련 문서 검색
    docs = pdf_retriever.invoke(question)

    # 검색된 문서의 내용 가져오기
    retrieved_doc = docs[1].page_content

    print("\nRetrieved document: \n\n", retrieved_doc)

    # 평가 결과 출력
    print("\n" + "=" * 80)

    res = cast(
        GradeDocuments,
        retrieval_grader.invoke({"question": question, "document": retrieved_doc}),
    )
    print(res)
