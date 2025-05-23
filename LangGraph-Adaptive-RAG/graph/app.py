from dotenv import load_dotenv

load_dotenv()

import os

print(f"Current working directory: {os.getcwd()}")


from graph.state import GraphState
from graph.chains.router import question_router, RouteQuery
from graph.chains.hallucination_grader import hallucination_grader, GradeHallucinations
from graph.chains.answer_grader import answer_grader, GradeAnswer
from graph.consts import (
    WEB_SEARCH,
    RETRIEVE,
    GRADE_DOCUMENTS,
    GENERATE,
    TRANSFORM_QUERY,
)

from graph.nodes import web_search, generate, grade_documents, transform_query, retrieve


from typing import cast
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver


# 질문 라우팅 분기 함수
def route_question(state: GraphState):
    print("==== [ROUTE QUESTION] ====")
    # 질문 가져오기
    question = state["question"]
    # 질문 라우팅
    source = cast(RouteQuery, question_router.invoke({"question": question}))

    # 질문 라우팅 결과에 따른 노드 라우팅
    if source.datasource == "web_search":
        print("==== [ROUTE QUESTION TO WEB SEARCH] ====")
        return "web_search"

    elif source.datasource == "vectorstore":
        print("==== [ROUTE QUESTION TO VECTORSTORE] ====")
        return "vectorstore"


# 문서 관련성 평가 함수
def decide_to_generate(state):
    print("==== [DECISION TO GENERATE] ====")
    # 질문과 문서 검색 결과 가져오기
    question = state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # 모든 문서가 관련성 없는 경우 질문 재작성
        print(
            "==== [DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY] ===="
        )
        return "transform_query"

    else:
        # 관련성 있는 문서가 있는 경우 답변 생성
        print("==== [DECISION: GENERATE] ====")
        return "generate"


# 환각 평가 함수
def hallucination_check(state):
    print("==== [CHECK HALLUCINATIONS] ====")
    # 질문과 문서 검색 결과 가져오기
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # 환각 평가
    score = cast(
        GradeHallucinations,
        hallucination_grader.invoke({"documents": documents, "generation": generation}),
    )
    grade = score.binary_score

    # Hallucination 통과 여부 확인
    if grade == "yes":
        print("==== [DECISION: GENERATION IS GROUNDED IN DOCUMENTS] ====")

        # 답변의 관련성(Relevance) 평가
        print("==== [GRADE GENERATED ANSWER vs QUESTION] ====")
        score = cast(
            GradeAnswer,
            answer_grader.invoke({"question": question, "generation": generation}),
        )
        grade = score.binary_score

        # 관련성 평가 결과에 따른 처리
        if grade == "yes":
            print("==== [DECISION: GENERATED ANSWER ADDRESSES QUESTION] ====")
            return "relevant"

        else:
            print("==== [DECISION: GENERATED ANSWER DOES NOT ADDRESS QUESTION] ====")
            return "not relevant"

    else:
        print("==== [DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY] ====")
        return "hallucination"


# 그래프 상태 초기화
workflow = StateGraph(GraphState)

# 노드 정의
workflow.add_node(WEB_SEARCH, web_search)  # 웹 검색
workflow.add_node(RETRIEVE, retrieve)  # 문서 검색
workflow.add_node(GRADE_DOCUMENTS, grade_documents)  # 문서 평가
workflow.add_node(GENERATE, generate)  # 답변 생성
workflow.add_node(TRANSFORM_QUERY, transform_query)  # 쿼리 변환

# 그래프 빌드
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": WEB_SEARCH,  # 웹 검색으로 라우팅
        "vectorstore": RETRIEVE,  # 벡터스토어로 라우팅
    },
)
workflow.add_edge(WEB_SEARCH, GENERATE)  # 웹 검색 후 답변 생성
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)  # 문서 검색 후 평가
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        "transform_query": TRANSFORM_QUERY,  # 쿼리 변환 필요
        "generate": GENERATE,  # 답변 생성 가능
    },
)
workflow.add_edge(TRANSFORM_QUERY, RETRIEVE)  # 쿼리 변환 후 문서 검색
workflow.add_conditional_edges(
    GENERATE,
    hallucination_check,
    {
        "hallucination": GENERATE,  # Hallucination 발생 시 재생성
        "relevant": END,  # 답변의 관련성 여부 통과
        "not relevant": TRANSFORM_QUERY,  # 답변의 관련성 여부 통과 실패 시 쿼리 변환
    },
)

# 그래프 컴파일
app = workflow.compile(checkpointer=MemorySaver())


if __name__ == "__main__":

    # Mermaid 형식의 PNG 이미지로 변환하여 표시
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")
