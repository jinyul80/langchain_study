from graph.state import GraphState
from graph.chains.query_rewriter import question_rewriter


# 질문 재작성 노드
def transform_query(state: GraphState):
    print("==== [TRANSFORM QUERY] ====")
    # 질문과 문서 검색 결과 가져오기
    question = state["question"]
    documents = state["documents"]

    # 질문 재작성
    better_question = question_rewriter.invoke({"question": question})

    return {"documents": documents, "question": better_question}
