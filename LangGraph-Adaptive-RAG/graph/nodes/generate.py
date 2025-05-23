from graph.state import GraphState
from graph.chains.generation import genaration_chain


# 답변 생성 노드
def generate(state: GraphState):
    print("==== [GENERATE] ====")
    # 질문과 문서 검색 결과 가져오기
    question = state["question"]
    documents = state["documents"]

    # RAG 답변 생성
    generation = genaration_chain.invoke({"context": documents, "question": question})

    return {"documents": documents, "question": question, "generation": generation}
