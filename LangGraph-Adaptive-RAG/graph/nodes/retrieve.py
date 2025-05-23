from graph.state import GraphState
from graph.chains.retriever import pdf_retriever


# 문서 검색 노드
def retrieve(state: GraphState):
    print("==== [RETRIEVE] ====")
    question = state["question"]

    # 문서 검색 수행
    documents = pdf_retriever.invoke(question)

    return {"documents": documents, "question": question}
