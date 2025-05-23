from graph.state import GraphState
from tools.web_search import web_search_tool
from langchain_core.documents import Document


# 웹 검색 노드
def web_search(state: GraphState):
    print("==== [WEB SEARCH] ====")
    # 질문과 문서 검색 결과 가져오기
    question = state["question"]

    # 웹 검색 수행
    web_results = web_search_tool.invoke({"query": question})
    web_results_docs = [
        Document(
            page_content=web_result["content"],
            metadata={"source": web_result["url"]},
        )
        for web_result in web_results
    ]

    return {"documents": web_results_docs, "question": question}
