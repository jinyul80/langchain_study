from graph.state import GraphState
from graph.chains.retrieval_grader import retrieval_grader, GradeDocuments
from typing import cast


# 문서 관련성 평가 노드
def grade_documents(state: GraphState):
    print("==== [CHECK DOCUMENT RELEVANCE TO QUESTION] ====")
    # 질문과 문서 검색 결과 가져오기
    question = state["question"]
    documents = state["documents"]

    # 각 문서에 대한 관련성 점수 계산
    filtered_docs = []
    for doc in documents:
        score = cast(
            GradeDocuments,
            retrieval_grader.invoke(
                {"question": question, "document": doc.page_content}
            ),
        )
        grade = score.binary_score
        if grade == "yes":
            # print("---GRADE: DOCUMENT RELEVANT---")
            # 관련성이 있는 문서 추가
            filtered_docs.append(doc)
        else:
            # 관련성이 없는 문서는 건너뛰기
            # print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue

    print(
        f"Check documents: {len(documents)}, Filtered Documents: {len(filtered_docs)}"
    )

    return {"documents": filtered_docs, "question": question}
