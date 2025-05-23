from tools.rag.pdf import PDFRetrievalChain

# PDF 문서를 로드합니다.
pdf = PDFRetrievalChain(["data/SPRI_AI_Brief_2023년12월호_F.pdf"]).create_chain()

# retriever 생성
pdf_retriever = pdf.retriever

# chain 생성
pdf_chain = pdf.chain


if __name__ == "__main__":
    res = pdf_retriever.invoke("What is the latest AI Brief Report on DEC 2023?")

    print(res)
