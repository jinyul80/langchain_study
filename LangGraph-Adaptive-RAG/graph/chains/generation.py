from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from graph.consts import MODEL_NAME

from dotenv import load_dotenv

load_dotenv()


# LangChain Hub에서 프롬프트 가져오기(RAG 프롬프트는 자유롭게 수정 가능)
prompt = hub.pull("teddynote/rag-prompt")

# LLM 초기화 및 함수 호출을 통한 구조화된 출력 생성
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)

# RAG 체인 생성
genaration_chain = prompt | llm | StrOutputParser()
