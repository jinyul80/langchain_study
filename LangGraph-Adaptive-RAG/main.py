from graph.app import app
from langchain_teddynote.messages import stream_graph
from langchain_core.runnables import RunnableConfig
import uuid

# config 설정(재귀 최대 횟수, thread_id)
config = RunnableConfig(recursion_limit=20, configurable={"thread_id": uuid.uuid4()})

if __name__ == "__main__":

    question = "한국의 수도는 어디인가요? 수도의 인구는 몇명인가요?"
    inputs = {"question": question}

    stream_graph(app, inputs, config)
