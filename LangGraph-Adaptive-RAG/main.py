import streamlit as st
from graph.app import app
from langchain_teddynote.messages import stream_graph
from langchain_core.runnables import RunnableConfig
import uuid

st.set_page_config(page_title="LangGraph RAG", page_icon="📚")
st.header("LangGraph RAG 📚")

# config 설정(재귀 최대 횟수, thread_id)
config = RunnableConfig(recursion_limit=20, configurable={"thread_id": uuid.uuid4()})

# Streamlit UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("질문을 입력해주세요."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("답변을 생성 중입니다..."):
            inputs = {"question": question}
            response = app.invoke(inputs, config)

            # app.invoke의 결과가 딕셔너리 형태일 경우 'answer' 키를 사용하거나,
            # 전체 응답을 문자열로 변환하여 표시합니다.
            if isinstance(response, dict) and "generation" in response:
                full_response = response["generation"]
            else:
                full_response = str(response)  # 다른 형태의 응답을 문자열로 변환

            st.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
