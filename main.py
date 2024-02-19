# from dotenv import load_dotenv
# load_dotenv()
#from langchain_openai.llms import OpenAI
from langchain_openai.chat_models import ChatOpenAI
import streamlit as st

# llm = OpenAI()
# result = llm.invoke("내가 좋아하는 동물은 ")


chat_model = ChatOpenAI()
st.title('인공지능 시인')
content = st.text_input('시의 주제를 제시해주세요')


if st.button('시 작성 요청하기'):
    with st.spinner('시 작성중 ...'):
        result = chat_model.invoke(content + "에 대한 시를 써줘")
    st.write(result)




