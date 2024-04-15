import time

from dotenv import load_dotenv
load_dotenv()
#from langchain_openai.llms import OpenAI
# import streamlit as st
from langchain.llms import CTransformers

# llm = OpenAI()
# result = llm.invoke("내가 좋아하는 동물은 "):
#
# llm = CTransformers(
#     model ="llama-2-7b-chat.ggmlv3.q8_0.bin",
#     model_type ="llama"
# )
#
# # chat_model = ChatOpenAI()
# st.title('인공지능 시인')
# content = st.text_input('시의 주제를 제시해주세요')
#
#
# if st.button('시 작성 요청하기'):
#     with st.spinner('시 작성중 ...'):
#         result = llm.invoke(content + "에 대한 시를 써줘")
#     st.write(result)


# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain_openai.chat_models import ChatOpenAI
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain.chains import RetrievalQA
#
# loader = PyPDFLoader("unsu.pdf")
# pages = loader.load_and_split()
#
# #Split
# text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size = 300,
#     chunk_overlap  = 20,
#     length_function = len,
#     is_separator_regex = False,
# )
# texts = text_splitter.split_documents(pages)
#
# #Embedding
# embeddings_model = OpenAIEmbeddings()
#
# # load it into Chroma
# db = Chroma.from_documents(texts, embeddings_model, persist_directory="chroma")
#
# #Question
# question = "환자가 먹고 싶어하는 음식은 무엇이야?"
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
# qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
# result = qa_chain({"query": question})
# print(result)



# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import RetrievalQA
# import streamlit as st
# import tempfile
# import os
#
# #제목
# st.title("ChatPDF")
# st.write("---")
#
# #파일 업로드
# uploaded_file = st.file_uploader("Choose a file")
# st.write("---")
#
# def pdf_to_document(uploaded_file):
#     temp_dir = tempfile.TemporaryDirectory()
#     temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
#     with open(temp_filepath, "wb") as f:
#         f.write(uploaded_file.getvalue())
#     loader = PyPDFLoader(temp_filepath)
#     pages = loader.load_and_split()
#     return pages
#
# #업로드 되면 동작하는 코드
# if uploaded_file is not None:
#     pages = pdf_to_document(uploaded_file)
#
#     #Split
#     text_splitter = RecursiveCharacterTextSplitter(
#         # Set a really small chunk size, just to show.
#         chunk_size = 300,
#         chunk_overlap  = 20,
#         length_function = len,
#         is_separator_regex = False,
#     )
#     texts = text_splitter.split_documents(pages)
#
#     #Embedding
#     embeddings_model = OpenAIEmbeddings()
#
#     # load it into Chroma
#     db = Chroma.from_documents(texts, embeddings_model)
#
#     #Question
#     st.header("PDF에게 질문해보세요!!")
#     question = st.text_input('질문을 입력하세요')
#
#     if st.button('질문하기'):
#         with st.spinner('Wait for it...'):
#             llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
#             qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
#             result = qa_chain({"query": question})
#             st.write(result["result"])



# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os
from streamlit_extras.buy_me_a_coffee import button
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import WebBaseLoader
from langchain import hub
from llama_cpp import Llama
from langchain.llms import LlamaCpp
import pandas as pd
button(username="ybtour", floating=True, width=221)


#제목
st.title("노랑풍선 ChatFile")
st.write("---")

#OpenAI KEY 입력 받기
openai_key = st.text_input('OPEN_AI_API_KEY', type="password")

webUrl = st.text_input('webUrl 입력 - 크롤링')


# loadfile = CSVLoader(file_path='/content/sample_data/california_housing_test.csv')
# data = loadfile.load()
# print(data)

genre = st.radio(
    "LLm 타입을 선택하세요.",
    [":rainbow[gpt-3.5-turbo]", "llama-2-7b-chat.ggmlv3.q8_0.bin"])

if genre == ':rainbow[gpt-3.5-turbo]':
    st.write('gpt-3.5-turbo 를 선택했습니다.')
else:
    st.write("llama-2-7b-chat.ggmlv3.q8_0.bin 를 선택했습니다.")


#Question
st.header("노랑풍선 ChatFile 에게 질문해보세요!!")
question = st.text_input('질문을 입력하세요')

#Stream 받아 줄 Hander 만들기
from langchain.callbacks.base import BaseCallbackHandler
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text=initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text+=token
        self.container.markdown(self.text)

if st.button('질문하기'):
    with st.spinner('Wait for it...'):
        chat_box = st.empty()
        stream_hander = StreamHandler(chat_box)
        if genre == ':rainbow[gpt-3.5-turbo]' and webUrl != '':
            print('질문하기 시작')
            web_loader = WebBaseLoader([
                webUrl  # LangChain Introduction
                ]
            )

            data = web_loader.load()

            text_splitter2 = RecursiveCharacterTextSplitter(
                chunk_size = 500,
                chunk_overlap = 0
            )

            all_splits = text_splitter2.split_documents(data)
            print(all_splits[0])

            vectorstore = Chroma.from_documents(
                documents=all_splits,
                embedding=OpenAIEmbeddings()
            )

            prompt = hub.pull("rlm/rag-prompt")

            print(prompt)

            model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_key, streaming=True, callbacks=[stream_hander])

            # llm = CTransformers(
            #     model ="llama-2-7b-chat.ggmlv3.q8_0.bin",
            #     model_type ="llama"
            # )
            qa_chain = RetrievalQA.from_chain_type(
                llm=model,
                retriever=vectorstore.as_retriever(),
                chain_type_kwargs={"prompt": prompt}
            )

            qa_chain({"query": question})






