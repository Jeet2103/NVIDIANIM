import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

llm = ChatNVIDIA(model = "meta/llama-3.1-405b-instruct")

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./DATA_FILE")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000, chunk_overlap = 200
        )
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)


st.title("NVIDIA NIM Demo")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
Question : {input}

"""
)

if st.button("Document Embedding"):
    vector_embedding()
    st.write("FAISS Vectorstore DB Is Ready Using NvidiaEmbedding")

question = st.chat_input(placeholder="Enter your question from Documents")
if question:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    rag_chain = create_retrieval_chain(retriever,document_chain)
    response = rag_chain.invoke({"input":question})
    st.success(response['answer'])

## With a streamlit expander
    with st.expander("Document Similarity search"):
        ## Find the relevant chunks
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("-----------------------------")