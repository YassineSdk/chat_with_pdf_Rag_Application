import streamlit as st 
from PyPDF2 import PdfReader
from functions import show_pdf , get_text , text_preproccesing , getting_answer
import base64
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI



def main():
    st.set_page_config(page_title="Chat with your pdf",layout="wide")
    st.header("ASK YOUR PDF 🗫 ")
    st.divider()
    col1,col2 = st.columns(2)
    
    #uploading and showing the file 
    with col1:
        pdf = st.file_uploader(label="appload you pdf",type="pdf")
        if pdf is not None:
            show_pdf(pdf)

    # extracting the pdf text 
    with col2 :
        if pdf is not None:
            text = get_text(pdf)
            if "knowledge_base" not in st.session_state:
                st.session_state.knowledge_base = text_preproccesing(text)
            user_question = st.text_input("ask a question about ur pdf")
            ask_question = st.button(label="ask your question")
            if user_question:
                getting_answer(st.session_state.knowledge_base, user_question)


if __name__ == "__main__":
    main()

