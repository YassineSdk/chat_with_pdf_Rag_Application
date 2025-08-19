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
from streamlit_extras.stylable_container import stylable_container



def main():

    st.set_page_config(page_title="Chat with your pdf",layout="wide",page_icon="🗫")
# loading the Popping font
    st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
        }
    </style>
    """,
    unsafe_allow_html=True
)
    st.markdown("""
    <div style="
        background-color:#4B9CD3; 
        padding:30px 20px; 
        border-radius:12px; 
        text-align:center; 
        color:white;
        box-shadow: 1px 1px 10px rgba(0,0,0,0.15);
        font-family: 'Poppins', sans-serif;
        ">
        <h1 style="font-size:60px; margin-bottom:5px;">PDF Chat Master</h1>
        <p style="font-size:20px; margin-top:0;">The era of talking with your PDFs</p>
    </div>
    """, unsafe_allow_html=True)

    # margin
    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
    # Create 4 horizontal boxes using columns
    col1, col2, col3 = st.columns(3)

    box_style = """
            <div style="
                background-color:#f5f7fa;
                padding:15px 10px;
                border-radius:10px;
                text-align:center;
                box-shadow: 1px 1px 5px rgba(0,0,0,0.1);
                margin:5px;
                font-family: 'Poppins', sans-serif;
                font-size: 22px;
                font-weight: 600;
                line-height: 1.5;
                ">
                {}
            </div>
    """

    with col1:
        st.markdown(box_style.format("🟦<br><b>Step 1</b><br>Upload File"), unsafe_allow_html=True)

    with col2:
        st.markdown(box_style.format("🟩<br><b>Step 2</b><br>Processing"), unsafe_allow_html=True)

    with col3:
        st.markdown(box_style.format("🟨<br><b>Step 3</b><br>Explore your PDF"), unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)

    with stylable_container(
        key="hero-section",
        css_styles="""
            {
                background-image: url('Rag_pipeline.jpg');
                background-size: cover;
                background-position: center;
                border-radius: 20px;
                height: 1px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 1.5rem;
                font-weight: 600;
            }
        """
    ):
        pass

    with st.expander(label="Rag Application Pipeline ⚙️"):
        st.image("Rag_pipeline.jpg", use_container_width=True,caption='RAG Pipline')
    col_1,col_2 = st.columns(2)
    
    #uploading and showing the file 
    with col_1:
        pdf = st.file_uploader(label="appload you pdf",type="pdf")
        if pdf is not None:
            show_pdf(pdf)

    # extracting the pdf text 
    with col_2 :
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

