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


    st.title("CHAT WITH YOR PDF")
    with st.container():
        st.markdown("""
    ####  How to Use the App  

    > 🟦 **Step 1: Upload**  
    > Upload your file or paste your text into the app.  

    > 🟩 **Step 2: Processing**  
    > The app automatically analyzes and processes your input.  

    > 🟨 **Step 3: Explore**  
    > Ask questions, view insights, or generate visualizations.  

    > 🟪 **Step 4: Save / Export**  
    > Download the results or save them for later use.  
    """)

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
        st.write("🚀 Chat with Your PDFs 📄")
    st.divider()
    with st.expander(label="Rag Application Pipeline ⚙️"):
        st.image("Rag_pipeline.jpg", use_container_width=True,caption='RAG Pipline')
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

