import streamlit as st 
from PyPDF2 import PdfReader
import base64
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain_community.llms import Ollama


def show_pdf(pdf):
        base64_pdf = base64.b64encode(pdf.read()).decode("utf-8")
        pdf = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="700"></iframe>'
        with st.expander("📑 View PDF"):
            st.markdown(pdf, unsafe_allow_html=True)


def get_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
            text += page.extract_text()
    return text



def text_preproccesing(text):
    """
    splitting the pdf text into chunks and then embedding this chunks in 
                    order to create a Knowledge base
    """
    # splitting the data 
    with st.spinner("🤖 preproccessing the pdf file..."):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200 ,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # embedding the chunks
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embedder = HuggingFaceEmbeddings(model_name=model_name)
        Knowledge_base = FAISS.from_texts(chunks,embedder)
    st.success("✅ the pdf is ready for chatting ")

    
    # Initialize LLM 
    llm = Ollama(model="mistral") 
    chain = load_qa_chain(llm,chain_type='stuff')
    # asking the question 
    user_question = st.text_input("ask a question about ur pdf")

    if user_question:
        with st.spinner("🤖 Generating answer..."):
            doc = Knowledge_base.similarity_search(user_question,k=3)
            response = chain.run(input_documents=doc,question=user_question)
        st.success("✅ Answer generated")
        st.text_area("Answer", value=response, height=250)












