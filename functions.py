import streamlit as st 
from PyPDF2 import PdfReader
import base64
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv



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
    if text is not None :
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
        st.toast("✅ The PDF is ready for chatting!", icon="🤖")
    return Knowledge_base





def getting_answer(Knowledge_base,user_question):
    """
    this function initializes the LLm and finds the chunks relvent to the query 
    to finally generate the answer
    """
    # Initialize LLM 
    # getiing the api key
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.2
    )

    chain = load_qa_chain(llm,chain_type='stuff')
    # asking the question 
    #printing the response
    if  len(user_question) > 10:
        with st.spinner("🤖 Generating answer..."):
            doc = Knowledge_base.similarity_search(user_question,k=3)
            response = chain.run(input_documents=doc,question=user_question)
        st.toast("✅ Answer generated")
        st.text_area("Answer", value=response, height=250)
    else :
        st.error('please ask a complete question!!')












