import os
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Load the Google API Key from the environment variable
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

#to manage frontend
import streamlit as st
st.header("My First documenent")
with st.sidebar:
    st.title("Your documents")
    file=st.file_uploader("Upload pdf file",type="pdf")
    
#to extract text

if file is not None:
    Pdf_reader=PdfReader(file)
    text=""
    for page in Pdf_reader.pages:
        text+=page.extract_text()
        #st.write(text)
        
#to convert into chunks
    
    text_splitter=RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks=text_splitter.split_text(text)
    #st.write(chunks)
        
#generating embeddings
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    

#creating vector stores
    
    vector_stores= FAISS.from_texts(chunks,embeddings)

#get user question

    question=st.text_input("Type your question here")
    
#do similarity search

    if question:
        match=vector_stores.similarity_search(question)
        #st.write(match)
    
#define model

        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
        chain=load_qa_chain(model,chain_type="stuff")
    
#output results

        response=chain.run(input_documents=match,question=question)
        st.write(response)