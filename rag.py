import os
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import tkinter as tk
from tkinter import filedialog
import ctypes

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  
except Exception:
    pass

load_dotenv()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")

root = tk.Tk()
root.withdraw()  
pdf_path = filedialog.askopenfilename(
    title="Select a PDF file", 
    filetypes=(("PDF Files", "*.pdf"), ("All Files", "*.*"))
)

if not pdf_path:
    print("No file selected. Exiting.")
    exit()

pdf_loader = PyPDFLoader(pdf_path)
documents = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()  
vectorstore = FAISS.from_documents(chunks, embeddings)

llm = OpenAI(temperature=0)  

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", 
    retriever=vectorstore.as_retriever()
)

while True:
    user_query = input("Please enter your query (or type 'exit' to quit): ")
    if user_query.lower() == 'exit':
        print("Exiting...")
        break
    
    response = qa_chain.run(user_query)
    print(response)
