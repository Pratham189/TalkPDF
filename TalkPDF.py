import os
import fitz  # For PyMuPDF
import pdfplumber  # For PDF text extraction
from bs4 import BeautifulSoup  # For web scraping
import requests
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from groq import Groq  # Initialize Groq model
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Function to extract text from PDF using PyMuPDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")  # Extracts raw text from PDF pages
    return text

# Function to extract text from PDF using pdfplumber
def extract_text_using_pdfplumber(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()  # Extracts text using pdfplumber
    return text

# Function to scrape web data using BeautifulSoup
def scrape_web_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")  # Extract all paragraphs from the page
        return [para.get_text() for para in paragraphs]
    else:
        return []

# Initialize Groq client for text generation
client = Groq(api_key=os.environ.get('GROQ_API_KEY'))  # API key for Groq

# Function to generate an answer based on PDF and web data
def generate_answer(query, pdf_text, web_data):
    combined_text = pdf_text + " ".join(web_data)  # Combine PDF and web data
    embeddings = OpenAIEmbeddings()  # Initialize OpenAI embeddings
    docsearch = FAISS.from_texts([combined_text], embeddings)  # Vector search

    # Retrieve top documents for query
    docs = docsearch.similarity_search(query, k=3)

    # Format the documents into one string
    retrieved_docs = " ".join([doc.page_content for doc in docs])

    # Use Groq LLM to generate an answer
    prompt = PromptTemplate(
        input_variables=["question", "document"],
        template="""
        You are a helpful assistant. Use the provided document to answer the question:
        {document}
        Question: {question}
        """
    )
    chain = LLMChain(llm=client, prompt=prompt)  # Chain for reasoning
    result = chain.run({"question": query, "document": retrieved_docs})
    return result

# Define an endpoint for uploading PDFs and answering questions
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...), question: str = ""):
    # Save the uploaded PDF file
    pdf_path = f"./uploads/{file.filename}"
    os.makedirs("./uploads", exist_ok=True)  # Ensure the upload directory exists
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    # Extract text from the uploaded PDF
    pdf_text = extract_text_from_pdf(pdf_path)

    # Scrape additional data from a sample URL (can be replaced with dynamic URL input)
    web_data = scrape_web_data("https://example.com")  # Replace with actual URL if needed

    # Generate the answer using LLM
    answer = generate_answer(question, pdf_text, web_data)

    return {"answer": answer}

# Define a test route
@app.get("/")
async def root():
    return {"message": "Welcome to TalkPDF - Upload a PDF and ask your questions!"}
