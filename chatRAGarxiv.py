import os
import streamlit as st
import requests
import PyPDF2
from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SpacyEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Directory for downloaded papers
PAPER_DIR = "arxivpapers"
os.makedirs(PAPER_DIR, exist_ok=True)

# Initialize embeddings
embeddings = SpacyEmbeddings(model_name="en_core_web_md")

# Function to split text into chunks
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

# Function to store text embeddings in FAISS
def vector_store(text_chunks):
    vector_db = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_db.save_local("faiss_db")

# Function to extract text from an uploaded PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PyPDF2.PdfReader(uploaded_file)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to search ArXiv and download papers
def search_and_download_arxiv_papers(query, max_results=3):
    loader = ArxivLoader(query=query, load_max_docs=max_results)
    papers = loader.get_summaries_as_docs()
    paper_texts = {}

    for paper in papers:
        arxiv_id = paper.metadata['Entry ID'].split("/")[-1]
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        pdf_path = os.path.join(PAPER_DIR, f"{arxiv_id}.pdf")

        response = requests.get(pdf_url, stream=True)
        if response.status_code == 200:
            with open(pdf_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            st.success(f"📄 Downloaded ArXiv Paper: {arxiv_id}")

            # Extract text from the downloaded PDF
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                paper_texts[arxiv_id] = text
        else:
            st.error(f"⚠️ Failed to download: {pdf_url}")

    return paper_texts

# Function to load FAISS and ask GPT a question
def answer_question(question):
    vector_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever()

    # Read api_key from key.txt
    with open("key.txt", "r") as file:
        api_key = file.read().strip()
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, api_key=api_key, max_completion_tokens=100)
    
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    response = qa_chain.run(question)
    return response


def main():
    st.title("📄 Chat with PDF and ArXiv Papers")

    # Initialize session state to track document availability
    if "document_ready" not in st.session_state:
        st.session_state["document_ready"] = False
    st.subheader("Select the way to get your document.")
    option = st.radio("Options:", ["📤 Upload your PDF", "🔍 Search in ArXiv"])

    # Case 1: User uploads a PDF
    if option == "📤 Upload your PDF":
        uploaded_file = st.file_uploader("📎 Upload a PDF file", type=["pdf"])

        if uploaded_file is not None:
            with st.spinner("Extracting text from uploaded PDF..."):
                extracted_text = extract_text_from_pdf(uploaded_file)
                text_chunks = get_chunks(extracted_text)
                vector_store(text_chunks)
                st.session_state["document_ready"] = True  # Mark as ready
            st.success("✅ PDF processed successfully!")
            st.text_area("Extracted Text:", extracted_text[:2000], height=300)

    # Case 2: User searches ArXiv
    elif option == "🔍 Search in ArXiv":
        query = st.text_input("🔍 Enter a search query for ArXiv:")
        max_results = st.selectbox("Number of papers to download:", options=list(range(1, 11)), index=2)

        if st.button("Search & Process"):
            with st.spinner("Searching and downloading papers..."):
                papers = search_and_download_arxiv_papers(query, max_results)

            if papers:
                st.success(f"✅ Retrieved {len(papers)} papers from ArXiv.")
                full_text = "\n".join(papers.values())
                text_chunks = get_chunks(full_text)
                vector_store(text_chunks)
                st.session_state["document_ready"] = True  # Mark as ready
                for paper_id, text in papers.items():
                    st.subheader(f"📄 Paper ID: {paper_id}")
                    st.text_area(f"Extracted Text (First 2000 chars):", text[:2000], height=300)
       
            else:
                st.error("❌ No papers found for this query.")

    # **Ask Question Section** (Only show if a document is ready)
    if st.session_state["document_ready"]:
        st.subheader("❓ Ask a Question About the Paper(s)")
        user_question = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            with st.spinner("I am thinking..."):
                answer = answer_question(user_question)
                st.success("✅ Answer.")
                st.write(answer)
    else:
        st.warning("⚠️ Please upload a PDF or search ArXiv before asking questions.")

# Run the app
if __name__ == "__main__":
    main()
