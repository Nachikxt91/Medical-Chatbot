from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

# Extract text from pdf files
def load_pdf_files(data):  # FIX: was inconsistent naming
    """Load PDF files from directory"""
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

# Filter to only keep source and page_content
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Keep only source metadata and page_content"""
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

# Split documents into smaller chunks
def text_split(minimal_docs):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    text_chunks = text_splitter.split_documents(minimal_docs)
    return text_chunks

# Download embeddings model
def download_embeddings():  # FIX: function name matches import
    """Download and return HuggingFace embeddings model"""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'}  # Added for stability
    )
    return embeddings
