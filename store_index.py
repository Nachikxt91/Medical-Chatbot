from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

# Load and process documents
print("Loading PDF documents...")
documents = load_pdf_files("data/")

print("Filtering documents...")
minimal_docs = filter_to_minimal_docs(documents)

print("Splitting text into chunks...")
text_chunks = text_split(minimal_docs)

print(f"Created {len(text_chunks)} text chunks")

# Download embeddings
print("Loading embeddings model...")
embeddings = download_embeddings()

# Create Pinecone vector store
print("Creating Pinecone index...")
index_name = os.getenv("PINECONE_INDEX_NAME")

vectorstore = PineconeVectorStore.from_documents(
    text_chunks,
    embeddings,
    index_name=index_name
)

print("✅ Vector store created successfully!")
