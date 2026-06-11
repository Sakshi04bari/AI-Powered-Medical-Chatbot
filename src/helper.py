from typing import List

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Extract Data From the PDF File
def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def remove_duplicates(docs):
    seen = set()
    unique_docs = []
    for doc in docs:
        content = " ".join(dict.fromkeys(doc.page_content.split()))  # Remove repeated words
        if content not in seen:
            seen.add(content)
            doc.page_content = content
            unique_docs.append(doc)
    return unique_docs


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Strips each Document to only page_content + source metadata.
    """
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


# Split Data into Text Chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# Download the Embeddings from HuggingFace
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    return embeddings
