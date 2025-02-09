import os
import glob
import argparse
import requests
import html
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings.ollama import (
    OllamaEmbeddings,
)  # Updated import for OllamaEmbeddings


FAISS_PATH = r"C:\Users\El-Amrani\RAG\FAISS_DB"
DOC_FILES = r"C:\Users\El-Amrani\RAG\data"


def get_embedding_function():
    """Initialize the Ollama embedding function."""
    # embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings


def load_or_create_faiss(chunks):
    """Load or create a FAISS vector store."""
    embedding_function = get_embedding_function()
    if os.path.exists(FAISS_PATH):
        print(f"Loading existing FAISS index from {FAISS_PATH}")
        vector_store = FAISS.load_local(
            FAISS_PATH,
            embedding_function,
            allow_dangerous_deserialization=True,  # Enable this flag for trusted data
        )
    else:
        print("Creating new FAISS index...")
        vector_store = FAISS.from_texts(chunks, embedding_function)
        vector_store.save_local(FAISS_PATH)
        print(f"FAISS index saved to {FAISS_PATH}")
    return vector_store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docs_dir",
        type=str,
        default=DOC_FILES,
    )
    parser.add_argument(
        "--ollama_url", type=str, default="http://localhost:11434/api/generate"
    )

    args = parser.parse_args()

    print(f"Using data dir {args.docs_dir}")
    print(f"Using FAISS index path {FAISS_PATH}")
    print(f"Using Ollama model URL {args.ollama_url}")

    # Step 1: Load PDFs
    files = glob.glob(f"{args.docs_dir}/*.pdf", recursive=False)
    print(f"Matched files: {files}")

    loader = DirectoryLoader(
        args.docs_dir,
        loader_cls=PyMuPDFLoader,
        recursive=False,
        silent_errors=True,
        show_progress=True,
        glob="*.pdf",
    )

    docs = loader.load()
    print(f"📄 Loaded {len(docs)} PDF documents.")

    # Step 2: Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for i, doc in enumerate(docs):
        print(f"📜 Splitting Document {i + 1}...")
        splits = text_splitter.split_text(doc.page_content)
        chunks.extend(splits)
        print(f"Document {i + 1} split into {len(splits)} chunks.")
    print(f"Total chunks created: {len(chunks)}")

    # Step 3: Load or create FAISS vector store
    vector_store = load_or_create_faiss(chunks)

    # Step 4: Chat with the RAG system
    while True:
        user_query = input("Enter your query (or 'exit' to quit): ").strip()
        if user_query.lower() == "exit":
            print("Exiting...")
            break

        # Step 5: Retrieve relevant documents
        results = vector_store.similarity_search(user_query, k=3)
        context = "\n\n".join([res.page_content for res in results])
        print(f"Context retrieved: {len(results)} chunks")

        # Step 6: Query the Ollama model
        payload = {
            "prompt": f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:",
            "model": "deepseek-r1:1.5b",
            "stream": False,
        }
        try:
            response = requests.post(args.ollama_url, json=payload)
            response.raise_for_status()
            data = response.json()

            # ✅ Extract the raw response
            raw_response = data.get("response", "No response received")

            # ✅ Decode HTML entities (e.g., \u003c becomes <)
            clean_response = html.unescape(raw_response)
            print(clean_response)
            # print(
            #     f"Ollama Response: {response.json().get('response', 'No response key in JSON')}"
            # )
        except requests.exceptions.RequestException as e:
            print(f"Error querying Ollama model: {e}")


if __name__ == "__main__":
    main()
