import os

# import glob
import argparse
import requests
import html
import shutil
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# Paths
FAISS_PATH = r"C:\Users\El-Amrani\RAG\FAISS_DB"
DOC_FILES = r"C:\Users\El-Amrani\RAG\data"

# ------------------------------- #
#  ✅ Embedding & FAISS Functions  #
# ------------------------------- #


def get_embedding_function():
    """Use a dedicated embedding model for FAISS."""
    return OllamaEmbeddings(model="nomic-embed-text")


def load_or_create_faiss(chunks):
    """Load or create a FAISS vector store with metadata tracking."""
    embedding_function = get_embedding_function()

    if os.path.exists(FAISS_PATH):
        print(f"📂 Loading existing FAISS index from {FAISS_PATH}")
        vector_store = FAISS.load_local(
            FAISS_PATH, embedding_function, allow_dangerous_deserialization=True
        )
    else:
        print("✨ Creating new FAISS index...")
        vector_store = FAISS.from_documents(chunks, embedding_function)
        vector_store.save_local(FAISS_PATH)
        print(f"✅ FAISS index saved to {FAISS_PATH}")

    return vector_store


# -------------------------------- #
#  ✅ Document Processing Functions #
# -------------------------------- #


def load_documents():
    """Load all PDFs from the specified directory."""
    loader = DirectoryLoader(
        DOC_FILES, loader_cls=PyMuPDFLoader, recursive=False, glob="*.pdf"
    )
    return loader.load()


def split_documents(documents):
    """Split documents into chunks, keeping metadata (source file & page number)."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    return calculate_chunk_ids(chunks)


def calculate_chunk_ids(chunks):
    """Assigns unique IDs based on source file & page number."""
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown.pdf")
        page = chunk.metadata.get("page", "unknown")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

    return chunks


def clear_database():
    """Deletes the FAISS index to reset the database."""
    if os.path.exists(FAISS_PATH):
        shutil.rmtree(FAISS_PATH)
        print("🚀 Database cleared!")


# -------------------------------- #
#  ✅ Chat & Retrieval Functions   #
# -------------------------------- #


def query_ollama(user_query, context):
    """Query DeepSeek model with retrieved context."""
    payload = {
        "model": "deepseek-r1:1.5b",
        "prompt": f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:",
        "stream": False,
        "temperature": 0.7,
    }

    url = "http://localhost:11434/api/generate"
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        raw_response = data.get("response", "No response received")
        clean_response = html.unescape(raw_response)

        if "<think>" in clean_response:
            clean_response = clean_response.split("</think>")[-1].strip()

        return clean_response

    except requests.exceptions.RequestException as e:
        return f"Error querying Ollama model: {e}"


# ------------------------------- #
#  ✅ Main Execution Function     #
# ------------------------------- #


def main():
    """Main function for processing PDFs and running the RAG chat."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    if args.reset:
        clear_database()

    print(f"📂 Using data dir: {DOC_FILES}")
    print(f"🗄️ Using FAISS index path: {FAISS_PATH}")

    documents = load_documents()
    chunks = split_documents(documents)
    print(f"📄 Processed {len(documents)} documents into {len(chunks)} chunks.")

    vector_store = load_or_create_faiss(chunks)

    while True:
        user_query = input("\n💬 Enter your query (or 'exit' to quit): ").strip()
        if user_query.lower() == "exit":
            print("👋 Exiting...")
            break

        # Step 4: Retrieve relevant chunks
        results = vector_store.similarity_search(user_query, k=20)
        if not results:
            print("⚠️ No relevant context found!")
            continue

        # Build context with metadata (showing file + page number)
        context = ""
        for i, doc in enumerate(results):
            source = doc.metadata.get("source", "unknown.pdf")
            page = doc.metadata.get("page", "unknown")

            print(f"Context based on the **[{source} - Page {page}]**")

            context += f"\n📄 **[{source} - Page {page}]**\n{doc.page_content}\n"

        print(f"📚 Context retrieved from {len(results)} chunks")

        # Step 5: Ask DeepSeek model
        answer = query_ollama(user_query, context)
        print(f"\n🤖 **DeepSeek Answer:**\n{answer}\n")


if __name__ == "__main__":
    main()
