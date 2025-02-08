import os
import glob
import argparse
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docs_dir",
        type=str,
        default=r"C:\Users\El-Amrani\RAG\data",
    )
    parser.add_argument("--persist_dir", type=str, default="data_faiss")
    args = parser.parse_args()

    print(f"Using data dir {args.docs_dir}")
    print(f"Using index path {args.persist_dir}")

    # Debug: Print matched files
    files = glob.glob(
        f"{args.docs_dir}/*.pdf", recursive=False
    )  # Restrict to top-level only
    print(f"Matched files: {files}")

    # Load PDFs
    loader = DirectoryLoader(
        args.docs_dir,
        loader_cls=PyMuPDFLoader,
        recursive=False,  # Disable recursive folder scanning
        silent_errors=True,
        show_progress=True,
        glob="*.pdf",  # Only load PDFs in the current directory
    )

    docs = loader.load()
    print(f"📄 Loaded {len(docs)} PDF documents.")

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Each chunk will have up to 1000 characters
        chunk_overlap=200,  # Allow 200-character overlap between chunks
    )

    # Split the content of each document
    all_splits = []
    for i, doc in enumerate(docs):
        print(f"📜 Splitting Document {i + 1}...")
        splits = text_splitter.split_text(doc.page_content)
        all_splits.extend(splits)
        print(f"Document {i + 1} split into {len(splits)} chunks.")

    # Print some of the split chunks for verification
    for i, chunk in enumerate(all_splits[:5]):  # Show the first 5 chunks
        print(f"Chunk {i + 1}: {chunk[:200]}...")  # Truncated preview of each chunk


if __name__ == "__main__":
    main()
