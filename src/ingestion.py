
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter 


def load_document(file_path: str) -> str:
    """Load a text document and return its contents."""
    path = Path(file_path)
    return path.read_text(encoding="utf-8")


def split_document(text: str) -> list[str]:
    """Split document text into smaller, overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n" , ". " , " " , ""],
    )

    return splitter.split_text(text)


if __name__ == "__main__":
    document_path = "data/documents/company_policy.txt"

    text = load_document(document_path)
    chunks = split_document(text)

    print(f"Original document length: {len(text)} characters")
    print(f"Total chunks created: {len(chunks)}")

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {i} ---")
        print(chunk)
