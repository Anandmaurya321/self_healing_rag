
from sentence_transformers import SentenceTransformer
from ingestion import split_document , load_document

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model():
    """Load a pretrained embedding model."""
    return SentenceTransformer(MODEL_NAME)

def generate_embeddings(chunks , model):
    """Convert text into numerical data"""
    embeddings = model.encode(
        chunks,
        normalize_embeddings = True,
        show_progress_bar = True
    )
    return embeddings


if __name__ == "__main__":
    
    # 1. load and split the document 
    document_path = "data/documents/company_policy.txt"

    document_text = load_document(document_path)

    chunks = split_document(document_text)

    """Model is loading here"""
    model = load_embedding_model()
    embeddings = generate_embeddings(chunks , model)

    # Inspect the result 

    print("chunks count" , len(chunks))

    print("Embeddings Matrix Shape", embeddings.shape)

    for i, embedding in enumerate(embeddings, start=1):
        print(f"\nChunk {i}: {chunks[i - 1][:70]}...")
        print("Vector dimensions:", len(embedding))
        print("First 5 values:", embedding[:5])

    print("Everything is working perfectly")
