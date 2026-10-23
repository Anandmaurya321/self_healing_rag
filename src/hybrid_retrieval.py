
from ingestion import load_document , split_document
from embedding import load_embedding_model , generate_embeddings
from dense_retrieval import store_in_chromadb , semantic_search 
from bm25_retrieval import build_bm25_index , bm25_search

def hybrid_search(query , chunks , bm25, embedding_model, collection, n_results=2 , fusion_k=60):
    """Combining BM25 and dense vector using RRF: Reciprocal Rank Fusion"""
   
    
    sparse_results = bm25_search(query , chunks, bm25 , n_results)
    
    dense_results = semantic_search(query , embedding_model , collection , n_results)


    ## Store Combined Score for each Chunks : RRF : Reciprocal Rank Fusion
    fused = {}

    for results in (dense_results, sparse_results):
        for rank, result in enumerate(results, start=1):
            chunk_id = result["id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "id": chunk_id,
                    "document": result["document"],
                    "rrf_score": 0.0,
                }

            fused[chunk_id]["rrf_score"] += (
                1.0 / (fusion_k + rank)
            )

    # Sort by combined score, highest first.
    ranked_results = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    return ranked_results[:min(n_results , len(ranked_results))]


if __name__ == "__main__":

    document_path = "data/documents/company_policy.txt"
    document_text = load_document(document_path)
    document_chunks = split_document(document_text)
    

    queries = [
        "How many days can I work remotely?",
        "How long do I have to submit travel expense claims?",
        "Who should I contact for payroll problems?",
        "What is the company's parental leave policy?",
    ]



    bm25 = build_bm25_index(document_chunks)
    embedding_model = load_embedding_model()
    embeddings = embedding_model.encode(document_chunks)
    collection = store_in_chromadb(document_chunks , embeddings)


    for query in queries:
        
        results = hybrid_search(
            query,
            document_chunks,
            bm25,
            embedding_model,
            collection,
            n_results=2,
            fusion_k=60
        )

        print(f"\n{'=' * 60}")
        print("Question:", query)

        for rank, result in enumerate(results, start=1):
            print(
                f"Rank: {rank} | "
                f"ID: {result['id']} | "
                f"RRF Score: {result['rrf_score']:.6f}"
            )





## Thanking You ::>>>>
## Thanking You ::>>>>>
## Thanking You ::>>>>>>>