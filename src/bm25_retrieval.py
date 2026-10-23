
from ingestion import load_document , split_document
from rank_bm25 import BM25Okapi
import re

## It just make the word tokens in lowercase which we used for creating indices.
def tokenize(text: str)-> list[str]:
    """Convert text into lowercase word tokens. Using Regular Expression"""
    return re.findall(r"\b\w+\b", text.lower())  


## bm25 automatically created its indices on the basis of tokens it have
def build_bm25_index(chunks:list[str])->BM25Okapi:
    """Build bm25 index over the document chunks""" 
    tokenized_chunks = [tokenize(chunk) for chunk in chunks]
    return BM25Okapi(tokenized_chunks)


## Simple BM25_Search
def bm25_search(query: str , chunks:list[str] , bm25: BM25Okapi , n_results: int=2
)-> list[dict]:
    """Return the top n_results chunks ranked by BM25 score"""
    query_token = tokenize(query)
    scores = bm25.get_scores(query_token)
    ranked_indices = sorted(
            range(len(chunks)),
            key=lambda i: scores[i],
            reverse=True,
    )

    results = []

    for index in ranked_indices[:min(n_results , len(chunks))]:
            results.append({
                "id": f"policy_{index}",
                "document": chunks[index],
                "score": float(scores[index]),
            })
    
    return results



if __name__ == "__main__":
    document_path = "data/documents/company_policy.txt"

    text = load_document(document_path) ## Extract all the text 
    chunks = split_document(text) ## Convert it into chunks using recursive chunking 

    bm25 = build_bm25_index(chunks) ## Create bm25_indices which will return the created bm25 which we use for bm25_search 

    queries = [
        "How many days can I work remotely?",
        "How long do I have to submit travel expense claims?",
        "Who should I contact for payroll problems?",
    ]

    for query in queries:
        results = bm25_search(query, chunks, bm25, n_results=2)

        print(f"\n{'=' * 60}")
        print("Question:", query)

        for rank, result in enumerate(results, start=1):
            print(f"\nRank: {rank}")
            print(f"ID: {result['id']}")
            print(f"BM25 Score: {result['score']:.4f}")
            print(f"Document: {result['document']}")



## Thanking You ::>>>>>>
