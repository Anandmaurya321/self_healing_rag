
## Now lets check the performance of retrival on the basis of current strategies

from ingestion import load_document , split_document
from embedding import load_embedding_model , generate_embeddings
from sentence_transformers import util


## Here we are required to create one more basic function check the score of the chunks on the basis of querry

def semantic_search(querry , model , embeddings , chunks):

    querry_embeddings = model.encode(
        querry,
        normalize_embeddings = True,
        show_progress_bar = True
    )

    ## Now lets make the cosine similarity with each chunks and get its scores

    scores = util.cos_sim(querry_embeddings , embeddings)[0]
    sorted_score_ind = scores.argsort(descending=True).tolist()

    ## Now we have the index which gives the sorted score

    return [
        (chunks[i], float(scores[i]))
        for i in sorted_score_ind
    ]


if __name__ == "__main__":
    
    data_path = "data/documents/company_policy.txt"
    querry1 = "Can I work remotely, and how many days are allowed?"
    querry2 = "what are the supports provided by the company?"
    # 1. Load data and perform chunking
    document_text = load_document(data_path)
    chunks = split_document(document_text) ## split by recursive approach 

    # 2. Perform Embeddings
    model = load_embedding_model()
    embeddings = generate_embeddings(chunks , model)

    # 3. Now lets check the semantic_search chunks ordering
    results = semantic_search(querry2 , model , embeddings , chunks)

    print(f"\nUser question: {querry2}")
    print("\nRanked retrieval results:")

    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"\nRank {rank} | Similarity: {score:.4f}")
        print(chunk)

    


