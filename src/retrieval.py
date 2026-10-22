
## Now lets check the performance of retrival on the basis of current strategies
import chromadb
from pathlib import Path
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


### Now lets store embeddings in chroma_db
def store_in_chromadb(chunks , embeddings):
    client = chromadb.PersistentClient( ## Select the database
        path="/chromadb"
    )

    collection = client.get_or_create_collection( ## Select the collection 
        name="company_policy"
    )

    collection.upsert( ## Insert and Update the Collection
        ids=[f"policy_{i}" for i in range(len(chunks))], ## store the ids 
        documents=chunks, ## chunks 
        embeddings = embeddings ## and Embeddings
    )

    print("Collection Created Successfuly")
    return collection ## Return your created collection




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




    ## Now lets store the Embeddings in chromadb
    collection = store_in_chromadb(chunks , embeddings)

    stored_data = collection.get(
        include = ['documents']
    )

    # Now lets verify the stored record now
    print("Verify the stored record for now")

    for doc_id, document in zip(
        stored_data["ids"],
        stored_data["documents"],
    ):
        print(f"\nID: {doc_id}")
        print(f"Document: {document[:100]}...")



    # 3. Now lets check the semantic_search chunks ordering
    results = semantic_search(querry2 , model , embeddings , chunks)

    print(f"\nUser question: {querry2}")
    print("\nRanked retrieval results:")

    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"\nRank {rank} | Similarity: {score:.4f}")
        print(chunk)

    


