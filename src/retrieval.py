
## Now lets check the performance of retrival on the basis of current strategies
import chromadb
from pathlib import Path
from ingestion import load_document , split_document
from embedding import load_embedding_model , generate_embeddings
from sentence_transformers import util


## Here we are required to create one more basic function check the score of the chunks on the basis of querry

## Now lets get the data from chroma_db instead of manual passing it
def semantic_search(query , model , collection , n_results):

    query_embeddings = model.encode(
        query,
        normalize_embeddings = True,
        show_progress_bar = True
    )

    """Retrieve the most similar chunks using Chromadb"""
    ## Now lets search for all the vectors with distance from querry
    # Search The Stored Vector
    results = collection.query(
        query_embeddings = [query_embeddings.tolist()],
        n_results = n_results,
        include = ["documents" , "distances"]
    )

    return [
        {
            "document" : document,
            "distance" : distance
        }
        for document, distance in zip(
            results["documents"][0],
            results["distances"][0],
        )
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
    query1 = "Can I work remotely, and how many days are allowed?"
    query2 = "what are the supports provided by the company?"
    
    
    
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

    # 3. Now lets check the semantic search documents and its distance : 
    results = semantic_search(query2 , model , collection , 2)

    print("\n-- Chromadb Search Result --")
    print(f"\nUser question: {query2}")

    for rank , result in enumerate(results , start=1):
        print(f"\n Rank: {rank}")
        print(f"\n Distance: {result['distance']:.4f}")
        print(f"\n Document: {result['document']}")

    


