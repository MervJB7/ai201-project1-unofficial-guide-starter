import chromadb
from sentence_transformers import SentenceTransformer
from ingest import load_documents, chunk_documents

def embed_and_store(chunks, collection_name="anime_guide"):
    """Embed chunks and store them in ChromaDB."""
    
    # Set up ChromaDB (stores locally in a folder called chroma_db/)
    client = chromadb.PersistentClient(path="chroma_db")
    
    # Delete existing collection if it exists (so we can re-run cleanly)
    try:
        client.delete_collection(collection_name)
        print("Deleted existing collection.")
    except:
        pass
    
    collection = client.create_collection(collection_name)
    
    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Embed all chunks
    print(f"Embedding {len(chunks)} chunks...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Store in ChromaDB in batches
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        
        collection.add(
            ids=[f"chunk_{i+j}" for j in range(len(batch))],
            embeddings=[e.tolist() for e in batch_embeddings],
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"], "chunk_index": c["chunk_index"]} for c in batch]
        )
        print(f"Stored chunks {i} to {i+len(batch)}")
    
    print(f"\nDone! {len(chunks)} chunks stored in ChromaDB.")
    return collection


def retrieve(query, collection_name="anime_guide", k=3):
    """Retrieve the top-k most relevant chunks for a query."""
    
    # Load model and collection
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection(collection_name)
    
    # Embed the query
    query_embedding = model.encode(query).tolist()
    
    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    # Format results
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "distance": results["distances"][0][i]
        })
    
    return chunks


if __name__ == "__main__":
    # Build the vector store
    print("=== LOADING AND CHUNKING DOCUMENTS ===")
    documents = load_documents()
    chunks = chunk_documents(documents)
    
    print("\n=== EMBEDDING AND STORING ===")
    embed_and_store(chunks)
    
    # Test retrieval with 3 evaluation questions
    print("\n=== TESTING RETRIEVAL ===")
    test_queries = [
        "What anime should I watch if I liked Death Note?",
        "What is a good starter anime for beginners?",
        "What are the best isekai anime?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        results = retrieve(query)
        for r in results:
            print(f"[distance: {r['distance']:.3f}] (source: {r['source']})")
            print(r["text"][:200])
            print()