import chromadb
from sentence_transformers import SentenceTransformer
from ingest import load_documents, chunk_documents

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_PATH = "chroma_db"

# Lazily-built, process-wide singletons. Loading the embedding model and opening
# the ChromaDB client are both expensive, so we do each once and reuse it across
# every retrieve() call instead of rebuilding them on every query.
_model = None
_client = None
_collections = {}


def get_model():
    """Return the shared embedding model, loading it once on first use."""
    global _model
    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_client():
    """Return the shared ChromaDB client, opening it once on first use."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def get_collection(collection_name="anime_guide"):
    """Return a cached handle to the named ChromaDB collection."""
    if collection_name not in _collections:
        _collections[collection_name] = get_client().get_collection(collection_name)
    return _collections[collection_name]


def embed_and_store(chunks, collection_name="anime_guide"):
    """Embed chunks and store them in ChromaDB."""

    client = get_client()

    # Delete existing collection if it exists (so we can re-run cleanly)
    try:
        client.delete_collection(collection_name)
        print("Deleted existing collection.")
    except Exception:
        pass

    collection = client.create_collection(collection_name)
    # Keep the retrieval cache in sync with the freshly rebuilt collection.
    _collections[collection_name] = collection

    model = get_model()

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

    # Reuse the cached model and collection rather than rebuilding per call.
    model = get_model()
    collection = get_collection(collection_name)

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