import os
from dotenv import load_dotenv
from groq import Groq
from embed import retrieve

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise SystemExit(
        "GROQ_API_KEY is not set. Add it to your .env file (see .env.example)."
    )

client = Groq(api_key=api_key)

def ask(question):
    """Retrieve relevant chunks and generate a grounded answer."""
    
    # Step 1: Retrieve relevant chunks
    chunks = retrieve(question, k=3)
    
    # Step 2: Build context string from chunks
    context = ""
    sources = []
    for i, chunk in enumerate(chunks):
        context += f"[Document {i+1} - {chunk['source']}]\n{chunk['text']}\n\n"
        if chunk['source'] not in sources:
            sources.append(chunk['source'])
    
    # Step 3: Build grounded prompt
    prompt = f"""You are an anime guide assistant. Answer the user's question using ONLY the information provided in the documents below. 

If the documents don't contain enough information to answer the question, say "I don't have enough information on that in my current knowledge base."

Do not use any outside knowledge. Only use what is in the documents.

Documents:
{context}

Question: {question}

Answer:"""
    
    # Step 4: Generate response
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            # Keep generation deterministic and tightly grounded in the context.
            temperature=0,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        # Surface a friendly message instead of crashing the caller (e.g. the UI).
        answer = f"Sorry, I couldn't generate an answer right now (error: {e})."

    return {
        "answer": answer,
        "sources": sources,
        "chunks": chunks
    }


if __name__ == "__main__":
    # Test end-to-end generation
    test_questions = [
        "What anime should I watch if I liked Death Note?",
        "What is a good starter anime for beginners?",
        "What are the best isekai anime?"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        print("-" * 50)
        result = ask(question)
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")