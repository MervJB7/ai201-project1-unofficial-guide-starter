import gradio as gr
from query import ask
from embed import get_model, get_collection

def handle_query(question):
    if not question.strip():
        return "Please enter a question.", ""

    result = ask(question)
    if result["sources"]:
        sources = "\n".join(f"• {s}" for s in result["sources"])
    else:
        sources = "No sources found."
    return result["answer"], sources

with gr.Blocks(title="Anime Unofficial Guide") as demo:
    gr.Markdown("# 🎌 The Unofficial Anime Guide")
    gr.Markdown("Ask anything about anime — recommendations, watch orders, reviews, and more.")
    
    with gr.Row():
        inp = gr.Textbox(
            label="Your question",
            placeholder="e.g. What anime should I watch if I liked Death Note?",
            lines=2
        )
    
    btn = gr.Button("Ask", variant="primary")
    
    with gr.Row():
        answer = gr.Textbox(label="Answer", lines=10)
        sources = gr.Textbox(label="Sources", lines=10)
    
    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])

if __name__ == "__main__":
    # Warm up the embedding model and vector store at startup so the first user
    # query is fast instead of paying the one-time load cost mid-request.
    print("Warming up (loading embedding model and vector store)...")
    get_model()
    get_collection()
    demo.launch()