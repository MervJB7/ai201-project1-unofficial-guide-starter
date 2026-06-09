# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain



---

## Documents

Fan-generated anime knowledge — recommendations, watch orders, genre guides,
and show reviews sourced from Reddit (r/anime, r/Animesuggest) and
MyAnimeList. This knowledge is valuable because it reflects real community
opinions and nuanced taste-matching that official platforms like Crunchyroll
or Netflix don't provide. A new fan can't easily search "good psychological
anime for someone who liked Death Note" and get a trustworthy, personalized
answer from any single official source — that knowledge lives in scattered
forum threads and community discussions.

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | Reddit | A general list of anime recommendations | https://www.reddit.com/r/AnimeReviews/comments/1pi8zxv/my_anime_list_looking_for_recommendations/
| 2 | Reddit | Anime that are hidden gems but very good | https://www.reddit.com/r/anime/comments/1edei41/a_simple_list_of_anime_for_people_looking_for/
| 3 | Reddit | Best and worst anime genere | https://www.reddit.com/r/anime/comments/17k9qor/best_and_worst_genres_of_anime/?limit=500
| 4 | My Anime List | Top Reviews for Death Note | https://myanimelist.net/manga/21/Death_Note/reviews
| 5 | My Anime List | Top Reviews for Attack on Titan | https://myanimelist.net/anime/16498/Shingeki_no_Kyojin/reviews
| 6 | My Anime List | Top Reviews for Full Metal Alchemist: Brootherhood | https://myanimelist.net/manga/21/Death_Note/reviews
| 7 | FandomSpot | Best Isekai Anime | https://www.fandomspot.com/best-isekai-anime/
| 8 | FandomSpot | Ultimate Beginners Guide for Anime | https://www.fandomspot.com/beginner-anime/
| 9 | Reddit | Best Slice of Life Anime | https://www.reddit.com/r/anime/comments/1ki0g4a/best_slice_of_life_anime/
| 10 | Reddit | The Ultimate Anime Watch Order Guide | https://www.reddit.com/r/anime/wiki/watch_order

---

## Chunking Strategy

**Chunk size:** 500 characters

**Overlap:** 50 characters

**Reasoning:** Documents are a mix of short reviews (1–3 sentences) and
longer Reddit posts (multiple paragraphs), with key facts sometimes
self-contained and sometimes spread across sentences. 500 characters is
large enough to capture a complete opinion or recommendation with context,
while short reviews will naturally become their own single chunk. The 50
character overlap prevents key facts from being cut across a chunk boundary
and lost entirely. LangChain's RecursiveCharacterTextSplitter will be used
because it tries to split on paragraph breaks first, then sentences, then
characters — respecting natural content boundaries instead of cutting
mid-thought.

**Chunk size:**

**Overlap:**

**Reasoning:**

---

## Retrieval Approach

**Embedding model:** all-MiniLM-L6-v2 via sentence-transformers

**Top-k:** 3 chunks per query

**Top-k reasoning:** Settled on 3 rather than 4 during implementation. With
`temperature=0` and a strict grounding prompt, a 4th chunk frequently added a
low-relevance, high-distance result that diluted the context and nudged the
model toward weaker or hedged answers. Retrieving 3 kept the context tighter
and better-grounded across the evaluation questions.

**Production tradeoff reflection:** all-MiniLM-L6-v2 runs fully locally
with no API cost or rate limits, which makes it ideal for this project.
For a real production deployment I would weigh several tradeoffs: OpenAI's
text-embedding-3-small offers higher accuracy for English text but adds
per-request cost and a dependency on an external API. A multilingual model
like paraphrase-multilingual-MiniLM-L12-v2 would better handle Japanese
anime titles and non-English queries. For a domain-specific system, a model
fine-tuned on anime or media review text would likely outperform a
general-purpose model on niche terminology and abbreviations like "FMA"
or "AoT". Latency is also a consideration at scale — local models avoid
network round-trips but are constrained by local hardware.

**Embedding model:**

**Top-k:**

**Production tradeoff reflection:**

---

## Evaluation Plan


| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | What anime should I watch if I liked Death Note? | Should recommend psychological thrillers like Monster, Code Geass, or The Promised Neverland based on community reviews
| 2 | What do fans think of Attack on Titan's ending? | Should reflect the mixed/divided community opinion captured in collected AoT reviews 
| 3 | What is a good starter anime for someone completely new to anime? | Should return beginner-friendly picks like FMA Brotherhood, AoT, or Your Lie in April from the beginner guide 
| 4 | What are the best isekai anime according to fans? | Should surface community-recommended isekai titles from the genre thread
| 5 | Is Sword Art Online worth watching? | Should reflect the community consensus (enjoyable but divisive/overrated) from overrated/underrated threads 

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. **Mixed document length producing weak chunks:** Short 1–2 sentence
   reviews may produce chunks too small to carry enough semantic meaning
   for the embedding model to match accurately. Retrieval may return
   fragments that don't fully answer the query.

2. **Anime title abbreviations and alternate names:** The embedding model
   may not handle abbreviations like "FMA", "AoT", or "HxH" well,
   causing retrieval to miss relevant chunks when a user queries by full
   title or vice versa. Japanese titles vs. English titles could cause
   the same problem.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

┌─────────────────────────────────────────────────────────────┐
│                        INDEXING                             │
│                                                             │
│  [.txt files]                                               │
│       ↓                                                     │
│  Document Ingestion                                         │
│  (Python open())                                            │
│       ↓                                                     │
│  Chunking                                                   │
│  (LangChain RecursiveCharacterTextSplitter                  │
│   chunk_size=500, overlap=50)                               │
│       ↓                                                     │
│  Embedding + Vector Store                                   │
│  (sentence-transformers all-MiniLM-L6-v2 → ChromaDB)       │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                        QUERYING                             │
│                                                             │
│  [User question]                                            │
│       ↓                                                     │
│  Embed query                                                │
│  (all-MiniLM-L6-v2)                                         │
│       ↓                                                     │
│  Retrieval                                                  │
│  (ChromaDB semantic search, top-k=3)                        │
│       ↓                                                     │
│  Generation                                                 │
│  (Groq llama-3.3-70b-versatile)                             │
│       ↓                                                     │
│  [Answer + source citations]                                │
│  (Gradio UI)                                                │
└─────────────────────────────────────────────────────────────┘
---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:**
I will give Claude my Documents section (file locations and types) and
my Chunking Strategy section (chunk_size=500, overlap=50,
RecursiveCharacterTextSplitter). I will ask it to implement two functions:
load_documents() that reads all .txt files from the documents/ folder and
returns a list of {text, source} dicts, and chunk_documents() that applies
my specified chunking strategy and returns a list of chunks with source
metadata attached. I will verify the output by printing 5 random chunks
and confirming they are readable, complete thoughts with no HTML artifacts.

**Milestone 4 — Embedding and retrieval:**
I will give Claude my Retrieval Approach section and my architecture
diagram. I will ask it to implement embed_and_store() that embeds all
chunks using all-MiniLM-L6-v2 and stores them in ChromaDB with source
metadata, and retrieve() that takes a query string and returns the top 4
most relevant chunks with their source filenames. I will verify by running
3 of my evaluation questions and checking that returned chunks visibly
relate to each question.

**Milestone 5 — Generation and interface:**
I will give Claude my grounding requirement (answers must come only from
retrieved chunks, not LLM general knowledge), my output format (answer +
source list), and the Gradio skeleton from the project instructions. I
will ask it to implement ask() that builds a Groq prompt from retrieved
chunks and returns a grounded answer, and a Gradio interface with a
question input, answer output, and sources output. I will verify grounding
by asking a question my documents don't cover and confirming the system
says it doesn't have enough information.