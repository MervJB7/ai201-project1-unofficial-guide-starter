# The Unofficial Guide — Project 1

> **How to use this template:**
> Complete each section *after* you've built and tested the corresponding part of your system.
> Do not write placeholder text — if a section isn't done yet, leave it blank and come back.
> Every section below is required for submission. One-liners will not receive full credit.

---

## Domain

<!-- What topic or category of knowledge does your system cover?
     Why is this knowledge valuable, and why is it hard to find through official channels?
     Example: "Student reviews of CS professors at [university] — useful because official
     course descriptions don't reflect teaching style, exam difficulty, or workload." -->

Fan-generated anime knowledge — recommendations, watch orders, genre guides,
and show reviews sourced from Reddit and MyAnimeList. This knowledge is
valuable because it reflects real community opinions and nuanced
taste-matching that official platforms like Crunchyroll or Netflix don't
provide. A new fan can't easily search "good psychological anime for someone
who liked Death Note" and get a trustworthy, personalized answer from any
single official source — that knowledge lives in scattered forum threads
and community discussions.


---

## Document Sources

<!-- List every source you collected documents from.
     Be specific: include URLs, subreddit names, forum thread titles, or file names.
     Aim for variety — sources that together cover different subtopics or perspectives. -->

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

<!-- Describe your chunking approach with enough specificity that someone else could reproduce it.
     Include:
     - Chunk size (characters or tokens) and why that size fits your documents
     - Overlap size and why (or why not) you used overlap
     - Any preprocessing you did before chunking (e.g., stripping HTML, removing headers)
     - What your final chunk count was across all documents -->

**Chunk size:** 500 characters

**Overlap:** 50 characters

**Why these choices fit your documents:** Documents are a mix of short
Reddit comments (1–3 sentences) and longer articles (multiple paragraphs).
500 characters is large enough to capture a complete opinion or
recommendation with its full context, while short comments naturally
become their own single chunk without being split. The 50 character overlap
prevents key facts from being cut across chunk boundaries and lost.
LangChain's RecursiveCharacterTextSplitter was used because it splits on
paragraph breaks first, then sentences, then characters — respecting
natural content boundaries instead of cutting mid-thought.

Before chunking, documents were cleaned using a custom clean_text()
function that stripped Reddit UI boilerplate (e.g. "permalinkembedsave"),
vote/timestamp metadata, and collapse/expand toggles. After splitting,
three filters were applied to remove weak chunks:
- Chunks shorter than 100 characters (fragments with no standalone meaning)
- Chunks where less than 50% of characters are letters (mostly
  punctuation or symbols)
- Chunks containing profanity

**Final chunk count:** 865 chunks across 12 documents

---

## Embedding Model

<!-- Name the embedding model you used and explain your choice.
     Then answer: if you were deploying this system for real users and cost wasn't a constraint,
     what tradeoffs would you weigh in choosing a different model?
     Consider: context length limits, multilingual support, accuracy on domain-specific text,
     latency, and local vs. API-hosted. -->

**Model used:** all-MiniLM-L6-v2 via sentence-transformers

**Production tradeoff reflection:** all-MiniLM-L6-v2 runs fully locally
with no API cost or rate limits, making it ideal for this project. For a
real production deployment I would weigh several tradeoffs. OpenAI's
text-embedding-3-small offers higher accuracy for English text but adds
per-request cost and an external API dependency. A multilingual model like
paraphrase-multilingual-MiniLM-L12-v2 would better handle Japanese anime
titles and non-English queries. For a domain-specific system, a model
fine-tuned on anime or media review text would likely outperform a
general-purpose model on niche terminology and abbreviations like "FMA"
or "AoT". Latency is also a consideration at scale — local models avoid
network round-trips but are constrained by local hardware.

---

## Grounded Generation

<!-- Explain how your system enforces grounding — how does it prevent the LLM from answering
     beyond the retrieved documents?
     Describe both your system prompt (what instruction you gave the model) and any structural
     choices (e.g., how you formatted the context, whether you filtered low-relevance chunks).
     Do not just say "I told it to use the documents" — show the actual instruction or explain
     the mechanism. -->


**System prompt grounding instruction:**
The LLM is instructed using the following prompt template in query.py:

"You are an anime guide assistant. Answer the user's question using ONLY
the information provided in the documents below. If the documents don't
contain enough information to answer the question, say 'I don't have
enough information on that in my current knowledge base.' Do not use any
outside knowledge. Only use what is in the documents."

The retrieved chunks are injected into the prompt under a "Documents:"
header, each labeled with its source filename and an index number
(e.g. "[Document 1 - similar_to_death_note.txt]"). This structure makes
the boundaries between documents explicit so the model knows exactly
what it is and isn't allowed to draw from.

**How source attribution is surfaced in the response:**
During retrieval, each returned chunk carries a "source" metadata field
containing its original filename. The ask() function in query.py collects
unique source filenames from all retrieved chunks and returns them
alongside the generated answer as a separate "sources" list. The Gradio
interface in app.py displays this list in a dedicated "Sources" output
box, so the user always sees which documents the answer came from
regardless of what the LLM writes in its response. Attribution is
therefore guaranteed programmatically — it does not depend on the LLM
choosing to cite sources itself.

---

## Evaluation Report

<!-- Run your 5 test questions from planning.md through your system and record the results.
     Be honest — a partially accurate or inaccurate result that you explain well is more
     valuable than a suspiciously perfect result. -->

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What anime should I watch if I liked Death Note? | Psychological thrillers like Monster, Code Geass, The Promised Neverland | Recommended Steins;Gate for its similar suspense-building style; correct source (similar_to_death_note.txt, dist 0.609) but gave only one pick instead of several  | Relevant | Partially accurate |
| 2 | What do fans think of Attack on Titan's ending? | Mixed/divided community opinion from AoT reviews | "I don't have enough information…" — attack_on_titan_reviews.txt did not surface; retrieved off-topic starter-guide chunks (dist 0.825+) | Off-target  | Inaccurate (honest refusal, but failed to find existing content) |
| 3 | What is a good starter anime for someone new? | Beginner picks like FMA Brotherhood, AoT, Your Lie in April | Correctly recommended FMA Brotherhood with genre and episode details from Top 30 Best Starter Anime (dist 0.626) | Relevant | Accurate |
| 4 | What are the best isekai anime according to fans? | Community isekai picks from the genre thread | Hedged with "not enough info" but then listed Overlord, KonoSuba, Tanya, Re:Zero, Shield Hero, Isekai Quartet from 40 Best Isekai (dist 0.453) | Relevant | Partially accurate (over-cautious refusal despite strong retrieved context) |
| 5 | Is Sword Art Online worth watching? | Enjoyable but divisive/overrated community consensus | Described SAO as having a bland protagonist but still fun with superb animation and hyped fights; reflects divisive-but-enjoyable consensus. Third chunk was noise (dist 1.062) | Partially relevant | Accurate |

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

<!-- Identify at least one question where retrieval or generation did not work as expected.
     Write a specific explanation of *why* it failed, tied to a part of the pipeline.

     "The answer was wrong" is not an explanation.

     "The relevant information was split across a chunk boundary, so retrieval returned
     only half the context — the model didn't have enough to answer correctly" is an explanation.

     "The embedding model treated the professor's nickname as out-of-vocabulary and returned
     results from an unrelated review" is an explanation. -->

**Question that failed:**
"What do fans think of Attack on Titan's ending?" (Q2)
**What the system returned:**
"I don't have enough information on that in my current knowledge base." The three retrieved chunks all came from Top 30 Best Starter Anime For Beginners.txt at poor cosine distances (0.825, 0.867, 0.901) — attack_on_titan_reviews.txt did not appear in the top-3 at all.
**Root cause (tied to a specific pipeline stage):**
This is a retrieval / embedding semantic-mismatch failure, not a generation failure. The query embedding is dominated by the concept "ending / finale / how it concluded." But attack_on_titan_reviews.txt contains 

**What you would change to fix it:**
1. Add a distance threshold in retrieve() — discard chunks above ~0.8 so genuinely irrelevant matches (like the 0.825+ starter-guide chunks here) are dropped before reaching the LLM. This makes the "not enough information" response intentional and detectable rather than incidental.

2. Fix the data gap — the curated attack_on_titan_reviews.txt summarizes reviews but omits any discussion of the ending. Either add review content that covers the finale/chapter 139 reception, or accept that the corpus genuinely can't answer ending-specific questions (in which case the refusal is the correct behavior, and the only honest fix is documenting that scope limit).
---

## Spec Reflection

<!-- Reflect on how planning.md shaped your implementation.
     Answer both questions with at least 2–3 sentences each. -->

**One way the spec helped you during implementation:**

Challenge #1 predicted that "short 1–2 sentence reviews may produce chunks too small to carry enough semantic meaning," and that prediction translated directly into a concrete safeguard that wasn't part of the original Chunking Strategy: the is_weak_chunk() filter and MIN_CHUNK_LENGTH = 100 guard in ingest.py. Because the risk was named up front, I built the defense in from the start instead of discovering empty/fragment chunks during evaluation. Similarly, the Chunking Strategy section's explicit reasoning — mixed short reviews vs. long Reddit posts — justified choosing RecursiveCharacterTextSplitter with 500/50, so the implementation decision was already settled before any code was written.

**One way your implementation diverged from the spec, and why:**

The biggest divergence is the corpus and the cleaning pipeline. The spec's Documents section planned to use raw MyAnimeList review dumps (e.g. "Top Reviews for Death Note / Attack on Titan / FMA Brotherhood"), but during implementation those raw scrapes proved enormous and noisy — the FMA dump alone produced 681 chunks that drowned out everything else in retrieval. So I moved to curated review summaries (death_note_reviews.txt, attack_on_titan_reviews.txt, fma_brotherhood_reviews.txt), each condensed to ~10 focused chunks. The pipeline also grew significant text-cleaning logic the spec never anticipated — Reddit boilerplate/UI stripping and a profanity filter in clean_text() (ingest.py) — added purely because the real scraped data was far messier than the plan assumed. The change was justified: balanced, deduplicated sources gave each anime fair representation at retrieval time, which the raw dumps actively prevented.

---

## AI Usage

<!-- Describe at least 2 specific instances where you used an AI tool during this project.
     For each: what did you give the AI as input, what did it produce, and what did you
     change, override, or direct differently?

     "I used Claude to help me code" is not sufficient.
     "I gave Claude my Chunking Strategy section from planning.md and asked it to implement
     chunk_text(). It returned a function using a fixed character split. I overrode the
     chunk size from 500 to 200 because my documents are short reviews, not long guides." -->

**Instance 1**

- *What I gave the AI:*

The symptom — "embed.py is ignoring the new documents I added and is holding onto the old one" — plus access to the repo (embed.py, ingest.py, the documents/ folder, and the ChromaDB store).

- *What it produced:*

Instead of accepting my framing, it inspected the actual vector store and showed that embed.py was embedding every file correctly (all 1,546 chunks were present). It traced the real cause to retrieval, not embedding: the leftover raw Top MAL reviews for FMA Brotherhood.txt (270 KB → 681 chunks) was statistically drowning out my new curated fma_brotherhood_reviews.txt (10 chunks), so the new file never reached the top-k. It then removed the stale dump and rebuilt the store.
- *What I changed or overrode:*

I overrode its working style — it kept trying to run diagnostic Python scripts, and I directed it to just report results in chat and run the evaluation command itself rather than have me create files. I also chose option (b) when it offered to fix the resulting top-k mismatch — updating planning.md to match the code (k=3) rather than reverting the code to k=4.

**Instance 2**

- *What I gave the AI:*
My Chunking Strategy section from planning.md (chunk_size=500, overlap=50, RecursiveCharacterTextSplitter) and the Documents section describing Reddit- and MyAnimeList-sourced .txt files, asking it to implement load_documents() and chunk_documents().
- *What it produced:*
The two functions per spec, plus — after seeing how noisy the raw scraped text was — additional cleaning I hadn't specified: a clean_text() step stripping Reddit UI artifacts (comment-action rows, "load more comments", vote/timestamp lines), an is_weak_chunk() filter enforcing MIN_CHUNK_LENGTH = 100, and a PROFANITY word-list filter that drops chunks entirely.
- *What I changed or overrode:*
I kept the cleaning and weak-chunk filter (they directly addressed the "weak chunks from short reviews" risk in my plan), but I directed the profanity list to exclude "hell" because it appears constantly in legitimate anime content (titles, idioms like "what the hell") — overriding the more aggressive default filter that would have discarded valid chunks.
