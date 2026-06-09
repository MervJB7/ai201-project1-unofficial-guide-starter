import os
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Reddit's comment-action row gets scraped as one whitespace-stripped token,
# always starting with "permalink" and followed by some subset of these actions
# in varying combinations (e.g. "permalinkembedsaveparentreportreply" or
# "permalinksavereportreply"). We match the whole run rather than enumerate every
# permutation. "permalink" never appears in real prose, so this is safe.
REDDIT_ACTION_ROW = re.compile(
    r"permalink(?:embed|save|parent|report|reply|share|context|give ?award|hide|delete)*"
)

# Other Reddit UI fragments to strip outright.
REDDIT_BOILERPLATE = [
    "load more comments",
    "continue this thread",
    "give award",
]

# Standalone UI lines (a line that is exactly one of these, ignoring case) left
# over from Reddit scraping. We drop the whole line rather than the substring so
# we don't mangle legitimate prose that happens to contain the word.
REDDIT_UI_LINES = {"report", "reply", "share", "permalink", "embed", "save", "parent"}

# A chunk shorter than this (after cleaning) is almost always a stray review
# header, timestamp, or fragment with too little context to be useful.
MIN_CHUNK_LENGTH = 100

# Curse/swear/profanity words. Any chunk containing one of these (as a whole
# word) is dropped entirely so it is never embedded or retrieved. Edit this set
# to tune strictness. Note: "hell" is intentionally excluded because it appears
# constantly in legitimate anime content (titles, idioms like "what the hell").
PROFANITY = frozenset({
    # f-word family
    "fuck", "fucks", "fucked", "fucking", "fuckin", "fucker", "fuckers",
    "fuckface", "clusterfuck", "motherfucker", "motherfuckers", "motherfucking",
    # s-word family
    "shit", "shits", "shitty", "shitting", "shithead", "bullshit",
    # b-word family
    "bitch", "bitches", "bitching", "bastard", "bastards",
    # a-word family
    "ass", "asses", "asshole", "assholes", "dumbass", "jackass", "asshat",
    # d-word family
    "dick", "dicks", "dickhead", "douche", "douchebag", "damn", "damned", "dammit",
    # crude anatomy / sexual
    "cock", "cocks", "cunt", "cunts", "pussy", "prick", "twat",
    "slut", "sluts", "whore", "whores", "piss", "pissed", "pissing",
    # slurs
    "fag", "faggot", "faggots", "retard", "retarded",
    "nigger", "niggers", "nigga", "niggas",
    # misc / british
    "crap", "wanker", "bollocks", "bugger", "goddamn", "goddamned",
})

PROFANITY_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in PROFANITY) + r")\b",
    re.IGNORECASE,
)


def contains_profanity(text):
    """Return True if the text contains any curse/swear/profanity word."""
    return PROFANITY_PATTERN.search(text) is not None


def clean_text(text):
    """Strip scraping artifacts and normalize whitespace in raw document text.

    Note: characters like curly quotes, em/en dashes, accented letters, and
    Japanese kana are valid and meaningful (e.g. anime titles), so we keep them.
    They may render as "?" in some terminals, but they embed correctly.
    """
    # Remove Reddit's comment collapse/expand toggles that prefix usernames,
    # e.g. "[-]username" or "[+]username" (the bracket holds an en-dash or plus).
    text = re.sub(r"\[[-–—+]\]", "", text)

    # Drop the U+FFFD replacement char if any genuinely corrupted bytes exist.
    text = text.replace("�", "")

    # Remove Reddit comment-action rows and other UI boilerplate.
    text = REDDIT_ACTION_ROW.sub("", text)
    for token in REDDIT_BOILERPLATE:
        text = text.replace(token, "")

    # Remove standalone vote/timestamp metadata like "117 points 7 years ago".
    text = re.sub(r"\d+\s*points?\s+\d+\s+\w+\s+ago", "", text)

    # Drop lines that are nothing but a leftover Reddit UI word.
    lines = [ln for ln in text.splitlines()
             if ln.strip().lower() not in REDDIT_UI_LINES]
    text = "\n".join(lines)

    # Collapse runs of blank lines and trailing spaces left behind by removals.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def is_weak_chunk(text):
    """Return True for chunks too short or too low in actual word content to keep."""
    if len(text) < MIN_CHUNK_LENGTH:
        return True
    # Reject chunks that are mostly punctuation/symbols rather than words.
    letters = sum(c.isalpha() for c in text)
    if letters < len(text) * 0.5:
        return True
    return False


def load_documents(folder_path="documents"):
    """Load all .txt files from the documents folder."""
    documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt") and filename != ".gitkeep":
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            text = clean_text(text)
            documents.append({
                "text": text,
                "source": filename
            })
            print(f"Loaded: {filename} ({len(text)} characters)")

    return documents


def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = []

    for doc in documents:
        split_texts = splitter.split_text(doc["text"])
        # Index within this source, incremented only for chunks we keep.
        chunk_index = 0
        for text in split_texts:
            text = text.strip()
            if is_weak_chunk(text):
                continue
            # Drop any chunk containing curse/swear/profanity words entirely.
            if contains_profanity(text):
                continue
            chunks.append({
                "text": text,
                "source": doc["source"],
                "chunk_index": chunk_index
            })
            chunk_index += 1

    return chunks


if __name__ == "__main__":
    # Load documents
    print("=== LOADING DOCUMENTS ===")
    documents = load_documents()
    print(f"\nTotal documents loaded: {len(documents)}")
    
    # Chunk documents
    print("\n=== CHUNKING DOCUMENTS ===")
    chunks = chunk_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    
    # Inspect 5 random chunks
    print("\n=== SAMPLE CHUNKS ===")
    import random
    samples = random.sample(chunks, min(5, len(chunks)))
    for i, chunk in enumerate(samples):
        print(f"\n--- Chunk {i+1} (from: {chunk['source']}) ---")
        print(chunk["text"])
        print(f"[{len(chunk['text'])} characters]")