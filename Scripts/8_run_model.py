# =============================================================================
# STEP 7 — RAG-augmented inference
# Retrieves relevant Unity 6 doc chunks from ChromaDB, injects them into
# the prompt, then calls your fine-tuned model via Ollama.
#
# Usage:
#   python 7_inference.py                        # interactive REPL
#   python 7_inference.py "How do I use Jobs?"  # single query
# =============================================================================

import sys
import ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Config ───────────────────────────────────────────────────────────────────
OLLAMA_MODEL = "unity-assistant"
EMBED_MODEL  = "BAAI/bge-small-en-v1.5"
CHROMA_DIR   = "./unity_rag_db"
TOP_K        = 4      # number of doc chunks to retrieve per query
STREAM       = True   # stream tokens as they arrive
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Unity 6 expert assistant. You write optimized C# scripts for Unity 6.0+.

RULES:
- Only use APIs confirmed in the provided Unity 6 documentation context below
- Always use modern Unity 6 patterns (InputSystem, Awaitable, linearVelocity, etc.)
- After every script, explain each technical choice and why it is better

Unity 6 documentation context:
{context}"""


def load_retriever():
    print("Loading embedding model and RAG index...", end=" ", flush=True)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    print("ready.")
    return vectordb


def retrieve(vectordb, question: str) -> str:
    docs = vectordb.similarity_search(question, k=TOP_K)
    chunks = []
    seen_sources = set()
    for doc in docs:
        source = doc.metadata.get("source", "").split("/")[-1]
        if source not in seen_sources:
            chunks.append(f"[{source}]\n{doc.page_content}")
            seen_sources.add(source)
    return "\n\n---\n\n".join(chunks)


def ask(vectordb, question: str) -> str:
    context = retrieve(vectordb, question)
    system  = SYSTEM_PROMPT.format(context=context)

    if STREAM:
        print("\n" + "─" * 60)
        full_response = ""
        for chunk in ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system",  "content": system},
                {"role": "user",    "content": question},
            ],
            stream=True,
        ):
            token = chunk["message"]["content"]
            print(token, end="", flush=True)
            full_response += token
        print("\n" + "─" * 60)
        return full_response
    else:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system",  "content": system},
                {"role": "user",    "content": question},
            ],
        )
        return response["message"]["content"]


def repl(vectordb):
    print("\nUnity 6 Assistant — RAG + Fine-tuned Mistral 7B")
    print("Type 'quit' to exit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        ask(vectordb, question)


# ── Example queries to test the system ───────────────────────────────────────
EXAMPLE_QUERIES = [
    "Write a Unity 6 player controller using the new Input System and Rigidbody.linearVelocity",
    "Create a Burst-compiled parallel job that processes enemy transform positions",
    "How do I replace StartCoroutine with async Awaitable in Unity 6?",
    "Write a Unity 6 ScriptableObject event system with UnityAction",
    "Create a URP custom render feature that adds a fullscreen outline effect",
]


if __name__ == "__main__":
    vectordb = load_retriever()

    if len(sys.argv) > 1:
        # Single query from command line
        question = " ".join(sys.argv[1:])
        ask(vectordb, question)
    else:
        # Interactive mode
        repl(vectordb)
