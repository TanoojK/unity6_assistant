
#   1. Download docs from:
#      https://docs.unity3d.com/6000.3/Documentation/Manual/OfflineDocumentation.html
#   2. Unzip the downloaded file and place the resulting folder as-is in the same directory as this script, renaming it to "unity_docs_local"
#      Result: unity_docs_local\Manual\ and unity_docs_local\ScriptReference\

import time
from pathlib import Path
from bs4 import BeautifulSoup

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Config 
LOCAL_DOCS_DIR = "./unity_docs_local"   # unzipped offline docs go here
CHROMA_DIR     = "./unity_rag_db"
EMBED_MODEL    = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE     = 1000
CHUNK_OVERLAP  = 100
SCRAPE_YOUTUBE = True                   # set False to skip transcripts
BATCH_SIZE     = 500                    # chunks per ChromaDB batch insert

# Pages to skip— boilerplate, redirect pages, or too generic to be useful
SKIP_KEYWORDS = [
    "UnityManual", "docdata", "StaticFiles", "Images",
    "uploads", "VideoPlayer", "StyleSheets",
    "search", "Search", "404", "index",
]

# Only index these high-value ScriptReference pages (avoids 10k+ API stubs)
SCRIPTREF_ALLOWLIST = [
    # Physics
    "Rigidbody", "Physics", "Collider", "RaycastCommand", "CharacterController",
    # Input
    "InputAction", "InputSystem", "PlayerInput", "Keyboard", "Mouse", "Gamepad",
    # Async
    "Awaitable", "AsyncOperation", "Coroutine",
    # Rendering
    "Camera", "Light", "RenderTexture", "Graphics", "Shader",
    # Animation
    "Animator", "AnimationClip", "AnimationCurve", "Avatar",
    # GameObject / Transform
    "GameObject", "Transform", "MonoBehaviour", "ScriptableObject",
    # UI
    "UIDocument", "VisualElement", "Button", "Canvas",
    # Audio
    "AudioSource", "AudioClip", "AudioMixer",
    # Addressables
    "Addressables", "AssetReference", "AsyncOperationHandle",
    # Jobs / Burst
    "IJob", "IJobParallelFor", "JobHandle", "BurstCompile", "NativeArray",
    # Particles
    "ParticleSystem",
    # NavMesh
    "NavMeshAgent", "NavMesh",
    # Misc
    "Time", "Debug", "Application", "Screen", "Resources",
]

# YouTube tutorials (optional)
YOUTUBE_VIDEOS = {
    "ZnMNOqADg5E": "Unity Animator Controller tutorial",
    "vApG8aYD5aI": "Unity Blend Trees explained",
    "hFB9F8cdSXM": "Unity Animation Rigging tutorial",
    "dXqNNk5fJY8": "Cinemachine Unity 6 setup",
    "OX_6_bKpIgY": "Shader Graph Unity 6 basics",
    "THcqbdm5vBo": "Unity VFX Graph tutorial",
    "Xtm04ORRPBY": "Unity Timeline tutorial",
}


# html parsing

def parse_html(path: Path) -> str | None:
    """Extract clean text from a Unity 6.3 offline doc HTML file."""
    try:
        html = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")


        main = (
            soup.find("div", id="content-wrap")   # Unity 6.3 offline docs
            or soup.find("div", id="master-wrapper")
            or soup.find("article")
            or soup.find("main")
        )
        if not main:
            return None

        # Remove sidebar and nav noise from within the content
        for tag in main(["script", "style", "noscript"]):
            tag.decompose()
        for tag in main.find_all("div", id="sidebar"):
            tag.decompose()
        for tag in main.find_all("div", class_=["breadcrumbs", "feedback",
                                                 "footer", "header-wrapper"]):
            tag.decompose()

        text = main.get_text(separator="\n", strip=True)
        return text if len(text) > 200 else None

    except Exception as e:
        print(f"  parse error {path.name}: {e}")
        return None


def should_skip(path: Path) -> bool:
    name = path.stem
    for kw in SKIP_KEYWORDS:
        if kw.lower() in name.lower():
            return True
    return False


def in_allowlist(path: Path) -> bool:
    name = path.stem
    for kw in SCRIPTREF_ALLOWLIST:
        if name.startswith(kw):
            return True
    return False


# Load Documents

def load_documents():
    docs_dir = Path(LOCAL_DOCS_DIR)
    if not docs_dir.exists():
        print(f"ERROR: '{LOCAL_DOCS_DIR}' not found.")
        print("Download and unzip Unity 6.3 offline docs there first.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    docs = []

    # 1. Manual/ — all pages 
    manual_dir = docs_dir / "Manual"
    if manual_dir.exists():
        pages = [p for p in manual_dir.rglob("*.html") if not should_skip(p)]
        print(f"Indexing Manual: {len(pages)} pages...")
        ok = skip = 0
        for path in pages:
            text = parse_html(path)
            if not text:
                skip += 1
                continue
            for j, chunk in enumerate(splitter.split_text(text)):
                docs.append({
                    "text":   chunk,
                    "source": f"Manual/{path.name}",
                    "chunk":  j,
                })
            ok += 1
        print(f"  Done — {ok} parsed, {skip} skipped, {len(docs)} chunks so far")
    else:
        print(f"WARNING: Manual/ folder not found in {LOCAL_DOCS_DIR}")

    # 2. ScriptReference/ — allowlisted pages only
    scriptref_dir = docs_dir / "ScriptReference"
    if scriptref_dir.exists():
        all_pages   = list(scriptref_dir.rglob("*.html"))
        pages       = [p for p in all_pages if in_allowlist(p) and not should_skip(p)]
        print(f"\nIndexing ScriptReference: {len(pages)} / {len(all_pages)} pages (allowlisted)...")
        ok = skip = 0
        before = len(docs)
        for path in pages:
            text = parse_html(path)
            if not text:
                skip += 1
                continue
            for j, chunk in enumerate(splitter.split_text(text)):
                docs.append({
                    "text":   chunk,
                    "source": f"ScriptReference/{path.name}",
                    "chunk":  j,
                })
            ok += 1
        added = len(docs) - before
        print(f"  Done — {ok} parsed, {skip} skipped, {added} chunks added")
    else:
        print(f"WARNING: ScriptReference/ folder not found in {LOCAL_DOCS_DIR}")

    # 3. YouTube transcripts (optional) 
    if SCRAPE_YOUTUBE:
        print(f"\nFetching YouTube transcripts...")
        yt_chunks = fetch_transcripts(splitter)
        docs.extend(yt_chunks)
        print(f"  {len(yt_chunks)} transcript chunks added")

    return docs


# YOUTUBE TRANSCRIPTS


def fetch_transcripts(splitter):
    chunks = []
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        print("  youtube-transcript-api not installed — skipping")
        return chunks

    for video_id, title in YOUTUBE_VIDEOS.items():
        try:
            # Works with both old and new versions of the library
            api = YouTubeTranscriptApi()
            transcript = api.fetch(video_id)
            full_text = " ".join([t.text for t in transcript])
            for chunk in splitter.split_text(full_text):
                chunks.append({
                    "text":   chunk,
                    "source": f"YouTube: {title}",
                    "chunk":  0,
                })
            print(f"  OK  {title}")
        except Exception:
            # Fallback for older library versions
            try:
                data = YouTubeTranscriptApi.get_transcript(video_id)
                full_text = " ".join([t["text"] for t in data])
                for chunk in splitter.split_text(full_text):
                    chunks.append({
                        "text":   chunk,
                        "source": f"YouTube: {title}",
                        "chunk":  0,
                    })
                print(f"  OK  {title} (legacy API)")
            except Exception as e:
                print(f"  SKIP {title}: {e}")
        time.sleep(0.5)
    return chunks

# Build the index


def build_index(docs):
    print(f"\nLoading embedding model: {EMBED_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"Embedding {len(docs)} chunks into ChromaDB in batches of {BATCH_SIZE}...")

    # Batch inserts prevent memory spikes on large doc sets
    vectordb = None
    for i in range(0, len(docs), BATCH_SIZE):
        batch    = docs[i : i + BATCH_SIZE]
        texts    = [d["text"]   for d in batch]
        metas    = [{"source": d["source"], "chunk": d["chunk"]} for d in batch]
        end      = min(i + BATCH_SIZE, len(docs))
        print(f"  Batch {i}–{end} / {len(docs)}...")

        if vectordb is None:
            vectordb = Chroma.from_texts(
                texts=texts,
                embedding=embeddings,
                metadatas=metas,
                persist_directory=CHROMA_DIR,
            )
        else:
            vectordb.add_texts(texts=texts, metadatas=metas)

    size_mb = sum(
        f.stat().st_size for f in Path(CHROMA_DIR).rglob("*") if f.is_file()
    ) / 1e6
    print(f"\nRAG index built → {CHROMA_DIR}/  ({size_mb:.0f} MB, {len(docs)} chunks)")
    return vectordb


# Retrieval test

def test_retrieval(vectordb):
    print("\n── Retrieval test ──────────────────────────────────────────")
    queries = [
        "Rigidbody linearVelocity Unity 6",
        "Awaitable async coroutine replacement",
        "BurstCompile IJobParallelFor example",
        "How do I set up a blend tree for 2D movement?",
        "Cinemachine 3 Input System setup",
        "How to bake lighting in Unity 6?",
    ]
    all_ok = True
    for q in queries:
        results = vectordb.similarity_search(q, k=2)
        print(f"\n  Q: {q}")
        for r in results:
            src = r.metadata["source"]
            print(f"    → {src}")
            print(f"      {r.page_content[:120].strip()}...")
        # Flag if results look off-topic
        sources = " ".join(r.metadata["source"] for r in results)
        if "ProBuilder" in sources and "blend" in q.lower():
            print("  ⚠  WARNING: retrieval looks wrong for this query")
            all_ok = False
    if all_ok:
        print("\n✅ All retrieval results look on-topic.")
    else:
        print("\n⚠  Some results look off — consider rebuilding with a larger chunk size.")



# Main

if __name__ == "__main__":
    print("=" * 60)
    print("Unity 6.3 RAG Index Builder")
    print("=" * 60)

    docs = load_documents()

    if not docs:
        print("\nNo documents loaded. Check LOCAL_DOCS_DIR path and unzip status.")
    else:
        print(f"\nTotal chunks to embed: {len(docs)}")
        vdb = build_index(docs)
        test_retrieval(vdb)
        print("\nDone! Run 7_inference.py to start the assistant.")