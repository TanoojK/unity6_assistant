# Unity 6 AI Assistant

A locally-running AI assistant for Unity 6 game development. Ask it to write C# scripts,
explain Unity 6 APIs, set up animations, configure Cinemachine, or anything else Unity-related
and get complete, optimized answers with explanations.

Built by fine-tuning **Mistral 7B Instruct** with **QLoRA** on 730+ Unity 6 C# scripts,
augmented with **RAG** over Unity 6 documentation, editor guides, and video transcripts.
Runs entirely on your local machine via **Ollama** — no internet needed after setup.

---

## What it can answer

- *"Write a Unity 6 player controller using the new Input System"*
- *"How do I replace StartCoroutine with Awaitable in Unity 6?"*
- *"Create a Burst-compiled parallel job for enemy AI"*
- *"How do I set up a 2D blend tree for character movement?"*
- *"Write a URP custom render feature for outlines"*
- *"How do I use Cinemachine 3 with the new Input System?"*

---

## Architecture

```
User query
    |
    v
ChromaDB RAG  <-- Unity 6 docs + editor guides + YouTube transcripts
    |
    v
Augmented prompt
    |
    v
Ollama  <-- Mistral 7B GGUF + QLoRA LoRA adapter
    |
    v
Unity 6 C# script + step-by-step explanation
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Base model | Mistral 7B Instruct v0.3 |
| Fine-tuning | QLoRA rank 16 via unsloth |
| Training | Google Colab T4 GPU (free) |
| Dataset generation | Groq Llama 3.3 70B (batched 3 scripts/request) |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector DB | ChromaDB |
| Inference | Ollama (GGUF + LoRA adapter) |
| Training pairs | 731 instruction pairs |
| Disk footprint | ~5.5 GB total |

---

## Requirements

- Windows / Linux / Mac
- Python 3.10+
- Git
- [Ollama](https://ollama.com) installed
- Mistral 7B GGUF model (Q4_K_M recommended)
- 8 GB RAM minimum (16 GB recommended)
- 12 GB free disk space
- GPU optional for inference (CPU works, just slower)

---

## Quick Setup (use pre-built model)

If you just want to run the assistant without rebuilding the dataset or fine-tuning:

### Step 1 — Clone the repo

```bash
git clone https://github.com/TanoojK/unity_6_assistant
cd unity-6-assistant
```

### Step 2 — Create a virtual environment

```bash
python -m venv unity_assistant

# Windows
unity_assistant\Scripts\activate

# Mac / Linux
source unity_assistant/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

# Step 4 — Download the GGUF Model

Download the prebuilt GGUF model from Hugging Face:

https://huggingface.co/TanoojK/unity-gguf

Place the model inside:

```text id="p0zk9t"
models/
    unity-assistant.gguf
```

This GGUF already contains:

* Mistral 7B
* Unity 6 fine-tuning
* LoRA merged into the model

No LoRA conversion is required.

### Step 5 — Build the RAG index

Fetches Unity 6 docs, editor guides, and video transcripts into ChromaDB.
Run once — takes about 30 minutes.

```bash
python scripts/5_build_rag.py
```

Output saved to `./unity_rag_db/` (~350 MB).

Optional: for richer editor knowledge (animation, Cinemachine, Timeline):

```bash
pip install youtube-transcript-api
python scripts/5_build_rag.py
```

### Step 7 — Create the Ollama Model

Generate the Ollama `Modelfile` automatically:

```bash id="4lyi0g"
python scripts/7_create_modelfile.py
```

This script:

* finds the GGUF model
* creates a Modelfile
* configures system prompts and inference settings

Then register the model with Ollama:

```bash id="y4kkg5"
ollama create unity-assistant -f Modelfile
```

Verify:

```bash id="76v09v"
ollama list
```

---

### Step 8 — Run the Assistant

```bash id="s6qtk1"
python scripts/8_run_model.py
```

Or directly:

```bash id="i9wx6m"
ollama run unity-assistant
```


## Full Pipeline — Build from Scratch

This section explains every script in the pipeline, what it does, how to configure it,
and what output to expect. Follow these steps if you want to reproduce or extend the dataset
and fine-tuning yourself.

---

### Script 1 — `scripts/1_scraper.py`

**What it does:**
Searches GitHub for Unity C# scripts using the GitHub Code Search API.
Targets Unity 6 specific patterns (InputSystem, DOTS, Jobs, Addressables, UI Toolkit)
across 13 targeted search queries. Filters out stubs, generated files, third-party plugins,
and test files. Saves each script as a JSON file containing the code and metadata.

**Before running:**
1. Get a free GitHub token at https://github.com/settings/tokens (no special scopes needed)
2. Open `scripts/1_scraper.py` and set your token:

```python
GITHUB_TOKEN = "your_github_token_here"
```

**Run:**

```bash
python scripts/1_scraper.py
```

**What to expect:**
- Takes 1–2 hours (rate-limited to 30 requests/min by GitHub)
- Scrapes up to 100 scripts per query across 13 queries (~1,300 raw files)
- ~40–60% are filtered out as stubs or plugins
- Output: `./raw_scripts/` folder containing ~700–800 `.cs` files

**Output example:**

```
Query: language:csharp UnityEngine.InputSystem extension:cs
  [1] PlayerController.cs  modern=2 legacy=1
  [2] InputHandler.cs      modern=3 legacy=0
  ...
Done. Scraped 847 scripts -> ./raw_scripts/
```

---

### Script 2 — `scripts/2_transform_pairs.py`

**What it does:**
Reads each raw `.cs` script and sends batches of 3 to a language model (Groq, Gemini, or
Ollama). The model migrates old Unity patterns to Unity 6 equivalents and generates a
realistic developer instruction + optimized response pair for each script.
Saves progress every 5 pairs so you can stop and resume at any time.

**Before running:**
1. Get a free Groq API key at https://console.groq.com (no credit card needed)
2. Open `scripts/2_transform_pairs.py` and configure:

```python
PROVIDER     = "groq"                    # "groq" | "gemini" | "ollama"
GROQ_API_KEY = "your_groq_api_key_here"
GROQ_MODEL   = "llama-3.3-70b-versatile"
RESUME       = True   # always keep True — saves progress across sessions
```

**Run:**

```bash
python scripts/2_transform_pairs.py

# If your raw_scripts folder is elsewhere:
python scripts/2_transform_pairs.py F:\path\to\raw_scripts
```

**What to expect:**
- Groq free tier: 100,000 tokens/day = ~96 pairs/day
- Script stops itself when daily limit is hit and prints how to resume
- Resume the next day by just running the same command — done pairs are skipped
- Full dataset of ~730 pairs takes approximately 7 days of overnight runs
- Output: `./unity6_pairs.jsonl`

**Output example:**

```
Provider      : groq
Batch size    : 3 scripts/request
Scripts todo  : 760
Est. time     : 9 min  (rate limit will pause it after ~96 pairs)

[1/254] PlayerController.cs | InputHandler.cs | BulletBehaviour.cs  3/3 OK  (4s)  ETA=18min
[2/254] EnemyAI.cs | GameManager.cs | UIManager.cs                  3/3 OK  (6s)  ETA=17min
...
Daily token limit hit — stopping cleanly.
Progress saved. Run again tomorrow to continue.
```

**Unity 6 migrations applied automatically:**

| Old pattern | Unity 6 replacement |
|---|---|
| `Input.GetAxis / GetKey` | `InputAction.ReadValue<> / triggered` |
| `StartCoroutine / IEnumerator` | `async Awaitable + NextFrameAsync()` |
| `FindObjectOfType` | `[SerializeField]` injection |
| `Camera.main` in Update | cached in `Awake()` |
| `GetComponent<T>` in Update | cached in `Awake()` |
| `Resources.Load` | `Addressables.LoadAssetAsync` |
| `rb.velocity` | `rb.linearVelocity` |
| `OnGUI / UI.Text` | `UIDocument + UI Toolkit` |

---

### Bonus — `normalize_features.py`

**What it does:**
The LLM labels features inconsistently across pairs — `InputSystem` and `Input System`
and `New Input System` all mean the same thing. This script maps all variants to a
single canonical name so dataset statistics are accurate.

**Run after `2_transform_pairs.py` is complete:**

```bash
python normalize_features.py
```

**Output example:**

```
Normalized 731 pairs

Feature distribution after normalization:
   317x  ████████████████████  InputSystem
   225x  ██████████████        Awaitable
   140x  █████████             Addressables
    85x  █████                 DOTS
```

---

### Bonus — `check_token_lengths.py`

**What it does:**
Tokenizes every pair using the Mistral tokenizer and checks none exceed the context
window. Important before fine-tuning — truncated pairs produce bad training data.

**Run before `3_prepare_finetune.py`:**

```bash
pip install transformers
python check_token_lengths.py
```

**Output example:**

```
Total pairs     : 731
Min tokens      : 452
Max tokens      : 1906
Avg tokens      : 753
Over 4096       : 0   <-- all pairs fit, no truncation
```

---

### Script 3 — `scripts/3_prepare_finetune.py`

**What it does:**
Converts `unity6_pairs.jsonl` into the Mistral instruct chat format that unsloth
expects for training. Wraps each pair in the system prompt and `[INST]` template.
Filters out pairs that are too short (stubs) or too long (over context window).

**Run:**

```bash
python scripts/3_prepare_finetune.py
```

**What to expect:**
- Takes under 1 minute
- Output: `./mistral_finetune.jsonl`

**Output example:**

```
Input pairs : 731
Output pairs: 718   (13 filtered — too short or missing C# code)
-> mistral_finetune.jsonl
```

---

### Script 4 — `scripts/4_finetune.py` (run on Google Colab)

**What it does:**
Fine-tunes Mistral 7B Instruct using QLoRA (rank 16) via unsloth on your dataset.
Trains for 3 epochs using 8-bit AdamW optimizer. Saves LoRA adapters (~150 MB) —
not the full model — so disk usage stays low. Deletes HuggingFace weights after
training to free disk space.

**This script requires a GPU with at least 8 GB VRAM.**
Use Google Colab free T4 GPU (16 GB VRAM) — no local GPU needed.

**Steps:**

1. Upload `mistral_finetune.jsonl` to Google Drive at `My Drive/unity_assistant/`
2. Open [colab.research.google.com](https://colab.research.google.com)
3. Set Runtime → Change runtime type → **T4 GPU**
4. Mount Drive and run the training cells from `scripts/4_finetune.py`

**Important — save checkpoints to Drive so disconnects don't lose progress:**

In the training config set:
```python
output_dir = "/content/drive/MyDrive/unity_assistant/checkpoints"
save_strategy = "epoch"   # saves after every epoch
```

If Colab disconnects, rerun all cells — it auto-resumes from the last checkpoint.

**What to expect:**
- Model download: ~5 min (4.5 GB, first run only)
- Training: ~2 hours for 731 pairs across 3 epochs
- Loss should drop from ~1.2 to ~0.3 over training
- Output: `unity_lora_adapter/` folder in Drive (~150 MB)

**Output example:**

```
Trainable params: 41.9M / 7241.7M (0.58%)
Training samples: 731

Step  10  loss: 1.219
Step  50  loss: 0.479
Step 110  loss: 0.333
Step 200  loss: 0.251
Step 288  loss: 0.198

Training complete!
Adapter saved: /content/drive/MyDrive/unity_assistant/unity_lora_adapter
Size: 148 MB
```

**After training:**
Download `unity_lora_adapter/` from Google Drive to your PC.

---

### Script 5 — `scripts/5_build_rag.py`

**What it does:**
Fetches 70+ Unity 6 documentation pages covering scripting APIs, editor workflows,
animation, Cinemachine, Timeline, Shader Graph, VFX Graph, physics, lighting,
NavMesh, audio, terrain, and ProBuilder. Optionally fetches YouTube transcripts for
editor workflow knowledge. Chunks all text and embeds it into a local ChromaDB
vector database using the `BAAI/bge-small-en-v1.5` embedding model (130 MB).

**Run:**

```bash
python scripts/5_build_rag.py
```

**Optional — include YouTube transcripts:**

```bash
pip install youtube-transcript-api
python scripts/5_build_rag.py
```

**What to expect:**
- Embedding model downloads on first run (~130 MB)
- Fetches and processes all doc pages (~30 minutes)
- Output: `./unity_rag_db/` (~350 MB)
- Runs a retrieval test at the end to confirm everything works

**Output example:**

```
Loading embedding model: BAAI/bge-small-en-v1.5 (130 MB)...
Fetching 70 Unity 6 doc pages...
  [1/70] AnimationOverview -> 12 chunks
  [2/70] AnimatorControllers -> 18 chunks
  ...
Fetching YouTube transcripts...
  OK  Unity Animator Controller tutorial  (24891 chars)
  OK  Unity Blend Trees explained  (18432 chars)
  ...
Embedding 3847 chunks into ChromaDB...
RAG index built -> ./unity_rag_db/  (342 MB, 3847 chunks)

Retrieval test:
  Q: How do I set up a blend tree for 2D movement?
    -> AnimationBlendTrees: A 2D blend tree uses two parameters...
  Q: Rigidbody linearVelocity Unity 6
    -> Rigidbody-linearVelocity: linearVelocity replaces velocity...
```

---

### Script 6 — `scripts/6_convert_lora.sh`

**What it does:**
Converts the HuggingFace LoRA adapter (`.safetensors`) to GGUF format for Ollama,
then creates an Ollama Modelfile combining your base GGUF with the adapter and
registers a `unity-assistant` model with Ollama.

**Before running:**
1. Clone llama.cpp: `git clone --depth 1 https://github.com/ggerganov/llama.cpp`
2. Open `scripts/6_convert_lora.sh` and update the two paths at the top:

```bash
BASE_GGUF="./your_model.gguf"         # path to your Mistral GGUF
LORA_DIR="./unity_lora_adapter"       # path to downloaded adapter
```

**Run:**

```bash
# Mac / Linux
chmod +x scripts/6_convert_lora.sh
./scripts/6_convert_lora.sh

# Windows — run the conversion manually:
python llama.cpp/convert_lora_to_gguf.py `
    --base .\your_model.gguf `
    --lora .\unity_lora_adapter `
    --outfile .\unity_lora_adapter\adapter_model.gguf

# Then create and register the Ollama model:
python create_modelfile.py
ollama create unity-assistant -f Modelfile
```

**What to expect:**
- Conversion takes 1–2 minutes
- Produces `unity_lora_adapter/adapter_model.gguf` (~150 MB)
- Registers `unity-assistant` in Ollama

**Output example:**

```
[1/3] Converting LoRA adapter to GGUF format...
LoRA GGUF saved: unity_lora_adapter/adapter_model.gguf  (148 MB)

[2/3] Creating Ollama Modelfile...
Modelfile created.

[3/3] Registering model with Ollama...
unity-assistant: success

Test with:
  ollama run unity-assistant
```

---

### Script 7 — `scripts/7_inference.py`

**What it does:**
The main inference pipeline. Embeds the user query using bge-small, retrieves the
top 4 most relevant chunks from ChromaDB, injects them into the system prompt as
Unity 6 documentation context, then calls the fine-tuned model via Ollama and
streams the response token by token.

**Run:**

```bash
# Interactive REPL
python scripts/7_inference.py

# Single query
python scripts/7_inference.py "Write a Unity 6 player controller using the new Input System"
```

**Configuration** (top of file):

```python
OLLAMA_MODEL = "unity-assistant"     # name registered in Step 6
EMBED_MODEL  = "BAAI/bge-small-en-v1.5"
CHROMA_DIR   = "./unity_rag_db"
TOP_K        = 4     # number of doc chunks retrieved per query
STREAM       = True  # stream tokens as they arrive
```

**What to expect:**

```
Loading embedding model and RAG index... ready.

Unity 6 Assistant - RAG + Fine-tuned Mistral 7B
Type 'quit' to exit.

You: Write a Unity 6 player controller using the new Input System

------------------------------------------------------------
using UnityEngine;
using UnityEngine.InputSystem;

[RequireComponent(typeof(Rigidbody))]
public class PlayerController : MonoBehaviour
{
    [SerializeField] float _moveSpeed = 5f;
    [SerializeField] float _jumpForce = 7f;

    Rigidbody _rb;
    InputAction _moveAction;
    InputAction _jumpAction;

    void Awake()
    {
        _rb = GetComponent<Rigidbody>();
        ...
    }
}

## Why this is optimized
- Rigidbody.linearVelocity replaces the deprecated .velocity property in Unity 6
- InputAction cached in Awake() avoids per-frame GetComponent overhead
- [RequireComponent] guarantees Rigidbody exists at edit time
------------------------------------------------------------
```

---

## Project Structure

```
unity-6-assistant/
    scripts/
        1_scraper.py                 # GitHub API scraper for Unity C# scripts
        2_transform_pairs.py         # Dataset generation (Groq / Gemini / Ollama)
        3_prepare_finetune.py        # Format dataset for Mistral fine-tuning
        4_finetune.py                # QLoRA training with unsloth
        5_build_rag.py               # Build ChromaDB RAG index from Unity 6 docs
        6_convert_lora.sh            # Convert LoRA adapter to GGUF for Ollama
        7_inference.py               # RAG-augmented inference pipeline
    Transform_Pairs_T4_Colab.ipynb   # Colab notebook for dataset gen on free T4 GPU
    normalize_features.py            # Normalize feature labels in dataset
    check_token_lengths.py           # Verify all pairs fit in context window
    transform_pairs_gemini.py        # Gemini-specific batched transform
    Modelfile.template               # Ollama Modelfile template
    requirements.txt
    samples/
        sample_pairs.jsonl           # Example training pairs
    README.md
    .gitignore
```

---

## Storage Budget

| File | Size | When needed |
|---|---|---|
| Mistral 7B GGUF Q4_K_M | ~4.6 GB | always |
| LoRA adapter | ~150 MB | always |
| bge-small-en-v1.5 | ~130 MB | always |
| ChromaDB RAG index | ~350 MB | always |
| HF model weights | ~4.5 GB | training only — delete after |
| **Production total** | **~5.2 GB** | |

---

## Dataset Statistics

```
Total training pairs  : 731
Min tokens per pair   : 452
Max tokens per pair   : 1906
Average tokens        : 753
Pairs over 4096 limit : 0  (all fit in 2048 tokens)

Top Unity 6 features:
  317x  InputSystem
  225x  Awaitable
  140x  Addressables
   85x  DOTS / ECS
   77x  UIToolkit
   63x  ScriptableObject
   59x  Unity.Mathematics
   57x  Unity.Collections
```

---

## Model 

- LoRA Adapter: [HuggingFace Models](https://huggingface.co/TanoojK/unity-gguf)

