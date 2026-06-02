import os
import json
import time
import re
from pathlib import Path
from collections import Counter

# select one provider: "groq", "gemini", or "ollama"
PROVIDER = "groq"   

GROQ_API_KEY   = os.getenv("GROQ_TOKEN")
GROQ_MODEL     = "llama-3.3-70b-versatile"

GEMINI_API_KEY = os.getenv("GEMINI_TOKEN")
GEMINI_MODEL   = "models/gemini-2.5-flash"

OLLAMA_MODEL   = "qwen2.5-coder:14b"

# Batch sizes that are tuned to each provider's token/request limits
BATCH_SIZES = {
    "groq":   3,    # 6k TPM -> 3 scripts per request max
    "gemini": 10,   # 250k TPM, 5 RPM -> 10 scripts safe
    "ollama": 5,    # no limits, 5 is quality sweet spot
}

# RPM limits
RPM_LIMITS = {
    "groq":   28,   
    "gemini": 5,    
    "ollama": 999,
}

RESUME    = True
LAST_CALL = [0.0]
# ─────────────────────────────────────────────────────────────────────────────

SKIP_PATTERNS = [
    r"^v[A-Z]", r"PlayMaker", r"Photon", r"DoozyUI",
    r"NodeCanvas", r"BehaviorDesigner", r"OdinInspector",
    r"^OOTII", r"ProBuilder", r"^A_",
]

PROMPT_BATCH = """You are building a training dataset for a Unity 6 AI assistant.
Given multiple Unity C# scripts, produce one training pair per script.
Use EXACTLY this format — one <PAIR> block per script in the same order:

<PAIR>
<INSTRUCTION>
A realistic developer question that leads to this script. Be specific.
Example: "Write a Unity 6 player controller using the new Input System and Rigidbody.linearVelocity"
</INSTRUCTION>

<RESPONSE>
The complete C# script updated for Unity 6.

Apply ALL these migrations where the original uses old patterns:
- Input.GetAxis/GetKey       -> InputAction.ReadValue<>/triggered
- StartCoroutine/IEnumerator -> async Awaitable + Awaitable.NextFrameAsync()
- FindObjectOfType           -> [SerializeField] injection
- Camera.main in Update      -> cached in Awake()
- GetComponent in Update     -> cached in Awake()
- Resources.Load             -> Addressables.LoadAssetAsync
- rb.velocity                -> rb.linearVelocity
- OnGUI/UI.Text              -> UIDocument + UI Toolkit

After the script write:
## Why this is optimized
- 3 to 6 bullets explaining each Unity 6 choice and why it improves performance
</RESPONSE>

<FEATURES>comma-separated Unity 6 APIs used</FEATURES>
<MIGRATED>comma-separated migrations applied</MIGRATED>
</PAIR>

IMPORTANT: Produce exactly one <PAIR>...</PAIR> block per script in the same order as given."""


def call_groq(prompt):
    from groq import Groq
    wait = max(0, (60.0 / RPM_LIMITS["groq"]) - (time.time() - LAST_CALL[0]))
    if wait > 0:
        time.sleep(wait)
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a Unity 6 C# expert. Produce exactly one <PAIR> block per script in order."},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=6000,
        temperature=0.3,
    )
    LAST_CALL[0] = time.time()
    return resp.choices[0].message.content


def call_gemini(prompt):
    from google import genai
    from google.genai import types
    wait = max(0, (60.0 / RPM_LIMITS["gemini"]) - (time.time() - LAST_CALL[0]))
    if wait > 0:
        time.sleep(wait)
    client = genai.Client(api_key=GEMINI_API_KEY)
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="You are a Unity 6 C# expert. Produce exactly one <PAIR> block per script in order.",
            temperature=0.3,
            max_output_tokens=8000,
        ),
    )
    LAST_CALL[0] = time.time()
    return resp.text


def call_ollama(prompt):
    import ollama
    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": "You are a Unity 6 C# expert. Produce exactly one <PAIR> block per script in order."},
            {"role": "user",   "content": prompt},
        ],
        options={"temperature": 0.3, "num_predict": 6000},
    )
    return resp["message"]["content"]


PROVIDERS = {
    "groq":   call_groq,
    "gemini": call_gemini,
    "ollama": call_ollama,
}


# Parsing
def extract_tag(text, tag):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def extract_pairs(text):
    return re.findall(r"<PAIR>(.*?)</PAIR>", text, re.DOTALL)


# Pre-Filter

def should_skip(filename, code):
    name = Path(filename).stem
    for pat in SKIP_PATTERNS:
        if re.search(pat, name):
            return True, "plugin pattern"
    if "using UnityEngine" not in code:
        return True, "no UnityEngine"
    if not any(x in code for x in ["MonoBehaviour", "ISystem", "SystemBase", "ScriptableObject"]):
        return True, "no MonoBehaviour/ISystem"
    real_lines = [l for l in code.splitlines() if l.strip() and not l.strip().startswith("//")]
    if len(real_lines) < 15:
        return True, f"too short ({len(real_lines)} lines)"
    return False, ""


#Loading the files

def load_record(path):
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if text.startswith("{"):
        return json.loads(text)
    return {
        "sha":      path.stem[:12],
        "filename": path.name,
        "repo":     "local",
        "html_url": "",
        "code":     text,
        "hints": {
            "modern_score": 0,
            "legacy_score": int("StartCoroutine" in text or "Input.Get" in text),
        },
    }



# Batch transformation


def transform_batch(records, call_fn):
    batch_size   = len(records)
    code_limit   = max(800, 2500 // batch_size)

    scripts_text = ""
    for idx, record in enumerate(records):
        scripts_text += f"\n\n--- SCRIPT {idx + 1}: {record['filename']} ---\n"
        scripts_text += f"```csharp\n{record['code'][:code_limit]}\n```"

    notes = "\n".join([
        f"Script {i + 1}: {'Migrate all legacy patterns to Unity 6.' if r['hints']['legacy_score'] > 0 else 'Modern script - keep APIs, add explanation only.'}"
        for i, r in enumerate(records)
    ])

    full_prompt = f"{PROMPT_BATCH}\n\nScripts to transform:{scripts_text}\n\nNotes:\n{notes}"

    for attempt in range(3):
        try:
            raw        = call_fn(full_prompt)
            pair_texts = extract_pairs(raw)

            if len(pair_texts) != len(records):
                raise ValueError(f"Expected {len(records)} pairs, got {len(pair_texts)}")

            results = []
            for pair_text, record in zip(pair_texts, records):
                instruction = extract_tag(pair_text, "INSTRUCTION")
                response    = extract_tag(pair_text, "RESPONSE")
                features    = extract_tag(pair_text, "FEATURES")
                migrated    = extract_tag(pair_text, "MIGRATED")

                if not instruction or not response:
                    print(f"\n  missing tags for {record['filename']}", end="")
                    results.append(None)
                    continue
                if len(response) < 200:
                    print(f"\n  too short for {record['filename']}", end="")
                    results.append(None)
                    continue

                results.append({
                    "instruction":     instruction,
                    "response":        response,
                    "unity6_features": [f.strip() for f in (features or "").split(",") if f.strip()],
                    "migrated_from":   [m.strip() for m in (migrated or "").split(",") if m.strip()],
                    "source_sha":      record["sha"],
                    "source_repo":     record.get("repo", "local"),
                    "was_legacy":      record["hints"]["legacy_score"] > 0,
                })
            return results

        except ValueError as e:
            if attempt < 2:
                print(f"\n  Retry attempt {attempt + 1} (failed: {e})")
                time.sleep(3)
            else:
                print(f"\n  SKIP batch (all retries failed: {e})")
                return [None] * len(records)

        except Exception as e:
            err = str(e).lower()
            print(f"\n  ERROR: {e}")
            if "503" in err or "unavailable" in err:
                wait = 30 * (attempt + 1)
                print(f"  Service overloaded — sleeping {wait}s...")
                time.sleep(wait)
                continue
            
            if "429" in err or "quota" in err or "rate" in err or "resource" in err:
                error_str = str(e)

                if "per day" in error_str.lower() or "tokens per day" in error_str.lower():
                    print(f"\n  Daily token limit hit — stopping.")
                    print(f"  Resume tomorrow with: python 2_transform_pairs.py")
                    import sys; sys.exit(0)
                
                wait_match = re.search(r'try again in (\d+)m([\d.]+)s', error_str)
                if wait_match:
                    minutes = int(wait_match.group(1))
                    seconds = float(wait_match.group(2))
                    wait    = minutes * 60 + seconds + 5   # +5 seconds buffer
                else:
                    wait    = 75 * (attempt + 1)
                
                print(f"  Rate limited — sleeping {wait:.0f}s (as requested by API)...")
                time.sleep(wait)
                continue
            return [None] * len(records)

    return [None] * len(records)


# Stats

def print_stats(pairs, total, pre_filtered, skipped):
    sent = total - pre_filtered
    print("\n" + "=" * 52)
    print("  DATASET QUALITY REPORT")
    print("=" * 52)
    print(f"  Total scripts        : {total}")
    print(f"  Pre-filtered         : {pre_filtered}  (plugins/stubs)")
    print(f"  Sent to LLM          : {sent}")
    print(f"  Skipped/failed       : {skipped}")
    print(f"  Valid pairs          : {len(pairs)}")
    if sent:
        print(f"  Success rate         : {len(pairs)/sent*100:.1f}%")
    migr = sum(1 for p in pairs if p.get("was_legacy"))
    print(f"  Legacy-migrated      : {migr}")
    print(f"  Already-modern       : {len(pairs) - migr}")

    feat_counter = Counter()
    for p in pairs:
        for f in p.get("unity6_features", []):
            feat_counter[f.strip()] += 1
    if feat_counter:
        max_count = feat_counter.most_common(1)[0][1]
        print(f"\n  Top Unity 6 features in dataset:")
        for feat, count in feat_counter.most_common(10):
            bar = "█" * max(1, count * 20 // max_count)
            print(f"  {count:4d}  {bar:<20}  {feat}")
    print("=" * 52)


# FLUSH

def _flush(pairs, path):
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"  [checkpoint: {len(pairs)} pairs saved]")


# Main

def build_pairs(raw_dir="./raw_scripts", output="./unity6_pairs.jsonl"):
    call_fn    = PROVIDERS[PROVIDER]
    batch_size = BATCH_SIZES[PROVIDER]
    rpm_limit  = RPM_LIMITS[PROVIDER]
    min_delay  = 60.0 / rpm_limit

    files = sorted(
        list(Path(raw_dir).glob("*.cs")) +
        list(Path(raw_dir).glob("*.json"))
    )

    # Resume: load already-done SHAs
    done_shas = set()
    pairs     = []
    if RESUME and Path(output).exists():
        with open(output) as f:
            for line in f:
                try:
                    p = json.loads(line)
                    done_shas.add(p["source_sha"])
                    pairs.append(p)
                except Exception:
                    pass
        if done_shas:
            print(f"Resuming — {len(done_shas)} pairs already done")

    # Load and filter all records upfront
    todo         = []
    pre_filtered = 0

    for path in files:
        try:
            record = load_record(path)
        except Exception as e:
            print(f"READ ERROR {path.name}: {e}")
            continue

        if record["sha"] in done_shas:
            continue

        skip, reason = should_skip(record["filename"], record["code"])
        if skip:
            pre_filtered += 1
            continue

        todo.append(record)

    total_batches = (len(todo) + batch_size - 1) // batch_size
    est_min       = total_batches * min_delay / 60
    model_name    = GROQ_MODEL if PROVIDER == "groq" else GEMINI_MODEL if PROVIDER == "gemini" else OLLAMA_MODEL

    print(f"\nProvider      : {PROVIDER}")
    print(f"Model         : {model_name}")
    print(f"Batch size    : {batch_size} scripts/request")
    print(f"Scripts todo  : {len(todo)}")
    print(f"Pre-filtered  : {pre_filtered}")
    print(f"Total batches : {total_batches}")
    print(f"Est. time     : {est_min:.0f} min  ({est_min/60:.1f}h)")
    print(f"Saving to     : {output}\n")

    start_time = time.time()
    skipped    = 0
    processed  = 0

    for batch_start in range(0, len(todo), batch_size):
        batch   = todo[batch_start: batch_start + batch_size]
        batch_n = batch_start // batch_size + 1
        names   = " | ".join(r["filename"] for r in batch)

        print(f"[{batch_n}/{total_batches}] {names}", end=" ", flush=True)

        t0      = time.time()
        results = transform_batch(batch, call_fn)
        elapsed = time.time() - t0
        processed += len(batch)

        ok = 0
        for result in results:
            if result:
                pairs.append(result)
                ok += 1
            else:
                skipped += 1

        avg     = (time.time() - start_time) / max(processed, 1)
        eta_min = (len(todo) - batch_start - len(batch)) * avg / 60
        print(f"{ok}/{len(batch)} OK  ({elapsed:.0f}s)  ETA={eta_min:.0f}min")

        if batch_n % 5 == 0 and pairs:
            _flush(pairs, output)

    _flush(pairs, output)
    print_stats(pairs, len(files), pre_filtered, skipped)


if __name__ == "__main__":
    import sys
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "./raw_scripts"
    try:
        build_pairs(raw_dir=raw_dir)
    except KeyboardInterrupt:
        print("\nStopped — progress saved to unity6_pairs.jsonl")
