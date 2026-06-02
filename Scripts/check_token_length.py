import json
import sys
from pathlib import Path

try:
    from transformers import AutoTokenizer
except ImportError:
    print("ERROR: transformers not installed.")
    print("Run: pip install transformers")
    sys.exit(1)

#  Config 
PAIRS_FILE   = "unity6_pairs.jsonl"
MODEL_ID     = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_TOKENS   = 4096
SHOW_LONGEST = 5     # print the N longest pairs for inspection


SYSTEM_PROMPT = (
    "You are a Unity 6 expert assistant. You write optimized C# scripts "
    "for Unity 6.0+ using modern APIs: InputSystem, Awaitable, linearVelocity, "
    "Addressables, UIToolkit, DOTS, Jobs/Burst. After every script explain each "
    "Unity 6 choice and why it improves performance."
)


def format_pair(pair: dict) -> str:
    """Format a pair into the Mistral instruct template for token counting."""
    instruction = pair.get("instruction", "")
    response    = pair.get("response", "")
    return (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{instruction} [/INST] {response} </s>"
    )


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(PAIRS_FILE)

    if not path.exists():
        print(f"ERROR: {path} not found.")
        print("Run 2_transform_pairs.py first.")
        sys.exit(1)

    # Load pairs
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    print(f"Loaded {len(pairs)} pairs from {path}")

    # Load tokenizer
    print(f"Loading tokenizer: {MODEL_ID}...")
    print("(This downloads ~500 KB of tokenizer files on first run)")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"\nERROR loading tokenizer: {e}")
        print("If you're offline, use: AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')")
        sys.exit(1)

    # Tokenize all pairs
    print(f"Tokenizing {len(pairs)} pairs...")
    lengths = []
    over_limit = []

    for i, pair in enumerate(pairs):
        text   = format_pair(pair)
        tokens = tokenizer(text, return_tensors=None)["input_ids"]
        length = len(tokens)
        lengths.append((length, i, pair))
        if length > MAX_TOKENS:
            over_limit.append((length, i, pair))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(pairs)}...")

    # Stats
    lengths_only = [l[0] for l in lengths]
    min_len = min(lengths_only)
    max_len = max(lengths_only)
    avg_len = sum(lengths_only) / len(lengths_only)

    print(f"\n{'='*50}")
    print(f"Total pairs     : {len(pairs)}")
    print(f"Min tokens      : {min_len}")
    print(f"Max tokens      : {max_len}")
    print(f"Avg tokens      : {avg_len:.0f}")
    print(f"Over {MAX_TOKENS}       : {len(over_limit)}", end="")

    if len(over_limit) == 0:
        print("  All pairs fit . no truncation will occur")
    else:
        print(f"   ⚠️  These will be truncated during training!")
        print(f"\nPairs over {MAX_TOKENS} tokens:")
        for length, idx, pair in sorted(over_limit, reverse=True):
            instr = pair.get("instruction", "")[:80]
            print(f"  [{idx}] {length} tokens — {instr}...")

    # Show longest pairs
    print(f"\nTop {SHOW_LONGEST} longest pairs:")
    for length, idx, pair in sorted(lengths, reverse=True)[:SHOW_LONGEST]:
        instr = pair.get("instruction", "")[:80]
        bar   = "█" * int(30 * length / MAX_TOKENS)
        print(f"  [{idx:4d}] {length:4d} tokens  {bar}  {instr}...")

    # Token distribution buckets
    print(f"\nToken length distribution:")
    buckets = [(0, 512), (512, 1024), (1024, 2048), (2048, 3072), (3072, 4096), (4096, 99999)]
    for lo, hi in buckets:
        count = sum(1 for l in lengths_only if lo <= l < hi)
        bar   = "█" * int(20 * count / len(lengths_only))
        label = f"{lo}–{hi}" if hi < 99999 else f"{lo}+"
        print(f"  {label:12s}  {count:4d}  {bar}")

    if len(over_limit) > 0:
        print(f"\n Recommendation: filter out the {len(over_limit)} over-limit pairs")
        print("   3_prepare_finetune.py does this automatically with max_length=4096")
    else:
        print(f"\nSafe to run 3_prepare_finetune.py")


if __name__ == "__main__":
    main()