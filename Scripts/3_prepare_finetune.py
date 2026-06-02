import json
from pathlib import Path
from collections import Counter

SYSTEM_PROMPT = """You are a Unity 6 expert assistant. You write optimized, production-ready
C# scripts exclusively for Unity 6.0+. You use modern Unity 6 APIs:
- New Input System (InputAction, PlayerInput)
- Awaitable async (await Awaitable.NextFrameAsync, WaitForSecondsAsync)
- DOTS ECS (ISystem, ComponentData, EntityManager) for performance-critical code
- Jobs System + Burst Compiler (IJobParallelFor, [BurstCompile])
- Addressables for asset loading
- UI Toolkit (UIDocument, VisualElement, UXML)
- Universal Render Pipeline (URP) shader and render feature patterns
- Rigidbody.linearVelocity (not the deprecated .velocity)

After every script you explain:
1. Which Unity 6 APIs were used and what they replace
2. Why each technical choice improves performance or maintainability
3. Any Unity 6 version-specific notes"""


def to_mistral_chat(pair: dict) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": pair["instruction"]},
            {"role": "assistant", "content": pair["response"]},
        ]
    }


def prepare(
    input_file:  str = "./unity6_pairs.jsonl",
    output_file: str = "./mistral_finetune.jsonl",
    min_resp:    int = 300,
    max_resp:    int = 6000,
):
    raw = [json.loads(l) for l in Path(input_file).read_text().splitlines() if l.strip()]
    print(f"Input pairs: {len(raw)}")

    valid, skipped_short, skipped_long, skipped_no_code = 0, 0, 0, 0
    feature_counter = Counter()

    with open(output_file, "w") as out:
        for p in raw:
            resp = p.get("response", "")
            resp_len = len(resp)

            if resp_len < min_resp:
                skipped_short += 1
                continue
            if resp_len > max_resp:
                skipped_long += 1
                continue
            if "using UnityEngine" not in resp and "```csharp" not in resp: # Must contain actual C# code
                skipped_no_code += 1
                continue

            out.write(json.dumps(to_mistral_chat(p)) + "\n")
            valid += 1
            for feat in p.get("unity6_features", []):
                feature_counter[feat] += 1

    print(f"\nOutput pairs : {valid}")
    print(f"Skipped — too short  : {skipped_short}")
    print(f"Skipped — too long   : {skipped_long}")
    print(f"Skipped — no code    : {skipped_no_code}")
    print(f"\nTop Unity 6 features in dataset:")
    for feat, count in feature_counter.most_common(10):
        print(f"  {count:4d}x  {feat}")
    print(f"\nReady for fine-tuning → {output_file}")


if __name__ == "__main__":
    prepare()
