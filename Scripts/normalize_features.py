import json
import sys
from pathlib import Path
from collections import Counter


FEATURE_MAP = {
    # Input System
    "input system":         "InputSystem",
    "new input system":     "InputSystem",
    "inputsystem":          "InputSystem",
    "unity input system":   "InputSystem",
    "inputaction":          "InputSystem",
    "playerinput":          "InputSystem",

    # Awaitable / Async
    "awaitable":            "Awaitable",
    "async awaitable":      "Awaitable",
    "async/await":          "Awaitable",
    "async await":          "Awaitable",
    "awaitable coroutine":  "Awaitable",
    "coroutine":            "Awaitable",

    # Addressables
    "addressables":         "Addressables",
    "addressable assets":   "Addressables",
    "unity addressables":   "Addressables",
    "addressableassets":    "Addressables",

    # DOTS / ECS
    "dots":                 "DOTS",
    "ecs":                  "DOTS",
    "entities":             "DOTS",
    "unity dots":           "DOTS",
    "unity ecs":            "DOTS",
    "isystem":              "DOTS",
    "icomponentdata":       "DOTS",

    # Jobs / Burst
    "jobs":                 "Jobs/Burst",
    "burst":                "Jobs/Burst",
    "job system":           "Jobs/Burst",
    "burst compiler":       "Jobs/Burst",
    "jobs/burst":           "Jobs/Burst",
    "ijob":                 "Jobs/Burst",
    "ijobparallelfor":      "Jobs/Burst",
    "burstcompile":         "Jobs/Burst",

    # UI Toolkit
    "ui toolkit":           "UIToolkit",
    "uitoolkit":            "UIToolkit",
    "uidocument":           "UIToolkit",
    "uielements":           "UIToolkit",
    "unity ui toolkit":     "UIToolkit",
    "visual element":       "UIToolkit",

    # Physics
    "linearvelocity":       "Physics",
    "linear velocity":      "Physics",
    "rigidbody":            "Physics",
    "physics":              "Physics",
    "rb.linearvelocity":    "Physics",

    # Cinemachine
    "cinemachine":          "Cinemachine",
    "cinemachine 3":        "Cinemachine",
    "virtual camera":       "Cinemachine",
    "cinemachinecamera":    "Cinemachine",

    # URP / Rendering
    "urp":                  "URP",
    "universal render pipeline": "URP",
    "shader graph":         "ShaderGraph",
    "shadergraph":          "ShaderGraph",
    "vfx graph":            "VFXGraph",
    "vfxgraph":             "VFXGraph",

    # Animation
    "animator":             "Animation",
    "animation":            "Animation",
    "blend tree":           "Animation",
    "blendtree":            "Animation",
    "animatorcontroller":   "Animation",

    # ScriptableObject
    "scriptableobject":     "ScriptableObject",
    "scriptable object":    "ScriptableObject",
}
# ─────────────────────────────────────────────────────────────────────────────


def normalize(feature: str) -> str:
    """Return canonical name for a feature string."""
    return FEATURE_MAP.get(feature.lower().strip(), feature.strip())


def normalize_pair(pair: dict) -> dict:
    """Normalize features list inside a single pair dict."""
    if "features" not in pair:
        return pair
    raw = pair["features"]
    if isinstance(raw, list):
        pair["features"] = [normalize(f) for f in raw]
    elif isinstance(raw, str):
        pair["features"] = [normalize(raw)]
    return pair


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("unity6_pairs.jsonl")

    if not path.exists():
        print(f"ERROR: {path} not found.")
        print("Run 2_transform_pairs.py first to generate it.")
        sys.exit(1)

    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    print(f"Loaded {len(pairs)} pairs from {path}")

    # Normalize
    normalized = [normalize_pair(p) for p in pairs]

    # Write back
    with open(path, "w", encoding="utf-8") as f:
        for pair in normalized:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Normalized {len(normalized)} pairs")

    # Feature distribution
    counter = Counter()
    for pair in normalized:
        for feat in pair.get("features", []):
            counter[feat] += 1

    if counter:
        print("\nFeature distribution after normalization:")
        max_count = max(counter.values())
        bar_width  = 20
        for feat, count in counter.most_common():
            bar = "█" * int(bar_width * count / max_count)
            print(f"  {count:4d}x  {bar:<{bar_width}}  {feat}")


if __name__ == "__main__":
    main()