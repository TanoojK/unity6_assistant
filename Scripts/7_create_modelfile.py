import glob

files = glob.glob(r"F:\cs_llm\proj\unity_gguf\*.gguf")
if not files:
    print("ERROR: No .gguf file found in F:\\cs_llm\\proj\\unity_gguf\\")
    print("Check the folder name and contents.")
else:
    gguf_path = files[0].replace("\\", "\\\\")
    content = f"""FROM {gguf_path}

SYSTEM \"\"\"You are a Unity 6 expert assistant. You write optimized C# scripts
for Unity 6.0+ using modern APIs: InputSystem, Awaitable, linearVelocity,
Addressables, UIToolkit, DOTS, Jobs/Burst. After every script explain each
Unity 6 choice and why it improves performance.\"\"\"

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 2048
"""
    with open(r"F:\cs_llm\proj\Modelfile", "w") as f:
        f.write(content)
    print(f"Modelfile created using: {gguf_path}")
    