import os

directory = 'venv/lib/python3.9/site-packages/llama_index/core'
fixed_count = 0

for root, dirs, files in os.walk(directory):
    for filename in files:
        if filename.endswith(".py"):
            filepath = os.path.join(root, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if content.startswith("from __future__ import annotations\n"):
                rest_of_file = content[len("from __future__ import annotations\n"):]
                # If there's another future import further down, my injection broke it. Remove my injection.
                if "from __future__ import" in rest_of_file:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(rest_of_file)
                    fixed_count += 1

print(f"Fixed {fixed_count} files that had double __future__ imports.")
