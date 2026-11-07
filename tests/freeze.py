"""
Freezes the current bot as a baseline for the strength gate.

    .venv/Scripts/python -m tests.freeze            # -> tests/baseline/
    .venv/Scripts/python -m tests.freeze v2         # -> tests/baseline_v2/

Copies model.py and pk_*.py into a package, renaming modules with a bl_
prefix and rewriting imports, so the frozen copy and the working copy can
be imported side by side. Not submitted (tests/ is never uploaded).
"""

import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def freeze(tag=""):
    name = "baseline" + (f"_{tag}" if tag else "")
    folder = os.path.join(ROOT, "tests", name)
    os.makedirs(folder, exist_ok=True)
    package = f"tests.{name}"
    sources = [os.path.join(ROOT, "model.py")] + sorted(
        glob.glob(os.path.join(ROOT, "pk_*.py")))
    for path in sources:
        module = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            text = f.read()
        text = re.sub(r"^from (pk_\w+|model) import",
                      lambda m: f"from {package}.bl_{m.group(1)} import",
                      text, flags=re.M)
        with open(os.path.join(folder, f"bl_{module}.py"), "w",
                  encoding="utf-8") as f:
            f.write(text)
    with open(os.path.join(folder, "__init__.py"), "w") as f:
        f.write('"""Frozen bot for the strength gate."""\n')
    print(f"froze {len(sources)} files into tests/{name}/ "
          f"(use --baseline {package}.bl_model:Model)")


if __name__ == "__main__":
    freeze(sys.argv[1] if len(sys.argv) > 1 else "")
