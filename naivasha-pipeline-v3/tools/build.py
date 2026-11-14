"""Build notebooks (.ipynb) and IDE-ready percent-format scripts (.py) from tools/src/*.py.

Source files use the "percent" cell format ("# %%" = code cell, "# %% [markdown]" = text cell)
and may contain "# @@snippet_name@@" lines that are replaced with tools/snippets/snippet_name.py.

Usage:  python tools/build.py
"""
import re
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC, SNIP = ROOT / "tools" / "src", ROOT / "tools" / "snippets"


def expand(text):
    def repl(m):
        return (SNIP / f"{m.group(1)}.py").read_text().rstrip("\n")
    return re.sub(r"^# @@(\w+)@@\s*$", repl, text, flags=re.M)


def split_cells(text):
    cells, kind, buf = [], None, []
    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if kind == "markdown":
            body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.split("\n"))
            cells.append(nbf.v4.new_markdown_cell(body))
        elif body.strip():
            cells.append(nbf.v4.new_code_cell(body))
    for line in text.split("\n"):
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        else:
            buf.append(line)
    flush()
    return cells


def main():
    (ROOT / "notebooks").mkdir(exist_ok=True)
    (ROOT / "scripts").mkdir(exist_ok=True)
    for src in sorted(SRC.glob("*.py")):
        text = expand(src.read_text())
        (ROOT / "scripts" / src.name).write_text(text.rstrip("\n") + "\n")
        nb = nbf.v4.new_notebook()
        nb.cells = split_cells(text)
        nb.metadata = {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        }
        nbf.write(nb, ROOT / "notebooks" / (src.stem + ".ipynb"))
        print(f"built {src.stem}: {len(nb.cells)} cells")


if __name__ == "__main__":
    main()
