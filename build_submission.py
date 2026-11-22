"""Pack the project into one notebook: care_finder_submission.ipynb.

    python build_submission.py

The packed notebook carries carefinder.py, the three YAML files and the HTML template as
text cells, writes them to a temporary folder when it runs, and imports the library from
there. So the code that runs is byte for byte the code in this folder. Re-run this script
after any change; do not edit the packed notebook by hand.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = "care_finder_notebook.py"
TARGET = "care_finder_submission.ipynb"
SEP = "\n\n# COMMAND ----------\n\n"

# Shown in this order. Settings first: they are what a reader should question.
PACKED = [
    ("scoring.yaml", "Every number the method uses: distance limits, penalties, grade values, travel-time assumptions."),
    ("services.yaml", "What counts as evidence for each service, as plain term lists. Not clinically reviewed."),
    ("answer.html.j2", "The HTML template of the answer."),
    ("location_aliases.yaml", "Other names for places: old names, spellings, names in Indian scripts. A first draft."),
    ("carefinder.py", "The library: cleaning, evidence grading, place lookup, ranking, rendering. Pure pandas."),
    ("app.py", "The web app (section 8). A thin screen over the library; not needed to run this notebook."),
    ("app.yaml", "How Databricks Apps starts the web app."),
    ("requirements.txt", "Python packages the web app needs."),
]

INTRO = """# Care Finder: where can this patient get this service, and why trust the answer?

Two inputs (a service and a location), a ranked answer, and for every recommendation the exact
record entries it rests on.

**How it decides.** `score = evidence x trust x proximity`, each between 0 and 1.

- **Evidence**: does the record state the service, and does a separate entry back it up? Found by fixed
  term rules on the record's own text. Every claim on an answer is a verbatim entry from the record.
- **Trust**: starts at 1 and loses points for data problems in the record (location conflicts, merged
  records, no first-party source).
- **Proximity**: falls with distance, faster for services where minutes matter.

A language model is optional (section 7). It only splits a typed question into service and place, and
writes search terms for a service that is not predefined. It never ranks, scores or quotes.

**This is one self-contained notebook.** Section 0 holds the settings and the library as text; running
it writes them to a temporary folder and imports the library. The sections after it hold no logic:
they load data, call the library, save tables and show results.

| Section | What it holds |
|---|---|
| 0 | Settings (`scoring.yaml`, `services.yaml`), answer template, place names, the library `carefinder.py`, the web app `app.py` |
| 1 to 2 | Load the data, run the pipeline |
| 3 | Data-quality findings and diagnostics |
| 4 | Save and reload the result tables |
| 5 | **Ask: service + location** |
| 6 | Validation sample |
| 7 | Optional: questions in plain words, in any language |
| 8 | Optional: write the folder for the web app |
| 9 | **District view**: where care is out of reach, joined with the NFHS-5 survey; maps, priorities, recommendations |

**Run order.** Run all once (about a minute). After that, change the widgets at the top and re-run only
section 5 or 7."""

SECTION0 = """## 0. Settings and code

Each cell below holds one file as text. Everything that is a judgment call is in the first two."""

LOADER = '''# Write the files above to a temporary folder and import the library from it.
import importlib, json, os, sys, tempfile
from collections import Counter

import numpy as np
import pandas as pd

PACK_DIR = os.path.join(tempfile.gettempdir(), "care_finder_packed")
os.makedirs(PACK_DIR, exist_ok=True)
for _name, _text in FILES.items():
    with open(os.path.join(PACK_DIR, _name), "w", encoding="utf-8") as _fh:
        _fh.write(_text)
if PACK_DIR not in sys.path:
    sys.path.insert(0, PACK_DIR)
sys.modules.pop("carefinder", None)      # so that an edited cell above takes effect on re-run
import carefinder as cf
print({k: v for k, v in cf.load_settings().items() if k != "directory"})'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(keepends=True)}


def read(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as fh:
        return fh.read()


def packed_cell(name, about, first):
    text = read(name)
    if "'''" in text or text.rstrip("\n").endswith("\\"):
        raise ValueError(f"{name} cannot be embedded as a raw string: it contains ''' or ends with a backslash.")
    head = f"# {name}: {about}\n" + ("FILES = {}\n" if first else "")
    return code(f"{head}FILES[{name!r}] = r'''{text}'''")


def build():
    parts = read(SOURCE).split(SEP)
    parts[0] = parts[0].replace("# Databricks notebook source\n", "", 1)
    cells, replaced = [], False
    for i, part in enumerate(parts):
        body = part.strip("\n")
        if body.startswith("# MAGIC %md"):
            text = "\n".join(line[len("# MAGIC"):].lstrip(" ") if line.startswith("# MAGIC ") else "" for line in body.split("\n"))
            text = text.replace("%md\n", "", 1).replace("%md", "", 1)
            if i == 0:
                text = INTRO
            elif text.lstrip().startswith("## 0."):
                text = SECTION0
            cells.append(md(text))
        elif "import carefinder as cf" in body:
            cells += [packed_cell(n, about, first=(j == 0)) for j, (n, about) in enumerate(PACKED)]
            cells.append(code(LOADER))
            replaced = True
        else:
            cells.append(code(body))
    if not replaced:
        raise ValueError("Did not find the cell that imports carefinder in " + SOURCE)
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    with open(os.path.join(HERE, TARGET), "w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)
    # Check: running the packed cells reproduces every file exactly.
    ns = {}
    for c in cells:
        src = "".join(c["source"])
        if c["cell_type"] == "code" and "FILES[" in src and "r'''" in src:
            exec(src, ns)
    for name, _ in PACKED:
        assert ns["FILES"][name] == read(name), f"{name} was changed by packing"
    return len(cells), os.path.getsize(os.path.join(HERE, TARGET))


if __name__ == "__main__":
    n, size = build()
    print(f"wrote {TARGET}: {n} cells, {size / 1024:.0f} KB")
