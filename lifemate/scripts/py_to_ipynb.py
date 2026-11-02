"""Đổi file .py dạng percent (# %% / # %% [markdown]) thành .ipynb.

Dùng: python3 scripts/py_to_ipynb.py notebooks/lifemate_pipeline.py
"""
import json
import re
import sys
from pathlib import Path


def convert(src_path):
    src = Path(src_path).read_text(encoding="utf-8")
    cells = []
    for chunk in re.split(r"^# %%", src, flags=re.M)[1:]:
        header, _, body = chunk.partition("\n")
        body = body.strip("\n")
        if header.strip() == "[markdown]":
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.split("\n"))
            cell = {"cell_type": "markdown", "metadata": {}, "source": text}
        else:
            cell = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": body}
        lines = cell["source"].split("\n")
        cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        cells.append(cell)
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path(src_path).with_suffix(".ipynb")
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{out}: {len(cells)} cells")


if __name__ == "__main__":
    convert(sys.argv[1])
