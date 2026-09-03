"""
get_real_data.py - fetch training corpora into Data_Training/sources/.

What was wrong with the previous version:

  1. It wrote HuggingFaceFW/fineweb-edu, an English-only corpus, into a file
     named languages/en_ar.txt. The Arabic half never existed. For an
     Arabic-first model that was the actual data problem, not a naming nit.
  2. It reassigned `kwargs` inside the fineweb-edu branch, silently throwing
     away any data_dir the caller had passed.
  3. It pulled bigcode/starcoderdata data_dir="python" into a file named
     programming/code_cpp_assembly.txt.
  4. bigcode/starcoderdata is gated. Without an accepted licence and a token
     the whole script raised instead of skipping one source.

Each source now writes the language it actually contains, failures are
per-source, and gated datasets are reported rather than fatal.

Usage:
    python get_real_data.py                  # all ungated sources
    HF_TOKEN=hf_xxx python get_real_data.py  # include gated sources
"""

from __future__ import annotations

import os
import sys

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("datasets not installed:  pip install datasets")

OUT_ROOT = os.path.join("Data_Training", "sources")
LANG_DIR = os.path.join(OUT_ROOT, "languages")
CODE_DIR = os.path.join(OUT_ROOT, "programming")

MIN_LINE = 40
MAX_LINE = 600

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def save_texts(
    dataset: str,
    out_file: str,
    *,
    config: str | None = None,
    data_dir: str | None = None,
    text_col: str = "text",
    limit: int = 2000,
) -> int:
    """Stream `dataset` and write up to `limit` cleaned lines to `out_file`.

    Returns the number of lines written. Never raises: a failed source is
    reported and skipped so the remaining sources still run.
    """
    label = f"{dataset}" + (f":{config}" if config else "") + (f" [{data_dir}]" if data_dir else "")
    print(f"-> {label}")

    kwargs: dict[str, object] = {"split": "train", "streaming": True}
    if data_dir:
        kwargs["data_dir"] = data_dir
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN

    args = [dataset] + ([config] if config else [])

    try:
        ds = load_dataset(*args, **kwargs)
    except Exception as err:                      # gated, offline, renamed
        print(f"   SKIP  {type(err).__name__}: {str(err).splitlines()[0][:160]}")
        return 0

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    count = 0
    try:
        with open(out_file, "w", encoding="utf-8") as fh:
            for row in ds:
                text = row.get(text_col) or row.get("text") or row.get("content") or ""
                if not text:
                    continue
                for line in str(text).split("\n"):
                    line = line.strip()
                    if MIN_LINE < len(line) < MAX_LINE:
                        fh.write(line + "\n")
                        count += 1
                        if count >= limit:
                            raise StopIteration
    except StopIteration:
        pass
    except Exception as err:
        print(f"   PARTIAL  {type(err).__name__}: {str(err).splitlines()[0][:160]}")

    size = os.path.getsize(out_file) if os.path.exists(out_file) else 0
    print(f"   {count} lines, {size} bytes -> {out_file}")
    return count


SOURCES = [
    # Arabic. Ungated, so this one actually runs without a token.
    dict(
        dataset="wikimedia/wikipedia",
        config="20231101.ar",
        out_file=os.path.join(LANG_DIR, "ar.txt"),
        text_col="text",
        limit=4000,
    ),
    # English.
    dict(
        dataset="HuggingFaceFW/fineweb-edu",
        out_file=os.path.join(LANG_DIR, "en.txt"),
        text_col="text",
        limit=2000,
    ),
    # C source. Gated: accept the licence on the dataset page, then set
    # HF_TOKEN. Skipped with a message otherwise.
    dict(
        dataset="bigcode/starcoderdata",
        data_dir="c",
        out_file=os.path.join(CODE_DIR, "code_c.txt"),
        text_col="content",
        limit=2000,
    ),
]


def main() -> int:
    os.makedirs(LANG_DIR, exist_ok=True)
    os.makedirs(CODE_DIR, exist_ok=True)

    if not HF_TOKEN:
        print("note: HF_TOKEN unset - gated sources will be skipped\n")

    total = 0
    for source in SOURCES:
        out_file = source.pop("out_file")
        total += save_texts(out_file=out_file, **source)

    print(f"\ntotal lines = {total}")
    if total == 0:
        print("FAIL: no data written")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
