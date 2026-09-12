"""Russian to English, so the model this project trained can be asked about it.

`Helsinki-NLP/opus-mt-ru-en` rather than the local LLM server the original
pipeline called. It is 300 MB, runs on the same card as everything else, needs no
server and no key, and -- the part that matters here -- it is a fixed artefact, so
the translation behind a published number can be reproduced. An LLM behind an
HTTP endpoint cannot be pinned, and a number produced through one is not
checkable a year later.

Greedy decoding, no beams. Beam search would give slightly better translations and
make the result depend on a beam width nobody recorded; this is the same argument
the split makes for a fixed seed. What matters is that the translation is *the
same translation* every time, not that it is the best available.

**One sentence at a time, and that is not a detail.** Marian's opus-mt models are
trained on sentence pairs. Handed two sentences they routinely translate the
first and emit end-of-sequence, and nothing in the output says so. The first
version of this module passed whole rows: **88% of the multi-sentence rows in the
Russian evaluation set came back with sentences missing**, a quarter of all rows
lost more than a quarter of their characters, and the approach built on it scored
0.3631 -- a number that was measuring the harness rather than the idea. Splitting
first and rejoining after costs one pass over a regex and fixes it.

**The output is cleaned the way the training set was.** The English model was
trained on text that had been through `data/clean.py` -- entity masking,
placeholder normalisation, emoticon stripping, the lot. Feeding it raw translator
output asks it about a surface form it never saw. :func:`to_model_english` runs
the same three passes the dataset build runs, in the same order.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from emotion_timeline.data import clean

MODEL = "Helsinki-NLP/opus-mt-ru-en"
BACKWARDS = "Helsinki-NLP/opus-mt-en-ru"

#: A second engine, an order of magnitude larger, so "a better translator would
#: fix it" is a measurement rather than an argument. NLLB needs the language on
#: both ends rather than one model per direction.
NLLB = "facebook/nllb-200-distilled-600M"
NLLB_CODES = {"ru": "rus_Cyrl", "en": "eng_Latn"}

#: Comfortably longer than any single sentence in the corpus, which is the only
#: thing this ever sees now.
MAX_TOKENS = 192

#: Sentence boundaries, kept deliberately simple: a terminator followed by
#: whitespace. Abbreviations will occasionally split early, which costs a little
#: context and never drops a clause -- the failure this exists to prevent.
SENTENCE = re.compile(r"(?<=[.!?\u2026])\s+")


def sentences(text: str) -> list[str]:
    """The pieces to translate separately, in order. Never empty for non-empty text."""
    parts = [part.strip() for part in SENTENCE.split(text) if part.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def to_model_english(text: str) -> str:
    """Translator output in the surface form the English model was trained on."""
    return clean.tidy(clean.clean_late(clean.clean_early(text)))


def _decode(  # pragma: no cover - needs the model and a GPU
    pieces: list[str],
    model_id: str,
    source: str,
    target: str,
    batch_size: int,
    max_tokens: int,
    progress: object,
) -> list[str]:
    """Every sentence through one engine, in one flat batch pass."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    nllb = model_id == NLLB
    tokenizer = (
        AutoTokenizer.from_pretrained(model_id, src_lang=NLLB_CODES[source])
        if nllb
        else AutoTokenizer.from_pretrained(model_id)
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    extra = (
        {"forced_bos_token_id": tokenizer.convert_tokens_to_ids(NLLB_CODES[target])} if nllb else {}
    )

    out: list[str] = []
    with torch.no_grad():
        for start in range(0, len(pieces), batch_size):
            encoded = tokenizer(
                pieces[start : start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_tokens,
            ).to(device)
            generated = model.generate(**encoded, max_new_tokens=max_tokens, num_beams=1, **extra)
            out.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
            if callable(progress) and start % (batch_size * 20) == 0:
                progress(f"  translated {min(start + batch_size, len(pieces)):,}/{len(pieces):,}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def translate(
    texts: Sequence[str],
    batch_size: int = 64,
    max_tokens: int = MAX_TOKENS,
    model_id: str = MODEL,
    progress: object = None,
    clean_output: bool = True,
    source: str = "ru",
    target: str = "en",
) -> list[str]:  # pragma: no cover - needs the model and a GPU
    """English for each row, in order, translated a sentence at a time.

    Every sentence of every row goes into one flat batch, so a row of eight
    sentences costs no more passes than eight rows of one, and the pieces are
    reassembled at the end.
    """
    pieces: list[str] = []
    owners: list[int] = []
    for index, text in enumerate(texts):
        for piece in sentences(text):
            pieces.append(piece)
            owners.append(index)

    translated = _decode(pieces, model_id, source, target, batch_size, max_tokens, progress)
    out = [""] * len(texts)
    for owner, piece in zip(owners, translated, strict=True):
        out[owner] = f"{out[owner]} {piece}".strip()
    return [to_model_english(row) for row in out] if clean_output else out
