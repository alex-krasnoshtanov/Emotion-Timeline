"""Russian to English, so the model this project trained can be asked about it.

`Helsinki-NLP/opus-mt-ru-en` rather than the local LLM server the original
pipeline called. It is 300 MB, runs on the same card as everything else, needs no
server and no key, and -- the part that matters here -- it is a fixed artefact, so
the translation behind a published number can be reproduced. An LLM behind an
HTTP endpoint cannot be pinned, and a number produced through one is not
checkable a year later.

Greedy decoding, no beams. Beam search would give slightly better translations and
make the result depend on a beam width nobody recorded; this is the same argument
the split makes for a fixed seed. Approach A is being measured against a native
Russian model, so what matters is that the translation is *the same translation*
every time, not that it is the best available.
"""

from __future__ import annotations

from collections.abc import Sequence

MODEL = "Helsinki-NLP/opus-mt-ru-en"

#: The corpus's longest row is 357 characters, so this truncates nothing real.
MAX_TOKENS = 192


def translate(
    texts: Sequence[str],
    batch_size: int = 64,
    max_tokens: int = MAX_TOKENS,
    model_id: str = MODEL,
    progress: object = None,
) -> list[str]:  # pragma: no cover - needs the model and a GPU
    """English for each row, in order."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    out: list[str] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_tokens,
            ).to(device)
            generated = model.generate(**encoded, max_new_tokens=max_tokens, num_beams=1)
            out.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
            if callable(progress) and start % (batch_size * 10) == 0:
                progress(f"  translated {min(start + batch_size, len(texts)):,}/{len(texts):,}")
    return out
