import os

import ctranslate2
from tokenizer import LightMarianTokenizer

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_OPUS_MT_MODEL_DIR = os.path.join(_BASE_DIR, "opus-mt-mul-en-ct2")

translator = ctranslate2.Translator(
    _OPUS_MT_MODEL_DIR,
    device="cpu",
    inter_threads=2,
    intra_threads=4,
    compute_type="int8",
)

tokenizer = LightMarianTokenizer(
    source_spm_path=os.path.join(_OPUS_MT_MODEL_DIR, "source.spm"),
    target_spm_path=os.path.join(_OPUS_MT_MODEL_DIR, "target.spm"),
    vocab_path=os.path.join(_OPUS_MT_MODEL_DIR, "vocab.json"),
    target_vocab_path=os.path.join(_OPUS_MT_MODEL_DIR, "target_vocab.json"),
)


def translate(text: str) -> str:
    source_tokens = tokenizer.encode(text, add_special_tokens=True)

    results = translator.translate_batch(
        [source_tokens],
        max_decoding_length=128,
        no_repeat_ngram_size=4,
        beam_size=2,
    )

    output_tokens = results[0].hypotheses[0]
    return tokenizer.decode(output_tokens, skip_special_tokens=True)
