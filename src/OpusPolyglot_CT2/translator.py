import ctranslate2
from .tokenizer import LightMarianTokenizer

translator = ctranslate2.Translator(
    "./opus-mt-mul-en-ct2",
    device="cpu",
    inter_threads=2,
    intra_threads=4,
    compute_type="int8",
)

tokenizer = LightMarianTokenizer(
    source_spm_path="./opus-mt-mul-en-ct2/source.spm",
    target_spm_path="./opus-mt-mul-en-ct2/target.spm",
    vocab_path="./opus-mt-mul-en-ct2/vocab.json",
    target_vocab_path="./opus-mt-mul-en-ct2/target_vocab.json",
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
