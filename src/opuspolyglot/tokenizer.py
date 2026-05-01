import json
import re
from pathlib import Path

import sentencepiece as spm
from sacremoses import MosesPunctNormalizer

LANG_CODE_RE = re.compile(r">>.+<<")
DEFAULT_SPECIAL_TOKENS = {"<unk>", "<pad>", "</s>", "<eop>", "<eod>"}
SPIECE_UNDERLINE = "▁"


class LightMarianTokenizer:
    def __init__(
        self,
        source_spm_path: str,
        target_spm_path: str,
        vocab_path: str | None = None,
        target_vocab_path: str | None = None,
        source_lang: str | None = None,
    ):
        self.spm_source = spm.SentencePieceProcessor(model_file=source_spm_path)
        self.spm_target = spm.SentencePieceProcessor(model_file=target_spm_path)
        self.normalizer = MosesPunctNormalizer(source_lang).normalize

        self.supported_language_codes = set()
        self.special_tokens = set(DEFAULT_SPECIAL_TOKENS)

        if vocab_path and Path(vocab_path).exists():
            vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
            self.supported_language_codes |= {
                tok
                for tok in vocab.keys()
                if tok.startswith(">>") and tok.endswith("<<")
            }

        if target_vocab_path and Path(target_vocab_path).exists():
            target_vocab = json.loads(
                Path(target_vocab_path).read_text(encoding="utf-8")
            )
            self.supported_language_codes |= {
                tok
                for tok in target_vocab.keys()
                if tok.startswith(">>") and tok.endswith("<<")
            }

        self.special_tokens |= self.supported_language_codes

    def normalize(self, text: str) -> str:
        return self.normalizer(text) if text else ""

    def remove_language_code(self, text: str):
        match = LANG_CODE_RE.match(text)
        code = [match.group(0)] if match else []
        if match:
            text = LANG_CODE_RE.sub("", text)
        return code, text

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        use_target_tokenizer: bool = False,
    ):
        if not use_target_tokenizer:
            text = self.normalize(text)

        code, text = self.remove_language_code(text)
        sp_model = self.spm_target if use_target_tokenizer else self.spm_source
        pieces = sp_model.encode(text, out_type=str)

        if add_special_tokens:
            return code + pieces + ["</s>"]
        return code + pieces

    def decode(
        self,
        tokens,
        *,
        use_source_tokenizer: bool = False,
        skip_special_tokens: bool = False,
    ) -> str:
        sp_model = self.spm_source if use_source_tokenizer else self.spm_target

        current_sub_tokens = []
        out_string = ""

        for token in tokens:
            if token in self.special_tokens:
                if current_sub_tokens:
                    out_string += sp_model.decode_pieces(current_sub_tokens)
                    current_sub_tokens = []

                if not skip_special_tokens:
                    out_string += token + " "
            else:
                current_sub_tokens.append(token)

        if current_sub_tokens:
            out_string += sp_model.decode_pieces(current_sub_tokens)

        return out_string.replace(SPIECE_UNDERLINE, " ").strip()
