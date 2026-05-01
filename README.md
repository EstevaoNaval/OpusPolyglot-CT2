# OpusPolyglot-CT2

OpusPolyglot-CT2 translates from multiple languages to English using [CTranslate2](https://github.com/opennmt/ctranslate2) and a light MarianTokenizer, avoiding the need for Transformers and heavy DeepLearning libs.

For neural machine translation, it's used [Opus-MT](https://github.com/Helsinki-NLP/OPUS-MT) [opus-mt-mul-en](https://huggingface.co/Helsinki-NLP/opus-mt-mul-en) model.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/EstevaoNaval/OpusPolyglot-CT2
```

For Developers:

```bash
pip install -e .
```

## Quick use

Run from the repository root so the model files resolve at `./opus-mt-mul-en-ct2`.

```python
from opuspolyglot.translator import translate

print(translate("Olá mundo!"))
```

## License

This project is licensed under the Apache v2.0 - see the [LICENSE](LICENSE) file for details.

## References

- [Opus-MT](https://github.com/Helsinki-NLP/OPUS-MT)
- [opus-mt-mul-en](https://huggingface.co/Helsinki-NLP/opus-mt-mul-en)
