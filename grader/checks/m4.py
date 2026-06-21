"""M4 checkpoint: char tokenizer. Parameterized by the tokenizer class."""

from __future__ import annotations

from grader.checks._util import property_check

_S = "hello world\nthe quick brown fox"


def check_tokenizer(TokenizerClass):
    def roundtrip():
        tk = TokenizerClass(_S)
        return (
            tk.decode(tk.encode(_S)) == _S
            and tk.encode(tk.decode(list(range(tk.vocab_size)))) == list(range(tk.vocab_size))
        )

    def vocab_integrity():
        tk = TokenizerClass(_S)
        return tk.vocab_size == len(set(_S)) and all(0 <= i < tk.vocab_size for i in tk.encode(_S))

    return [
        property_check("roundtrip", roundtrip),
        property_check("vocab_integrity", vocab_integrity),
    ]


def build_checks():
    from reference.tokenizer.char_tokenizer import CharTokenizer
    return check_tokenizer(CharTokenizer)
