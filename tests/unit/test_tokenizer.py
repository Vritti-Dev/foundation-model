from reference.tokenizer.char_tokenizer import CharTokenizer
S = "hello world\nthe quick brown fox"

def test_roundtrip_identity():
    tk = CharTokenizer(S)
    assert tk.decode(tk.encode(S)) == S
    assert tk.encode(tk.decode(list(range(tk.vocab_size)))) == list(range(tk.vocab_size))

def test_vocab_integrity():
    tk = CharTokenizer(S)
    assert tk.vocab_size == len(set(S))
    assert all(0 <= i < tk.vocab_size for i in tk.encode(S))
