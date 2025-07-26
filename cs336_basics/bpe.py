import os
from collections import defaultdict

def merge(indices, max_pair, token_id):
    """Merge the indices based on the max_pair and return the new indices."""
    index1, index2 = max_pair
    new_indices = []
    skip_next = False

    for i in range(len(indices)):
        if skip_next:
            skip_next = False
            continue
        if i < len(indices) - 1 and indices[i] == index1 and indices[i + 1] == index2:
            new_indices.append(token_id)
            skip_next = True
        else:
            new_indices.append(indices[i])

    return new_indices

def create_bpe_tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    with open(input_path, "rb") as f:
        corpus = f.read()

    utf8_decoded = corpus.decode("utf-8")
    print(utf8_decoded)
    utf8_encoded = utf8_decoded.encode("utf-8")
    print(utf8_encoded)
    vocabs = {}
    merges = []

    for special_token in special_tokens:
        token_byte = special_token.encode("utf-8")
        token_id = len(vocabs)
        vocabs[token_id] = token_byte
    
    for i in range(256):
        token_byte = bytes([i])
        if token_byte not in vocabs.values():
            token_id = len(vocabs)
            vocabs[token_id] = token_byte

    indices = list(map(int, utf8_encoded))
    special_token_bytes = set(token.encode("utf-8") for token in special_tokens)
    while len(vocabs) < vocab_size:
        counts = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):
            if vocabs[index1] in special_token_bytes or vocabs[index2] in special_token_bytes:
                continue
            counts[(index1, index2)] += 1
        if not counts:
            break
        max_pair = max(counts, key=counts.get)
        index1, index2 = max_pair
        token_id = len(vocabs)
        vocabs[token_id] = vocabs[index1] + vocabs[index2]
        merges.append((vocabs[index1], vocabs[index2])) 
        indices = merge(indices, max_pair, token_id)

    return vocabs, merges

if __name__ == "__main__":
    vocab, merges = create_bpe_tokenizer(
        input_path="test.txt",
        vocab_size=1000,
        special_tokens=["<unk>", "<pad>", "<s>", "</s>"],
    )
    print(vocab)
    print(merges)
