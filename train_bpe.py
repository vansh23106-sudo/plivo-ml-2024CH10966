"""Train a fast Byte-Level BPE Tokenizer on train_corpus.txt.
Produces bpe_vocab.json for tokenizer.py to load.
"""
import json
import os
import time
from collections import Counter


def train_bpe(text_path, target_vocab_size=1024, out_path="bpe_vocab.json"):
    print(f"Training BPE tokenizer to vocab size {target_vocab_size} on {text_path}...")
    t0 = time.time()
    with open(text_path, "rb") as f:
        raw_bytes = f.read()

    ids = list(raw_bytes)

    # Split into word chunks to accelerate pair counting
    words = []
    current_word = []
    for b in ids:
        current_word.append(b)
        if b in (32, 10):
            words.append(tuple(current_word))
            current_word = []
    if current_word:
        words.append(tuple(current_word))

    word_counts = Counter(words)
    merges = {}  # "p0,p1" -> new_id
    vocab = {i: bytes([i]) for i in range(256)}
    num_merges = target_vocab_size - 256

    for i in range(num_merges):
        pair_counts = Counter()
        for word, count in word_counts.items():
            for pair in zip(word[:-1], word[1:]):
                pair_counts[pair] += count

        if not pair_counts:
            break

        best_pair, freq = pair_counts.most_common(1)[0]
        new_id = 256 + i
        pair_key = f"{best_pair[0]},{best_pair[1]}"
        merges[pair_key] = new_id
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

        new_word_counts = {}
        for word, count in word_counts.items():
            new_word = []
            j = 0
            while j < len(word):
                if j < len(word) - 1 and (word[j], word[j + 1]) == best_pair:
                    new_word.append(new_id)
                    j += 2
                else:
                    new_word.append(word[j])
                    j += 1
            new_word_counts[tuple(new_word)] = count
        word_counts = new_word_counts

        if (i + 1) % 250 == 0 or i == num_merges - 1:
            print(f"Merge {i+1:4d}/{num_merges}: pair {best_pair} -> token {new_id} ({time.time()-t0:.1f}s)")

    data = {
        "vocab_size": len(vocab),
        "merges": merges,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved BPE vocab ({len(vocab)} tokens) to {out_path} in {time.time()-t0:.2f}s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/train_corpus.txt")
    ap.add_argument("--vocab_size", type=int, default=1024)
    ap.add_argument("--out", default="bpe_vocab.json")
    args = ap.parse_args()
    train_bpe(args.data, args.vocab_size, args.out)
