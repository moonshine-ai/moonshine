#!/usr/bin/env python3
"""
Convert tokenizer files to the BinTokenizer format used by moonshine.

Supports:
- SentencePiece .model files (protobuf format)
- HuggingFace tokenizer.json files (BPE / WordPiece dict vocabularies, and
  Unigram list vocabularies with scores)

The flat BinTokenizer format is a run of entries, one per token id:
- 1 byte: length (if < 128)
- 2 bytes: length (if >= 128, first byte has high bit set)
- N bytes: token string

Empty tokens (length 0) are stored as a single 0x00 byte.

A Unigram vocabulary needs more than the spellings: the encoding is the
Viterbi path that maximizes the summed piece scores, so the scores have to
travel with the file. Those are written in the headed form, which
``core/bin-tokenizer/bin-tokenizer.h`` documents:

    FF FF 'B' 'T' 'O' 'K'   magic (a flat file cannot start with FF FF)
    u8  version = 1
    u8  encoding            0 longest-match, 1 BPE, 2 Unigram
    u32 count               little-endian
    u32 unk_id              little-endian, 0xFFFFFFFF when none
    entries                 exactly `count`, as in the flat form
    f32 scores[count]       little-endian, Unigram only

Vocabularies without scores keep the flat form, so existing files are
unchanged byte for byte.

Usage:
    python convert_tokenizer.py <input_file> <output_file>
    python convert_tokenizer.py tokenizer.model tokenizer.bin
    python convert_tokenizer.py tokenizer.json tokenizer.bin
"""

import argparse
import json
import struct
import sys
from pathlib import Path

HEADER_MAGIC = b"\xff\xffBTOK"
HEADER_VERSION = 1
ENCODING_LONGEST_MATCH = 0
ENCODING_BPE = 1
ENCODING_UNIGRAM = 2
NO_UNK_ID = 0xFFFFFFFF


def encode_entries(tokens: list[bytes]) -> bytes:
    """The flat run of length-prefixed entries."""
    out = bytearray()
    for token_bytes in tokens:
        length = len(token_bytes)
        if length == 0:
            # Empty token
            out += b"\x00"
        elif length < 128:
            # Single byte length
            out.append(length)
            out += token_bytes
        else:
            # Two byte length: first byte has high bit set
            # length = (second_byte * 128) + first_byte - 128
            # So: first_byte = (length % 128) + 128, second_byte = length // 128
            out.append((length % 128) + 128)
            out.append(length // 128)
            out += token_bytes
    return bytes(out)


def bin_tokenizer_bytes(tokens: list[bytes], scores: list[float] | None = None,
                        encoding: int | None = None, unk_id: int | None = None) -> bytes:
    """The file contents: flat when there is nothing but spellings, headed otherwise."""
    if scores is None and encoding is None:
        return encode_entries(tokens)
    if encoding is None:
        encoding = ENCODING_UNIGRAM
    if encoding == ENCODING_UNIGRAM and (scores is None or len(scores) != len(tokens)):
        raise ValueError("Unigram vocabularies need one score per token")
    out = bytearray(HEADER_MAGIC)
    out += struct.pack("<BBII", HEADER_VERSION, encoding, len(tokens),
                       NO_UNK_ID if unk_id is None else unk_id)
    out += encode_entries(tokens)
    if encoding == ENCODING_UNIGRAM:
        out += struct.pack(f"<{len(scores)}f", *scores)
    return bytes(out)


def write_bin_tokenizer(tokens: list[bytes], output_path: str, scores: list[float] | None = None,
                        encoding: int | None = None, unk_id: int | None = None) -> None:
    """Write tokens (and, for Unigram, their scores) to BinTokenizer format."""
    with open(output_path, "wb") as f:
        f.write(bin_tokenizer_bytes(tokens, scores, encoding, unk_id))


def convert_sentencepiece(input_path: str, output_path: str) -> None:
    """Convert SentencePiece .model file to BinTokenizer format."""
    try:
        import sentencepiece as spm
    except ImportError:
        print("Error: sentencepiece package required. Install with: pip install sentencepiece")
        sys.exit(1)

    sp = spm.SentencePieceProcessor()
    sp.load(input_path)

    vocab_size = sp.get_piece_size()
    print(f"Vocabulary size: {vocab_size}")

    # Extract all tokens
    tokens = []
    for i in range(vocab_size):
        piece = sp.id_to_piece(i)
        # SentencePiece uses ▁ (U+2581) for space, convert to standard representation
        token_bytes = piece.encode("utf-8")
        tokens.append(token_bytes)

    # Print some stats
    print(f"Special tokens:")
    print(f"  PAD (0): {repr(sp.id_to_piece(0))}")
    print(f"  EOS (1): {repr(sp.id_to_piece(1))}")
    print(f"  BOS (2): {repr(sp.id_to_piece(2))}")
    print(f"  UNK (3): {repr(sp.id_to_piece(3))}")

    write_bin_tokenizer(tokens, output_path)
    print(f"Wrote {len(tokens)} tokens to {output_path}")


def unigram_from_huggingface_json(data: dict) -> tuple[list[bytes], list[float], int]:
    """Spellings, scores and unknown id of a ``tokenizer.json`` whose model is Unigram."""
    model = data["model"]
    tokens = [piece.encode("utf-8") for piece, _score in model["vocab"]]
    scores = [float(score) for _piece, score in model["vocab"]]
    unk_id = model.get("unk_id")
    for added in data.get("added_tokens", []):
        token_id = added.get("id")
        content = added.get("content", "")
        if token_id is None:
            continue
        if token_id >= len(tokens):
            tokens.extend([b""] * (token_id - len(tokens) + 1))
            scores.extend([0.0] * (token_id - len(scores) + 1))
        tokens[token_id] = content.encode("utf-8")
    return tokens, scores, unk_id


def convert_huggingface_json(input_path: str, output_path: str) -> None:
    """Convert HuggingFace tokenizer.json to BinTokenizer format."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Get the vocabulary from the model section
    model = data.get("model", {})
    vocab = model.get("vocab", {})

    if not vocab:
        print("Error: Could not find vocabulary in tokenizer.json")
        sys.exit(1)

    if model.get("type") == "Unigram":
        tokens, scores, unk_id = unigram_from_huggingface_json(data)
        print(f"Unigram vocabulary size: {len(tokens)} (unk_id {unk_id})")
        for i in range(min(10, len(tokens))):
            print(f"  {i}: {repr(tokens[i].decode('utf-8', errors='replace'))} {scores[i]:.3f}")
        write_bin_tokenizer(tokens, output_path, scores=scores, encoding=ENCODING_UNIGRAM,
                            unk_id=unk_id)
        print(f"Wrote {len(tokens)} tokens with scores to {output_path}")
        return

    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # Create token list indexed by ID
    tokens = [b""] * vocab_size
    for token_str, token_id in vocab.items():
        if token_id < vocab_size:
            tokens[token_id] = token_str.encode("utf-8")

    # Handle added_tokens which may override or add special tokens
    added_tokens = data.get("added_tokens", [])
    for added in added_tokens:
        token_id = added.get("id")
        content = added.get("content", "")
        if token_id is not None and token_id < len(tokens):
            tokens[token_id] = content.encode("utf-8")
        elif token_id is not None and token_id >= len(tokens):
            # Extend tokens list if needed
            tokens.extend([b""] * (token_id - len(tokens) + 1))
            tokens[token_id] = content.encode("utf-8")

    # Print some sample tokens
    print(f"Sample tokens:")
    for i in range(min(10, len(tokens))):
        print(f"  {i}: {repr(tokens[i].decode('utf-8', errors='replace'))}")

    write_bin_tokenizer(tokens, output_path)
    print(f"Wrote {len(tokens)} tokens to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert tokenizer files to BinTokenizer format"
    )
    parser.add_argument("input", help="Input tokenizer file (.model or .json)")
    parser.add_argument("output", help="Output .bin file")

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    if input_path.suffix == ".model":
        print(f"Converting SentencePiece model: {input_path}")
        convert_sentencepiece(str(input_path), args.output)
    elif input_path.suffix == ".json":
        print(f"Converting HuggingFace tokenizer: {input_path}")
        convert_huggingface_json(str(input_path), args.output)
    else:
        print(f"Error: Unknown file type: {input_path.suffix}")
        print("Supported formats: .model (SentencePiece), .json (HuggingFace)")
        sys.exit(1)

    # Verify the output
    output_size = Path(args.output).stat().st_size
    print(f"Output file size: {output_size:,} bytes")


if __name__ == "__main__":
    main()
