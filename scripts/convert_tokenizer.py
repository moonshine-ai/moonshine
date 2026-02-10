#!/usr/bin/env python3
"""
Convert tokenizer files to the BinTokenizer format used by moonshine.

Supports:
- SentencePiece .model files (protobuf format)
- HuggingFace tokenizer.json files

The BinTokenizer format is a simple binary format where each token is stored as:
- 1 byte: length (if < 128)
- 2 bytes: length (if >= 128, first byte has high bit set)
- N bytes: token string

Empty tokens (length 0) are stored as a single 0x00 byte.

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


def write_bin_tokenizer(tokens: list[bytes], output_path: str) -> None:
    """Write tokens to BinTokenizer format."""
    with open(output_path, 'wb') as f:
        for token_bytes in tokens:
            length = len(token_bytes)
            if length == 0:
                # Empty token
                f.write(b'\x00')
            elif length < 128:
                # Single byte length
                f.write(bytes([length]))
                f.write(token_bytes)
            else:
                # Two byte length: first byte has high bit set
                # length = (second_byte * 128) + first_byte - 128
                # So: first_byte = (length % 128) + 128, second_byte = length // 128
                first_byte = (length % 128) + 128
                second_byte = length // 128
                f.write(bytes([first_byte, second_byte]))
                f.write(token_bytes)


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
        token_bytes = piece.encode('utf-8')
        tokens.append(token_bytes)
    
    # Print some stats
    print(f"Special tokens:")
    print(f"  PAD (0): {repr(sp.id_to_piece(0))}")
    print(f"  EOS (1): {repr(sp.id_to_piece(1))}")
    print(f"  BOS (2): {repr(sp.id_to_piece(2))}")
    print(f"  UNK (3): {repr(sp.id_to_piece(3))}")
    
    write_bin_tokenizer(tokens, output_path)
    print(f"Wrote {len(tokens)} tokens to {output_path}")


def convert_huggingface_json(input_path: str, output_path: str) -> None:
    """Convert HuggingFace tokenizer.json to BinTokenizer format."""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get the vocabulary from the model section
    model = data.get('model', {})
    vocab = model.get('vocab', {})
    
    if not vocab:
        print("Error: Could not find vocabulary in tokenizer.json")
        sys.exit(1)
    
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create token list indexed by ID
    tokens = [b''] * vocab_size
    for token_str, token_id in vocab.items():
        if token_id < vocab_size:
            tokens[token_id] = token_str.encode('utf-8')
    
    # Handle added_tokens which may override or add special tokens
    added_tokens = data.get('added_tokens', [])
    for added in added_tokens:
        token_id = added.get('id')
        content = added.get('content', '')
        if token_id is not None and token_id < len(tokens):
            tokens[token_id] = content.encode('utf-8')
        elif token_id is not None and token_id >= len(tokens):
            # Extend tokens list if needed
            tokens.extend([b''] * (token_id - len(tokens) + 1))
            tokens[token_id] = content.encode('utf-8')
    
    # Print some sample tokens
    print(f"Sample tokens:")
    for i in range(min(10, len(tokens))):
        print(f"  {i}: {repr(tokens[i].decode('utf-8', errors='replace'))}")
    
    write_bin_tokenizer(tokens, output_path)
    print(f"Wrote {len(tokens)} tokens to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert tokenizer files to BinTokenizer format'
    )
    parser.add_argument('input', help='Input tokenizer file (.model or .json)')
    parser.add_argument('output', help='Output .bin file')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    if input_path.suffix == '.model':
        print(f"Converting SentencePiece model: {input_path}")
        convert_sentencepiece(str(input_path), args.output)
    elif input_path.suffix == '.json':
        print(f"Converting HuggingFace tokenizer: {input_path}")
        convert_huggingface_json(str(input_path), args.output)
    else:
        print(f"Error: Unknown file type: {input_path.suffix}")
        print("Supported formats: .model (SentencePiece), .json (HuggingFace)")
        sys.exit(1)
    
    # Verify the output
    output_size = Path(args.output).stat().st_size
    print(f"Output file size: {output_size:,} bytes")


if __name__ == '__main__':
    main()
