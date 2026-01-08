"""
Tokenizer utilities for Moonshine.

Uses the same byte-level BPE tokenizer as Llama 1/2 with vocab size 32000 + 768 special tokens.
"""
from typing import List
import os


class MoonshineTokenizer:
    """
    Byte-level BPE tokenizer for Moonshine.

    This is a simplified version. In practice, you should use the actual
    Llama tokenizer or a compatible implementation.
    """

    def __init__(self, vocab_size: int = 32768):
        """
        Initialize tokenizer.

        Args:
            vocab_size: Vocabulary size (32000 + 768 special tokens)
        """
        self.vocab_size = vocab_size
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0

        # Try to use the tokenizers library
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import BPE

            # This is a placeholder - in practice, load the actual Llama tokenizer
            self._tokenizer = None
            self._use_hf_tokenizer = False
        except ImportError:
            self._tokenizer = None
            self._use_hf_tokenizer = False

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs (without BOS/EOS)
        """
        if self._tokenizer is not None:
            return self._tokenizer.encode(text).ids
        else:
            # Fallback: simple character-level encoding for demonstration
            # In practice, you MUST use a proper BPE tokenizer
            return [ord(c) % self.vocab_size for c in text if ord(c) < self.vocab_size]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        if skip_special_tokens:
            token_ids = [
                t
                for t in token_ids
                if t not in [self.bos_token_id, self.eos_token_id, self.pad_token_id]
            ]

        if self._tokenizer is not None:
            return self._tokenizer.decode(token_ids)
        else:
            # Fallback: simple character-level decoding
            return "".join([chr(t) for t in token_ids if t < 128])


def get_tokenizer(tokenizer_path: str = None) -> MoonshineTokenizer:
    """
    Get the Moonshine tokenizer.

    Args:
        tokenizer_path: Optional path to tokenizer files

    Returns:
        MoonshineTokenizer instance
    """
    # TODO: Load actual Llama tokenizer from HuggingFace or local path
    # For now, return a simple tokenizer
    return MoonshineTokenizer(vocab_size=32768)


def load_llama_tokenizer(tokenizer_path: str = None):
    """
    Load the actual Llama tokenizer.

    This is a helper function for users who want to use the proper Llama tokenizer.

    Args:
        tokenizer_path: Path to Llama tokenizer files or HuggingFace model ID

    Returns:
        Tokenizer instance

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> # Use with vocab_size 32000, add 768 special tokens for Moonshine
    """
    try:
        from transformers import AutoTokenizer

        if tokenizer_path is None:
            # Try to use a compatible tokenizer
            print(
                "WARNING: Using GPT2 tokenizer as fallback. "
                "For best results, use the Llama tokenizer."
            )
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        return tokenizer
    except ImportError:
        raise ImportError(
            "transformers library is required to load Llama tokenizer. "
            "Install with: pip install transformers"
        )


class TextNormalizer:
    """
    Text normalization utilities for preprocessing training data.

    Based on the paper's preprocessing steps.
    """

    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize text for training.

        This includes:
        - Lowercasing
        - Removing or replacing ambiguous unicode characters
        - Normalizing punctuation

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        import unicodedata

        # Lowercase
        text = text.lower()

        # Normalize unicode
        text = unicodedata.normalize("NFKC", text)

        # Remove control characters
        text = "".join(char for char in text if not unicodedata.category(char).startswith("C"))

        # Normalize whitespace
        text = " ".join(text.split())

        return text

    @staticmethod
    def prepare_caption(caption: str, pseudo_label: str, threshold: float = 0.3) -> str:
        """
        Prepare a caption by comparing with pseudo-label.

        This implements the filtering heuristic from the paper for noisily-labeled speech.

        Args:
            caption: Human-generated caption
            pseudo_label: Pseudo-label from Whisper
            threshold: Levenshtein distance threshold for filtering

        Returns:
            Prepared caption if distance < threshold, else empty string
        """
        # Normalize both
        caption_norm = TextNormalizer.normalize(caption)
        pseudo_norm = TextNormalizer.normalize(pseudo_label)

        # Compute normalized Levenshtein distance
        distance = TextNormalizer.levenshtein_distance(caption_norm, pseudo_norm)
        max_len = max(len(caption_norm), len(pseudo_norm))
        normalized_distance = distance / max(max_len, 1)

        # Filter if distance is too high
        if normalized_distance > threshold:
            return ""

        return caption

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """
        Compute Levenshtein distance between two strings.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Levenshtein distance
        """
        if len(s1) < len(s2):
            return TextNormalizer.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]
