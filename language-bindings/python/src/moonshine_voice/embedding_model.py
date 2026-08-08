"""Embedding model access via the native embedding C API.

Internal to the binding: :class:`~moonshine_voice.agent_flow.AgentFlow` is
the supported way to match spoken phrases, and it owns an embedding model on
the application's behalf.  Nothing here is exported from ``moonshine_voice``.
"""

import ctypes
from pathlib import Path
from typing import List, Optional

from moonshine_voice.moonshine_api import _MoonshineLib
from moonshine_voice.errors import MoonshineError, check_error
from moonshine_voice.download import EmbeddingModelArch


class EmbeddingModel:
    """Loads an embedding model and scores text against it.

    Satisfies the :class:`~moonshine_voice.agent_flow.EmbeddingBackend`
    protocol, which is how ``AgentFlow`` matches utterances to phrases:

        >>> model = EmbeddingModel("path/to/embedding-model")
        >>> a = model.calculate_embedding("turn on the lights")
        >>> b = model.calculate_embedding("switch on the lights")
        >>> model.distance(a, b)
        0.92
    """

    def __init__(
        self,
        model_path: str | Path,
        model_arch: EmbeddingModelArch = EmbeddingModelArch.GEMMA_300M,
        model_variant: str = "fp32",
    ):
        """
        Initialize an embedding model.

        Args:
            model_path: Path to the directory containing the embedding model files
                       (ONNX model and tokenizer.bin).
            model_arch: The embedding model architecture to use.
                       Currently only GEMMA_300M is supported.
            model_variant: Model variant to load: "fp32", "fp16", "q8", "q4",
                          or "q4f16". Default is "q4" for efficiency.
        """
        self._lib_wrapper = _MoonshineLib()
        self._lib = self._lib_wrapper.lib
        self._handle = None
        self._setup_function_signatures()

        # Create the embedding model
        model_path_bytes = str(model_path).encode("utf-8")
        model_variant_bytes = model_variant.encode("utf-8") if model_variant else None

        handle = self._lib.moonshine_create_embedding_model(
            model_path_bytes,
            model_arch.value,
            model_variant_bytes,
        )

        if handle < 0:
            raise MoonshineError(f"Failed to create embedding model from {model_path}")

        self._handle = handle

    def _setup_function_signatures(self):
        """Setup ctypes function signatures for the embedding model C API."""
        lib = self._lib

        # Create embedding model
        lib.moonshine_create_embedding_model.restype = ctypes.c_int32
        lib.moonshine_create_embedding_model.argtypes = [
            ctypes.c_char_p,  # model_path
            ctypes.c_uint32,  # model_arch
            ctypes.c_char_p,  # model_variant
        ]

        # Free embedding model
        lib.moonshine_free_embedding_model.restype = None
        lib.moonshine_free_embedding_model.argtypes = [ctypes.c_int32]

        # Calculate embedding
        lib.moonshine_calculate_embedding.restype = ctypes.c_int32
        lib.moonshine_calculate_embedding.argtypes = [
            ctypes.c_int32,  # embedding_model_handle
            ctypes.c_char_p,  # sentence
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),  # out_embedding
            ctypes.POINTER(ctypes.c_uint64),  # out_embedding_size
            ctypes.c_char_p,  # model_name (nullable)
        ]

        # Free embedding
        lib.moonshine_free_embedding.restype = None
        lib.moonshine_free_embedding.argtypes = [
            ctypes.POINTER(ctypes.c_float),
        ]

        # Calculate embedding distance (cosine similarity)
        lib.moonshine_calculate_embedding_distance.restype = ctypes.c_int32
        lib.moonshine_calculate_embedding_distance.argtypes = [
            ctypes.c_int32,  # embedding_model_handle
            ctypes.POINTER(ctypes.c_float),  # embedding_a
            ctypes.POINTER(ctypes.c_float),  # embedding_b
            ctypes.c_uint64,  # embedding_size
            ctypes.POINTER(ctypes.c_float),  # out_similarity
        ]

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def close(self):
        """Free the embedding model resources."""
        if self._handle is not None:
            self._lib.moonshine_free_embedding_model(self._handle)
            self._handle = None

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "_handle"):
            self.close()

    def calculate_embedding(
        self, sentence: str, *, model_name: Optional[str] = None
    ) -> List[float]:
        """
        Calculate the embedding vector for a sentence.

        Args:
            sentence: The input text to embed.
            model_name: Reserved for future use; currently ignored by the native
                library. Pass *None*.

        Returns:
            A list of floats representing the embedding vector.
        """
        if self._handle is None:
            raise MoonshineError("Embedding model is not initialized")

        out_ptr = ctypes.POINTER(ctypes.c_float)()
        out_size = ctypes.c_uint64(0)
        model_bytes = model_name.encode("utf-8") if model_name else None
        error = self._lib.moonshine_calculate_embedding(
            self._handle,
            sentence.encode("utf-8"),
            ctypes.byref(out_ptr),
            ctypes.byref(out_size),
            model_bytes,
        )
        check_error(error)
        n = int(out_size.value)
        result = [float(out_ptr[i]) for i in range(n)]
        self._lib.moonshine_free_embedding(out_ptr)
        return result

    def distance(
        self, embedding_a: List[float], embedding_b: List[float]
    ) -> float:
        """
        Compute the cosine similarity between two embedding vectors.

        Args:
            embedding_a: The first embedding vector.
            embedding_b: The second embedding vector. Must be the same length
                as *embedding_a*.

        Returns:
            Cosine similarity in the range [-1, 1].  1 means identical,
            0 means orthogonal, -1 means opposite.
        """
        if self._handle is None:
            raise MoonshineError("Embedding model is not initialized")
        if len(embedding_a) != len(embedding_b):
            raise ValueError(
                f"Embedding sizes differ: {len(embedding_a)} vs {len(embedding_b)}"
            )

        n = len(embedding_a)
        arr_a = (ctypes.c_float * n)(*embedding_a)
        arr_b = (ctypes.c_float * n)(*embedding_b)
        out = ctypes.c_float(0.0)

        error = self._lib.moonshine_calculate_embedding_distance(
            self._handle, arr_a, arr_b, ctypes.c_uint64(n), ctypes.byref(out)
        )
        check_error(error)
        return float(out.value)
