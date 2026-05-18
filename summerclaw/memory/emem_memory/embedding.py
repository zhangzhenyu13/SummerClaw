"""EMem embedder — embedding generation for EMem memory algorithm.

Supports two backends:
1. **Provider embeddings** (default) — delegates to the LLM provider's ``embed()``
   method, which uses the same API credentials and endpoint as chat.
2. **Sentence-Transformers** (optional, via ``pip install summerclaw-ai[emem]``).

Provides the ``batch_encode(texts, instruction=None, norm=True)`` interface
expected by ContentStore and the retrieval pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    pass


class EMemEmbedder:
    """Embedding model wrapper for EMem memory algorithm.

    When a ``provider`` is supplied, embedding API calls are routed through
    ``provider.embed()``, which uses the same credentials and retry logic as
    the LLM chat calls.  For local-only usage, set ``use_local=True`` and
    install ``sentence-transformers``.

    Attributes:
        model_name: The embedding model name (e.g. ``"text-embedding-3-small"``).
        batch_size: Max batch size per API call.
        normalize: Whether to L2-normalize output vectors.
    """

    _DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
    _DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,       # fallback when no provider is given
        api_base: str | None = None,       # fallback when no provider is given
        batch_size: int = 32,
        normalize: bool = True,
        use_local: bool = False,
        provider: Any = None,              # LLMProvider instance (preferred)
    ):
        self.model_name = model_name or (
            self._DEFAULT_LOCAL_MODEL if use_local else self._DEFAULT_OPENAI_MODEL
        )
        self.batch_size = batch_size
        self.normalize = normalize
        self._use_local = use_local
        self._provider = provider
        # Keep these as fallback for backward-compat when provider is None
        self._api_key = api_key
        self._api_base = api_base
        self._local_model: Any = None
        self._openai_client: Any = None   # only used when _provider is None
        self._initialized = False

    @classmethod
    def from_config(
        cls,
        embedding_config: Any,
        *,
        provider: Any = None,
        fallback_api_key: str | None = None,
        fallback_api_base: str | None = None,
        batch_size: int | None = None,
        normalize: bool | None = None,
    ) -> "EMemEmbedder":
        """Create an EMemEmbedder from an EmbeddingConfig.

        Args:
            embedding_config: An ``EmbeddingConfig`` instance.
            provider: Optional LLMProvider for embedding API calls (preferred path).
            fallback_api_key: API key when embedding_config has none and no provider.
            fallback_api_base: API base URL when embedding_config has none and no provider.
            batch_size: Override batch size.
            normalize: Override normalize flag.
        """
        import os

        cfg_provider = getattr(embedding_config, "provider", "auto")
        use_local = cfg_provider == "local"

        model_name = getattr(embedding_config, "model", None) or None

        # If a provider is given, prefer it for API credentials
        if provider is not None and not use_local:
            api_key = getattr(embedding_config, "api_key", None) or None
            api_base = getattr(embedding_config, "api_base", None) or None
        else:
            api_key = getattr(embedding_config, "api_key", None) or None
            if not api_key:
                api_key = fallback_api_key or os.environ.get("OPENAI_API_KEY")
            api_base = getattr(embedding_config, "api_base", None) or None
            if not api_base:
                api_base = fallback_api_base or os.environ.get("OPENAI_BASE_URL")

        bs = batch_size if batch_size is not None else getattr(embedding_config, "batch_size", 16)
        norm = normalize if normalize is not None else getattr(embedding_config, "normalize", True)

        return cls(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            batch_size=bs,
            normalize=norm,
            use_local=use_local,
            provider=provider,
        )

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        if self._use_local:
            self._init_local()
        elif self._provider is not None:
            # Provider path — no separate client init needed
            logger.info(
                f"EMemEmbedder using provider embeddings: model={self.model_name}"
            )
        else:
            self._init_openai()

        self._initialized = True

    def _init_openai(self) -> None:
        """Legacy path: create a standalone OpenAI client when no provider is given."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAI embeddings. "
                "Install with: pip install openai"
            ) from exc

        import os

        api_key = self._api_key or os.environ.get("OPENAI_API_KEY", "")
        api_base = self._api_base or os.environ.get("OPENAI_BASE_URL", None)

        if not api_key:
            raise ValueError(
                "OpenAI API key is required for embedding generation. "
                "Set OPENAI_API_KEY env var or pass api_key to EMemEmbedder."
            )

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if api_base:
            client_kwargs["base_url"] = api_base

        self._openai_client = OpenAI(**client_kwargs)
        logger.info(f"EMemEmbedder using standalone OpenAI API: model={self.model_name}")

    def _init_local(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers package is required for local embeddings. "
                "Install with: pip install summerclaw-ai[emem]"
            ) from exc

        self._local_model = SentenceTransformer(self.model_name)
        logger.info(f"EMemEmbedder using local model: {self.model_name}")

    def batch_encode(
        self,
        texts: list[str],
        instruction: str | None = None,
        norm: bool | None = None,
    ) -> list[np.ndarray]:
        """Encode a batch of texts into embedding vectors.

        Args:
            texts: List of input strings.
            instruction: Optional instruction prefix (for models like E5/Instructor).
            norm: Override instance ``normalize`` setting.

        Returns:
            List of numpy arrays, one per input text.
        """
        self._ensure_initialized()

        if norm is None:
            norm = self.normalize

        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            if instruction and self._use_local:
                # Some local models support instruction prefixes
                batch = [f"{instruction}: {t}" for t in batch]

            if self._use_local:
                vectors = self._encode_local(batch)
            elif self._provider is not None:
                vectors = self._encode_via_provider(batch, instruction)
            else:
                vectors = self._encode_openai(batch, instruction)

            all_embeddings.extend(vectors)

        if norm:
            all_embeddings = [
                v / (np.linalg.norm(v) + 1e-12) for v in all_embeddings
            ]

        return all_embeddings

    def _encode_openai(
        self, texts: list[str], instruction: str | None,
    ) -> list[np.ndarray]:
        assert self._openai_client is not None

        # OpenAI doesn't support instruction natively; we prepend if needed
        if instruction:
            input_texts = [f"{instruction}: {t}" for t in texts]
        else:
            input_texts = texts

        response = self._openai_client.embeddings.create(
            model=self.model_name,
            input=input_texts,
        )
        return [
            np.array(item.embedding, dtype=np.float32)
            for item in response.data
        ]

    def _encode_via_provider(
        self, texts: list[str], instruction: str | None,
    ) -> list[np.ndarray]:
        """Encode texts through the LLM provider's embed() method."""
        assert self._provider is not None

        if instruction:
            input_texts = [f"{instruction}: {t}" for t in texts]
        else:
            input_texts = texts

        embeddings = self._provider.embed(input_texts, self.model_name)
        return [np.array(e, dtype=np.float32) for e in embeddings]

    def _encode_local(self, texts: list[str]) -> list[np.ndarray]:
        assert self._local_model is not None
        vectors = self._local_model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we normalize after
        )
        return [np.array(v, dtype=np.float32) for v in vectors]

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string (used for retrieval)."""
        results = self.batch_encode([query])
        return results[0]

    @property
    def dim(self) -> int:
        """Return embedding dimension (probe with a dummy text)."""
        self._ensure_initialized()
        probe = self.batch_encode(["dimension probe"])
        return len(probe[0])
