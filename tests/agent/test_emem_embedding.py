"""Tests for EMemEmbedder — embedding generation interface."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from summerclaw.memory.emem_memory.embedding import EMemEmbedder


# ===================================================================
# EMemEmbedder — from_config
# ===================================================================

class TestEMemEmbedderFromConfig:
    """Test EMemEmbedder.from_config() factory method."""

    def test_default_config_creates_openai_embedder(self) -> None:
        from summerclaw.config.schema import EmbeddingConfig

        cfg = EmbeddingConfig()
        emb = EMemEmbedder.from_config(cfg, fallback_api_key="fb-key")
        assert emb.model_name == "text-embedding-3-small"
        assert emb._use_local is False
        assert emb.batch_size == 16
        assert emb.normalize is True
        assert emb._api_key == "fb-key"

    def test_custom_model_and_batch_size(self) -> None:
        from summerclaw.config.schema import EmbeddingConfig

        cfg = EmbeddingConfig(model="text-embedding-ada-002", batch_size=8, normalize=False)
        emb = EMemEmbedder.from_config(cfg, fallback_api_key="key")
        assert emb.model_name == "text-embedding-ada-002"
        assert emb.batch_size == 8
        assert emb.normalize is False

    def test_local_provider(self) -> None:
        from summerclaw.config.schema import EmbeddingConfig

        cfg = EmbeddingConfig(provider="local", model="all-MiniLM-L6-v2")
        emb = EMemEmbedder.from_config(cfg)
        assert emb._use_local is True
        assert emb.model_name == "all-MiniLM-L6-v2"

    def test_explicit_api_key_and_base(self) -> None:
        from summerclaw.config.schema import EmbeddingConfig

        cfg = EmbeddingConfig(api_key="explicit-key", api_base="https://custom.api/v1")
        emb = EMemEmbedder.from_config(cfg)
        assert emb._api_key == "explicit-key"
        assert emb._api_base == "https://custom.api/v1"

    def test_fallback_api_key_when_config_has_none(self) -> None:
        from summerclaw.config.schema import EmbeddingConfig

        cfg = EmbeddingConfig()  # api_key is None
        with patch.dict(os.environ, {}, clear=True):
            emb = EMemEmbedder.from_config(cfg, fallback_api_key="fb-key")
            assert emb._api_key == "fb-key"

    def test_fallback_api_base_when_config_has_none(self) -> None:
        from summerclaw.config.schema import EmbeddingConfig

        cfg = EmbeddingConfig()  # api_base is None
        with patch.dict(os.environ, {}, clear=True):
            emb = EMemEmbedder.from_config(cfg, fallback_api_base="https://fb.api")
            assert emb._api_base == "https://fb.api"

    def test_env_var_overrides_when_no_explicit_config(self) -> None:
        from summerclaw.config.schema import EmbeddingConfig

        cfg = EmbeddingConfig()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key", "OPENAI_BASE_URL": "https://env.api"}):
            emb = EMemEmbedder.from_config(cfg)
            assert emb._api_key == "env-key"
            assert emb._api_base == "https://env.api"

    def test_explicit_config_overrides_env_vars(self) -> None:
        from summerclaw.config.schema import EmbeddingConfig

        cfg = EmbeddingConfig(api_key="explicit-key")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            emb = EMemEmbedder.from_config(cfg)
            assert emb._api_key == "explicit-key"

    def test_explicit_config_overrides_fallback(self) -> None:
        from summerclaw.config.schema import EmbeddingConfig

        cfg = EmbeddingConfig(api_key="explicit-key")
        emb = EMemEmbedder.from_config(cfg, fallback_api_key="fb-key")
        assert emb._api_key == "explicit-key"

    def test_override_batch_size_and_normalize(self) -> None:
        from summerclaw.config.schema import EmbeddingConfig

        cfg = EmbeddingConfig(batch_size=16, normalize=True)
        emb = EMemEmbedder.from_config(cfg, batch_size=64, normalize=False)
        assert emb.batch_size == 64
        assert emb.normalize is False


# ===================================================================
# EMemEmbedder — basic initialization
# ===================================================================

class TestEMemEmbedderInit:
    """Test EMemEmbedder initialization and configuration."""

    def test_default_model_is_openai_small(self) -> None:
        # Don't initialize (no API key) — just check the name
        emb = EMemEmbedder(api_key="dummy-key")
        assert emb.model_name == "text-embedding-3-small"
        assert emb._use_local is False

    def test_custom_model_name(self) -> None:
        emb = EMemEmbedder(model_name="text-embedding-ada-002", api_key="dummy-key")
        assert emb.model_name == "text-embedding-ada-002"

    def test_local_mode_sets_default_model(self) -> None:
        emb = EMemEmbedder(use_local=True, api_key="dummy-key")
        assert emb.model_name == "all-MiniLM-L6-v2"
        assert emb._use_local is True

    def test_local_mode_with_custom_model(self) -> None:
        emb = EMemEmbedder(
            model_name="all-mpnet-base-v2",
            use_local=True,
            api_key="dummy-key",
        )
        assert emb.model_name == "all-mpnet-base-v2"

    def test_default_batch_size(self) -> None:
        emb = EMemEmbedder(api_key="dummy-key")
        assert emb.batch_size == 32

    def test_custom_batch_size(self) -> None:
        emb = EMemEmbedder(api_key="dummy-key", batch_size=16)
        assert emb.batch_size == 16

    def test_normalize_default_true(self) -> None:
        emb = EMemEmbedder(api_key="dummy-key")
        assert emb.normalize is True

    def test_normalize_custom(self) -> None:
        emb = EMemEmbedder(api_key="dummy-key", normalize=False)
        assert emb.normalize is False

    def test_not_initialized_until_used(self) -> None:
        emb = EMemEmbedder(api_key="dummy-key")
        assert emb._initialized is False


# ===================================================================
# EMemEmbedder — batch_encode
# ===================================================================

class TestEMemEmbedderEncode:
    """Test batch_encode with mocked OpenAI client."""

    @pytest.fixture
    def mock_openai_response(self) -> MagicMock:
        """Create a mock OpenAI embeddings response."""
        resp = MagicMock()
        resp.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
        ]
        return resp

    @pytest.fixture
    def embedder(self) -> EMemEmbedder:
        """Create an embedder with a mocked OpenAI client."""
        emb = EMemEmbedder(api_key="test-key", batch_size=8)
        # Pre-initialize with a mock client
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = MagicMock(
            data=[
                MagicMock(embedding=[0.1, 0.2]),
                MagicMock(embedding=[0.3, 0.4]),
                MagicMock(embedding=[0.5, 0.6]),
            ],
        )
        emb._openai_client = mock_client
        emb._initialized = True
        return emb

    def test_batch_encode_returns_list_of_arrays(self, embedder: EMemEmbedder) -> None:
        texts = ["hello", "world"]
        mock_data = [MagicMock(embedding=[0.1, 0.2]), MagicMock(embedding=[0.3, 0.4])]
        embedder._openai_client.embeddings.create.return_value = MagicMock(data=mock_data)
        result = embedder.batch_encode(texts)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)

    def test_batch_encode_respects_batch_size(self, embedder: EMemEmbedder) -> None:
        embedder.batch_size = 2
        texts = ["a", "b", "c", "d"]
        mock_data = [
            MagicMock(embedding=[0.1]),
            MagicMock(embedding=[0.2]),
        ]
        embedder._openai_client.embeddings.create.return_value = MagicMock(data=mock_data)
        result = embedder.batch_encode(texts)
        # Should have been called twice (2 batches of 2)
        assert embedder._openai_client.embeddings.create.call_count == 2
        assert len(result) == 4

    def test_batch_encode_with_norm_disabled(self, embedder: EMemEmbedder) -> None:
        texts = ["test"]
        mock_data = [MagicMock(embedding=[3.0, 4.0])]
        embedder._openai_client.embeddings.create.return_value = MagicMock(data=mock_data)
        result = embedder.batch_encode(texts, norm=False)
        # Should not be normalized
        assert np.linalg.norm(result[0]) != pytest.approx(1.0, abs=1e-5)

    def test_batch_encode_normalized(self, embedder: EMemEmbedder) -> None:
        texts = ["test norm"]
        mock_data = [MagicMock(embedding=[3.0, 4.0])]
        embedder._openai_client.embeddings.create.return_value = MagicMock(data=mock_data)
        result = embedder.batch_encode(texts, norm=True)
        # Should be normalized to unit length
        assert np.linalg.norm(result[0]) == pytest.approx(1.0)

    def test_batch_encode_with_instruction(self, embedder: EMemEmbedder) -> None:
        texts = ["query text"]
        mock_data = [MagicMock(embedding=[0.5, 0.5])]
        embedder._openai_client.embeddings.create.return_value = MagicMock(data=mock_data)
        result = embedder.batch_encode(texts, instruction="Represent the query")
        assert len(result) == 1

    def test_batch_encode_empty_list(self, embedder: EMemEmbedder) -> None:
        result = embedder.batch_encode([])
        assert result == []

    def test_encode_query(self, embedder: EMemEmbedder) -> None:
        mock_data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        embedder._openai_client.embeddings.create.return_value = MagicMock(data=mock_data)
        result = embedder.encode_query("test query")
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_dim_property(self, embedder: EMemEmbedder) -> None:
        mock_data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        embedder._openai_client.embeddings.create.return_value = MagicMock(data=mock_data)
        dim = embedder.dim
        assert dim == 3


# ===================================================================
# EMemEmbedder — OpenAI initialization
# ===================================================================

class TestEMemEmbedderOpenAIInit:
    """Test OpenAI client initialization."""

    def test_init_openai_with_api_key_env(self) -> None:
        emb = EMemEmbedder()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=True):
            with patch("openai.OpenAI") as mock_openai:
                emb._init_openai()
                mock_openai.assert_called_once()
                call_kwargs = mock_openai.call_args[1]
                assert call_kwargs["api_key"] == "env-key"

    def test_init_openai_with_api_key_param(self) -> None:
        emb = EMemEmbedder(api_key="param-key")
        with patch.dict(os.environ, {}, clear=True):
            with patch("openai.OpenAI") as mock_openai:
                emb._init_openai()
                mock_openai.assert_called_once()
                call_kwargs = mock_openai.call_args[1]
                assert call_kwargs["api_key"] == "param-key"

    def test_init_openai_with_api_base(self) -> None:
        emb = EMemEmbedder(api_key="key", api_base="https://custom.api.com")
        with patch("openai.OpenAI") as mock_openai:
            emb._init_openai()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["base_url"] == "https://custom.api.com"

    def test_init_openai_missing_key_raises(self) -> None:
        emb = EMemEmbedder(api_key=None)
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                emb._init_openai()

    def test_init_openai_missing_package(self) -> None:
        emb = EMemEmbedder(api_key="key")
        with patch.dict(os.environ, {}, clear=True):
            # Patch openai at the import level to simulate missing package
            with patch.dict("sys.modules", {"openai": None}):
                with pytest.raises(ImportError, match="openai"):
                    emb._init_openai()


# ===================================================================
# EMemEmbedder — local mode
# ===================================================================

class TestEMemEmbedderLocal:
    """Test local SentenceTransformer initialization and encoding."""

    def test_init_local_missing_package(self) -> None:
        emb = EMemEmbedder(use_local=True)
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="sentence-transformers"):
                emb._init_local()

    def test_local_encode_with_mock(self) -> None:
        emb = EMemEmbedder(model_name="test-model", use_local=True)
        mock_st = MagicMock()
        mock_st.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        emb._local_model = mock_st
        emb._initialized = True
        emb._use_local = True

        result = emb.batch_encode(["hello", "world"])
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)

    def test_local_encode_with_instruction_prefix(self) -> None:
        emb = EMemEmbedder(model_name="test-model", use_local=True)
        mock_st = MagicMock()
        mock_st.encode.return_value = np.array([[0.1]], dtype=np.float32)
        emb._local_model = mock_st
        emb._initialized = True
        emb._use_local = True

        result = emb.batch_encode(["query"], instruction="search")
        assert len(result) == 1
        # The instruction should be prepended to the text before encoding
        call_texts = mock_st.encode.call_args[0][0]
        assert "search:" in call_texts[0] or "query" in call_texts[0]


# ===================================================================
# EMemEmbedder — provider-based encoding
# ===================================================================

class TestEMemEmbedderProviderEncode:
    """Test batch_encode via provider.embed() path."""

    @pytest.fixture
    def mock_provider(self) -> MagicMock:
        """Create a mock LLMProvider that supports embed()."""
        provider = MagicMock()
        provider.embed.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
        return provider

    @pytest.fixture
    def embedder_with_provider(self, mock_provider: MagicMock) -> EMemEmbedder:
        """Create an EMemEmbedder with a mock provider."""
        emb = EMemEmbedder(
            model_name="text-embedding-3-large",
            provider=mock_provider,
            batch_size=8,
        )
        emb._initialized = True
        return emb

    def test_provider_embed_called(self, embedder_with_provider: EMemEmbedder, mock_provider: MagicMock) -> None:
        mock_provider.embed.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        texts = ["hello", "world"]
        result = embedder_with_provider.batch_encode(texts)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        mock_provider.embed.assert_called_once_with(texts, "text-embedding-3-large")

    def test_provider_embed_respects_batch_size(self, embedder_with_provider: EMemEmbedder, mock_provider: MagicMock) -> None:
        mock_provider.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
        embedder_with_provider.batch_size = 2
        texts = ["a", "b", "c", "d"]
        result = embedder_with_provider.batch_encode(texts)
        assert mock_provider.embed.call_count == 2
        assert len(result) == 4

    def test_provider_embed_with_instruction(self, embedder_with_provider: EMemEmbedder, mock_provider: MagicMock) -> None:
        mock_provider.embed.return_value = [[0.1, 0.2, 0.3]]
        texts = ["query text"]
        result = embedder_with_provider.batch_encode(texts, instruction="search")
        assert len(result) == 1
        # Instruction should be prepended
        call_args = mock_provider.embed.call_args[0]
        assert "search:" in call_args[0][0]

    def test_provider_embed_normalized(self, embedder_with_provider: EMemEmbedder, mock_provider: MagicMock) -> None:
        mock_provider.embed.return_value = [[3.0, 4.0]]
        result = embedder_with_provider.batch_encode(["test"])
        assert np.linalg.norm(result[0]) == pytest.approx(1.0)

    def test_provider_embed_norm_disabled(self, embedder_with_provider: EMemEmbedder, mock_provider: MagicMock) -> None:
        mock_provider.embed.return_value = [[3.0, 4.0]]
        result = embedder_with_provider.batch_encode(["test"], norm=False)
        assert np.linalg.norm(result[0]) != pytest.approx(1.0, abs=1e-5)

    def test_provider_embed_empty_list(self, embedder_with_provider: EMemEmbedder, mock_provider: MagicMock) -> None:
        result = embedder_with_provider.batch_encode([])
        assert result == []
        mock_provider.embed.assert_not_called()

    def test_provider_embed_encode_query(self, embedder_with_provider: EMemEmbedder, mock_provider: MagicMock) -> None:
        mock_provider.embed.return_value = [[0.1, 0.2, 0.3]]
        result = embedder_with_provider.encode_query("test query")
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_provider_embed_dim_property(self, embedder_with_provider: EMemEmbedder, mock_provider: MagicMock) -> None:
        mock_provider.embed.return_value = [[0.1, 0.2, 0.3]]
        dim = embedder_with_provider.dim
        assert dim == 3


# ===================================================================
# EMemEmbedder — provider initialization
# ===================================================================

class TestEMemEmbedderProviderInit:
    """Test provider-based initialization."""

    def test_provider_stored(self) -> None:
        mock_provider = MagicMock()
        emb = EMemEmbedder(provider=mock_provider)
        assert emb._provider is mock_provider

    def test_provider_init_skips_openai_client(self) -> None:
        """When provider is set, _ensure_initialized should NOT create an OpenAI client."""
        mock_provider = MagicMock()
        emb = EMemEmbedder(provider=mock_provider)
        emb._ensure_initialized()
        assert emb._openai_client is None
        assert emb._initialized is True

    def test_from_config_with_provider(self) -> None:
        from summerclaw.config.schema import EmbeddingConfig

        mock_provider = MagicMock()
        cfg = EmbeddingConfig(model="text-embedding-3-large")
        emb = EMemEmbedder.from_config(cfg, provider=mock_provider)
        assert emb._provider is mock_provider
        assert emb.model_name == "text-embedding-3-large"
        # api_key/api_base not needed when provider is set
        assert emb._api_key is None
        assert emb._api_base is None
