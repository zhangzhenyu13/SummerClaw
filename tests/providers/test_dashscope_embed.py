"""Tests for OpenAICompatProvider.embed() DashScope multimodal routing.

Verify that:
1. ``_is_dashscope_multimodal_model`` correctly classifies DashScope models
2. ``embed()`` routes DashScope multimodal models to the native API
3. ``embed()`` uses OpenAI-compatible endpoint for text-embedding-v* and non-dashscope
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nanobot.providers.openai_compat_provider import OpenAICompatProvider


# ===================================================================
# Model classification
# ===================================================================


class TestDashScopeModelClassification:
    """Tests for _is_dashscope_multimodal_model."""

    @pytest.mark.parametrize("model", [
        "tongyi-embedding-vision-plus",
        "tongyi-embedding-vision-flash",
        "tongyi-embedding-vision-plus-2026-03-06",
        "tongyi-embedding-vision-flash-2026-03-06",
        "qwen3-vl-embedding",
        "qwen2.5-vl-embedding",
        "multimodal-embedding-v1",
        "TONGYI-EMBEDDING-VISION-PLUS",
        "Qwen3-VL-Embedding",
    ])
    def test_multimodal_model_returns_true(self, model: str) -> None:
        assert OpenAICompatProvider._is_dashscope_multimodal_model(model) is True

    @pytest.mark.parametrize("model", [
        "text-embedding-v1",
        "text-embedding-v2",
        "text-embedding-v3",
        "text-embedding-v4",
        "text-embedding-3-small",
        "text-embedding-ada-002",
        "qwen-plus",
        "qwen-turbo",
        "qwen-max",
        "qwen3-235b-a22b",
        "gpt-4o",
        "claude-3-opus",
        "",
    ])
    def test_non_multimodal_model_returns_false(self, model: str) -> None:
        assert OpenAICompatProvider._is_dashscope_multimodal_model(model) is False


# ===================================================================
# embed() routing logic
# ===================================================================


class TestEmbedRouting:
    """Tests for embed() routing between OpenAI-compat and native API."""

    def test_empty_texts_returns_empty_list(self) -> None:
        """embed() returns [] immediately when texts is empty."""
        provider = OpenAICompatProvider(api_key="test-key", spec=None)
        result = provider.embed([], "text-embedding-v3")
        assert result == []

    def test_non_dashscope_spec_uses_openai_compat(self) -> None:
        """When spec is not dashscope, uses the standard /embeddings endpoint."""
        from nanobot.providers.registry import ProviderSpec

        openai_spec = ProviderSpec(
            name="openai",
            keywords=("openai", "gpt"),
            env_key="OPENAI_API_KEY",
            display_name="OpenAI",
            backend="openai_compat",
        )

        provider = OpenAICompatProvider(api_key="test-key", spec=openai_spec)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
        ]
        mock_client.embeddings.create.return_value = mock_response
        provider._sync_client = mock_client

        result = provider.embed(["hello", "world"], "tongyi-embedding-vision-flash")
        # Even though model is multimodal, with non-dashscope spec it goes through
        # the standard openai compat endpoint
        assert len(result) == 2
        mock_client.embeddings.create.assert_called_once_with(
            model="tongyi-embedding-vision-flash",
            input=["hello", "world"],
        )

    def test_dashscope_multimodal_routes_to_native_api(self) -> None:
        """When spec is dashscope and model is multimodal, uses native API."""
        from nanobot.providers.registry import ProviderSpec

        dashscope_spec = ProviderSpec(
            name="dashscope",
            keywords=("qwen", "dashscope"),
            env_key="DASHSCOPE_API_KEY",
            display_name="DashScope",
            backend="openai_compat",
            default_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        provider = OpenAICompatProvider(api_key="test-key", spec=dashscope_spec)

        # Mock the native API call
        with patch.object(
            provider,
            "_dashscope_multimodal_embed",
            return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        ) as mock_native:
            result = provider.embed(
                ["hello", "world"],
                "tongyi-embedding-vision-flash",
            )
            assert len(result) == 2
            mock_native.assert_called_once_with(
                ["hello", "world"],
                "tongyi-embedding-vision-flash",
            )

    def test_dashscope_text_embedding_uses_openai_compat(self) -> None:
        """DashScope text-embedding-v* models use the OpenAI-compat endpoint."""
        from nanobot.providers.registry import ProviderSpec

        dashscope_spec = ProviderSpec(
            name="dashscope",
            keywords=("qwen", "dashscope"),
            env_key="DASHSCOPE_API_KEY",
            display_name="DashScope",
            backend="openai_compat",
            default_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        provider = OpenAICompatProvider(api_key="test-key", spec=dashscope_spec)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2]),
        ]
        mock_client.embeddings.create.return_value = mock_response
        provider._sync_client = mock_client

        result = provider.embed(["test"], "text-embedding-v3")
        assert len(result) == 1
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-v3",
            input=["test"],
        )

    def test_dashscope_qwen_chat_model_uses_openai_compat(self) -> None:
        """DashScope with a non-embedding chat model (qwen-plus) uses OpenAI-compat."""
        from nanobot.providers.registry import ProviderSpec

        dashscope_spec = ProviderSpec(
            name="dashscope",
            keywords=("qwen", "dashscope"),
            env_key="DASHSCOPE_API_KEY",
            display_name="DashScope",
            backend="openai_compat",
            default_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        provider = OpenAICompatProvider(api_key="test-key", spec=dashscope_spec)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2]),
        ]
        mock_client.embeddings.create.return_value = mock_response
        provider._sync_client = mock_client

        result = provider.embed(["test"], "qwen-plus")
        assert len(result) == 1
        mock_client.embeddings.create.assert_called_once()


# ===================================================================
# _dashscope_multimodal_embed — native API call
# ===================================================================


class TestDashScopeNativeEmbed:
    """Tests for _dashscope_multimodal_embed."""

    def test_successful_batch_call(self) -> None:
        """Single batch of texts returns correct embeddings."""
        from nanobot.providers.registry import ProviderSpec

        dashscope_spec = ProviderSpec(
            name="dashscope",
            keywords=("qwen", "dashscope"),
            env_key="DASHSCOPE_API_KEY",
            display_name="DashScope",
            backend="openai_compat",
            default_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        provider = OpenAICompatProvider(api_key="test-key", spec=dashscope_spec)

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "output": {
                "embeddings": [
                    {"embedding": [0.1, 0.2, 0.3]},
                    {"embedding": [0.4, 0.5, 0.6]},
                ],
            },
        }

        with patch("requests.post", return_value=mock_resp):
            result = provider._dashscope_multimodal_embed(
                ["hello", "world"],
                "tongyi-embedding-vision-flash",
            )
            assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    def test_large_batch_splits_into_sub_batches(self) -> None:
        """Texts exceeding batch_size=10 are split."""
        from nanobot.providers.registry import ProviderSpec

        dashscope_spec = ProviderSpec(
            name="dashscope",
            keywords=("qwen", "dashscope"),
            env_key="DASHSCOPE_API_KEY",
            display_name="DashScope",
            backend="openai_compat",
            default_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        provider = OpenAICompatProvider(api_key="test-key", spec=dashscope_spec)

        texts = [f"text-{i}" for i in range(25)]  # 25 texts → 3 batches

        call_count = 0

        def make_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            payload = kwargs.get("json", {})
            contents = payload.get("input", {}).get("contents", [])
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {
                "output": {
                    "embeddings": [
                        {"embedding": [float(i)] * 3}
                        for i, _ in enumerate(contents)
                    ],
                },
            }
            return resp

        with patch("requests.post", side_effect=make_response):
            result = provider._dashscope_multimodal_embed(
                texts,
                "tongyi-embedding-vision-flash",
            )
            assert len(result) == 25
            assert call_count == 3  # ceil(25/10) = 3 batches

    def test_batch_failure_falls_back_to_single(self) -> None:
        """When a batch fails, individual texts are retried one-by-one."""
        from nanobot.providers.registry import ProviderSpec

        dashscope_spec = ProviderSpec(
            name="dashscope",
            keywords=("qwen", "dashscope"),
            env_key="DASHSCOPE_API_KEY",
            display_name="DashScope",
            backend="openai_compat",
            default_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        provider = OpenAICompatProvider(api_key="test-key", spec=dashscope_spec)

        texts = ["hello", "world"]
        call_log = []

        def make_response(*args, **kwargs):
            payload = kwargs.get("json", {})
            contents = payload.get("input", {}).get("contents", [])
            call_log.append(len(contents))

            resp = MagicMock()
            if len(contents) > 1:
                # Batch call fails
                resp.raise_for_status.side_effect = Exception("batch error")
            else:
                # Single-text call succeeds
                resp.raise_for_status = MagicMock()
                resp.json.return_value = {
                    "output": {
                        "embeddings": [
                            {"embedding": [0.1, 0.2]},
                        ],
                    },
                }
            return resp

        with patch("requests.post", side_effect=make_response):
            result = provider._dashscope_multimodal_embed(
                texts,
                "tongyi-embedding-vision-flash",
            )
            # First batch(2) fails, then 2 single-text calls succeed
            assert call_log == [2, 1, 1]
            assert len(result) == 2

    def test_correct_url_and_headers(self) -> None:
        """Verify the native API endpoint URL and auth headers."""
        from nanobot.providers.registry import ProviderSpec

        dashscope_spec = ProviderSpec(
            name="dashscope",
            keywords=("qwen", "dashscope"),
            env_key="DASHSCOPE_API_KEY",
            display_name="DashScope",
            backend="openai_compat",
            default_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        provider = OpenAICompatProvider(api_key="my-api-key", spec=dashscope_spec)

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "output": {
                "embeddings": [
                    {"embedding": [0.1]},
                ],
            },
        }

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_resp
            provider._dashscope_multimodal_embed(["test"], "tongyi-embedding-vision-flash")

            call_kwargs = mock_post.call_args
            # Check URL
            assert call_kwargs[0][0] == (
                "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
                "multimodal-embedding/multimodal-embedding"
            )
            # Check headers
            headers = call_kwargs[1]["headers"]
            assert headers["Authorization"] == "Bearer my-api-key"
            assert headers["Content-Type"] == "application/json"

    def test_payload_format(self) -> None:
        """Verify the JSON payload format matches DashScope native API."""
        from nanobot.providers.registry import ProviderSpec

        dashscope_spec = ProviderSpec(
            name="dashscope",
            keywords=("qwen", "dashscope"),
            env_key="DASHSCOPE_API_KEY",
            display_name="DashScope",
            backend="openai_compat",
            default_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        provider = OpenAICompatProvider(api_key="test-key", spec=dashscope_spec)

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "output": {
                "embeddings": [
                    {"embedding": [0.1, 0.2]},
                    {"embedding": [0.3, 0.4]},
                ],
            },
        }

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_resp
            provider._dashscope_multimodal_embed(
                ["hello", "world"],
                "tongyi-embedding-vision-flash",
            )

            json_payload = mock_post.call_args[1]["json"]
            assert json_payload["model"] == "tongyi-embedding-vision-flash"
            assert json_payload["input"]["contents"] == [
                {"text": "hello"},
                {"text": "world"},
            ]

    def test_empty_response_embeddings(self) -> None:
        """When API returns no embeddings, result is empty."""
        from nanobot.providers.registry import ProviderSpec

        dashscope_spec = ProviderSpec(
            name="dashscope",
            keywords=("qwen", "dashscope"),
            env_key="DASHSCOPE_API_KEY",
            display_name="DashScope",
            backend="openai_compat",
            default_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        provider = OpenAICompatProvider(api_key="test-key", spec=dashscope_spec)

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"output": {"embeddings": []}}

        with patch("requests.post", return_value=mock_resp):
            result = provider._dashscope_multimodal_embed(
                ["test"],
                "tongyi-embedding-vision-flash",
            )
            assert result == []


# ===================================================================
# Integration: embed() with complete mock chain
# ===================================================================


class TestEmbedIntegration:
    """End-to-end tests for embed() with both routing paths."""

    @pytest.mark.parametrize("model", [
        "tongyi-embedding-vision-plus",
        "tongyi-embedding-vision-flash",
        "qwen3-vl-embedding",
        "qwen2.5-vl-embedding",
        "multimodal-embedding-v1",
    ])
    def test_all_multimodal_models_route_to_native(self, model: str) -> None:
        """Every known DashScope multimodal model goes to native API."""
        from nanobot.providers.registry import ProviderSpec

        dashscope_spec = ProviderSpec(
            name="dashscope",
            keywords=("qwen", "dashscope"),
            env_key="DASHSCOPE_API_KEY",
            display_name="DashScope",
            backend="openai_compat",
            default_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        provider = OpenAICompatProvider(api_key="test-key", spec=dashscope_spec)

        with patch.object(
            provider,
            "_dashscope_multimodal_embed",
            return_value=[[0.1, 0.2]],
        ) as mock_native:
            result = provider.embed(["test"], model)
            assert len(result) == 1
            mock_native.assert_called_once()

    @pytest.mark.parametrize("model", [
        "text-embedding-v1",
        "text-embedding-v2",
        "text-embedding-v3",
        "text-embedding-v4",
    ])
    def test_dashscope_text_embed_models_use_openai_compat(self, model: str) -> None:
        """DashScope text-embedding models stay on OpenAI-compat."""
        from nanobot.providers.registry import ProviderSpec

        dashscope_spec = ProviderSpec(
            name="dashscope",
            keywords=("qwen", "dashscope"),
            env_key="DASHSCOPE_API_KEY",
            display_name="DashScope",
            backend="openai_compat",
            default_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        provider = OpenAICompatProvider(api_key="test-key", spec=dashscope_spec)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2])]
        mock_client.embeddings.create.return_value = mock_response
        provider._sync_client = mock_client

        result = provider.embed(["test"], model)
        assert len(result) == 1
        assert mock_client.embeddings.create.called
