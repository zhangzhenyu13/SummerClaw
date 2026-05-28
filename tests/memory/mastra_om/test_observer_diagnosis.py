#!/usr/bin/env python3
"""Diagnose mastra_om Observer/Reflector in an isolated temp workspace."""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger
from summerclaw.config.loader import load_config, resolve_config_env_vars
from summerclaw.summerclaw import _make_provider
from summerclaw.memory.mastra_om_memory import MastraOMStore, MastraOMConsolidator

# Configure loguru to output everything to stdout with full tracebacks
logger.remove()
logger.add(sys.stdout, level="DEBUG", backtrace=True, diagnose=True)


async def test_observer():
    """Test Observer LLM call with mock data."""
    print("=" * 80)
    print("Starting mastra_om Observer diagnosis...")
    print("=" * 80)

    # Load config (will read ~/.summerclaw/config.json)
    print("\n[1] Loading config...")
    try:
        config = resolve_config_env_vars(load_config())
        print(f"✓ Config loaded successfully")
        print(f"  - Default model: {config.agents.defaults.model}")
        print(f"  - Memory algorithm: {config.agents.defaults.memory_algorithm}")
        print(f"  - Context window: {config.agents.defaults.context_window_tokens} tokens")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        logger.exception("Config load failed")
        return

    # Get provider
    print("\n[2] Getting LLM provider...")
    try:
        provider = _make_provider(config)
        print(f"✓ Provider initialized: {type(provider).__name__}")
        print(f"  - Default model: {provider.default_model if hasattr(provider, 'default_model') else 'N/A'}")
    except Exception as e:
        print(f"✗ Failed to get provider: {e}")
        logger.exception("Provider init failed")
        return

    # Create temp workspace
    print("\n[3] Creating temporary workspace...")
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        print(f"✓ Temp workspace: {workspace}")

        # Initialize store
        print("\n[4] Initializing MastraOMStore...")
        try:
            store = MastraOMStore(workspace, algo_name="mastra_om_memory")
            print(f"✓ Store initialized")
            print(f"  - Memory dir: {store.memory_dir}")
            print(f"  - Observations file: {store.observations_file}")
            print(f"  - History file: {store.history_file}")
        except Exception as e:
            print(f"✗ Failed to init store: {e}")
            logger.exception("Store init failed")
            return

        # Mock consolidator
        print("\n[5] Initializing MastraOMConsolidator...")
        try:
            # Mock functions for consolidator
            def mock_build_messages(**kwargs):
                return [{"role": "user", "content": "test"}]

            def mock_get_tool_definitions():
                return []

            consolidator = MastraOMConsolidator(
                store=store,
                provider=provider,
                model=config.agents.defaults.model,
                sessions=None,  # Not needed for this test
                context_window_tokens=config.agents.defaults.context_window_tokens or 100000,
                build_messages=mock_build_messages,
                get_tool_definitions=mock_get_tool_definitions,
            )
            print(f"✓ Consolidator initialized")
            print(f"  - Model: {consolidator.model}")
            print(f"  - Message tokens threshold: {consolidator.message_tokens_threshold}")
            print(f"  - Observation tokens threshold: {consolidator.observation_tokens_threshold}")
        except Exception as e:
            print(f"✗ Failed to init consolidator: {e}")
            logger.exception("Consolidator init failed")
            return

        # Prepare mock messages
        print("\n[6] Preparing mock conversation data...")
        mock_messages = [
            {
                "role": "user",
                "content": "你好，我正在开发一个AI助手项目",
                "timestamp": "2026-05-28 10:00",
            },
            {
                "role": "assistant",
                "content": "你好！很高兴帮助你开发AI助手项目。请告诉我更多细节。",
                "timestamp": "2026-05-28 10:01",
            },
            {
                "role": "user",
                "content": "我使用Python和FastAPI框架，主要功能是对话和任务管理",
                "timestamp": "2026-05-28 10:02",
            },
        ]
        print(f"✓ Mock data ready: {len(mock_messages)} messages")

        # Test Observer
        print("\n[7] Testing Observer LLM call...")
        print("-" * 80)
        try:
            result = await consolidator._observe_messages(mock_messages, existing_observations="")
            print("-" * 80)

            if result is None:
                print("✗ Observer returned None")
            else:
                print(f"✓ Observer returned result")
                print(f"  - Observations length: {len(result.get('observations', ''))} chars")
                print(f"  - Current task: {result.get('current_task', 'n/a')}")
                print(f"  - Suggested continuation: {result.get('suggested_continuation', 'n/a')}")
                print(f"  - Degenerate: {result.get('degenerate', False)}")

                if result.get('observations'):
                    print(f"\n  Observations preview:")
                    obs_lines = result['observations'].split('\n')[:10]
                    for line in obs_lines:
                        print(f"    {line}")

        except Exception as e:
            print(f"✗ Observer call failed with exception: {e}")
            logger.exception("Observer call failed")

        # Check if files were created
        print("\n[8] Checking generated files...")
        if store.observations_file.exists():
            print(f"✓ OBSERVATIONS.md created ({store.observations_file.stat().st_size} bytes)")
        else:
            print(f"✗ OBSERVATIONS.md not created")

        if store.history_file.exists():
            print(f"✓ history.jsonl created ({store.history_file.stat().st_size} bytes)")
        else:
            print(f"✗ history.jsonl not created")

        if store.om_ops_file.exists():
            print(f"✓ om-ops.jsonl created ({store.om_ops_file.stat().st_size} bytes)")
        else:
            print(f"✗ om-ops.jsonl not created")

        # Test full observe_and_store
        print("\n[9] Testing full observe_and_store() flow...")
        print("-" * 80)
        try:
            obs_text = await consolidator.observe_and_store(mock_messages)
            print("-" * 80)

            if obs_text is None:
                print("✗ observe_and_store returned None")
            else:
                print(f"✓ observe_and_store returned {len(obs_text)} chars")

        except Exception as e:
            print(f"✗ observe_and_store failed: {e}")
            logger.exception("observe_and_store failed")

        # Final file check
        print("\n[10] Final file status...")
        for f in [store.observations_file, store.history_file, store.om_ops_file]:
            if f.exists():
                print(f"✓ {f.name}: {f.stat().st_size} bytes")
                # Show first 200 chars
                content = f.read_text(encoding='utf-8')
                print(f"  Preview: {content[:200]}")
            else:
                print(f"✗ {f.name}: not created")

    print("\n" + "=" * 80)
    print("Diagnosis complete")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_observer())
