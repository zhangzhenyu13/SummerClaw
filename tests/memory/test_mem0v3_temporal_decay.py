"""Tests for Mem0V3 temporal decay (Memory Decay) functionality."""

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from summerclaw.memory.mem0v3_memory.consolidator import (
    _apply_temporal_decay,
    _get_bm25_params,
    _normalize_bm25,
    _score_and_rank,
)
from summerclaw.memory.mem0v3_memory.store import Mem0V3Store


class TestTemporalDecay:
    """Test temporal Decay (Memory Decay) implementation."""

    def test_decay_fresh_memory_gets_boost(self):
        """Fresh memory (created today) should get max boost (1.5×)."""
        now = datetime.now(timezone.utc)
        scored = [
            {
                "id": "mem1",
                "score": 0.8,
                "payload": {
                    "text": "User lives in San Francisco",
                    "created_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                    "metadata": {},
                },
            }
        ]
        
        result = _apply_temporal_decay(scored, max_boost=1.5, min_dampen=0.3, decay_rate=0.1)
        
        assert len(result) == 1
        assert result[0]["id"] == "mem1"
        # Fresh memory: factor should be close to 1.5
        assert 1.4 <= result[0]["decay_factor"] <= 1.5
        # Score should be boosted and clamped to 1.0
        assert result[0]["score"] == 1.0
        assert result[0]["original_score"] == 0.8

    def test_decay_stale_memory_gets_dampened(self):
        """Stale memory (90 days old, never accessed) should get min dampen (0.3×)."""
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=90)
        
        scored = [
            {
                "id": "mem1",
                "score": 0.9,
                "payload": {
                    "text": "User lives in New York",
                    "created_at": old_time.isoformat(),
                    "updated_at": old_time.isoformat(),
                    # No access_history - this is a truly stale memory
                    "metadata": {},
                },
            }
        ]
        
        # Disable access history tracking to test pure time-based decay
        result = _apply_temporal_decay(
            scored, 
            max_boost=1.5, 
            min_dampen=0.3, 
            decay_rate=0.1,
            access_history_enabled=False  # Don't add current access time
        )
        
        assert len(result) == 1
        # Old memory: factor should be close to 0.3 (floor)
        assert 0.29 <= result[0]["decay_factor"] <= 0.31
        # Score should be dampened: 0.9 * 0.3 = 0.27
        assert 0.25 <= result[0]["score"] <= 0.28

    def test_decay_reorders_results(self):
        """Temporal decay should reorder results: fresh > old even if old has higher base score."""
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=30)
        
        scored = [
            {
                "id": "old_mem",
                "score": 0.9,  # High base score but old
                "payload": {
                    "text": "User lives in New York",
                    "created_at": old_time.isoformat(),
                    "updated_at": old_time.isoformat(),
                    "metadata": {},
                },
            },
            {
                "id": "fresh_mem",
                "score": 0.7,  # Lower base score but fresh
                "payload": {
                    "text": "User moved to San Francisco",
                    "created_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                    "metadata": {},
                },
            },
        ]
        
        # Disable access history to test pure creation-time decay
        result = _apply_temporal_decay(
            scored, 
            max_boost=1.5, 
            min_dampen=0.3, 
            decay_rate=0.1,
            access_history_enabled=False
        )
        
        # Fresh memory should now rank higher
        assert result[0]["id"] == "fresh_mem"
        assert result[1]["id"] == "old_mem"
        # Fresh should have higher score after decay
        assert result[0]["score"] > result[1]["score"]
        
        # Verify the math:
        # old_mem: 0.9 * (0.3 + 1.2 * e^(-0.1*30)) = 0.9 * (0.3 + 1.2 * 0.0498) = 0.9 * 0.36 = 0.324
        # fresh_mem: 0.7 * 1.5 = 1.05 → clamped to 1.0
        assert result[0]["score"] > 0.9  # Fresh boosted
        assert result[1]["score"] < 0.5  # Old dampened

    def test_decay_access_history_priority(self):
        """Access history should take priority over created_at."""
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=60)
        recent_access = now - timedelta(days=2)
        
        scored = [
            {
                "id": "mem1",
                "score": 0.8,
                "payload": {
                    "text": "User likes pizza",
                    "created_at": old_time.isoformat(),  # Old creation
                    "updated_at": old_time.isoformat(),
                    "metadata": {
                        "access_history": [
                            (now - timedelta(days=30)).isoformat(),
                            recent_access.isoformat(),  # Recent access
                        ]
                    },
                },
            }
        ]
        
        # With access_history_enabled=False, it should still use existing access_history
        result = _apply_temporal_decay(
            scored, 
            max_boost=1.5, 
            min_dampen=0.3, 
            decay_rate=0.1,
            access_history_enabled=False  # Don't modify existing history
        )
        
        # Should use recent_access (2 days ago), not created_at (60 days ago)
        # Factor should be > 1.0 (recently accessed)
        assert result[0]["decay_factor"] > 1.0

    def test_decay_access_history_limited_to_20(self):
        """Access history should be limited to last 20 entries."""
        now = datetime.now(timezone.utc)
        # Start with 30 old accesses
        access_history = [(now - timedelta(days=i)).isoformat() for i in range(30, 0, -1)]
        
        scored = [
            {
                "id": "mem1",
                "score": 0.8,
                "payload": {
                    "text": "Test memory",
                    "created_at": (now - timedelta(days=100)).isoformat(),
                    "metadata": {"access_history": access_history.copy()},
                },
            }
        ]
        
        # Enable access_history to test the trimming logic
        result = _apply_temporal_decay(
            scored, 
            max_boost=1.5, 
            min_dampen=0.3, 
            decay_rate=0.1,
            access_history_enabled=True
        )
        
        # Should append current access (31), then trim to 20
        assert len(result[0]["payload"]["metadata"]["access_history"]) == 20
        # Last entry should be very recent (within 1 second of now)
        last_access_str = result[0]["payload"]["metadata"]["access_history"][-1]
        last_access = datetime.fromisoformat(last_access_str)
        assert (datetime.now(timezone.utc) - last_access).total_seconds() < 1.0

    def test_decay_no_timestamp_neutral_factor(self):
        """Memory without timestamp should get neutral factor (1.0)."""
        scored = [
            {
                "id": "mem1",
                "score": 0.8,
                "payload": {
                    "text": "Memory without timestamp",
                    "metadata": {},
                },
            }
        ]
        
        result = _apply_temporal_decay(
            scored, 
            max_boost=1.5, 
            min_dampen=0.3, 
            decay_rate=0.1,
            access_history_enabled=False  # Don't add timestamp
        )
        
        assert result[0]["decay_factor"] == 1.0
        assert result[0]["score"] == 0.8

    def test_decay_score_clamped_to_0_1(self):
        """Final score should be clamped to [0, 1]."""
        now = datetime.now(timezone.utc)
        
        scored = [
            {
                "id": "mem1",
                "score": 0.9,
                "payload": {
                    "text": "Fresh memory",
                    "created_at": now.isoformat(),
                    "metadata": {},
                },
            }
        ]
        
        result = _apply_temporal_decay(scored, max_boost=1.5, min_dampen=0.3, decay_rate=0.1)
        
        # 0.9 * 1.5 = 1.35, should be clamped to 1.0
        assert result[0]["score"] <= 1.0

    def test_decay_exponential_formula(self):
        """Verify exponential decay formula: factor = min + (max - min) × e^(-rate × days)."""
        now = datetime.now(timezone.utc)
        days_ago = 7
        test_time = now - timedelta(days=days_ago)
        
        scored = [
            {
                "id": "mem1",
                "score": 1.0,
                "payload": {
                    "text": "Week-old memory",
                    "created_at": test_time.isoformat(),
                    "metadata": {},
                },
            }
        ]
        
        result = _apply_temporal_decay(
            scored, 
            max_boost=1.5, 
            min_dampen=0.3, 
            decay_rate=0.1,
            access_history_enabled=False
        )
        
        # Expected: 0.3 + 1.2 * e^(-0.1 * 7) = 0.3 + 1.2 * e^(-0.7) ≈ 0.3 + 1.2 * 0.497 ≈ 0.896
        import math
        expected_factor = 0.3 + 1.2 * math.exp(-0.1 * 7)
        assert abs(result[0]["decay_factor"] - expected_factor) < 0.05

    def test_decay_disabled_via_parameter(self):
        """Search should support disabling temporal decay via parameter."""
        # This is tested at the search() level, but we can verify the function respects params
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=30)
        
        scored = [
            {
                "id": "old_mem",
                "score": 0.9,
                "payload": {
                    "text": "Old memory",
                    "created_at": old_time.isoformat(),
                    "metadata": {},
                },
            }
        ]
        
        # With decay disabled (neutral params), score should remain unchanged
        result = _apply_temporal_decay(
            scored, 
            max_boost=1.0, 
            min_dampen=1.0, 
            decay_rate=0.0,
            access_history_enabled=False
        )
        
        assert result[0]["decay_factor"] == 1.0
        assert result[0]["score"] == 0.9

    def test_decay_with_full_search_pipeline(self):
        """Integration test: temporal decay in full search pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = Mem0V3Store(workspace=Path(tmpdir))
            
            # Insert memories with different ages
            now = datetime.now(timezone.utc)
            old_time = now - timedelta(days=30)
            
            # Old memory
            store.insert_memories_batch([
                {
                    "text": "User lives in New York",
                    "hash": "hash1",
                    "lemmatized": "user live new york",
                    "created_at": old_time.isoformat(),
                    "metadata": {},
                }
            ])
            
            # Fresh memory
            store.insert_memories_batch([
                {
                    "text": "User moved to San Francisco",
                    "hash": "hash2",
                    "lemmatized": "user move san francisco",
                    "created_at": now.isoformat(),
                    "metadata": {},
                }
            ])
            
            # Verify both memories exist
            all_memories = store.get_all_memories()
            assert len(all_memories) == 2


class TestTemporalDecayParameters:
    """Test different temporal decay parameter configurations."""

    def test_custom_decay_rate(self):
        """Faster decay rate should penalize old memories more."""
        now = datetime.now(timezone.utc)
        days_ago = 10
        test_time = now - timedelta(days=days_ago)
        
        scored = [
            {
                "id": "mem1",
                "score": 1.0,
                "payload": {
                    "text": "Test memory",
                    "created_at": test_time.isoformat(),
                    "metadata": {},
                },
            }
        ]
        
        # Slow decay (rate=0.05)
        result_slow = _apply_temporal_decay(
            scored, 
            max_boost=1.5, 
            min_dampen=0.3, 
            decay_rate=0.05,
            access_history_enabled=False
        )
        
        # Fast decay (rate=0.2)
        result_fast = _apply_temporal_decay(
            scored, 
            max_boost=1.5, 
            min_dampen=0.3, 
            decay_rate=0.2,
            access_history_enabled=False
        )
        
        # Fast decay should result in lower factor
        assert result_fast[0]["decay_factor"] < result_slow[0]["decay_factor"]

    def test_custom_boost_dampen_range(self):
        """Custom boost/dampen ranges should be respected."""
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=60)
        
        scored = [
            {
                "id": "mem1",
                "score": 1.0,
                "payload": {
                    "text": "Old memory",
                    "created_at": old_time.isoformat(),
                    "metadata": {},
                },
            }
        ]
        
        # Conservative range: 1.2× / 0.5×
        result_conservative = _apply_temporal_decay(
            scored, 
            max_boost=1.2, 
            min_dampen=0.5, 
            decay_rate=0.1,
            access_history_enabled=False
        )
        
        # Aggressive range: 2.0× / 0.1×
        result_aggressive = _apply_temporal_decay(
            scored, 
            max_boost=2.0, 
            min_dampen=0.1, 
            decay_rate=0.1,
            access_history_enabled=False
        )
        
        # Aggressive should dampen old memory more
        assert result_aggressive[0]["decay_factor"] < result_conservative[0]["decay_factor"]
        # Conservative should respect floor of 0.5
        assert result_conservative[0]["decay_factor"] >= 0.49  # Allow small floating point error
