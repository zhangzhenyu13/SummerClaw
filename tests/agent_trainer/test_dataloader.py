"""Unit tests for agent_trainer data loader."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from summerclaw.agent_trainer.datasets.loader import DataLoader, DataSplit


class TestDataSplit:
    def test_len(self):
        split = DataSplit("train", [{"id": "1"}, {"id": "2"}, {"id": "3"}])
        assert len(split) == 3
        assert split.name == "train"

    def test_sample(self):
        items = [{"id": str(i)} for i in range(10)]
        split = DataSplit("train", items)
        sample = split.sample(5, seed=42)
        assert len(sample) == 5
        # All sampled items are from the original
        sampled_ids = {s["id"] for s in sample}
        assert sampled_ids.issubset({str(i) for i in range(10)})

    def test_sample_larger_than_split(self):
        items = [{"id": "1"}, {"id": "2"}]
        split = DataSplit("train", items)
        sample = split.sample(10)
        assert len(sample) == 2

    def test_sample_deterministic_with_seed(self):
        items = [{"id": str(i)} for i in range(20)]
        split = DataSplit("train", items)
        s1 = split.sample(5, seed=123)
        s2 = split.sample(5, seed=123)
        assert [x["id"] for x in s1] == [x["id"] for x in s2]

    def test_iter_batches(self):
        items = [{"id": str(i)} for i in range(10)]
        split = DataSplit("train", items)
        batches = list(split.iter_batches(3, seed=42))
        assert len(batches) == 4  # ceil(10/3) = 4
        assert len(batches[0]) == 3
        assert len(batches[-1]) == 1  # 10 % 3 = 1

    def test_iter_batches_exact(self):
        items = [{"id": str(i)} for i in range(6)]
        split = DataSplit("train", items)
        batches = list(split.iter_batches(3))
        assert len(batches) == 2
        assert all(len(b) == 3 for b in batches)


class TestDataLoader:
    def test_load_splits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create train split
            train_dir = os.path.join(tmpdir, "train")
            os.makedirs(train_dir)
            with open(os.path.join(train_dir, "items.json"), "w") as f:
                json.dump([{"id": "1", "question": "q1"}], f)

            # Create val split
            val_dir = os.path.join(tmpdir, "val")
            os.makedirs(val_dir)
            with open(os.path.join(val_dir, "items.json"), "w") as f:
                json.dump([{"id": "2", "question": "q2"}], f)

            loader = DataLoader(tmpdir)
            assert "train" in loader.split_names
            assert "val" in loader.split_names
            assert len(loader.train) == 1
            assert len(loader.val) == 1

    def test_missing_split_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(tmpdir)
            with pytest.raises(KeyError, match="Split 'train' not found"):
                _ = loader.train

    def test_invalid_json_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = os.path.join(tmpdir, "train")
            os.makedirs(train_dir)
            with open(os.path.join(train_dir, "items.json"), "w") as f:
                json.dump({"not": "a list"}, f)

            with pytest.raises(ValueError, match="expected a JSON array"):
                DataLoader(tmpdir)

    def test_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = os.path.join(tmpdir, "train")
            os.makedirs(train_dir)
            with open(os.path.join(train_dir, "items.json"), "w") as f:
                json.dump([{"id": "1"}, {"id": "2"}, {"id": "3"}], f)

            loader = DataLoader(tmpdir)
            summary = loader.summary()
            assert summary == {"train": 3}

    def test_get_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = os.path.join(tmpdir, "train")
            os.makedirs(train_dir)
            with open(os.path.join(train_dir, "items.json"), "w") as f:
                json.dump([{"id": "1"}], f)

            loader = DataLoader(tmpdir)
            split = loader.get_split("train")
            assert len(split) == 1

            with pytest.raises(KeyError):
                loader.get_split("test")
