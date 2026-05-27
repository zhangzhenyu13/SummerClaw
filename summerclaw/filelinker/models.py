"""Data models for FileLinker."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class FileLinkToken:
    """Metadata for a single P2P download link."""

    token: str
    file_path: str
    original_name: str
    file_size: int
    content_type: str
    channel: str
    chat_id: str
    created_at: float
    expires_at: float
    download_count: int = 0
    max_downloads: int = 0  # 0 = unlimited

    # ── serialisation helpers (for .index.json persistence) ──

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> FileLinkToken:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @property
    def is_expired(self) -> bool:
        import time
        return time.time() > self.expires_at

    def is_download_exhausted(self) -> bool:
        if self.max_downloads == 0:
            return False
        return self.download_count >= self.max_downloads
