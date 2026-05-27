"""FileLinker middleware — intercepts OutboundMessage to replace large files with P2P links."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from loguru import logger

from summerclaw.bus.events import OutboundMessage
from summerclaw.channels.base import BaseChannel

if TYPE_CHECKING:
    from summerclaw.filelinker.service import FileLinkerService


def format_size(size_bytes: int) -> str:
    """Human-readable file size string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


class FileLinkerMiddleware:
    """Intercepts ``OutboundMessage`` and replaces large file media with P2P download links."""

    _service: FileLinkerService | None = None

    @classmethod
    def set_service(cls, service: FileLinkerService | None) -> None:
        cls._service = service

    @classmethod
    async def intercept(cls, msg: OutboundMessage, channel: BaseChannel) -> OutboundMessage:
        """Process *msg*: replace oversized media with P2P link text.

        Returns a (possibly new) ``OutboundMessage``.  The original is returned
        unchanged when FileLinker is disabled or no media qualifies.
        """
        if cls._service is None or not cls._service.config.enabled:
            return msg

        if not msg.media:
            return msg

        remaining_media: list[str] = []
        link_texts: list[str] = []

        for media_path in msg.media:
            if cls._service.should_use_link(channel.name, media_path):
                try:
                    file_size = cls._service.get_file_size(media_path)
                    link_url = await cls._service.create_link(
                        file_path=media_path,
                        original_name=os.path.basename(media_path),
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                    )
                    link_texts.append(
                        f"📎 [{os.path.basename(media_path)}]({link_url}) "
                        f"({format_size(file_size)})"
                    )
                except Exception as exc:
                    logger.warning(
                        "FileLinker link creation failed for {}: {}", media_path, exc
                    )
                    remaining_media.append(media_path)
            else:
                remaining_media.append(media_path)

        if not link_texts:
            return msg

        new_content = msg.content
        link_block = "\n".join(link_texts)

        # Append Tailscale client setup instructions for the receiver
        tailscale_hint = (
            "\n\n---\n"
            "📡 **P2P 直传说明**：以上链接通过 Tailscale 内网传输，需要先加入同一网络才能下载。\n"
            "1️⃣ 下载 Tailscale 客户端：https://login.tailscale.com/download\n"
            "2️⃣ 使用你的账号登录并连接到同一个 Tailnet 网络\n"
            "3️⃣ 连接成功后，点击上方链接即可高速下载文件"
        )
        new_content = (
            f"{msg.content}\n\n{link_block}{tailscale_hint}"
            if msg.content
            else f"{link_block}{tailscale_hint}"
        )

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=new_content,
            reply_to=msg.reply_to,
            media=remaining_media,
            metadata=msg.metadata,
        )
