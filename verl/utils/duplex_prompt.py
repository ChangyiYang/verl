# Copyright 2026 Liquid AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Lightweight prompt parsing helpers for duplex audio rollouts."""

from __future__ import annotations

from typing import Any


def extract_duplex_audio_and_system_prompt(messages: list[dict[str, Any]]) -> tuple[Any, str | None]:
    """Extract a unique audio item and at most one system message."""
    audio = None
    system_prompt = None
    system_seen = False
    for message in messages:
        content = message.get("content")
        if message.get("role") == "system":
            if system_seen:
                raise ValueError("Duplex prompt accepts at most one system message")
            system_seen = True
            if isinstance(content, str):
                system_prompt = content or None
            elif isinstance(content, list):
                text_parts = [item.get("text", "") for item in content if isinstance(item, dict)]
                system_prompt = "\n".join(part for part in text_parts if part) or None
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "audio":
                continue
            candidate = item.get("audio", item.get("audio_url"))
            if candidate is not None:
                if audio is not None:
                    raise ValueError("Duplex rollout currently accepts exactly one audio item per prompt")
                audio = candidate
    return audio, system_prompt


__all__ = ["extract_duplex_audio_and_system_prompt"]
