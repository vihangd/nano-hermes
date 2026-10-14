"""Background learning calls can use a different model from the turn.

A Pi user can run a small, fast model on the foreground and let a stronger
one write the lessons, insights and curated principles the small model then
reads (context-space distillation, DuoMem arXiv:2606.29961).
"""
from __future__ import annotations

from types import SimpleNamespace

import nano_hermes
from conftest import _make_loop
from nano_hermes.config import background_model
from nano_hermes.memory.cheatsheet import extract_cheatsheet_lesson


class _RecordingProvider:
    def __init__(self) -> None:
        self.models: list[str | None] = []

    async def chat_with_retry(self, *args, model=None, **kwargs):
        self.models.append(model)
        return SimpleNamespace(content="SKIP")


def test_falls_back_to_the_loop_model():
    hook = SimpleNamespace(config=SimpleNamespace(background_model=None),
                           _loop=SimpleNamespace(model="small"))
    assert background_model(hook) == "small"


async def test_extraction_uses_the_background_model(tmp_path):
    loop = _make_loop(tmp_path)
    hook = nano_hermes.install(loop, config={"background_model": "teacher-72b"})
    provider = _RecordingProvider()
    loop.provider = provider
    messages = [
        {"role": "user", "content": "migrate the database"},
        {"role": "assistant", "content": "done"},
    ]
    await extract_cheatsheet_lesson(hook, messages, "success")
    assert provider.models == ["teacher-72b"]
