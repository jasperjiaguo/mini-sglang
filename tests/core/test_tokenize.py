from __future__ import annotations

from typing import Any

import torch
from minisgl.core import SamplingParams
from minisgl.message import TokenizeMsg
from minisgl.tokenizer.tokenize import TokenizeManager


class _FakeTokenizer:
    def __init__(self) -> None:
        self.template_kwargs: dict[str, Any] | None = None

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        self.template_kwargs = kwargs
        return "rendered prompt"

    def encode(self, prompt: str, return_tensors: str) -> torch.Tensor:
        assert prompt == "rendered prompt"
        assert return_tensors == "pt"
        return torch.tensor([[1, 2, 3]], dtype=torch.int64)


def test_tokenizer_forwards_chat_template_kwargs() -> None:
    tokenizer = _FakeTokenizer()
    manager = TokenizeManager(tokenizer)  # type: ignore[arg-type]
    message = TokenizeMsg(
        uid=0,
        text=[{"role": "user", "content": "hello"}],
        sampling_params=SamplingParams(),
        chat_template_kwargs={"enable_thinking": False},
    )

    result = manager.tokenize([message])

    assert result[0].tolist() == [1, 2, 3]
    assert tokenizer.template_kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }


def test_tokenize_message_serializes_chat_template_kwargs() -> None:
    message = TokenizeMsg(
        uid=7,
        text=[{"role": "user", "content": "hello"}],
        sampling_params=SamplingParams(max_tokens=8),
        chat_template_kwargs={"enable_thinking": False},
    )

    restored = message.decoder(message.encoder(message))

    assert isinstance(restored, TokenizeMsg)
    assert restored.chat_template_kwargs == {"enable_thinking": False}
