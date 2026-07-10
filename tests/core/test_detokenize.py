from __future__ import annotations

from minisgl.message import DetokenizeMsg
from minisgl.tokenizer.detokenize import DetokenizeManager


class _FakeTokenizer:
    eos_token_id = 0

    def decode(self, token_ids: list[int]) -> str:
        text = {
            (): "",
            (1,): " old",
            (1, 2): " old-old",
            (2,): "-old",
        }
        return text[tuple(token_ids)]


def test_same_request_tokens_in_one_reply_are_detokenized_sequentially() -> None:
    manager = DetokenizeManager(_FakeTokenizer())  # type: ignore[arg-type]

    outputs = manager.detokenize(
        [
            DetokenizeMsg(uid=3, next_token=1, finished=False),
            DetokenizeMsg(uid=3, next_token=2, finished=True),
        ]
    )

    assert outputs == [" old", "-old"]
    assert 3 not in manager.decode_map
