from __future__ import annotations

import torch
from minisgl.kernel.speculative import (
    DESC_ALLOC_OFFSET,
    DESC_CACHED_LEN,
    DESC_DEVICE_LEN,
    DESC_DRAFT_LEN,
    DESC_FIXED_FIELDS,
    DESC_FORWARD_LEN,
    DESC_IGNORE_EOS,
    DESC_IS_REAL,
    DESC_ROW_OFFSET,
    DESC_TABLE_IDX,
    DESC_VERIFY_LEN,
    RESULT_ACCEPTED_DRAFTS,
    RESULT_EMITTED_LEN,
    prepare_verify_inputs,
    reconcile_verify_tokens,
)


def _descriptors() -> torch.Tensor:
    descriptors = torch.zeros((2, DESC_FIXED_FIELDS + 2), dtype=torch.int64)
    descriptors[0, DESC_TABLE_IDX] = 0
    descriptors[0, DESC_CACHED_LEN] = 2
    descriptors[0, DESC_DEVICE_LEN] = 3
    descriptors[0, DESC_FORWARD_LEN] = 3
    descriptors[0, DESC_VERIFY_LEN] = 3
    descriptors[0, DESC_ROW_OFFSET] = 0
    descriptors[0, DESC_ALLOC_OFFSET] = 0
    descriptors[0, DESC_IS_REAL] = 1
    descriptors[0, DESC_DRAFT_LEN] = 2
    descriptors[0, DESC_IGNORE_EOS] = 1
    descriptors[0, DESC_FIXED_FIELDS : DESC_FIXED_FIELDS + 2] = torch.tensor([7, 8])

    descriptors[1, DESC_TABLE_IDX] = 1
    descriptors[1, DESC_CACHED_LEN] = 1
    descriptors[1, DESC_DEVICE_LEN] = 2
    descriptors[1, DESC_FORWARD_LEN] = 2
    descriptors[1, DESC_VERIFY_LEN] = 1
    descriptors[1, DESC_ROW_OFFSET] = 3
    descriptors[1, DESC_ALLOC_OFFSET] = 3
    descriptors[1, DESC_IS_REAL] = 1
    descriptors[1, DESC_DRAFT_LEN] = 0
    descriptors[1, DESC_IGNORE_EOS] = 1
    return descriptors


def test_prepare_verify_inputs_fuses_drafts_mappings_and_page_table() -> None:
    descriptors = _descriptors()
    allocated = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
    token_pool = torch.zeros((2, 8), dtype=torch.int32)
    token_pool[0, 2] = 5
    token_pool[1, 1] = 6
    page_table = torch.full((2, 8), -1, dtype=torch.int32)
    positions = torch.empty(5, dtype=torch.int32)
    position_indices = torch.empty(5, dtype=torch.int64)
    request_indices = torch.empty(5, dtype=torch.int64)
    input_ids = torch.empty(5, dtype=torch.int32)
    out_loc = torch.empty(5, dtype=torch.int32)

    prepare_verify_inputs(
        descriptors,
        allocated,
        token_pool,
        page_table,
        positions,
        position_indices,
        request_indices,
        input_ids,
        out_loc,
        num_requests=2,
        descriptor_width=descriptors.size(1),
        max_forward_width=3,
        dummy_page=99,
    )

    assert input_ids.tolist() == [5, 7, 8, 6, 0]
    assert positions.tolist() == [2, 3, 4, 1, 2]
    assert position_indices.tolist() == positions.tolist()
    assert request_indices.tolist() == [0, 0, 0, 1, 1]
    assert out_loc.tolist() == [10, 11, 12, 13, 99]
    assert page_table[0, 2:5].tolist() == [10, 11, 12]
    assert page_table[1, 1:3].tolist() == [13, 99]


def test_reconcile_verify_tokens_accepts_and_writes_pending_tokens() -> None:
    descriptors = _descriptors()
    target_tokens = torch.tensor([7, 9, 100, 42, 100], dtype=torch.int32)
    token_pool = torch.zeros((2, 8), dtype=torch.int32)
    results = torch.empty((2, 2), dtype=torch.int32)

    reconcile_verify_tokens(
        descriptors,
        target_tokens,
        token_pool,
        results,
        num_requests=2,
        descriptor_width=descriptors.size(1),
        max_draft_tokens=2,
        eos_token_id=99,
    )

    assert results[:, RESULT_EMITTED_LEN].tolist() == [2, 1]
    assert results[:, RESULT_ACCEPTED_DRAFTS].tolist() == [1, 0]
    assert token_pool[0, 4].item() == 9
    assert token_pool[1, 2].item() == 42


def test_reconcile_verify_tokens_truncates_at_eos() -> None:
    descriptors = _descriptors()[:1]
    descriptors[0, DESC_IGNORE_EOS] = 0
    target_tokens = torch.tensor([7, 8, 9], dtype=torch.int32)
    token_pool = torch.zeros((1, 8), dtype=torch.int32)
    results = torch.empty((1, 2), dtype=torch.int32)

    reconcile_verify_tokens(
        descriptors,
        target_tokens,
        token_pool,
        results,
        num_requests=1,
        descriptor_width=descriptors.size(1),
        max_draft_tokens=2,
        eos_token_id=7,
    )

    assert results[0, RESULT_EMITTED_LEN].item() == 1
    assert results[0, RESULT_ACCEPTED_DRAFTS].item() == 1
    assert token_pool[0, 3].item() == 7
