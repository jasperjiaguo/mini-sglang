from __future__ import annotations

import torch

DESC_TABLE_IDX = 0
DESC_CACHED_LEN = 1
DESC_DEVICE_LEN = 2
DESC_FORWARD_LEN = 3
DESC_VERIFY_LEN = 4
DESC_ROW_OFFSET = 5
DESC_ALLOC_OFFSET = 6
DESC_IS_REAL = 7
DESC_DRAFT_LEN = 8
DESC_IGNORE_EOS = 9
DESC_FIXED_FIELDS = 10

RESULT_EMITTED_LEN = 0
RESULT_ACCEPTED_DRAFTS = 1
RESULT_FIELDS = 2


def prepare_verify_inputs(
    descriptors: torch.Tensor,
    allocated: torch.Tensor,
    token_pool: torch.Tensor,
    page_table: torch.Tensor,
    positions: torch.Tensor,
    position_indices: torch.Tensor,
    request_indices: torch.Tensor,
    input_ids: torch.Tensor,
    out_loc: torch.Tensor,
    *,
    num_requests: int,
    descriptor_width: int,
    max_forward_width: int,
    dummy_page: int,
) -> None:
    if descriptors.is_cuda:
        from .triton.speculative import prepare_verify_inputs_triton

        prepare_verify_inputs_triton(
            descriptors,
            allocated,
            token_pool,
            page_table,
            positions,
            position_indices,
            request_indices,
            input_ids,
            out_loc,
            num_requests=num_requests,
            descriptor_width=descriptor_width,
            max_forward_width=max_forward_width,
            dummy_page=dummy_page,
        )
        return

    for request_index in range(num_requests):
        descriptor = descriptors[request_index]
        table_idx = int(descriptor[DESC_TABLE_IDX])
        cached_len = int(descriptor[DESC_CACHED_LEN])
        forward_len = int(descriptor[DESC_FORWARD_LEN])
        verify_len = int(descriptor[DESC_VERIFY_LEN])
        row_offset = int(descriptor[DESC_ROW_OFFSET])
        alloc_offset = int(descriptor[DESC_ALLOC_OFFSET])
        is_real = bool(descriptor[DESC_IS_REAL])
        draft_len = int(descriptor[DESC_DRAFT_LEN])
        for local_offset in range(forward_len):
            row = row_offset + local_offset
            position = cached_len + local_offset
            positions[row] = position
            position_indices[row] = position
            request_indices[row] = table_idx
            if local_offset == 0:
                input_ids[row] = token_pool[table_idx, cached_len]
            elif local_offset <= draft_len:
                input_ids[row] = descriptor[DESC_FIXED_FIELDS + local_offset - 1]
            else:
                input_ids[row] = 0
            page = (
                int(allocated[alloc_offset + local_offset])
                if is_real and local_offset < verify_len
                else dummy_page
            )
            page_table[table_idx, position] = page
            out_loc[row] = page


def reconcile_verify_tokens(
    descriptors: torch.Tensor,
    target_tokens: torch.Tensor,
    token_pool: torch.Tensor,
    results: torch.Tensor,
    *,
    num_requests: int,
    descriptor_width: int,
    max_draft_tokens: int,
    eos_token_id: int,
) -> None:
    if descriptors.is_cuda:
        from .triton.speculative import reconcile_verify_tokens_triton

        reconcile_verify_tokens_triton(
            descriptors,
            target_tokens,
            token_pool,
            results,
            num_requests=num_requests,
            descriptor_width=descriptor_width,
            max_draft_tokens=max_draft_tokens,
            eos_token_id=eos_token_id,
        )
        return

    row_width = token_pool.size(1)
    del row_width  # the 2D CPU fallback indexes token_pool directly
    for request_index in range(num_requests):
        descriptor = descriptors[request_index]
        table_idx = int(descriptor[DESC_TABLE_IDX])
        device_len = int(descriptor[DESC_DEVICE_LEN])
        row_offset = int(descriptor[DESC_ROW_OFFSET])
        draft_len = int(descriptor[DESC_DRAFT_LEN])
        ignore_eos = bool(descriptor[DESC_IGNORE_EOS])
        accepted_drafts = 0
        for draft_index in range(draft_len):
            draft_id = int(descriptor[DESC_FIXED_FIELDS + draft_index])
            if draft_id != int(target_tokens[row_offset + draft_index]):
                break
            accepted_drafts += 1
        emitted_len = accepted_drafts + 1
        if not ignore_eos:
            for token_index in range(emitted_len):
                if int(target_tokens[row_offset + token_index]) == eos_token_id:
                    emitted_len = token_index + 1
                    break
        accepted_drafts = min(accepted_drafts, emitted_len)
        pending_token = target_tokens[row_offset + emitted_len - 1]
        token_pool[table_idx, device_len + emitted_len - 1] = pending_token
        results[request_index, RESULT_EMITTED_LEN] = emitted_len
        results[request_index, RESULT_ACCEPTED_DRAFTS] = accepted_drafts
