import triton
import triton.language as tl


@triton.jit
def _prepare_verify_inputs_kernel(
    descriptors,
    allocated,
    token_pool,
    page_table,
    positions,
    position_indices,
    request_indices,
    input_ids,
    out_loc,
    token_pool_stride,
    page_table_stride,
    descriptor_width: tl.constexpr,
    dummy_page,
    BLOCK_WIDTH: tl.constexpr,
    DESC_TABLE_IDX: tl.constexpr,
    DESC_CACHED_LEN: tl.constexpr,
    DESC_FORWARD_LEN: tl.constexpr,
    DESC_VERIFY_LEN: tl.constexpr,
    DESC_ROW_OFFSET: tl.constexpr,
    DESC_ALLOC_OFFSET: tl.constexpr,
    DESC_IS_REAL: tl.constexpr,
    DESC_DRAFT_LEN: tl.constexpr,
    DESC_FIXED_FIELDS: tl.constexpr,
):
    request_index = tl.program_id(0)
    descriptor = descriptors + request_index * descriptor_width
    table_idx = tl.load(descriptor + DESC_TABLE_IDX).to(tl.int64)
    cached_len = tl.load(descriptor + DESC_CACHED_LEN).to(tl.int64)
    forward_len = tl.load(descriptor + DESC_FORWARD_LEN).to(tl.int32)
    verify_len = tl.load(descriptor + DESC_VERIFY_LEN).to(tl.int32)
    row_offset = tl.load(descriptor + DESC_ROW_OFFSET).to(tl.int64)
    alloc_offset = tl.load(descriptor + DESC_ALLOC_OFFSET).to(tl.int64)
    is_real = tl.load(descriptor + DESC_IS_REAL).to(tl.int1)
    draft_len = tl.load(descriptor + DESC_DRAFT_LEN).to(tl.int32)

    offsets = tl.arange(0, BLOCK_WIDTH)
    mask = offsets < forward_len
    rows = row_offset + offsets
    sequence_positions = cached_len + offsets
    tl.store(positions + rows, sequence_positions, mask=mask)
    tl.store(position_indices + rows, sequence_positions, mask=mask)
    tl.store(request_indices + rows, table_idx, mask=mask)

    pending_ids = tl.load(
        token_pool + table_idx * token_pool_stride + cached_len + offsets * 0,
        mask=mask & (offsets == 0),
        other=0,
    )
    draft_ids = tl.load(
        descriptor + DESC_FIXED_FIELDS + offsets - 1,
        mask=mask & (offsets > 0) & (offsets <= draft_len),
        other=0,
    )
    tokens = tl.where(offsets == 0, pending_ids, draft_ids)
    tl.store(input_ids + rows, tokens, mask=mask)

    real_kv = mask & is_real & (offsets < verify_len)
    pages = tl.load(allocated + alloc_offset + offsets, mask=real_kv, other=dummy_page)
    tl.store(
        page_table + table_idx * page_table_stride + sequence_positions,
        pages,
        mask=mask,
    )
    tl.store(out_loc + rows, pages, mask=mask)


@triton.jit
def _reconcile_verify_tokens_kernel(
    descriptors,
    target_tokens,
    token_pool,
    results,
    token_pool_stride,
    descriptor_width: tl.constexpr,
    eos_token_id,
    MAX_DRAFT_TOKENS: tl.constexpr,
    DESC_TABLE_IDX: tl.constexpr,
    DESC_DEVICE_LEN: tl.constexpr,
    DESC_ROW_OFFSET: tl.constexpr,
    DESC_DRAFT_LEN: tl.constexpr,
    DESC_IGNORE_EOS: tl.constexpr,
    DESC_FIXED_FIELDS: tl.constexpr,
    RESULT_FIELDS: tl.constexpr,
    RESULT_EMITTED_LEN: tl.constexpr,
    RESULT_ACCEPTED_DRAFTS: tl.constexpr,
):
    request_index = tl.program_id(0)
    descriptor = descriptors + request_index * descriptor_width
    table_idx = tl.load(descriptor + DESC_TABLE_IDX).to(tl.int64)
    device_len = tl.load(descriptor + DESC_DEVICE_LEN).to(tl.int64)
    row_offset = tl.load(descriptor + DESC_ROW_OFFSET).to(tl.int64)
    draft_len = tl.load(descriptor + DESC_DRAFT_LEN).to(tl.int32)
    ignore_eos = tl.load(descriptor + DESC_IGNORE_EOS).to(tl.int1)

    accepted_drafts = 0
    accepting = True
    for draft_index in tl.static_range(0, MAX_DRAFT_TOKENS):
        valid = draft_index < draft_len
        draft_id = tl.load(
            descriptor + DESC_FIXED_FIELDS + draft_index,
            mask=valid,
            other=-1,
        )
        target_id = tl.load(target_tokens + row_offset + draft_index, mask=valid, other=-2)
        matches = valid & accepting & (draft_id == target_id)
        accepted_drafts += matches.to(tl.int32)
        accepting = accepting & ((~valid) | matches)

    emitted_len = accepted_drafts + 1
    eos_found = False
    for token_index in tl.static_range(0, MAX_DRAFT_TOKENS + 1):
        valid = token_index < emitted_len
        token_id = tl.load(
            target_tokens + row_offset + token_index,
            mask=valid,
            other=-1,
        )
        hit = (~ignore_eos) & valid & (~eos_found) & (token_id == eos_token_id)
        emitted_len = tl.where(hit, token_index + 1, emitted_len)
        eos_found = eos_found | hit

    accepted_drafts = tl.minimum(accepted_drafts, emitted_len)
    pending_token = tl.load(target_tokens + row_offset + emitted_len - 1)
    tl.store(
        token_pool + table_idx * token_pool_stride + device_len + emitted_len - 1,
        pending_token,
    )
    result = results + request_index * RESULT_FIELDS
    tl.store(result + RESULT_EMITTED_LEN, emitted_len)
    tl.store(result + RESULT_ACCEPTED_DRAFTS, accepted_drafts)


def prepare_verify_inputs_triton(
    descriptors,
    allocated,
    token_pool,
    page_table,
    positions,
    position_indices,
    request_indices,
    input_ids,
    out_loc,
    *,
    num_requests: int,
    descriptor_width: int,
    max_forward_width: int,
    dummy_page: int,
) -> None:
    from minisgl.kernel.speculative import (
        DESC_ALLOC_OFFSET,
        DESC_CACHED_LEN,
        DESC_DRAFT_LEN,
        DESC_FIXED_FIELDS,
        DESC_FORWARD_LEN,
        DESC_IS_REAL,
        DESC_ROW_OFFSET,
        DESC_TABLE_IDX,
        DESC_VERIFY_LEN,
    )

    block_width = triton.next_power_of_2(max_forward_width)
    _prepare_verify_inputs_kernel[(num_requests,)](
        descriptors,
        allocated,
        token_pool,
        page_table,
        positions,
        position_indices,
        request_indices,
        input_ids,
        out_loc,
        token_pool.stride(0),
        page_table.stride(0),
        descriptor_width=descriptor_width,
        dummy_page=dummy_page,
        BLOCK_WIDTH=block_width,
        DESC_TABLE_IDX=DESC_TABLE_IDX,
        DESC_CACHED_LEN=DESC_CACHED_LEN,
        DESC_FORWARD_LEN=DESC_FORWARD_LEN,
        DESC_VERIFY_LEN=DESC_VERIFY_LEN,
        DESC_ROW_OFFSET=DESC_ROW_OFFSET,
        DESC_ALLOC_OFFSET=DESC_ALLOC_OFFSET,
        DESC_IS_REAL=DESC_IS_REAL,
        DESC_DRAFT_LEN=DESC_DRAFT_LEN,
        DESC_FIXED_FIELDS=DESC_FIXED_FIELDS,
        num_warps=1,
    )


def reconcile_verify_tokens_triton(
    descriptors,
    target_tokens,
    token_pool,
    results,
    *,
    num_requests: int,
    descriptor_width: int,
    max_draft_tokens: int,
    eos_token_id: int,
) -> None:
    from minisgl.kernel.speculative import (
        DESC_DEVICE_LEN,
        DESC_DRAFT_LEN,
        DESC_FIXED_FIELDS,
        DESC_IGNORE_EOS,
        DESC_ROW_OFFSET,
        DESC_TABLE_IDX,
        RESULT_ACCEPTED_DRAFTS,
        RESULT_EMITTED_LEN,
        RESULT_FIELDS,
    )

    _reconcile_verify_tokens_kernel[(num_requests,)](
        descriptors,
        target_tokens,
        token_pool,
        results,
        token_pool.stride(0),
        descriptor_width=descriptor_width,
        eos_token_id=eos_token_id,
        MAX_DRAFT_TOKENS=max_draft_tokens,
        DESC_TABLE_IDX=DESC_TABLE_IDX,
        DESC_DEVICE_LEN=DESC_DEVICE_LEN,
        DESC_ROW_OFFSET=DESC_ROW_OFFSET,
        DESC_DRAFT_LEN=DESC_DRAFT_LEN,
        DESC_IGNORE_EOS=DESC_IGNORE_EOS,
        DESC_FIXED_FIELDS=DESC_FIXED_FIELDS,
        RESULT_FIELDS=RESULT_FIELDS,
        RESULT_EMITTED_LEN=RESULT_EMITTED_LEN,
        RESULT_ACCEPTED_DRAFTS=RESULT_ACCEPTED_DRAFTS,
        num_warps=1,
    )
