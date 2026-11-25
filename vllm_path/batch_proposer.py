"""vLLM 의 SuffixDecodingProposer.propose 를 묶음 호출(tree_batch)로 바꿔 끼우는 모듈.

vLLM 소스는 고치지 않는다. gpu_measure.py 가 --proposer batch|check 일 때 propose 대신 부른다.
  BatchProposer : 스텝마다 갱신 한 번, 추측 한 번만 C++ 를 부른다 (전역 트리는 --mode, 스레드는 --threads)
  CheckProposer : vLLM 원래 propose 결과를 그대로 돌려주면서, 같은 입력을 묶음 호출에도 넣어 추측을 비교한다

원래 propose(vllm/v1/spec_decode/suffix_decoding.py)와 다른 점
  - 순서: 원래는 요청마다 갱신 -> 추측, 묶음은 모든 요청 갱신 -> 모든 요청 추측.
    그래서 같은 스텝에 다른 요청이 붙인 토큰까지 보고 추측하며, 원래와 추측이 조금 다를 수 있다
  - 전역 트리에서 오래된 응답을 지우지 않는다. 캐시 상한에 닿지 않는 조건에서만 쓴다
  - 중단됐다가 다시 시작한 요청(preemption)의 옛 응답도 지우지 못해서, 그런 횟수를 restarted 로 센다
"""
import sys
from itertools import chain
from pathlib import Path

import numpy as np

DEFAULT_MODULE_DIR = Path(__file__).resolve().parent.parent / "bin" / "vllm_path"
LOCAL_SEQ = 0   # 개인 트리 안의 시퀀스 번호 (cache.py 와 같음)


def _pack_lists(lists):
    """토큰 목록 여러 개를 한 번에 한 줄 배열과 시작 위치 배열로 만든다."""
    offsets = np.zeros(len(lists) + 1, dtype=np.int32)
    np.cumsum(np.fromiter(map(len, lists), np.int32, len(lists)), out=offsets[1:])
    flat = np.fromiter(chain.from_iterable(lists), np.int32, int(offsets[-1]))
    return flat, offsets


def _gather_contexts(token_ids, rows, ends, depth):
    """rows 번째 줄마다 ends 앞의 최대 depth 토큰을 반복문 없이 한 줄 배열로 모은다."""
    lengths = np.minimum(ends, depth)
    offsets = np.zeros(len(rows) + 1, dtype=np.int32)
    np.cumsum(lengths, out=offsets[1:])
    # 한 줄 배열의 각 칸이 원래 표의 몇 번째 줄, 몇 번째 칸에서 오는지 계산한다
    columns = np.arange(offsets[-1]) + np.repeat(ends - lengths - offsets[:-1], lengths)
    flat = token_ids[np.repeat(rows, lengths), columns]
    return flat.astype(np.int32, copy=False), offsets


def _step_inputs(original, input_batch, sampled_token_ids):
    """원래 propose 와 같은 규칙으로 이번 스텝에 추측할 요청을 고른다.

    (batch 안의 위치, 요청 ID, 지금까지 토큰 수) 목록을 돌려준다.
    """
    picked = []
    for i, sampled_ids in enumerate(sampled_token_ids):
        if not sampled_ids:   # 프리필이 덜 끝난 요청
            continue
        num_tokens = int(input_batch.num_tokens_no_spec[i])
        if num_tokens >= original.max_model_len:
            continue
        picked.append((i, input_batch.req_ids[i], num_tokens))
    return picked


class BatchProposer:
    def __init__(self, original, mode, threads, module_dir=None):
        sys.path.insert(0, str(module_dir or DEFAULT_MODULE_DIR))
        import tree_batch

        self.tree_batch = tree_batch
        self.original = original   # vLLM 의 SuffixDecodingProposer (설정값을 가져온다)
        self.mode = mode
        self.threads = threads
        self.reset()

    def reset(self):
        """전역 트리와 요청 상태를 모두 비운다."""
        self.tree = self.tree_batch.BatchTree(self.mode, self.original.max_tree_depth, self.threads)
        self.active = {}      # 요청 ID -> (개인 트리 자리, 전역 트리 seq_id)
        self.free_slots = []
        self.seen = set()
        self.next_seq_id = 0
        self.restarted = 0

    def fill(self, records):
        """지난 응답들을 전역 트리에 미리 넣는다. records 는 (프롬프트, 응답) 목록."""
        first = self.next_seq_id
        for k, (_, response) in enumerate(records):
            self.tree.prepare_seq(first + k, len(response))
        for k, (_, response) in enumerate(records):
            self.tree.extend_global(first + k, response)
        self.next_seq_id = first + len(records)
        self.tree.end_step()

    @property
    def cached_requests(self):
        return self.next_seq_id

    def _start(self, req_id, prompt):
        if req_id in self.seen:
            self.restarted += 1
        self.seen.add(req_id)
        slot = self.free_slots.pop() if self.free_slots else self.tree.num_requests()
        seq_id = self.next_seq_id
        self.next_seq_id += 1
        self.tree.prepare_seq(seq_id, self.original.max_model_len)
        self.tree.start_request(slot, prompt)
        self.active[req_id] = (slot, seq_id)

    def propose(self, num_speculative_tokens, input_batch, sampled_token_ids, slot_mappings=None):
        original = self.original
        assert num_speculative_tokens == original.num_speculative_tokens
        max_len, active = original.max_model_len, self.active
        num_tokens = input_batch.num_tokens_no_spec[:len(sampled_token_ids)].tolist()

        # 요청마다 해야 하는 일(자리 찾기, 새 요청 시작)만 반복문에서 하고, 토큰은 뒤에서 한 번에 모은다
        rows, slots, seq_ids, new_tokens = [], [], [], []
        for i, (req_id, sampled_ids) in enumerate(zip(input_batch.req_ids, sampled_token_ids)):
            if not sampled_ids or num_tokens[i] >= max_len:   # 프리필 중이거나 최대 길이에 닿은 요청
                continue
            entry = active.get(req_id)
            if entry is None:
                index = input_batch.req_id_to_index[req_id]
                num_prompt = input_batch.num_prompt_tokens[index]
                self._start(req_id, np.ascontiguousarray(input_batch.token_ids_cpu[index, :num_prompt], np.int32))
                entry = active[req_id]
            rows.append(i)
            slots.append(entry[0])
            seq_ids.append(entry[1])
            new_tokens.append(sampled_ids)

        drafts = [[] for _ in sampled_token_ids]
        if rows:
            row_array = np.array(rows, dtype=np.int64)
            slot_array = np.array(slots, dtype=np.int32)
            token_flat, token_offsets = _pack_lists(new_tokens)
            self.tree.update(slot_array, np.array(seq_ids, dtype=np.int32), token_flat, token_offsets)

            ends = np.array([num_tokens[i] for i in rows], dtype=np.int64)
            limits = np.minimum(original.num_speculative_tokens, max_len - ends - 1)
            # 추측 길이 상한은 모델 최대 길이에 가까운 요청만 다르다. 상한이 같은 요청끼리 한 번에 부른다
            for limit in np.unique(limits).tolist():
                group = np.nonzero(limits == limit)[0]
                context_flat, context_offsets = _gather_contexts(
                    input_batch.token_ids_cpu, row_array[group], ends[group], original.max_tree_depth)
                tokens, _, _, offsets, _, _ = self.tree.speculate(
                    slot_array[group], context_flat, context_offsets, limit,
                    original.max_spec_factor, 0.0, original.min_token_prob)
                token_list, offset_list = tokens.tolist(), offsets.tolist()
                for n, row in enumerate(row_array[group].tolist()):
                    drafts[row] = token_list[offset_list[n]:offset_list[n + 1]]
            self.tree.end_step()

        # 이번 batch 에 없는 요청은 끝난 것으로 보고 자리를 돌려놓는다
        for req_id in active.keys() - input_batch.req_id_to_index.keys():
            self.free_slots.append(active.pop(req_id)[0])
        return drafts


class CheckProposer:
    """원래 propose 의 결과를 돌려주면서 묶음 호출의 추측이 맞는지 센다 (속도 측정에는 쓰지 않는다).

    비교 대상 두 가지
      phase : 진짜 SuffixDecodingCache 를 묶음과 같은 순서(모두 갱신 -> 모두 추측)로 부른 것. 같아야 정상
      vllm  : vLLM 원래 propose (요청마다 갱신 -> 추측). 순서가 달라 조금 다를 수 있다
    """

    def __init__(self, batch, original_propose):
        self.batch = batch
        self.original_propose = original_propose
        self.reset()

    def reset(self):
        from arctic_inference.suffix_decoding import SuffixDecodingCache

        self.batch.reset()
        self.reference = SuffixDecodingCache(max_tree_depth=self.batch.original.max_tree_depth,
                                             max_cached_requests=-1)
        self.compared = self.same_phase = self.same_vllm = 0

    def fill(self, records):
        self.batch.fill(records)
        for k, (prompt, response) in enumerate(records):
            self.reference.start_request(f"cached{k}", prompt)
            self.reference.add_active_response(f"cached{k}", response)
            self.reference.stop_request(f"cached{k}")

    def _reference_propose(self, input_batch, sampled_token_ids):
        original, cache = self.batch.original, self.reference
        picked = _step_inputs(original, input_batch, sampled_token_ids)
        for i, req_id, _ in picked:
            if req_id not in cache.active_requests:
                if req_id in cache.cached_requests:
                    cache.evict_cached_response(req_id)
                index = input_batch.req_id_to_index[req_id]
                cache.start_request(req_id, input_batch.token_ids_cpu[index, :input_batch.num_prompt_tokens[index]])
            cache.add_active_response(req_id, sampled_token_ids[i])
        drafts = [[] for _ in sampled_token_ids]
        for i, req_id, num_tokens in picked:
            start = max(0, num_tokens - original.max_tree_depth)
            drafts[i] = cache.speculate(
                req_id, input_batch.token_ids_cpu[i, start:num_tokens],
                max_spec_tokens=min(original.num_speculative_tokens, original.max_model_len - num_tokens - 1),
                max_spec_factor=original.max_spec_factor, min_token_prob=original.min_token_prob).token_ids
        for req_id in cache.active_requests - input_batch.req_id_to_index.keys():
            cache.stop_request(req_id)
        return drafts

    def propose(self, num_speculative_tokens, input_batch, sampled_token_ids, slot_mappings=None):
        from_vllm = self.original_propose(self.batch.original, num_speculative_tokens, input_batch,
                                          sampled_token_ids, slot_mappings)
        from_phase = self._reference_propose(input_batch, sampled_token_ids)
        from_batch = self.batch.propose(num_speculative_tokens, input_batch, sampled_token_ids)
        for i, sampled_ids in enumerate(sampled_token_ids):
            if sampled_ids:
                self.compared += 1
                self.same_phase += list(from_batch[i]) == list(from_phase[i])
                self.same_vllm += list(from_batch[i]) == list(from_vllm[i])
        return from_vllm
