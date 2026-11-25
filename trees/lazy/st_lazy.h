// Copyright 2025 Snowflake Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <atomic>
#include <cassert>
#include <deque>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "../../third_party/arctic/int32_map.h"
#include "../../third_party/arctic/suffix_tree.h"   // ::Draft (측정 틀과 같은 타입)
#include "../../common/locks.h"                      // [lazy] 삭제 대기 목록 보호용 Mutex

namespace lazy {
using ::Draft;

struct Group;

struct Node {
    // Number of suffixes from the root that end at or pass through this node.
    int64_t count = 0;

    // Token referenced by this node. Node can refer to a sequence of tokens,
    // this is just the ID of the first token.
    int token = 0;

    // Number of tokens in this node.
    int length = 0;

    // Reference sequence ID and starting index for the tokens in this node.
    // Implements the path compression optimization to achieve O(N) memory.
    int ref_seq = 0;
    int ref_idx = -1;

    // This map tracks all the suffixes that end at this node. Maps seq_id to
    // the end index of that suffix (may be truncated due to tree depth). Used
    // to find a new ref_seq and ref_idx if the reference sequence is deleted.
    Int32Map<int> endpoints;

    // Pointer to parent node.
    Node* parent = nullptr;

    // Children nodes, the key should always be the first token of the child.
    Int32Map<std::unique_ptr<Node>> children;

    // All the children of each node are kept in order of decreasing count in
    // a doubly linked list for efficient speculation. head_child points to the
    // first child (highest count) and tail_child points to the last child.
    Node* head_child = nullptr;
    Node* tail_child = nullptr;

    // Pointers to the next and previous siblings in the doubly linked list.
    Node* next_sibling = nullptr;
    Node* prev_sibling = nullptr;

    // To enable efficient reordering of the siblings list when counts change,
    // nodes with the same count are grouped together. Each node holds a shared
    // pointer to its group, and the groups also form a doubly linked list.
    std::shared_ptr<Group> group = nullptr;

    Node() = default;

    Node(int64_t count, int token, int length, int ref_seq, int ref_idx)
        : count(count),
          token(token),
          length(length),
          ref_seq(ref_seq),
          ref_idx(ref_idx) {}

    // Memory usage of this node.
    size_t memory_usage() const {
        size_t total = sizeof(*this);
        total += children.memory_usage();
        total += endpoints.memory_usage();
        return total;
    }
};

struct Group {
    // Pointer to the head node of this group. All nodes before the head node
    // have a strictly higher count, and all nodes after the head node have a
    // lower or equal count.
    Node* head = nullptr;

    // Pointers to the next and previous groups in the doubly linked list.
    Group* next = nullptr;
    Group* prev = nullptr;

    Group(Node* head) : head(head) {}
};


class SuffixTree {
public:

    SuffixTree(int max_depth);

    int num_seqs() const {
        return static_cast<int>(_seqs.size());
    }

    // Append a new element to the sequence with id seq_id.
    void append(int seq_id, int token);

    // Append multiple new elements to the sequence with id seq_id.
    void extend(int seq_id, std::span<const int32_t> tokens);

    // Remove the sequence with id seq_id.
    void remove(int seq_id);

    // Given a context, speculate the next tokens using the suffix tree.
    Draft speculate(std::span<const int32_t> context,
                    int max_spec_tokens,
                    float max_spec_factor,
                    float max_spec_offset,
                    float min_token_prob,
                    bool use_tree_spec);

    // Check the integrity of the suffix tree, return empty string if ok,
    // otherwise return an error message.
    std::string check_integrity();

    // Estimate memory usage of the suffix tree, for debugging only. It
    // walks the entire tree so can be slow.
    size_t estimate_memory() const;

    // [suffix-tree-sync 추가] 트리 "내용"의 요약값 (--check 용, 한 스레드에서만 호출).
    // 루트에서 각 노드까지의 토큰열과 count 만 보고, 형제 순서·ref_seq 같은 표현 차이는 무시한다.
    // → 갱신 순서가 달라도 같은 토큰들을 넣었으면 같은 값이 나와야 한다.
    uint64_t content_digest();

    // [lazy] [한 스레드] 병렬 구간 전에 seq_id 자리를 만들고 max_len 만큼 reserve 한다.
    // 병렬 구간에서는 _seqs/_active_nodes 맵을 읽기만 하게 되고(rehash 없음), 토큰 vector 도
    // 재할당되지 않아 다른 스레드가 ref_seq 로 그 토큰을 읽어도 안전하다.
    // 삭제 대기 중인 id 가 다시 쓰이면(Arctic 은 seq_id 가 2^31 에서 한 바퀴 돌면 재사용)
    // 여기서 먼저 물리적으로 지운다.
    void prepare_seq(int seq_id, int max_len);

    // [lazy] 삭제는 세 단계로 미룬다 (제거 = 표시 → 나중에 정리, Lazy 동기화).
    //  1) mark_removed(id)      : 큐에 넣기만 (O(1)). extend 와 동시에 불러도 된다.
    //  2) _apply_marks()        : 큐의 seq 의 모든 접미사 경로에서 count 를 줄인다 (형제 재정렬 포함,
    //                             원본 remove 앞부분과 같음). count 0 이 된 노드("좀비")는 떼지 않고
    //                             형제 리스트 맨 뒤에 남긴다. 추측은 count 0 노드를 없는 것으로 본다.
    //                             갱신이 돌지 않는 시점(갱신 구간 뒤 첫 speculate, end_step)에 한 번에.
    //  3) flush_removed()       : 좀비 떼기·노드 합치기·ref 갱신·_seqs 정리 (end_step, 한 스레드).
    // 지울 seq 의 토큰(_seqs)은 3) 까지 남아 있어 그 seq 를 ref_seq 로 가리키는 노드도 유효하다.
    // 없는 id, 이미 대기 중인 id 는 무시.
    // 전제: 대기 중인 seq_id 로 flush 전에 다시 extend 하지 않는다 (다시 쓰려면 먼저 prepare_seq).
    void mark_removed(int seq_id);

    // [lazy] 물리적 삭제 (위 3단계). 다른 호출과 겹치지 않는 한 스레드(end_step)에서만 부른다.
    // 아직 줄이지 않은 것(2단계)도 먼저 마친다. 지운 seq 수를 돌려준다.
    int flush_removed();

private:

    // [lazy] 삭제 대기: _queued(1단계, 아직 count 안 줄임) → _marked(2단계 끝, 물리적 삭제 대기).
    // _pending_set = 둘을 합친 집합 (중복 삭제·id 재사용 확인용). _pending_mu 로 보호.
    std::vector<int32_t> _queued;
    std::vector<int32_t> _marked;
    Int32Map<char> _pending_set;
    Mutex _pending_mu;
    std::atomic<bool> _marks_pending{false};   // _queued 가 비어 있지 않음 (speculate 가 먼저 확인)
    Mutex _apply_mu;                           // 2단계를 한 스레드만 하도록

    void _apply_marks();                       // 2단계, 1스레드 (end_step, prepare_seq)
    void _apply_marks_if_pending();            // speculate 앞에서: 추측 스레드들이 나눠서 2단계
    void _build_batch();                       // 큐 → 서브트리별 할 일 목록 (한 스레드)
    void _run_batch_group(int g, bool lock_root);   // 서브트리 g 의 접미사들 count 줄이기
    bool _batch_built = false;                 // _apply_mu 로 보호
    std::vector<std::pair<int32_t, int32_t>> _batch[64];   // 서브트리별 (seq_id, 시작 위치), serial 순서
    std::atomic<int> _batch_next{0};           // 다음에 가져갈 서브트리 번호
    std::atomic<int> _batch_done{0};           // 끝난 서브트리 수

    // [lazy] 락. 순서는 항상 루트 락 -> 서브트리 락 (교착 방지), 서브트리 락은 한 번에 하나만.
    //  - 서브트리 락: 첫 토큰(깊이 1 노드)을 해시한 번호의 락이 그 서브트리 안의 모든 노드를 보호
    //  - 루트 락: 루트의 count·endpoints·children 과 루트 자식(깊이 1 노드)들의 형제 리스트를 보호
    // 측정 틀이 병렬로 부르는 extend 만 잡는다 (갱신끼리의 경쟁). count 줄이기(2단계)와 flush 는
    // 갱신이 돌지 않을 때 한 스레드로 하고, speculate 는 갱신과 겹치지 않는다(read_with_update=false).
    static constexpr int kSubtreeLocks = 64;
    struct alignas(64) PaddedLock { Mutex m; };   // 락마다 캐시 줄 하나 (거짓 공유 방지)
    alignas(64) Mutex _root_lock;
    PaddedLock _subtree_locks[kSubtreeLocks];

    // [lazy] node 의 자식 중 count 0 인 것(형제 리스트 맨 뒤에 모여 있음)을 서브트리째 떼어 낸다.
    // 떼어 낸 노드 수를 돌려준다. node 를 보호하는 락(깊이 0 이면 루트 락, 아니면 서브트리 락)을 잡고 부른다.
    int _purge_zombies(Node* node);

    // [lazy] 물리적 삭제 본체 (seq 하나, 한 스레드)
    void _physical_remove(int seq_id);

    // Maximum depth of the suffix tree.
    int _max_depth;

    // The root node of the suffix tree.
    std::unique_ptr<Node> _root;

    // Mapping from seq id to its sequence of tokens (vectors of int32_t).
    Int32Map<std::vector<int32_t>> _seqs;

    // For each sequence, a sliding window of active nodes. Maintains at most
    // _max_depth active nodes for each sequence. Queue is shifted when a new
    // token is added to the sequence. Each active node is in the queue for at
    // most _max_depth iterations before being removed.
    Int32Map<std::deque<Node*>> _active_nodes;

    std::pair<Node*, int> _match_context(std::span<const int32_t> context);

    Draft _speculate_path(Node* node, int idx, int max_spec_tokens,
                          float min_token_prob);

    Draft _speculate_tree(Node* node, int idx, int max_spec_tokens,
                          float min_token_prob);

    std::string _check_node_integrity(Node* node);
};

}  // namespace lazy
