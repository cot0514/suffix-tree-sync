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

    // [lazy] 논리적 삭제: seq 의 모든 suffix 경로에서 count 만 줄이고(형제 재정렬 포함, 원본과 같음)
    // 대기 목록에 넣는다. count 가 0 이 된 노드("좀비")는 떼지 않고 형제 리스트 맨 뒤에 남긴다.
    //  - 추측(_match_context)은 count 0 노드를 없는 것으로 보므로 원본 remove 직후와 같은 결과
    //  - 갱신(append)은 활성 노드의 자식을 보기 전에 맨 뒤 좀비를 잘라 낸다 (_purge_zombies)
    //  - 지울 seq 의 토큰(_seqs)은 flush 때까지 남아 있어 그 seq 를 ref_seq 로 가리키는 노드도 유효
    // 전제: 대기 중인 seq_id 로 flush 전에 다시 extend 하지 않는다 (측정 틀은 지난 답변만 지움).
    void mark_removed(int seq_id);

    // [lazy] 물리적 삭제: 대기 목록 seq 의 suffix 경로를 다시 걸으며 좀비 떼기·노드 합치기·
    // ref 갱신을 하고 _seqs/_active_nodes 에서 지운다.
    // 다른 호출과 겹치지 않는 한 스레드(end_step)에서만 부른다. 지운 개수를 돌려준다.
    int flush_removed();

    size_t pending_removals() const { return _pending_remove.size(); }

private:

    // [lazy] 삭제 대기 목록 (mark_removed 가 여러 스레드에서 동시에 불려도 되도록 락으로 보호)
    std::vector<int32_t> _pending_remove;
    Mutex _pending_mu;

    // [lazy] node 의 자식 중 count 0 인 것(형제 리스트 맨 뒤에 모여 있음)을 서브트리째 떼어 낸다.
    // 떼어 낸 노드 수를 돌려준다.
    int _purge_zombies(Node* node);

    // [lazy] 물리적 삭제 본체 (seq 하나)
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
