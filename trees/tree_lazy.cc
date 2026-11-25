// --mode=lazy : 삭제는 표시만 하고 실제 제거는 나중에 (Lazy 동기화)
//  1) remove()        = 큐에 넣기만 (O(1)). extend 와 동시에 불러도 된다.
//  2) 첫 speculate    = 갱신 구간이 끝난 뒤, 추측 스레드들이 서브트리별로 나눠 지운 seq 의 count 를 줄인다
//                       (count 0 노드 = 지워짐 표시, 추측은 건너뜀). 서브트리 안에서는 serial 과 같은 순서라
//                       동점 형제 순서까지 serial 과 같다.
//  3) end_step()      = 좀비 떼기·노드 합치기·ref 갱신·_seqs 정리 (한 스레드)
//  - extend() 끼리는 서브트리 락 64개 + 루트 락 (첫 토큰 서브트리마다 락, 루트 쪽을 고칠 때만 루트 락).
//  - 노드를 떼거나 합치는 구조 변경과 count 줄이기가 병렬 갱신 구간 밖에서만 일어나므로
//    갱신은 갱신끼리만 동기화하면 되고, 삭제 호출 자체는 갱신과 동시에 해도 된다 (par_remove=true).
// CSV extra0 = end_step 에서 정리한 seq 수, extra1 = 루트 락 잡은 횟수, extra2 = 떼어 낸 좀비 수,
//     extra3 = 2) 에서 count 를 줄인 노드 수
#include "../common/locks.h"
#include "../common/tree_iface.h"
#include "lazy/st_lazy.h"

class LazyTree : public ConcurrentTree {
    lazy::SuffixTree t;
public:
    explicit LazyTree(int depth) : t(depth) {}
    // {par_update, par_remove, par_read, read_with_update}
    // 추측은 락을 잡지 않으므로 갱신·삭제와 겹치면 안 된다 → read_with_update=false
    TreeCaps caps() const override { return {true, true, true, false}; }
    void prepare_seq(int id, int max_len) override { t.prepare_seq(id, max_len); }
    void extend(int id, std::span<const int32_t> tok) override { t.extend(id, tok); }
    void remove(int id) override { t.mark_removed(id); }
    Draft speculate(std::span<const int32_t> ctx, const SpecParams& p) override {
        return t.speculate(ctx, p.max_spec_tokens, p.max_spec_factor, p.max_spec_offset,
                           p.min_token_prob, p.use_tree_spec);
    }
    void end_step() override {
        tctr().extra[0] += t.flush_removed();
    }
    std::string check() override { return t.check_integrity(); }
    uint64_t digest() override { return t.content_digest(); }   // --check 가 serial 과 비교
};
REGISTER_TREE("lazy", "remove = queue; counts in batch before speculate (per subtree); physical removal in end_step", LazyTree);
