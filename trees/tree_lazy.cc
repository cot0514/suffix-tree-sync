// --mode=lazy : 삭제는 표시만 하고(논리적 삭제), 실제 제거는 스텝 끝 end_step() 에서 한꺼번에 (Lazy)
// 2단계: remove() = 삭제 대기 목록에 넣기, end_step() = 목록의 seq 를 원본 remove 로 제거.
//        갱신(extend)은 아직 락이 없어 한 스레드 (par_update=false).
#include "../common/locks.h"
#include "../common/tree_iface.h"
#include "lazy/st_lazy.h"

class LazyTree : public ConcurrentTree {
    lazy::SuffixTree t;
public:
    explicit LazyTree(int depth) : t(depth) {}
    // {par_update, par_remove, par_read, read_with_update}
    // 갱신 병렬화(3단계) 전이라 par_update=false. 측정 틀은 par_update && par_remove 일 때만
    // 삭제를 갱신과 같은 병렬 구간에 넣으므로 par_remove 도 아직 false 로 둔다.
    TreeCaps caps() const override { return {false, false, true, false}; }
    void prepare_seq(int /*id*/, int /*max_len*/) override {}
    void extend(int id, std::span<const int32_t> tok) override { t.extend(id, tok); }
    void remove(int id) override { t.mark_removed(id); }   // 논리적 삭제 (트리는 그대로)
    Draft speculate(std::span<const int32_t> ctx, const SpecParams& p) override {
        return t.speculate(ctx, p.max_spec_tokens, p.max_spec_factor, p.max_spec_offset,
                           p.min_token_prob, p.use_tree_spec);
    }
    void end_step() override {                                // 물리적 삭제 (한 스레드)
        tctr().extra[0] += t.flush_removed();                 // extra0 = 실제로 지운 seq 수
    }
    std::string check() override { return t.check_integrity(); }
    uint64_t digest() override { return t.content_digest(); }   // --check 가 serial 과 비교
};
REGISTER_TREE("lazy", "remove = mark only, physical removal deferred to end_step (Lazy)", LazyTree);
