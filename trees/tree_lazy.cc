// --mode=lazy : (설명을 쓰세요)
#include "../common/locks.h"
#include "../common/tree_iface.h"
#include "lazy/st_lazy.h"

class LazyTree : public ConcurrentTree {
    lazy::SuffixTree t;
public:
    explicit LazyTree(int depth) : t(depth) {}
    // 구현이 진행되면 true 로 바꾼다: {par_update, par_remove, par_read, read_with_update}
    TreeCaps caps() const override { return {false, false, true, false}; }
    void prepare_seq(int /*id*/, int /*max_len*/) override {}   // 필요하면 st_lazy 에 reserve 함수를 만들어 호출
    void extend(int id, std::span<const int32_t> tok) override { t.extend(id, tok); }
    void remove(int id) override { t.remove(id); }
    Draft speculate(std::span<const int32_t> ctx, const SpecParams& p) override {
        return t.speculate(ctx, p.max_spec_tokens, p.max_spec_factor, p.max_spec_offset,
                           p.min_token_prob, p.use_tree_spec);
    }
    void end_step() override {}                                 // Lazy 정리 등
    std::string check() override { return t.check_integrity(); }
    uint64_t digest() override { return t.content_digest(); }   // --check 가 serial 과 비교
};
REGISTER_TREE("lazy", "(description)", LazyTree);
