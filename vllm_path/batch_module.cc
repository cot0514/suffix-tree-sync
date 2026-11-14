// tree_batch: 요청 여러 개를 파이썬 호출 한 번에 처리하는 묶음 모듈 (README 7장 2·3단계)
// ---------------------------------------------------------------------------
//  - 전역 트리: trees/ 에 등록된 방식(--mode 와 같은 이름)을 ConcurrentTree 로 부른다
//  - 개인 트리: 요청마다 Arctic 원본 SuffixTree (vLLM 의 SuffixDecodingCache 와 같은 구성)
//  - 스텝마다 update 한 번, speculate 한 번만 부르면 모든 요청이 처리된다
//  - 입력은 토큰을 한 줄로 이은 배열과 시작 위치 배열(오프셋). 복사 없이 numpy 를 그대로 읽는다
//  - threads > 1 이면 (3단계) 스레드 풀이 구간별로 요청을 나눠 처리한다. 병렬 여부는 tree_bench 와
//    같게 전역 트리의 caps() 를 따른다. 일하는 동안에는 GIL 을 풀고, numpy 를 만들기 전에 다시 잡는다.
// 공통 틀(common/)과 trees/ 는 고치지 않고 가져다 쓰기만 한다.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "../common/tree_iface.h"

namespace nb = nanobind;

using Int32Array1D = nb::ndarray<int32_t, nb::numpy, nb::shape<-1>,
                                 nb::device::cpu, nb::any_contig>;

// 벡터를 복사 없이 numpy 배열로 넘긴다. 배열이 사라질 때 벡터도 같이 지워진다.
template <class T>
static nb::object to_numpy(std::vector<T>&& v) {
    v.reserve(1);   // 빈 배열이어도 데이터 위치가 널이 되지 않게 한다
    auto* owned = new std::vector<T>(std::move(v));
    nb::capsule owner(owned, [](void* p) noexcept {
        delete static_cast<std::vector<T>*>(p);
    });
    return nb::cast(nb::ndarray<nb::numpy, T, nb::ndim<1>>(
        owned->data(), {owned->size()}, owner));
}

// 한 줄로 이은 배열에서 k 번째 조각을 꺼낸다 (범위가 틀리면 예외)
static std::span<const int32_t> piece(const Int32Array1D& flat,
                                      const Int32Array1D& offsets, size_t k) {
    int64_t a = offsets.data()[k];
    int64_t b = offsets.data()[k + 1];
    if (a < 0 || b < a || static_cast<size_t>(b) > flat.size()) {
        throw std::invalid_argument("offsets out of range");
    }
    return std::span<const int32_t>(flat.data() + a, static_cast<size_t>(b - a));
}

// ---------------------------------------------------------------- 스레드 풀 (tree_bench 의 Pool 과 같은 구조)
// 스텝마다 스레드를 새로 만들지 않고 배리어로 깨운다. 작업 스레드의 g_tid 는 1.. (호출한 스레드는 0)
struct Pool {
    int n;
    std::vector<std::thread> th;
    std::barrier<> go, done;
    std::function<void(int)> job;
    bool stop = false;

    explicit Pool(int n_) : n(n_), go(n_ + 1), done(n_ + 1) {
        for (int t = 0; t < n; t++) {
            th.emplace_back([this, t] {
                g_tid = t + 1;
                for (;;) {
                    go.arrive_and_wait();
                    if (stop) return;
                    job(t);
                    done.arrive_and_wait();
                }
            });
        }
    }
    void run(std::function<void(int)> f) {
        job = std::move(f);
        go.arrive_and_wait();
        done.arrive_and_wait();
    }
    ~Pool() {
        stop = true;
        go.arrive_and_wait();
        for (auto& t : th) t.join();
    }
};

// 한 구간 안의 예외를 모아 둔다. 작업 스레드 밖으로 예외가 나가면 프로세스가 죽는다.
struct ErrorBox {
    std::mutex mu;
    std::exception_ptr first;
    void capture() {
        std::lock_guard<std::mutex> g(mu);
        if (!first) first = std::current_exception();
    }
    void rethrow() { if (first) std::rethrow_exception(first); }
};

// 0..n-1 을 공유 카운터로 chunk 개씩 나눠 가진다. pool 이 없으면 호출한 스레드가 모두 처리한다.
template <class F>
static void par_for(Pool* pool, size_t n, size_t chunk, ErrorBox& err, F&& f) {
    auto body = [&](size_t k) {
        try { f(k); } catch (...) { err.capture(); }
    };
    if (!pool) {
        for (size_t k = 0; k < n; k++) body(k);
        return;
    }
    std::atomic<size_t> next{0};
    pool->run([&](int) {
        for (size_t k; (k = next.fetch_add(chunk)) < n;)
            for (size_t j = k; j < std::min(k + chunk, n); j++) body(j);
    });
}

class BatchTree {
public:
    BatchTree(const std::string& mode, int depth, int threads = 1) : depth_(depth) {
        auto& reg = tree_registry();
        auto it = reg.find(mode);
        if (it == reg.end()) throw std::invalid_argument("unknown mode: " + mode);
        if (threads < 1 || threads > MAX_THREADS) {
            throw std::invalid_argument("threads must be 1.." + std::to_string(MAX_THREADS));
        }
        global_ = it->second.make(depth);
        caps_ = global_->caps();
        threads_ = threads;
        if (threads > 1) pool_ = std::make_unique<Pool>(threads);
    }

    // 전역 트리에 쓸 seq_id 자리를 미리 만든다 (병렬 갱신 방식에서 필요)
    void prepare_seq(int seq_id, int max_len) { global_->prepare_seq(seq_id, max_len); }

    // 사전 채우기용: 전역 트리에 응답 하나를 통째로 넣는다
    void extend_global(int seq_id, const Int32Array1D& tokens) {
        global_->extend(seq_id, std::span<const int32_t>(tokens.data(), tokens.size()));
    }

    // 요청들의 개인 트리를 만들고 프롬프트를 넣는다 (요청 번호 = 배열 순서)
    void start_requests(const Int32Array1D& prompt_flat, const Int32Array1D& prompt_off) {
        if (prompt_off.size() < 1) throw std::invalid_argument("empty offsets");
        size_t n = prompt_off.size() - 1;
        local_.clear();
        for (size_t k = 0; k < n; k++) {
            local_.push_back(std::make_unique<SuffixTree>(depth_));
            local_.back()->extend(0, piece(prompt_flat, prompt_off, k));
        }
    }

    // 스텝 갱신: 요청마다 개인 트리와 전역 트리에 이번에 확정된 토큰을 붙인다.
    // 구간 1) 개인 트리 (요청마다 독립이라 항상 병렬)  구간 2) 전역 트리 (caps().par_update 일 때만 병렬)
    void update(const Int32Array1D& req_idx, const Int32Array1D& seq_ids,
                const Int32Array1D& tok_flat, const Int32Array1D& tok_off) {
        size_t n = req_idx.size();
        check_sizes(n, seq_ids.size(), tok_off.size());
        // 입력 검사는 GIL 을 잡은 채로 끝낸다 (일하는 동안 파이썬 객체는 건드리지 않는다)
        std::vector<std::span<const int32_t>> toks(n);
        for (size_t k = 0; k < n; k++) {
            toks[k] = piece(tok_flat, tok_off, k);
            if (req_idx.data()[k] < 0 || static_cast<size_t>(req_idx.data()[k]) >= local_.size()) {
                throw std::out_of_range("request index out of range");
            }
        }
        const int32_t* ridx = req_idx.data();
        const int32_t* sids = seq_ids.data();
        ErrorBox err;
        {
            nb::gil_scoped_release nogil;
            double t0 = now_s();
            par_for(pool_.get(), n, 4, err, [&](size_t k) {
                local_[ridx[k]]->extend(0, toks[k]);
            });
            double t1 = now_s();
            par_for(caps_.par_update ? pool_.get() : nullptr, n, 1, err, [&](size_t k) {
                global_->extend(sids[k], toks[k]);
            });
            double t2 = now_s();
            t_local_ += t1 - t0;
            t_global_ += t2 - t1;
        }
        err.rethrow();
    }

    // 스텝 추측: 요청마다 개인·전역 추측 중 점수가 높은 쪽을 고른다 (cache.py 와 같은 규칙)
    // 반환: (토큰, 부모, 확률, 요청별 시작 위치, 점수, 일치 길이)
    nb::tuple speculate(const Int32Array1D& req_idx,
                        const Int32Array1D& ctx_flat, const Int32Array1D& ctx_off,
                        int max_spec_tokens, float max_spec_factor,
                        float max_spec_offset, float min_token_prob,
                        bool use_tree_spec) {
        size_t n = req_idx.size();
        check_sizes(n, n, ctx_off.size());
        SpecParams sp;
        sp.max_spec_tokens = max_spec_tokens;
        sp.max_spec_factor = max_spec_factor;
        sp.max_spec_offset = max_spec_offset;
        sp.min_token_prob = min_token_prob;
        sp.use_tree_spec = use_tree_spec;

        std::vector<std::span<const int32_t>> ctxs(n);
        for (size_t k = 0; k < n; k++) {
            ctxs[k] = piece(ctx_flat, ctx_off, k);
            if (req_idx.data()[k] < 0 || static_cast<size_t>(req_idx.data()[k]) >= local_.size()) {
                throw std::out_of_range("request index out of range");
            }
        }
        const int32_t* ridx = req_idx.data();

        // 요청마다 자기 칸에 결과를 쓰고, 끝난 뒤 순서대로 이어 붙인다 (스레드 수와 무관하게 같은 결과)
        std::vector<Draft> best(n);
        ErrorBox err;
        {
            nb::gil_scoped_release nogil;
            double t0 = now_s();
            par_for(caps_.par_read ? pool_.get() : nullptr, n, 4, err, [&](size_t k) {
                Draft d1 = local_[ridx[k]]->speculate(
                    ctxs[k], sp.max_spec_tokens, sp.max_spec_factor, sp.max_spec_offset,
                    sp.min_token_prob, sp.use_tree_spec);
                Draft d2 = global_->speculate(ctxs[k], sp);
                best[k] = d1.score >= d2.score ? std::move(d1) : std::move(d2);
            });
            t_spec_ += now_s() - t0;
        }
        err.rethrow();

        // GIL 을 다시 잡은 뒤 numpy 를 만든다
        std::vector<int32_t> tokens, parents, offsets(1, 0), match_lens;
        std::vector<float> probs, scores;
        for (const Draft& b : best) {
            tokens.insert(tokens.end(), b.token_ids.begin(), b.token_ids.end());
            parents.insert(parents.end(), b.parents.begin(), b.parents.end());
            probs.insert(probs.end(), b.probs.begin(), b.probs.end());
            offsets.push_back(static_cast<int32_t>(tokens.size()));
            scores.push_back(b.score);
            match_lens.push_back(b.match_len);
        }
        return nb::make_tuple(to_numpy(std::move(tokens)), to_numpy(std::move(parents)),
                              to_numpy(std::move(probs)), to_numpy(std::move(offsets)),
                              to_numpy(std::move(scores)), to_numpy(std::move(match_lens)));
    }

    // [한 스레드] 스텝이 끝날 때 부른다 (전역 트리의 end_step: 나중에 정리하는 방식 등에서 필요)
    void end_step() { global_->end_step(); }

    // 구조 검사와 트리 내용 요약 (정확성 확인용)
    std::string check() { return global_->check(); }
    uint64_t digest() { return global_->digest(); }

    int num_requests() const { return static_cast<int>(local_.size()); }
    int threads() const { return threads_; }

    // 측정용: update 의 개인 구간·전역 구간, speculate 구간 시간 합(초), 락 대기 시간 합(초)
    nb::tuple stats() const {
        double lw = 0;
        for (auto& c : g_ctr) lw += c.lock_wait_s;
        return nb::make_tuple(t_local_, t_global_, t_spec_, lw);
    }
    void reset_stats() {
        t_local_ = t_global_ = t_spec_ = 0;
        for (auto& c : g_ctr) c.lock_wait_s = 0;
    }

private:
    static void check_sizes(size_t n, size_t n_other, size_t off_size) {
        if (n != n_other || off_size != n + 1) {
            throw std::invalid_argument("array sizes do not match");
        }
    }

    int depth_;
    int threads_ = 1;
    TreeCaps caps_{};
    double t_local_ = 0, t_global_ = 0, t_spec_ = 0;
    std::unique_ptr<ConcurrentTree> global_;
    std::vector<std::unique_ptr<SuffixTree>> local_;
    std::unique_ptr<Pool> pool_;   // 맨 뒤에 둔다: 가장 먼저 정리되어 스레드가 트리보다 먼저 끝난다
};

NB_MODULE(tree_batch, m) {
    nb::set_leak_warnings(false);

    nb::class_<BatchTree>(m, "BatchTree")
        .def(nb::init<const std::string&, int, int>(),
             nb::arg("mode"), nb::arg("depth"), nb::arg("threads") = 1)
        .def("prepare_seq", &BatchTree::prepare_seq)
        .def("extend_global", &BatchTree::extend_global)
        .def("start_requests", &BatchTree::start_requests)
        .def("update", &BatchTree::update)
        .def("speculate", &BatchTree::speculate,
             nb::arg("req_idx"), nb::arg("ctx_flat"), nb::arg("ctx_off"),
             nb::arg("max_spec_tokens") = 16, nb::arg("max_spec_factor") = 1.0f,
             nb::arg("max_spec_offset") = 0.0f, nb::arg("min_token_prob") = 0.1f,
             nb::arg("use_tree_spec") = false)
        .def("end_step", &BatchTree::end_step)
        .def("check", &BatchTree::check)
        .def("digest", &BatchTree::digest)
        .def("stats", &BatchTree::stats)
        .def("reset_stats", &BatchTree::reset_stats)
        .def("threads", &BatchTree::threads)
        .def("num_requests", &BatchTree::num_requests);

    // 쓸 수 있는 전역 트리 방식 이름
    m.def("modes", [] {
        std::vector<std::string> names;
        for (auto& kv : tree_registry()) names.push_back(kv.first);
        return names;
    });
}
