// 공통 락 (subtree_lock 담당만 수정). 락 종류 비교용 TAS/TTAS/backoff 락은 node_lock 담당이 trees/node_lock/ 쪽 자기 파일에 만든다.
#pragma once

// Windows(MinGW)의 std::shared_mutex / std::mutex 는 winpthreads 를 거쳐 경합 시 매우 느림
// → Windows 에서는 SRWLOCK (MSVC std::shared_mutex 와 같은 기본 락)을 쓴다.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
struct RWLock {
    SRWLOCK l = SRWLOCK_INIT;
    void lock() { AcquireSRWLockExclusive(&l); }
    void unlock() { ReleaseSRWLockExclusive(&l); }
    bool try_lock() { return TryAcquireSRWLockExclusive(&l); }
    void lock_shared() { AcquireSRWLockShared(&l); }
    void unlock_shared() { ReleaseSRWLockShared(&l); }
};
using Mutex = RWLock;   // 배타 락만 쓰면 mutex 로 동작
#else
#include <atomic>
#include <shared_mutex>
#include <thread>
using RWLock = std::shared_mutex;

// Linux 의 std::mutex 는 경합 시 futex 로 잠들었다 깨어나서, 임계 구역이 짧은 서브트리 락에서는
// 그 비용이 락 보유 시간보다 훨씬 크다 (WSL2, 4스레드 전역 갱신 구간 3.3ms → 스핀락 1.1ms).
// → 짧게 도는 TTAS 스핀락을 쓴다. 코어보다 스레드가 많아 락을 쥔 스레드가 밀려나는 경우를 위해
//   일정 횟수 돈 뒤에는 CPU 를 양보한다.
struct SpinMutex {
    std::atomic<int> flag{0};
    bool try_lock() {
        return flag.load(std::memory_order_relaxed) == 0 &&
               flag.exchange(1, std::memory_order_acquire) == 0;
    }
    void lock() {
        for (int spins = 0;; ) {
            if (try_lock()) return;
            while (flag.load(std::memory_order_relaxed) != 0) {
                if (++spins < 2000) cpu_relax();
                else { std::this_thread::yield(); spins = 0; }
            }
        }
    }
    void unlock() { flag.store(0, std::memory_order_release); }
private:
    static void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#elif defined(__aarch64__)
        asm volatile("yield" ::: "memory");
#endif
    }
};
using Mutex = SpinMutex;
#endif
