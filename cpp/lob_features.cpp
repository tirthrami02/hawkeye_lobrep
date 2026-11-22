// Streaming order-book-imbalance feature kernel (O(levels) per tick, no allocations).
// Mirrors lobrep.features for the (spread, OBI@1/3/5, microprice offset) columns, to show how the
// feature stage of the pipeline would be moved to a low-latency path.
//   g++ -O3 -std=c++17 cpp/lob_features.cpp -o lob_features && ./lob_features
#include <array>
#include <chrono>
#include <cstdio>
#include <random>

constexpr int L = 10;
struct Book { std::array<double, L> ap, av, bp, bv; };
struct Feat { double spread, obi1, obi3, obi5, micro; };

static inline double obi(const Book& b, int d) {
    double bs = 0, as = 0;
    for (int i = 0; i < d; ++i) { bs += b.bv[i]; as += b.av[i]; }
    return (bs - as) / (bs + as + 1e-9);
}

Feat compute(const Book& b) {
    double mid = 0.5 * (b.ap[0] + b.bp[0]);
    double micro = (b.ap[0] * b.bv[0] + b.bp[0] * b.av[0]) / (b.bv[0] + b.av[0] + 1e-9);
    return {(b.ap[0] - b.bp[0]) / mid, obi(b, 1), obi(b, 3), obi(b, 5), (micro - mid) / mid};
}

int main() {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> u(50.0, 500.0);
    Book b;
    for (int i = 0; i < L; ++i) { b.ap[i] = 100.01 + 0.01 * i; b.bp[i] = 99.99 - 0.01 * i; }
    const long N = 5'000'000; double sink = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (long n = 0; n < N; ++n) {
        for (int i = 0; i < L; ++i) { b.av[i] = u(rng); b.bv[i] = u(rng); }
        Feat f = compute(b); sink += f.obi5 + f.micro;
    }
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%ld books in %.3fs (%.0f ns/book incl. RNG)  checksum=%g\n", N, s, s / N * 1e9, sink);
}
