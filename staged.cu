#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <string_view>
#include <vector>

void check(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        std::cerr << expression << ": " << cudaGetErrorString(status) << '\n';
        std::exit(1);
    }
}
#define CUDA(expr) check((expr), #expr)
using Bf16 = __nv_bfloat16;

__global__ void reduce(const Bf16* own, const Bf16* peer, Bf16* out, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        out[i] = __hadd(own[i], peer[i]);
    }
}

struct Rank {
    Bf16 *input, *output, *host, *mapped, *remote;
    cudaStream_t stream, send, compute;
    cudaEvent_t published, consumed, arrived, reduced;
};

class Pair {
public:
    explicit Pair(size_t bytes) {
        for (int i = 0; i < 2; ++i) {
            Rank& rank = ranks[i];
            CUDA(cudaSetDevice(i));
            CUDA(cudaMalloc(&rank.input, bytes));
            CUDA(cudaMalloc(&rank.output, bytes));
            CUDA(cudaMalloc(&rank.remote, bytes));
            CUDA(cudaHostAlloc(&rank.host, bytes, cudaHostAllocPortable | cudaHostAllocMapped));
            CUDA(cudaStreamCreateWithFlags(&rank.stream, cudaStreamNonBlocking));
            CUDA(cudaStreamCreateWithFlags(&rank.send, cudaStreamNonBlocking));
            CUDA(cudaStreamCreateWithFlags(&rank.compute, cudaStreamNonBlocking));
            CUDA(cudaEventCreateWithFlags(&rank.arrived, cudaEventDisableTiming));
            CUDA(cudaEventCreateWithFlags(&rank.reduced, cudaEventDisableTiming));
            CUDA(cudaEventCreateWithFlags(&rank.published, cudaEventDisableTiming));
            CUDA(cudaEventCreateWithFlags(&rank.consumed, cudaEventDisableTiming));
        }
        for (int i = 0; i < 2; ++i) {
            CUDA(cudaSetDevice(i));
            CUDA(cudaHostGetDevicePointer(&ranks[i].mapped, ranks[1 - i].host, 0));
        }
    }

    ~Pair() {
        for (int i = 0; i < 2; ++i) {
            CUDA(cudaSetDevice(i));
            Rank& rank = ranks[i];
            CUDA(cudaStreamDestroy(rank.stream));
            CUDA(cudaStreamDestroy(rank.send));
            CUDA(cudaStreamDestroy(rank.compute));
            CUDA(cudaEventDestroy(rank.arrived));
            CUDA(cudaEventDestroy(rank.reduced));
            CUDA(cudaEventDestroy(rank.published));
            CUDA(cudaEventDestroy(rank.consumed));
            CUDA(cudaFree(rank.input));
            CUDA(cudaFree(rank.output));
            CUDA(cudaFree(rank.remote));
            CUDA(cudaFreeHost(rank.host));
        }
    }

    void setup(const std::array<std::vector<Bf16>, 2>& source, size_t bytes) {
        for (int i = 0; i < 2; ++i) {
            CUDA(cudaSetDevice(i));
            CUDA(cudaMemcpy(ranks[i].input, source[i].data(), bytes, cudaMemcpyHostToDevice));
            CUDA(cudaDeviceSynchronize());
        }
    }

    void poison_output(size_t bytes) {
        for (int i = 0; i < 2; ++i) {
            CUDA(cudaSetDevice(i));
            CUDA(cudaMemset(ranks[i].output, 0xff, bytes));
            CUDA(cudaDeviceSynchronize());
        }
    }

    cudaGraphExec_t capture(int count, int iterations, bool direct_read) {
        CUDA(cudaSetDevice(0));
        CUDA(cudaStreamBeginCapture(ranks[0].stream, cudaStreamCaptureModeGlobal));
        CUDA(cudaEventRecord(ranks[0].consumed, ranks[0].stream));
        CUDA(cudaSetDevice(1));
        CUDA(cudaStreamWaitEvent(ranks[1].stream, ranks[0].consumed));
        const size_t bytes = count * sizeof(Bf16);
        for (int iteration = 0; iteration < iterations; ++iteration) {
            for (int i = 0; i < 2; ++i) {
                CUDA(cudaSetDevice(i));
                Rank& rank = ranks[i];
                Rank& other = ranks[1 - i];
                if (iteration != 0) CUDA(cudaStreamWaitEvent(rank.stream, other.consumed));
                CUDA(cudaMemcpyAsync(rank.host, rank.input, bytes, cudaMemcpyDeviceToHost, rank.stream));
                CUDA(cudaEventRecord(rank.published, rank.stream));
            }
            for (int i = 0; i < 2; ++i) {
                CUDA(cudaSetDevice(i));
                Rank& rank = ranks[i];
                Rank& other = ranks[1 - i];
                CUDA(cudaStreamWaitEvent(rank.stream, other.published));
                if (!direct_read) {
                    CUDA(cudaMemcpyAsync(rank.remote, other.host, bytes,
                                         cudaMemcpyHostToDevice, rank.stream));
                    CUDA(cudaEventRecord(rank.consumed, rank.stream));
                }
                const Bf16* peer = direct_read ? rank.mapped : rank.remote;
                reduce<<<std::min((count + 255) / 256, 256), 256, 0, rank.stream>>>(
                    rank.input, peer, rank.output, count);
                CUDA(cudaGetLastError());
                if (direct_read) CUDA(cudaEventRecord(rank.consumed, rank.stream));
            }
        }
        CUDA(cudaEventRecord(ranks[1].published, ranks[1].stream));
        CUDA(cudaSetDevice(0));
        CUDA(cudaStreamWaitEvent(ranks[0].stream, ranks[1].published));
        cudaGraph_t graph;
        CUDA(cudaStreamEndCapture(ranks[0].stream, &graph));
        cudaGraphExec_t executable;
        CUDA(cudaGraphInstantiateWithFlags(&executable, graph, 0));
        CUDA(cudaGraphDestroy(graph));
        return executable;
    }

    cudaGraphExec_t capture_pipeline(int count, int iterations, int chunk_elements, bool overlap_reduce = false) {
        CUDA(cudaSetDevice(0));
        CUDA(cudaStreamBeginCapture(ranks[0].stream, cudaStreamCaptureModeGlobal));
        CUDA(cudaEventRecord(ranks[0].consumed, ranks[0].stream));
        CUDA(cudaStreamWaitEvent(ranks[0].send, ranks[0].consumed));
        CUDA(cudaSetDevice(1));
        CUDA(cudaStreamWaitEvent(ranks[1].stream, ranks[0].consumed));
        CUDA(cudaStreamWaitEvent(ranks[1].send, ranks[0].consumed));
        if (overlap_reduce) {
            for (int i = 0; i < 2; ++i) {
                CUDA(cudaSetDevice(i));
                CUDA(cudaStreamWaitEvent(ranks[i].compute, ranks[0].consumed));
            }
        }
        for (int iteration = 0; iteration < iterations; ++iteration) {
            if (iteration) {
                for (int i = 0; i < 2; ++i) {
                    CUDA(cudaSetDevice(i));
                    // The peer must finish reading this rank's slab before reuse.
                    CUDA(cudaStreamWaitEvent(ranks[i].send, ranks[1 - i].consumed));
                    if (overlap_reduce) {
                        CUDA(cudaStreamWaitEvent(ranks[i].stream, ranks[i].reduced));
                    }
                }
            }
            for (int start = 0; start < count; start += chunk_elements) {
                const int length = std::min(chunk_elements, count - start);
                const size_t bytes = length * sizeof(Bf16);
                for (int i = 0; i < 2; ++i) {
                    CUDA(cudaSetDevice(i));
                    Rank& rank = ranks[i];
                    CUDA(cudaMemcpyAsync(rank.host + start, rank.input + start, bytes,
                                         cudaMemcpyDeviceToHost, rank.send));
                    CUDA(cudaEventRecord(rank.published, rank.send));
                }
                for (int i = 0; i < 2; ++i) {
                    CUDA(cudaSetDevice(i));
                    Rank& rank = ranks[i];
                    Rank& other = ranks[1 - i];
                    CUDA(cudaStreamWaitEvent(rank.stream, other.published));
                    CUDA(cudaMemcpyAsync(rank.remote + start, other.host + start, bytes,
                                         cudaMemcpyHostToDevice, rank.stream));
                    if (overlap_reduce) {
                        CUDA(cudaEventRecord(rank.arrived, rank.stream));
                        CUDA(cudaStreamWaitEvent(rank.compute, rank.arrived));
                        reduce<<<std::min((length + 255) / 256, 256), 256, 0, rank.compute>>>(
                            rank.input + start, rank.remote + start, rank.output + start, length);
                        CUDA(cudaGetLastError());
                    }
                }
            }
            for (int i = 0; i < 2; ++i) {
                CUDA(cudaSetDevice(i));
                Rank& rank = ranks[i];
                CUDA(cudaEventRecord(rank.consumed, rank.stream));
                if (overlap_reduce) {
                    CUDA(cudaEventRecord(rank.reduced, rank.compute));
                    continue;
                }
                reduce<<<std::min((count + 255) / 256, 256), 256, 0, rank.stream>>>(
                    rank.input, rank.remote, rank.output, count);
                CUDA(cudaGetLastError());
            }
        }
        if (overlap_reduce) {
            for (int i = 0; i < 2; ++i) {
                CUDA(cudaSetDevice(i));
                CUDA(cudaStreamWaitEvent(ranks[i].stream, ranks[i].reduced));
            }
        }
        CUDA(cudaEventRecord(ranks[1].consumed, ranks[1].stream));
        CUDA(cudaSetDevice(0));
        CUDA(cudaStreamWaitEvent(ranks[0].stream, ranks[1].consumed));
        cudaGraph_t graph;
        CUDA(cudaStreamEndCapture(ranks[0].stream, &graph));
        cudaGraphExec_t executable;
        CUDA(cudaGraphInstantiateWithFlags(&executable, graph, 0));
        CUDA(cudaGraphDestroy(graph));
        return executable;
    }

    double measure(cudaGraphExec_t graph, int iterations) {
        CUDA(cudaSetDevice(0));
        const auto start = std::chrono::steady_clock::now();
        CUDA(cudaGraphLaunch(graph, ranks[0].stream));
        CUDA(cudaStreamSynchronize(ranks[0].stream));
        return std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count() / iterations;
    }

    void verify(const std::array<std::vector<Bf16>, 2>& source, int count) {
        std::vector<Bf16> host(count);
        for (int i = 0; i < 2; ++i) {
            CUDA(cudaSetDevice(i));
            CUDA(cudaMemcpy(host.data(), ranks[i].output, count * sizeof(Bf16),
                            cudaMemcpyDeviceToHost));
            for (int j = 0; j < count; ++j) {
                const Bf16 expected = __hadd(source[i][j], source[1 - i][j]);
                if (__bfloat16_as_ushort(host[j]) != __bfloat16_as_ushort(expected)) {
                    std::cerr << "Incorrect rank " << i << ", element " << j << '\n';
                    std::exit(2);
                }
            }
        }
    }

private:
    Rank ranks[2];
};

int main(int argc, char** argv) {
    bool single = false;
    bool sweep = false;
    bool prefill = false;
    unsigned seed = 3090;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--single") {
            single = true;
        } else if (arg == "--sweep") {
            sweep = true;
        } else if (arg == "--prefill") {
            prefill = true;
            single = true;
        } else if (arg == "--repeat") {
            seed = 6290;
        } else {
            std::cerr << "Usage: staged [--single] [--sweep] [--prefill] [--repeat]\n";
            return 1;
        }
    }
    constexpr int hidden = 5120;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> values(-10.0f, 10.0f);
    std::vector<int> shapes{1, 2, 4, 8, 16, 32, 128, 512, 1024};
    if (prefill) {
        shapes = {4096, 8192};
    } else if (sweep) {
        shapes = {32, 128, 512, 1024};
    }
    Pair pair(shapes.back() * hidden * sizeof(Bf16));
    for (int tokens : shapes) {
        int count = tokens * hidden;
        int iterations = single ? 1 : std::clamp((1 << 23) / (count * 2), 8, 128);
        std::array<std::vector<Bf16>, 2> source{std::vector<Bf16>(count), std::vector<Bf16>(count)};
        std::vector<cudaGraphExec_t> graphs{
            pair.capture(count, iterations, false),
            pair.capture(count, iterations, true),
            pair.capture_pipeline(count, iterations, 128 * 1024),
            pair.capture_pipeline(count, iterations, 512 * 1024),
        };
        if (sweep) {
            for (int chunk_kib : {32, 64, 128, 512, 2048}) {
                graphs.push_back(pair.capture_pipeline(count, iterations, chunk_kib * 1024 / sizeof(Bf16)));
            }
            for (int chunk_kib : {64, 256, 512}) {
                graphs.push_back(pair.capture_pipeline(count, iterations, chunk_kib * 1024 / sizeof(Bf16), true));
            }
        }
        std::vector<size_t> order(graphs.size());
        std::iota(order.begin(), order.end(), 0);
        for (int round = -1; round < 7; ++round) {
            for (auto& rank : source) {
                for (Bf16& value : rank) value = __float2bfloat16(values(rng));
            }
            pair.setup(source, count * sizeof(Bf16));
            std::shuffle(order.begin(), order.end(), rng);
            for (size_t mode : order) {
                pair.poison_output(count * sizeof(Bf16));
                const double time = pair.measure(graphs[mode], iterations);
                pair.verify(source, count);
                if (round >= 0) {
                    std::cout << tokens << ',' << mode << ',' << round << ','
                              << time << '\n';
                }
            }
        }
        for (auto graph : graphs) CUDA(cudaGraphExecDestroy(graph));
    }
}
