// SPDX-License-Identifier: GPL-3.0-or-later
// Two-process adaptation of koenjw-wq/rtx3090-host-staged-allreduce.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

constexpr size_t ChunkBytes = 1 << 20;
constexpr size_t MaxBytes = 80 << 20;
constexpr int MaxChunks = MaxBytes / ChunkBytes;
constexpr size_t HeaderBytes = 4096;
constexpr size_t MappingBytes = HeaderBytes + 2 * MaxBytes;
struct alignas(64) Flag { unsigned long long value; char pad[56]; };
struct Header {
    Flag consumed[2];
    unsigned long long published[2][MaxChunks];
};
static_assert(sizeof(Header) <= HeaderBytes);
thread_local std::string error_message;

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
__device__ unsigned long long timer_ns() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}
__device__ unsigned long long load_host(const unsigned long long* address) {
    unsigned long long value;
    asm volatile("ld.global.cv.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
    return value;
}
__global__ void next_sequence(unsigned long long* sequence) { ++*sequence; }
__global__ void wait_flag(const unsigned long long* flag,
                          const unsigned long long* sequence, bool previous) {
    const unsigned long long target = *sequence - (previous ? 1 : 0);
    const unsigned long long began = timer_ns();
    while (load_host(flag) < target) {
        if (timer_ns() - began > 60000000000ULL) {
            assert(false && "Host-staged all-reduce peer progress timed out");
            return;
        }
        __nanosleep(128);
    }
    __threadfence_system();
}
__global__ void publish_flag(unsigned long long* flag,
                             const unsigned long long* sequence) {
    __threadfence_system();
    *reinterpret_cast<volatile unsigned long long*>(flag) = *sequence;
    __threadfence_system();
}
__global__ void add_bf16(const __nv_bfloat16* own, const __nv_bfloat16* peer,
                         __nv_bfloat16* output, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) output[i] = __hadd(own[i], peer[i]);
}
struct Context {
    int device = -1;
    int rank = -1;
    void* mapping = MAP_FAILED;
    void* mapped_device = nullptr;
    bool registered = false;
    unsigned long long* sequence = nullptr;
    __nv_bfloat16* remote = nullptr;
    cudaStream_t send = nullptr, receive = nullptr;
    cudaEvent_t ready = nullptr, sent = nullptr, received = nullptr;
    ~Context() {
        if (device >= 0) cudaSetDevice(device);
        if (send) cudaStreamDestroy(send);
        if (receive) cudaStreamDestroy(receive);
        if (ready) cudaEventDestroy(ready);
        if (sent) cudaEventDestroy(sent);
        if (received) cudaEventDestroy(received);
        if (remote) cudaFree(remote);
        if (sequence) cudaFree(sequence);
        if (registered) cudaHostUnregister(mapping);
        if (mapping != MAP_FAILED) munmap(mapping, MappingBytes);
    }
};
extern "C" const char* hs_error() { return error_message.c_str(); }
extern "C" void hs_unlink(const char* name) { shm_unlink(name); }
extern "C" void hs_close(void* handle) { delete static_cast<Context*>(handle); }
extern "C" void* hs_open(const char* name, int rank, int device, int create) {
    Context* context = new Context;
    int fd = -1;
    try {
        if (rank < 0 || rank > 1) throw std::runtime_error("Expected two ranks");
        context->device = device;
        context->rank = rank;
        check(cudaSetDevice(device));
        cudaDeviceProp properties;
        check(cudaGetDeviceProperties(&properties, device));
        if (properties.major != 8 || properties.minor != 6 ||
            std::string(properties.name).find("RTX 3090") == std::string::npos)
            throw std::runtime_error("This backend is qualified only for RTX 3090");
        fd = shm_open(name, O_RDWR | (create ? O_CREAT | O_EXCL : 0), 0600);
        if (fd < 0) throw std::runtime_error(std::strerror(errno));
        if (create && ftruncate(fd, MappingBytes)) throw std::runtime_error(std::strerror(errno));
        struct stat info;
        if (fstat(fd, &info) || info.st_size != MappingBytes)
            throw std::runtime_error("Shared mapping has an unexpected size");
        context->mapping = mmap(nullptr, MappingBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        fd = -1;
        if (context->mapping == MAP_FAILED) throw std::runtime_error(std::strerror(errno));
        if (create) std::memset(context->mapping, 0, MappingBytes);
        check(cudaHostRegister(context->mapping, MappingBytes, cudaHostRegisterPortable | cudaHostRegisterMapped));
        context->registered = true;
        check(cudaHostGetDevicePointer(&context->mapped_device, context->mapping, 0));
        check(cudaMalloc(&context->remote, MaxBytes));
        check(cudaMalloc(&context->sequence, sizeof(unsigned long long)));
        check(cudaMemset(context->sequence, 0, sizeof(unsigned long long)));
        check(cudaStreamCreateWithFlags(&context->send, cudaStreamNonBlocking));
        check(cudaStreamCreateWithFlags(&context->receive, cudaStreamNonBlocking));
        check(cudaEventCreateWithFlags(&context->ready, cudaEventDisableTiming));
        check(cudaEventCreateWithFlags(&context->sent, cudaEventDisableTiming));
        check(cudaEventCreateWithFlags(&context->received, cudaEventDisableTiming));
        check(cudaDeviceSynchronize());
        return context;
    } catch (const std::exception& error) {
        error_message = error.what();
        if (fd >= 0) close(fd);
        delete context;
        if (create) shm_unlink(name);
        return nullptr;
    }
}
extern "C" int hs_reduce(void* handle, const void* input, void* output,
                          size_t bytes, void* stream_pointer) {
    try {
        auto* context = static_cast<Context*>(handle);
        if (!context || !bytes || bytes > MaxBytes || bytes % 2)
            throw std::runtime_error("Invalid host-staged reduction arguments");
        check(cudaSetDevice(context->device));
        auto main = static_cast<cudaStream_t>(stream_pointer);
        auto* header = static_cast<Header*>(context->mapped_device);
        auto* own_host = static_cast<char*>(context->mapping) + HeaderBytes + context->rank * MaxBytes;
        auto* peer_host = static_cast<char*>(context->mapping) + HeaderBytes + (1 - context->rank) * MaxBytes;
        next_sequence<<<1, 1, 0, main>>>(context->sequence);
        check(cudaGetLastError());
        check(cudaEventRecord(context->ready, main));
        check(cudaStreamWaitEvent(context->send, context->ready));
        check(cudaStreamWaitEvent(context->receive, context->ready));
        wait_flag<<<1, 1, 0, context->send>>>(&header->consumed[1 - context->rank].value, context->sequence, true);
        for (size_t offset = 0, chunk = 0; offset < bytes; offset += ChunkBytes, ++chunk) {
            const size_t length = bytes - offset < ChunkBytes ? bytes - offset : ChunkBytes;
            check(cudaMemcpyAsync(own_host + offset, static_cast<const char*>(input) + offset,
                                  length, cudaMemcpyDeviceToHost, context->send));
            publish_flag<<<1, 1, 0, context->send>>>(&header->published[context->rank][chunk], context->sequence);
            check(cudaGetLastError());
            wait_flag<<<1, 1, 0, context->receive>>>(&header->published[1 - context->rank][chunk], context->sequence, false);
            check(cudaGetLastError());
            check(cudaMemcpyAsync(reinterpret_cast<char*>(context->remote) + offset, peer_host + offset,
                                  length, cudaMemcpyHostToDevice, context->receive));
        }
        publish_flag<<<1, 1, 0, context->receive>>>(&header->consumed[context->rank].value, context->sequence);
        check(cudaGetLastError());
        check(cudaEventRecord(context->sent, context->send));
        check(cudaEventRecord(context->received, context->receive));
        check(cudaStreamWaitEvent(main, context->sent));
        check(cudaStreamWaitEvent(main, context->received));
        add_bf16<<<256, 256, 0, main>>>(static_cast<const __nv_bfloat16*>(input), context->remote,
                                      static_cast<__nv_bfloat16*>(output), bytes / 2);
        check(cudaGetLastError());
        return 0;
    } catch (const std::exception& error) {
        error_message = error.what();
        return -1;
    }
}
