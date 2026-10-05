#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

// ------------------------------------------------------------
// SHA-256 constants
// ------------------------------------------------------------

__device__ __constant__ uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__device__ __forceinline__ uint32_t rotr(uint32_t x, uint32_t n)
{
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z)
{
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z)
{
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ uint32_t ep0(uint32_t x)
{
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

__device__ __forceinline__ uint32_t ep1(uint32_t x)
{
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

__device__ __forceinline__ uint32_t sig0(uint32_t x)
{
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ uint32_t sig1(uint32_t x)
{
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

__device__ void sha256_transform_words(
    uint32_t state[8],
    const uint32_t message_words[16])
{
    uint32_t w[64];

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        w[i] = message_words[i];
    }

    #pragma unroll
    for (int i = 16; i < 64; i++) {
        w[i] = sig1(w[i - 2]) +
               w[i - 7] +
               sig0(w[i - 15]) +
               w[i - 16];
    }

    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + ep1(e) + ch(e, f, g) + K[i] + w[i];
        uint32_t t2 = ep0(a) + maj(a, b, c);

        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

template <bool HasPrecomputedFirstWords>
__device__ void sha256_transform_rolling(
    uint32_t state[8],
    uint32_t schedule[16],
    uint32_t precomputed_w16,
    uint32_t precomputed_w17,
    uint32_t precomputed_w18_base,
    uint32_t precomputed_w19_base)
{
    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    #pragma unroll
    for (int t = 0; t < 64; t++) {
        uint32_t word;

        if (t < 16) {
            word = schedule[t];
        }
        else {
            const int slot = t & 15;
            if (HasPrecomputedFirstWords && t == 16) {
                word = precomputed_w16;
            }
            else if (HasPrecomputedFirstWords && t == 17) {
                word = precomputed_w17;
            }
            else if (HasPrecomputedFirstWords && t == 18) {
                word = precomputed_w18_base +
                       sig0(schedule[(t - 15) & 15]);
            }
            else if (HasPrecomputedFirstWords && t == 19) {
                word = precomputed_w19_base +
                       schedule[(t - 16) & 15];
            }
            else {
                word = sig1(schedule[(t - 2) & 15]) +
                       schedule[(t - 7) & 15] +
                       sig0(schedule[(t - 15) & 15]) +
                       schedule[slot];
            }
            schedule[slot] = word;
        }

        uint32_t t1 = h + ep1(e) + ch(e, f, g) + K[t] + word;
        uint32_t t2 = ep0(a) + maj(a, b, c);

        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

__device__ void sha256_80_from_midstate_words(
    const uint32_t tail_words[3],
    uint32_t nonce_word,
    const uint32_t midstate[8],
    const uint32_t precomputed_schedule_words[4],
    uint32_t first_hash_words[8])
{
    uint32_t state[8];
    uint32_t schedule[16] = {0};

    #pragma unroll
    for (int i = 0; i < 8; i++)
        state[i] = midstate[i];

    #pragma unroll
    for (int i = 0; i < 3; i++)
        schedule[i] = tail_words[i];

    // Header bytes encode the nonce little-endian; SHA-256 reads big-endian words.
    schedule[3] = nonce_word;
    schedule[4] = 0x80000000;
    // 80 bytes = 640 bits = 0x280
    schedule[15] = 0x00000280;

    sha256_transform_rolling<true>(
        state,
        schedule,
        precomputed_schedule_words[0],
        precomputed_schedule_words[1],
        precomputed_schedule_words[2],
        precomputed_schedule_words[3]
    );

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        first_hash_words[i] = state[i];
    }
}

__device__ void sha256_32_from_words(
    const uint32_t first_hash_words[8],
    uint32_t final_hash_words[8])
{
    uint32_t state[8] = {
        0x6a09e667,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19
    };

    uint32_t schedule[16] = {0};

    #pragma unroll
    for (int i = 0; i < 8; i++)
        schedule[i] = first_hash_words[i];

    schedule[8] = 0x80000000;

    // 32 bytes = 256 bits = 0x100
    schedule[15] = 0x00000100;

    sha256_transform_rolling<false>(state, schedule, 0, 0, 0, 0);

    #pragma unroll
    for (int i = 0; i < 8; i++)
        final_hash_words[i] = state[i];
}

__device__ bool hash_meets_target(
    const uint32_t hash_words[8],
    const uint32_t target_words[8])
{
    // Compare from the most-significant Bitcoin hash/target word.
    for (int i = 0; i < 8; i++) {
        uint32_t hash_word = hash_words[7 - i];
        hash_word =
            (hash_word >> 24) |
            ((hash_word >> 8) & 0x0000ff00) |
            ((hash_word << 8) & 0x00ff0000) |
            (hash_word << 24);

        if (hash_word < target_words[i])
            return true;
        if (hash_word > target_words[i])
            return false;
    }

    return true;
}

uint32_t read_be32(const uint8_t *bytes)
{
    return ((uint32_t)bytes[0] << 24) |
           ((uint32_t)bytes[1] << 16) |
           ((uint32_t)bytes[2] << 8) |
           (uint32_t)bytes[3];
}

uint32_t read_le32(const uint8_t *bytes)
{
    return (uint32_t)bytes[0] |
           ((uint32_t)bytes[1] << 8) |
           ((uint32_t)bytes[2] << 16) |
           ((uint32_t)bytes[3] << 24);
}

bool parse_uint32(const char *text, uint32_t *value)
{
    if (text == nullptr || *text == '\0' || *text == '-')
        return false;

    char *end = nullptr;
    unsigned long long parsed = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' || parsed > 0xffffffffULL)
        return false;

    *value = static_cast<uint32_t>(parsed);
    return true;
}

int hex_digit(char value)
{
    if (value >= '0' && value <= '9')
        return value - '0';
    if (value >= 'a' && value <= 'f')
        return value - 'a' + 10;
    if (value >= 'A' && value <= 'F')
        return value - 'A' + 10;
    return -1;
}

bool parse_header_hex(const char *text, uint8_t header[80])
{
    if (std::strlen(text) != 160)
        return false;

    for (int i = 0; i < 80; i++) {
        int high = hex_digit(text[i * 2]);
        int low = hex_digit(text[i * 2 + 1]);
        if (high < 0 || low < 0)
            return false;
        header[i] = static_cast<uint8_t>((high << 4) | low);
    }

    return true;
}

bool compact_bits_to_target(uint32_t bits, uint8_t target[32])
{
    memset(target, 0, 32);

    uint32_t exponent = bits >> 24;
    uint32_t mantissa = bits & 0x007fffff;
    bool negative = mantissa != 0 && (bits & 0x00800000) != 0;
    bool overflow = mantissa != 0 &&
        (exponent > 34 ||
         (mantissa > 0xff && exponent > 33) ||
         (mantissa > 0xffff && exponent > 32));

    if (negative || overflow || mantissa == 0)
        return false;

    if (exponent <= 3) {
        mantissa >>= 8 * (3 - exponent);
        for (uint32_t i = 0; i < 3; i++)
            target[i] = (mantissa >> (8 * i)) & 0xff;
    }
    else {
        uint32_t offset = exponent - 3;
        for (uint32_t i = 0; i < 3 && offset + i < 32; i++)
            target[offset + i] = (mantissa >> (8 * i)) & 0xff;
    }

    for (uint32_t i = 0; i < 32; i++) {
        if (target[i] != 0)
            return true;
    }

    return false;
}

__device__ void hash_header_nonce(
    const uint32_t *tail_words,
    const uint32_t *midstate,
    const uint32_t *precomputed_schedule_words,
    uint32_t nonce,
    uint32_t final_hash_words[8])
{
    uint32_t nonce_word =
        ((nonce & 0x000000ff) << 24) |
        ((nonce & 0x0000ff00) << 8) |
        ((nonce & 0x00ff0000) >> 8) |
        ((nonce & 0xff000000) >> 24);

    uint32_t first_hash_words[8];

    sha256_80_from_midstate_words(
        tail_words,
        nonce_word,
        midstate,
        precomputed_schedule_words,
        first_hash_words
    );
    sha256_32_from_words(first_hash_words, final_hash_words);
}

__global__ void prepare_midstate_kernel(
    const uint8_t *base_header,
    const uint32_t *tail_words,
    uint32_t *midstate,
    uint32_t *precomputed_schedule_words)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    uint32_t state[8] = {
        0x6a09e667,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19
    };
    uint32_t message_words[16];

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int offset = i * 4;
        message_words[i] =
            ((uint32_t)base_header[offset] << 24) |
            ((uint32_t)base_header[offset + 1] << 16) |
            ((uint32_t)base_header[offset + 2] << 8) |
            (uint32_t)base_header[offset + 3];
    }

    sha256_transform_words(state, message_words);

    #pragma unroll
    for (int i = 0; i < 8; i++)
        midstate[i] = state[i];

    // For this first-hash tail layout, W[16] and W[17] do not
    // depend on the nonce, so compute them once before the scan.
    precomputed_schedule_words[0] =
        sig0(tail_words[1]) + tail_words[0];
    precomputed_schedule_words[1] =
        sig1(0x00000280) + sig0(tail_words[2]) + tail_words[1];
    precomputed_schedule_words[2] =
        sig1(precomputed_schedule_words[0]) + tail_words[2];
    precomputed_schedule_words[3] =
        sig1(precomputed_schedule_words[1]) + sig0(0x80000000);
}

__global__ void mine_kernel(
    const uint32_t *tail_words,
    const uint32_t *target_words,
    const uint32_t *midstate,
    const uint32_t *precomputed_schedule_words,
    uint32_t start_nonce,
    uint32_t nonce_count,
    uint32_t *found_nonce)
{
    uint32_t nonce_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (nonce_index >= nonce_count)
        return;

    uint32_t nonce = start_nonce + nonce_index;
    uint32_t final_hash_words[8];

    hash_header_nonce(
        tail_words,
        midstate,
        precomputed_schedule_words,
        nonce,
        final_hash_words
    );

    if (hash_meets_target(final_hash_words, target_words))
        atomicMin(found_nonce, nonce);
}

__global__ void verify_nonce_kernel(
    const uint32_t *tail_words,
    const uint32_t *midstate,
    const uint32_t *precomputed_schedule_words,
    uint32_t nonce,
    uint32_t *hash_words)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        hash_header_nonce(
            tail_words,
            midstate,
            precomputed_schedule_words,
            nonce,
            hash_words
        );
}

int main(int argc, char **argv)
{
    bool scan_mode = false;
    uint32_t scan_start = 0;
    uint32_t scan_count = 0;
    uint8_t header[80] = {0};

    if (argc == 7 &&
        std::strcmp(argv[1], "--scan-header") == 0 &&
        std::strcmp(argv[3], "--start") == 0 &&
        std::strcmp(argv[5], "--count") == 0) {
        if (!parse_header_hex(argv[2], header) ||
            !parse_uint32(argv[4], &scan_start) ||
            !parse_uint32(argv[6], &scan_count) ||
            scan_count == 0 ||
            (uint64_t)scan_start + scan_count > 0x100000000ULL) {
            std::cerr
                << "Usage: cuda_miner --scan-header <80-byte-hex> "
                << "--start <uint32> --count <1..uint32>\n";
            return 2;
        }
        scan_mode = true;
    }
    else if (argc != 1) {
        std::cerr
            << "Usage: cuda_miner [--scan-header <80-byte-hex> "
            << "--start <uint32> --count <1..uint32>]\n";
        return 2;
    }

    // Same synthetic header used by our CPU benchmark:
    //
    // version     = 0x20000000
    // prev hash   = 0
    // merkle root = 0
    // timestamp   = 1728000000
    // bits        = 0x1f00ffff
    //
    // Nonce gets replaced by the GPU.

    if (!scan_mode) {
        uint32_t version = 0x20000000;
        uint32_t timestamp = 1728000000;
        uint32_t bits = 0x1f00ffff;

        memcpy(header + 0, &version, 4);
        memcpy(header + 68, &timestamp, 4);
        memcpy(header + 72, &bits, 4);
    }

    uint32_t bits = read_le32(header + 72);

    uint8_t target[32];
    if (!compact_bits_to_target(bits, target)) {
        std::cerr << "Invalid compact target in nBits.\n";
        return 1;
    }

    uint32_t tail_words[3] = {
        read_be32(header + 64),
        read_be32(header + 68),
        read_be32(header + 72)
    };
    uint32_t target_words[8];
    for (int i = 0; i < 8; i++) {
        int offset = 31 - i * 4;
        target_words[i] =
            ((uint32_t)target[offset] << 24) |
            ((uint32_t)target[offset - 1] << 16) |
            ((uint32_t)target[offset - 2] << 8) |
            (uint32_t)target[offset - 3];
    }

    uint8_t *device_header = nullptr;
    uint32_t *device_tail_words = nullptr;
    uint32_t *device_target_words = nullptr;
    uint32_t *device_result = nullptr;
    uint32_t *device_midstate = nullptr;
    uint32_t *device_precomputed_schedule_words = nullptr;
    uint32_t *device_hash_words = nullptr;

    cudaMalloc(&device_header, 80);
    cudaMalloc(&device_tail_words, sizeof(tail_words));
    cudaMalloc(&device_target_words, sizeof(target_words));
    cudaMalloc(&device_result, sizeof(uint32_t));
    cudaMalloc(&device_midstate, 8 * sizeof(uint32_t));
    cudaMalloc(&device_precomputed_schedule_words, 4 * sizeof(uint32_t));
    cudaMalloc(&device_hash_words, 8 * sizeof(uint32_t));

    cudaMemcpy(
        device_header,
        header,
        80,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        device_tail_words,
        tail_words,
        sizeof(tail_words),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        device_target_words,
        target_words,
        sizeof(target_words),
        cudaMemcpyHostToDevice
    );

    prepare_midstate_kernel<<<1, 1>>>(
        device_header,
        device_tail_words,
        device_midstate,
        device_precomputed_schedule_words
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr
            << "CUDA midstate kernel error: "
            << cudaGetErrorString(err)
            << "\n";
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr
            << "CUDA midstate preparation error: "
            << cudaGetErrorString(err)
            << "\n";
        return 1;
    }

    uint32_t initial_result = 0xffffffff;

    cudaMemcpy(
        device_result,
        &initial_result,
        sizeof(uint32_t),
        cudaMemcpyHostToDevice
    );

    if (scan_mode) {
        const int threads = 256;
        const unsigned blocks = static_cast<unsigned>(
            ((uint64_t)scan_count + threads - 1) / threads
        );

        mine_kernel<<<blocks, threads>>>(
            device_tail_words,
            device_target_words,
            device_midstate,
            device_precomputed_schedule_words,
            scan_start,
            scan_count,
            device_result
        );

        cudaError_t scan_error = cudaGetLastError();
        if (scan_error == cudaSuccess)
            scan_error = cudaDeviceSynchronize();
        if (scan_error != cudaSuccess) {
            std::cerr
                << "CUDA scan error: "
                << cudaGetErrorString(scan_error)
                << "\n";
            return 1;
        }

        uint32_t found_nonce;
        cudaMemcpy(
            &found_nonce,
            device_result,
            sizeof(uint32_t),
            cudaMemcpyDeviceToHost
        );

        if (found_nonce == 0xffffffff) {
            std::cout << "NONE\n";
        }
        else {
            verify_nonce_kernel<<<1, 1>>>(
                device_tail_words,
                device_midstate,
                device_precomputed_schedule_words,
                found_nonce,
                device_hash_words
            );

            scan_error = cudaGetLastError();
            if (scan_error == cudaSuccess)
                scan_error = cudaDeviceSynchronize();
            if (scan_error != cudaSuccess) {
                std::cerr
                    << "CUDA hash verification error: "
                    << cudaGetErrorString(scan_error)
                    << "\n";
                return 1;
            }

            uint32_t found_hash_words[8];
            cudaMemcpy(
                found_hash_words,
                device_hash_words,
                sizeof(found_hash_words),
                cudaMemcpyDeviceToHost
            );

            std::cout << "FOUND " << found_nonce << " ";
            for (int i = 7; i >= 0; i--) {
                for (int byte = 0; byte < 4; byte++)
                    std::printf(
                        "%02x",
                        static_cast<unsigned>(
                            (found_hash_words[i] >> (byte * 8)) & 0xff
                        )
                    );
            }
            std::cout << "\n";
        }

        cudaFree(device_header);
        cudaFree(device_tail_words);
        cudaFree(device_target_words);
        cudaFree(device_result);
        cudaFree(device_midstate);
        cudaFree(device_precomputed_schedule_words);
        cudaFree(device_hash_words);
        return 0;
    }

    const uint32_t nonce_count = 500000000;
    const uint32_t warmup_nonce_count = 1000000;
    const int measured_runs = 5;

    const int threads = 256;
    const unsigned blocks = static_cast<unsigned>(
        ((uint64_t)nonce_count + threads - 1) / threads
    );
    const unsigned warmup_blocks =
        (warmup_nonce_count + threads - 1) / threads;

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    mine_kernel<<<warmup_blocks, threads>>>(
        device_tail_words,
        device_target_words,
        device_midstate,
        device_precomputed_schedule_words,
        0,
        warmup_nonce_count,
        device_result
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr
            << "CUDA warm-up error: "
            << cudaGetErrorString(err)
            << "\n";
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr
            << "CUDA warm-up synchronization error: "
            << cudaGetErrorString(err)
            << "\n";
        return 1;
    }

    float run_milliseconds[5];
    for (int run = 0; run < measured_runs; run++) {
        cudaMemcpy(
            device_result,
            &initial_result,
            sizeof(uint32_t),
            cudaMemcpyHostToDevice
        );

        cudaEventRecord(start);
        mine_kernel<<<blocks, threads>>>(
            device_tail_words,
            device_target_words,
            device_midstate,
            device_precomputed_schedule_words,
            0,
            nonce_count,
            device_result
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr
                << "CUDA mining error: "
                << cudaGetErrorString(err)
                << "\n";
            return 1;
        }

        cudaEventRecord(stop);
        err = cudaEventSynchronize(stop);
        if (err != cudaSuccess) {
            std::cerr
                << "CUDA benchmark synchronization error: "
                << cudaGetErrorString(err)
                << "\n";
            return 1;
        }

        err = cudaEventElapsedTime(&run_milliseconds[run], start, stop);
        if (err != cudaSuccess) {
            std::cerr
                << "CUDA event timing error: "
                << cudaGetErrorString(err)
                << "\n";
            return 1;
        }
    }

    uint32_t found_nonce;
    cudaMemcpy(
        &found_nonce,
        device_result,
        sizeof(uint32_t),
        cudaMemcpyDeviceToHost
    );

    uint32_t found_hash_words[8];
    if (found_nonce != 0xffffffff) {
        verify_nonce_kernel<<<1, 1>>>(
            device_tail_words,
            device_midstate,
            device_precomputed_schedule_words,
            found_nonce,
            device_hash_words
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr
                << "CUDA verification error: "
                << cudaGetErrorString(err)
                << "\n";
            return 1;
        }

        cudaMemcpy(
            found_hash_words,
            device_hash_words,
            sizeof(found_hash_words),
            cudaMemcpyDeviceToHost
        );
    }

    std::cout << "GPU hashes tested: "
              << nonce_count << "\n";
    std::cout << "Warm-up hashes: "
              << warmup_nonce_count << "\n";
    double total_hashrate = 0.0;
    for (int run = 0; run < measured_runs; run++) {
        double seconds = run_milliseconds[run] / 1000.0;
        double hashrate = nonce_count / seconds;
        total_hashrate += hashrate;
        std::cout << "Run " << (run + 1)
                  << ": " << seconds << " seconds, "
                  << hashrate << " H/s\n";
    }
    std::cout << "Average hashrate: "
              << total_hashrate / measured_runs << " H/s\n";

    std::cout << "Target: ";
    for (int i = 31; i >= 0; i--)
        std::printf("%02x", target[i]);
    std::cout << "\n";

    if (found_nonce != 0xffffffff) {
        std::cout << "Lowest candidate nonce: "
                  << found_nonce << "\n";
        std::cout << "Displayed hash: ";
        for (int i = 7; i >= 0; i--) {
            for (int byte = 0; byte < 4; byte++)
                std::printf(
                    "%02x",
                    static_cast<unsigned>(
                        (found_hash_words[i] >> (byte * 8)) & 0xff
                    )
                );
        }
        std::cout << "\n";
    }
    else {
        std::cout << "No candidate found.\n";
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(device_header);
    cudaFree(device_tail_words);
    cudaFree(device_target_words);
    cudaFree(device_result);
    cudaFree(device_midstate);
    cudaFree(device_precomputed_schedule_words);
    cudaFree(device_hash_words);

    return 0;
}