#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

// ------------------------------------------------------------
// Scan-kernel build options
// ------------------------------------------------------------
//
// The defaults are the validated configuration. Each option can be turned
// off (or the count changed) with -D at build time to measure it alone; every
// combination computes exactly the same SHA256d and applies the same full
// 256-bit target comparison.

// Resume the first hash after rounds 0-3, whose nonce-independent part is
// computed once per header instead of once per nonce.
#ifndef MINER_RESUME_FIRST_HASH
#define MINER_RESUME_FIRST_HASH 1
#endif

// Stop the second hash after round 60, when the most significant word of the
// final hash is already known, and decide on that word alone unless it equals
// the target's most significant word.
#ifndef MINER_EARLY_SECOND_HASH
#define MINER_EARLY_SECOND_HASH 1
#endif

// Consecutive nonces hashed by each GPU thread per kernel launch.
#ifndef MINER_NONCES_PER_THREAD
#define MINER_NONCES_PER_THREAD 1
#endif

// GPU threads per block of a scan launch.
#ifndef MINER_THREADS_PER_BLOCK
#define MINER_THREADS_PER_BLOCK 256
#endif

// ------------------------------------------------------------
// Version rolling (BIP 320)
// ------------------------------------------------------------
//
// BIP 320 reserves bits 13-28 of the block version for miners. Variant i of
// a header is that header with i added to this field, which must be zero in
// the header as supplied. The variants differ only in the first header
// block, so they share the message schedule of the second one; see
// mine_versions_kernel. Only SCANV requests use this; SCAN is unaffected.
const int VERSION_ROLL_SHIFT = 13;
const uint32_t VERSION_ROLL_MASK = 0xffffu << VERSION_ROLL_SHIFT;
const int MAX_VERSIONS = 16;

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

// Rounds [FirstRound, EndRound) of the same rolling-schedule compression as
// sha256_transform_rolling, applied to the working variables v = {a..h}. The
// caller supplies the starting variables and does the feed-forward addition.
template <bool HasPrecomputedFirstWords, int FirstRound, int EndRound>
__device__ __forceinline__ void sha256_rounds_rolling(
    uint32_t v[8],
    uint32_t schedule[16],
    uint32_t precomputed_w16,
    uint32_t precomputed_w17,
    uint32_t precomputed_w18_base,
    uint32_t precomputed_w19_base)
{
    uint32_t a = v[0];
    uint32_t b = v[1];
    uint32_t c = v[2];
    uint32_t d = v[3];
    uint32_t e = v[4];
    uint32_t f = v[5];
    uint32_t g = v[6];
    uint32_t h = v[7];

    #pragma unroll
    for (int t = FirstRound; t < EndRound; t++) {
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

    v[0] = a;
    v[1] = b;
    v[2] = c;
    v[3] = d;
    v[4] = e;
    v[5] = f;
    v[6] = g;
    v[7] = h;
}

// Rounds [FirstRound, EndRound) of the compression with a message schedule
// that has already been expanded, applied to the working variables v.
template <int FirstRound, int EndRound>
__device__ __forceinline__ void sha256_rounds_expanded(
    uint32_t v[8],
    const uint32_t w[64])
{
    uint32_t a = v[0];
    uint32_t b = v[1];
    uint32_t c = v[2];
    uint32_t d = v[3];
    uint32_t e = v[4];
    uint32_t f = v[5];
    uint32_t g = v[6];
    uint32_t h = v[7];

    #pragma unroll
    for (int t = FirstRound; t < EndRound; t++) {
        uint32_t t1 = h + ep1(e) + ch(e, f, g) + K[t] + w[t];
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

    v[0] = a;
    v[1] = b;
    v[2] = c;
    v[3] = d;
    v[4] = e;
    v[5] = f;
    v[6] = g;
    v[7] = h;
}

// First hash, resumed after round 3.
//
// Round t of the second header block consumes message word W[t] only, and the
// nonce is W[3]; W[0..2] are the Merkle-root tail, time and bits. So the
// working variables after rounds 0-2 do not depend on the nonce. In round 3
//
//   T1 = h + ep1(e) + ch(e, f, g) + K[3] + W[3]     T2 = ep0(a) + maj(a, b, c)
//
// only the "+ W[3]" term involves the nonce. After round 3 therefore
//
//   a = (T1 - W[3] + T2) + W[3]      e = (d + T1 - W[3]) + W[3]
//   b, c, d = old a, b, c            f, g, h = old e, f, g
//
// and resume_state holds those eight nonce-independent values, prepared once
// per header by prepare_midstate_kernel.
__device__ __forceinline__ void sha256_80_resumed(
    const uint32_t tail_words[3],
    uint32_t nonce_word,
    const uint32_t midstate[8],
    const uint32_t precomputed_schedule_words[4],
    const uint32_t resume_state[8],
    uint32_t first_hash_words[8])
{
    uint32_t schedule[16] = {0};

    #pragma unroll
    for (int i = 0; i < 3; i++)
        schedule[i] = tail_words[i];

    schedule[3] = nonce_word;
    schedule[4] = 0x80000000;
    schedule[15] = 0x00000280;

    uint32_t v[8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        v[i] = resume_state[i];
    v[0] += nonce_word;
    v[4] += nonce_word;

    sha256_rounds_rolling<true, 4, 64>(
        v,
        schedule,
        precomputed_schedule_words[0],
        precomputed_schedule_words[1],
        precomputed_schedule_words[2],
        precomputed_schedule_words[3]
    );

    #pragma unroll
    for (int i = 0; i < 8; i++)
        first_hash_words[i] = midstate[i] + v[i];
}

// Most significant 32 bits of the double hash, as the target comparison reads
// them, from 61 of the second hash's 64 rounds.
//
// Rounds 61-63 only shift e to f, g and then h, so the value of e after round
// 60 is h after round 63, and final word 7 is 0x5be0cd19 + that value. Word 7
// holds the last four hash bytes, which are the most significant bytes of the
// hash as a little-endian 256-bit number.
__device__ __forceinline__ uint32_t sha256_32_top_word(
    const uint32_t first_hash_words[8])
{
    uint32_t v[8] = {
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
    schedule[15] = 0x00000100;

    sha256_rounds_rolling<false, 0, 61>(v, schedule, 0, 0, 0, 0);

    uint32_t word = 0x5be0cd19 + v[4];
    return (word >> 24) |
           ((word >> 8) & 0x0000ff00) |
           ((word << 8) & 0x00ff0000) |
           (word << 24);
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

// Midstate of the header's first block with `variant` added to its BIP 320
// version field, and the nonce-independent state after round 3 of the first
// hash's second block; see sha256_80_resumed.
__device__ void header_block_state(
    const uint8_t *base_header,
    const uint32_t *tail_words,
    uint32_t variant,
    uint32_t *midstate,
    uint32_t *resume_state)
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

    // The version is the header's first four bytes, little-endian.
    uint32_t version =
        (message_words[0] >> 24) |
        ((message_words[0] >> 8) & 0x0000ff00) |
        ((message_words[0] << 8) & 0x00ff0000) |
        (message_words[0] << 24);
    version += variant << VERSION_ROLL_SHIFT;
    message_words[0] =
        (version >> 24) |
        ((version >> 8) & 0x0000ff00) |
        ((version << 8) & 0x00ff0000) |
        (version << 24);

    sha256_transform_words(state, message_words);

    #pragma unroll
    for (int i = 0; i < 8; i++)
        midstate[i] = state[i];

    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    #pragma unroll
    for (int t = 0; t < 3; t++) {
        uint32_t t1 = h + ep1(e) + ch(e, f, g) + K[t] + tail_words[t];
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

    uint32_t t1_without_nonce = h + ep1(e) + ch(e, f, g) + K[3];
    uint32_t t2 = ep0(a) + maj(a, b, c);

    resume_state[0] = t1_without_nonce + t2;
    resume_state[1] = a;
    resume_state[2] = b;
    resume_state[3] = c;
    resume_state[4] = d + t1_without_nonce;
    resume_state[5] = e;
    resume_state[6] = f;
    resume_state[7] = g;
}

__global__ void prepare_midstate_kernel(
    const uint8_t *base_header,
    const uint32_t *tail_words,
    uint32_t *midstate,
    uint32_t *precomputed_schedule_words)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    // The resume state is stored after the four schedule words.
    header_block_state(
        base_header,
        tail_words,
        0,
        midstate,
        precomputed_schedule_words + 4
    );

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

// One thread per version variant; variant i writes its midstate and then its
// resume state to version_states + 16 * i.
__global__ void prepare_versions_kernel(
    const uint8_t *base_header,
    const uint32_t *tail_words,
    uint32_t *version_states)
{
    if (blockIdx.x != 0 || threadIdx.x >= MAX_VERSIONS)
        return;

    uint32_t *states = version_states + 16 * threadIdx.x;
    header_block_state(base_header, tail_words, threadIdx.x, states, states + 8);
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
    // Per-scan inputs are read once per thread, not once per nonce.
    uint32_t tail[3];
    uint32_t mid[8];
    uint32_t target[8];
    uint32_t precomputed[4];
    uint32_t resume_state[8];

    #pragma unroll
    for (int i = 0; i < 3; i++)
        tail[i] = tail_words[i];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        mid[i] = midstate[i];
        target[i] = target_words[i];
        resume_state[i] = precomputed_schedule_words[4 + i];
    }
    #pragma unroll
    for (int i = 0; i < 4; i++)
        precomputed[i] = precomputed_schedule_words[i];

    const uint64_t first_index =
        ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) *
        MINER_NONCES_PER_THREAD;

    for (uint32_t i = 0; i < MINER_NONCES_PER_THREAD; i++) {
        const uint64_t nonce_index = first_index + i;
        if (nonce_index >= nonce_count)
            return;

        const uint32_t nonce = start_nonce + (uint32_t)nonce_index;
        // Header bytes encode the nonce little-endian; SHA-256 reads big-endian words.
        const uint32_t nonce_word =
            ((nonce & 0x000000ff) << 24) |
            ((nonce & 0x0000ff00) << 8) |
            ((nonce & 0x00ff0000) >> 8) |
            ((nonce & 0xff000000) >> 24);

        uint32_t first_hash_words[8];
#if MINER_RESUME_FIRST_HASH
        sha256_80_resumed(
            tail,
            nonce_word,
            mid,
            precomputed,
            resume_state,
            first_hash_words
        );
#else
        sha256_80_from_midstate_words(
            tail,
            nonce_word,
            mid,
            precomputed,
            first_hash_words
        );
#endif

#if MINER_EARLY_SECOND_HASH
        // target[0] is the target's most significant word, so this decides
        // every case except equality exactly as the full comparison would.
        const uint32_t top_word = sha256_32_top_word(first_hash_words);
        if (top_word > target[0])
            continue;
        if (top_word < target[0]) {
            atomicMin(found_nonce, nonce);
            continue;
        }
        // Equal most significant words: the lower words decide, so finish the
        // hash and run the unchanged full 256-bit comparison.
#endif
        uint32_t final_hash_words[8];
        sha256_32_from_words(first_hash_words, final_hash_words);

        if (hash_meets_target(final_hash_words, target))
            atomicMin(found_nonce, nonce);
    }
}

// Everything a version-rolling scan needs about one header. It is passed to
// the kernel by value, so threads read it as kernel parameters instead of
// loading it from global memory.
template <int Versions>
struct VersionWork {
    uint32_t tail[3];
    // W[16], W[17] and the nonce-independent parts of W[18] and W[19].
    uint32_t schedule[4];
    uint32_t target[8];
    // Per variant: the midstate, then the resume state.
    uint32_t state[Versions][16];
};

// Hashes `Versions` version variants of the header for each nonce.
//
// The variants differ only in the header's first block, that is in the
// midstate. The second block (Merkle-root tail, time, bits, nonce, padding)
// is identical, and the SHA-256 message schedule depends on the block alone,
// so W[16..63] of the first hash is expanded once per nonce and used by every
// variant. Each variant then runs its own rounds and its own second hash,
// with the same shortcuts and the same target comparison as mine_kernel.
//
// found receives the lowest (nonce << 32 | variant) that met the target and
// is left untouched otherwise.
template <int Versions>
__global__ void mine_versions_kernel(
    const VersionWork<Versions> work,
    uint32_t start_nonce,
    uint32_t nonce_count,
    unsigned long long *found)
{
    const uint64_t nonce_index =
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (nonce_index >= nonce_count)
        return;

    const uint32_t nonce = start_nonce + (uint32_t)nonce_index;
    // Header bytes encode the nonce little-endian; SHA-256 reads big-endian words.
    const uint32_t nonce_word =
        ((nonce & 0x000000ff) << 24) |
        ((nonce & 0x0000ff00) << 8) |
        ((nonce & 0x00ff0000) >> 8) |
        ((nonce & 0xff000000) >> 24);

    uint32_t w[64] = {0};

    #pragma unroll
    for (int i = 0; i < 3; i++)
        w[i] = work.tail[i];

    w[3] = nonce_word;
    w[4] = 0x80000000;
    // 80 bytes = 640 bits = 0x280
    w[15] = 0x00000280;
    w[16] = work.schedule[0];
    w[17] = work.schedule[1];
    w[18] = work.schedule[2] + sig0(w[3]);
    w[19] = work.schedule[3] + w[3];

    #pragma unroll
    for (int t = 20; t < 64; t++)
        w[t] = sig1(w[t - 2]) + w[t - 7] + sig0(w[t - 15]) + w[t - 16];

    #pragma unroll
    for (int variant = 0; variant < Versions; variant++) {
        uint32_t v[8];
        #pragma unroll
        for (int i = 0; i < 8; i++)
            v[i] = work.state[variant][8 + i];
        v[0] += nonce_word;
        v[4] += nonce_word;

        sha256_rounds_expanded<4, 64>(v, w);

        uint32_t first_hash_words[8];
        #pragma unroll
        for (int i = 0; i < 8; i++)
            first_hash_words[i] = work.state[variant][i] + v[i];

        const unsigned long long key =
            ((unsigned long long)nonce << 32) | (unsigned)variant;

#if MINER_EARLY_SECOND_HASH
        // target[0] is the target's most significant word, so this decides
        // every case except equality exactly as the full comparison would.
        const uint32_t top_word = sha256_32_top_word(first_hash_words);
        if (top_word > work.target[0])
            continue;
        if (top_word < work.target[0]) {
            atomicMin(found, key);
            continue;
        }
        // Equal most significant words: the lower words decide, so finish the
        // hash and run the unchanged full 256-bit comparison.
#endif
        uint32_t target[8];
        #pragma unroll
        for (int i = 0; i < 8; i++)
            target[i] = work.target[i];

        uint32_t final_hash_words[8];
        sha256_32_from_words(first_hash_words, final_hash_words);

        if (hash_meets_target(final_hash_words, target))
            atomicMin(found, key);
    }
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

struct DeviceBuffers {
    uint8_t *header = nullptr;
    uint32_t *tail_words = nullptr;
    uint32_t *target_words = nullptr;
    uint32_t *result = nullptr;
    uint32_t *midstate = nullptr;
    uint32_t *precomputed_schedule_words = nullptr;
    uint32_t *hash_words = nullptr;
    // Version rolling: 16 words per variant and the 64-bit scan result.
    uint32_t *version_states = nullptr;
    unsigned long long *version_result = nullptr;
};

// Host copy of what prepare_midstate_kernel and prepare_versions_kernel
// produced for the loaded header; the source of every VersionWork.
struct HostVersionWork {
    uint32_t tail[3];
    uint32_t schedule[4];
    uint32_t target[8];
    uint32_t state[MAX_VERSIONS][16];
};

bool cuda_ok(cudaError_t err, const char *what)
{
    if (err == cudaSuccess)
        return true;

    std::cerr
        << "CUDA " << what << " error: "
        << cudaGetErrorString(err)
        << "\n";
    return false;
}

// Allocated once per process and reused for every header and nonce range.
bool allocate_device_buffers(DeviceBuffers &device)
{
    return
        cuda_ok(cudaMalloc(&device.header, 80), "allocation") &&
        cuda_ok(cudaMalloc(&device.tail_words, 3 * sizeof(uint32_t)), "allocation") &&
        cuda_ok(cudaMalloc(&device.target_words, 8 * sizeof(uint32_t)), "allocation") &&
        cuda_ok(cudaMalloc(&device.result, sizeof(uint32_t)), "allocation") &&
        cuda_ok(cudaMalloc(&device.midstate, 8 * sizeof(uint32_t)), "allocation") &&
        // Four schedule words followed by the eight resume-state words.
        cuda_ok(cudaMalloc(&device.precomputed_schedule_words, 12 * sizeof(uint32_t)), "allocation") &&
        cuda_ok(cudaMalloc(&device.hash_words, 8 * sizeof(uint32_t)), "allocation") &&
        cuda_ok(cudaMalloc(&device.version_states, MAX_VERSIONS * 16 * sizeof(uint32_t)), "allocation") &&
        cuda_ok(cudaMalloc(&device.version_result, sizeof(unsigned long long)), "allocation");
}

void free_device_buffers(DeviceBuffers &device)
{
    cudaFree(device.header);
    cudaFree(device.tail_words);
    cudaFree(device.target_words);
    cudaFree(device.result);
    cudaFree(device.midstate);
    cudaFree(device.precomputed_schedule_words);
    cudaFree(device.hash_words);
    cudaFree(device.version_states);
    cudaFree(device.version_result);
    device = DeviceBuffers();
}

// Upload one header, its target and its midstate. Every nonce range of that
// header can then be scanned without repeating this work.
bool load_header(
    const DeviceBuffers &device,
    const uint8_t header[80],
    uint8_t target[32])
{
    uint32_t bits = read_le32(header + 72);

    if (!compact_bits_to_target(bits, target)) {
        std::cerr << "Invalid compact target in nBits.\n";
        return false;
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

    if (!cuda_ok(
            cudaMemcpy(device.header, header, 80, cudaMemcpyHostToDevice),
            "header upload") ||
        !cuda_ok(
            cudaMemcpy(
                device.tail_words,
                tail_words,
                sizeof(tail_words),
                cudaMemcpyHostToDevice
            ),
            "header upload") ||
        !cuda_ok(
            cudaMemcpy(
                device.target_words,
                target_words,
                sizeof(target_words),
                cudaMemcpyHostToDevice
            ),
            "target upload"))
        return false;

    prepare_midstate_kernel<<<1, 1>>>(
        device.header,
        device.tail_words,
        device.midstate,
        device.precomputed_schedule_words
    );

    return cuda_ok(cudaGetLastError(), "midstate kernel") &&
           cuda_ok(cudaDeviceSynchronize(), "midstate preparation");
}

// Scan [start, start + count) of the loaded header. found_nonce is 0xffffffff
// when no nonce met the target; otherwise found_hash_words holds its hash.
bool scan_range(
    const DeviceBuffers &device,
    uint32_t start,
    uint32_t count,
    uint32_t *found_nonce,
    uint32_t found_hash_words[8])
{
    uint32_t initial_result = 0xffffffff;

    if (!cuda_ok(
            cudaMemcpy(
                device.result,
                &initial_result,
                sizeof(uint32_t),
                cudaMemcpyHostToDevice
            ),
            "result reset"))
        return false;

    const int threads = MINER_THREADS_PER_BLOCK;
    const uint64_t nonces_per_block =
        (uint64_t)threads * MINER_NONCES_PER_THREAD;
    const unsigned blocks = static_cast<unsigned>(
        ((uint64_t)count + nonces_per_block - 1) / nonces_per_block
    );

    mine_kernel<<<blocks, threads>>>(
        device.tail_words,
        device.target_words,
        device.midstate,
        device.precomputed_schedule_words,
        start,
        count,
        device.result
    );

    if (!cuda_ok(cudaGetLastError(), "scan") ||
        !cuda_ok(cudaDeviceSynchronize(), "scan") ||
        !cuda_ok(
            cudaMemcpy(
                found_nonce,
                device.result,
                sizeof(uint32_t),
                cudaMemcpyDeviceToHost
            ),
            "result download"))
        return false;

    if (*found_nonce == 0xffffffff)
        return true;

    verify_nonce_kernel<<<1, 1>>>(
        device.tail_words,
        device.midstate,
        device.precomputed_schedule_words,
        *found_nonce,
        device.hash_words
    );

    return cuda_ok(cudaGetLastError(), "hash verification") &&
           cuda_ok(cudaDeviceSynchronize(), "hash verification") &&
           cuda_ok(
               cudaMemcpy(
                   found_hash_words,
                   device.hash_words,
                   8 * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost
               ),
               "hash download");
}

bool supported_version_count(uint32_t versions)
{
    return versions == 2 || versions == 4 || versions == 8 || versions == 16;
}

// Prepare every version variant of the header that load_header has just
// loaded, and copy what the scan kernel needs to the host.
bool load_versions(const DeviceBuffers &device, HostVersionWork &work)
{
    prepare_versions_kernel<<<1, MAX_VERSIONS>>>(
        device.header,
        device.tail_words,
        device.version_states
    );

    return cuda_ok(cudaGetLastError(), "version kernel") &&
           cuda_ok(
               cudaMemcpy(
                   work.tail,
                   device.tail_words,
                   sizeof(work.tail),
                   cudaMemcpyDeviceToHost
               ),
               "version download") &&
           cuda_ok(
               cudaMemcpy(
                   work.schedule,
                   device.precomputed_schedule_words,
                   sizeof(work.schedule),
                   cudaMemcpyDeviceToHost
               ),
               "version download") &&
           cuda_ok(
               cudaMemcpy(
                   work.target,
                   device.target_words,
                   sizeof(work.target),
                   cudaMemcpyDeviceToHost
               ),
               "version download") &&
           cuda_ok(
               cudaMemcpy(
                   work.state,
                   device.version_states,
                   sizeof(work.state),
                   cudaMemcpyDeviceToHost
               ),
               "version download");
}

template <int Versions>
cudaError_t launch_versions(
    const HostVersionWork &host,
    unsigned blocks,
    uint32_t start,
    uint32_t count,
    unsigned long long *found)
{
    VersionWork<Versions> work;
    std::memcpy(work.tail, host.tail, sizeof(work.tail));
    std::memcpy(work.schedule, host.schedule, sizeof(work.schedule));
    std::memcpy(work.target, host.target, sizeof(work.target));
    std::memcpy(work.state, host.state, sizeof(work.state));

    mine_versions_kernel<Versions><<<blocks, MINER_THREADS_PER_BLOCK>>>(
        work,
        start,
        count,
        found
    );
    return cudaGetLastError();
}

// Scan [start, start + count) of the first `versions` variants of the loaded
// header. *found tells whether a hash met the target; if so the lowest nonce,
// its lowest qualifying variant and that header's hash are returned.
bool scan_versions(
    const DeviceBuffers &device,
    const HostVersionWork &work,
    uint32_t versions,
    uint32_t start,
    uint32_t count,
    bool *found,
    uint32_t *found_nonce,
    uint32_t *found_variant,
    uint32_t found_hash_words[8])
{
    // No (nonce, variant) pair has this value, so unlike the 32-bit result of
    // scan_range it cannot be confused with nonce 0xffffffff.
    const unsigned long long no_result = ~0ULL;
    unsigned long long result = no_result;

    if (!cuda_ok(
            cudaMemcpy(
                device.version_result,
                &result,
                sizeof(result),
                cudaMemcpyHostToDevice
            ),
            "result reset"))
        return false;

    const unsigned blocks = static_cast<unsigned>(
        ((uint64_t)count + MINER_THREADS_PER_BLOCK - 1) /
        MINER_THREADS_PER_BLOCK
    );

    cudaError_t launched = cudaErrorInvalidValue;
    switch (versions) {
    case 2:
        launched = launch_versions<2>(work, blocks, start, count, device.version_result);
        break;
    case 4:
        launched = launch_versions<4>(work, blocks, start, count, device.version_result);
        break;
    case 8:
        launched = launch_versions<8>(work, blocks, start, count, device.version_result);
        break;
    case 16:
        launched = launch_versions<16>(work, blocks, start, count, device.version_result);
        break;
    }

    if (!cuda_ok(launched, "scan") ||
        !cuda_ok(cudaDeviceSynchronize(), "scan") ||
        !cuda_ok(
            cudaMemcpy(
                &result,
                device.version_result,
                sizeof(result),
                cudaMemcpyDeviceToHost
            ),
            "result download"))
        return false;

    *found = result != no_result;
    if (!*found)
        return true;

    *found_nonce = static_cast<uint32_t>(result >> 32);
    *found_variant = static_cast<uint32_t>(result);
    if (*found_variant >= versions) {
        std::cerr << "CUDA scan returned a version variant outside the request.\n";
        return false;
    }

    // The same independent full hash as scan_range, from this variant's midstate.
    verify_nonce_kernel<<<1, 1>>>(
        device.tail_words,
        device.version_states + 16 * *found_variant,
        device.precomputed_schedule_words,
        *found_nonce,
        device.hash_words
    );

    return cuda_ok(cudaGetLastError(), "hash verification") &&
           cuda_ok(cudaDeviceSynchronize(), "hash verification") &&
           cuda_ok(
               cudaMemcpy(
                   found_hash_words,
                   device.hash_words,
                   8 * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost
               ),
               "hash download");
}

void print_hash_words(const uint32_t hash_words[8])
{
    for (int i = 7; i >= 0; i--) {
        for (int byte = 0; byte < 4; byte++)
            std::printf(
                "%02x",
                static_cast<unsigned>(
                    (hash_words[i] >> (byte * 8)) & 0xff
                )
            );
    }
}

void print_scan_result(uint32_t found_nonce, const uint32_t found_hash_words[8])
{
    if (found_nonce == 0xffffffff) {
        std::printf("NONE\n");
    }
    else {
        std::printf("FOUND %u ", found_nonce);
        print_hash_words(found_hash_words);
        std::printf("\n");
    }
    std::fflush(stdout);
}

bool valid_scan_range(uint32_t start, uint32_t count)
{
    return count != 0 && (uint64_t)start + count <= 0x100000000ULL;
}

// Persistent mode. One request per stdin line:
//
//   SCAN <id> <80-byte-hex> <start> <count>
//   SCANV <id> <80-byte-hex> <start> <count> <versions>
//
// and exactly one stdout line in reply, echoing the request id:
//
//   <id> NONE
//   <id> FOUND <nonce> <hash>               (SCAN)
//   <id> FOUND <nonce> <hash> <variant>     (SCANV)
//
// SCANV scans the nonce range for each of <versions> version variants of the
// header (see "Version rolling" above) and reports the variant that was hit.
//
// End of input ends the session. Any malformed request or CUDA error is
// reported on stderr and ends the process with a nonzero status and no reply.
int serve(DeviceBuffers &device)
{
    std::printf("READY\n");
    std::fflush(stdout);

    uint8_t loaded_header[80];
    bool header_loaded = false;
    bool versions_loaded = false;
    HostVersionWork version_work;
    std::string line;

    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        std::istringstream fields(line);
        std::string command, id_text, header_hex, start_text, count_text;
        std::string versions_text, extra;
        fields >> command >> id_text >> header_hex >> start_text >> count_text;
        const bool rolling = command == "SCANV";
        if (rolling)
            fields >> versions_text;

        uint8_t header[80];
        uint32_t request_id = 0;
        uint32_t start = 0;
        uint32_t count = 0;
        uint32_t versions = 1;

        if ((command != "SCAN" && !rolling) ||
            (fields >> extra) ||
            !parse_uint32(id_text.c_str(), &request_id) ||
            !parse_header_hex(header_hex.c_str(), header) ||
            !parse_uint32(start_text.c_str(), &start) ||
            !parse_uint32(count_text.c_str(), &count) ||
            !valid_scan_range(start, count) ||
            (rolling &&
             (!parse_uint32(versions_text.c_str(), &versions) ||
              !supported_version_count(versions) ||
              (read_le32(header) & VERSION_ROLL_MASK) != 0))) {
            std::cerr << "Malformed request; expected: "
                      << "SCAN <id> <80-byte-hex> <start> <count> or "
                      << "SCANV <id> <80-byte-hex> <start> <count> <2|4|8|16> "
                      << "with version bits 13-28 clear\n";
            return 2;
        }

        // The nonce bytes are replaced on the GPU, so only the first 76
        // bytes identify the work already prepared on the device.
        if (!header_loaded || std::memcmp(header, loaded_header, 76) != 0) {
            uint8_t target[32];
            header_loaded = false;
            versions_loaded = false;
            if (!load_header(device, header, target))
                return 1;
            std::memcpy(loaded_header, header, 80);
            header_loaded = true;
        }

        if (rolling) {
            if (!versions_loaded) {
                if (!load_versions(device, version_work))
                    return 1;
                versions_loaded = true;
            }

            bool found;
            uint32_t found_nonce;
            uint32_t found_variant;
            uint32_t found_hash_words[8];
            if (!scan_versions(
                    device,
                    version_work,
                    versions,
                    start,
                    count,
                    &found,
                    &found_nonce,
                    &found_variant,
                    found_hash_words))
                return 1;

            if (found) {
                std::printf("%u FOUND %u ", request_id, found_nonce);
                print_hash_words(found_hash_words);
                std::printf(" %u\n", found_variant);
            }
            else {
                std::printf("%u NONE\n", request_id);
            }
            std::fflush(stdout);
            continue;
        }

        uint32_t found_nonce;
        uint32_t found_hash_words[8];
        if (!scan_range(device, start, count, &found_nonce, found_hash_words))
            return 1;

        std::printf("%u ", request_id);
        print_scan_result(found_nonce, found_hash_words);
    }

    return 0;
}

int main(int argc, char **argv)
{
    bool scan_mode = false;
    bool serve_mode = false;
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
            !valid_scan_range(scan_start, scan_count)) {
            std::cerr
                << "Usage: cuda_miner --scan-header <80-byte-hex> "
                << "--start <uint32> --count <1..uint32>\n";
            return 2;
        }
        scan_mode = true;
    }
    else if (argc == 2 && std::strcmp(argv[1], "--serve") == 0) {
        serve_mode = true;
    }
    else if (argc != 1) {
        std::cerr
            << "Usage: cuda_miner [--serve | --scan-header <80-byte-hex> "
            << "--start <uint32> --count <1..uint32>]\n";
        return 2;
    }

    // Sleep while a kernel runs instead of spinning a CPU core. Host-side
    // only; on a thermally limited laptop the idle core leaves the GPU more
    // power, which measured as higher sustained throughput.
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    DeviceBuffers device;
    if (!allocate_device_buffers(device))
        return 1;

    if (serve_mode) {
        int status = serve(device);
        free_device_buffers(device);
        return status;
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

    uint8_t target[32];
    if (!load_header(device, header, target))
        return 1;

    if (scan_mode) {
        uint32_t found_nonce;
        uint32_t found_hash_words[8];
        if (!scan_range(
                device,
                scan_start,
                scan_count,
                &found_nonce,
                found_hash_words))
            return 1;

        print_scan_result(found_nonce, found_hash_words);
        free_device_buffers(device);
        return 0;
    }

    uint32_t initial_result = 0xffffffff;

    cudaMemcpy(
        device.result,
        &initial_result,
        sizeof(uint32_t),
        cudaMemcpyHostToDevice
    );

    const uint32_t nonce_count = 500000000;
    const uint32_t warmup_nonce_count = 1000000;
    const int measured_runs = 5;

    const int threads = MINER_THREADS_PER_BLOCK;
    const uint64_t nonces_per_block =
        (uint64_t)threads * MINER_NONCES_PER_THREAD;
    const unsigned blocks = static_cast<unsigned>(
        ((uint64_t)nonce_count + nonces_per_block - 1) / nonces_per_block
    );
    const unsigned warmup_blocks = static_cast<unsigned>(
        ((uint64_t)warmup_nonce_count + nonces_per_block - 1) /
        nonces_per_block
    );

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    mine_kernel<<<warmup_blocks, threads>>>(
        device.tail_words,
        device.target_words,
        device.midstate,
        device.precomputed_schedule_words,
        0,
        warmup_nonce_count,
        device.result
    );

    cudaError_t err = cudaGetLastError();
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
            device.result,
            &initial_result,
            sizeof(uint32_t),
            cudaMemcpyHostToDevice
        );

        cudaEventRecord(start);
        mine_kernel<<<blocks, threads>>>(
            device.tail_words,
            device.target_words,
            device.midstate,
            device.precomputed_schedule_words,
            0,
            nonce_count,
            device.result
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
        device.result,
        sizeof(uint32_t),
        cudaMemcpyDeviceToHost
    );

    uint32_t found_hash_words[8];
    if (found_nonce != 0xffffffff) {
        verify_nonce_kernel<<<1, 1>>>(
            device.tail_words,
            device.midstate,
            device.precomputed_schedule_words,
            found_nonce,
            device.hash_words
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
            device.hash_words,
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

    free_device_buffers(device);

    return 0;
}
