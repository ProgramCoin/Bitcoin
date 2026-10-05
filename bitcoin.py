import hashlib
import struct


# Perform Bitcoin's double SHA-256.
def double_sha256(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


# Convert Bitcoin's compact nBits representation into a full target.
def bits_to_target(bits: int) -> int:
    exponent = bits >> 24
    mantissa = bits & 0xFFFFFF

    if exponent <= 3:
        return mantissa >> (8 * (3 - exponent))

    return mantissa << (8 * (exponent - 3))


# Convert a full target back into compact nBits.
def target_to_bits(target: int) -> int:
    size = (target.bit_length() + 7) // 8

    if size <= 3:
        compact = target << (8 * (3 - size))
    else:
        compact = target >> (8 * (size - 3))

    compact &= 0xFFFFFF

    if compact & 0x00800000:
        compact >>= 8
        size += 1

    return (size << 24) | compact


# Build the complete 80-byte Bitcoin block header.
def build_header(
    version: int,
    prev_hash: str,
    merkle_root: str,
    timestamp: int,
    bits: int,
    nonce: int,
) -> bytes:

    header = (
        struct.pack("<I", version)
        + bytes.fromhex(prev_hash)[::-1]
        + bytes.fromhex(merkle_root)[::-1]
        + struct.pack("<I", timestamp)
        + struct.pack("<I", bits)
        + struct.pack("<I", nonce)
    )

    if len(header) != 80:
        raise ValueError(
            f"Bitcoin block header must be 80 bytes, got {len(header)}"
        )

    return header


# Prepare the portion of the header that never changes during nonce scanning.
#
# The first 64 bytes can be processed once by SHA-256.
# The remaining 16 bytes consist of:
#
# timestamp  = 4 bytes
# bits       = 4 bytes
# nonce      = 4 bytes
# padding    = handled automatically by hashlib
def prepare_mining_header(
    version: int,
    prev_hash: str,
    merkle_root: str,
    timestamp: int,
    bits: int,
):
    fixed_header = (
        struct.pack("<I", version)
        + bytes.fromhex(prev_hash)[::-1]
        + bytes.fromhex(merkle_root)[::-1]
        + struct.pack("<I", timestamp)
        + struct.pack("<I", bits)
    )

    if len(fixed_header) != 76:
        raise ValueError(
            f"Fixed header must be 76 bytes, got {len(fixed_header)}"
        )

    # SHA-256 processes the first 64 bytes only once.
    first_chunk = fixed_header[:64]

    # These are the first 12 bytes of the second SHA-256 chunk.
    second_chunk_prefix = fixed_header[64:76]

    # Prepare the SHA-256 state after processing the first 64 bytes.
    sha256_midstate = hashlib.sha256(first_chunk)

    return sha256_midstate, second_chunk_prefix


# Hash one nonce using the precomputed SHA-256 state.
def hash_nonce(
    sha256_midstate,
    second_chunk_prefix: bytes,
    nonce: int,
) -> bytes:

    nonce_bytes = struct.pack("<I", nonce)

    # Copy the prepared SHA-256 state instead of hashing
    # the first 64 bytes again.
    first_hash = sha256_midstate.copy()

    first_hash.update(second_chunk_prefix)
    first_hash.update(nonce_bytes)

    digest = first_hash.digest()

    # Bitcoin uses double SHA-256.
    return hashlib.sha256(digest).digest()


# Convert a raw Bitcoin hash into the conventional displayed form.
def display_hash(raw_hash: bytes) -> str:
    return raw_hash[::-1].hex()


# Determine whether a hash satisfies the target.
def hash_meets_target(raw_hash: bytes, target: int) -> bool:
    hash_value = int.from_bytes(
        raw_hash[::-1],
        byteorder="big",
    )

    return hash_value <= target