import argparse
import time

from bitcoin import (
    prepare_mining_header,
    hash_nonce,
    display_hash,
    bits_to_target,
    hash_meets_target,
)


# Mine a range of Bitcoin nonces.
def mine(
    version: int,
    prev_hash: str,
    merkle_root: str,
    timestamp: int,
    bits: int,
    start_nonce: int = 0,
    end_nonce: int = 0xFFFFFFFF,
    report_interval: float = 2.0,
):

    target = bits_to_target(bits)

    print("=" * 70)
    print("Bitcoin Educational Miner - Optimized CPU")
    print("=" * 70)

    print(f"Target : {target:064x}")
    print(f"Nonce  : {start_nonce:,} -> {end_nonce:,}")
    print()

    # Prepare the invariant portion of the block header.
    sha256_midstate, second_chunk_prefix = prepare_mining_header(
        version=version,
        prev_hash=prev_hash,
        merkle_root=merkle_root,
        timestamp=timestamp,
        bits=bits,
    )

    nonce = start_nonce
    hashes = 0

    start_time = time.perf_counter()
    last_report = start_time

    while nonce <= end_nonce:

        # Hash this nonce using the precomputed SHA-256 state.
        raw_hash = hash_nonce(
            sha256_midstate,
            second_chunk_prefix,
            nonce,
        )

        hashes += 1

        # Check whether the hash satisfies the target.
        if hash_meets_target(raw_hash, target):

            elapsed = time.perf_counter() - start_time

            print()
            print()
            print("🎉 VALID HASH FOUND")
            print("=" * 70)

            print(f"Nonce:       {nonce}")
            print(f"Hash:        {display_hash(raw_hash)}")
            print(f"Target:      {target:064x}")
            print(f"Hashes:      {hashes:,}")
            print(f"Time:        {elapsed:.4f} seconds")

            if elapsed > 0:
                print(
                    f"Hashrate:    "
                    f"{hashes / elapsed:,.2f} H/s"
                )

            print("=" * 70)

            return nonce, display_hash(raw_hash)

        nonce += 1

        # Periodically display mining statistics.
        now = time.perf_counter()

        if now - last_report >= report_interval:

            elapsed = now - start_time

            hashrate = (
                hashes / elapsed
                if elapsed > 0
                else 0
            )

            print(
                f"\rNonce: {nonce:,} | "
                f"Hashes: {hashes:,} | "
                f"Hashrate: {hashrate:,.0f} H/s",
                end="",
                flush=True,
            )

            last_report = now

    elapsed = time.perf_counter() - start_time

    print()
    print()
    print("Nonce range exhausted.")
    print(f"Hashes tested: {hashes:,}")

    if elapsed > 0:
        print(
            f"Average hashrate: "
            f"{hashes / elapsed:,.2f} H/s"
        )

    return None


# Allow decimal and hexadecimal CLI values.
def parse_int(value: str) -> int:
    return int(value, 0)


def main():

    parser = argparse.ArgumentParser(
        description="Optimized educational Bitcoin CPU miner"
    )

    parser.add_argument(
        "--version",
        type=parse_int,
        default=0x20000000,
        help="Block version",
    )

    parser.add_argument(
        "--prev-hash",
        required=True,
        help="Previous block hash (64 hexadecimal characters)",
    )

    parser.add_argument(
        "--merkle-root",
        required=True,
        help="Merkle root (64 hexadecimal characters)",
    )

    parser.add_argument(
        "--timestamp",
        type=int,
        required=True,
        help="Block timestamp",
    )

    parser.add_argument(
        "--bits",
        type=parse_int,
        required=True,
        help="Compact Bitcoin difficulty (nBits)",
    )

    parser.add_argument(
        "--start",
        type=parse_int,
        default=0,
        help="Starting nonce",
    )

    parser.add_argument(
        "--end",
        type=parse_int,
        default=0xFFFFFFFF,
        help="Ending nonce",
    )

    args = parser.parse_args()

    # Validate hashes before starting.
    if len(args.prev_hash) != 64:
        parser.error(
            "Previous hash must contain exactly 64 hexadecimal characters"
        )

    if len(args.merkle_root) != 64:
        parser.error(
            "Merkle root must contain exactly 64 hexadecimal characters"
        )

    try:
        bytes.fromhex(args.prev_hash)
        bytes.fromhex(args.merkle_root)
    except ValueError:
        parser.error(
            "Previous hash and Merkle root must contain only hexadecimal characters"
        )

    mine(
        version=args.version,
        prev_hash=args.prev_hash,
        merkle_root=args.merkle_root,
        timestamp=args.timestamp,
        bits=args.bits,
        start_nonce=args.start,
        end_nonce=args.end,
    )


if __name__ == "__main__":
    main()