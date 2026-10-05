import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path
from typing import Callable

from bitcoin import bits_to_target, double_sha256


DEFAULT_CORE_CLI = Path(
    r"C:\Program Files\Bitcoin\daemon\bitcoin-cli.exe"
)
DEFAULT_CUDA_MINER = Path(__file__).with_name("cuda_miner.exe")
NONCE_LIMIT = 1 << 32
WITNESS_COMMITMENT_PREFIX = bytes.fromhex("6a24aa21a9ed")
RAW_STRING_RPC_METHODS = {
    "getbestblockhash",
    "getblockhash",
    "getnewaddress",
    "getrawchangeaddress",
    "sendrawtransaction",
    "sendtoaddress",
}


class StaleTemplate(Exception):
    pass


def compact_size(value: int) -> bytes:
    if value < 0:
        raise ValueError("CompactSize cannot encode a negative value")
    if value < 0xFD:
        return bytes((value,))
    if value <= 0xFFFF:
        return b"\xfd" + struct.pack("<H", value)
    if value <= 0xFFFFFFFF:
        return b"\xfe" + struct.pack("<I", value)
    return b"\xff" + struct.pack("<Q", value)


def script_number(value: int) -> bytes:
    if value < 0:
        raise ValueError("Coinbase height must be nonnegative")
    if value == 0:
        return b"\x00"
    if value <= 16:
        return bytes((0x50 + value,))

    result = bytearray()
    while value:
        result.append(value & 0xFF)
        value >>= 8
    if result[-1] & 0x80:
        result.append(0)
    encoded = bytes(result)
    return compact_size(len(encoded)) + encoded


def create_coinbase(
    height: int,
    coinbase_flags: bytes,
    extra_nonce: int,
    coinbase_value: int,
    payout_script: bytes,
    witness_commitment_script: bytes | None,
) -> tuple[bytes, bytes]:
    script_sig = (
        script_number(height)
        + coinbase_flags
        + struct.pack("<Q", extra_nonce)
    )
    if not 2 <= len(script_sig) <= 100:
        raise ValueError("Constructed coinbase scriptSig is outside consensus limits")

    tx_input = (
        compact_size(1)
        + b"\x00" * 32
        + struct.pack("<I", 0xFFFFFFFF)
        + compact_size(len(script_sig))
        + script_sig
        + struct.pack("<I", 0xFFFFFFFF)
    )
    tx_outputs = (
        compact_size(1 + (witness_commitment_script is not None))
        + struct.pack("<Q", coinbase_value)
        + compact_size(len(payout_script))
        + payout_script
    )
    if witness_commitment_script is not None:
        tx_outputs += (
            struct.pack("<Q", 0)
            + compact_size(len(witness_commitment_script))
            + witness_commitment_script
        )

    stripped_tx = (
        struct.pack("<i", 2)
        + tx_input
        + tx_outputs
        + struct.pack("<I", 0)
    )
    if witness_commitment_script is None:
        return stripped_tx, double_sha256(stripped_tx)

    full_tx = (
        struct.pack("<i", 2)
        + b"\x00\x01"
        + tx_input
        + tx_outputs
        + compact_size(1)
        + compact_size(32)
        + b"\x00" * 32
        + struct.pack("<I", 0)
    )
    return full_tx, double_sha256(stripped_tx)


def merkle_root(raw_hashes: list[bytes]) -> bytes:
    if not raw_hashes:
        raise ValueError("Cannot compute a Merkle root without leaves")

    level = raw_hashes
    while len(level) > 1:
        if len(level) & 1:
            level = level + [level[-1]]
        level = [
            double_sha256(level[index] + level[index + 1])
            for index in range(0, len(level), 2)
        ]
    return level[0]


def template_transactions(template: dict) -> tuple[list[bytes], list[bytes]]:
    tx_data = []
    txids = []
    for index, transaction in enumerate(template.get("transactions", [])):
        if not isinstance(transaction, dict):
            raise ValueError(f"Template transaction {index} is not an object")
        try:
            raw_tx = bytes.fromhex(transaction["data"])
            txid = bytes.fromhex(transaction["txid"])[::-1]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Template transaction {index} is missing valid data/txid"
            ) from error
        if len(txid) != 32 or not raw_tx:
            raise ValueError(f"Template transaction {index} has invalid length")
        tx_data.append(raw_tx)
        txids.append(txid)
    return tx_data, txids


def get_witness_commitment(
    template: dict,
    transactions: list[dict],
) -> bytes | None:
    template_commitment = template.get("default_witness_commitment")
    if template_commitment is None:
        if any("hash" in transaction for transaction in transactions):
            raise ValueError("Segwit template is missing its witness commitment")
        return None

    try:
        commitment_script = bytes.fromhex(template_commitment)
    except (TypeError, ValueError) as error:
        raise ValueError("Template witness commitment is invalid hex") from error
    if (
        len(commitment_script) < len(WITNESS_COMMITMENT_PREFIX) + 32
        or not commitment_script.startswith(WITNESS_COMMITMENT_PREFIX)
    ):
        raise ValueError("Template witness commitment has an invalid script")

    witness_hashes = [b"\x00" * 32]
    for index, transaction in enumerate(transactions):
        try:
            wtxid = bytes.fromhex(transaction.get("hash", transaction["txid"]))[::-1]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Template transaction {index} has an invalid witness hash"
            ) from error
        if len(wtxid) != 32:
            raise ValueError(f"Template transaction {index} has an invalid witness hash")
        witness_hashes.append(wtxid)

    witness_root = merkle_root(witness_hashes)
    expected_script = (
        WITNESS_COMMITMENT_PREFIX
        + double_sha256(witness_root + b"\x00" * 32)
    )
    if commitment_script != expected_script:
        raise ValueError(
            "Template witness commitment does not match its transaction set"
        )
    return commitment_script


def build_header(template: dict, coinbase_txid: bytes, transaction_txids: list[bytes]) -> bytes:
    previous_hash = bytes.fromhex(template["previousblockhash"])[::-1]
    root = merkle_root([coinbase_txid, *transaction_txids])
    return (
        struct.pack("<I", template["version"])
        + previous_hash
        + root
        + struct.pack("<I", template["curtime"])
        + struct.pack("<I", int(template["bits"], 16))
        + b"\x00" * 4
    )


def validate_template_target(template: dict) -> tuple[int, int]:
    bits_text = template.get("bits")
    target_text = template.get("target")
    if (
        not isinstance(bits_text, str)
        or len(bits_text) != 8
        or any(character not in "0123456789abcdefABCDEF" for character in bits_text)
    ):
        raise RuntimeError("Block template bits must be exactly 8 hexadecimal characters")
    if (
        not isinstance(target_text, str)
        or len(target_text) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in target_text)
    ):
        raise RuntimeError(
            "Block template target must be exactly 64 hexadecimal characters"
        )

    bits = int(bits_text, 16)
    gbt_target = int(target_text, 16)
    decoded_target = bits_to_target(bits)
    if not 0 < decoded_target < (1 << 256):
        raise RuntimeError("Block template bits decode to an invalid 256-bit target")
    if decoded_target != gbt_target:
        raise RuntimeError(
            "Block template target mismatch: "
            f"bits={bits_text}, decoded_target={decoded_target:064x}, "
            f"gbt_target={target_text.lower()}"
        )
    return bits, decoded_target


def bitcoin_cli_command(
    cli: Path,
    network: str,
    datadir: Path | None,
    conf: Path | None,
    *args: str,
) -> list[str]:
    command = [str(cli)]
    if network == "regtest":
        command.append("-regtest")
    if datadir is not None:
        command.append(f"-datadir={datadir}")
    if conf is not None:
        command.append(f"-conf={conf}")
    command.extend(args)
    return command


def rpc(
    cli: Path,
    network: str,
    datadir: Path | None,
    conf: Path | None,
    method: str,
    *params: object,
) -> object:
    command = bitcoin_cli_command(
        cli,
        network,
        datadir,
        conf,
        method,
        *(
            param
            if isinstance(param, str)
            else json.dumps(param, separators=(",", ":"))
            for param in params
        ),
    )
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Bitcoin Core RPC {method} failed: {detail}")
    if not result.stdout.strip():
        if method == "submitblock":
            return None
        raise RuntimeError(f"Bitcoin Core RPC {method} returned no result")
    if method in RAW_STRING_RPC_METHODS or method == "submitblock":
        return result.stdout.strip()
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Bitcoin Core RPC {method} returned invalid JSON"
        ) from error


def mine_chunk(
    cuda_miner: Path,
    header: bytes,
    start: int,
    count: int,
) -> tuple[int, str] | None:
    result = subprocess.run(
        [
            str(cuda_miner),
            "--scan-header",
            header.hex(),
            "--start",
            str(start),
            "--count",
            str(count),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"CUDA miner failed: {detail}")

    output = result.stdout.strip()
    if output == "NONE":
        return None
    fields = output.split()
    if len(fields) != 3 or fields[0] != "FOUND":
        raise RuntimeError(f"Unexpected CUDA miner output: {output!r}")
    return int(fields[1]), fields[2]


def find_nonce(
    cuda_miner: Path,
    header: bytes,
    bits: int,
    chunk_size: int,
    stale_check: Callable[[], bool],
) -> tuple[int, bytes] | None:
    target = bits_to_target(bits)
    start = 0
    while start < NONCE_LIMIT:
        if stale_check():
            raise StaleTemplate
        count = min(chunk_size, NONCE_LIMIT - start)
        candidate = mine_chunk(cuda_miner, header, start, count)
        if candidate is not None:
            nonce, displayed_hash = candidate
            if not start <= nonce < start + count:
                raise RuntimeError("CUDA miner returned a nonce outside its assigned range")
            nonce_header = header[:76] + struct.pack("<I", nonce)
            actual_hash = double_sha256(nonce_header)
            if actual_hash[::-1].hex() != displayed_hash:
                raise RuntimeError("CUDA hash does not match the CPU SHA-256d result")
            if int.from_bytes(actual_hash[::-1], "big") > target:
                raise RuntimeError("CUDA returned a hash that does not meet the target")
            return nonce, actual_hash
        start += count

    # The CUDA result sentinel is 0xffffffff; check that final nonce on the CPU.
    last_header = header[:76] + struct.pack("<I", 0xFFFFFFFF)
    last_hash = double_sha256(last_header)
    if int.from_bytes(last_hash[::-1], "big") <= target:
        return 0xFFFFFFFF, last_hash
    return None


def mine_one_block(
    cli: Path,
    cuda_miner: Path,
    network: str,
    datadir: Path | None,
    conf: Path | None,
    chunk_size: int,
    extra_nonce: int,
    payout_script: bytes,
) -> bool:
    template = rpc(
        cli,
        network,
        datadir,
        conf,
        "getblocktemplate",
        {"rules": ["segwit"]},
    )
    if not isinstance(template, dict):
        raise RuntimeError("getblocktemplate returned an unexpected response")
    required = ("height", "version", "previousblockhash", "curtime", "bits")
    if any(key not in template for key in required):
        raise RuntimeError("Block template is missing required header fields")
    bits, target = validate_template_target(template)
    print(
        f"Template target cross-check: bits={template['bits']}, "
        f"decoded_target={target:064x}, gbt_target={template['target'].lower()}",
        flush=True,
    )

    coinbase_aux = template.get("coinbaseaux", {})
    flags_hex = coinbase_aux.get("flags", "") if isinstance(coinbase_aux, dict) else ""
    try:
        coinbase_flags = bytes.fromhex(flags_hex)
    except ValueError as error:
        raise RuntimeError("Template coinbase flags are not valid hexadecimal") from error

    transactions = template.get("transactions", [])
    if not isinstance(transactions, list):
        raise RuntimeError("Block template transactions field is invalid")
    try:
        witness_commitment = get_witness_commitment(template, transactions)
        coinbase, coinbase_txid = create_coinbase(
            int(template["height"]),
            coinbase_flags,
            extra_nonce,
            int(template["coinbasevalue"]),
            payout_script,
            witness_commitment,
        )
        transaction_data, transaction_txids = template_transactions(template)
        header = build_header(template, coinbase_txid, transaction_txids)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid block template: {error}") from error
    print(
        f"Mining {network} block {template['height']} "
        f"(bits={template['bits']}, chunk={chunk_size:,})",
        flush=True,
    )
    expected_previous = template["previousblockhash"]
    try:
        candidate = find_nonce(
            cuda_miner,
            header,
            bits,
            chunk_size,
            lambda: rpc(
                cli,
                network,
                datadir,
                conf,
                "getbestblockhash",
            ) != expected_previous,
        )
    except StaleTemplate:
        print("Template became stale; requesting new work.")
        return False
    if candidate is None:
        print(
            "Nonce space exhausted; changing the coinbase extranonce "
            "and retrying the template."
        )
        return False

    nonce, raw_hash = candidate
    mined_header = header[:76] + struct.pack("<I", nonce)
    block = (
        mined_header
        + compact_size(1 + len(transaction_data))
        + coinbase
        + b"".join(transaction_data)
    )
    block_hash = raw_hash[::-1].hex()
    print(f"Found nonce {nonce}; block hash {block_hash}", flush=True)

    if rpc(cli, network, datadir, conf, "getbestblockhash") != expected_previous:
        print("Template became stale before submission; requesting new work.")
        return False

    submission = rpc(cli, network, datadir, conf, "submitblock", block.hex())
    if submission is not None:
        raise RuntimeError(f"Bitcoin Core rejected the block: {submission}")

    height = rpc(cli, network, datadir, conf, "getblockcount")
    if not isinstance(height, int) or height < int(template["height"]):
        raise RuntimeError(
            "submitblock returned null but the regtest chain height did not advance"
        )
    print(f"Bitcoin Core accepted the block; current height is {height}.")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solo mine Bitcoin Core block templates with CUDA."
    )
    parser.add_argument(
        "--network",
        choices=("regtest", "mainnet"),
        default="regtest",
        help="Bitcoin Core network (default: regtest)",
    )
    parser.add_argument(
        "--bitcoin-cli",
        type=Path,
        default=DEFAULT_CORE_CLI,
        help="Path to Bitcoin Core bitcoin-cli.exe",
    )
    parser.add_argument(
        "--cuda-miner",
        type=Path,
        default=DEFAULT_CUDA_MINER,
        help="Path to the compiled cuda_miner.exe",
    )
    parser.add_argument(
        "--datadir",
        type=Path,
        help="Bitcoin Core datadir, if the node uses a non-default datadir",
    )
    parser.add_argument(
        "--bitcoin-conf",
        type=Path,
        help="Path to the bitcoin.conf used by Bitcoin Core",
    )
    parser.add_argument(
        "--payout-address",
        help="Required on mainnet; address receiving the block reward",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=1,
        help="Number of blocks to mine; use 0 to continue until interrupted",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500000000,
        help="Nonces per CUDA process invocation (default: 500000000)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.blocks < 0:
        print("--blocks must be nonnegative", file=sys.stderr)
        return 2
    if not 1 <= args.chunk_size <= 0xFFFFFFFF:
        print("--chunk-size must be from 1 through 4294967295", file=sys.stderr)
        return 2
    if not args.bitcoin_cli.is_file():
        print(f"bitcoin-cli.exe not found: {args.bitcoin_cli}", file=sys.stderr)
        return 2
    if not args.cuda_miner.is_file():
        print(f"CUDA miner executable not found: {args.cuda_miner}", file=sys.stderr)
        return 2
    if args.bitcoin_conf is not None and not args.bitcoin_conf.is_file():
        print(f"Bitcoin Core config file not found: {args.bitcoin_conf}", file=sys.stderr)
        return 2
    if args.network == "mainnet" and not args.payout_address:
        print("--payout-address is required for mainnet", file=sys.stderr)
        return 2
    if args.network == "regtest" and args.payout_address:
        print("--payout-address is only supported for mainnet", file=sys.stderr)
        return 2

    try:
        chain = rpc(
            args.bitcoin_cli,
            args.network,
            args.datadir,
            args.bitcoin_conf,
            "getblockchaininfo",
        )
        expected_chain = "main" if args.network == "mainnet" else "regtest"
        if not isinstance(chain, dict) or chain.get("chain") != expected_chain:
            raise RuntimeError(
                f"Connected Bitcoin Core node is not on {expected_chain}"
            )
        if (
            args.network == "mainnet"
            and chain.get("initialblockdownload") is True
        ):
            raise RuntimeError(
                "Bitcoin Core is still syncing; wait for initial block download to finish"
            )

        if args.network == "mainnet":
            address_info = rpc(
                args.bitcoin_cli,
                args.network,
                args.datadir,
                args.bitcoin_conf,
                "validateaddress",
                args.payout_address,
            )
            if (
                not isinstance(address_info, dict)
                or address_info.get("isvalid") is not True
                or not isinstance(address_info.get("scriptPubKey"), str)
            ):
                raise RuntimeError(
                    "Bitcoin Core rejected the mainnet payout address"
                )
            try:
                payout_script = bytes.fromhex(address_info["scriptPubKey"])
            except ValueError as error:
                raise RuntimeError(
                    "Bitcoin Core returned an invalid payout script"
                ) from error
        else:
            payout_script = b"\x51"

        mined = 0
        extra_nonce = 0
        while args.blocks == 0 or mined < args.blocks:
            if mine_one_block(
                args.bitcoin_cli,
                args.cuda_miner,
                args.network,
                args.datadir,
                args.bitcoin_conf,
                args.chunk_size,
                extra_nonce,
                payout_script,
            ):
                mined += 1
                extra_nonce = 0
            else:
                extra_nonce += 1
        return 0
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 130
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
