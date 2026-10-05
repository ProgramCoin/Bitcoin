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
MAX_MONEY = 21_000_000 * 100_000_000
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


def read_compact_size(data: bytes, offset: int) -> tuple[int, int]:
    if offset >= len(data):
        raise ValueError("Truncated CompactSize")
    prefix = data[offset]
    offset += 1
    if prefix < 0xFD:
        return prefix, offset

    width = {0xFD: 2, 0xFE: 4, 0xFF: 8}[prefix]
    if offset + width > len(data):
        raise ValueError("Truncated CompactSize value")
    value = int.from_bytes(data[offset : offset + width], "little")
    minimum = {2: 0xFD, 4: 0x10000, 8: 0x100000000}[width]
    if value < minimum:
        raise ValueError("Non-canonical CompactSize")
    return value, offset + width


def require_transaction_bytes(
    data: bytes,
    offset: int,
    length: int,
    field: str,
) -> int:
    if length < 0 or offset + length > len(data):
        raise ValueError(f"Truncated transaction {field}")
    return offset + length


def transaction_hashes(raw_transaction: bytes) -> tuple[bytes, bytes]:
    if len(raw_transaction) < 10:
        raise ValueError("Transaction is too short")

    offset = 4
    has_witness = False
    if raw_transaction[offset] == 0 and raw_transaction[offset + 1] != 0:
        if raw_transaction[offset + 1] != 1:
            raise ValueError("Transaction uses unsupported witness flags")
        has_witness = True
        offset += 2

    inputs_start = offset
    input_count, offset = read_compact_size(raw_transaction, offset)
    if input_count == 0 or input_count > len(raw_transaction) - offset:
        raise ValueError("Transaction has an invalid input count")
    for _ in range(input_count):
        offset = require_transaction_bytes(
            raw_transaction, offset, 36, "input outpoint"
        )
        script_length, offset = read_compact_size(raw_transaction, offset)
        offset = require_transaction_bytes(
            raw_transaction, offset, script_length, "input script"
        )
        offset = require_transaction_bytes(
            raw_transaction, offset, 4, "input sequence"
        )

    output_count, offset = read_compact_size(raw_transaction, offset)
    if output_count == 0 or output_count > (len(raw_transaction) - offset) // 9:
        raise ValueError("Transaction has an invalid output count")
    for _ in range(output_count):
        offset = require_transaction_bytes(
            raw_transaction, offset, 8, "output value"
        )
        script_length, offset = read_compact_size(raw_transaction, offset)
        offset = require_transaction_bytes(
            raw_transaction, offset, script_length, "output script"
        )

    outputs_end = offset
    has_witness_data = False
    if has_witness:
        for _ in range(input_count):
            item_count, offset = read_compact_size(raw_transaction, offset)
            if item_count > len(raw_transaction) - offset:
                raise ValueError("Transaction has an invalid witness item count")
            has_witness_data = has_witness_data or item_count > 0
            for _ in range(item_count):
                item_length, offset = read_compact_size(raw_transaction, offset)
                offset = require_transaction_bytes(
                    raw_transaction, offset, item_length, "witness item"
                )
        if not has_witness_data:
            raise ValueError("Transaction has a superfluous witness record")

    offset = require_transaction_bytes(raw_transaction, offset, 4, "locktime")
    if offset != len(raw_transaction):
        raise ValueError("Transaction has trailing data")

    stripped_transaction = (
        raw_transaction[:4]
        + raw_transaction[inputs_start:outputs_end]
        + raw_transaction[offset - 4 : offset]
    )
    return (
        double_sha256(stripped_transaction),
        double_sha256(raw_transaction),
    )


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
            expected_txid = bytes.fromhex(transaction["txid"])
            expected_wtxid = bytes.fromhex(transaction["hash"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Template transaction {index} is missing valid data/txid/hash"
            ) from error
        if len(expected_txid) != 32 or len(expected_wtxid) != 32 or not raw_tx:
            raise ValueError(f"Template transaction {index} has invalid length")
        try:
            txid, wtxid = transaction_hashes(raw_tx)
        except ValueError as error:
            raise ValueError(
                f"Template transaction {index} is malformed: {error}"
            ) from error
        if txid[::-1] != expected_txid:
            raise ValueError(
                f"Template transaction {index} txid does not match its data"
            )
        if wtxid[::-1] != expected_wtxid:
            raise ValueError(
                f"Template transaction {index} witness hash does not match its data"
            )
        tx_data.append(raw_tx)
        txids.append(txid)
    return tx_data, txids


def get_witness_commitment(
    template: dict,
    transactions: list[dict],
) -> bytes | None:
    template_commitment = template.get("default_witness_commitment")
    if template_commitment is None:
        if any(transaction_has_witness(transaction) for transaction in transactions):
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
            wtxid = bytes.fromhex(transaction["hash"])[::-1]
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


def serialize_block(
    header: bytes,
    coinbase: bytes,
    transactions: list[bytes],
) -> bytes:
    if len(header) != 80:
        raise ValueError(f"Bitcoin block header must be 80 bytes, got {len(header)}")
    return (
        header
        + compact_size(1 + len(transactions))
        + coinbase
        + b"".join(transactions)
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


def template_uint(template: dict, field: str, maximum: int) -> int:
    value = template.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= maximum
    ):
        raise RuntimeError(
            f"Block template {field} must be an integer from 0 through {maximum}"
        )
    return value


def parse_nonce_range(value: object) -> tuple[int, int]:
    if (
        not isinstance(value, str)
        or len(value) != 16
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise RuntimeError(
            "Block template noncerange must be exactly 16 hexadecimal characters"
        )
    lower = int(value[:8], 16)
    upper = int(value[8:], 16)
    if lower > upper:
        raise RuntimeError(
            "Block template noncerange has its lower bound above its upper bound"
        )
    # Bitcoin Core's default range spans both endpoints of the uint32 nonce.
    return lower, upper


def transaction_has_witness(transaction: dict) -> bool:
    raw_transaction = bytes.fromhex(transaction["data"])
    return (
        len(raw_transaction) >= 6
        and raw_transaction[4] == 0
        and raw_transaction[5] != 0
    )


def validate_template(
    template: dict,
) -> tuple[int, int, int, int]:
    required_fields = (
        "version",
        "previousblockhash",
        "height",
        "bits",
        "target",
        "curtime",
        "mintime",
        "noncerange",
        "mutable",
        "transactions",
        "coinbasevalue",
    )
    missing = [field for field in required_fields if field not in template]
    if missing:
        raise RuntimeError(
            "Block template is missing required fields: " + ", ".join(missing)
        )

    template_uint(template, "version", 0xFFFFFFFF)
    template_uint(template, "height", 0x7FFFFFFF)
    template_uint(template, "coinbasevalue", MAX_MONEY)
    curtime = template_uint(template, "curtime", 0xFFFFFFFF)
    mintime = template_uint(template, "mintime", 0xFFFFFFFF)
    if curtime < mintime:
        raise RuntimeError(
            f"Block template curtime {curtime} is earlier than mintime {mintime}"
        )

    previous_hash = template["previousblockhash"]
    if (
        not isinstance(previous_hash, str)
        or len(previous_hash) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in previous_hash)
    ):
        raise RuntimeError(
            "Block template previousblockhash must be exactly 64 hexadecimal characters"
        )

    bits, target = validate_template_target(template)
    nonce_start, nonce_end = parse_nonce_range(template["noncerange"])

    mutable = template["mutable"]
    if not isinstance(mutable, list) or any(
        not isinstance(item, str) for item in mutable
    ):
        raise RuntimeError("Block template mutable must be an array of strings")

    transactions = template["transactions"]
    if not isinstance(transactions, list):
        raise RuntimeError("Block template transactions field must be an array")
    for index, transaction in enumerate(transactions):
        if not isinstance(transaction, dict):
            raise RuntimeError(f"Template transaction {index} must be an object")
        try:
            raw_data = transaction["data"]
            txid = transaction["txid"]
        except KeyError as error:
            raise RuntimeError(
                f"Template transaction {index} is missing {error.args[0]}"
            ) from error
        if (
            not isinstance(raw_data, str)
            or not raw_data
            or len(raw_data) % 2
            or any(character not in "0123456789abcdefABCDEF" for character in raw_data)
        ):
            raise RuntimeError(
                f"Template transaction {index} data must be nonempty hexadecimal"
            )
        if (
            not isinstance(txid, str)
            or len(txid) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in txid)
        ):
            raise RuntimeError(
                f"Template transaction {index} txid must be 64 hexadecimal characters"
            )
        try:
            witness_txid = transaction["hash"]
        except KeyError as error:
            raise RuntimeError(
                f"Template transaction {index} is missing hash"
            ) from error
        if (
            not isinstance(witness_txid, str)
            or len(witness_txid) != 64
            or any(
                character not in "0123456789abcdefABCDEF"
                for character in witness_txid
            )
        ):
            raise RuntimeError(
                f"Template transaction {index} hash must be 64 hexadecimal characters"
            )

    commitment = template.get("default_witness_commitment")
    if commitment is not None and (
        not isinstance(commitment, str)
        or len(commitment) % 2
        or any(character not in "0123456789abcdefABCDEF" for character in commitment)
    ):
        raise RuntimeError(
            "Block template default_witness_commitment must be hexadecimal"
        )
    if commitment is None and any(transaction_has_witness(tx) for tx in transactions):
        raise RuntimeError("Segwit template is missing its witness commitment")

    return bits, target, nonce_start, nonce_end


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
    nonce_start: int,
    nonce_end: int,
    stale_check: Callable[[], bool],
) -> tuple[int, bytes] | None:
    target = bits_to_target(bits)
    start = nonce_start
    while start <= nonce_end:
        if stale_check():
            raise StaleTemplate
        count = min(chunk_size, nonce_end - start + 1)
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

    # CUDA uses 0xffffffff as its no-result sentinel, so verify that nonce on
    # the CPU when it is included in the template's permitted range.
    if nonce_start <= 0xFFFFFFFF <= nonce_end:
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
    bits, target, nonce_start, nonce_end = validate_template(template)
    print(
        f"Template target cross-check: bits={template['bits']}, "
        f"decoded_target={target:064x}, gbt_target={template['target'].lower()}",
        flush=True,
    )
    print(
        f"Template nonce range: {nonce_start:08x}..{nonce_end:08x}; "
        f"time={template['curtime']} (minimum={template['mintime']})",
        flush=True,
    )

    coinbase_aux = template.get("coinbaseaux", {})
    flags_hex = coinbase_aux.get("flags", "") if isinstance(coinbase_aux, dict) else ""
    try:
        coinbase_flags = bytes.fromhex(flags_hex)
    except ValueError as error:
        raise RuntimeError("Template coinbase flags are not valid hexadecimal") from error

    transactions = template["transactions"]
    try:
        witness_commitment = get_witness_commitment(template, transactions)
        coinbase, coinbase_txid = create_coinbase(
            int(template["height"]),
            coinbase_flags,
            extra_nonce,
            template["coinbasevalue"],
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
            nonce_start,
            nonce_end,
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
    block = serialize_block(mined_header, coinbase, transaction_data)
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
