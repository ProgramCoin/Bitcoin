import argparse
import ctypes
import json
import math
import os
import re
import socket
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NoReturn

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
# A serialized block is far larger than the Windows command-line limit, so
# these methods receive their parameters through bitcoin-cli -stdin.
STDIN_RPC_METHODS = {"submitblock"}
REGTEST_PAYOUT_SCRIPT = b"\x51"
CUDA_SHUTDOWN_TIMEOUT = 10.0
# A tip check that has not answered in this long counts as a failed check
# instead of leaving the miner waiting with an idle GPU.
TIP_CHECK_TIMEOUT = 10.0
# Consecutive failed tip checks tolerated on mainnet before new GPU work stops.
# One failed bitcoin-cli call is treated as transient.
TIP_CHECK_FAILURE_LIMIT = 3
# Consecutive failed health polls before the monitor asks mining to pause.
MONITOR_FAILURES_TO_PAUSE = 2
# Waits between recovery attempts after mining paused; the last one repeats
# until --recovery-timeout is reached.
RECOVERY_DELAYS = (5.0, 10.0, 20.0, 40.0, 60.0)
# A tip confirmed this recently lets the first chunk of follow-on work start
# before its own tip check instead of after it.
TIP_CONFIRMATION_SECONDS = 2.0
# A template older than this is replaced at the next nonce-space rollover;
# until then exhausted nonce space only rolls the coinbase extranonce.
TEMPLATE_REFRESH_SECONDS = 30.0
STATUS_INTERVAL = 5.0
SUBMIT_TIMEOUT = 120.0
# Waits before each further submitblock attempt after one failed without
# Core reporting the block as known.
SUBMIT_RETRY_DELAYS = (1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0, 60.0, 60.0, 60.0)
# submitblock results meaning Core stored a block that is not on its best
# chain. Core has not fully validated such a block and does not relay it.
SIDE_CHAIN_SUBMISSION_RESULTS = {"inconclusive", "duplicate-inconclusive"}
# A coinbase output is spendable by the wallet at this many confirmations.
COINBASE_MATURITY_CONFIRMATIONS = 101
BLOCK_MONITOR_POLL_SECONDS = 60.0
HEX_PATTERN = re.compile(r"[0-9a-fA-F]+\Z")


class StaleTemplate(Exception):
    pass


class RpcError(RuntimeError):
    """bitcoin-cli did not deliver a usable answer (as opposed to Core's answer being bad)."""


class MiningPaused(Exception):
    """Connectivity loss was confirmed between chunks; no GPU work is in flight."""


def keep_system_awake(enable: bool) -> None:
    """Ask Windows not to sleep on idle while mainnet work is running.

    GPU load does not count as activity, so an unattended machine would
    otherwise suspend in the middle of mining. No power setting is changed:
    the request belongs to this thread and ends with it. The display may
    still turn off, and closing the lid or sleeping by hand still sleeps.
    """
    if sys.platform != "win32":
        return
    es_continuous, es_system_required = 0x80000000, 0x00000001
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            es_continuous | (es_system_required if enable else 0)
        )
    except (AttributeError, OSError):
        pass


def socks5_ready(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout) as connection:
            connection.settimeout(timeout)
            connection.sendall(b"\x05\x01\x00")
            response = bytearray()
            while len(response) < 2:
                chunk = connection.recv(2 - len(response))
                if not chunk:
                    return False
                response.extend(chunk)
            return response == b"\x05\x00"
    except OSError:
        return False


def ensure_tor_ready(
    host: str,
    port: int,
    executable: Path | None,
    timeout: float,
    poll_interval: float,
) -> subprocess.Popen[bytes] | None:
    if socks5_ready(host, port, min(1.0, timeout)):
        print(f"[TOR] SOCKS5 proxy already available at {host}:{port}.")
        print("[TOR] Tor ready.")
        return None

    print(f"[TOR] Tor not detected on {host}:{port}.")
    if executable is None:
        raise RuntimeError(
            "Tor SOCKS5 is unavailable and no Tor executable was configured; "
            "use --tor-executable to allow automatic startup; CUDA mining was prevented"
        )
    if not executable.is_file():
        raise RuntimeError(f"Configured Tor executable not found: {executable}")

    print("[TOR] Starting Tor...")
    endpoint = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        owned_process = subprocess.Popen(
            [str(executable), "--SocksPort", endpoint],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
    except OSError as error:
        raise RuntimeError(f"Could not start configured Tor executable: {error}") from error

    deadline = time.monotonic() + timeout
    print("[TOR] Waiting for SOCKS5 proxy...")
    try:
        while time.monotonic() < deadline:
            exit_code = owned_process.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"Tor exited with status {exit_code} before its SOCKS5 "
                    "proxy became ready"
                )
            if socks5_ready(host, port, min(1.0, poll_interval)):
                print("[TOR] SOCKS5 proxy detected.")
                print("[TOR] Tor ready.")
                return owned_process
            time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))

        raise RuntimeError(
            f"Tor SOCKS5 proxy did not become ready at {host}:{port} "
            f"within {timeout:g} seconds; CUDA mining was prevented"
        )
    except (RuntimeError, KeyboardInterrupt):
        stop_owned_tor(owned_process)
        raise


def stop_owned_tor(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        process.terminate()
    except OSError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except OSError:
            return
        process.wait()


def preflight_mainnet(
    cli: Path,
    datadir: Path | None,
    conf: Path | None,
    tor_host: str,
    tor_port: int,
    tor_executable: Path | None,
    tor_timeout: float,
    tor_poll_interval: float,
    require_onion_peers: bool,
    min_peers: int = 1,
) -> subprocess.Popen[bytes] | None:
    owned_tor = ensure_tor_ready(
        tor_host,
        tor_port,
        tor_executable,
        tor_timeout,
        tor_poll_interval,
    )
    try:
        chain = rpc(cli, "mainnet", datadir, conf, "getblockchaininfo")
        if not isinstance(chain, dict) or chain.get("chain") != "main":
            raise RuntimeError("Bitcoin Core RPC did not confirm the main chain")
        if chain.get("initialblockdownload") is not False:
            raise RuntimeError(
                "Bitcoin Core is in initial block download or did not report "
                "initialblockdownload=false; CUDA mining was prevented"
            )

        blocks = chain.get("blocks")
        headers = chain.get("headers")
        if (
            isinstance(blocks, bool)
            or not isinstance(blocks, int)
            or isinstance(headers, bool)
            or not isinstance(headers, int)
            or blocks != headers
        ):
            raise RuntimeError(
                "Bitcoin Core is not synchronized to its known header tip "
                f"(blocks={blocks!r}, headers={headers!r}); CUDA mining was prevented"
            )

        network_info = rpc(cli, "mainnet", datadir, conf, "getnetworkinfo")
        if (
            not isinstance(network_info, dict)
            or network_info.get("networkactive") is not True
        ):
            raise RuntimeError(
                "Bitcoin Core networking is inactive or not confirmed active; "
                "CUDA mining was prevented"
            )
        connections = network_info.get("connections")
        if (
            isinstance(connections, bool)
            or not isinstance(connections, int)
            or connections < 1
        ):
            raise RuntimeError(
                "Bitcoin Core has no connected peers; CUDA mining was prevented"
            )
        if connections < min_peers:
            raise RuntimeError(
                f"Bitcoin Core has {connections} connected peer(s) but "
                f"--min-peers requires {min_peers}; CUDA mining was prevented"
            )
        if require_onion_peers:
            peers = rpc(cli, "mainnet", datadir, conf, "getpeerinfo")
            if not isinstance(peers, list) or not any(
                isinstance(peer, dict)
                and (
                    peer.get("network") == "onion"
                    or (
                        isinstance(peer.get("addr"), str)
                        and ".onion" in peer["addr"].lower()
                    )
                )
                for peer in peers
            ):
                raise RuntimeError(
                    "Tor-only operation was required, but Bitcoin Core has no "
                    "connected onion peer; CUDA mining was prevented"
                )

        best_block = chain.get("bestblockhash")
        if (
            not isinstance(best_block, str)
            or len(best_block) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in best_block)
        ):
            raise RuntimeError(
                "Bitcoin Core did not report a valid best block hash; "
                "CUDA mining was prevented"
            )

        print(
            f"[CORE] Connected to mainnet at height {blocks}; "
            f"{connections} peer(s), headers synchronized."
        )
        print(f"[CORE] Chain: {chain['chain']}")
        print(
            f"[CORE] Synchronization: blocks={blocks}, headers={headers}, "
            "initialblockdownload=false"
        )
        print(f"[CORE] Peers: {connections}")
        print(f"[CORE] Height: {blocks}")
        print(f"[CORE] Best block: {best_block}")
        if require_onion_peers:
            print("[CORE] At least one onion peer is connected.")
        return owned_tor
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        subprocess.SubprocessError,
        KeyboardInterrupt,
    ):
        stop_owned_tor(owned_tor)
        raise


def resolve_payout_script(
    cli: Path,
    datadir: Path | None,
    conf: Path | None,
    payout_address: str,
    expected_script: bytes | None,
) -> bytes:
    address_info = rpc(
        cli,
        "mainnet",
        datadir,
        conf,
        "validateaddress",
        payout_address,
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
    if not payout_script:
        raise RuntimeError("Bitcoin Core returned an empty payout script")
    if payout_script == REGTEST_PAYOUT_SCRIPT:
        raise RuntimeError(
            "Refusing the anyone-can-spend regtest payout script on mainnet"
        )

    print(f"PAYOUT ADDRESS: {payout_address}")
    print(f"PAYOUT SCRIPT (Bitcoin Core): {payout_script.hex()}")
    if expected_script is None:
        print("EXPECTED PAYOUT SCRIPT: not supplied")
        print(
            "PAYOUT SCRIPT MATCH: NOT CHECKED "
            "(use --expected-payout-script to pin the payout)"
        )
        return payout_script

    print(f"EXPECTED PAYOUT SCRIPT: {expected_script.hex()}")
    if payout_script != expected_script:
        print("PAYOUT SCRIPT MATCH: NO")
        raise RuntimeError(
            "Payout script mismatch: Bitcoin Core resolved "
            f"{payout_script.hex()} but --expected-payout-script is "
            f"{expected_script.hex()}"
        )
    print("PAYOUT SCRIPT MATCH: YES")
    return payout_script


def verify_wallet_ownership(
    cli: Path,
    datadir: Path | None,
    conf: Path | None,
    payout_address: str,
    payout_script: bytes,
) -> None:
    try:
        wallet_info = rpc(
            cli,
            "mainnet",
            datadir,
            conf,
            "getaddressinfo",
            payout_address,
        )
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            "Wallet ownership check failed; Bitcoin Core's wallet could not "
            f"confirm the payout address ({error}); CUDA mining was prevented"
        ) from error
    if not isinstance(wallet_info, dict) or wallet_info.get("ismine") is not True:
        raise RuntimeError(
            "Wallet ownership check failed; Bitcoin Core's wallet does not "
            "report ismine=true for the payout address; CUDA mining was prevented"
        )
    # Only the public script is read from the wallet's answer; descriptors and
    # key metadata are never printed.
    wallet_script = wallet_info.get("scriptPubKey")
    if (
        not isinstance(wallet_script, str)
        or wallet_script.lower() != payout_script.hex()
    ):
        raise RuntimeError(
            "Wallet ownership check failed; the wallet's scriptPubKey for the "
            "payout address does not match the validated payout script; "
            "CUDA mining was prevented"
        )


def print_live_preflight_summary(
    payout_address: str,
    payout_script: bytes,
    expected_script: bytes,
) -> None:
    # Reached only after every check named here has passed without raising.
    print(
        "NETWORK: MAINNET\n"
        "MODE: LIVE\n"
        "CHAIN CHECK: PASS\n"
        "SYNC CHECK: PASS\n"
        "PEER CHECK: PASS\n"
        f"PAYOUT ADDRESS: {payout_address}\n"
        f"PAYOUT SCRIPT: {payout_script.hex()}\n"
        f"EXPECTED PAYOUT SCRIPT: {expected_script.hex()}\n"
        "SCRIPT MATCH: PASS\n"
        "WALLET OWNERSHIP: PASS\n"
        "SUBMISSION TRANSPORT: bitcoin-cli -stdin\n"
        "LIVE MAINNET PREFLIGHT: PASS",
        flush=True,
    )


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


def transaction_outputs(raw_transaction: bytes) -> list[tuple[int, bytes]]:
    # transaction_hashes rejects any malformed or truncated serialization, so
    # the offsets below are known to be in range.
    transaction_hashes(raw_transaction)
    offset = 6 if raw_transaction[4] == 0 else 4
    input_count, offset = read_compact_size(raw_transaction, offset)
    for _ in range(input_count):
        script_length, offset = read_compact_size(raw_transaction, offset + 36)
        offset += script_length + 4

    output_count, offset = read_compact_size(raw_transaction, offset)
    outputs = []
    for _ in range(output_count):
        value = int.from_bytes(raw_transaction[offset : offset + 8], "little")
        script_length, offset = read_compact_size(raw_transaction, offset + 8)
        outputs.append((value, raw_transaction[offset : offset + script_length]))
        offset += script_length
    return outputs


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


def verify_candidate_block(
    block: bytes,
    verified_header: bytes,
    target: int,
    template: dict,
    expected_previous: str,
    coinbase: bytes,
    coinbase_txid: bytes,
    transactions: list[bytes],
    payout_script: bytes,
    witness_commitment_script: bytes | None,
) -> None:
    if len(verified_header) != 80 or block[:80] != verified_header:
        raise ValueError("serialized header differs from the verified header")
    if int.from_bytes(double_sha256(block[:80])[::-1], "big") > target:
        raise ValueError("serialized header does not meet the template target")
    if (
        template["previousblockhash"] != expected_previous
        or block[4:36] != bytes.fromhex(expected_previous)[::-1]
    ):
        raise ValueError("header previous block hash differs from the template")

    transaction_count, offset = read_compact_size(block, 80)
    if (
        transaction_count != 1 + len(transactions)
        or block[offset : offset + len(coinbase)] != coinbase
        or block[offset + len(coinbase) :] != b"".join(transactions)
    ):
        raise ValueError("serialized transactions differ from the mined template")

    leaves = [transaction_hashes(coinbase)[0]]
    leaves.extend(transaction_hashes(transaction)[0] for transaction in transactions)
    if leaves[0] != coinbase_txid:
        raise ValueError("coinbase txid is not the first Merkle leaf")
    if merkle_root(leaves) != block[36:68]:
        raise ValueError("recomputed Merkle root differs from the header")

    expected_outputs = [(template["coinbasevalue"], payout_script)]
    if witness_commitment_script is not None:
        expected_outputs.append((0, witness_commitment_script))
    if not payout_script or transaction_outputs(coinbase) != expected_outputs:
        raise ValueError(
            "coinbase outputs do not pay the full coinbase value to the "
            "validated payout script"
        )


def submission_permitted(network: str, dry_run: bool, live_mainnet: bool) -> bool:
    if dry_run:
        return False
    if network == "mainnet":
        return live_mainnet
    return network == "regtest"


def save_unsubmitted_block(block_hash: str, block: bytes) -> Path | None:
    # Written under a temporary name and renamed, so the final name never
    # holds a partial block. The data is not forced to disk here; that is
    # flush_saved_block, which must not delay a submission.
    path = Path(__file__).with_name(f"unsubmitted_block_{block_hash}.hex")
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(block.hex(), encoding="ascii")
        os.replace(temporary, path)
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass
        return None
    return path


def flush_saved_block(path: Path) -> None:
    try:
        with open(path, "rb+") as saved:
            os.fsync(saved.fileno())
    except OSError:
        pass


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
            or len(raw_data) % 2
            or HEX_PATTERN.match(raw_data) is None
        ):
            raise RuntimeError(
                f"Template transaction {index} data must be nonempty hexadecimal"
            )
        if (
            not isinstance(txid, str)
            or len(txid) != 64
            or HEX_PATTERN.match(txid) is None
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
            or HEX_PATTERN.match(witness_txid) is None
        ):
            raise RuntimeError(
                f"Template transaction {index} hash must be 64 hexadecimal characters"
            )

    commitment = template.get("default_witness_commitment")
    if commitment is not None and (
        not isinstance(commitment, str)
        or len(commitment) % 2
        or (commitment != "" and HEX_PATTERN.match(commitment) is None)
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
    timeout: float | None = None,
) -> object:
    arguments = [
        param
        if isinstance(param, str)
        else json.dumps(param, separators=(",", ":"))
        for param in params
    ]
    stdin_text = None
    if method in STDIN_RPC_METHODS:
        command = bitcoin_cli_command(cli, network, datadir, conf, "-stdin", method)
        stdin_text = "\n".join(arguments)
    else:
        command = bitcoin_cli_command(
            cli,
            network,
            datadir,
            conf,
            method,
            *arguments,
        )
    try:
        result = subprocess.run(
            command,
            input=stdin_text,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise RpcError(
            f"Bitcoin Core RPC {method} timed out after {timeout:g} seconds"
        ) from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RpcError(f"Bitcoin Core RPC {method} failed: {detail}")
    if not result.stdout.strip():
        if method == "submitblock":
            return None
        raise RpcError(f"Bitcoin Core RPC {method} returned no result")
    if method in RAW_STRING_RPC_METHODS or method == "submitblock":
        return result.stdout.strip()
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RpcError(
            f"Bitcoin Core RPC {method} returned invalid JSON"
        ) from error


def check_node_health(
    cli: Path,
    datadir: Path | None,
    conf: Path | None,
    tor_host: str,
    tor_port: int,
    min_peers: int,
    require_onion_peers: bool,
) -> None:
    """Raise when mainnet work would go stale unnoticed; never starts or restarts anything."""
    if not socks5_ready(tor_host, tor_port):
        raise RuntimeError(
            f"Tor SOCKS5 proxy at {tor_host}:{tor_port} is not answering"
        )
    chain = rpc(cli, "mainnet", datadir, conf, "getblockchaininfo", timeout=TIP_CHECK_TIMEOUT)
    if not isinstance(chain, dict) or chain.get("chain") != "main":
        raise RuntimeError("Bitcoin Core RPC did not confirm the main chain")
    if chain.get("initialblockdownload") is not False:
        raise RuntimeError("Bitcoin Core reports initial block download")
    network_info = rpc(cli, "mainnet", datadir, conf, "getnetworkinfo", timeout=TIP_CHECK_TIMEOUT)
    if not isinstance(network_info, dict) or network_info.get("networkactive") is not True:
        raise RuntimeError("Bitcoin Core networking is inactive")
    connections = network_info.get("connections")
    if (
        isinstance(connections, bool)
        or not isinstance(connections, int)
        or connections < max(1, min_peers)
    ):
        raise RuntimeError(
            f"Bitcoin Core has {connections!r} connected peer(s); "
            f"{max(1, min_peers)} required"
        )
    if require_onion_peers:
        peers = rpc(cli, "mainnet", datadir, conf, "getpeerinfo", timeout=TIP_CHECK_TIMEOUT)
        if not isinstance(peers, list) or not any(
            isinstance(peer, dict) and peer.get("network") == "onion" for peer in peers
        ):
            raise RuntimeError("Bitcoin Core has no connected onion peer")


class NodeMonitor:
    """Polls node health on its own thread and only records the outcome.

    It never touches the CUDA process and never prints; the mining thread asks
    reason() between chunks. One failed poll is not a reason to pause.
    """

    def __init__(self, check: Callable[[], None], interval: float) -> None:
        self.check = check
        self.interval = interval
        self.lock = threading.Lock()
        self.failures = 0
        self.last_error = ""
        self.stopped = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="node-monitor", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        # Preflight has just passed, so the first poll waits a full interval.
        while not self.stopped.wait(self.interval):
            self.poll()

    def poll(self) -> None:
        try:
            self.check()
        except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as error:
            with self.lock:
                self.failures += 1
                self.last_error = str(error)
        else:
            with self.lock:
                self.failures = 0

    def reason(self) -> str | None:
        with self.lock:
            if self.failures >= MONITOR_FAILURES_TO_PAUSE:
                return f"{self.failures} consecutive health checks failed: {self.last_error}"
        return None

    def reset(self) -> None:
        with self.lock:
            self.failures = 0

    def close(self) -> None:
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(timeout=TIP_CHECK_TIMEOUT)


def wait_for_recovery(check: Callable[[], None], timeout: float) -> None:
    """Block until check() passes, with backoff; give up after timeout seconds."""
    deadline = time.monotonic() + timeout
    attempt = 0
    while True:
        try:
            check()
            return
        except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as error:
            delay = RECOVERY_DELAYS[min(attempt, len(RECOVERY_DELAYS) - 1)]
            attempt += 1
            if time.monotonic() + delay > deadline:
                raise RuntimeError(
                    f"Connectivity did not recover within {timeout:g} seconds "
                    f"(last check: {error}); mining stopped"
                ) from error
            print(
                f"[MONITOR] Not ready ({error}); retrying in {delay:g} s.",
                flush=True,
            )
            time.sleep(delay)


@dataclass(frozen=True)
class Work:
    """One fully validated block template; never modified after it is built."""

    template: dict
    bits: int
    target: int
    nonce_start: int
    nonce_end: int
    coinbase_flags: bytes
    witness_commitment: bytes | None
    transaction_data: list[bytes]
    transaction_txids: list[bytes]
    fetched_at: float


def fetch_work(
    cli: Path,
    network: str,
    datadir: Path | None,
    conf: Path | None,
) -> Work | None:
    """Fetch and validate a template; None when it no longer builds on Core's tip."""
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
    current_tip = rpc(cli, network, datadir, conf, "getbestblockhash")
    if template.get("previousblockhash") != current_tip:
        return None
    bits, target, nonce_start, nonce_end = validate_template(template)

    coinbase_aux = template.get("coinbaseaux", {})
    flags_hex = coinbase_aux.get("flags", "") if isinstance(coinbase_aux, dict) else ""
    try:
        coinbase_flags = bytes.fromhex(flags_hex)
    except ValueError as error:
        raise RuntimeError("Template coinbase flags are not valid hexadecimal") from error
    try:
        witness_commitment = get_witness_commitment(template, template["transactions"])
        transaction_data, transaction_txids = template_transactions(template)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid block template: {error}") from error
    return Work(
        template,
        bits,
        target,
        nonce_start,
        nonce_end,
        coinbase_flags,
        witness_commitment,
        transaction_data,
        transaction_txids,
        time.monotonic(),
    )


class TemplatePrefetcher:
    """Prepares the next Work on a background thread while the GPU scans.

    The thread only runs bitcoin-cli and pure functions and hands over a
    finished, immutable Work; it never touches the CUDA process.
    """

    RETRY_SECONDS = 5.0

    def __init__(self, fetch: Callable[[], Work | None]) -> None:
        self.fetch = fetch
        self.lock = threading.Lock()
        self.result: Work | None = None
        self.thread: threading.Thread | None = None
        self.next_request = 0.0

    def request(self) -> None:
        with self.lock:
            if (
                self.result is not None
                or (self.thread is not None and self.thread.is_alive())
                or time.monotonic() < self.next_request
            ):
                return
            self.next_request = time.monotonic() + self.RETRY_SECONDS
            self.thread = threading.Thread(
                target=self._run, name="template-prefetch", daemon=True
            )
            self.thread.start()

    def _run(self) -> None:
        try:
            work = self.fetch()
        except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError):
            # The mining thread keeps rolling its current template and finds
            # out about a persistent failure through its own checks.
            return
        with self.lock:
            self.result = work

    def ready(self) -> bool:
        with self.lock:
            return self.result is not None

    def take(self) -> Work | None:
        with self.lock:
            work, self.result = self.result, None
            return work

    def discard(self) -> None:
        self.take()

    def close(self) -> None:
        if self.thread is not None:
            self.thread.join(timeout=TIP_CHECK_TIMEOUT)


class MiningSession:
    """Mainnet state shared across templates.

    tip_changed and pause_reason are called only by the thread that drives
    CUDA. The monitor and the prefetcher run on their own threads.
    """

    def __init__(
        self,
        cli: Path,
        datadir: Path | None,
        conf: Path | None,
        monitor: NodeMonitor,
        prefetcher: TemplatePrefetcher,
    ) -> None:
        self.cli = cli
        self.datadir = datadir
        self.conf = conf
        self.monitor = monitor
        self.prefetcher = prefetcher
        self.tip_failures = 0
        self.tip_error = ""
        self.accepted_block: str | None = None
        self.confirmed_tip: tuple[str, float] | None = None

    def tip_confirmed_recently(self, expected_previous: str) -> bool:
        return (
            self.confirmed_tip is not None
            and self.confirmed_tip[0] == expected_previous
            and time.monotonic() - self.confirmed_tip[1] <= TIP_CONFIRMATION_SECONDS
        )

    def tip_changed(self, expected_previous: str, fetched_at: float) -> bool:
        try:
            tip = rpc(
                self.cli,
                "mainnet",
                self.datadir,
                self.conf,
                "getbestblockhash",
                timeout=TIP_CHECK_TIMEOUT,
            )
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            # Unknown is not stale: the chunk in flight is still read, so a
            # candidate in it is kept. Repeated failures pause new work.
            self.tip_failures += 1
            self.tip_error = str(error)
            print(
                f"\n[MONITOR] Tip check failed "
                f"({self.tip_failures}/{TIP_CHECK_FAILURE_LIMIT}): {error}",
                flush=True,
            )
            return False
        self.tip_failures = 0
        if tip != expected_previous:
            self.confirmed_tip = None
            return True
        self.confirmed_tip = (tip, time.monotonic())
        if time.monotonic() - fetched_at >= TEMPLATE_REFRESH_SECONDS:
            self.prefetcher.request()
        return False

    def pause_reason(self) -> str | None:
        if self.tip_failures >= TIP_CHECK_FAILURE_LIMIT:
            return (
                f"{self.tip_failures} consecutive tip checks failed: {self.tip_error}"
            )
        return self.monitor.reason()

    def reset(self) -> None:
        self.tip_failures = 0
        self.confirmed_tip = None
        self.monitor.reset()
        self.prefetcher.discard()

    def close(self) -> None:
        self.monitor.close()
        self.prefetcher.close()


class CudaMiner:
    """One `cuda_miner --serve` process reused for every chunk of a session.

    The process is started on the first scan, so nothing touches the GPU until
    a template is ready. Any protocol violation or process death is fatal for
    the session: the process is killed and never restarted.
    """

    def __init__(self, executable: Path) -> None:
        self.executable = executable
        self.process: subprocess.Popen[str] | None = None
        self.failed = False
        self.request_id = 0
        self.pending_id: str | None = None
        self.pending_abandoned = False
        # Result of the last abandoned chunk once it has been read, and the
        # block context needed to check it; see preserve_abandoned_candidate.
        self.abandoned_result: tuple[int, str] | None = None
        self.abandoned_work: tuple | None = None

    def _start(self) -> None:
        try:
            self.process = subprocess.Popen(
                [str(self.executable), "--serve"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                # Keep console Ctrl+C away from the child so that close()
                # ends it in an orderly way and it frees its CUDA resources.
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        except OSError as error:
            self.failed = True
            raise RuntimeError(f"Could not start CUDA miner: {error}") from error
        try:
            ready = self.process.stdout.readline()
        except (OSError, ValueError) as error:
            self._fail(f"lost contact with the CUDA process ({error})")
        if ready != "READY\n":
            self._fail(f"CUDA process did not report READY (got {ready!r})")

    def _fail(self, detail: str) -> NoReturn:
        process, self.process = self.process, None
        self.failed = True
        if process is not None:
            diagnostics = ""
            process.kill()
            try:
                _, diagnostics = process.communicate(timeout=CUDA_SHUTDOWN_TIMEOUT)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            detail += f"; exit status {process.returncode}"
            if diagnostics and diagnostics.strip():
                detail += f": {diagnostics.strip()}"
        raise RuntimeError(f"CUDA miner failed: {detail}")

    def scan(self, header: bytes, start: int, count: int) -> tuple[int, str] | None:
        self.start_scan(header, start, count)
        return self.finish_scan()

    def start_scan(self, header: bytes, start: int, count: int) -> None:
        """Send one request and return at once; the GPU scans in the background."""
        if self.failed:
            raise RuntimeError(
                "CUDA miner failed earlier in this session and was not restarted"
            )
        if self.process is None:
            self._start()
        if self.pending_id is not None:
            if not self.pending_abandoned:
                self._fail("a scan was started before the previous result was read")
            # Replies come back in request order, so the abandoned scan's
            # reply must be consumed before new work; it is never returned
            # as a scan result.
            self.abandoned_result = self._read_reply()

        self.request_id = (self.request_id + 1) & 0xFFFFFFFF
        request_id = str(self.request_id)
        try:
            self.process.stdin.write(
                f"SCAN {request_id} {header.hex()} {start} {count}\n"
            )
            self.process.stdin.flush()
        except (OSError, ValueError) as error:
            self._fail(f"lost contact with the CUDA process ({error})")
        self.pending_id = request_id
        self.pending_abandoned = False

    def abandon_scan(self) -> None:
        """Mark the scan in flight as stale work; its result will never be returned."""
        if self.pending_id is not None:
            self.pending_abandoned = True

    def drain_abandoned(self) -> tuple[int, str] | None:
        """Read (once) what the abandoned chunk found; None if there was none."""
        if self.pending_id is not None and self.pending_abandoned and not self.failed:
            self.abandoned_result = self._read_reply()
        result, self.abandoned_result = self.abandoned_result, None
        return result

    def finish_scan(self) -> tuple[int, str] | None:
        if self.failed:
            raise RuntimeError(
                "CUDA miner failed earlier in this session and was not restarted"
            )
        if self.pending_id is None or self.pending_abandoned:
            self._fail("no scan result is waiting to be read")
        return self._read_reply()

    def _read_reply(self) -> tuple[int, str] | None:
        expected_id = self.pending_id
        self.pending_id = None
        self.pending_abandoned = False
        try:
            line = self.process.stdout.readline()
        except (OSError, ValueError) as error:
            self._fail(f"lost contact with the CUDA process ({error})")
        if not line.endswith("\n"):
            self._fail(f"CUDA process ended without a complete reply (got {line!r})")

        fields = line.split()
        if fields[:1] == [expected_id]:
            if fields[1:] == ["NONE"]:
                return None
            if (
                len(fields) == 4
                and fields[1] == "FOUND"
                and fields[2].isascii()
                and fields[2].isdigit()
                and int(fields[2]) <= 0xFFFFFFFF
                and len(fields[3]) == 64
                and all(character in "0123456789abcdef" for character in fields[3])
            ):
                return int(fields[2]), fields[3]
        self._fail(f"Unexpected CUDA miner output: {line!r}")

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        try:
            # End of input asks the process to free its CUDA resources and exit.
            try:
                process.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                process.wait(timeout=CUDA_SHUTDOWN_TIMEOUT)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        except BaseException:
            process.kill()
            raise
        finally:
            for stream in (process.stdout, process.stderr):
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass


def start_chunk(
    cuda_miner: CudaMiner,
    header: bytes,
    start: int,
    count: int,
) -> None:
    cuda_miner.start_scan(header, start, count)


def finish_chunk(cuda_miner: CudaMiner) -> tuple[int, str] | None:
    return cuda_miner.finish_scan()


def abandon_chunk(cuda_miner: CudaMiner) -> None:
    cuda_miner.abandon_scan()


def find_nonce(
    cuda_miner: CudaMiner,
    header: bytes,
    bits: int,
    chunk_size: int,
    nonce_start: int,
    nonce_end: int,
    stale_check: Callable[[], bool],
    pause_check: Callable[[], str | None] | None = None,
    overlap_first_check: bool = False,
) -> tuple[int, bytes] | None:
    target = bits_to_target(bits)
    start = nonce_start
    count = 0
    total_hashes = 0
    scan_started = time.perf_counter()
    last_status = scan_started - STATUS_INTERVAL
    if start <= nonce_end:
        if pause_check is not None and (reason := pause_check()):
            raise MiningPaused(reason)
        count = min(chunk_size, nonce_end - start + 1)
        if overlap_first_check:
            # The caller has just seen this tip confirmed, so the first chunk
            # is treated like every later one: launched, then checked.
            chunk_started = time.perf_counter()
            start_chunk(cuda_miner, header, start, count)
            if stale_check():
                abandon_chunk(cuda_miner)
                raise StaleTemplate
        else:
            # No GPU work starts on a template that is already stale.
            if stale_check():
                raise StaleTemplate
            chunk_started = time.perf_counter()
            start_chunk(cuda_miner, header, start, count)
    while count:
        # The tip check belonging to this chunk has already returned "still
        # current"; only then is the chunk's result read.
        candidate = finish_chunk(cuda_miner)
        chunk_seconds = time.perf_counter() - chunk_started
        total_hashes += count
        now = time.perf_counter()
        elapsed = max(now - scan_started, 1e-9)
        if candidate is not None or now - last_status >= STATUS_INTERVAL:
            last_status = now
            print(
                f"\rHashing nonce {start:08x}..{start + count - 1:08x} "
                f"of {nonce_start:08x}..{nonce_end:08x} | "
                f"{total_hashes:,} hashes | "
                f"{count / max(chunk_seconds, 1e-9):,.0f} H/s chunk | "
                f"{total_hashes / elapsed:,.0f} H/s average",
                end="",
                flush=True,
            )
        if candidate is not None:
            print()
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

        next_start = start + count
        if next_start > nonce_end:
            break
        next_count = min(chunk_size, nonce_end - next_start + 1)
        # Nothing is in flight here, so this is where a confirmed loss of
        # connectivity stops new work.
        if pause_check is not None and (reason := pause_check()):
            print()
            raise MiningPaused(reason)
        # Launch the next chunk before asking Core about the tip, so the GPU
        # works during the RPC instead of idling. The check still happens at
        # the same moment as before; if it reports a new tip (or fails), the
        # chunk already running is abandoned and its result is never read.
        next_started = time.perf_counter()
        start_chunk(cuda_miner, header, next_start, next_count)
        if stale_check():
            abandon_chunk(cuda_miner)
            print()
            raise StaleTemplate
        start, count, chunk_started = next_start, next_count, next_started

    if total_hashes:
        print()
    # CUDA uses 0xffffffff as its no-result sentinel, so verify that nonce on
    # the CPU when it is included in the template's permitted range.
    if nonce_start <= 0xFFFFFFFF <= nonce_end:
        last_header = header[:76] + struct.pack("<I", 0xFFFFFFFF)
        last_hash = double_sha256(last_header)
        if int.from_bytes(last_hash[::-1], "big") <= target:
            return 0xFFFFFFFF, last_hash
    return None


def known_block_state(
    cli: Path,
    network: str,
    datadir: Path | None,
    conf: Path | None,
    block_hash: str,
) -> str | None:
    """'active', 'side' or None (Core does not know the block, or did not answer).

    Core answers getblockheader for any stored header, including a block it
    has not fully validated, so only confirmations >= 0 means accepted.
    """
    try:
        header = rpc(
            cli, network, datadir, conf, "getblockheader", block_hash,
            timeout=TIP_CHECK_TIMEOUT,
        )
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return None
    confirmations = header.get("confirmations") if isinstance(header, dict) else None
    if isinstance(confirmations, bool) or not isinstance(confirmations, int):
        return None
    return "active" if confirmations >= 0 else "side"


def submit_candidate(
    cli: Path,
    network: str,
    datadir: Path | None,
    conf: Path | None,
    block: bytes,
    block_hash: str,
    saved_path: Path | None,
) -> str:
    """Submit a verified block; returns 'accepted', 'side-chain' or 'stale'.

    A failed or timed-out call says nothing about whether Core received the
    block, so Core is asked before each retry. Resubmitting is harmless: Core
    answers 'duplicate' for a block it already has.
    """
    block_hex = block.hex()
    delays = (0.0,) + (SUBMIT_RETRY_DELAYS if network == "mainnet" else ())
    last_error: Exception | None = None
    for attempt, delay in enumerate(delays, start=1):
        if delay:
            print(
                f"Retrying submitblock in {delay:g} s "
                f"(attempt {attempt} of {len(delays)}).",
                flush=True,
            )
            time.sleep(delay)
        try:
            submission = rpc(
                cli, network, datadir, conf, "submitblock", block_hex,
                timeout=SUBMIT_TIMEOUT,
            )
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            last_error = error
            print(
                f"SUBMISSION ATTEMPT {attempt} DID NOT COMPLETE: {error}",
                flush=True,
            )
        else:
            if submission is None:
                print(
                    "SUBMITTED: submitblock response=None "
                    "(bitcoin-cli returned no text for Core's null result)",
                    flush=True,
                )
                return "accepted"
            if submission in SIDE_CHAIN_SUBMISSION_RESULTS:
                return "side-chain"
            if submission != "duplicate":
                print(f"REJECTED: submitblock response={submission!r}", flush=True)
                if submission == "prev-blk-not-found":
                    return "stale"
                raise RuntimeError(f"Bitcoin Core rejected the block: {submission}")
            last_error = RuntimeError(
                "Core reported the block as a duplicate but did not confirm its status"
            )

        # Either the call failed or Core says it already has the block: what
        # Core holds decides, not the failed call.
        state = known_block_state(cli, network, datadir, conf, block_hash)
        if state == "active":
            print(
                f"SUBMITTED: Bitcoin Core already has block {block_hash} "
                "on its active chain.",
                flush=True,
            )
            return "accepted"
        if state == "side":
            return "side-chain"

    if saved_path is None and network != "mainnet":
        saved_path = save_unsubmitted_block(block_hash, block)
    print(
        f"SUBMISSION FAILED: submitblock did not complete for {block_hash}; "
        "the block is NOT confirmed as accepted.",
        flush=True,
    )
    if saved_path is not None:
        print(
            f"Serialized block saved to {saved_path}; resubmit it with: "
            f'Get-Content "{saved_path}" | & "{cli}" -stdin submitblock',
            flush=True,
        )
    raise RuntimeError(
        f"submitblock failed for candidate {block_hash}: {last_error}"
    ) from last_error


def preserve_abandoned_candidate(cuda_miner: object, network: str) -> None:
    """Check what a chunk abandoned as stale found, before new GPU work starts.

    A valid block there builds on a tip Core has already replaced. Core stores
    such a block but never relays it, so it is saved and not submitted.
    """
    if not isinstance(cuda_miner, CudaMiner):
        return
    reply = cuda_miner.drain_abandoned()
    context, cuda_miner.abandoned_work = cuda_miner.abandoned_work, None
    if reply is None or context is None:
        return
    (
        header, target, template, expected_previous, coinbase, coinbase_txid,
        transactions, payout_script, witness_commitment,
    ) = context
    nonce, displayed_hash = reply
    mined_header = header[:76] + struct.pack("<I", nonce)
    digest = double_sha256(mined_header)
    block_hash = digest[::-1].hex()
    if block_hash != displayed_hash:
        raise RuntimeError("CUDA hash does not match the CPU SHA-256d result")
    if int.from_bytes(digest[::-1], "big") > target:
        raise RuntimeError("CUDA returned a hash that does not meet the target")
    block = serialize_block(mined_header, coinbase, transactions)
    try:
        verify_candidate_block(
            block, mined_header, target, template, expected_previous, coinbase,
            coinbase_txid, transactions, payout_script, witness_commitment,
        )
    except ValueError as error:
        raise RuntimeError(
            f"Stale candidate failed local consistency checks: {error}"
        ) from error
    saved_path = save_unsubmitted_block(block_hash, block) if network == "mainnet" else None
    print(
        f"STALE CANDIDATE: block {block_hash} solves the previous tip, which "
        "Core had already replaced; it was not submitted because Core does "
        "not relay a block that competes with its tip."
        + (f" Saved to {saved_path}." if saved_path is not None else ""),
        flush=True,
    )


def mine_one_block(
    cli: Path,
    cuda_miner: CudaMiner,
    network: str,
    datadir: Path | None,
    conf: Path | None,
    chunk_size: int,
    extra_nonce: int,
    payout_script: bytes,
    dry_run: bool = False,
    live_mainnet: bool = False,
    payout_address: str | None = None,
    session: MiningSession | None = None,
) -> bool:
    work = session.prefetcher.take() if session is not None else None
    if work is None:
        work = fetch_work(cli, network, datadir, conf)
    if work is None:
        print(
            "Template previousblockhash does not match Core's current tip; "
            "discarding it before CUDA and requesting a fresh template.",
            flush=True,
        )
        return False
    template = work.template
    target = work.target
    print(
        f"Template target cross-check: bits={template['bits']}, "
        f"decoded_target={target:064x}, gbt_target={template['target'].lower()}",
        flush=True,
    )
    print(
        f"Template nonce range: {work.nonce_start:08x}..{work.nonce_end:08x}; "
        f"time={template['curtime']} (minimum={template['mintime']})",
        flush=True,
    )
    # The abandoned chunk is still running; its result is read here, when new
    # GPU work would have had to wait for it anyway.
    preserve_abandoned_candidate(cuda_miner, network)
    print(
        f"{'Dry-running' if dry_run else 'Mining'} {network} block "
        f"{template['height']} "
        f"(bits={template['bits']}, chunk={chunk_size:,})",
        flush=True,
    )
    expected_previous = template["previousblockhash"]
    if session is None:
        scan_checks: tuple = (
            lambda: rpc(
                cli,
                network,
                datadir,
                conf,
                "getbestblockhash",
                timeout=TIP_CHECK_TIMEOUT,
            ) != expected_previous,
        )
    else:
        scan_checks = (
            lambda: session.tip_changed(expected_previous, work.fetched_at),
            session.pause_reason,
        )

    # Exhausting the 32-bit nonce space only needs a different coinbase, so
    # the validated template is kept and the extranonce rolled. The caller's
    # extra_nonce is the high half, which keeps every coinbase of a session
    # distinct.
    roll = 0
    while True:
        try:
            coinbase, coinbase_txid = create_coinbase(
                int(template["height"]),
                work.coinbase_flags,
                ((extra_nonce & 0xFFFFFFFF) << 32) | roll,
                template["coinbasevalue"],
                payout_script,
                work.witness_commitment,
            )
            header = build_header(template, coinbase_txid, work.transaction_txids)
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"Invalid block template: {error}") from error
        # After a rollover, or when a prefetched template builds on the tip
        # that the previous chunk's check confirmed moments ago, waiting for
        # another check before the first chunk would only idle the GPU.
        overlap = roll > 0 or (
            session is not None and session.tip_confirmed_recently(expected_previous)
        )
        try:
            candidate = find_nonce(
                cuda_miner,
                header,
                work.bits,
                chunk_size,
                work.nonce_start,
                work.nonce_end,
                *scan_checks,
                **({"overlap_first_check": True} if overlap else {}),
            )
        except StaleTemplate:
            if isinstance(cuda_miner, CudaMiner) and cuda_miner.pending_abandoned:
                cuda_miner.abandoned_work = (
                    header, target, template, expected_previous, coinbase,
                    coinbase_txid, work.transaction_data, payout_script,
                    work.witness_commitment,
                )
            if session is not None:
                session.prefetcher.discard()
            print("Template became stale; requesting new work.")
            return False
        if candidate is not None:
            break

        age = time.monotonic() - work.fetched_at
        if session is not None and session.prefetcher.ready():
            print("Nonce space exhausted; switching to the refreshed template.")
            return False
        if age >= TEMPLATE_REFRESH_SECONDS * (1 if session is None else 2):
            print(
                "Nonce space exhausted; changing the coinbase extranonce "
                "and retrying the template."
            )
            return False
        roll = (roll + 1) & 0xFFFFFFFF

    nonce, raw_hash = candidate
    mined_header = header[:76] + struct.pack("<I", nonce)
    block = serialize_block(mined_header, coinbase, work.transaction_data)
    block_hash = double_sha256(mined_header)[::-1].hex()
    if block_hash != raw_hash[::-1].hex():
        raise RuntimeError("Local block-header hash changed after candidate verification")
    try:
        verify_candidate_block(
            block,
            mined_header,
            target,
            template,
            expected_previous,
            coinbase,
            coinbase_txid,
            work.transaction_data,
            payout_script,
            work.witness_commitment,
        )
    except ValueError as error:
        raise RuntimeError(
            f"Candidate block failed local consistency checks and was not "
            f"submitted: {error}"
        ) from error
    print(
        f"CANDIDATE FOUND: height={template['height']} nonce={nonce} "
        f"hash={block_hash}",
        flush=True,
    )

    # A verified mainnet block is on disk before anything else can fail. The
    # write goes through the OS cache; forcing it to disk happens beside the
    # submission, not ahead of it.
    saved_path: Path | None = None
    flusher: threading.Thread | None = None
    if network == "mainnet":
        saved_path = save_unsubmitted_block(block_hash, block)
        if saved_path is None:
            print("WARNING: the candidate block could not be saved to disk.", flush=True)
        else:
            print(f"CANDIDATE SAVED: {saved_path}", flush=True)
            flusher = threading.Thread(
                target=flush_saved_block, args=(saved_path,), daemon=True
            )
            flusher.start()

    if not submission_permitted(network, dry_run, live_mainnet):
        if flusher is not None:
            flusher.join()
        if not dry_run:
            raise RuntimeError(
                "Mainnet submission requires --live-mainnet; "
                f"candidate {block_hash} was not submitted"
            )
        print(
            f"DRY RUN: candidate {block_hash} verified locally and not submitted.",
            flush=True,
        )
        return False

    # No tip check first: Core decides whether the block extends its chain,
    # and asking beforehand would only delay the broadcast.
    try:
        outcome = submit_candidate(
            cli, network, datadir, conf, block, block_hash, saved_path
        )
    finally:
        if flusher is not None:
            flusher.join()
    if outcome != "accepted":
        if outcome == "side-chain":
            print(
                f"NOT ON ACTIVE CHAIN: Bitcoin Core stored block {block_hash}, "
                f"but another block holds height {template['height']}. Core "
                "does not relay a block that competes with its tip, so this "
                "block earns the reward only if the chain reorganizes onto it.",
                flush=True,
            )
        print(
            "Core did not connect the block to its best chain; "
            "requesting a fresh template.",
            flush=True,
        )
        return False
    if network == "mainnet":
        banner = "=" * 72
        print(
            f"{banner}\n"
            "*** MAINNET BLOCK ACCEPTED BY BITCOIN CORE ***\n"
            f"BLOCK HASH: {block_hash}\n"
            f"BLOCK HEIGHT: {template['height']}\n"
            f"PAYOUT ADDRESS: {payout_address}\n"
            f"PAYOUT SCRIPT: {payout_script.hex()}\n"
            f"COINBASE TXID: {coinbase_txid[::-1].hex()}\n"
            f"COINBASE VALUE: {template['coinbasevalue']} satoshis\n"
            f"{banner}",
            flush=True,
        )
        if session is not None:
            session.accepted_block = block_hash

    try:
        status = submitted_block_status(
            cli,
            network,
            datadir,
            conf,
            int(template["height"]),
            block_hash,
        )
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        if network != "mainnet":
            raise
        print(
            f"WARNING: active-chain status of {block_hash} could not be "
            f"confirmed: {error}",
            flush=True,
        )
        return True
    # Live mainnet stops after any accepted block so the result can be inspected.
    if status == "ACTIVE CHAIN":
        print(
            f"ACCEPTED: Bitcoin Core recognizes block {block_hash}.",
            flush=True,
        )
        print(
            f"ACTIVE CHAIN: block {block_hash} is active at height "
            f"{template['height']}.",
            flush=True,
        )
        return True
    if status == "STALE/ORPHANED":
        print(
            f"STALE/ORPHANED: Core knows block {block_hash}, "
            "but it is not on the active chain.",
            flush=True,
        )
        return network == "mainnet"

    print(
        f"SUBMITTED: Core knows block {block_hash}, "
        "but active-chain inclusion is not confirmed.",
        flush=True,
    )
    return network == "mainnet"


def monitor_block(
    cli: Path,
    network: str,
    datadir: Path | None,
    conf: Path | None,
    block_hash: str,
    poll_seconds: float = BLOCK_MONITOR_POLL_SECONDS,
) -> int:
    """Report a block's chain status until its coinbase is spendable.

    Read-only. Acceptance by this node is not a guaranteed reward: the block
    must stay on the active chain for COINBASE_MATURITY_CONFIRMATIONS.
    """
    print(
        f"MONITORING block {block_hash} until its coinbase has "
        f"{COINBASE_MATURITY_CONFIRMATIONS} confirmations (Ctrl+C stops "
        "watching; --monitor-block resumes it).",
        flush=True,
    )
    coinbase_txid: str | None = None
    last_report: tuple | None = None
    while True:
        try:
            header = rpc(
                cli, network, datadir, conf, "getblockheader", block_hash,
                timeout=SUBMIT_TIMEOUT,
            )
            confirmations = header.get("confirmations") if isinstance(header, dict) else None
            if isinstance(confirmations, bool) or not isinstance(confirmations, int):
                raise RuntimeError("Bitcoin Core returned an invalid confirmation count")
            if coinbase_txid is None:
                stored = rpc(
                    cli, network, datadir, conf, "getblock", block_hash, 1,
                    timeout=SUBMIT_TIMEOUT,
                )
                coinbase_txid = stored["tx"][0]
            try:
                wallet_tx = rpc(
                    cli, network, datadir, conf, "gettransaction", coinbase_txid,
                    timeout=SUBMIT_TIMEOUT,
                )
                categories = sorted(
                    {detail.get("category", "?") for detail in wallet_tx.get("details", [])}
                )
                wallet = "/".join(categories) if categories else "known, no wallet output"
            except (OSError, RuntimeError, subprocess.SubprocessError):
                wallet = "not reported by the wallet"
            report: tuple = (confirmations, wallet)
            if report != last_report:
                height = header.get("height")
                if confirmations < 0:
                    print(
                        f"NOT ON ACTIVE CHAIN: block {block_hash} (height {height}) "
                        "was reorganized out or never became active; its "
                        f"coinbase cannot be spent. Wallet: {wallet}.",
                        flush=True,
                    )
                elif confirmations < COINBASE_MATURITY_CONFIRMATIONS:
                    print(
                        f"ACTIVE CHAIN: height {height}, {confirmations} "
                        f"confirmation(s); coinbase {coinbase_txid} matures in "
                        f"{COINBASE_MATURITY_CONFIRMATIONS - confirmations} more "
                        f"block(s). Wallet: {wallet}.",
                        flush=True,
                    )
                else:
                    print(
                        f"COINBASE MATURE: block {block_hash} has {confirmations} "
                        f"confirmations; coinbase {coinbase_txid} is spendable. "
                        f"Wallet: {wallet}.",
                        flush=True,
                    )
                    return 0
        except (OSError, RuntimeError, KeyError, TypeError, IndexError, subprocess.SubprocessError) as error:
            report = ("unavailable", str(error))
            if report != last_report:
                print(
                    f"[MONITOR] Block status unavailable ({error}); still watching.",
                    flush=True,
                )
        last_report = report
        time.sleep(poll_seconds)


def submitted_block_status(
    cli: Path,
    network: str,
    datadir: Path | None,
    conf: Path | None,
    candidate_height: int,
    candidate_hash: str,
) -> str:
    chain_height = rpc(cli, network, datadir, conf, "getblockcount")
    if isinstance(chain_height, bool) or not isinstance(chain_height, int):
        raise RuntimeError("Bitcoin Core returned an invalid chain height")

    if chain_height >= candidate_height:
        active_hash = rpc(
            cli,
            network,
            datadir,
            conf,
            "getblockhash",
            candidate_height,
        )
        if active_hash == candidate_hash:
            return "ACTIVE CHAIN"

    header = rpc(cli, network, datadir, conf, "getblockheader", candidate_hash)
    if not isinstance(header, dict) or header.get("hash") != candidate_hash:
        raise RuntimeError(
            "Bitcoin Core did not return the submitted block's header"
        )
    confirmations = header.get("confirmations")
    if isinstance(confirmations, bool) or not isinstance(confirmations, int):
        raise RuntimeError(
            "Bitcoin Core returned an invalid block confirmation count"
        )
    if confirmations < 0:
        return "STALE/ORPHANED"
    return "SUBMITTED"


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
        "--mainnet",
        dest="network",
        action="store_const",
        const="mainnet",
        default=argparse.SUPPRESS,
        help="Select Bitcoin mainnet (equivalent to --network mainnet)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Continuously scan live mainnet templates without submitting blocks",
    )
    parser.add_argument(
        "--live-mainnet",
        action="store_true",
        help=(
            "Acknowledge live mainnet mining: valid blocks are submitted with "
            "submitblock (requires --mainnet; cannot be combined with --dry-run)"
        ),
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
        "--tor-host",
        default="127.0.0.1",
        help="Tor SOCKS5 host for mainnet preflight (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--tor-port",
        type=int,
        default=9150,
        help="Tor SOCKS5 port for mainnet preflight (default: 9150)",
    )
    parser.add_argument(
        "--tor-executable",
        type=Path,
        help="Standalone tor.exe to start if its SOCKS5 endpoint is unavailable",
    )
    parser.add_argument(
        "--tor-startup-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for Tor SOCKS5 readiness (default: 120)",
    )
    parser.add_argument(
        "--tor-poll-interval",
        type=float,
        default=0.5,
        help="Seconds between Tor readiness checks (default: 0.5)",
    )
    parser.add_argument(
        "--require-onion-peers",
        action="store_true",
        help="Require at least one connected onion peer before mainnet mining",
    )
    parser.add_argument(
        "--min-peers",
        type=int,
        default=1,
        help="Connected peers required to start and to keep mining on mainnet (default: 1)",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=15.0,
        help="Seconds between background health checks on mainnet (default: 15)",
    )
    parser.add_argument(
        "--recovery-timeout",
        type=float,
        default=3600.0,
        help=(
            "Seconds to keep retrying after mining paused for lost "
            "connectivity before giving up (default: 3600)"
        ),
    )
    parser.add_argument(
        "--monitor-block",
        metavar="HASH",
        help=(
            "Mine nothing; report this block's confirmations until its "
            "coinbase is spendable"
        ),
    )
    parser.add_argument(
        "--payout-address",
        help="Required on mainnet; address receiving the block reward",
    )
    parser.add_argument(
        "--expected-payout-script",
        help=(
            "Required with --live-mainnet, recommended with --dry-run; "
            "scriptPubKey hex that Bitcoin Core must resolve the payout "
            "address to, or startup aborts"
        ),
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
        default=250000000,
        help="Nonces per CUDA process invocation (default: 250000000)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.monitor_block is not None:
        block_hash = args.monitor_block.lower()
        if len(block_hash) != 64 or HEX_PATTERN.match(block_hash) is None:
            print("--monitor-block needs a 64-character block hash", file=sys.stderr)
            return 2
        if not args.bitcoin_cli.is_file():
            print(f"bitcoin-cli.exe not found: {args.bitcoin_cli}", file=sys.stderr)
            return 2
        keep_system_awake(True)
        try:
            return monitor_block(
                args.bitcoin_cli, args.network, args.datadir, args.bitcoin_conf,
                block_hash,
            )
        except KeyboardInterrupt:
            print("\nStopped by user.")
            return 130
        finally:
            keep_system_awake(False)
    if args.dry_run and args.network != "mainnet":
        print("--dry-run requires --mainnet", file=sys.stderr)
        return 2
    if args.live_mainnet and args.network != "mainnet":
        print("--live-mainnet requires --mainnet", file=sys.stderr)
        return 2
    if args.live_mainnet and args.dry_run:
        print(
            "--live-mainnet and --dry-run are mutually exclusive",
            file=sys.stderr,
        )
        return 2
    if args.network == "mainnet" and not (args.dry_run or args.live_mainnet):
        print(
            "Mainnet submission is gated: --mainnet alone never mines or "
            "submits. Add --dry-run (never submits) or --live-mainnet "
            "(submits valid blocks).",
            file=sys.stderr,
        )
        return 2
    if args.blocks < 0:
        print("--blocks must be nonnegative", file=sys.stderr)
        return 2
    if not 1 <= args.chunk_size <= 0xFFFFFFFF:
        print("--chunk-size must be from 1 through 4294967295", file=sys.stderr)
        return 2
    if not 1 <= args.tor_port <= 65535:
        print("--tor-port must be from 1 through 65535", file=sys.stderr)
        return 2
    if args.min_peers < 1:
        print("--min-peers must be at least 1", file=sys.stderr)
        return 2
    if not (
        math.isfinite(args.monitor_interval)
        and args.monitor_interval > 0
        and math.isfinite(args.recovery_timeout)
        and args.recovery_timeout > 0
    ):
        print(
            "--monitor-interval and --recovery-timeout must be positive",
            file=sys.stderr,
        )
        return 2
    if args.tor_startup_timeout <= 0 or args.tor_poll_interval <= 0:
        print(
            "--tor-startup-timeout and --tor-poll-interval must be positive",
            file=sys.stderr,
        )
        return 2
    if (
        not args.tor_host.strip()
        or not math.isfinite(args.tor_startup_timeout)
        or not math.isfinite(args.tor_poll_interval)
    ):
        print(
            "--tor-host must be nonempty and Tor timing values must be finite",
            file=sys.stderr,
        )
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
    expected_payout_script: bytes | None = None
    if args.expected_payout_script is not None:
        if args.network != "mainnet":
            print(
                "--expected-payout-script is only supported for mainnet",
                file=sys.stderr,
            )
            return 2
        try:
            expected_payout_script = bytes.fromhex(args.expected_payout_script)
        except ValueError:
            expected_payout_script = b""
        if not expected_payout_script:
            print(
                "--expected-payout-script must be nonempty hexadecimal",
                file=sys.stderr,
            )
            return 2
    if args.live_mainnet and expected_payout_script is None:
        print(
            "--expected-payout-script is required with --live-mainnet",
            file=sys.stderr,
        )
        return 2

    owned_tor: subprocess.Popen[bytes] | None = None
    session: MiningSession | None = None
    cuda_miner = CudaMiner(args.cuda_miner)

    def health_check() -> None:
        check_node_health(
            args.bitcoin_cli,
            args.datadir,
            args.bitcoin_conf,
            args.tor_host,
            args.tor_port,
            args.min_peers,
            args.require_onion_peers,
        )

    try:
        if args.network == "mainnet":
            keep_system_awake(True)
            print("NETWORK: MAINNET")
            if args.live_mainnet:
                print("MODE: LIVE (valid blocks WILL be submitted with submitblock)")
            else:
                print("MODE: DRY RUN (submitblock is never called)")
            owned_tor = preflight_mainnet(
                args.bitcoin_cli,
                args.datadir,
                args.bitcoin_conf,
                args.tor_host,
                args.tor_port,
                args.tor_executable,
                args.tor_startup_timeout,
                args.tor_poll_interval,
                args.require_onion_peers,
                args.min_peers,
            )
            payout_script = resolve_payout_script(
                args.bitcoin_cli,
                args.datadir,
                args.bitcoin_conf,
                args.payout_address,
                expected_payout_script,
            )
            if args.live_mainnet:
                if expected_payout_script is None or payout_script != expected_payout_script:
                    raise RuntimeError(
                        "Live mainnet requires a pinned payout script that "
                        "matches Bitcoin Core; CUDA mining was prevented"
                    )
                verify_wallet_ownership(
                    args.bitcoin_cli,
                    args.datadir,
                    args.bitcoin_conf,
                    args.payout_address,
                    payout_script,
                )
                print_live_preflight_summary(
                    args.payout_address,
                    payout_script,
                    expected_payout_script,
                )
            session = MiningSession(
                args.bitcoin_cli,
                args.datadir,
                args.bitcoin_conf,
                NodeMonitor(health_check, args.monitor_interval),
                TemplatePrefetcher(
                    lambda: fetch_work(
                        args.bitcoin_cli, "mainnet", args.datadir, args.bitcoin_conf
                    )
                ),
            )
            session.monitor.start()
        else:
            chain = rpc(
                args.bitcoin_cli,
                args.network,
                args.datadir,
                args.bitcoin_conf,
                "getblockchaininfo",
            )
            if not isinstance(chain, dict) or chain.get("chain") != "regtest":
                raise RuntimeError(
                    "Bitcoin Core RPC did not confirm the regtest chain; the "
                    "anyone-can-spend regtest payout was refused"
                )
            payout_script = REGTEST_PAYOUT_SCRIPT

        mined = 0
        extra_nonce = 0
        while args.dry_run or args.blocks == 0 or mined < args.blocks:
            try:
                block_mined = mine_one_block(
                    args.bitcoin_cli,
                    cuda_miner,
                    args.network,
                    args.datadir,
                    args.bitcoin_conf,
                    args.chunk_size,
                    extra_nonce,
                    payout_script,
                    dry_run=args.dry_run,
                    live_mainnet=args.live_mainnet,
                    payout_address=args.payout_address,
                    session=session,
                )
            except (MiningPaused, RpcError) as pause:
                # Regtest keeps failing fast; on mainnet no candidate is
                # pending here and no GPU work is in flight.
                if session is None:
                    raise
                print(f"[MONITOR] Mining paused: {pause}", flush=True)
                wait_for_recovery(health_check, args.recovery_timeout)
                session.reset()
                print(
                    "[MONITOR] Bitcoin Core and Tor answer again; resuming "
                    "with a fresh template.",
                    flush=True,
                )
                extra_nonce += 1
                continue
            if block_mined:
                mined += 1
                extra_nonce = 0
                if args.network == "mainnet":
                    print(
                        "Mining stopped after the accepted mainnet block; "
                        "inspect the result before restarting."
                    )
                    break
            else:
                extra_nonce += 1
        if session is not None and session.accepted_block is not None:
            # Mining is over; free the GPU and watch the block mature.
            cuda_miner.close()
            session.close()
            return monitor_block(
                args.bitcoin_cli,
                "mainnet",
                args.datadir,
                args.bitcoin_conf,
                session.accepted_block,
            )
        return 0
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 130
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        prefix = "Mainnet miner stopped: " if args.network == "mainnet" else ""
        print(f"{prefix}{error}", file=sys.stderr)
        return 1
    finally:
        try:
            cuda_miner.close()
        finally:
            if session is not None:
                session.close()
            stop_owned_tor(owned_tor)
            keep_system_awake(False)


if __name__ == "__main__":
    raise SystemExit(main())
