import argparse
import hashlib
import io
import random
import tempfile
import threading
import unittest
import struct
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from bitcoin import (
    bits_to_target,
    build_header,
    double_sha256,
    hash_meets_target,
    target_to_bits,
)
import regtest_miner
from regtest_miner import (
    CudaMiner,
    MiningPaused,
    MiningSession,
    NodeMonitor,
    RpcError,
    StaleTemplate,
    TemplatePrefetcher,
    check_node_health,
    fetch_work,
    monitor_block,
    save_unsubmitted_block,
    wait_for_recovery,
    build_header as build_template_header,
    create_coinbase,
    ensure_tor_ready,
    find_nonce,
    get_witness_commitment,
    main,
    mine_one_block,
    merkle_root,
    parse_args,
    parse_nonce_range,
    preflight_mainnet,
    read_compact_size,
    rpc,
    script_number,
    serialize_block,
    submission_permitted,
    submitted_block_status,
    template_transactions,
    transaction_hashes,
    transaction_outputs,
    socks5_ready,
    validate_template,
    validate_template_target,
    verify_candidate_block,
)


GENESIS_BITS = "1d00ffff"
GENESIS_TARGET = (
    "00000000ffff0000000000000000000000000000000000000000000000000000"
)
LEGACY_TRANSACTION = (
    "020000000111111111111111111111111111111111111111111111111111111111"
    "11111111010000000151ffffffff01e803000000000000015100000000"
)
WITNESS_TRANSACTION = (
    LEGACY_TRANSACTION[:8]
    + "0001"
    + LEGACY_TRANSACTION[8:-8]
    + "0102abcd"
    + LEGACY_TRANSACTION[-8:]
)
LEGACY_TXID = "7e4874381b1c2968a2494f6b46f1b46ac84519561a5a0b6bebd501c0ef80f0c2"
WITNESS_WTXID = "19a312fe7125dff8317e28015f97be565f8aa1d64b6f216b6c03341be7a595fe"
WITNESS_ROOT = "03a1a7f1a90c0346837509e5323f5385efc26547ad936ced50fa948e050e65e5"
WITNESS_COMMITMENT = (
    "6a24aa21a9ed9876be8d19e6c50d7952e4e66abba21f694440bcfbf09456661a212c510c8964"
)
COINBASE_TRANSACTION = (
    "02000000010000000000000000000000000000000000000000000000000000000000000000"
    "ffffffff09510000000000000000ffffffff0100f2052a01000000015100000000"
)
GENESIS_MERKLE_ROOT = (
    "4a5e1e4baab89f3a32518a88c31bc87f618f76673e2cc77ab2127b7afdeda33b"
)
GENESIS_HEADER = (
    "01000000"
    + "00" * 32
    + "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
    + "29ab5f49ffff001d1dac2b7c"
)
CUDA_MINER = Path(__file__).with_name("cuda_miner.exe")
# BIP173 P2WPKH test vector.
PAYOUT_ADDRESS = "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4"
PAYOUT_SCRIPT = "0014751e76e8199196d454941c45d1b3a323f1433bd6"


class TemplateTargetTests(unittest.TestCase):
    def test_accepts_matching_genesis_compact_bits_and_target(self) -> None:
        bits, target = validate_template_target(
            {"bits": GENESIS_BITS, "target": GENESIS_TARGET}
        )

        self.assertEqual(bits, 0x1D00FFFF)
        self.assertEqual(
            target,
            int(GENESIS_TARGET, 16),
        )

    def test_rejects_mismatched_gbt_target(self) -> None:
        mismatch = f"{int(GENESIS_TARGET, 16) - 1:064x}"

        with self.assertRaisesRegex(RuntimeError, "target mismatch"):
            validate_template_target({"bits": GENESIS_BITS, "target": mismatch})

    def test_rejects_malformed_bits(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "bits must be exactly"):
            validate_template_target(
                {"bits": "1d00fffg", "target": GENESIS_TARGET}
            )

    def test_rejects_malformed_target(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "target must be exactly"):
            validate_template_target(
                {"bits": GENESIS_BITS, "target": GENESIS_TARGET[:-1]}
            )


def valid_template() -> dict:
    return {
        "version": 1,
        "previousblockhash": "00" * 32,
        "height": 1,
        "bits": GENESIS_BITS,
        "target": GENESIS_TARGET,
        "curtime": 1_700_000_000,
        "mintime": 1_700_000_000,
        "noncerange": "00000000ffffffff",
        "mutable": ["time", "transactions"],
        "transactions": [],
        "coinbasevalue": 5_000_000_000,
    }


def easy_template() -> dict:
    template = valid_template()
    template["bits"] = "207fffff"
    template["target"] = f"{bits_to_target(0x207FFFFF):064x}"
    template["transactions"] = [
        {"data": LEGACY_TRANSACTION, "txid": LEGACY_TXID, "hash": LEGACY_TXID}
    ]
    return template


def cpu_find_nonce(_cuda_miner, header, bits, *_scan_arguments):
    target = bits_to_target(bits)
    for nonce in range(10_000):
        digest = double_sha256(header[:76] + struct.pack("<I", nonce))
        if int.from_bytes(digest[::-1], "big") <= target:
            return nonce, digest
    raise AssertionError("Could not find a valid test-vector nonce")


def patch_chunks(result_for):
    """Patch start_chunk/finish_chunk with a function of the pending request."""
    requests = []

    def start(cuda_miner, header, start, count):
        requests.append((cuda_miner, header, start, count))

    def finish(_cuda_miner):
        return result_for(*requests[-1])

    return (
        patch("regtest_miner.start_chunk", side_effect=start),
        patch("regtest_miner.finish_chunk", side_effect=finish),
    )


def miner_args(**overrides) -> argparse.Namespace:
    values = dict(
        network="mainnet",
        dry_run=False,
        live_mainnet=False,
        blocks=1,
        chunk_size=10,
        bitcoin_cli=Path(__file__),
        cuda_miner=Path(__file__),
        bitcoin_conf=None,
        datadir=None,
        payout_address=PAYOUT_ADDRESS,
        expected_payout_script=None,
        tor_host="127.0.0.1",
        tor_port=9150,
        tor_executable=None,
        tor_startup_timeout=1.0,
        tor_poll_interval=0.01,
        require_onion_peers=False,
        min_peers=1,
        monitor_interval=15.0,
        recovery_timeout=3600.0,
        monitor_block=None,
        version_rolling=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def live_args(**overrides) -> argparse.Namespace:
    values = dict(live_mainnet=True, expected_payout_script=PAYOUT_SCRIPT)
    values.update(overrides)
    return miner_args(**values)


class FakeCore:
    """In-memory stand-in for the rpc() function; never reaches a real node."""

    def __init__(self, template=None, chain="main") -> None:
        self.template = template if template is not None else easy_template()
        self.chain_info = {
            "chain": chain,
            "initialblockdownload": False,
            "blocks": 100,
            "headers": 100,
            "bestblockhash": self.template["previousblockhash"],
        }
        self.address_info = {"isvalid": True, "scriptPubKey": PAYOUT_SCRIPT}
        self.wallet_info: object = {
            "address": PAYOUT_ADDRESS,
            "scriptPubKey": PAYOUT_SCRIPT,
            "ismine": True,
            "desc": "wpkh([d34db33f/84h/0h/0h/0/0]02aa)#secretlooking",
            "hdkeypath": "m/84h/0h/0h/0/0",
        }
        self.wallet_error: Exception | None = None
        self.tips: list[str] = []
        self.submit_result: str | None = None
        self.submit_error: Exception | None = None
        # What getblockheader reports for a submitted block; None = unknown block.
        self.header_confirmations: int | None = None
        self.methods: list[str] = []
        self.networks: set[str] = set()
        self.submitted: list[str] = []

    def __call__(self, _cli, network, _datadir, _conf, method, *params, **_kwargs):
        self.methods.append(method)
        self.networks.add(network)
        if method == "getblockchaininfo":
            return self.chain_info
        if method == "getnetworkinfo":
            return {"networkactive": True, "connections": 8}
        if method == "validateaddress":
            return self.address_info
        if method == "getaddressinfo":
            if self.wallet_error is not None:
                raise self.wallet_error
            return self.wallet_info
        if method == "getblocktemplate":
            return self.template
        if method == "getbestblockhash":
            if self.tips:
                return self.tips.pop(0)
            return self.template["previousblockhash"]
        if method == "submitblock":
            self.submitted.append(params[0])
            if self.submit_error is not None:
                raise self.submit_error
            return self.submit_result
        if method == "getblockcount":
            return self.template["height"]
        if method == "getblockhash":
            header = bytes.fromhex(self.submitted[-1][:160])
            return double_sha256(header)[::-1].hex()
        if method == "getblockheader":
            if self.header_confirmations is None:
                raise RpcError(
                    "Bitcoin Core RPC getblockheader failed: error code: -5 Block not found"
                )
            return {
                "hash": params[0],
                "height": self.template["height"],
                "confirmations": self.header_confirmations,
            }
        raise AssertionError(f"Unexpected RPC method: {method}")


class TorStartupTests(unittest.TestCase):
    def test_socks5_probe_negotiates_no_authentication(self) -> None:
        connection = MagicMock()
        connection.__enter__.return_value.recv.return_value = b"\x05\x00"
        with patch("regtest_miner.socket.create_connection", return_value=connection):
            self.assertTrue(socks5_ready("127.0.0.1", 9150))

        connection.__enter__.return_value.sendall.assert_called_once_with(
            b"\x05\x01\x00"
        )

    def test_existing_tor_socks_endpoint_is_reused(self) -> None:
        with (
            patch("regtest_miner.socks5_ready", return_value=True),
            patch("regtest_miner.subprocess.Popen") as popen,
            redirect_stdout(io.StringIO()),
        ):
            owned = ensure_tor_ready(
                "127.0.0.1", 9150, Path(__file__), 1.0, 0.01
            )

        self.assertIsNone(owned)
        popen.assert_not_called()

    def test_absent_tor_is_started_and_waited_for(self) -> None:
        process = MagicMock()
        process.poll.return_value = None
        with (
            patch("regtest_miner.socks5_ready", side_effect=[False, True]),
            patch("regtest_miner.subprocess.Popen", return_value=process) as popen,
            redirect_stdout(io.StringIO()),
        ):
            owned = ensure_tor_ready(
                "127.0.0.1", 9150, Path(__file__), 1.0, 0.01
            )

        self.assertIs(owned, process)
        self.assertEqual(
            popen.call_args.args[0],
            [str(Path(__file__)), "--SocksPort", "127.0.0.1:9150"],
        )

    def test_tor_startup_timeout_stops_owned_process(self) -> None:
        process = MagicMock()
        process.poll.return_value = None
        with (
            patch("regtest_miner.socks5_ready", return_value=False),
            patch("regtest_miner.subprocess.Popen", return_value=process),
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(RuntimeError, "CUDA mining was prevented"):
                ensure_tor_ready(
                    "127.0.0.1", 9150, Path(__file__), 0.01, 0.001
                )

        process.terminate.assert_called_once()


class StartupPreflightTests(unittest.TestCase):
    def run_preflight(self, rpc_side_effect, require_onion_peers=False):
        with patch("regtest_miner.ensure_tor_ready", return_value=None), patch(
            "regtest_miner.rpc", side_effect=rpc_side_effect
        ) as rpc_mock:
            result = preflight_mainnet(
                Path("bitcoin-cli.exe"),
                None,
                None,
                "127.0.0.1",
                9150,
                None,
                1.0,
                0.01,
                require_onion_peers,
            )
        return result, rpc_mock

    def test_bitcoin_core_rpc_unavailable_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "RPC unavailable"):
            self.run_preflight([RuntimeError("RPC unavailable")])

    def test_initial_block_download_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "initial block download"):
            self.run_preflight(
                [{"chain": "main", "initialblockdownload": True}]
            )

    def test_no_connected_peers_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no connected peers"):
            self.run_preflight(
                [
                    {
                        "chain": "main",
                        "initialblockdownload": False,
                        "blocks": 100,
                        "headers": 100,
                    },
                    {
                        "networkactive": True,
                        "connections": 0,
                    },
                ]
            )

    def test_tor_only_mode_requires_an_onion_peer(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no connected onion peer"):
            self.run_preflight(
                [
                    {
                        "chain": "main",
                        "initialblockdownload": False,
                        "blocks": 100,
                        "headers": 100,
                    },
                    {
                        "networkactive": True,
                        "connections": 1,
                    },
                    [],
                ],
                require_onion_peers=True,
            )

    def test_stale_template_is_discarded_before_cuda(self) -> None:
        with (
            patch(
                "regtest_miner.rpc",
                side_effect=[valid_template(), "ff" * 32],
            ),
            patch("regtest_miner.find_nonce") as find_nonce,
            redirect_stdout(io.StringIO()),
        ):
            result = mine_one_block(
                Path("bitcoin-cli.exe"),
                Path("cuda_miner.exe"),
                "mainnet",
                None,
                None,
                1,
                0,
                b"\x51",
                dry_run=True,
            )

        self.assertFalse(result)
        find_nonce.assert_not_called()

    def test_successful_preflight_reaches_cuda_chunk_launch(self) -> None:
        args = miner_args(dry_run=True, blocks=0, chunk_size=1)
        template = valid_template()
        template["noncerange"] = "0000000000000000"
        rpc_methods = []

        def fake_rpc(_cli, _network, _datadir, _conf, method, *_params, **_kwargs):
            rpc_methods.append(method)
            if method == "getblockchaininfo":
                return {
                    "chain": "main",
                    "initialblockdownload": False,
                    "blocks": 100,
                    "headers": 100,
                    "bestblockhash": template["previousblockhash"],
                }
            if method == "getnetworkinfo":
                return {"networkactive": True, "connections": 1}
            if method == "validateaddress":
                return {"isvalid": True, "scriptPubKey": PAYOUT_SCRIPT}
            if method == "getblocktemplate":
                return template
            if method == "getbestblockhash":
                return template["previousblockhash"]
            self.fail(f"Unexpected RPC method: {method}")

        mine_calls = 0

        def mine_once_then_interrupt(*call_args, **call_kwargs):
            nonlocal mine_calls
            mine_calls += 1
            if mine_calls > 1:
                raise KeyboardInterrupt
            return mine_one_block(*call_args, **call_kwargs)

        # The one-nonce range is exhausted at once; the same template is then
        # reused with a rolled extranonce, where the interrupt arrives.
        with (
            patch("regtest_miner.parse_args", return_value=args),
            patch("regtest_miner.ensure_tor_ready", return_value=None),
            patch("regtest_miner.rpc", side_effect=fake_rpc),
            patch("regtest_miner.mine_one_block", side_effect=mine_once_then_interrupt),
            patch("regtest_miner.start_chunk") as mine_chunk,
            patch("regtest_miner.finish_chunk", side_effect=[None, KeyboardInterrupt]),
            redirect_stdout(io.StringIO()),
        ):
            result = main()

        self.assertEqual(result, 130)
        self.assertEqual(mine_chunk.call_count, 2)
        self.assertEqual(rpc_methods.count("getblocktemplate"), 1)
        first_header, second_header = (call.args[1] for call in mine_chunk.call_args_list)
        self.assertEqual(first_header[:36], second_header[:36])
        self.assertNotEqual(first_header[36:68], second_header[36:68])
        self.assertEqual(
            rpc_methods[:4],
            [
                "getblockchaininfo",
                "getnetworkinfo",
                "validateaddress",
                "getblocktemplate",
            ],
        )
        self.assertNotIn("submitblock", rpc_methods)


class TemplateValidationTests(unittest.TestCase):
    def test_default_chunk_size(self) -> None:
        with patch("sys.argv", ["regtest_miner.py"]):
            args = parse_args()

        self.assertEqual(args.chunk_size, 250_000_000)

    def test_mainnet_dry_run_cli_forms(self) -> None:
        for argv in (
            [
                "regtest_miner.py",
                "--mainnet",
                "--dry-run",
                "--payout-address",
                "bc1qexample",
            ],
            [
                "regtest_miner.py",
                "--network",
                "mainnet",
                "--dry-run",
                "--payout-address",
                "bc1qexample",
            ],
        ):
            with self.subTest(argv=argv), patch("sys.argv", argv):
                args = parse_args()
                self.assertEqual(args.network, "mainnet")
                self.assertTrue(args.dry_run)

    def test_mainnet_alone_remains_gated(self) -> None:
        error_output = io.StringIO()
        with (
            patch("regtest_miner.parse_args", return_value=miner_args()),
            patch("regtest_miner.rpc") as rpc_mock,
            patch("regtest_miner.start_chunk") as mine_chunk,
            redirect_stderr(error_output),
        ):
            result = main()

        self.assertEqual(result, 2)
        self.assertIn("gated", error_output.getvalue())
        rpc_mock.assert_not_called()
        mine_chunk.assert_not_called()

    def test_mainnet_dry_run_rejects_unsynchronized_core(self) -> None:
        args = miner_args(dry_run=True, blocks=0)
        error_output = io.StringIO()
        with (
            patch("regtest_miner.parse_args", return_value=args),
            patch("regtest_miner.ensure_tor_ready", return_value=None),
            patch(
                "regtest_miner.rpc",
                return_value={
                    "chain": "main",
                    "initialblockdownload": False,
                    "blocks": 100,
                    "headers": 101,
                },
            ),
            redirect_stderr(error_output),
            redirect_stdout(io.StringIO()),
        ):
            result = main()

        self.assertEqual(result, 1)
        self.assertIn("not synchronized", error_output.getvalue())

    def test_dry_run_candidate_is_never_submitted(self) -> None:
        template = valid_template()
        template["bits"] = "207fffff"
        template["target"] = f"{bits_to_target(0x207FFFFF):064x}"

        def regtest_candidate(
            _cuda_miner: Path,
            header: bytes,
            bits: int,
            _chunk_size: int,
            _nonce_start: int,
            _nonce_end: int,
            _stale_check: object,
        ) -> tuple[int, bytes]:
            target = bits_to_target(bits)
            for nonce in range(10_000):
                nonce_header = header[:76] + struct.pack("<I", nonce)
                digest = double_sha256(nonce_header)
                if int.from_bytes(digest[::-1], "big") <= target:
                    return nonce, digest
            self.fail("Could not find a valid regtest test-vector nonce")

        with (
            patch(
                "regtest_miner.rpc",
                side_effect=[
                    template,
                    template["previousblockhash"],
                    template["previousblockhash"],
                ],
            ) as rpc_mock,
            patch("regtest_miner.find_nonce", side_effect=regtest_candidate),
            redirect_stdout(io.StringIO()),
        ):
            result = mine_one_block(
                Path("bitcoin-cli.exe"),
                Path("cuda_miner.exe"),
                "regtest",
                None,
                None,
                10,
                0,
                b"\x51",
                dry_run=True,
            )

        self.assertFalse(result)
        self.assertNotIn(
            "submitblock",
            [call.args[4] for call in rpc_mock.call_args_list],
        )

    def test_accepts_valid_template_and_returns_inclusive_nonce_bounds(self) -> None:
        bits, target, nonce_start, nonce_end = validate_template(valid_template())

        self.assertEqual(bits, 0x1D00FFFF)
        self.assertEqual(target, int(GENESIS_TARGET, 16))
        self.assertEqual((nonce_start, nonce_end), (0, 0xFFFFFFFF))

    def test_rejects_invalid_required_template_fields(self) -> None:
        invalid_values = (
            ("version", True),
            ("previousblockhash", "00" * 31),
            ("height", -1),
            ("coinbasevalue", -1),
            ("curtime", 1_699_999_999),
            ("mintime", "1700000000"),
            ("mutable", ["time", 1]),
            ("transactions", {}),
        )
        for field, invalid_value in invalid_values:
            with self.subTest(field=field):
                template = valid_template()
                template[field] = invalid_value
                with self.assertRaises(RuntimeError):
                    validate_template(template)

    def test_rejects_missing_required_fields(self) -> None:
        template = valid_template()
        del template["noncerange"]

        with self.assertRaisesRegex(RuntimeError, "noncerange"):
            validate_template(template)

    def test_rejects_invalid_nonce_range(self) -> None:
        for noncerange in ("00000000", "00000010ffffffff0", "ffffffff00000000"):
            with self.subTest(noncerange=noncerange):
                with self.assertRaises(RuntimeError):
                    parse_nonce_range(noncerange)

    def test_requires_commitment_for_witness_transaction(self) -> None:
        template = valid_template()
        template["transactions"] = [
            {
                "data": "020000000001",
                "txid": "00" * 32,
                "hash": "00" * 32,
            }
        ]

        with self.assertRaisesRegex(RuntimeError, "missing its witness commitment"):
            validate_template(template)

    def test_nonce_scan_covers_only_template_range_in_chunks(self) -> None:
        scanned = []

        def record_scan(cuda_miner, header, start, count):
            self.assertEqual(cuda_miner, "cuda_miner.exe")
            self.assertEqual(header, bytes(80))
            scanned.append((start, count))
            return None

        with (
            patch("regtest_miner.start_chunk", side_effect=record_scan),
            patch("regtest_miner.finish_chunk", return_value=None),
            redirect_stdout(io.StringIO()),
        ):
            result = find_nonce(
                "cuda_miner.exe",
                bytes(80),
                0x1D00FFFF,
                2,
                0x10,
                0x13,
                lambda: False,
            )

        self.assertIsNone(result)
        self.assertEqual(scanned, [(0x10, 2), (0x12, 2)])


class LiveMainnetTests(unittest.TestCase):
    def mine(self, core, network="mainnet", payout_script=None, **kwargs):
        output = io.StringIO()
        script = bytes.fromhex(PAYOUT_SCRIPT) if payout_script is None else payout_script
        with (
            patch("regtest_miner.rpc", side_effect=core),
            patch(
                "regtest_miner.find_nonce",
                side_effect=kwargs.pop("find_nonce", cpu_find_nonce),
            ),
            patch("regtest_miner.save_unsubmitted_block", return_value=None) as save,
            patch("regtest_miner.time.sleep"),
            redirect_stdout(output),
        ):
            self.save_mock = save
            self.output = output
            return mine_one_block(
                Path("bitcoin-cli.exe"),
                Path("cuda_miner.exe"),
                network,
                None,
                None,
                10,
                0,
                script,
                **kwargs,
            )

    def run_main(self, core, args):
        self.output = io.StringIO()
        self.errors = io.StringIO()
        with (
            patch("regtest_miner.parse_args", return_value=args),
            patch("regtest_miner.ensure_tor_ready", return_value=None),
            patch("regtest_miner.rpc", side_effect=core),
            patch(
                "regtest_miner.find_nonce", side_effect=cpu_find_nonce
            ) as self.find_nonce,
            patch("regtest_miner.start_chunk") as self.mine_chunk,
            patch("regtest_miner.subprocess.Popen") as self.popen,
            patch("regtest_miner.save_unsubmitted_block", return_value=None),
            patch("regtest_miner.monitor_block", return_value=0) as self.monitor_block,
            patch("regtest_miner.time.sleep"),
            redirect_stdout(self.output),
            redirect_stderr(self.errors),
        ):
            return main()

    def test_submission_gate_requires_mainnet_acknowledgement(self) -> None:
        self.assertFalse(submission_permitted("mainnet", False, False))
        self.assertFalse(submission_permitted("mainnet", True, False))
        self.assertFalse(submission_permitted("mainnet", True, True))
        self.assertFalse(submission_permitted("regtest", True, False))
        self.assertFalse(submission_permitted("testnet", False, True))
        self.assertTrue(submission_permitted("mainnet", False, True))
        self.assertTrue(submission_permitted("regtest", False, False))

    def test_live_flag_cli_parsing_and_conflicts(self) -> None:
        with patch(
            "sys.argv",
            [
                "regtest_miner.py",
                "--mainnet",
                "--live-mainnet",
                "--payout-address",
                PAYOUT_ADDRESS,
                "--expected-payout-script",
                PAYOUT_SCRIPT,
            ],
        ):
            args = parse_args()
        self.assertTrue(args.live_mainnet)
        self.assertFalse(args.dry_run)
        self.assertEqual(args.expected_payout_script, PAYOUT_SCRIPT)

        with patch("sys.argv", ["regtest_miner.py", "--mainnet"]):
            self.assertFalse(parse_args().live_mainnet)

        for overrides in (
            dict(dry_run=True),
            dict(network="regtest", payout_address=None),
            dict(payout_address=None),
            dict(expected_payout_script="zz"),
            dict(expected_payout_script=""),
            dict(expected_payout_script=None),
            dict(
                live_mainnet=False,
                network="regtest",
                payout_address=None,
                expected_payout_script="51",
            ),
        ):
            with self.subTest(overrides=overrides):
                core = FakeCore()
                self.assertEqual(self.run_main(core, live_args(**overrides)), 2)
                self.assertEqual(core.methods, [])
                self.find_nonce.assert_not_called()
                self.mine_chunk.assert_not_called()

    def test_live_mainnet_requires_expected_payout_script(self) -> None:
        core = FakeCore()
        result = self.run_main(core, live_args(expected_payout_script=None))

        self.assertEqual(result, 2)
        self.assertIn(
            "--expected-payout-script is required with --live-mainnet",
            self.errors.getvalue(),
        )
        self.assertEqual(core.methods, [])
        self.find_nonce.assert_not_called()

    def test_wallet_ownership_failures_abort_before_cuda(self) -> None:
        other_script = "0014" + "11" * 20
        cases = {
            "ismine false": dict(wallet_info={"ismine": False, "scriptPubKey": PAYOUT_SCRIPT}),
            "ismine missing": dict(wallet_info={"scriptPubKey": PAYOUT_SCRIPT}),
            "ismine not boolean": dict(wallet_info={"ismine": "true", "scriptPubKey": PAYOUT_SCRIPT}),
            "watch-only": dict(
                wallet_info={"ismine": False, "iswatchonly": True, "scriptPubKey": PAYOUT_SCRIPT}
            ),
            "wallet script mismatch": dict(wallet_info={"ismine": True, "scriptPubKey": other_script}),
            "wallet script missing": dict(wallet_info={"ismine": True}),
            "unexpected response": dict(wallet_info=None),
            "no wallet loaded": dict(
                wallet_error=RuntimeError(
                    "Bitcoin Core RPC getaddressinfo failed: error code: -18 "
                    "No wallet is loaded."
                )
            ),
            "wallet not specified": dict(
                wallet_error=RuntimeError(
                    "Bitcoin Core RPC getaddressinfo failed: error code: -19 "
                    "Wallet file not specified"
                )
            ),
            "bitcoin-cli unavailable": dict(wallet_error=OSError("cannot start")),
        }
        for label, settings in cases.items():
            with self.subTest(label=label):
                core = FakeCore()
                for name, value in settings.items():
                    setattr(core, name, value)
                result = self.run_main(core, live_args())

                self.assertEqual(result, 1)
                self.assertIn("Wallet ownership check failed", self.errors.getvalue())
                self.assertEqual(core.methods[-1], "getaddressinfo")
                self.assertNotIn("getblocktemplate", core.methods)
                self.assertNotIn("submitblock", core.methods)
                self.assertNotIn("LIVE MAINNET PREFLIGHT", self.output.getvalue())
                self.find_nonce.assert_not_called()
                self.mine_chunk.assert_not_called()

    def test_no_cuda_work_starts_for_any_failed_live_preflight(self) -> None:
        def broken(**changes):
            core = FakeCore()
            for name, value in changes.items():
                if name in core.chain_info:
                    core.chain_info[name] = value
                else:
                    setattr(core, name, value)
            return core

        no_peers = FakeCore()
        original_call = no_peers.__call__

        def without_peers(cli, network, datadir, conf, method, *params, **kwargs):
            result = original_call(cli, network, datadir, conf, method, *params, **kwargs)
            if method == "getnetworkinfo":
                return {"networkactive": True, "connections": 0}
            return result

        cases = {
            "wrong chain": (broken(chain="test"), live_args()),
            "initial block download": (broken(initialblockdownload=True), live_args()),
            "headers ahead": (broken(headers=101), live_args()),
            "no best block": (broken(bestblockhash=None), live_args()),
            "no peers": (without_peers, live_args()),
            "invalid address": (broken(address_info={"isvalid": False}), live_args()),
            "script mismatch": (
                FakeCore(),
                live_args(expected_payout_script="0014" + "22" * 20),
            ),
            "not mine": (
                broken(wallet_info={"ismine": False, "scriptPubKey": PAYOUT_SCRIPT}),
                live_args(),
            ),
            "wallet rpc failure": (
                broken(wallet_error=RuntimeError("No wallet is loaded.")),
                live_args(),
            ),
            "wallet script mismatch": (
                broken(wallet_info={"ismine": True, "scriptPubKey": "51"}),
                live_args(),
            ),
        }
        for label, (core, args) in cases.items():
            with self.subTest(label=label):
                result = self.run_main(core, args)

                self.assertEqual(result, 1)
                self.assertNotIn("LIVE MAINNET PREFLIGHT: PASS", self.output.getvalue())
                self.find_nonce.assert_not_called()
                self.mine_chunk.assert_not_called()
                self.popen.assert_not_called()
                methods = no_peers.methods if core is without_peers else core.methods
                self.assertNotIn("getblocktemplate", methods)
                self.assertNotIn("submitblock", methods)

    def test_tor_unavailable_aborts_live_mainnet_before_any_rpc(self) -> None:
        core = FakeCore()
        errors = io.StringIO()
        with (
            patch("regtest_miner.parse_args", return_value=live_args()),
            patch("regtest_miner.socks5_ready", return_value=False),
            patch("regtest_miner.rpc", side_effect=core),
            patch("regtest_miner.find_nonce") as find_nonce_mock,
            patch("regtest_miner.start_chunk") as mine_chunk_mock,
            redirect_stdout(io.StringIO()),
            redirect_stderr(errors),
        ):
            result = main()

        self.assertEqual(result, 1)
        self.assertIn("Tor SOCKS5 is unavailable", errors.getvalue())
        self.assertEqual(core.methods, [])
        find_nonce_mock.assert_not_called()
        mine_chunk_mock.assert_not_called()

    def test_successful_live_preflight_prints_summary_then_mines(self) -> None:
        core = FakeCore()
        result = self.run_main(core, live_args())

        self.assertEqual(result, 0)
        self.assertEqual(
            core.methods[:5],
            [
                "getblockchaininfo",
                "getnetworkinfo",
                "validateaddress",
                "getaddressinfo",
                "getblocktemplate",
            ],
        )
        self.find_nonce.assert_called_once()
        output = self.output.getvalue()
        summary = (
            "NETWORK: MAINNET\n"
            "MODE: LIVE\n"
            "CHAIN CHECK: PASS\n"
            "SYNC CHECK: PASS\n"
            "PEER CHECK: PASS\n"
            f"PAYOUT ADDRESS: {PAYOUT_ADDRESS}\n"
            f"PAYOUT SCRIPT: {PAYOUT_SCRIPT}\n"
            f"EXPECTED PAYOUT SCRIPT: {PAYOUT_SCRIPT}\n"
            "SCRIPT MATCH: PASS\n"
            "WALLET OWNERSHIP: PASS\n"
            "SUBMISSION TRANSPORT: bitcoin-cli -stdin\n"
            "LIVE MAINNET PREFLIGHT: PASS\n"
        )
        self.assertIn(summary, output)
        self.assertLess(output.index(summary), output.index("Mining mainnet block"))
        for secret in ("secretlooking", "hdkeypath", "d34db33f", "wpkh("):
            self.assertNotIn(secret, output + self.errors.getvalue())

    def test_dry_run_and_regtest_do_not_require_the_wallet(self) -> None:
        core = FakeCore()
        core.wallet_error = AssertionError("dry-run must not query the wallet")

        def rpc_with_interrupt(cli, network, datadir, conf, method, *params, **kwargs):
            if method == "getblocktemplate" and "getblocktemplate" in core.methods:
                raise KeyboardInterrupt
            return core(cli, network, datadir, conf, method, *params, **kwargs)

        self.assertEqual(
            self.run_main(rpc_with_interrupt, miner_args(dry_run=True)), 130
        )
        self.assertNotIn("getaddressinfo", core.methods)
        self.assertNotIn("submitblock", core.methods)
        self.assertNotIn("LIVE MAINNET PREFLIGHT", self.output.getvalue())

        regtest = FakeCore(chain="regtest")
        regtest.wallet_error = AssertionError("regtest must not query the wallet")
        args = miner_args(network="regtest", payout_address=None)
        self.assertEqual(self.run_main(regtest, args), 0)
        self.assertNotIn("getaddressinfo", regtest.methods)
        self.assertEqual(regtest.methods.count("submitblock"), 1)

    def test_mainnet_candidate_without_live_flag_is_not_submitted(self) -> None:
        core = FakeCore()
        with self.assertRaisesRegex(RuntimeError, "requires --live-mainnet"):
            self.mine(core)

        self.assertNotIn("submitblock", core.methods)

    def test_mainnet_dry_run_candidate_is_never_submitted(self) -> None:
        core = FakeCore()
        self.assertFalse(self.mine(core, dry_run=True))
        self.assertFalse(self.mine(core, dry_run=True, live_mainnet=True))

        self.assertNotIn("submitblock", core.methods)
        self.assertIn("DRY RUN", self.output.getvalue())

    def test_main_dry_run_verifies_candidates_without_submitting(self) -> None:
        core = FakeCore()

        def rpc_with_interrupt(cli, network, datadir, conf, method, *params, **kwargs):
            if method == "getblocktemplate" and "getblocktemplate" in core.methods:
                raise KeyboardInterrupt
            return core(cli, network, datadir, conf, method, *params, **kwargs)

        result = self.run_main(
            rpc_with_interrupt,
            miner_args(dry_run=True, expected_payout_script=PAYOUT_SCRIPT),
        )

        self.assertEqual(result, 130)
        self.assertNotIn("submitblock", core.methods)
        self.assertIn("MODE: DRY RUN", self.output.getvalue())
        self.assertIn("DRY RUN: candidate", self.output.getvalue())

    def test_invalid_payout_address_aborts_before_mining(self) -> None:
        for address_info in (
            {"isvalid": False},
            {"isvalid": True},
            {"isvalid": True, "scriptPubKey": ""},
            {"isvalid": True, "scriptPubKey": "zz"},
            {"isvalid": True, "scriptPubKey": "51"},
            None,
        ):
            with self.subTest(address_info=address_info):
                core = FakeCore()
                core.address_info = address_info
                result = self.run_main(core, live_args())

                self.assertEqual(result, 1)
                self.assertNotIn("getblocktemplate", core.methods)
                self.assertNotIn("submitblock", core.methods)
                self.mine_chunk.assert_not_called()

    def test_payout_script_mismatch_aborts_before_mining(self) -> None:
        core = FakeCore()
        result = self.run_main(
            core,
            miner_args(
                live_mainnet=True,
                expected_payout_script=PAYOUT_SCRIPT[:-2] + "d7",
            ),
        )

        self.assertEqual(result, 1)
        self.assertIn("Payout script mismatch", self.errors.getvalue())
        self.assertIn("PAYOUT SCRIPT MATCH: NO", self.output.getvalue())
        self.assertNotIn("getblocktemplate", core.methods)
        self.assertNotIn("submitblock", core.methods)

    def test_wrong_chain_aborts_before_mining(self) -> None:
        for chain in ("test", "regtest", "signet", None):
            with self.subTest(chain=chain):
                core = FakeCore(chain=chain)
                result = self.run_main(core, live_args())

                self.assertEqual(result, 1)
                self.assertIn("main chain", self.errors.getvalue())
                self.assertEqual(core.methods, ["getblockchaininfo"])

    def test_initial_block_download_aborts_before_mining(self) -> None:
        core = FakeCore()
        core.chain_info["initialblockdownload"] = True
        result = self.run_main(core, live_args())

        self.assertEqual(result, 1)
        self.assertIn("initial block download", self.errors.getvalue())
        self.assertEqual(core.methods, ["getblockchaininfo"])

    def test_unreachable_core_aborts_before_mining(self) -> None:
        def unreachable(*_arguments, **_kwargs):
            raise RuntimeError("Bitcoin Core RPC getblockchaininfo failed")

        result = self.run_main(unreachable, live_args())

        self.assertEqual(result, 1)
        self.mine_chunk.assert_not_called()

    def test_no_tip_check_delays_submission_and_core_decides_staleness(self) -> None:
        core = FakeCore()
        # A new tip that a pre-submission check would have seen. Core is the
        # judge instead: it stores the block as a competitor.
        core.tips = [core.template["previousblockhash"], "ff" * 32]
        core.submit_result = "inconclusive"

        self.assertFalse(self.mine(core, live_mainnet=True))
        self.assertEqual(
            core.methods, ["getblocktemplate", "getbestblockhash", "submitblock"]
        )
        output = self.output.getvalue()
        self.assertIn("NOT ON ACTIVE CHAIN", output)
        self.assertIn("does not relay", output)
        self.assertNotIn("REJECTED", output)
        self.assertNotIn("ACCEPTED", output)

    def test_verified_mainnet_candidate_is_saved_before_any_further_rpc(self) -> None:
        for mode in (dict(live_mainnet=True), dict(dry_run=True)):
            with self.subTest(mode=mode):
                core = FakeCore()
                seen_before_save = []
                output = io.StringIO()
                with (
                    patch("regtest_miner.rpc", side_effect=core),
                    patch("regtest_miner.find_nonce", side_effect=cpu_find_nonce),
                    patch(
                        "regtest_miner.save_unsubmitted_block",
                        side_effect=lambda *_a: seen_before_save.append(list(core.methods)),
                    ) as save,
                    redirect_stdout(output),
                ):
                    mine_one_block(
                        Path("bitcoin-cli.exe"), Path("cuda_miner.exe"), "mainnet",
                        None, None, 10, 0, bytes.fromhex(PAYOUT_SCRIPT), **mode,
                    )

                save.assert_called_once()
                self.assertEqual(
                    seen_before_save, [["getblocktemplate", "getbestblockhash"]]
                )
                saved_hash, saved_block = save.call_args.args
                self.assertEqual(saved_hash, double_sha256(saved_block[:80])[::-1].hex())
                self.assertEqual(
                    transaction_outputs(saved_block[81 : len(saved_block) - len(bytes.fromhex(LEGACY_TRANSACTION))]),
                    [(5_000_000_000, bytes.fromhex(PAYOUT_SCRIPT))],
                )
                if "dry_run" in mode:
                    self.assertNotIn("submitblock", core.methods)
                    self.assertIn("DRY RUN", output.getvalue())
                else:
                    self.assertEqual(core.submitted, [saved_block.hex()])

    def test_regtest_blocks_are_not_saved_when_accepted(self) -> None:
        core = FakeCore(chain="regtest")
        self.assertTrue(self.mine(core, network="regtest", payout_script=b"\x51"))
        self.save_mock.assert_not_called()

    def test_block_file_is_written_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            with patch("regtest_miner.__file__", str(Path(folder) / "regtest_miner.py")):
                saved = save_unsubmitted_block("ab" * 32, b"\x01\x02\xff")
                self.assertEqual(saved, Path(folder) / f"unsubmitted_block_{'ab' * 32}.hex")
                self.assertEqual(saved.read_text(encoding="ascii"), "0102ff")
                self.assertEqual([path.name for path in Path(folder).iterdir()], [saved.name])
                regtest_miner.flush_saved_block(saved)

                with patch("regtest_miner.os.replace", side_effect=OSError("disk full")):
                    self.assertIsNone(save_unsubmitted_block("cd" * 32, b"\x03"))
                # A failed save leaves neither a partial block nor a temporary file.
                self.assertEqual([path.name for path in Path(folder).iterdir()], [saved.name])

    def test_cpu_invalid_cuda_candidate_prevents_submission(self) -> None:
        template = easy_template()
        template["bits"] = GENESIS_BITS
        template["target"] = GENESIS_TARGET
        core = FakeCore(template)
        with (
            patch("regtest_miner.rpc", side_effect=core),
            patch("regtest_miner.start_chunk"),
            patch("regtest_miner.finish_chunk", return_value=(3, "00" * 32)),
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(RuntimeError, "does not match the CPU"):
                mine_one_block(
                    Path("bitcoin-cli.exe"),
                    Path("cuda_miner.exe"),
                    "mainnet",
                    None,
                    None,
                    10,
                    0,
                    bytes.fromhex(PAYOUT_SCRIPT),
                    live_mainnet=True,
                )
        self.assertNotIn("submitblock", core.methods)

        def consistent_but_above_target(_cuda_miner, header, start, _count):
            digest = double_sha256(header[:76] + struct.pack("<I", start))
            return start, digest[::-1].hex()

        core = FakeCore(template)
        start_patch, finish_patch = patch_chunks(consistent_but_above_target)
        with (
            patch("regtest_miner.rpc", side_effect=core),
            start_patch,
            finish_patch,
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(RuntimeError, "does not meet the target"):
                mine_one_block(
                    Path("bitcoin-cli.exe"),
                    Path("cuda_miner.exe"),
                    "mainnet",
                    None,
                    None,
                    10,
                    0,
                    bytes.fromhex(PAYOUT_SCRIPT),
                    live_mainnet=True,
                )
        self.assertNotIn("submitblock", core.methods)

    def test_pre_submission_check_rejects_header_above_target(self) -> None:
        template = easy_template()
        template["bits"] = GENESIS_BITS
        template["target"] = GENESIS_TARGET
        core = FakeCore(template)

        def unverified_nonce(_cuda_miner, header, *_scan_arguments):
            return 0, double_sha256(header[:76] + bytes(4))

        with self.assertRaisesRegex(RuntimeError, "does not meet the template target"):
            self.mine(core, live_mainnet=True, find_nonce=unverified_nonce)
        self.assertNotIn("submitblock", core.methods)

    def test_valid_candidate_is_submitted_and_accepted(self) -> None:
        core = FakeCore()

        self.assertTrue(
            self.mine(core, live_mainnet=True, payout_address=PAYOUT_ADDRESS)
        )
        self.assertEqual(core.methods.count("submitblock"), 1)
        self.assertEqual(core.networks, {"mainnet"})

        block = bytes.fromhex(core.submitted[0])
        header = block[:80]
        self.assertLessEqual(
            int.from_bytes(double_sha256(header)[::-1], "big"),
            bits_to_target(0x207FFFFF),
        )
        legacy = bytes.fromhex(LEGACY_TRANSACTION)
        self.assertEqual(block[80], 2)
        self.assertTrue(block.endswith(legacy))
        coinbase = block[81 : len(block) - len(legacy)]
        self.assertEqual(
            transaction_outputs(coinbase),
            [(5_000_000_000, bytes.fromhex(PAYOUT_SCRIPT))],
        )
        coinbase_txid = double_sha256(coinbase)
        self.assertEqual(
            header[36:68],
            merkle_root([coinbase_txid, bytes.fromhex(LEGACY_TXID)[::-1]]),
        )

        output = self.output.getvalue()
        block_hash = double_sha256(header)[::-1].hex()
        self.assertIn("MAINNET BLOCK ACCEPTED", output)
        self.assertIn(f"BLOCK HASH: {block_hash}", output)
        self.assertIn("BLOCK HEIGHT: 1", output)
        self.assertIn(f"PAYOUT ADDRESS: {PAYOUT_ADDRESS}", output)
        self.assertIn(f"COINBASE TXID: {coinbase_txid[::-1].hex()}", output)
        self.assertIn("ACTIVE CHAIN", output)

    def test_main_stops_after_accepted_live_mainnet_block(self) -> None:
        core = FakeCore()
        result = self.run_main(
            core,
            miner_args(
                live_mainnet=True,
                blocks=0,
                expected_payout_script=PAYOUT_SCRIPT,
            ),
        )

        self.assertEqual(result, 0)
        self.assertEqual(core.methods.count("submitblock"), 1)
        self.assertEqual(core.methods.count("getblocktemplate"), 1)
        self.assertLess(
            core.methods.index("validateaddress"),
            core.methods.index("getblocktemplate"),
        )
        output = self.output.getvalue()
        for expected in (
            "NETWORK: MAINNET",
            "MODE: LIVE",
            f"PAYOUT ADDRESS: {PAYOUT_ADDRESS}",
            f"PAYOUT SCRIPT (Bitcoin Core): {PAYOUT_SCRIPT}",
            f"EXPECTED PAYOUT SCRIPT: {PAYOUT_SCRIPT}",
            "PAYOUT SCRIPT MATCH: YES",
            "[CORE] Chain: main",
            "[CORE] Synchronization: blocks=100, headers=100",
            "[CORE] Peers: 8",
            "[CORE] Height: 100",
            f"[CORE] Best block: {'00' * 32}",
            "MAINNET BLOCK ACCEPTED",
            "Mining stopped after the accepted mainnet block",
        ):
            self.assertIn(expected, output)

    def test_rejection_string_is_not_success(self) -> None:
        for reason in ("high-hash", "bad-cb-amount", "duplicate-invalid", "rejected"):
            with self.subTest(reason=reason):
                core = FakeCore()
                core.submit_result = reason
                with self.assertRaisesRegex(RuntimeError, f"rejected the block: {reason}"):
                    self.mine(core, live_mainnet=True)

                self.assertIn(f"REJECTED: submitblock response='{reason}'", self.output.getvalue())
                self.assertNotIn("ACCEPTED", self.output.getvalue())
                self.assertNotIn("getblockcount", core.methods)

    def test_stale_rejection_requests_fresh_template(self) -> None:
        for reason in ("inconclusive", "duplicate-inconclusive", "prev-blk-not-found"):
            with self.subTest(reason=reason):
                core = FakeCore()
                core.submit_result = reason

                self.assertFalse(self.mine(core, live_mainnet=True))
                output = self.output.getvalue()
                # Core stored an inconclusive block; that is not a rejection.
                self.assertEqual("REJECTED" in output, reason == "prev-blk-not-found")
                self.assertEqual("NOT ON ACTIVE CHAIN" in output, reason != "prev-blk-not-found")
                self.assertIn("requesting a fresh template", output)
                self.assertNotIn("ACCEPTED", output)
                self.assertEqual(core.methods.count("submitblock"), 1)

    def test_duplicate_is_resolved_by_asking_core_about_the_block(self) -> None:
        core = FakeCore()
        core.submit_result = "duplicate"
        core.header_confirmations = 3
        self.assertTrue(self.mine(core, live_mainnet=True, payout_address=PAYOUT_ADDRESS))
        self.assertIn("already has block", self.output.getvalue())
        self.assertIn("MAINNET BLOCK ACCEPTED", self.output.getvalue())
        self.assertEqual(core.methods.count("submitblock"), 1)

        core = FakeCore()
        core.submit_result = "duplicate"
        core.header_confirmations = -1
        self.assertFalse(self.mine(core, live_mainnet=True))
        self.assertIn("NOT ON ACTIVE CHAIN", self.output.getvalue())
        self.assertNotIn("ACCEPTED", self.output.getvalue())
        self.assertEqual(core.methods.count("submitblock"), 1)

    def test_failed_submit_call_is_accepted_when_core_has_the_block(self) -> None:
        # The call timed out, but Core had already processed the block.
        core = FakeCore()
        core.submit_error = RpcError("Bitcoin Core RPC submitblock timed out after 120 seconds")
        core.header_confirmations = 1

        self.assertTrue(self.mine(core, live_mainnet=True, payout_address=PAYOUT_ADDRESS))
        self.assertEqual(core.methods.count("submitblock"), 1)
        output = self.output.getvalue()
        self.assertIn("DID NOT COMPLETE", output)
        self.assertIn("already has block", output)
        self.assertIn("MAINNET BLOCK ACCEPTED", output)
        self.assertNotIn("SUBMISSION FAILED", output)

    def test_failed_submit_is_retried_until_core_answers(self) -> None:
        core = FakeCore()
        attempts = 0

        def fail_twice(cli, network, datadir, conf, method, *params, **kwargs):
            nonlocal attempts
            if method == "submitblock":
                attempts += 1
                core.submit_error = RpcError("connection refused") if attempts <= 2 else None
            return core(cli, network, datadir, conf, method, *params, **kwargs)

        self.assertTrue(self.mine(fail_twice, live_mainnet=True, payout_address=PAYOUT_ADDRESS))
        self.assertEqual(core.methods.count("submitblock"), 3)
        # Every attempt sent the identical verified block.
        self.assertEqual(len(set(core.submitted)), 1)
        self.assertIn("MAINNET BLOCK ACCEPTED", self.output.getvalue())

    def test_main_reports_rejection_as_failure(self) -> None:
        core = FakeCore()
        core.submit_result = "bad-txnmrklroot"
        result = self.run_main(core, live_args())

        self.assertEqual(result, 1)
        self.assertIn("bad-txnmrklroot", self.errors.getvalue())
        self.assertNotIn("ACCEPTED", self.output.getvalue())

    def test_submitblock_rpc_failure_is_not_success(self) -> None:
        for error in (
            RuntimeError("Bitcoin Core RPC submitblock failed: timeout"),
            OSError("bitcoin-cli could not be started"),
        ):
            with self.subTest(error=error):
                core = FakeCore()
                core.submit_error = error
                with self.assertRaisesRegex(RuntimeError, "submitblock failed"):
                    self.mine(core, live_mainnet=True)

                self.assertIn("SUBMISSION FAILED", self.output.getvalue())
                self.assertNotIn("ACCEPTED", self.output.getvalue())
                self.assertNotIn("getblockcount", core.methods)
                # Bounded: one first attempt plus the fixed retry schedule,
                # every one of them the same block, saved exactly once.
                self.assertEqual(
                    len(core.submitted), 1 + len(regtest_miner.SUBMIT_RETRY_DELAYS)
                )
                self.assertEqual(len(set(core.submitted)), 1)
                self.save_mock.assert_called_once()
                saved_hash, saved_block = self.save_mock.call_args.args
                self.assertEqual(saved_block.hex(), core.submitted[0])
                self.assertEqual(
                    saved_hash, double_sha256(saved_block[:80])[::-1].hex()
                )

    def test_tampered_block_fails_pre_submission_checks(self) -> None:
        template = easy_template()
        payout = bytes.fromhex(PAYOUT_SCRIPT)
        other = bytes.fromhex("0014" + "11" * 20)
        transactions, txids = template_transactions(template)
        target = bits_to_target(0x207FFFFF)
        previous = template["previousblockhash"]

        def build(script, value=template["coinbasevalue"]):
            coinbase, txid = create_coinbase(1, b"", 0, value, script, None)
            header = build_template_header(template, txid, txids)
            nonce, _digest = cpu_find_nonce(None, header, 0x207FFFFF)
            header = header[:76] + struct.pack("<I", nonce)
            return header, coinbase, txid

        header, coinbase, txid = build(payout)
        block = serialize_block(header, coinbase, transactions)
        verify_candidate_block(
            block, header, target, template, previous,
            coinbase, txid, transactions, payout, None,
        )

        other_header, other_coinbase, other_txid = build(other)
        low_header, low_coinbase, low_txid = build(payout, 1)
        cases = {
            "payout script": (
                serialize_block(other_header, other_coinbase, transactions),
                other_header, target, previous, other_coinbase, other_txid, payout,
            ),
            "coinbase value": (
                serialize_block(low_header, low_coinbase, transactions),
                low_header, target, previous, low_coinbase, low_txid, payout,
            ),
            "swapped coinbase": (
                serialize_block(header, other_coinbase, transactions),
                header, target, previous, other_coinbase, other_txid, other,
            ),
            "header": (block, other_header, target, previous, coinbase, txid, payout),
            "target": (block, header, 0, previous, coinbase, txid, payout),
            "previous hash": (block, header, target, "ff" * 32, coinbase, txid, payout),
            "first leaf": (block, header, target, previous, coinbase, bytes(32), payout),
            "dropped transaction": (
                serialize_block(header, coinbase, []),
                header, target, previous, coinbase, txid, payout,
            ),
        }
        for label, (blk, hdr, tgt, prev, cb, cb_txid, script) in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    verify_candidate_block(
                        blk, hdr, tgt, template, prev,
                        cb, cb_txid, transactions, script, None,
                    )

    def test_witness_commitment_is_the_only_other_coinbase_output(self) -> None:
        commitment = bytes.fromhex(WITNESS_COMMITMENT)
        payout = bytes.fromhex(PAYOUT_SCRIPT)
        coinbase, _txid = create_coinbase(1, b"", 0, 625, payout, commitment)

        self.assertEqual(
            transaction_outputs(coinbase),
            [(625, payout), (0, commitment)],
        )

    def test_regtest_block_is_still_submitted(self) -> None:
        core = FakeCore(chain="regtest")

        self.assertTrue(self.mine(core, network="regtest", payout_script=b"\x51"))
        self.assertEqual(core.methods.count("submitblock"), 1)
        self.assertEqual(core.networks, {"regtest"})
        self.assertNotIn("MAINNET", self.output.getvalue())

    def test_regtest_main_mines_requested_blocks(self) -> None:
        core = FakeCore(chain="regtest")
        args = miner_args(network="regtest", payout_address=None, blocks=2)

        self.assertEqual(self.run_main(core, args), 0)
        self.assertEqual(core.methods.count("submitblock"), 2)
        self.assertEqual(core.networks, {"regtest"})
        self.assertNotIn("validateaddress", core.methods)

    def test_regtest_refuses_node_on_another_chain(self) -> None:
        core = FakeCore(chain="main")
        args = miner_args(network="regtest", payout_address=None)

        self.assertEqual(self.run_main(core, args), 1)
        self.assertIn("regtest chain", self.errors.getvalue())
        self.assertEqual(core.methods, ["getblockchaininfo"])

    def test_submitblock_is_sent_through_stdin_not_argv(self) -> None:
        block_hex = "ab" * 100_000
        for stdout, expected in (("", None), ("high-hash\n", "high-hash")):
            with self.subTest(stdout=stdout):
                completed = subprocess.CompletedProcess([], 0, stdout, "")
                with patch(
                    "regtest_miner.subprocess.run", return_value=completed
                ) as run:
                    result = rpc(
                        Path("bitcoin-cli.exe"), "mainnet", None, None,
                        "submitblock", block_hex,
                    )

                self.assertEqual(result, expected)
                command = run.call_args.args[0]
                self.assertEqual(command[-2:], ["-stdin", "submitblock"])
                self.assertNotIn("-regtest", command)
                self.assertNotIn(block_hex, command)
                self.assertEqual(run.call_args.kwargs["input"], block_hex)

        failed = subprocess.CompletedProcess([], 1, "", "error: timeout")
        with patch("regtest_miner.subprocess.run", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "submitblock failed"):
                rpc(
                    Path("bitcoin-cli.exe"), "mainnet", None, None,
                    "submitblock", block_hex,
                )

    def test_other_rpc_methods_do_not_use_stdin(self) -> None:
        completed = subprocess.CompletedProcess([], 0, "{}", "")
        with patch("regtest_miner.subprocess.run", return_value=completed) as run:
            rpc(Path("bitcoin-cli.exe"), "regtest", None, None, "getblockchaininfo")

        self.assertEqual(
            run.call_args.args[0],
            ["bitcoin-cli.exe", "-regtest", "getblockchaininfo"],
        )
        self.assertIsNone(run.call_args.kwargs["input"])


class FakeCudaProcess:
    """Scripted stand-in for a `cuda_miner --serve` child process."""

    def __init__(self, responder=None, ready="READY\n") -> None:
        self.responder = responder or (lambda fields: f"{fields[1]} NONE\n")
        self.pending: list[object] = [ready]
        self.requests: list[str] = []
        self.events: list[tuple] = []
        self.ready_read = False
        self.returncode: int | None = None
        self.exit_status = 1
        self.stderr_text = ""
        self.stdin_error: Exception | None = None
        self.stdin_closed = False
        self.killed = False
        self.hang_on_exit = False
        self.wait_timeouts: list[float | None] = []
        process = self

        class Stdin:
            def write(self, text: str) -> None:
                if process.stdin_error is not None:
                    raise process.stdin_error
                process.requests.append(text)
                process.events.append(("send", text.split()[1]))
                process.pending.append(process.responder(text.split()))

            def flush(self) -> None:
                pass

            def close(self) -> None:
                process.stdin_closed = True

        class Stdout:
            def readline(self) -> str:
                reply = process.pending.pop(0) if process.pending else ""
                if isinstance(reply, BaseException):
                    raise reply
                if process.ready_read:
                    process.events.append(("read", (reply.split() or [""])[0]))
                process.ready_read = True
                return reply

            def close(self) -> None:
                pass

        self.stdin = Stdin()
        self.stdout = Stdout()
        self.stderr = Stdout()

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        if self.returncode is None:
            self.returncode = self.exit_status

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        if self.hang_on_exit and not self.killed:
            raise subprocess.TimeoutExpired("cuda_miner", timeout)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        if self.returncode is None:
            self.returncode = self.exit_status
        return "", self.stderr_text


def cpu_scan_responder(fields: list[str]) -> str:
    _command, request_id, header_hex, start, count = fields
    header = bytes.fromhex(header_hex)
    target = bits_to_target(struct.unpack("<I", header[72:76])[0])
    for nonce in range(int(start), int(start) + int(count)):
        digest = double_sha256(header[:76] + struct.pack("<I", nonce))
        if int.from_bytes(digest[::-1], "big") <= target:
            return f"{request_id} FOUND {nonce} {digest[::-1].hex()}\n"
    return f"{request_id} NONE\n"


class PersistentCudaMinerTests(unittest.TestCase):
    HEADER_A = bytes(range(80))
    HEADER_B = bytes(range(1, 81))

    def started(self, process: FakeCudaProcess):
        popen = patch("regtest_miner.subprocess.Popen", return_value=process)
        mock = popen.start()
        self.addCleanup(popen.stop)
        return CudaMiner(Path("cuda_miner.exe")), mock

    def test_one_process_serves_many_chunks_and_a_new_header(self) -> None:
        process = FakeCudaProcess()
        miner, popen = self.started(process)
        popen.assert_not_called()

        self.assertIsNone(miner.scan(self.HEADER_A, 0, 10))
        self.assertIsNone(miner.scan(self.HEADER_A, 10, 10))
        self.assertIsNone(miner.scan(self.HEADER_B, 0, 10))

        popen.assert_called_once()
        self.assertEqual(popen.call_args.args[0], ["cuda_miner.exe", "--serve"])
        self.assertEqual(popen.call_args.kwargs["stdin"], subprocess.PIPE)
        self.assertEqual(popen.call_args.kwargs["stdout"], subprocess.PIPE)
        self.assertEqual(popen.call_args.kwargs["stderr"], subprocess.PIPE)
        self.assertEqual(
            process.requests,
            [
                f"SCAN 1 {self.HEADER_A.hex()} 0 10\n",
                f"SCAN 2 {self.HEADER_A.hex()} 10 10\n",
                f"SCAN 3 {self.HEADER_B.hex()} 0 10\n",
            ],
        )

        miner.close()
        miner.close()
        self.assertTrue(process.stdin_closed)
        self.assertEqual(len(process.wait_timeouts), 1)
        self.assertIsNotNone(process.wait_timeouts[0])
        self.assertFalse(process.killed)
        self.assertIsNone(miner.process)

    def test_found_reply_is_parsed(self) -> None:
        process = FakeCudaProcess(lambda fields: f"{fields[1]} FOUND 7 {'ab' * 32}\n")
        miner, _popen = self.started(process)

        self.assertEqual(miner.scan(self.HEADER_A, 0, 10), (7, "ab" * 32))
        self.assertEqual(
            miner.scan(self.HEADER_A, 0, 10), (7, "ab" * 32)
        )

    def test_find_nonce_scans_every_chunk_through_one_process(self) -> None:
        process = FakeCudaProcess()
        miner, popen = self.started(process)
        stale_checks = []

        with redirect_stdout(io.StringIO()):
            result = find_nonce(
                miner,
                self.HEADER_A,
                0x1D00FFFF,
                5,
                0,
                19,
                lambda: stale_checks.append(len(process.requests)) or False,
            )

        self.assertIsNone(result)
        popen.assert_called_once()
        self.assertEqual(
            [request.split()[3:] for request in process.requests],
            [["0", "5"], ["5", "5"], ["10", "5"], ["15", "5"]],
        )
        # The first check precedes any GPU work; each later check runs while the
        # chunk it belongs to is already scanning.
        self.assertEqual(stale_checks, [0, 2, 3, 4])

    def test_malformed_reply_is_fatal_and_never_retried(self) -> None:
        good_hash = "ab" * 32
        replies = (
            "NONE\n",
            "2 NONE\n",
            "1 NONE extra\n",
            "1 NONE",
            "1 MAYBE\n",
            "1 FOUND 7\n",
            f"1 FOUND x {good_hash}\n",
            f"1 FOUND -7 {good_hash}\n",
            f"1 FOUND 4294967296 {good_hash}\n",
            f"1 FOUND 7 {good_hash[:-2]}\n",
            f"1 FOUND 7 {good_hash.upper()}\n",
            f"1 FOUND 7 {good_hash} extra\n",
            f"FOUND 7 {good_hash}\n",
            "garbage\n",
            "\n",
        )
        for reply in replies:
            with self.subTest(reply=reply):
                process = FakeCudaProcess(lambda _fields, reply=reply: reply)
                popen_patch = patch(
                    "regtest_miner.subprocess.Popen", return_value=process
                )
                with popen_patch as popen:
                    miner = CudaMiner(Path("cuda_miner.exe"))
                    with self.assertRaisesRegex(RuntimeError, "CUDA miner failed"):
                        miner.scan(self.HEADER_A, 0, 10)

                    self.assertTrue(process.killed)
                    self.assertTrue(miner.failed)
                    self.assertIsNone(miner.process)
                    with self.assertRaisesRegex(RuntimeError, "not restarted"):
                        miner.scan(self.HEADER_A, 0, 10)
                    popen.assert_called_once()

    def test_process_death_reports_status_and_diagnostics(self) -> None:
        process = FakeCudaProcess(lambda _fields: "")
        process.stderr_text = "CUDA scan error: out of memory\n"
        process.exit_status = 1
        miner, popen = self.started(process)

        with self.assertRaisesRegex(
            RuntimeError,
            "ended without a complete reply.*exit status 1.*out of memory",
        ):
            miner.scan(self.HEADER_A, 0, 10)

        self.assertTrue(miner.failed)
        with self.assertRaisesRegex(RuntimeError, "not restarted"):
            miner.scan(self.HEADER_A, 0, 10)
        popen.assert_called_once()

    def test_broken_pipe_and_read_errors_are_fatal(self) -> None:
        process = FakeCudaProcess()
        process.stdin_error = BrokenPipeError("pipe closed")
        miner, _popen = self.started(process)
        with self.assertRaisesRegex(RuntimeError, "lost contact"):
            miner.scan(self.HEADER_A, 0, 10)
        self.assertTrue(process.killed)

        process = FakeCudaProcess(lambda _fields: OSError("read failed"))
        with patch("regtest_miner.subprocess.Popen", return_value=process):
            miner = CudaMiner(Path("cuda_miner.exe"))
            with self.assertRaisesRegex(RuntimeError, "lost contact"):
                miner.scan(self.HEADER_A, 0, 10)
        self.assertTrue(process.killed)

    def test_startup_failures_are_fatal(self) -> None:
        for ready in ("", "NONE\n", "READY", "Usage: cuda_miner\n"):
            with self.subTest(ready=ready):
                process = FakeCudaProcess(ready=ready)
                process.stderr_text = "Usage: cuda_miner [--scan-header ...]"
                process.exit_status = 2
                with patch("regtest_miner.subprocess.Popen", return_value=process):
                    miner = CudaMiner(Path("cuda_miner.exe"))
                    with self.assertRaisesRegex(RuntimeError, "did not report READY"):
                        miner.scan(self.HEADER_A, 0, 10)
                self.assertEqual(process.requests, [])
                self.assertTrue(process.killed)
                self.assertTrue(miner.failed)

        with patch("regtest_miner.subprocess.Popen", side_effect=OSError("missing")):
            miner = CudaMiner(Path("cuda_miner.exe"))
            with self.assertRaisesRegex(RuntimeError, "Could not start CUDA miner"):
                miner.scan(self.HEADER_A, 0, 10)
        self.assertTrue(miner.failed)

    def test_close_kills_a_process_that_does_not_exit(self) -> None:
        process = FakeCudaProcess()
        process.hang_on_exit = True
        miner, _popen = self.started(process)
        miner.scan(self.HEADER_A, 0, 10)

        miner.close()

        self.assertTrue(process.stdin_closed)
        self.assertTrue(process.killed)
        self.assertEqual(len(process.wait_timeouts), 2)

    def test_close_without_a_started_process_does_nothing(self) -> None:
        with patch("regtest_miner.subprocess.Popen") as popen:
            CudaMiner(Path("cuda_miner.exe")).close()
        popen.assert_not_called()

    def run_main(self, core, args, process):
        self.output = io.StringIO()
        self.errors = io.StringIO()
        with (
            patch("regtest_miner.parse_args", return_value=args),
            patch("regtest_miner.ensure_tor_ready", return_value=None),
            patch("regtest_miner.rpc", side_effect=core),
            patch(
                "regtest_miner.subprocess.Popen", return_value=process
            ) as self.popen,
            patch("regtest_miner.save_unsubmitted_block", return_value=None),
            patch("regtest_miner.monitor_block", return_value=0),
            patch("regtest_miner.time.sleep"),
            redirect_stdout(self.output),
            redirect_stderr(self.errors),
        ):
            return main()

    def test_session_reuses_one_process_across_templates(self) -> None:
        core = FakeCore(chain="regtest")
        original = core.__call__

        def next_template_after_each_block(cli, network, datadir, conf, method, *params, **kwargs):
            result = original(cli, network, datadir, conf, method, *params, **kwargs)
            if method == "submitblock":
                core.template = dict(core.template, curtime=core.template["curtime"] + 1)
            return result

        process = FakeCudaProcess(cpu_scan_responder)
        args = miner_args(
            network="regtest", payout_address=None, blocks=3, chunk_size=1
        )
        result = self.run_main(next_template_after_each_block, args, process)

        self.assertEqual(result, 0)
        self.assertEqual(core.methods.count("submitblock"), 3)
        self.popen.assert_called_once()
        headers = [request.split()[2] for request in process.requests]
        self.assertEqual(len(set(headers)), 3)
        self.assertEqual(
            [int(request.split()[1]) for request in process.requests],
            list(range(1, len(process.requests) + 1)),
        )
        for block_hex, header_hex in zip(core.submitted, dict.fromkeys(headers)):
            self.assertEqual(block_hex[:152], header_hex[:152])
        # The session ends by closing the child's stdin and waiting for it.
        self.assertTrue(process.stdin_closed)
        self.assertEqual(len(process.wait_timeouts), 1)
        self.assertFalse(process.killed)

    def test_ctrl_c_during_a_scan_shuts_the_process_down(self) -> None:
        core = FakeCore(chain="regtest")
        process = FakeCudaProcess(lambda _fields: KeyboardInterrupt())
        args = miner_args(network="regtest", payout_address=None)

        result = self.run_main(core, args, process)

        self.assertEqual(result, 130)
        self.assertIn("Stopped by user", self.output.getvalue())
        self.assertEqual(len(process.requests), 1)
        self.assertTrue(process.stdin_closed)
        self.assertEqual(len(process.wait_timeouts), 1)
        self.assertNotIn("submitblock", core.methods)

    def test_ctrl_c_between_chunks_shuts_the_process_down(self) -> None:
        core = FakeCore(chain="regtest")
        tips = 0

        def interrupt_third_tip_query(cli, network, datadir, conf, method, *params, **kwargs):
            nonlocal tips
            if method == "getbestblockhash":
                tips += 1
                if tips == 3:
                    raise KeyboardInterrupt
            return core(cli, network, datadir, conf, method, *params, **kwargs)

        process = FakeCudaProcess()
        args = miner_args(network="regtest", payout_address=None, chunk_size=1)
        result = self.run_main(interrupt_third_tip_query, args, process)

        # The interrupt arrives in the tip check that overlaps chunk 2, so
        # CUDA is scanning and Core is being queried at the same moment.
        self.assertEqual(result, 130)
        self.assertEqual(len(process.requests), 2)
        self.assertEqual(
            process.events[-3:], [("send", "1"), ("read", "1"), ("send", "2")]
        )
        self.assertTrue(process.stdin_closed)
        self.assertEqual(len(process.wait_timeouts), 1)
        self.assertFalse(process.killed)
        self.assertNotIn("submitblock", core.methods)

    def test_cuda_errors_never_reach_submission(self) -> None:
        good_hash = "00" * 32

        def wrong_hash(fields):
            return f"{fields[1]} FOUND {fields[3]} {good_hash}\n"

        def nonce_outside_range(fields):
            return f"{fields[1]} FOUND {int(fields[3]) + int(fields[4])} {good_hash}\n"

        def stale_reply_for_previous_request(fields):
            return f"{int(fields[1]) - 1} NONE\n"

        cases = {
            "malformed": (lambda _fields: "FOUND\n", "CUDA miner failed"),
            "process died": (lambda _fields: "", "CUDA miner failed"),
            "wrong request id": (stale_reply_for_previous_request, "CUDA miner failed"),
            "hash disagrees with CPU": (wrong_hash, "does not match the CPU"),
            "nonce outside range": (nonce_outside_range, "outside its assigned range"),
        }
        for network, live in (("mainnet", True), ("regtest", False)):
            for label, (responder, message) in cases.items():
                with self.subTest(network=network, label=label):
                    core = FakeCore()
                    process = FakeCudaProcess(responder)
                    with (
                        patch("regtest_miner.rpc", side_effect=core),
                        patch("regtest_miner.subprocess.Popen", return_value=process),
                        redirect_stdout(io.StringIO()),
                    ):
                        miner = CudaMiner(Path("cuda_miner.exe"))
                        with self.assertRaisesRegex(RuntimeError, message):
                            mine_one_block(
                                Path("bitcoin-cli.exe"),
                                miner,
                                network,
                                None,
                                None,
                                10,
                                0,
                                bytes.fromhex(PAYOUT_SCRIPT),
                                live_mainnet=live,
                            )
                    self.assertNotIn("submitblock", core.methods)

    def test_main_stops_with_an_error_when_cuda_fails(self) -> None:
        core = FakeCore()
        process = FakeCudaProcess(lambda _fields: "")
        process.stderr_text = "CUDA scan error: unspecified launch failure"

        result = self.run_main(core, live_args(), process)

        self.assertEqual(result, 1)
        self.assertIn("unspecified launch failure", self.errors.getvalue())
        self.assertNotIn("submitblock", core.methods)
        self.assertNotIn("ACCEPTED", self.output.getvalue())
        self.popen.assert_called_once()

    @unittest.skipUnless(CUDA_MINER.is_file(), "Build cuda_miner.exe to run this vector")
    def test_real_persistent_process_known_vector_and_work_replacement(self) -> None:
        vector_header = (
            struct.pack("<I", 0x20000000)
            + bytes(64)
            + struct.pack("<I", 1_728_000_000)
            + struct.pack("<I", 0x1F00FFFF)
            + bytes(4)
        )
        expected = (
            107938,
            "00009dac139e241aac5c9bfda9a7526dd697145b3147332cdbc0bd3e7ff24b42",
        )
        hard_header = vector_header[:72] + struct.pack("<I", 0x17021EF0) + bytes(4)
        miner = CudaMiner(CUDA_MINER)
        self.addCleanup(miner.close)

        self.assertEqual(miner.scan(vector_header, 107938, 1), expected)
        process = miner.process
        self.assertEqual(miner.scan(vector_header, 100_000, 10_000), expected)
        self.assertEqual(miner.scan(vector_header, 0, 2_000_000), expected)
        self.assertIsNone(miner.scan(vector_header, 107_939, 1))
        # New work replaces the old header without restarting the process...
        self.assertIsNone(miner.scan(hard_header, 0, 2_000_000))
        # ...and returning to the first header gives the first answer again.
        self.assertEqual(miner.scan(vector_header, 107_000, 1_000), expected)
        self.assertEqual(
            expected[1],
            double_sha256(vector_header[:76] + struct.pack("<I", 107938))[::-1].hex(),
        )
        self.assertIs(miner.process, process)
        self.assertIsNone(process.poll())

        miner.close()
        self.assertEqual(process.returncode, 0)

    @unittest.skipUnless(CUDA_MINER.is_file(), "Build cuda_miner.exe to run this vector")
    def test_real_process_death_and_malformed_request_fail_closed(self) -> None:
        miner = CudaMiner(CUDA_MINER)
        self.addCleanup(miner.close)
        self.assertIsNone(miner.scan(bytes(72) + struct.pack("<I", 0x17021EF0) + bytes(4), 0, 1))
        miner.process.kill()
        miner.process.wait()
        with self.assertRaisesRegex(RuntimeError, "CUDA miner failed"):
            miner.scan(bytes(72) + struct.pack("<I", 0x17021EF0) + bytes(4), 1, 1)
        self.assertTrue(miner.failed)

        # An invalid nBits makes the real process exit nonzero without a reply.
        miner = CudaMiner(CUDA_MINER)
        self.addCleanup(miner.close)
        with self.assertRaisesRegex(RuntimeError, "Invalid compact target"):
            miner.scan(bytes(80), 0, 1)

        result = subprocess.run(
            [str(CUDA_MINER), "--serve"],
            input="SCAN 1 nothex 0 1\n",
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "READY\n")
        self.assertIn("Malformed request", result.stderr)


class OverlappedTipCheckTests(unittest.TestCase):
    """The tip check for chunk N+1 runs while chunk N+1 is already scanning."""

    HARD = bytes(72) + struct.pack("<I", 0x1D00FFFF) + bytes(4)
    EASY = bytes(range(72)) + struct.pack("<I", 0x207FFFFF) + bytes(4)

    def miner_with(self, process: FakeCudaProcess):
        popen = patch("regtest_miner.subprocess.Popen", return_value=process)
        self.popen = popen.start()
        self.addCleanup(popen.stop)
        return CudaMiner(Path("cuda_miner.exe"))

    def checker(self, process: FakeCudaProcess, verdicts):
        verdicts = list(verdicts)

        def stale_check() -> bool:
            verdict = verdicts.pop(0) if verdicts else False
            if isinstance(verdict, BaseException):
                process.events.append(("check", type(verdict).__name__))
                raise verdict
            process.events.append(("check", verdict))
            return verdict

        return stale_check

    def find(self, miner, header, bits, chunk, last_nonce, stale_check):
        with redirect_stdout(io.StringIO()):
            return find_nonce(miner, header, bits, chunk, 0, last_nonce, stale_check)

    def test_unchanged_tip_keeps_the_gpu_busy_and_checks_every_chunk(self) -> None:
        process = FakeCudaProcess()
        miner = self.miner_with(process)

        result = self.find(miner, self.HARD, 0x1D00FFFF, 5, 19, self.checker(process, []))

        self.assertIsNone(result)
        self.assertEqual(
            process.events,
            [
                ("check", False),                 # before any GPU work
                ("send", "1"), ("read", "1"),
                ("send", "2"), ("check", False), ("read", "2"),
                ("send", "3"), ("check", False), ("read", "3"),
                ("send", "4"), ("check", False), ("read", "4"),
            ],
        )
        self.assertEqual(
            [request.split()[3:] for request in process.requests],
            [["0", "5"], ["5", "5"], ["10", "5"], ["15", "5"]],
        )
        self.assertIsNone(miner.pending_id)
        self.popen.assert_called_once()

    def test_stale_template_before_the_first_chunk_starts_no_gpu_work(self) -> None:
        process = FakeCudaProcess()
        miner = self.miner_with(process)

        with self.assertRaises(StaleTemplate):
            self.find(miner, self.HARD, 0x1D00FFFF, 5, 19, self.checker(process, [True]))

        self.assertEqual(process.events, [("check", True)])
        self.popen.assert_not_called()

    def test_tip_change_during_a_scan_abandons_the_chunk_in_flight(self) -> None:
        # Request 3 would return a perfectly valid candidate; it must never be read
        # as a result because its tip check reported a new block.
        def responder(fields):
            return f"{fields[1]} NONE\n" if fields[1] in ("1", "2") else cpu_scan_responder(fields)

        process = FakeCudaProcess(responder)
        miner = self.miner_with(process)

        with self.assertRaises(StaleTemplate):
            self.find(
                miner, self.EASY, 0x207FFFFF, 50, 499,
                self.checker(process, [False, False, True]),
            )

        self.assertEqual(
            process.events[-5:],
            [("send", "2"), ("check", False), ("read", "2"), ("send", "3"), ("check", True)],
        )
        self.assertNotIn(("read", "3"), process.events)
        self.assertEqual(process.pending[0].split()[1], "FOUND")
        self.assertTrue(miner.pending_abandoned)
        with self.assertRaisesRegex(RuntimeError, "no scan result is waiting"):
            miner.finish_scan()

    def test_new_work_after_a_stale_chunk_discards_the_old_reply_first(self) -> None:
        process = FakeCudaProcess(cpu_scan_responder)
        miner = self.miner_with(process)
        miner.start_scan(self.EASY, 0, 50)
        stale_reply = process.pending[0]
        self.assertEqual(stale_reply.split()[1], "FOUND")
        miner.abandon_scan()

        new_header = bytes(range(1, 73)) + struct.pack("<I", 0x1D00FFFF) + bytes(4)
        miner.start_scan(new_header, 0, 50)
        result = miner.finish_scan()

        self.assertIsNone(result)
        self.assertEqual(
            process.events, [("send", "1"), ("read", "1"), ("send", "2"), ("read", "2")]
        )
        self.popen.assert_called_once()

    def test_abandoned_reply_is_still_validated(self) -> None:
        for reply in ("9 NONE\n", "garbage\n", "", "1 FOUND 7\n"):
            with self.subTest(reply=reply):
                process = FakeCudaProcess(lambda _fields, reply=reply: reply)
                with patch("regtest_miner.subprocess.Popen", return_value=process):
                    miner = CudaMiner(Path("cuda_miner.exe"))
                    miner.start_scan(self.HARD, 0, 5)
                    miner.abandon_scan()
                    with self.assertRaisesRegex(RuntimeError, "CUDA miner failed"):
                        miner.start_scan(self.HARD, 5, 5)
                self.assertTrue(process.killed)
                self.assertTrue(miner.failed)
                self.assertEqual(len(process.requests), 1)

    def test_scan_api_misuse_fails_closed(self) -> None:
        process = FakeCudaProcess()
        miner = self.miner_with(process)
        miner.start_scan(self.HARD, 0, 5)
        with self.assertRaisesRegex(RuntimeError, "before the previous result was read"):
            miner.start_scan(self.HARD, 5, 5)
        self.assertTrue(process.killed)
        self.assertEqual(len(process.requests), 1)

        process = FakeCudaProcess()
        with patch("regtest_miner.subprocess.Popen", return_value=process):
            miner = CudaMiner(Path("cuda_miner.exe"))
            miner.scan(self.HARD, 0, 5)
            with self.assertRaisesRegex(RuntimeError, "no scan result is waiting"):
                miner.finish_scan()
        self.assertTrue(miner.failed)

    def test_candidate_waiting_during_the_tip_check_is_used_only_after_it_passes(self) -> None:
        def responder(fields):
            return f"{fields[1]} NONE\n" if fields[1] == "1" else cpu_scan_responder(fields)

        process = FakeCudaProcess(responder)
        miner = self.miner_with(process)
        waiting = []

        def slow_check() -> bool:
            # CUDA has already answered chunk 2 (a candidate) but nothing may
            # read it until this check has returned.
            if process.requests[1:]:
                waiting.append(list(process.pending))
                self.assertNotIn(("read", "2"), process.events)
            process.events.append(("check", False))
            return False

        nonce, raw_hash = self.find(miner, self.EASY, 0x207FFFFF, 50, 499, slow_check)

        self.assertEqual(len(waiting), 1)
        self.assertEqual(waiting[0][0].split()[1], "FOUND")
        self.assertEqual(
            process.events,
            [("check", False), ("send", "1"), ("read", "1"),
             ("send", "2"), ("check", False), ("read", "2")],
        )
        self.assertTrue(50 <= nonce < 100)
        self.assertEqual(raw_hash, double_sha256(self.EASY[:76] + struct.pack("<I", nonce)))
        self.assertLessEqual(
            int.from_bytes(raw_hash[::-1], "big"), bits_to_target(0x207FFFFF)
        )

    def test_tip_check_failure_stops_the_scan_without_reading_the_chunk(self) -> None:
        for error in (
            RuntimeError("Bitcoin Core RPC getbestblockhash failed: connection refused"),
            RuntimeError("Bitcoin Core RPC getbestblockhash timed out after 60 seconds"),
            OSError("bitcoin-cli could not be started"),
        ):
            with self.subTest(error=error):
                process = FakeCudaProcess(cpu_scan_responder)
                with patch("regtest_miner.subprocess.Popen", return_value=process):
                    miner = CudaMiner(Path("cuda_miner.exe"))
                    with self.assertRaises(type(error)):
                        self.find(
                            miner, self.HARD, 0x1D00FFFF, 5, 19,
                            self.checker(process, [False, error]),
                        )
                self.assertEqual(
                    process.events[-2:], [("send", "2"), ("check", type(error).__name__)]
                )
                self.assertNotIn(("read", "2"), process.events)

    def test_tip_check_rpc_has_a_timeout_and_times_out_closed(self) -> None:
        with patch(
            "regtest_miner.subprocess.run",
            side_effect=subprocess.TimeoutExpired("bitcoin-cli", 60),
        ) as run:
            with self.assertRaisesRegex(RuntimeError, "getbestblockhash timed out after 60 seconds"):
                rpc(Path("bitcoin-cli.exe"), "mainnet", None, None, "getbestblockhash", timeout=60)
        self.assertEqual(run.call_args.kwargs["timeout"], 60)

        completed = subprocess.CompletedProcess([], 0, "00" * 32 + "\n", "")
        with patch("regtest_miner.subprocess.run", return_value=completed) as run:
            rpc(Path("bitcoin-cli.exe"), "mainnet", None, None, "getbestblockhash")
        self.assertIsNone(run.call_args.kwargs["timeout"])

    def run_block(self, core, process, network="mainnet", live=True, chunk=50):
        self.output = io.StringIO()
        with (
            patch("regtest_miner.rpc", side_effect=core),
            patch("regtest_miner.subprocess.Popen", return_value=process),
            patch("regtest_miner.save_unsubmitted_block", return_value=None),
            patch("regtest_miner.time.sleep"),
            redirect_stdout(self.output),
        ):
            miner = CudaMiner(Path("cuda_miner.exe"))
            return mine_one_block(
                Path("bitcoin-cli.exe"),
                miner,
                network,
                None,
                None,
                chunk,
                0,
                bytes.fromhex(PAYOUT_SCRIPT),
                live_mainnet=live,
                payout_address=PAYOUT_ADDRESS,
            )

    @staticmethod
    def candidate_in_second_chunk(fields):
        return f"{fields[1]} NONE\n" if fields[1] == "1" else cpu_scan_responder(fields)

    def test_candidate_from_an_overlapped_chunk_is_verified_and_submitted(self) -> None:
        core = FakeCore()
        process = FakeCudaProcess(self.candidate_in_second_chunk)

        self.assertTrue(self.run_block(core, process))

        # Template check, first-chunk check, overlapped check for chunk 2;
        # nothing stands between the verified candidate and submitblock.
        self.assertEqual(
            core.methods[:5],
            ["getblocktemplate", "getbestblockhash", "getbestblockhash",
             "getbestblockhash", "submitblock"],
        )
        self.assertEqual(
            process.events,
            [("send", "1"), ("read", "1"), ("send", "2"), ("read", "2")],
        )
        header = bytes.fromhex(core.submitted[0][:160])
        self.assertTrue(50 <= struct.unpack("<I", header[76:])[0] < 100)
        self.assertIn("MAINNET BLOCK ACCEPTED", self.output.getvalue())

    def test_tip_change_after_the_last_scan_check_is_left_to_core(self) -> None:
        core = FakeCore()
        previous = core.template["previousblockhash"]
        # Every scan check saw the old tip; the competing block arrives after.
        # The candidate is submitted at once and Core reports it as a competitor.
        core.tips = [previous, previous, previous, "ff" * 32]
        core.submit_result = "inconclusive"
        process = FakeCudaProcess(self.candidate_in_second_chunk)

        self.assertFalse(self.run_block(core, process))

        self.assertEqual(core.methods.count("getbestblockhash"), 3)
        self.assertEqual(core.methods.count("submitblock"), 1)
        self.assertIn("CANDIDATE FOUND", self.output.getvalue())
        self.assertIn("NOT ON ACTIVE CHAIN", self.output.getvalue())
        self.assertNotIn("ACCEPTED", self.output.getvalue())

    def test_first_chunk_candidate_goes_straight_to_submission(self) -> None:
        core = FakeCore()
        process = FakeCudaProcess(cpu_scan_responder)

        self.assertTrue(self.run_block(core, process))

        self.assertEqual(len(process.requests), 1)
        self.assertEqual(
            core.methods,
            ["getblocktemplate", "getbestblockhash", "getbestblockhash",
             "submitblock", "getblockcount", "getblockhash"],
        )

    def test_stale_verdict_discards_a_candidate_already_returned_by_cuda(self) -> None:
        for network, live in (("mainnet", True), ("regtest", False)):
            with self.subTest(network=network):
                core = FakeCore()
                previous = core.template["previousblockhash"]
                core.tips = [previous, previous, "ff" * 32]
                process = FakeCudaProcess(self.candidate_in_second_chunk)

                self.assertFalse(self.run_block(core, process, network=network, live=live))

                self.assertEqual(process.pending[0].split()[1], "FOUND")
                self.assertNotIn(("read", "2"), process.events)
                self.assertEqual(core.methods.count("getbestblockhash"), 3)
                self.assertNotIn("submitblock", core.methods)
                self.assertNotIn("CANDIDATE FOUND", self.output.getvalue())
                self.assertIn("Template became stale", self.output.getvalue())

    def test_scan_checks_and_submission_carry_their_timeouts(self) -> None:
        core = FakeCore()
        calls = []

        def recording(cli, network, datadir, conf, method, *params, **kwargs):
            calls.append((method, kwargs))
            return core(cli, network, datadir, conf, method, *params, **kwargs)

        process = FakeCudaProcess(self.candidate_in_second_chunk)
        self.assertTrue(self.run_block(recording, process))

        tip_calls = [kwargs for method, kwargs in calls if method == "getbestblockhash"]
        scan_timeout = {"timeout": regtest_miner.TIP_CHECK_TIMEOUT}
        self.assertEqual(tip_calls, [{}, scan_timeout, scan_timeout])
        self.assertEqual(
            [kwargs for method, kwargs in calls if method == "submitblock"],
            [{"timeout": regtest_miner.SUBMIT_TIMEOUT}],
        )

    def run_main(self, core, args, process):
        self.output = io.StringIO()
        self.errors = io.StringIO()
        with (
            patch("regtest_miner.parse_args", return_value=args),
            patch("regtest_miner.ensure_tor_ready", return_value=None),
            patch("regtest_miner.rpc", side_effect=core),
            patch("regtest_miner.subprocess.Popen", return_value=process) as self.popen,
            patch("regtest_miner.save_unsubmitted_block", return_value=None),
            patch("regtest_miner.monitor_block", return_value=0),
            patch("regtest_miner.socks5_ready", return_value=True),
            patch("regtest_miner.time.sleep"),
            redirect_stdout(self.output),
            redirect_stderr(self.errors),
        ):
            return main()

    def nth_tip_check(self, core, number, action):
        seen = 0

        def wrapped(cli, network, datadir, conf, method, *params, **kwargs):
            nonlocal seen
            if method == "getbestblockhash":
                seen += 1
                if seen == number:
                    return action()
            return core(cli, network, datadir, conf, method, *params, **kwargs)

        return wrapped

    def test_session_survives_a_stale_chunk_and_never_submits_its_candidate(self) -> None:
        core = FakeCore(chain="regtest")
        # Requests 1 and 2 belong to the first template; request 2 holds a
        # valid candidate when the overlapped check reports a new block.
        process = FakeCudaProcess(self.candidate_in_second_chunk)
        rpc_fake = self.nth_tip_check(core, 3, lambda: "ff" * 32)
        args = miner_args(network="regtest", payout_address=None, chunk_size=50)

        result = self.run_main(rpc_fake, args, process)

        self.assertEqual(result, 0)
        self.popen.assert_called_once()
        self.assertEqual(core.methods.count("getblocktemplate"), 2)
        self.assertEqual(core.methods.count("submitblock"), 1)
        self.assertEqual(
            process.events,
            [("send", "1"), ("read", "1"), ("send", "2"),
             ("read", "2"),            # drained and discarded, never returned
             ("send", "3"), ("read", "3")],
        )
        stale_header, fresh_header = (process.requests[i].split()[2] for i in (1, 2))
        self.assertNotEqual(stale_header[:152], fresh_header[:152])
        self.assertEqual(core.submitted[0][:152], fresh_header[:152])
        self.assertIn("Template became stale", self.output.getvalue())
        self.assertTrue(process.stdin_closed)
        self.assertFalse(process.killed)

    def test_tip_check_failure_during_a_regtest_scan_stops_the_miner(self) -> None:
        def fail():
            raise RuntimeError("Bitcoin Core RPC getbestblockhash timed out after 60 seconds")

        args = miner_args(network="regtest", payout_address=None, chunk_size=50)
        core = FakeCore(chain="regtest")
        process = FakeCudaProcess(self.candidate_in_second_chunk)
        result = self.run_main(self.nth_tip_check(core, 3, fail), args, process)

        self.assertEqual(result, 1)
        self.assertIn("timed out", self.errors.getvalue())
        self.assertEqual(process.events[-1], ("send", "2"))
        self.assertNotIn("submitblock", core.methods)
        self.assertNotIn("ACCEPTED", self.output.getvalue())
        self.assertTrue(process.stdin_closed)
        self.popen.assert_called_once()

    def test_one_failed_mainnet_tip_check_neither_pauses_nor_loses_the_candidate(self) -> None:
        def fail():
            raise RpcError("Bitcoin Core RPC getbestblockhash timed out after 10 seconds")

        core = FakeCore()
        # Chunk 2 holds a valid candidate while its tip check fails.
        process = FakeCudaProcess(self.candidate_in_second_chunk)
        result = self.run_main(
            self.nth_tip_check(core, 3, fail), live_args(chunk_size=50), process
        )

        self.assertEqual(result, 0)
        self.assertIn("Tip check failed (1/3)", self.output.getvalue())
        self.assertNotIn("Mining paused", self.output.getvalue())
        self.assertEqual(
            process.events, [("send", "1"), ("read", "1"), ("send", "2"), ("read", "2")]
        )
        self.assertEqual(core.methods.count("submitblock"), 1)
        self.assertIn("MAINNET BLOCK ACCEPTED", self.output.getvalue())

    def test_repeated_mainnet_tip_check_failures_pause_then_resume_on_fresh_work(self) -> None:
        core = FakeCore()
        tip_checks = 0

        def flaky(cli, network, datadir, conf, method, *params, **kwargs):
            nonlocal tip_checks
            if method == "getbestblockhash":
                tip_checks += 1
                # Checks 3-5 overlap chunks 2-4 of the first template.
                if 3 <= tip_checks <= 5:
                    raise RpcError("Bitcoin Core RPC getbestblockhash failed: connection refused")
                if tip_checks == 9:
                    raise KeyboardInterrupt
            return core(cli, network, datadir, conf, method, *params, **kwargs)

        process = FakeCudaProcess()   # never finds anything
        template = easy_template()
        template["bits"], template["target"] = GENESIS_BITS, GENESIS_TARGET
        core.template = template
        result = self.run_main(flaky, live_args(chunk_size=50, blocks=0), process)

        self.assertEqual(result, 130)
        self.assertEqual(core.methods.count("getblocktemplate"), 2)
        self.assertEqual(
            len({request.split()[2][:152] for request in process.requests}), 2
        )
        output = self.output.getvalue()
        self.assertIn("Tip check failed (3/3)", output)
        self.assertIn("[MONITOR] Mining paused: 3 consecutive tip checks failed", output)
        self.assertIn("resuming with a fresh template", output)
        # All four requests of the first template were answered before the
        # pause, so nothing was in flight; request 5 is already the new work.
        self.assertEqual(
            process.events[:9],
            [(kind, str(number)) for number in range(1, 5) for kind in ("send", "read")]
            + [("send", "5")],
        )
        headers = [request.split()[2][:152] for request in process.requests]
        self.assertEqual(len(set(headers[:4])), 1)
        self.assertNotEqual(headers[3], headers[4])
        self.assertNotIn("submitblock", core.methods)
        self.assertFalse(process.killed)

    def test_ctrl_c_while_cuda_scans_and_the_tip_check_runs(self) -> None:
        def interrupt():
            raise KeyboardInterrupt

        core = FakeCore()
        process = FakeCudaProcess(self.candidate_in_second_chunk)
        result = self.run_main(self.nth_tip_check(core, 3, interrupt), live_args(chunk_size=50), process)

        self.assertEqual(result, 130)
        self.assertIn("Stopped by user", self.output.getvalue())
        self.assertEqual(process.events, [("send", "1"), ("read", "1"), ("send", "2")])
        self.assertEqual(process.pending[0].split()[1], "FOUND")
        self.assertTrue(process.stdin_closed)
        self.assertEqual(len(process.wait_timeouts), 1)
        self.assertFalse(process.killed)
        self.assertNotIn("submitblock", core.methods)
        self.popen.assert_called_once()

    @unittest.skipUnless(CUDA_MINER.is_file(), "Build cuda_miner.exe to run this vector")
    def test_real_process_overlap_abandon_and_resynchronise(self) -> None:
        vector = (
            struct.pack("<I", 0x20000000) + bytes(64)
            + struct.pack("<I", 1_728_000_000) + struct.pack("<I", 0x1F00FFFF) + bytes(4)
        )
        expected = (107938, "00009dac139e241aac5c9bfda9a7526dd697145b3147332cdbc0bd3e7ff24b42")
        hard = vector[:72] + struct.pack("<I", 0x17021EF0) + bytes(4)
        miner = CudaMiner(CUDA_MINER)
        self.addCleanup(miner.close)

        # A real scan that finds the known candidate is abandoned as stale...
        miner.start_scan(vector, 100_000, 10_000)
        process = miner.process
        miner.abandon_scan()
        # ...and the next request gets its own answer, not the abandoned one.
        miner.start_scan(hard, 0, 1_000_000)
        self.assertIsNone(miner.finish_scan())
        miner.start_scan(vector, 107_000, 1_000)
        self.assertEqual(miner.finish_scan(), expected)

        checks = []
        with redirect_stdout(io.StringIO()):
            result = find_nonce(
                miner, vector, 0x1F00FFFF, 30_000, 0, 149_999,
                lambda: checks.append(miner.pending_id) or False,
            )
        self.assertEqual(result[0], 107938)
        self.assertEqual(result[1][::-1].hex(), expected[1])
        # First check before any request; the rest while a request is in flight.
        self.assertIsNone(checks[0])
        self.assertTrue(all(pending is not None for pending in checks[1:]))
        self.assertEqual(len(checks), 4)
        self.assertIs(miner.process, process)
        miner.close()
        self.assertEqual(process.returncode, 0)


def cpu_lowest_nonce(header: bytes, start: int, count: int) -> tuple[int, str] | None:
    """Lowest nonce in [start, start + count) whose SHA256d meets the header's own nBits."""
    target = bits_to_target(struct.unpack("<I", header[72:76])[0])
    for nonce in range(start, start + count):
        digest = double_sha256(header[:76] + struct.pack("<I", nonce))
        if int.from_bytes(digest[::-1], "big") <= target:
            return nonce, digest[::-1].hex()
    return None


@unittest.skipUnless(CUDA_MINER.is_file(), "Build cuda_miner.exe to run this vector")
class CudaKernelAgainstCpuTests(unittest.TestCase):
    """The scan kernel must agree with hashlib for every target shape.

    Every expectation is computed on the CPU. The kernel decides most nonces
    from the most significant hash word alone, so the cases where that word
    equals the target's most significant word, and lower words decide, are
    pinned explicitly: each was found by a 2^32 search and occurs with
    probability 2^-32 per nonce.
    """

    # (nBits, header seed, nonce, hash qualifies)
    TOP_WORD_EQUAL = (
        (0x1D00FFFF, 65535, 3160841370, True),
        (0x1D00FFFF, 66535, 3956457915, True),
        (0x1B00FFFF, 66535, 2044930437, False),
        (0x1B00FFFF, 67535, 21119194, False),
        (0x1E00FFFF, 65535, 3842344062, True),
        (0x1E00FFFF, 66535, 1141879043, True),
        (0x1E00FF01, 65281, 1433309259, False),
        (0x1E00FF01, 66281, 2558232714, False),
        (0x1E7FFF80, 65408, 1726876227, True),
        (0x1E7FFF80, 67408, 1244040955, True),
        (0x207FFFFF, 65535, 2974232656, False),
        (0x207FFFFF, 66535, 229882729, False),
        (0x17021EF0, 7920, 4170659115, False),
        (0x17021EF0, 8920, 1485342555, False),
    )

    @classmethod
    def setUpClass(cls) -> None:
        cls.miner = CudaMiner(CUDA_MINER)

    @classmethod
    def tearDownClass(cls) -> None:
        process = cls.miner.process
        cls.miner.close()
        if process is not None:
            assert process.returncode == 0, process.returncode

    @staticmethod
    def equality_header(seed: int, bits: int) -> bytes:
        return (
            struct.pack("<I", 0x20000000)
            + hashlib.sha256(b"eq-prev-%d" % seed).digest()
            + hashlib.sha256(b"eq-merkle-%d" % seed).digest()
            + struct.pack("<II", 1_791_500_000 + seed, bits)
            + bytes(4)
        )

    def assert_scan_matches_cpu(self, header: bytes, start: int, count: int) -> None:
        self.assertEqual(
            self.miner.scan(header, start, count),
            cpu_lowest_nonce(header, start, count),
            f"bits={header[72:76][::-1].hex()} start={start} count={count}",
        )

    def test_top_word_equal_cases_fall_back_to_the_full_comparison(self) -> None:
        accepted = rejected = 0
        for bits, seed, nonce, qualifies in self.TOP_WORD_EQUAL:
            with self.subTest(bits=f"{bits:08x}", nonce=nonce):
                header = self.equality_header(seed, bits)
                target = bits_to_target(bits)
                digest = double_sha256(header[:76] + struct.pack("<I", nonce))
                value = int.from_bytes(digest[::-1], "big")
                # The vector really is an equality case, and its verdict really
                # depends on the lower words; both established on the CPU.
                self.assertEqual(value >> 224, target >> 224)
                self.assertEqual(value <= target, qualifies)

                result = self.miner.scan(header, nonce, 1)
                if qualifies:
                    self.assertEqual(result, (nonce, digest[::-1].hex()))
                    accepted += 1
                else:
                    self.assertIsNone(result)
                    rejected += 1
                self.assert_scan_matches_cpu(header, nonce - 3, 7)
        self.assertEqual((accepted, rejected), (6, 8))

    def test_randomized_headers_targets_and_ranges(self) -> None:
        rng = random.Random(234)
        pool = (
            0x207FFFFF, 0x2000FFFF, 0x1F7FFFFF, 0x1F00FFFF, 0x1F0000FF, 0x1E7FFFFF,
            0x1E00FFFF, 0x1D00FFFF, 0x1F123456, 0x20000001, 0x2012AB00, 0x1F00FF01,
        )
        found = 0
        for _ in range(60):
            bits = rng.choice(pool)
            header = rng.randbytes(72) + struct.pack("<I", bits) + bytes(4)
            start = rng.choice(
                (0, rng.randrange(1 << 31), 0xFFFFFFFF - rng.randrange(1, 5000))
            )
            count = min(
                rng.choice((1, 2, 3, 5, 7, 31, 255, 257, 1000, 4097)),
                0x100000000 - start,
            )
            with self.subTest(bits=f"{bits:08x}", start=start, count=count):
                expected = cpu_lowest_nonce(header, start, count)
                self.assertEqual(self.miner.scan(header, start, count), expected)
                found += expected is not None
        # The fixed seed gives a mix of hits and misses; both must be exercised.
        self.assertTrue(5 <= found <= 55, found)

    def test_lowest_qualifying_nonce_and_range_boundaries(self) -> None:
        header = bytes(range(72)) + struct.pack("<I", 0x1F00FFFF) + bytes(4)
        first = cpu_lowest_nonce(header, 0, 600_000)
        self.assertIsNotNone(first)
        nonce = first[0]
        second = cpu_lowest_nonce(header, nonce + 1, 300_000)

        self.assertEqual(self.miner.scan(header, 0, 600_000), first)
        self.assertEqual(self.miner.scan(header, nonce, 1), first)
        self.assertEqual(self.miner.scan(header, nonce + 1, 300_000), second)
        # The range ends one nonce before the hit, for several odd lengths
        # that do not divide evenly into the kernel's threads or loops.
        for length in (1, 2, 3, 15, 16, 17, 33, 255, 257, 1023):
            with self.subTest(length=length):
                if nonce >= length:
                    self.assertIsNone(self.miner.scan(header, nonce - length, length))
                    self.assertEqual(
                        self.miner.scan(header, nonce - length, length + 1), first
                    )

    def test_end_of_nonce_space(self) -> None:
        for bits, start, count in (
            (0x207FFFFF, 0xFFFFFFF0, 15),
            (0x207FFFFF, 0xFFFFFF00, 255),
            (0x1F7FFFFF, 0xFFFF0000, 65535),
            (0x207FFFFF, 0xFFFFFFFE, 1),
        ):
            with self.subTest(start=start, count=count):
                header = bytes(range(8, 80)) + struct.pack("<I", bits) + bytes(4)
                self.assert_scan_matches_cpu(header, start, count)

    def test_known_vector_through_every_mode(self) -> None:
        vector = (
            struct.pack("<I", 0x20000000) + bytes(64)
            + struct.pack("<II", 1_728_000_000, 0x1F00FFFF) + bytes(4)
        )
        expected = (107938, "00009dac139e241aac5c9bfda9a7526dd697145b3147332cdbc0bd3e7ff24b42")
        self.assertEqual(cpu_lowest_nonce(vector, 107938, 1), expected)
        self.assertEqual(self.miner.scan(vector, 0, 200_000), expected)
        self.assertIsNone(self.miner.scan(vector, 0, 107938))

        benchmark = subprocess.run(
            [str(CUDA_MINER)], capture_output=True, text=True, check=False, timeout=120
        )
        self.assertEqual(benchmark.returncode, 0, benchmark.stderr)
        self.assertIn("Lowest candidate nonce: 107938", benchmark.stdout)
        self.assertIn(expected[1], benchmark.stdout)


def rolled_header(header: bytes, nonce: int, variant: int) -> bytes:
    """The header with `variant` added to its BIP 320 version field (bits 13-28)."""
    version = struct.unpack("<I", header[:4])[0] + (variant << 13)
    return struct.pack("<I", version) + header[4:76] + struct.pack("<I", nonce)


def cpu_lowest_rolled(
    header: bytes, start: int, count: int, versions: int, first_variant: int = 0
) -> tuple[int, str, int] | None:
    """Lowest nonce, then lowest version variant, whose SHA256d meets the header's nBits."""
    target = bits_to_target(struct.unpack("<I", header[72:76])[0])
    for nonce in range(start, start + count):
        for variant in range(first_variant, versions):
            digest = double_sha256(rolled_header(header, nonce, variant))
            if int.from_bytes(digest[::-1], "big") <= target:
                return nonce, digest[::-1].hex(), variant
    return None


def cpu_rolled_responder(fields: list[str]) -> str:
    if fields[0] == "SCAN":
        return cpu_scan_responder(fields)
    _command, request_id, header_hex, start, count, versions = fields
    result = cpu_lowest_rolled(
        bytes.fromhex(header_hex), int(start), int(count), int(versions)
    )
    if result is None:
        return f"{request_id} NONE\n"
    return f"{request_id} FOUND {result[0]} {result[1]} {result[2]}\n"


@unittest.skipUnless(CUDA_MINER.is_file(), "Build cuda_miner.exe to run this vector")
class VersionRollingKernelTests(unittest.TestCase):
    """SCANV must agree with hashlib for every variant, nonce and target shape.

    Every expectation is computed on the CPU from the rolled 80-byte header,
    so a kernel that skipped, repeated or mislabelled a variant would differ.
    """

    VERSION_COUNTS = (2, 4, 8, 16)

    @classmethod
    def setUpClass(cls) -> None:
        cls.miner = CudaMiner(CUDA_MINER)

    @classmethod
    def tearDownClass(cls) -> None:
        process = cls.miner.process
        cls.miner.close()
        if process is not None:
            assert process.returncode == 0, process.returncode

    def assert_scan_matches_cpu(
        self, header: bytes, start: int, count: int, versions: int
    ) -> tuple[int, str, int] | None:
        expected = cpu_lowest_rolled(header, start, count, versions)
        self.assertEqual(
            self.miner.scan(header, start, count, versions),
            expected,
            f"bits={header[72:76][::-1].hex()} start={start} count={count} "
            f"versions={versions}",
        )
        return expected

    def test_randomized_headers_targets_ranges_and_version_counts(self) -> None:
        rng = random.Random(320)
        pool = (
            0x207FFFFF, 0x2000FFFF, 0x1F7FFFFF, 0x1F00FFFF, 0x1F0000FF, 0x1E7FFFFF,
            0x1E00FFFF, 0x1D00FFFF, 0x1F123456, 0x20000001, 0x2012AB00, 0x1F00FF01,
        )
        found = 0
        variants = set()
        for _ in range(80):
            bits = rng.choice(pool)
            versions = rng.choice(self.VERSION_COUNTS)
            version = rng.getrandbits(32) & ~(0xFFFF << 13)
            header = (
                struct.pack("<I", version) + rng.randbytes(68)
                + struct.pack("<I", bits) + bytes(4)
            )
            start = rng.choice(
                (0, rng.randrange(1 << 31), 0xFFFFFFFF - rng.randrange(0, 5000))
            )
            count = min(
                rng.choice((1, 2, 3, 5, 7, 31, 255, 257, 1000)),
                0x100000000 - start,
            )
            with self.subTest(bits=f"{bits:08x}", start=start, count=count, versions=versions):
                expected = self.assert_scan_matches_cpu(header, start, count, versions)
                if expected is not None:
                    found += 1
                    variants.add(expected[2])
        # The fixed seed gives hits and misses, and hits on rolled versions.
        self.assertTrue(10 <= found <= 70, found)
        self.assertGreater(len(variants), 3, variants)

    def test_every_variant_is_hashed_from_its_own_rolled_header(self) -> None:
        # About one hash in sixteen qualifies, so over single-nonce scans each
        # of the sixteen variants is, sooner or later, the lowest one that does.
        header = (
            struct.pack("<I", 0x20000000) + hashlib.sha256(b"variants").digest() * 2
            + struct.pack("<II", 1_791_500_000, 0x200FFFFF) + bytes(4)
        )
        reported = set()
        for nonce in range(400):
            expected = self.assert_scan_matches_cpu(header, nonce, 1, 16)
            if expected is not None:
                reported.add(expected[2])
        self.assertEqual(reported, set(range(16)))

    def test_fewer_versions_never_report_a_higher_variant(self) -> None:
        header = (
            struct.pack("<I", 0x20000000) + hashlib.sha256(b"prefix").digest() * 2
            + struct.pack("<II", 1_791_500_000, 0x1F7FFFFF) + bytes(4)
        )
        for versions in self.VERSION_COUNTS:
            with self.subTest(versions=versions):
                expected = self.assert_scan_matches_cpu(header, 0, 3000, versions)
                self.assertIsNotNone(expected)
                self.assertLess(expected[2], versions)

    def test_top_word_equal_cases_fall_back_to_the_full_comparison(self) -> None:
        # Variant 0 is the unrolled header, so the pinned equality cases of
        # the plain kernel must be decided the same way by this one.
        accepted = 0
        for bits, seed, nonce, qualifies in CudaKernelAgainstCpuTests.TOP_WORD_EQUAL:
            with self.subTest(bits=f"{bits:08x}", nonce=nonce):
                header = CudaKernelAgainstCpuTests.equality_header(seed, bits)
                digest = double_sha256(header[:76] + struct.pack("<I", nonce))
                for versions in (2, 16):
                    result = self.assert_scan_matches_cpu(header, nonce, 1, versions)
                    if qualifies:
                        self.assertEqual(result, (nonce, digest[::-1].hex(), 0))
                        accepted += 1
                self.assert_scan_matches_cpu(header, nonce - 3, 7, 4)
        self.assertEqual(accepted, 12)

    def test_end_of_nonce_space_is_reported_by_the_gpu(self) -> None:
        # The plain scan cannot report nonce 0xffffffff (its "nothing found"
        # value); the version scan has no such value and must report it.
        for versions in self.VERSION_COUNTS:
            with self.subTest(versions=versions):
                # Half of all hashes meet this target; take a header whose
                # last nonce is a hit, as established on the CPU.
                header = next(
                    candidate
                    for candidate in (
                        struct.pack("<I", 0x20000000) + bytes([seed]) * 68
                        + struct.pack("<I", 0x207FFFFF) + bytes(4)
                        for seed in range(256)
                    )
                    if cpu_lowest_rolled(candidate, 0xFFFFFFFF, 1, versions) is not None
                )
                expected = self.assert_scan_matches_cpu(header, 0xFFFFFFFF, 1, versions)
                self.assertEqual(expected[0], 0xFFFFFFFF)
        for bits, start, count in (
            (0x1F7FFFFF, 0xFFFFFFF0, 16),
            (0x1F7FFFFF, 0xFFFFFF00, 256),
            (0x1F00FFFF, 0xFFFFF000, 4096),
            (0x1D00FFFF, 0xFFFFFFFE, 2),
        ):
            with self.subTest(start=start, count=count):
                header = (
                    struct.pack("<I", 0x20000000) + bytes(range(8, 76))
                    + struct.pack("<I", bits) + bytes(4)
                )
                self.assert_scan_matches_cpu(header, start, count, 4)

    def test_range_boundaries_around_a_hit(self) -> None:
        header = (
            struct.pack("<I", 0x20000000) + bytes(range(68))
            + struct.pack("<I", 0x1F00FFFF) + bytes(4)
        )
        first = cpu_lowest_rolled(header, 0, 40_000, 4)
        self.assertIsNotNone(first)
        nonce = first[0]
        self.assertEqual(self.miner.scan(header, 0, 40_000, 4), first)
        self.assertEqual(self.miner.scan(header, nonce, 1, 4), first)
        # The range ends one nonce before the hit, for lengths that do not
        # divide evenly into the kernel's blocks.
        for length in (1, 2, 3, 15, 16, 17, 33, 255, 257, 1023):
            with self.subTest(length=length):
                if nonce >= length:
                    self.assertIsNone(self.miner.scan(header, nonce - length, length, 4))
                    self.assertEqual(
                        self.miner.scan(header, nonce - length, length + 1, 4), first
                    )

    def test_plain_and_version_scans_share_one_loaded_header(self) -> None:
        header = (
            struct.pack("<I", 0x20000000) + hashlib.sha256(b"shared").digest() * 2
            + struct.pack("<II", 1_791_500_000, 0x1F7FFFFF) + bytes(4)
        )
        other = header[:40] + bytes(4) + header[44:]
        plain = cpu_lowest_nonce(header, 0, 3000)
        self.assertEqual(self.miner.scan(header, 0, 3000), plain)
        self.assert_scan_matches_cpu(header, 0, 3000, 4)
        self.assertEqual(self.miner.scan(header, 0, 3000), plain)
        self.assert_scan_matches_cpu(header, 0, 3000, 8)
        # New work replaces every variant's state, not only the plain one.
        self.assert_scan_matches_cpu(other, 0, 3000, 8)
        self.assert_scan_matches_cpu(header, 0, 3000, 8)
        self.assertEqual(self.miner.scan(other, 0, 3000), cpu_lowest_nonce(other, 0, 3000))

    def test_unsupported_version_requests_fail_closed(self) -> None:
        clear = (bytes(72) + struct.pack("<I", 0x207FFFFF) + bytes(4)).hex()
        rolled = (
            struct.pack("<I", 1 << 13) + bytes(68) + struct.pack("<I", 0x207FFFFF) + bytes(4)
        ).hex()
        for request in (
            f"SCANV 1 {clear} 0 1 3",
            f"SCANV 1 {clear} 0 1 1",
            f"SCANV 1 {clear} 0 1",
            f"SCANV 1 {clear} 0 1 4 4",
            f"SCANV 1 {rolled} 0 1 4",
            f"SCAN 1 {clear} 0 1 4",
        ):
            with self.subTest(request=request[:8] + request[-8:]):
                result = subprocess.run(
                    [str(CUDA_MINER), "--serve"],
                    input=request + "\n",
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "READY\n")
                self.assertIn("Malformed request", result.stderr)


class TransactionHashTests(unittest.TestCase):
    def test_legacy_transaction_id_and_witness_id_match(self) -> None:
        txid, wtxid = transaction_hashes(bytes.fromhex(LEGACY_TRANSACTION))

        self.assertEqual(txid[::-1].hex(), LEGACY_TXID)
        self.assertEqual(wtxid, txid)

    def test_witness_transaction_id_excludes_witness_and_wtxid_includes_it(self) -> None:
        txid, wtxid = transaction_hashes(bytes.fromhex(WITNESS_TRANSACTION))

        self.assertEqual(txid[::-1].hex(), LEGACY_TXID)
        self.assertEqual(wtxid[::-1].hex(), WITNESS_WTXID)

    def test_template_transaction_hashes_are_independently_checked(self) -> None:
        template = {
            "transactions": [
                {
                    "data": LEGACY_TRANSACTION,
                    "txid": LEGACY_TXID,
                    "hash": LEGACY_TXID,
                }
            ]
        }
        transaction_data, txids = template_transactions(template)

        self.assertEqual(transaction_data, [bytes.fromhex(LEGACY_TRANSACTION)])
        self.assertEqual(txids, [bytes.fromhex(LEGACY_TXID)[::-1]])

        template["transactions"][0]["txid"] = "00" * 32
        with self.assertRaisesRegex(ValueError, "txid does not match"):
            template_transactions(template)

        template["transactions"][0]["txid"] = LEGACY_TXID
        template["transactions"][0]["hash"] = "00" * 32
        with self.assertRaisesRegex(ValueError, "witness hash does not match"):
            template_transactions(template)

    def test_rejects_trailing_or_malformed_transaction_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "trailing data"):
            transaction_hashes(bytes.fromhex(LEGACY_TRANSACTION) + b"\x00")
        with self.assertRaisesRegex(ValueError, "Non-canonical CompactSize"):
            read_compact_size(bytes.fromhex("fdfc00"), 0)


class BitcoinProtocolTests(unittest.TestCase):
    def test_sha256d_known_empty_input_vector(self) -> None:
        self.assertEqual(
            double_sha256(b"").hex(),
            "5df6e0e2761359d30a8275058e299fcc0381534545f55cf43e41983f5d4c9456",
        )

    def test_compact_target_conversion_and_round_trip(self) -> None:
        target = bits_to_target(0x1D00FFFF)

        self.assertEqual(target, int(GENESIS_TARGET, 16))
        self.assertEqual(target_to_bits(target), 0x1D00FFFF)

    def test_target_comparison_accepts_equality(self) -> None:
        raw_hash = bytes.fromhex("12" * 32)
        hash_value = int.from_bytes(raw_hash[::-1], "big")

        self.assertTrue(hash_meets_target(raw_hash, hash_value))
        self.assertFalse(hash_meets_target(raw_hash, hash_value - 1))

    def test_genesis_header_serialization_and_hash(self) -> None:
        header = build_header(
            1,
            "00" * 32,
            GENESIS_MERKLE_ROOT,
            1_231_006_505,
            0x1D00FFFF,
            2_083_236_893,
        )

        self.assertEqual(header.hex(), GENESIS_HEADER)
        self.assertEqual(len(header), 80)
        self.assertEqual(
            double_sha256(header)[::-1].hex(),
            "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f",
        )

    def test_bip34_height_script_numbers(self) -> None:
        self.assertEqual(script_number(1), b"\x51")
        self.assertEqual(script_number(17), bytes.fromhex("0111"))
        self.assertEqual(script_number(128), bytes.fromhex("028000"))

    def test_coinbase_legacy_serialization(self) -> None:
        coinbase, txid = create_coinbase(
            1,
            b"",
            0,
            5_000_000_000,
            b"\x51",
            None,
        )

        self.assertEqual(coinbase.hex(), COINBASE_TRANSACTION)
        self.assertEqual(txid, double_sha256(coinbase))

    def test_extranonce_rebuilds_coinbase_merkle_root_and_header(self) -> None:
        template = valid_template()
        coinbase_zero, txid_zero = create_coinbase(
            1,
            b"",
            0,
            5_000_000_000,
            b"\x51",
            None,
        )
        coinbase_one, txid_one = create_coinbase(
            1,
            b"",
            1,
            5_000_000_000,
            b"\x51",
            None,
        )
        header_zero = build_template_header(template, txid_zero, [])
        header_one = build_template_header(template, txid_one, [])

        self.assertNotEqual(coinbase_zero, coinbase_one)
        self.assertNotEqual(merkle_root([txid_zero]), merkle_root([txid_one]))
        self.assertNotEqual(header_zero, header_one)

    def test_coinbase_witness_serialization(self) -> None:
        commitment = bytes.fromhex(WITNESS_COMMITMENT)
        full_coinbase, txid = create_coinbase(
            1,
            b"",
            0,
            5_000_000_000,
            b"\x51",
            commitment,
        )
        script_sig = b"\x51" + bytes(8)
        tx_input = (
            b"\x01"
            + bytes(32)
            + b"\xff" * 4
            + bytes((len(script_sig),))
            + script_sig
            + b"\xff" * 4
        )
        tx_outputs = (
            b"\x02"
            + struct.pack("<Q", 5_000_000_000)
            + b"\x01\x51"
            + bytes(8)
            + bytes((len(commitment),))
            + commitment
        )
        stripped_coinbase = struct.pack("<i", 2) + tx_input + tx_outputs + bytes(4)
        expected_full = (
            stripped_coinbase[:4]
            + b"\x00\x01"
            + tx_input
            + tx_outputs
            + b"\x01\x20"
            + bytes(32)
            + bytes(4)
        )

        self.assertEqual(full_coinbase, expected_full)
        self.assertEqual(txid, double_sha256(stripped_coinbase))

    def test_merkle_root_and_odd_leaf_duplication(self) -> None:
        leaves = [bytes.fromhex(byte * 32) for byte in ("01", "02", "03")]
        duplicated_leaves = leaves + [leaves[-1]]

        self.assertEqual(
            merkle_root(leaves).hex(),
            "223e023fadf1f053df26988871f893c821c28edf77d64a955e6c2a02d547bdac",
        )
        self.assertEqual(merkle_root(leaves), merkle_root(duplicated_leaves))

    def test_witness_merkle_root_and_commitment(self) -> None:
        witness_txid = bytes.fromhex(WITNESS_WTXID)[::-1]
        self.assertEqual(
            merkle_root([bytes(32), witness_txid]).hex(),
            WITNESS_ROOT,
        )

        transaction = {
            "data": WITNESS_TRANSACTION,
            "txid": LEGACY_TXID,
            "hash": WITNESS_WTXID,
        }
        template = {
            "default_witness_commitment": WITNESS_COMMITMENT,
        }
        self.assertEqual(
            get_witness_commitment(template, [transaction]),
            bytes.fromhex(WITNESS_COMMITMENT),
        )

    def test_full_block_serialization(self) -> None:
        header = bytes.fromhex(GENESIS_HEADER)
        coinbase = bytes.fromhex(COINBASE_TRANSACTION)
        transaction = bytes.fromhex(LEGACY_TRANSACTION)
        expected = (
            header
            + b"\x02"
            + coinbase
            + transaction
        )

        self.assertEqual(
            serialize_block(header, coinbase, [transaction]),
            expected,
        )
        with self.assertRaisesRegex(ValueError, "header must be 80 bytes"):
            serialize_block(b"", coinbase, [])

    @unittest.skipUnless(CUDA_MINER.is_file(), "Build cuda_miner.exe to run this vector")
    def test_cuda_known_sha256d_vector(self) -> None:
        header = (
            struct.pack("<I", 0x20000000)
            + bytes(64)
            + struct.pack("<I", 1_728_000_000)
            + struct.pack("<I", 0x1F00FFFF)
            + bytes(4)
        )
        result = subprocess.run(
            [
                str(CUDA_MINER),
                "--scan-header",
                header.hex(),
                "--start",
                "107938",
                "--count",
                "1",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            "FOUND 107938 "
            "00009dac139e241aac5c9bfda9a7526dd697145b3147332cdbc0bd3e7ff24b42",
        )


class ExtranonceRollingTests(unittest.TestCase):
    def run_block(self, core, process, session=None, **kwargs):
        self.output = io.StringIO()
        with (
            patch("regtest_miner.rpc", side_effect=core),
            patch("regtest_miner.subprocess.Popen", return_value=process),
            patch("regtest_miner.save_unsubmitted_block", return_value=None),
            redirect_stdout(self.output),
        ):
            miner = CudaMiner(Path("cuda_miner.exe"))
            return mine_one_block(
                Path("bitcoin-cli.exe"), miner, "mainnet", None, None, 50, 7,
                bytes.fromhex(PAYOUT_SCRIPT), live_mainnet=True,
                payout_address=PAYOUT_ADDRESS, session=session, **kwargs,
            )

    @staticmethod
    def small_range_template() -> dict:
        template = easy_template()
        template["noncerange"] = "0000000000000031"
        return template

    def test_exhausted_nonce_space_rolls_the_extranonce_on_the_same_template(self) -> None:
        core = FakeCore(self.small_range_template())

        def third_request_finds(fields):
            return f"{fields[1]} NONE\n" if fields[1] in ("1", "2") else cpu_scan_responder(fields)

        process = FakeCudaProcess(third_request_finds)
        self.assertTrue(self.run_block(core, process))

        self.assertEqual(core.methods.count("getblocktemplate"), 1)
        headers = [bytes.fromhex(request.split()[2]) for request in process.requests]
        self.assertEqual(len(headers), 3)
        # Same version, previous block, time and bits; only the Merkle root moves.
        self.assertEqual(len({header[:36] + header[68:76] for header in headers}), 1)
        self.assertEqual(len({header[36:68] for header in headers}), 3)
        self.assertEqual([request.split()[3:] for request in process.requests], [["0", "50"]] * 3)

        block = bytes.fromhex(core.submitted[0])
        self.assertEqual(block[:76], headers[2][:76])
        legacy = bytes.fromhex(LEGACY_TRANSACTION)
        coinbase = block[81 : len(block) - len(legacy)]
        # The rolled coinbase still pays everything to the validated script.
        self.assertEqual(
            transaction_outputs(coinbase),
            [(5_000_000_000, bytes.fromhex(PAYOUT_SCRIPT))],
        )
        self.assertEqual(
            block[36:68],
            merkle_root([double_sha256(coinbase), bytes.fromhex(LEGACY_TXID)[::-1]]),
        )
        # extra_nonce 7 from the caller in the high half, roll 2 in the low half.
        self.assertIn(struct.pack("<Q", (7 << 32) | 2), coinbase)

    def test_rollover_starts_the_gpu_before_its_tip_check_and_still_honours_it(self) -> None:
        core = FakeCore(self.small_range_template())
        previous = core.template["previousblockhash"]
        # Template check, first-chunk check, then the check that overlaps the
        # first chunk after the rollover reports a new block.
        core.tips = [previous, previous, "ff" * 32]
        order = []
        original = core.__call__

        def recording(cli, network, datadir, conf, method, *params, **kwargs):
            if method == "getbestblockhash":
                order.append("check")
            return original(cli, network, datadir, conf, method, *params, **kwargs)

        process = FakeCudaProcess(
            lambda fields: order.append(f"scan {fields[1]}") or (
                f"{fields[1]} NONE\n" if fields[1] == "1" else cpu_scan_responder(fields)
            )
        )
        self.assertFalse(self.run_block(recording, process))

        # Before the rollover the check precedes the scan; after it the scan
        # is already running when the check is made.
        self.assertEqual(order, ["check", "check", "scan 1", "scan 2", "check"])
        # The rolled chunk held a valid candidate; a stale verdict abandons it.
        self.assertEqual(process.pending[0].split()[1], "FOUND")
        self.assertNotIn(("read", "2"), process.events)
        self.assertNotIn("submitblock", core.methods)
        self.assertIn("Template became stale", self.output.getvalue())

    def test_old_template_is_replaced_instead_of_rolled(self) -> None:
        core = FakeCore(self.small_range_template())
        process = FakeCudaProcess()
        with patch("regtest_miner.TEMPLATE_REFRESH_SECONDS", 0.0):
            self.assertFalse(self.run_block(core, process))

        self.assertEqual(len(process.requests), 1)
        self.assertIn("Nonce space exhausted", self.output.getvalue())
        self.assertNotIn("submitblock", core.methods)

    def session(self, prefetcher):
        return MiningSession(
            Path("bitcoin-cli.exe"), None, None, NodeMonitor(lambda: None, 60.0), prefetcher
        )

    def test_prefetched_template_is_adopted_at_rollover_without_idle_rpc(self) -> None:
        core = FakeCore(self.small_range_template())
        fresh = dict(self.small_range_template(), curtime=1_700_000_030)
        with patch("regtest_miner.rpc", side_effect=FakeCore(fresh)):
            prepared = fetch_work(Path("bitcoin-cli.exe"), "mainnet", None, None)
        prefetcher = TemplatePrefetcher(lambda: prepared)
        session = self.session(prefetcher)
        process = FakeCudaProcess()

        def ready_after_first_scan(fields):
            prefetcher.result = prepared
            return f"{fields[1]} NONE\n"

        process.responder = ready_after_first_scan
        self.assertFalse(self.run_block(core, process, session=session))
        self.assertIn("switching to the refreshed template", self.output.getvalue())
        self.assertEqual(len(process.requests), 1)

        # The next call uses the prepared work: no getblocktemplate at all,
        # and its first chunk starts before the tip check because the same
        # tip was confirmed during the previous chunk.
        self.assertTrue(session.tip_confirmed_recently(core.template["previousblockhash"]))
        core.methods.clear()
        process = FakeCudaProcess(
            lambda fields: core.methods.append("scan") or cpu_scan_responder(fields)
        )
        self.assertTrue(self.run_block(core, process, session=session))
        self.assertNotIn("getblocktemplate", core.methods)
        self.assertEqual(core.methods[:2], ["scan", "getbestblockhash"])
        header = bytes.fromhex(process.requests[0].split()[2])
        self.assertEqual(struct.unpack("<I", header[68:72])[0], 1_700_000_030)
        self.assertFalse(prefetcher.ready())

    def test_stale_tip_discards_a_prefetched_template(self) -> None:
        core = FakeCore(self.small_range_template())
        with patch("regtest_miner.rpc", side_effect=FakeCore(self.small_range_template())):
            prepared = fetch_work(Path("bitcoin-cli.exe"), "mainnet", None, None)
        prefetcher = TemplatePrefetcher(lambda: prepared)
        prefetcher.result = prepared
        session = self.session(prefetcher)
        # The prepared work is taken, and its very first tip check is stale.
        core.tips = ["ff" * 32]
        process = FakeCudaProcess()

        self.assertFalse(self.run_block(core, process, session=session))
        self.assertEqual(process.requests, [])
        self.assertIn("Template became stale", self.output.getvalue())
        self.assertFalse(prefetcher.ready())

    def test_prefetch_runs_beside_scanning_without_touching_the_cuda_protocol(self) -> None:
        template = dict(self.small_range_template(), noncerange="00000000000000c7")
        core = FakeCore(template)
        with patch("regtest_miner.rpc", side_effect=FakeCore(dict(template, curtime=1_700_000_030))):
            prepared = fetch_work(Path("bitcoin-cli.exe"), "mainnet", None, None)
        in_fetch = threading.Event()
        release = threading.Event()
        fetch_threads = []

        def slow_fetch():
            fetch_threads.append(threading.current_thread())
            in_fetch.set()
            release.wait(5)
            return prepared

        prefetcher = TemplatePrefetcher(slow_fetch)
        session = self.session(prefetcher)
        scans = 0

        def responder(fields):
            nonlocal scans
            scans += 1
            if scans == 2:
                # A refresh is in progress on another thread during this scan.
                self.assertTrue(in_fetch.wait(5))
                self.assertFalse(prefetcher.ready())
            if scans == 4:
                release.set()
                prefetcher.thread.join(5)
            return f"{fields[1]} NONE\n"

        process = FakeCudaProcess(responder)
        # Every tip check asks for a refresh; only one fetch may ever run.
        with patch("regtest_miner.TEMPLATE_REFRESH_SECONDS", 0.0), patch.object(
            TemplatePrefetcher, "RETRY_SECONDS", 0.0
        ):
            self.assertFalse(self.run_block(core, process, session=session))

        self.assertEqual(scans, 4)
        self.assertEqual(len(fetch_threads), 1)
        self.assertIsNot(fetch_threads[0], threading.current_thread())
        # Strict request/reply alternation with consecutive ids throughout.
        self.assertEqual(
            process.events,
            [(kind, str(number)) for number in range(1, 5) for kind in ("send", "read")],
        )
        self.assertEqual(len({request.split()[2] for request in process.requests}), 1)
        self.assertIn("switching to the refreshed template", self.output.getvalue())
        self.assertIs(prefetcher.take(), prepared)

    def test_prefetch_failure_is_contained_and_not_restarted_in_a_tight_loop(self) -> None:
        calls = []

        def failing():
            calls.append(1)
            raise RpcError("Bitcoin Core RPC getblocktemplate failed")

        prefetcher = TemplatePrefetcher(failing)
        prefetcher.request()
        prefetcher.thread.join(5)
        prefetcher.request()
        prefetcher.request()

        self.assertEqual(len(calls), 1)
        self.assertFalse(prefetcher.ready())
        self.assertIsNone(prefetcher.take())


class VersionRollingTests(unittest.TestCase):
    """--version-rolling: the runner's side of SCANV, with every RPC mocked."""

    HEADER = bytes(4) + bytes(range(4, 80))

    def run_block(self, core, process, **kwargs):
        self.output = io.StringIO()
        with (
            patch("regtest_miner.rpc", side_effect=core),
            patch("regtest_miner.subprocess.Popen", return_value=process),
            patch("regtest_miner.save_unsubmitted_block", return_value=None),
            redirect_stdout(self.output),
        ):
            miner = CudaMiner(Path("cuda_miner.exe"))
            return mine_one_block(
                Path("bitcoin-cli.exe"), miner, "mainnet", None, None, 50, 7,
                bytes.fromhex(PAYOUT_SCRIPT), live_mainnet=True,
                payout_address=PAYOUT_ADDRESS, **kwargs,
            )

    @staticmethod
    def rolled_only_responder(fields: list[str]) -> str:
        """Report the lowest hit on a rolled version, never on the template's own."""
        _command, request_id, header_hex, start, count, versions = fields
        nonce, displayed, variant = cpu_lowest_rolled(
            bytes.fromhex(header_hex), int(start), int(count), int(versions), 1
        )
        return f"{request_id} FOUND {nonce} {displayed} {variant}\n"

    def test_flag_is_off_by_default_and_parsed(self) -> None:
        with patch("sys.argv", ["regtest_miner.py"]):
            self.assertFalse(parse_args().version_rolling)
        with patch("sys.argv", ["regtest_miner.py", "--version-rolling"]):
            self.assertTrue(parse_args().version_rolling)

    def test_request_and_reply_name_the_version_variant(self) -> None:
        process = FakeCudaProcess(lambda fields: f"{fields[1]} FOUND 7 {'ab' * 32} 3\n")
        with patch("regtest_miner.subprocess.Popen", return_value=process):
            miner = CudaMiner(Path("cuda_miner.exe"))
            self.assertEqual(miner.scan(self.HEADER, 0, 10, 4), (7, "ab" * 32, 3))
        self.assertEqual(process.requests, [f"SCANV 1 {self.HEADER.hex()} 0 10 4\n"])

    def test_reply_without_a_valid_variant_fails_closed(self) -> None:
        for versions, reply in (
            (4, f"FOUND 7 {'ab' * 32}"),
            (4, f"FOUND 7 {'ab' * 32} 4"),
            (4, f"FOUND 7 {'ab' * 32} -1"),
            (4, f"FOUND 7 {'ab' * 32} x"),
            (4, f"FOUND 7 {'ab' * 32} 1 1"),
            (1, f"FOUND 7 {'ab' * 32} 0"),
        ):
            with self.subTest(versions=versions, reply=reply[-6:]):
                process = FakeCudaProcess(lambda fields, reply=reply: f"{fields[1]} {reply}\n")
                with patch("regtest_miner.subprocess.Popen", return_value=process):
                    miner = CudaMiner(Path("cuda_miner.exe"))
                    with self.assertRaisesRegex(RuntimeError, "Unexpected CUDA miner output"):
                        miner.scan(self.HEADER, 0, 10, versions)
                self.assertTrue(miner.failed)
                self.assertTrue(process.killed)

    def test_nonce_range_is_covered_once_for_all_variants(self) -> None:
        process = FakeCudaProcess()
        with (
            patch("regtest_miner.subprocess.Popen", return_value=process),
            redirect_stdout(io.StringIO()) as output,
        ):
            miner = CudaMiner(Path("cuda_miner.exe"))
            hard = self.HEADER[:72] + struct.pack("<I", 0x1D00FFFF) + bytes(4)
            # 100 hashes per chunk is 25 nonces of 4 versions each.
            self.assertIsNone(
                find_nonce(miner, hard, 0x1D00FFFF, 100, 0, 109, lambda: False, version_count=4)
            )
            # Through the last nonce: the GPU reports it, no CPU check is needed.
            easy = self.HEADER[:72] + struct.pack("<I", 0x207FFFFF) + bytes(4)
            self.assertIsNone(
                find_nonce(
                    miner, easy, 0x207FFFFF, 100, 0xFFFFFFF0, 0xFFFFFFFF, lambda: False,
                    version_count=4,
                )
            )
        self.assertEqual(
            [request.split()[0] for request in process.requests], ["SCANV"] * 6
        )
        self.assertEqual(
            [tuple(map(int, request.split()[3:])) for request in process.requests],
            [
                (0, 25, 4), (25, 25, 4), (50, 25, 4), (75, 25, 4), (100, 10, 4),
                (0xFFFFFFF0, 16, 4),
            ],
        )
        # The status line counts hashes, not nonces.
        self.assertIn("| 100 hashes (4 versions per nonce) |", output.getvalue())
        self.assertIn("| 64 hashes (4 versions per nonce) |", output.getvalue())

    def test_rolled_block_carries_the_variant_version_and_passes_every_check(self) -> None:
        core = FakeCore()
        process = FakeCudaProcess(self.rolled_only_responder)
        self.assertTrue(self.run_block(core, process, version_rolling=True))

        count = regtest_miner.VERSION_ROLL_COUNT
        request = process.requests[0].split()
        self.assertEqual(request[0], "SCANV")
        self.assertEqual(request[3:], ["0", str(50 // count), str(count)])
        requested = bytes.fromhex(request[2])
        # The request holds the template's own version; the GPU rolls it.
        self.assertEqual(struct.unpack("<I", requested[:4])[0], core.template["version"])

        block = bytes.fromhex(core.submitted[0])
        variant = (struct.unpack("<I", block[:4])[0] - core.template["version"]) >> 13
        self.assertIn(variant, range(1, count))
        self.assertEqual(
            block[:4], struct.pack("<I", core.template["version"] + (variant << 13))
        )
        # Nothing but the version and the nonce differs from the requested header.
        self.assertEqual(block[4:76], requested[4:76])
        self.assertLessEqual(
            int.from_bytes(double_sha256(block[:80])[::-1], "big"),
            bits_to_target(0x207FFFFF),
        )
        legacy = bytes.fromhex(LEGACY_TRANSACTION)
        coinbase = block[81 : len(block) - len(legacy)]
        self.assertEqual(
            transaction_outputs(coinbase),
            [(5_000_000_000, bytes.fromhex(PAYOUT_SCRIPT))],
        )
        self.assertEqual(
            block[36:68],
            merkle_root([double_sha256(coinbase), bytes.fromhex(LEGACY_TXID)[::-1]]),
        )
        self.assertIn(f"version={block[:4][::-1].hex()}", self.output.getvalue())
        self.assertIn(f"versions per nonce={count})", self.output.getvalue())

    def test_hash_reported_for_the_wrong_variant_is_never_submitted(self) -> None:
        core = FakeCore()

        def wrong_variant(fields):
            reply = self.rolled_only_responder(fields).split()
            return " ".join(reply[:4] + [str(int(reply[4]) - 1)]) + "\n"

        process = FakeCudaProcess(wrong_variant)
        with self.assertRaisesRegex(RuntimeError, "does not match the CPU"):
            self.run_block(core, process, version_rolling=True)
        self.assertNotIn("submitblock", core.methods)

    def test_template_that_uses_the_rolled_bits_is_mined_unrolled(self) -> None:
        core = FakeCore(dict(easy_template(), version=0x20000000 | (1 << 20)))
        process = FakeCudaProcess(cpu_rolled_responder)
        self.assertTrue(self.run_block(core, process, version_rolling=True))

        self.assertEqual([request.split()[0] for request in process.requests], ["SCAN"])
        self.assertEqual(process.requests[0].split()[3:], ["0", "50"])
        self.assertIn("version rolling is off for this template", self.output.getvalue())
        self.assertNotIn("versions per nonce", self.output.getvalue())
        block = bytes.fromhex(core.submitted[0])
        self.assertEqual(block[:4], struct.pack("<I", 0x20000000 | (1 << 20)))

    def test_without_the_flag_requests_and_blocks_are_unchanged(self) -> None:
        core = FakeCore()
        process = FakeCudaProcess(cpu_rolled_responder)
        self.assertTrue(self.run_block(core, process))

        self.assertEqual([request.split()[0] for request in process.requests], ["SCAN"])
        self.assertEqual(process.requests[0].split()[3:], ["0", "50"])
        self.assertEqual(
            bytes.fromhex(core.submitted[0])[:4], struct.pack("<I", core.template["version"])
        )
        found_line = self.output.getvalue().split("CANDIDATE FOUND")[1].splitlines()[0]
        self.assertNotIn("version", found_line)

    def test_exhausted_nonce_space_rolls_the_extranonce_with_versions(self) -> None:
        count = regtest_miner.VERSION_ROLL_COUNT
        per_chunk = 50 // count
        template = easy_template()
        # Two chunks of nonces per header.
        template["noncerange"] = f"00000000{2 * per_chunk - 1:08x}"

        def third_request_finds(fields):
            if fields[1] in ("1", "2"):
                return f"{fields[1]} NONE\n"
            return cpu_rolled_responder(fields)

        core = FakeCore(template)
        process = FakeCudaProcess(third_request_finds)
        self.assertTrue(self.run_block(core, process, version_rolling=True))

        headers = [bytes.fromhex(request.split()[2]) for request in process.requests]
        # Two chunks exhaust the header, then a new coinbase gives a new
        # Merkle root and the range starts again.
        self.assertEqual(
            [request.split()[3:] for request in process.requests],
            [
                ["0", str(per_chunk), str(count)],
                [str(per_chunk), str(per_chunk), str(count)],
                ["0", str(per_chunk), str(count)],
            ],
        )
        self.assertEqual(headers[0], headers[1])
        self.assertNotEqual(headers[0][36:68], headers[2][36:68])
        self.assertEqual(headers[0][:36] + headers[0][68:], headers[2][:36] + headers[2][68:])
        self.assertEqual(core.methods.count("getblocktemplate"), 1)

    def test_candidate_in_an_abandoned_version_chunk_is_saved_with_its_version(self) -> None:
        core = FakeCore()
        previous = core.template["previousblockhash"]
        core.tips = [previous, previous, "ff" * 32]

        def responder(fields):
            if fields[1] in ("1", "3"):
                return f"{fields[1]} NONE\n"
            return self.rolled_only_responder(fields)

        process = FakeCudaProcess(responder)
        with (
            patch("regtest_miner.rpc", side_effect=core),
            patch("regtest_miner.subprocess.Popen", return_value=process),
            patch("regtest_miner.save_unsubmitted_block", return_value=Path("saved.hex")) as save,
            patch("regtest_miner.flush_saved_block"),
            redirect_stdout(io.StringIO()) as output,
        ):
            miner = CudaMiner(Path("cuda_miner.exe"))
            arguments = (
                Path("bitcoin-cli.exe"), miner, "mainnet", None, None, 50, 0,
                bytes.fromhex(PAYOUT_SCRIPT),
            )
            self.assertFalse(mine_one_block(*arguments, live_mainnet=True, version_rolling=True))
            save.assert_not_called()
            core.template = dict(core.template, curtime=core.template["curtime"] + 1)
            self.assertTrue(
                mine_one_block(
                    *arguments, live_mainnet=True, payout_address=PAYOUT_ADDRESS,
                    version_rolling=True,
                )
            )

        (stale_hash, stale_block), (_fresh_hash, fresh_block) = (
            call.args for call in save.call_args_list
        )
        stale_request = bytes.fromhex(process.requests[1].split()[2])
        self.assertEqual(stale_block[4:76], stale_request[4:76])
        self.assertNotEqual(stale_block[:4], stale_request[:4])
        self.assertEqual(stale_hash, double_sha256(stale_block[:80])[::-1].hex())
        self.assertLessEqual(int(stale_hash, 16), bits_to_target(0x207FFFFF))
        self.assertEqual(core.submitted, [fresh_block.hex()])
        self.assertIn(f"STALE CANDIDATE: block {stale_hash}", output.getvalue())

    def test_main_passes_the_flag_to_the_scan(self) -> None:
        core = FakeCore()
        process = FakeCudaProcess(cpu_rolled_responder)
        with (
            patch("regtest_miner.parse_args", return_value=live_args(version_rolling=True)),
            patch("regtest_miner.ensure_tor_ready", return_value=None),
            patch("regtest_miner.rpc", side_effect=core),
            patch("regtest_miner.subprocess.Popen", return_value=process),
            patch("regtest_miner.save_unsubmitted_block", return_value=None),
            patch("regtest_miner.monitor_block", return_value=0),
            patch("regtest_miner.keep_system_awake"),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(main(), 0)
        self.assertTrue(process.requests[0].startswith("SCANV "))
        self.assertTrue(process.requests[0].endswith(f" {regtest_miner.VERSION_ROLL_COUNT}\n"))
        self.assertEqual(len(core.submitted), 1)


class StaleCandidatePreservationTests(unittest.TestCase):
    def test_candidate_in_an_abandoned_chunk_is_saved_and_never_submitted(self) -> None:
        core = FakeCore()
        previous = core.template["previousblockhash"]
        # Chunk 2 holds a valid block when its tip check reports a new block.
        core.tips = [previous, previous, "ff" * 32]

        def responder(fields):
            return f"{fields[1]} NONE\n" if fields[1] in ("1", "3") else cpu_scan_responder(fields)

        process = FakeCudaProcess(responder)
        output = io.StringIO()
        with (
            patch("regtest_miner.rpc", side_effect=core),
            patch("regtest_miner.subprocess.Popen", return_value=process),
            patch("regtest_miner.save_unsubmitted_block", return_value=Path("saved.hex")) as save,
            patch("regtest_miner.flush_saved_block"),
            redirect_stdout(output),
        ):
            miner = CudaMiner(Path("cuda_miner.exe"))
            arguments = (
                Path("bitcoin-cli.exe"), miner, "mainnet", None, None, 50, 0,
                bytes.fromhex(PAYOUT_SCRIPT),
            )
            self.assertFalse(mine_one_block(*arguments, live_mainnet=True))
            self.assertNotIn(("read", "2"), process.events)
            save.assert_not_called()

            core.template = dict(core.template, curtime=core.template["curtime"] + 1)
            self.assertTrue(
                mine_one_block(*arguments, live_mainnet=True, payout_address=PAYOUT_ADDRESS)
            )

        self.assertEqual(save.call_count, 2)
        (stale_hash, stale_block), (fresh_hash, fresh_block) = (
            call.args for call in save.call_args_list
        )
        stale_request = bytes.fromhex(process.requests[1].split()[2])
        self.assertEqual(stale_block[:76], stale_request[:76])
        self.assertEqual(stale_hash, double_sha256(stale_block[:80])[::-1].hex())
        self.assertLessEqual(int(stale_hash, 16), bits_to_target(0x207FFFFF))
        # Only the block on the current tip was sent to Core.
        self.assertEqual(core.submitted, [fresh_block.hex()])
        self.assertNotEqual(stale_hash, fresh_hash)
        text = output.getvalue()
        self.assertIn(f"STALE CANDIDATE: block {stale_hash}", text)
        self.assertLess(text.index("STALE CANDIDATE"), text.rindex("Mining mainnet block"))
        # The abandoned reply was read before the next request was sent.
        self.assertEqual(
            process.events[2:6], [("send", "2"), ("read", "2"), ("send", "3"), ("read", "3")]
        )

    def test_abandoned_candidate_that_fails_cpu_verification_fails_closed(self) -> None:
        core = FakeCore()
        previous = core.template["previousblockhash"]
        core.tips = [previous, previous, "ff" * 32]

        def responder(fields):
            return f"{fields[1]} NONE\n" if fields[1] == "1" else f"{fields[1]} FOUND 60 {'00' * 32}\n"

        process = FakeCudaProcess(responder)
        with (
            patch("regtest_miner.rpc", side_effect=core),
            patch("regtest_miner.subprocess.Popen", return_value=process),
            patch("regtest_miner.save_unsubmitted_block", return_value=None) as save,
            redirect_stdout(io.StringIO()),
        ):
            miner = CudaMiner(Path("cuda_miner.exe"))
            arguments = (
                Path("bitcoin-cli.exe"), miner, "mainnet", None, None, 50, 0,
                bytes.fromhex(PAYOUT_SCRIPT),
            )
            self.assertFalse(mine_one_block(*arguments, live_mainnet=True))
            with self.assertRaisesRegex(RuntimeError, "does not match the CPU"):
                mine_one_block(*arguments, live_mainnet=True)

        save.assert_not_called()
        self.assertNotIn("submitblock", core.methods)


class MonitoringAndRecoveryTests(unittest.TestCase):
    def test_rpc_transport_failures_are_rpc_errors(self) -> None:
        failed = subprocess.CompletedProcess([], 1, "", "error: Could not connect to the server")
        with patch("regtest_miner.subprocess.run", return_value=failed):
            with self.assertRaises(RpcError):
                rpc(Path("bitcoin-cli.exe"), "mainnet", None, None, "getbestblockhash")
        self.assertTrue(issubclass(RpcError, RuntimeError))

    def health(self, chain=None, network=None, peers=None, tor=True, **kwargs):
        chain = chain or {"chain": "main", "initialblockdownload": False}
        network = network or {"networkactive": True, "connections": 8}

        def fake_rpc(_cli, _network, _datadir, _conf, method, *_params, **_kwargs):
            return {"getblockchaininfo": chain, "getnetworkinfo": network, "getpeerinfo": peers}[method]

        with (
            patch("regtest_miner.socks5_ready", return_value=tor),
            patch("regtest_miner.rpc", side_effect=fake_rpc),
        ):
            check_node_health(
                Path("bitcoin-cli.exe"), None, None, "127.0.0.1", 9150,
                kwargs.get("min_peers", 1), kwargs.get("require_onion_peers", False),
            )

    def test_health_check_passes_and_names_each_failure(self) -> None:
        self.health()
        self.health(min_peers=8)
        self.health(peers=[{"network": "onion"}], require_onion_peers=True)
        cases = {
            "Tor SOCKS5 proxy": dict(tor=False),
            "main chain": dict(chain={"chain": "test", "initialblockdownload": False}),
            "initial block download": dict(chain={"chain": "main", "initialblockdownload": True}),
            "networking is inactive": dict(network={"networkactive": False, "connections": 8}),
            "0 connected peer": dict(network={"networkactive": True, "connections": 0}),
            "9 required": dict(min_peers=9),
            "no connected onion peer": dict(peers=[{"network": "ipv4"}], require_onion_peers=True),
        }
        for message, arguments in cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    self.health(**arguments)

    def test_monitor_needs_consecutive_failures_and_recovers(self) -> None:
        outcomes = []

        def check():
            outcome = outcomes.pop(0)
            if outcome is not None:
                raise outcome

        monitor = NodeMonitor(check, 60.0)
        outcomes[:] = [RpcError("connection refused")]
        monitor.poll()
        self.assertIsNone(monitor.reason())          # one failure is transient
        outcomes[:] = [None, RuntimeError("no peers"), RuntimeError("no peers")]
        monitor.poll()
        monitor.poll()
        self.assertIsNone(monitor.reason())          # the success in between reset it
        monitor.poll()
        self.assertIn("2 consecutive health checks failed: no peers", monitor.reason())
        monitor.reset()
        self.assertIsNone(monitor.reason())

    def test_monitor_thread_polls_on_its_interval_and_stops(self) -> None:
        polled = threading.Event()
        monitor = NodeMonitor(polled.set, 0.01)
        monitor.start()
        self.assertTrue(polled.wait(5))
        monitor.close()
        self.assertFalse(monitor.thread.is_alive())

    def test_recovery_backs_off_then_returns(self) -> None:
        outcomes = [RpcError("refused"), RuntimeError("no peers"), None]

        def check():
            outcome = outcomes.pop(0)
            if outcome is not None:
                raise outcome

        output = io.StringIO()
        with patch("regtest_miner.time.sleep") as sleep, redirect_stdout(output):
            wait_for_recovery(check, 3600.0)

        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5.0, 10.0])
        self.assertEqual(output.getvalue().count("[MONITOR] Not ready"), 2)

    def test_recovery_is_bounded(self) -> None:
        def check():
            raise RpcError("refused")

        with patch("regtest_miner.time.sleep") as sleep, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "did not recover within 30 seconds"):
                wait_for_recovery(check, 30.0)
        # 5 + 10 + 20 fits in 30 s only up to the third wait; the fourth would not.
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5.0, 10.0, 20.0])

    def test_tip_check_failures_are_counted_not_treated_as_stale(self) -> None:
        answers: list[object] = []

        def fake_rpc(*_arguments, **_kwargs):
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer

        session = MiningSession(
            Path("bitcoin-cli.exe"), None, None,
            NodeMonitor(lambda: None, 60.0), TemplatePrefetcher(lambda: None),
        )
        with patch("regtest_miner.rpc", side_effect=fake_rpc), redirect_stdout(io.StringIO()):
            answers[:] = [RpcError("refused"), RpcError("refused"), "00" * 32]
            self.assertFalse(session.tip_changed("00" * 32, regtest_miner.time.monotonic()))
            self.assertFalse(session.tip_changed("00" * 32, regtest_miner.time.monotonic()))
            self.assertIsNone(session.pause_reason())
            self.assertFalse(session.tip_changed("00" * 32, regtest_miner.time.monotonic()))
            self.assertEqual(session.tip_failures, 0)
            answers[:] = [OSError("no cli")] * 3 + ["ff" * 32]
            for _ in range(3):
                self.assertFalse(session.tip_changed("00" * 32, regtest_miner.time.monotonic()))
            self.assertIn("3 consecutive tip checks failed", session.pause_reason())
            session.reset()
            self.assertIsNone(session.pause_reason())
            self.assertTrue(session.tip_changed("00" * 32, regtest_miner.time.monotonic()))

    def test_unhealthy_monitor_pauses_between_chunks_with_nothing_in_flight(self) -> None:
        process = FakeCudaProcess()
        reasons = [None, None, None, "2 consecutive health checks failed: no peers"]
        with (
            patch("regtest_miner.subprocess.Popen", return_value=process),
            redirect_stdout(io.StringIO()),
        ):
            miner = CudaMiner(Path("cuda_miner.exe"))
            with self.assertRaisesRegex(MiningPaused, "no peers"):
                find_nonce(
                    miner, OverlappedTipCheckTests.HARD, 0x1D00FFFF, 5, 0, 99,
                    lambda: False, lambda: reasons.pop(0),
                )

        self.assertEqual(
            process.events,
            [("send", "1"), ("read", "1"), ("send", "2"), ("read", "2"), ("send", "3"), ("read", "3")],
        )
        self.assertIsNone(miner.pending_id)

    def test_candidate_takes_priority_over_a_pause_request(self) -> None:
        process = FakeCudaProcess(cpu_scan_responder)
        with (
            patch("regtest_miner.subprocess.Popen", return_value=process),
            redirect_stdout(io.StringIO()),
        ):
            miner = CudaMiner(Path("cuda_miner.exe"))
            pauses = iter([None, "node unreachable", "node unreachable"])
            nonce, _digest = find_nonce(
                miner, OverlappedTipCheckTests.EASY, 0x207FFFFF, 50, 0, 499,
                lambda: False, lambda: next(pauses),
            )
        self.assertTrue(0 <= nonce < 50)

    def test_preflight_enforces_min_peers(self) -> None:
        core = FakeCore()
        with (
            patch("regtest_miner.ensure_tor_ready", return_value=None),
            patch("regtest_miner.rpc", side_effect=core),
            redirect_stdout(io.StringIO()),
        ):
            preflight_mainnet(
                Path("bitcoin-cli.exe"), None, None, "127.0.0.1", 9150, None,
                1.0, 0.01, False, 8,
            )
            with self.assertRaisesRegex(RuntimeError, "--min-peers requires 9"):
                preflight_mainnet(
                    Path("bitcoin-cli.exe"), None, None, "127.0.0.1", 9150, None,
                    1.0, 0.01, False, 9,
                )


class BlockMonitorTests(unittest.TestCase):
    HASH = "ab" * 32
    TXID = "cd" * 32

    def watch(self, confirmations, wallet=("immature",)):
        confirmations = list(confirmations)
        wallet_answers = list(wallet)
        calls = []

        def fake_rpc(_cli, network, _datadir, _conf, method, *params, **_kwargs):
            calls.append((network, method))
            if method == "getblockheader":
                value = confirmations.pop(0)
                if isinstance(value, Exception):
                    raise value
                return {"hash": params[0], "height": 900_000, "confirmations": value}
            if method == "getblock":
                return {"tx": [self.TXID, "ee" * 32]}
            if method == "gettransaction":
                answer = wallet_answers.pop(0) if len(wallet_answers) > 1 else wallet_answers[0]
                if isinstance(answer, Exception):
                    raise answer
                return {"details": [{"category": answer}]}
            raise AssertionError(method)

        output = io.StringIO()
        with (
            patch("regtest_miner.rpc", side_effect=fake_rpc),
            patch("regtest_miner.time.sleep") as sleep,
            redirect_stdout(output),
        ):
            result = monitor_block(Path("bitcoin-cli.exe"), "mainnet", None, None, self.HASH)
        self.calls = calls
        self.sleeps = sleep.call_count
        return result, output.getvalue()

    def test_reports_each_change_until_the_coinbase_is_mature(self) -> None:
        result, output = self.watch(
            [1, 1, 2, -1, -1, 100, 101],
            wallet=("immature", "immature", "immature", "orphan", "orphan", "immature", "generate"),
        )

        self.assertEqual(result, 0)
        lines = output.splitlines()
        self.assertIn("MONITORING block", lines[0])
        self.assertEqual(len(lines), 6)          # repeats of the same state print nothing
        self.assertIn("1 confirmation(s)", lines[1])
        self.assertIn("matures in 100 more block(s)", lines[1])
        self.assertIn("Wallet: immature", lines[1])
        self.assertIn("2 confirmation(s)", lines[2])
        self.assertIn("NOT ON ACTIVE CHAIN", lines[3])
        self.assertIn("Wallet: orphan", lines[3])
        self.assertIn("matures in 1 more block(s)", lines[4])
        self.assertIn("COINBASE MATURE", lines[5])
        self.assertIn("Wallet: generate", lines[5])
        self.assertEqual(self.sleeps, 6)
        self.assertEqual({network for network, _ in self.calls}, {"mainnet"})
        self.assertEqual([m for _, m in self.calls].count("getblock"), 1)
        self.assertNotIn("submitblock", [m for _, m in self.calls])

    def test_survives_rpc_and_wallet_failures(self) -> None:
        result, output = self.watch(
            [RpcError("connection refused"), RpcError("connection refused"), 101],
            wallet=(RpcError("wallet not loaded"),),
        )

        self.assertEqual(result, 0)
        self.assertEqual(output.count("Block status unavailable"), 1)
        self.assertIn("not reported by the wallet", output)

    def run_main(self, arguments):
        self.errors = io.StringIO()
        with (
            patch("sys.argv", ["regtest_miner.py", *arguments]),
            patch("regtest_miner.monitor_block", return_value=0) as monitor,
            patch("regtest_miner.mine_one_block") as mine,
            patch("regtest_miner.rpc") as rpc_mock,
            patch("pathlib.Path.is_file", return_value=True),
            redirect_stdout(io.StringIO()),
            redirect_stderr(self.errors),
        ):
            result = main()
        mine.assert_not_called()
        rpc_mock.assert_not_called()
        return result, monitor

    def test_monitor_block_mode_never_mines(self) -> None:
        with patch("regtest_miner.keep_system_awake") as awake:
            result, monitor = self.run_main(["--mainnet", "--monitor-block", self.HASH.upper()])
        self.assertEqual([call.args[0] for call in awake.call_args_list], [True, False])
        self.assertEqual(result, 0)
        self.assertEqual(monitor.call_args.args[1], "mainnet")
        self.assertEqual(monitor.call_args.args[4], self.HASH)

        result, monitor = self.run_main(["--mainnet", "--monitor-block", "xyz"])
        self.assertEqual(result, 2)
        monitor.assert_not_called()

    def test_mainnet_sessions_hold_off_idle_sleep_and_release_it(self) -> None:
        # Dry-run mining, ended by Ctrl+C.
        core = FakeCore()

        def interrupt_second_template(cli, network, datadir, conf, method, *params, **kwargs):
            if method == "getblocktemplate" and "getblocktemplate" in core.methods:
                raise KeyboardInterrupt
            return core(cli, network, datadir, conf, method, *params, **kwargs)

        with (
            patch("regtest_miner.parse_args", return_value=miner_args(dry_run=True)),
            patch("regtest_miner.ensure_tor_ready", return_value=None),
            patch("regtest_miner.rpc", side_effect=interrupt_second_template),
            patch("regtest_miner.find_nonce", side_effect=cpu_find_nonce),
            patch("regtest_miner.subprocess.Popen"),
            patch("regtest_miner.save_unsubmitted_block", return_value=None),
            patch("regtest_miner.keep_system_awake") as awake,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main(), 130)
        self.assertEqual([call.args[0] for call in awake.call_args_list], [True, False])

        # A failed preflight still releases the request.
        with (
            patch("regtest_miner.parse_args", return_value=live_args()),
            patch("regtest_miner.ensure_tor_ready", side_effect=RuntimeError("Tor SOCKS5 is unavailable")),
            patch("regtest_miner.subprocess.Popen"),
            patch("regtest_miner.keep_system_awake") as awake,
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(main(), 1)
        self.assertEqual([call.args[0] for call in awake.call_args_list], [True, False])

        # Regtest never asks.
        regtest = FakeCore(chain="regtest")
        with (
            patch("regtest_miner.parse_args", return_value=miner_args(network="regtest", payout_address=None)),
            patch("regtest_miner.rpc", side_effect=regtest),
            patch("regtest_miner.find_nonce", side_effect=cpu_find_nonce),
            patch("regtest_miner.subprocess.Popen"),
            patch("regtest_miner.keep_system_awake") as awake,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main(), 0)
        self.assertEqual([call.args[0] for call in awake.call_args_list], [False])

    def test_keep_system_awake_requests_and_clears_the_windows_state(self) -> None:
        with patch("regtest_miner.sys.platform", "win32"), patch("regtest_miner.ctypes") as fake:
            regtest_miner.keep_system_awake(True)
            regtest_miner.keep_system_awake(False)
        calls = fake.windll.kernel32.SetThreadExecutionState.call_args_list
        self.assertEqual([call.args[0] for call in calls], [0x80000001, 0x80000000])
        with patch("regtest_miner.sys.platform", "linux"), patch("regtest_miner.ctypes") as fake:
            regtest_miner.keep_system_awake(True)
        fake.windll.kernel32.SetThreadExecutionState.assert_not_called()

    def test_accepted_mainnet_block_is_then_monitored(self) -> None:
        core = FakeCore()
        with (
            patch("regtest_miner.parse_args", return_value=live_args()),
            patch("regtest_miner.ensure_tor_ready", return_value=None),
            patch("regtest_miner.rpc", side_effect=core),
            patch("regtest_miner.find_nonce", side_effect=cpu_find_nonce),
            patch("regtest_miner.subprocess.Popen"),
            patch("regtest_miner.save_unsubmitted_block", return_value=None),
            patch("regtest_miner.monitor_block", return_value=0) as monitor,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main(), 0)

        monitor.assert_called_once()
        block_hash = double_sha256(bytes.fromhex(core.submitted[0][:160]))[::-1].hex()
        self.assertEqual(monitor.call_args.args[4], block_hash)
        self.assertEqual(core.methods.count("submitblock"), 1)


class SubmittedBlockStatusTests(unittest.TestCase):
    def test_requires_exact_active_chain_hash(self) -> None:
        with patch(
            "regtest_miner.rpc",
            side_effect=[101, "ab" * 32],
        ) as rpc_mock:
            status = submitted_block_status(
                "bitcoin-cli.exe",
                "regtest",
                None,
                None,
                101,
                "ab" * 32,
            )

        self.assertEqual(status, "ACTIVE CHAIN")
        self.assertEqual(rpc_mock.call_count, 2)

    def test_identifies_known_side_chain_block_as_stale(self) -> None:
        with patch(
            "regtest_miner.rpc",
            side_effect=[
                101,
                "cd" * 32,
                {"hash": "ab" * 32, "height": 101, "confirmations": -1},
            ],
        ):
            status = submitted_block_status(
                "bitcoin-cli.exe",
                "regtest",
                None,
                None,
                101,
                "ab" * 32,
            )

        self.assertEqual(status, "STALE/ORPHANED")

    def test_does_not_infer_active_status_from_height_alone(self) -> None:
        with patch(
            "regtest_miner.rpc",
            side_effect=[
                100,
                {"hash": "ab" * 32, "height": 101, "confirmations": 0},
            ],
        ):
            status = submitted_block_status(
                "bitcoin-cli.exe",
                "regtest",
                None,
                None,
                101,
                "ab" * 32,
            )

        self.assertEqual(status, "SUBMITTED")


if __name__ == "__main__":
    unittest.main()
