import argparse
import io
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
from regtest_miner import (
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
    script_number,
    serialize_block,
    submitted_block_status,
    template_transactions,
    transaction_hashes,
    socks5_ready,
    validate_template,
    validate_template_target,
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
        args = argparse.Namespace(
            network="mainnet",
            dry_run=True,
            blocks=0,
            chunk_size=1,
            bitcoin_cli=Path(__file__),
            cuda_miner=Path(__file__),
            bitcoin_conf=None,
            datadir=None,
            payout_address="bc1qexample",
            tor_host="127.0.0.1",
            tor_port=9150,
            tor_executable=None,
            tor_startup_timeout=1.0,
            tor_poll_interval=0.01,
            require_onion_peers=False,
        )
        template = valid_template()
        template["noncerange"] = "0000000000000000"
        rpc_methods = []

        def fake_rpc(_cli, _network, _datadir, _conf, method, *_params):
            rpc_methods.append(method)
            if method == "getblockchaininfo":
                return {
                    "chain": "main",
                    "initialblockdownload": False,
                    "blocks": 100,
                    "headers": 100,
                }
            if method == "getnetworkinfo":
                return {"networkactive": True, "connections": 1}
            if method == "validateaddress":
                return {"isvalid": True, "scriptPubKey": "51"}
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

        with (
            patch("regtest_miner.parse_args", return_value=args),
            patch("regtest_miner.ensure_tor_ready", return_value=None),
            patch("regtest_miner.rpc", side_effect=fake_rpc),
            patch("regtest_miner.mine_one_block", side_effect=mine_once_then_interrupt),
            patch("regtest_miner.mine_chunk", return_value=None) as mine_chunk,
            redirect_stdout(io.StringIO()),
        ):
            result = main()

        self.assertEqual(result, 130)
        self.assertEqual(mine_chunk.call_count, 1)
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

    def test_live_mainnet_submission_remains_gated(self) -> None:
        args = argparse.Namespace(network="mainnet", dry_run=False)
        error_output = io.StringIO()
        with (
            patch("regtest_miner.parse_args", return_value=args),
            redirect_stderr(error_output),
        ):
            result = main()

        self.assertEqual(result, 2)
        self.assertIn("gated", error_output.getvalue())

    def test_mainnet_dry_run_rejects_unsynchronized_core(self) -> None:
        args = argparse.Namespace(
            network="mainnet",
            dry_run=True,
            blocks=0,
            chunk_size=10,
            bitcoin_cli=Path(__file__),
            cuda_miner=Path(__file__),
            bitcoin_conf=None,
            datadir=None,
            payout_address="bc1qexample",
            tor_host="127.0.0.1",
            tor_port=9150,
            tor_executable=None,
            tor_startup_timeout=1.0,
            tor_poll_interval=0.01,
            require_onion_peers=False,
        )
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
            patch("regtest_miner.mine_chunk", side_effect=record_scan),
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
