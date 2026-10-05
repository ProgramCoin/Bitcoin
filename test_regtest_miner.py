import unittest

from regtest_miner import validate_template_target


GENESIS_BITS = "1d00ffff"
GENESIS_TARGET = (
    "00000000ffff0000000000000000000000000000000000000000000000000000"
)


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


if __name__ == "__main__":
    unittest.main()
