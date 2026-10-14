"""Regression tests for bugs encountered in the original hybrid-model scan."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location("scan", Path(__file__).parents[1] / "scripts/scan_weights.py")
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)


class ScannerTests(unittest.TestCase):
    def test_auxiliary_head_is_excluded(self):
        names = {"model.language_model.layers.3.self_attn.k_proj.weight": "a",
                 "mtp.layers.0.self_attn.k_proj.weight": "b"}
        selected = scan.decoder_names(names, "model.language_model.layers.", "self_attn.k_proj.weight", 64)
        self.assertEqual(list(selected), [3])

    def test_sparse_ids_keep_large_offsets_and_skip_empty_pairs(self):
        tensors = {3: np.array([1., 0.], dtype=np.float32),
                   19: np.array([0., 1.], dtype=np.float32),
                   35: np.array([1., 0.], dtype=np.float32)}
        _, offsets, result = scan.similarities(tensors, 64, 16, (0, 63), 2)
        lookup = {row["offset"]: row for row in offsets}
        self.assertEqual(set(lookup), {16, 32})
        self.assertEqual(lookup[32]["n_pairs"], 1)
        self.assertAlmostEqual(lookup[32]["mean_cosine"], 1.)
        self.assertFalse(lookup[32]["rank_eligible"])
        self.assertEqual(result["top_offsets"][0]["offset"], 16)

    def test_inside_mean_differs_from_all_pair_mean(self):
        tensors = {0: np.array([1., 0.], dtype=np.float32),
                   1: np.array([0., 1.], dtype=np.float32),
                   2: np.array([0., 1.], dtype=np.float32)}
        _, _, result = scan.similarities(tensors, 3, 1, (1, 2), 1)
        period = result["periods"]["1"]
        self.assertAlmostEqual(period["all_valid_mean"], .5)
        self.assertAlmostEqual(period["inside_mean"], 1.)
        self.assertEqual(period["n_inside"], 1)

    def test_bfloat16_decoding(self):
        encoded = np.array([0x3F80, 0xC000], dtype="<u2").tobytes()
        np.testing.assert_array_equal(scan.decode_tensor(encoded, "BF16", [2]), [1., -2.])

    def test_server_ignoring_range_is_rejected_without_reading_body(self):
        class Response:
            status_code = 200
            headers = {}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def raise_for_status(self): pass
            def iter_content(self, *args):
                raise AssertionError("Must reject full shard before reading body")
        with patch.object(scan.requests, "get", return_value=Response()), patch.object(scan.time, "sleep"):
            with self.assertRaises(RuntimeError):
                scan.read_range("https://example.invalid/weights", 8, 15)


if __name__ == "__main__":
    unittest.main()
