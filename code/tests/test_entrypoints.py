"""Fast regression checks; no fitting and no changes to distributed research files."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from check_manifest import check_release
from vista import FrozenVISTA


class ReleaseChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = FrozenVISTA.load(ROOT / "models/fold_0.npz")

    def test_manifest_detects_missing_and_corrupted_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            content = b"original\n"
            manifest = {"files": [{"path": "result.csv", "bytes": len(content),
                                   "sha256": hashlib.sha256(content).hexdigest()}]}
            (root / "MANIFEST.json").write_text(json.dumps(manifest))
            with self.assertRaises(FileNotFoundError):
                check_release(root)
            (root / "result.csv").write_bytes(content)
            self.assertEqual(check_release(root), 1)
            (root / "result.csv").write_bytes(b"changed!\n")
            with self.assertRaises(ValueError):
                check_release(root)

    def test_manifest_rejects_parent_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "MANIFEST.json").write_text(json.dumps({"files": [{"path": "../outside"}]}))
            with self.assertRaises(ValueError):
                check_release(root)

    def test_optimized_verification_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for script, extra in [
                (ROOT / "verify_reproduction.py", ["--data-dir", folder, "--out", folder]),
                (ROOT / "supplement/verify_historical_records.py", []),
                (ROOT / "supplement/verify_selectors_and_directional.py", []),
            ]:
                result = subprocess.run([sys.executable, "-O", script, *extra],
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("without -O", result.stderr)

    def test_verifier_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "keep.txt").write_text("keep")
            result = subprocess.run([sys.executable, ROOT / "verify_reproduction.py",
                                     "--data-dir", folder, "--out", folder],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("new or empty output directory", result.stderr)
            self.assertEqual((root / "keep.txt").read_text(), "keep")

    def test_prediction_cli_from_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            x = np.random.default_rng(20261020).uniform(-1, 60, (17, 59))
            np.save(root / "input.npy", x)
            result = subprocess.run([sys.executable, ROOT / "predict.py",
                                     "--model", ROOT / "models/fold_0.npz",
                                     "--input", root / "input.npy",
                                     "--output", root / "result.npz"],
                                    cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with np.load(root / "result.npz") as actual:
                expected = self.model.predict(x)
                np.testing.assert_allclose(actual["spectra"], expected.spectra, rtol=0, atol=1e-12)
                np.testing.assert_array_equal(actual["annotation_index"], expected.annotation_index)

    def test_model_batching_and_class_preservation(self):
        x = np.random.default_rng(128).uniform(-2, 80, (31, 59))
        whole = self.model.predict(x)
        parts = [self.model.predict(part) for part in np.array_split(x, 3)]
        np.testing.assert_allclose(whole.spectra, np.concatenate([p.spectra for p in parts]),
                                   rtol=0, atol=1e-12)
        np.testing.assert_array_equal(self.model.probabilities(whole.spectra).argmax(axis=1),
                                      whole.annotation_index)

    def test_invalid_inputs_fail(self):
        for x in (np.empty((0, 59)), np.ones((2, 58)), np.ones(59),
                  np.full((2, 59), np.nan), np.full((2, 59), np.inf),
                  np.full((2, 59), np.finfo(float).max)):
            with self.subTest(shape=x.shape), self.assertRaises(ValueError):
                self.model.predict(x)

    def test_interpolation_grid_and_boundaries(self):
        waves = self.model.wavelength_nm
        x = np.stack((waves / 10, -waves / 20))
        np.testing.assert_array_equal(self.model.interpolate_prisma(x, waves), x)
        wider = np.array([waves[0] - 1, waves[-1] + 1])
        np.testing.assert_allclose(self.model.interpolate_prisma(wider[None, :], wider),
                                   waves[None, :], rtol=0, atol=1e-12)
        for grid in (waves[::-1], waves[:-1], np.r_[waves[:-1], np.nan]):
            with self.assertRaises(ValueError):
                self.model.interpolate_prisma(np.ones((2, len(grid))), grid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
