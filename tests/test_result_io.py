import hashlib
import os
import tempfile
import unittest

from rbdp.io import RunDirectory
from rbdp.io import config_hash
from rbdp.io import file_sha256
from rbdp.io import read_json
from rbdp.io import source_manifest
from rbdp.io import write_csv
from rbdp.io import write_json
from rbdp.io import write_npz


class ResultIOTest(unittest.TestCase):

    def test_config_hash_ignores_dictionary_insertion_order(self):
        self.assertEqual(
            config_hash({"a": 1, "b": [2, 3]}),
            config_hash({"b": [2, 3], "a": 1}),
        )

    def test_atomic_json_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "nested", "value.json")
            write_json(path, {"中文": "可追溯", "value": 3})
            self.assertEqual(read_json(path)["中文"], "可追溯")

    def test_file_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "payload.bin")
            with open(path, "wb") as handle:
                handle.write(b"rbdp")
            self.assertEqual(file_sha256(path), hashlib.sha256(b"rbdp").hexdigest())

    def test_run_directory_lifecycle_and_resume_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            run = RunDirectory(os.path.join(directory, "run"), {"seed": 1})
            self.assertFalse(run.reusable())
            run.start()
            self.assertEqual(read_json(os.path.join(run.path, "status.json"))["state"],
                             "running")
            run.complete({"acc": 0.5})
            self.assertTrue(run.reusable())
            with self.assertRaises(RuntimeError):
                run.start()
            changed = RunDirectory(run.path, {"seed": 2})
            self.assertFalse(changed.reusable())
            with self.assertRaises(ValueError):
                changed.start()

    def test_source_manifest_tracks_selected_files(self):
        with tempfile.TemporaryDirectory() as directory:
            first = os.path.join(directory, "a.py")
            second = os.path.join(directory, "b.json")
            with open(first, "w", encoding="utf-8") as handle:
                handle.write("value = 1\n")
            with open(second, "w", encoding="utf-8") as handle:
                handle.write("{}\n")
            manifest = source_manifest(directory, ["."])
            self.assertEqual(sorted(manifest["files"]), ["a.py", "b.json"])
            self.assertEqual(len(manifest["source_hash"]), 64)

    def test_csv_and_npz_are_written_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = os.path.join(directory, "rows.csv")
            npz_path = os.path.join(directory, "arrays.npz")
            write_csv(csv_path, [{"epoch": 1, "loss": 0.5}])
            write_npz(npz_path, values=[1, 2, 3])
            self.assertTrue(os.path.isfile(csv_path))
            self.assertTrue(os.path.isfile(npz_path))


if __name__ == "__main__":
    unittest.main()
