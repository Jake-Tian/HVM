import tempfile
import unittest
from pathlib import Path

from scripts.data.reorganize_outputs import PROJECT_ROOT
from utils.general import find_pkl_files


class UtilsStructureTest(unittest.TestCase):
    def test_graph_discovery_ignores_preabstraction_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            graph_dir = Path(directory)
            (graph_dir / "bedroom_01.pkl").touch()
            (graph_dir / "bedroom_01_preabstraction.pkl").touch()
            (graph_dir / "gym_02.pkl").touch()

            self.assertEqual(
                find_pkl_files(graph_dir),
                ["bedroom_01", "gym_02"],
            )

    def test_relocated_data_script_keeps_project_root(self):
        self.assertEqual(PROJECT_ROOT, Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    unittest.main()
