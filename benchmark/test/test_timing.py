"""CPU-only tests: python -m unittest discover -s benchmark/test -p 'test_*.py'."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "src_py"


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TimingTests(unittest.TestCase):
    def test_completion_order(self):
        for gray_scott in (False, True):
            for warmup in (0, 2):
                with self.subTest(gray_scott=gray_scott, warmup=warmup):
                    events, ticks = [], iter((6000, 12000))

                    def clock():
                        events.append("clock")
                        return next(ticks)

                    def fence(*, block):
                        self.assertTrue(block)
                        events.append("sync")

                    mocks = {
                        "cupynumeric": SimpleNamespace(float32=float, float64=float),
                        "legate.core": SimpleNamespace(get_legate_runtime=lambda:
                            SimpleNamespace(issue_execution_fence=fence)),
                        "legate.timing": SimpleNamespace(time=clock),
                    }
                    with patch.dict("sys.modules", mocks):
                        core = load(SOURCE / "core.py")
                        with patch.dict("sys.modules", {"core": core}):
                            gs = load(SOURCE / "benchmarks" / "grayscott.py")

                    class Probe:
                        def initialize(self):
                            events.append("initialize")
                            return None

                        def run(self, state):
                            events.append("run")

                    # Inherit the real GrayScott policy without creating GPU arrays.
                    cls = type("GrayScottProbe", (Probe, gs.GrayScott), {}) if gray_scott else Probe
                    bench = cls.__new__(cls)
                    result = core.trial(bench, warmup, 3, 6000)
                    step = ["run"] if gray_scott else ["run", "sync"]
                    self.assertEqual(events, ["initialize"] + step * warmup
                                     + ["clock"] + step * 3 + ["clock"])
                    self.assertEqual(result, (2.0, 0.003))


if __name__ == "__main__":
    unittest.main()
