import unittest
import ephysatlas.fixtures


class TestPidsFixtures(unittest.TestCase):
    def test_pid_fixtures(self):
        self.assertEqual(len(ephysatlas.fixtures.repro_ephys_pids), 104)
        self.assertEqual(len(ephysatlas.fixtures.benchmark_pids), 13)
        self.assertEqual(len(ephysatlas.fixtures.misaligned_pids), 294)
        self.assertEqual(len(ephysatlas.fixtures.nemo_test_pids), 135)

    def test_pid_qc_filter(self):
        qc = ephysatlas.fixtures.pid_qc_filter
        self.assertEqual((qc.VINTAGE, qc.LEVEL), ("2026_W42", "L3"))
        self.assertEqual(len(qc.excluded_pids()), 79)
        self.assertEqual(len(qc.excluded_pids(("alignment", "ap", "waveform"))), 59)
        self.assertEqual(len(qc.excluded_pids(include_review=True)), 106)
        labels = {label for entry in qc.pid_qc_filter.values() for label in entry[:4]}
        self.assertLessEqual(labels, {"good", "review", "bad"})
