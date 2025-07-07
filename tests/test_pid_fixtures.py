import unittest
import ephysatlas.fixtures


class TestPidsFixtures(unittest.TestCase):
    def test_pid_fixtures(self):
        self.assertEqual(len(ephysatlas.fixtures.repro_ephys_pids), 104)
        self.assertEqual(len(ephysatlas.fixtures.benchmark_pids), 13)
        self.assertEqual(len(ephysatlas.fixtures.misaligned_pids), 261)
        self.assertEqual(len(ephysatlas.fixtures.nemo_test_pids), 135)
