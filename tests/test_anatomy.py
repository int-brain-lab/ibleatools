import unittest
import ephysatlas.anatomy


class TestAnatomy(unittest.TestCase):
    def test_regions_transitions_and_instantiation(self):
        ba = ephysatlas.anatomy.ClassifierAtlas(res_um=50)
        transition_matrix, voxel_occurences, rids = (
            ephysatlas.anatomy.regions_transition_matrix(ba=ba)
        )
        self.assertEqual(voxel_occurences.size, 13)
        self.assertEqual(transition_matrix.size, 13 * 13)
        self.assertEqual(rids.size, 13)
