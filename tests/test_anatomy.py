import unittest
import ephysatlas.anatomy



class Test(unittest.TestCase):

    def test_split_void_null(self):
        brain_atlas = ephysatlas.anatomy.ClassifierAtlas()
