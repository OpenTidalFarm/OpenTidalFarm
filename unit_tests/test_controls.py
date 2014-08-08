import itertools
import unittest
import numpy as np
import opentidalfarm as otf

class TestControls(unittest.TestCase):
    """Tests Controls class and functionality."""

    def generate_combinations(self):
        options = 3
        combinations = []
        tf_count = itertools.combinations_with_replacement([True, False],
                                                           options)

        for tf in tf_count:
            permutations = list(set(itertools.permutations(tf, 3)))
            combinations +=  permutations

        return combinations


    def test___init__(self):
        self.assertRaises(ValueError, otf.Controls, 1.0, True, True)
        self.assertRaises(ValueError, otf.Controls, True, 1.0, True)
        self.assertRaises(ValueError, otf.Controls, True, True, 1.0)
        self.assertRaises(ValueError, otf.Controls, 1.0, 1.0, True)
        self.assertRaises(ValueError, otf.Controls, True, 1.0, 1.0)
        self.assertRaises(ValueError, otf.Controls, 1.0, True, 1.0)
        self.assertRaises(ValueError, otf.Controls, 1.0, 1.0, 1.0)

        def _test_valid(position, friction, dynamic_friction):
            # Check for valid initialization
            try:
                otf.Controls(position, friction, dynamic_friction)
            except ValueError:
                self.fail("Control failed to initialize with valid arguments")

        # Test no exceptions are raised for valid input.
        combinations = self.generate_combinations()
        for c in combinations:
            _test_valid(c[0], c[1], c[2])


    def test_property(self):
        combinations = self.generate_combinations()
        for c in combinations:
            control = otf.Controls(position=c[0],
                                   friction=c[1],
                                   dynamic_friction=c[2])
            self.assertEqual(control.position, c[0])
            self.assertEqual(control.friction, c[1])
            self.assertEqual(control.dynamic_friction, c[2])


if __name__=="__main__":
    otf.set_log_level(otf.ERROR)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestControls)
    unittest.TextTestRunner(verbosity=2).run(suite)
