import unittest
import numpy as np
import opentidalfarm as otf

class TestControls(unittest.TestCase):
    """Tests Controls class and functionality."""

    def test___init__(self):
        self.assertRaises(ValueError, otf.Controls, 1.0, True)
        self.assertRaises(ValueError, otf.Controls, 1.0, False)
        self.assertRaises(ValueError, otf.Controls, True, 1.0)
        self.assertRaises(ValueError, otf.Controls, False, 1.0)
        self.assertRaises(ValueError, otf.Controls, 1.0, 1.0)

        def _test_valid(position, friction):
            # Check for valid initialization
            try:
                otf.Controls(position, friction)
            except ValueError:
                self.fail("Control failed to initialize with valid arguments")

        # Test no exceptions are raised for valid input.
        _test_valid(True, True)
        _test_valid(False, False)
        _test_valid(True, False)
        _test_valid(False, True)


    def test_property(self):
        control = otf.Controls(position=True, friction=True)
        self.assertTrue(control.position)
        self.assertTrue(control.friction)

        control = otf.Controls(position=True, friction=False)
        self.assertTrue(control.position)
        self.assertFalse(control.friction)

        control = otf.Controls(position=False, friction=True)
        self.assertFalse(control.position)
        self.assertTrue(control.friction)

        control = otf.Controls(position=False, friction=False)
        self.assertFalse(control.position)
        self.assertFalse(control.friction)


if __name__=="__main__":
    otf.set_log_level(otf.ERROR)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestControls)
    unittest.TextTestRunner(verbosity=2).run(suite)
