import unittest
import opentidalfarm as otf

class TestTurbine(unittest.TestCase):
    """Tests Turbine class and functionality."""

    def setUp(self):
        self.diameter = 20.0
        self.minimum_distance = 40.0
        self.friction = 10.0


    def test_getters(self):
        turbine = otf.Turbine(self.diameter, self.minimum_distance,
                self.friction)

        # Test provded args
        self.assertEqual(turbine.diameter, self.diameter)
        self.assertEqual(turbine.minimum_distance, self.minimum_distance)
        self.assertEqual(turbine.friction, self.friction)
        # Test other properties
        self.assertEqual(turbine.radius, self.diameter*0.5)


    def test_setters(self):
        turbine = otf.Turbine(self.diameter, self.minimum_distance,
                self.friction+5)

        # Reset the friction.
        turbine.friction = self.friction
        self.assertEqual(turbine.friction, self.friction)

        # Give the turbine a location.
        position = (0.,0.)
        turbine.coordinates = position
        self.assertTupleEqual(turbine.coordinates, position)


    def test_copy_constructor(self):
        turbine1 = otf.Turbine(self.diameter, self.minimum_distance,
                self.friction)

        turbine2 = turbine1._copy_constructor()
        # Should be different objects but with the same parametrs.
        self.assertIsNot(turbine1, turbine2)
        self.assertEqual(turbine1.friction, turbine2.friction)
        self.assertEqual(turbine1.diameter, turbine2.diameter)
        self.assertEqual(turbine1.minimum_distance, turbine2.minimum_distance)

        # Check we can add a turbine with a position
        position = (10.,10.)
        turbine3 = turbine2._copy_constructor(position)
        self.assertIsNot(turbine2, turbine3)
        self.assertEqual(turbine2.friction, turbine3.friction)
        self.assertEqual(turbine2.diameter, turbine3.diameter)
        self.assertEqual(turbine2.minimum_distance, turbine3.minimum_distance)
        self.assertTupleEqual(turbine3.coordinates, position)


if __name__=="__main__":
    otf.set_log_level(otf.ERROR)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTurbine)
    unittest.TextTestRunner(verbosity=2).run(suite)
