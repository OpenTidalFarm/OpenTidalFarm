import unittest
import copy
import opentidalfarm as otf


class TestTurbineParameterisation(unittest.TestCase):
    """Tests TurbineParameterisation class."""

    def test_initialization(self):
        # Default initialisation
        parameterisation = otf.TurbineParameterisation()
        self.assertTrue(parameterisation._parameterisation["default"])
        self.assertFalse(parameterisation._parameterisation["thrust"])
        self.assertFalse(parameterisation._parameterisation["implicit thrust"])
        self.assertFalse(parameterisation._parameterisation["smeared"])

        parameterisation = otf.TurbineParameterisation(thrust=True)
        self.assertFalse(parameterisation._parameterisation["default"])
        self.assertTrue(parameterisation._parameterisation["thrust"])
        self.assertFalse(parameterisation._parameterisation["implicit thrust"])
        self.assertFalse(parameterisation._parameterisation["smeared"])

        parameterisation = otf.TurbineParameterisation(implicit_thrust=True)
        self.assertFalse(parameterisation._parameterisation["default"])
        self.assertFalse(parameterisation._parameterisation["thrust"])
        self.assertTrue(parameterisation._parameterisation["implicit thrust"])
        self.assertFalse(parameterisation._parameterisation["smeared"])

        parameterisation = otf.TurbineParameterisation(smeared=True)
        self.assertFalse(parameterisation._parameterisation["default"])
        self.assertFalse(parameterisation._parameterisation["thrust"])
        self.assertFalse(parameterisation._parameterisation["implicit thrust"])
        self.assertTrue(parameterisation._parameterisation["smeared"])

        self.assertRaises(ValueError, otf.TurbineParameterisation, default=True,
                thrust=True)

        self.assertRaises(ValueError, otf.TurbineParameterisation, default=True,
                thrust=True, implicit_thrust=True)

        self.assertRaises(ValueError, otf.TurbineParameterisation, default=True,
                thrust=True, implicit_thrust=True, smeared=True)

        self.assertRaises(ValueError, otf.TurbineParameterisation,
                default=False, thrust=False, implicit_thrust=True, smeared=True)


    def test___str__(self):
        parameterisation = otf.TurbineParameterisation()
        self.assertIsInstance(parameterisation.__str__(), str)


    def test_parameterisation(self):
        parameterisation = otf.TurbineParameterisation(default=True)
        self.assertTrue(parameterisation.default)

        parameterisation = otf.TurbineParameterisation(thrust=True)
        self.assertTrue(parameterisation.thrust)

        parameterisation = otf.TurbineParameterisation(implicit_thrust=True)
        self.assertTrue(parameterisation.implicit_thrust)

        parameterisation = otf.TurbineParameterisation(smeared=True)
        self.assertTrue(parameterisation.smeared)



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

    def test__get_controls(self):
        turbine = otf.Turbine(self.diameter, self.minimum_distance,
                self.friction)

        turbine._set_controls(None)
        self.assertRaises(ValueError, turbine._get_controls)



if __name__=="__main__":
    otf.set_log_level(otf.ERROR)
    # Parameterisation tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTurbineParameterisation)
    unittest.TextTestRunner(verbosity=2).run(suite)
    # Turbine tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTurbine)
    unittest.TextTestRunner(verbosity=2).run(suite)
