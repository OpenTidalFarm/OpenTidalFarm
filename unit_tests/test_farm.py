import unittest
import numpy as np
import opentidalfarm as otf

class TestRectangularFarm(unittest.TestCase):
    """Tests Farm class and functionality."""

    def setUp(self):
        self.turbine = otf.Turbine(diameter=1.0, minimum_distance=2.0,
                                   maximum_friction=21.0)
        self.farm = otf.RectangularFarm("./mesh.xml", 25, 75, 25, 75)


    def test_too_many_x(self):
        x_num = 1000
        y_num = 1
        self.assertRaises(ValueError, self.farm.add_regular_turbine_layout,
                          x_num, y_num, self.farm.site_x_start,
                          self.farm.site_x_end, self.farm.site_y_start,
                          self.farm.site_y_end, turbine=self.turbine)


    def test_too_many_y(self):
        x_num = 1
        y_num = 1000
        self.assertRaises(ValueError, self.farm.add_regular_turbine_layout,
                          x_num, y_num, self.farm.site_x_start,
                          self.farm.site_x_end, self.farm.site_y_start,
                          self.farm.site_y_end, turbine=self.turbine)


    def test_too_many_x_and_y(self):
        x_num = 1000
        y_num = 1000
        self.assertRaises(ValueError, self.farm.add_regular_turbine_layout,
                          x_num, y_num, self.farm.site_x_start,
                          self.farm.site_x_end, self.farm.site_y_start,
                          self.farm.site_y_end, turbine=self.turbine)


    def test_correct_number(self):
        x_num = 1
        y_num = 1
        self.farm.add_regular_turbine_layout(x_num, y_num, turbine=self.turbine)
        self.assertEqual(self.farm.number_of_turbines, x_num*y_num)


    def test_serialize(self):
        # Add turbines to the farm.
        self.farm.add_regular_turbine_layout(10, 10, turbine=self.turbine)
        # Test just turbine positions.
        controls = otf.Controls(position=True)
        m = self.farm.serialize(controls)
        self.assertEqual(len(m), self.farm.number_of_turbines*2)
        self.assertEqual(m[0], self.farm._turbine_positions[0])
        self.assertEqual(m[1], self.farm._turbine_positions[1])
        self.assertEqual(m[-2], self.farm._turbine_positions[-2])
        self.assertEqual(m[-1], self.farm._turbine_positions[-1])

        # Test for just turbine friction.
        controls = otf.Controls(friction=True)
        m = self.farm.serialize(controls)
        self.assertEqual(len(m), self.farm.number_of_turbines)
        self.assertEqual(m[0], self.farm._friction[0])
        self.assertEqual(m[1], self.farm._friction[1])
        self.assertEqual(m[-1], self.farm._friction[-1])

        # Test for turbine_friction and turbine_pos.
        controls = otf.Controls(position=True, friction=True)
        m = self.farm.serialize(controls)
        n_turbines = self.farm.number_of_turbines
        self.assertEqual(len(m), n_turbines*3)
        # Check friction values
        self.assertEqual(m[0], self.farm._friction[0])
        self.assertEqual(m[1], self.farm._friction[1])
        self.assertEqual(m[n_turbines-1], self.farm._friction[-1])

        # Check position values
        self.assertEqual(m[n_turbines], self.farm._turbine_positions[0])
        self.assertEqual(m[n_turbines+1], self.farm._turbine_positions[1])
        self.assertEqual(m[n_turbines+2], self.farm._turbine_positions[2])
        self.assertEqual(m[n_turbines+3], self.farm._turbine_positions[3])
        self.assertEqual(m[-2], self.farm._turbine_positions[-2])
        self.assertEqual(m[-1], self.farm._turbine_positions[-1])

        # Test no control
        controls = otf.Controls(position=False, friction=False)
        m = self.farm.serialize(controls)
        self.assertEqual(len(m), 0)


    def test_deserialize(self):
        controls = otf.Controls(position=True)
        # Add 2 turbines to the farm.
        self.farm.add_regular_turbine_layout(2, 1, turbine=self.turbine)
        # Create new turbine positions and update them.
        m = np.array([0, 0, 100, 100])
        self.farm.deserialize(m, controls)
        # Test the turbine positions are correct
        self.assertEqual(self.farm._turbine_positions[0], 0)
        self.assertEqual(self.farm._turbine_positions[1], 0)
        self.assertEqual(self.farm._turbine_positions[2], 100)
        self.assertEqual(self.farm._turbine_positions[3], 100)
        # And that we have enough turbines.
        self.assertEqual(self.farm.number_of_turbines, len(m)/2)

        # Test just friction
        m = np.array([10., 15.])
        controls = otf.Controls(friction=True)
        self.farm.deserialize(m, controls)
        # Test the turbine frictions are correct
        self.assertEqual(self.farm._friction[0], 10.)
        self.assertEqual(self.farm._friction[1], 15.)
        # And that we have enough turbines.
        self.assertEqual(self.farm.number_of_turbines, len(m))

        # Test for friction and position
        m = np.array([10., 15., 0., 0., 100., 100.])
        controls = otf.Controls(position=True, friction=True)
        self.farm.deserialize(m, controls)
        # Test the turbine positions are correct
        self.assertEqual(self.farm._turbine_positions[0], 0)
        self.assertEqual(self.farm._turbine_positions[1], 0)
        self.assertEqual(self.farm._turbine_positions[2], 100)
        self.assertEqual(self.farm._turbine_positions[3], 100)
        # Test the turbine frictions are correct
        self.assertEqual(self.farm._friction[0], 10.)
        self.assertEqual(self.farm._friction[1], 15.)
        # And that we have enough turbines.
        self.assertEqual(self.farm.number_of_turbines, len(m)/3)


    def test_boundary_constraints(self):
        # Add a 5x5 array of turbines
        self.farm.add_regular_turbine_layout(2, 2, turbine=self.turbine)
        lower, upper = self.farm.site_boundary_constraints()

        # Check we have enough bounds
        self.assertEqual(len(lower)/2, self.farm.number_of_turbines)
        self.assertEqual(len(upper)/2, self.farm.number_of_turbines)

        # Check the bounds are instances of dolfin_adjoint.Constant
        for l in lower:
            self.assertIsInstance(l, otf.Constant)

        for u in upper:
            self.assertIsInstance(u, otf.Constant)

        # Check entires are the same
        for i in range(1,len(lower)/2):
            self.assertEqual(lower[0], lower[2*i])
            self.assertEqual(lower[1], lower[2*i+1])
            self.assertEqual(upper[0], upper[2*i])
            self.assertEqual(upper[1], upper[2*i+1])


if __name__=="__main__":
    otf.set_log_level(otf.ERROR)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRectangularFarm)
    unittest.TextTestRunner(verbosity=2).run(suite)
