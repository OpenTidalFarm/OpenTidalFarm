import unittest
import numpy as np
import opentidalfarm as otf

class TestRectangularFarm(unittest.TestCase):
    """Tests Farm class and functionality."""

    def setUp(self):
        self.turbine = otf.Turbine(diameter=20.0, minimum_distance=40.0,
                                   friction=21.0)
        self.farm = otf.RectangularFarm(0, 1000, 0, 1000)


    def test_too_many_x(self):
        x_num = 1000
        y_num = 1
        self.assertRaises(ValueError, self.farm.add_regular_turbine_layout,
                          self.turbine, x_num, y_num, self.farm.site_x_start,
                          self.farm.site_x_end, self.farm.site_y_start,
                          self.farm.site_y_end)


    def test_too_many_y(self):
        x_num = 1
        y_num = 1000
        self.assertRaises(ValueError, self.farm.add_regular_turbine_layout,
                          self.turbine, x_num, y_num, self.farm.site_x_start,
                          self.farm.site_x_end, self.farm.site_y_start,
                          self.farm.site_y_end)


    def test_too_many_x_and_y(self):
        x_num = 1000
        y_num = 1000
        self.assertRaises(ValueError, self.farm.add_regular_turbine_layout,
                          self.turbine, x_num, y_num, self.farm.site_x_start,
                          self.farm.site_x_end, self.farm.site_y_start,
                          self.farm.site_y_end)


    def test_correct_number(self):
        x_num = 5
        y_num = 5
        self.farm.add_regular_turbine_layout(self.turbine, x_num, y_num)
        self.assertEqual(len(self.farm.turbines), x_num*y_num)


    def test_serialize(self):
        # Add turbines to the farm.
        self.farm.add_regular_turbine_layout(self.turbine, 10, 10)
        # Test just turbine positions.
        m = self.farm.serialize(turbine_pos=True, turbine_friction=False)
        self.assertEqual(len(m), len(self.farm.turbines)*2)
        self.assertEqual(m[0], self.farm.turbines[0].x)
        self.assertEqual(m[1], self.farm.turbines[0].y)
        self.assertEqual(m[-2], self.farm.turbines[-1].x)
        self.assertEqual(m[-1], self.farm.turbines[-1].y)

        # Test for just turbine friction.
        m = self.farm.serialize(turbine_pos=False, turbine_friction=True)
        self.assertEqual(len(m), len(self.farm.turbines))
        self.assertEqual(m[0], self.farm.turbines[0].friction)
        self.assertEqual(m[1], self.farm.turbines[1].friction)
        self.assertEqual(m[-1], self.farm.turbines[-1].friction)

        # Test for turbine_friction and turbine_pos.
        m = self.farm.serialize(turbine_pos=True, turbine_friction=True)
        n_turbines = len(self.farm.turbines)
        self.assertEqual(len(m), n_turbines*3)
        # Check friction values
        self.assertEqual(m[0], self.farm.turbines[0].friction)
        self.assertEqual(m[1], self.farm.turbines[1].friction)
        self.assertEqual(m[n_turbines-1], self.farm.turbines[-1].friction)

        # Check position values
        self.assertEqual(m[n_turbines], self.farm.turbines[0].x)
        self.assertEqual(m[n_turbines+1], self.farm.turbines[0].y)
        self.assertEqual(m[n_turbines+2], self.farm.turbines[1].x)
        self.assertEqual(m[n_turbines+3], self.farm.turbines[1].y)
        self.assertEqual(m[-2], self.farm.turbines[-1].x)
        self.assertEqual(m[-1], self.farm.turbines[-1].y)

        # Test no control
        m = self.farm.serialize(turbine_pos=False, turbine_friction=False)
        self.assertEqual(len(m), 0)


    def test_deserialize(self):
        self.farm.add_regular_turbine_layout(self.turbine, 2, 1)
        # Test just turbine_pos
        m = np.array([0, 0, 100, 100])
        self.farm.deserialize(m)
        # Test the turbine positions are correct
        self.assertEqual(self.farm.turbines[0].x, 0)
        self.assertEqual(self.farm.turbines[0].y, 0)
        self.assertEqual(self.farm.turbines[1].x, 100)
        self.assertEqual(self.farm.turbines[1].y, 100)
        # And that we have enough turbines.
        self.assertEqual(len(self.farm.turbines), len(m)/2)

        # Test just friction
        m = np.array([10., 15.])
        self.farm.deserialize(m)
        # Test the turbine frictions are correct
        self.assertEqual(self.farm.turbines[0].friction, 10.)
        self.assertEqual(self.farm.turbines[1].friction, 15.)
        # And that we have enough turbines.
        self.assertEqual(len(self.farm.turbines), len(m))

        # Test for friction and position
        m = np.array([10., 15., 0., 0., 100., 100.])
        self.farm.deserialize(m)
        # Test the turbine positions are correct
        self.assertEqual(self.farm.turbines[0].x, 0)
        self.assertEqual(self.farm.turbines[0].y, 0)
        self.assertEqual(self.farm.turbines[1].x, 100)
        self.assertEqual(self.farm.turbines[1].y, 100)
        # Test the turbine frictions are correct
        self.assertEqual(self.farm.turbines[0].friction, 10.)
        self.assertEqual(self.farm.turbines[1].friction, 15.)
        # And that we have enough turbines.
        self.assertEqual(len(self.farm.turbines), len(m)/3)

        # Test for no parameters
        m = np.array([])
        self.assertRaises(ValueError, self.farm.deserialize, m)


    def test_boundary_constraints(self):
        # Add a 5x5 array of turbines
        self.farm.add_regular_turbine_layout(self.turbine, 5, 5)
        lower, upper = self.farm.site_boundary_constraints()

        # Check we have enough bounds
        self.assertEqual(len(lower)/2, len(self.farm.turbines))
        self.assertEqual(len(upper)/2, len(self.farm.turbines))

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
