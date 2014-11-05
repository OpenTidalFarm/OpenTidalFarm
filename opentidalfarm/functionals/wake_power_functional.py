import numpy

class WakePowerFunctional(object):

    def __init__(self, farm):
        farm.turbine_cache.update(farm)
        self.farm = farm
        self.power_scale = 1.0


    def power(self, velocity_pairs):
        """Computes the power output of the farm."""
        total_power = 0.0
        for pair in velocity_pairs:
            total_power += self.power_individual(pair)
        return total_power


    def power_individual(self, velocity_pair):
        """Computes the power extracted at a given velocity."""
        magnitude = numpy.linalg.norm(velocity_pair, 2)
        return self.power_scale*magnitude**3
