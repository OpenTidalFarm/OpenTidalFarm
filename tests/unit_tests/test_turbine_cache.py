from opentidalfarm import *
import pytest

class TestTurbineCache(object):

    def create_farm(self, friction=1.0):
        turbine = BumpTurbine(diameter=20., friction=friction,
                              controls=Controls(position=True, friction=True))
        domain = RectangularDomain(0,0,100,100,10,10)
        return Farm(domain, turbine)

    def test_farm_add_turbine(self):
        position = numpy.random.rand(2)*100.
        friction = [numpy.random.rand()]

        farm = self.create_farm(friction[0])
        farm.add_turbine(position)

        # Check values are assigned correctly
        assert numpy.all(farm.turbine_cache.position==position)
        assert numpy.all(farm.turbine_cache.friction==friction)

        # Check type conversion is correct
        new_friction = [1.0]
        new_position = [(10.,20.)]
        farm.turbine_cache.friction = new_friction
        farm.turbine_cache.position = new_position

        assert numpy.all(farm.turbine_cache.position==new_position)
        assert numpy.all(farm.turbine_cache.friction==new_friction)
        assert isinstance(farm.turbine_cache.position, numpy.ndarray)
        assert isinstance(farm.turbine_cache.friction, numpy.ndarray)

    def test_turbine_cache_update(self):
        farm = self.create_farm(1.0)
        farm.add_turbine((50.,50.))

        new_position = [(10.,10.), (20.,20.)]
        new_friction = [5.0, 2.0]

        farm.turbine_cache.update(new_position, new_friction)
        assert numpy.all(farm.turbine_cache.position==new_position)
        assert numpy.all(farm.turbine_cache.friction==new_friction)

    def test_turbine_cache_initialize(self):
        farm = self.create_farm(1.0)
        farm.add_turbine((10.,20.))

        assert farm.turbine_cache.position_derivative is None
        assert farm.turbine_cache.friction_derivative is None
        assert farm.turbine_cache.turbine_field is None
        assert farm.turbine_cache.turbine_field_individual is None

        farm.turbine_cache._initialize()
        assert farm.turbine_cache.turbine_field is not None

