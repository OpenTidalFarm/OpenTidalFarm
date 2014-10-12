import dolfin_adjoint
import dolfin
import numpy

class MinimumDistanceConstraints(dolfin_adjoint.InequalityConstraint):
    """This class implements minimum distance constraints between turbines.

    .. note:: This class subclasses `dolfin_adjoint.InequalityConstraint`_. The
        following method names must not change:

        * ``length(self)``
        * ``function(self, m)``
        * ``jacobian(self, m)``


        _dolfin_adjoint.InequalityConstraint:
            http://dolfin-adjoint.org/documentation/api.html#dolfin_adjoint.InequalityConstraint

    """
    def __init__(self, turbine_positions, minimum_distance, controls):
        """Create MinimumDistanceConstraints

        :param serialized_turbines: The serialized turbine paramaterisation.
        :type serialized_turbines: numpy.ndarray.
        :param minimum_distance: The minimum distance allowed between turbines.
        :type minimum_distance: float.
        :raises: NotImplementedError


        """
        if len(turbine_positions)==0:
            raise NotImplementedError("Turbines must be deployed for distance "
                                      "constraints to be used.")


        self._turbines = numpy.asarray(turbine_positions).flatten().tolist()
        self._minimum_distance = minimum_distance
        self._controls = controls


    def _sl2norm(self, x):
        """Calculates the squared l2norm of a vector x."""
        return sum([v**2 for v in x])


    def length(self):
        """Returns the number of constraints ``len(function(m))``."""
        n_constraints = 0
        for i in range(len(self._turbines)):
          for j in range(len(self._turbines)):
            if i <= j:
              continue
            n_constraints += 1
        return n_constraints


    def function(self, m):
        """Return an object which must be zero for the point to be feasible.

        :param m: The serialized paramaterisation of the turbines.
        :tpye m: numpy.ndarray.
        :returns: numpy.ndarray -- each entry must be zero for a poinst to be
            feasible.

        """
        dolfin.log(dolfin.PROGRESS, "Calculating minimum distance constraints.")
        inequality_constraints = []
        for i in range(len(m)/2):
            for j in range(len(m)/2):
                if i <= j:
                    continue
                inequality_constraints.append(self._sl2norm([m[2*i]-m[2*j],
                                                             m[2*i+1]-m[2*j+1]])
                                              - self._minimum_distance**2)

        inequality_constraints = numpy.array(inequality_constraints)
        if any(inequality_constraints <= 0):
            dolfin.log(dolfin.WARNING,
                       "Minimum distance inequality constraints (should all "
                       "be > 0): %s" % inequality_constraints)
        return inequality_constraints


    def jacobian(self, m):
        """Returns the gradient of the constraint function.

        Return a list of vector-like objects representing the gradient of the
        constraint function with respect to the parameter m.

        :param m: The serialized paramaterisation of the turbines.
        :tpye m: numpy.ndarray.
        :returns: numpy.ndarray -- the gradient of the constraint function with
            respect to each input parameter m.

        """
        dolfin.log(dolfin.PROGRESS, "Calculating the jacobian of minimum "
                   "distance constraints function.")
        inequality_constraints = []

        for i in range(len(m)/2):
            for j in range(len(m)/2):
                if i <= j:
                    continue

                # Need to add space for zeros for the friction
                if self._controls.position and self._controls.friction:
                    prime_inequality_constraints = numpy.zeros(len(m*3/2))
                    friction_length = len(m)/2
                else:
                    prime_inequality_constraints = numpy.zeros(len(m))
                    friction_length = 0

                # Provide a shorter handle
                p_ineq_c = prime_inequality_constraints

                # The control vector contains the friction coefficients first,
                # so we need to shift here
                p_ineq_c[friction_length+2*i] = 2*(m[2*i] - m[2*j])
                p_ineq_c[friction_length+2*j] = -2*(m[2*i] - m[2*j])
                p_ineq_c[friction_length+2*i+1] = 2*(m[2*i+1] - m[2*j+1])
                p_ineq_c[friction_length+2*j+1] = -2*(m[2*i+1] - m[2*j+1])
                inequality_constraints.append(p_ineq_c)

        return numpy.array(inequality_constraints)
