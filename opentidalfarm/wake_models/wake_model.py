import analytical_wake

class AnalyticalWakeModel(object):
    """
    AnalyticalWakeModel class for interfacing with OpenTidalFarm
    """
    def __init__(self, config, flow_field, model_type='Jensen',
                 model_params=None):
        """
        Initialize the model
        """
        self.model = analytical_wake.AnalyticalWake(config, flow_field, 
                                                    model_type, model_params)

    def __call__(self, config, state, turbine_field=None, functional=None,
                 annotate=False, linear_solver="default",
                 preconditioner="default", u_source=None):
        """
        Update the turbine positions and return the total power in W
        """
        self.model.update_turbines(config.params["turbine_pos"])
        return self.model.total_power()


    def compute_functional(self, m):
        """
        Returns the power of the turbine array whilst computing dj -- saves
        computing the power and then recomputing the power when calculating the
        gradient
        """
        return self.model._total_power(m)


    def compute_gradient(self, m):
        """
        Returns the gradient of the functional for the parameter m
        """
        return self.model.grad(m)


    def compute_hessian(self, m):
        """
        Returns the hessian of the functional for the parameter m
        """
        return self.model.hess(m)
