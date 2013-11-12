import dolfin
import numpy
import ad

class ADDolfinVec(object):
    """
    A superclass used by ADDolfinVecX and ADDolfinVecY -- returns the flow in
    the direction of ind. If ad.adnumbers are used an ad.ADF object is returned
    with the value of the flow in the given direction which also holds
    information about the gradient of the flow field at that point.

    Class layout is largely based on most ad.admath functions.
    """
    def __init__(self, f, ind=None):
        self.f = f

        order = f.function_space().element().cell_shape()
        con = "DG" if "Discontinuous" in str(f.function_space()) else "CG"

        V = dolfin.VectorFunctionSpace(self.f.function_space().mesh(), con, order)
        self.dx = dolfin.project(self.f.dx(0), V)
        self.dy = dolfin.project(self.f.dx(1), V)
        self.grad = [self.dx, self.dy]
        self.ind = ind

    def __call__(self, x):
        if self.ind is not None and (isinstance(x[0], ad.ADF) and
                                     isinstance(x[1], ad.ADF)):
            ad_funcs = list(map(ad.to_auto_diff,x))
            val = self.f(x)[self.ind]
            variables = ad_funcs[0]._get_variables(ad_funcs)
            lc_wrt_args = numpy.array([self.grad[0](x)[self.ind], self.grad[1](x)[self.ind]])
            #TODO: look into why these qc and cp choices work
            qc_wrt_args = numpy.zeros(numpy.shape(lc_wrt_args))
            cp_wrt_args = 0.0

            lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = ad._apply_chain_rule(
                    ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args)

            return ad.ADF(val, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
        else:
            return self.f(x)[self.ind]


class ADDolfinVecX(ADDolfinVec):
    __doc__ = ADDolfinVec.__doc__
    def __init__(self, f):
        super(ADDolfinVecX, self).__init__(f, ind=0)

class ADDolfinVecY(ADDolfinVec):
    __doc__ = ADDolfinVec.__doc__
    def __init__(self, f):
        super(ADDolfinVecY, self).__init__(f, ind=1)


class ADDolfinExpression(object):
    def __init__(self, f):
        self.f = f

        order = f.function_space().element().cell_shape()
        con = "DG" if "Discontinuous" in str(f.function_space()) else "CG"
        V = dolfin.VectorFunctionSpace(self.f.function_space().mesh(),
                                       con, order)

        gradf = dolfin.Function(V)
        t = dolfin.TestFunction(V)
        F = (dolfin.inner(dolfin.grad(f), t) - dolfin.inner(gradf, t))*dolfin.dx
        dolfin.solve(F == 0, gradf)
        self.gradf = gradf


    def __call__(self, x):
        if isinstance(x[0], ad.ADF) and isinstance(x[1], ad.ADF):
            ad_funcs = list(map(ad.to_auto_diff,x))
            val = self.f(x)
            variables = ad_funcs[0]._get_variables(ad_funcs)
            lc_wrt_args = self.gradf(x)
            #TODO: look into why these qc and cp choices work
            qc_wrt_args = numpy.zeros(numpy.shape(lc_wrt_args))
            cp_wrt_args = 0.0

            lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = ad._apply_chain_rule(
                    ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args)

            return ad.ADF(val, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
        else:
            return self.f(x)
