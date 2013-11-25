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
    def element_order(self, V):
        """
        Returns the element polynomial order
        """
        return dolfin.TestFunction(V).element().degree()


    def __init__(self, f, ind=None):
        self.f = f
        order = self.element_order(f.function_space())

        if order < 2:
            raise ValueError("For AD, the dolfin function must be of polynomial degree > 1")

        # The gradient is always in DG and of one order less
        V = dolfin.VectorFunctionSpace(self.f.function_space().mesh(),
                                       "DG", order - 1)

        # get first order derivatives
        dolfin.info("Calculating first order derivatives ...")
        self.dx = dolfin.project(self.f.dx(0), V)
        self.dy = dolfin.project(self.f.dx(1), V)
        self.grad = [self.dx, self.dy]
        # second order derivatives and second order cross derivatives
        dolfin.info("Calculating second order derivatives ...")
        self.d2xx = dolfin.project(self.grad[0].dx(0), V)
        self.d2yy = dolfin.project(self.grad[1].dx(1), V)
        self.d2xy = dolfin.project(self.grad[0].dx(1), V)
        self.grad2 = [self.d2xx, self.d2yy]

        self.ind = ind

    def __call__(self, x):
        if self.ind is not None and (isinstance(x[0], ad.ADF) and
                                     isinstance(x[1], ad.ADF)):
            ad_funcs = list(map(ad.to_auto_diff,x))
            val = self.f(x)[self.ind]
            variables = ad_funcs[0]._get_variables(ad_funcs)
            # take gradient wrt ind
            lc_wrt_args = numpy.array([self.grad[0](x)[self.ind], self.grad[1](x)[self.ind]])
            qc_wrt_args = numpy.array([self.grad2[0](x)[self.ind], self.grad2[1](x)[self.ind]])
            cp_wrt_args = self.d2xy(x)[self.ind]

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
    def element_order(self, V):
        """
        Returns the element polynomial order
        """
        return dolfin.TestFunction(V).element().degree()


    def __init__(self, f):
        """
        Calculate the first and second derivatives of f across the function
        space
        """
        self.f = f
        order = self.element_order(f.function_space())

        if order < 2:
            raise ValueError("For AD, the dolfin function must be of polynomial degree > 1")

        # The gradient is always in DG and of one order less
        V = dolfin.VectorFunctionSpace(self.f.function_space().mesh(),
                                       "DG", order - 1)

        dolfin.info("Calculating first order derivatives ...")
        self.first_deriv = dolfin.project(dolfin.grad(self.f), V)
        # split to get df/dx and df/dy
        dfdx, dfdy = self.first_deriv.split()
        # get second derivative in x and y
        dolfin.info("Calculating second order derivatives ...")
        second_deriv_x = dolfin.project(dolfin.grad(dfdx), V)
        second_deriv_y = dolfin.project(dolfin.grad(dfdy), V)
        # get cross derivative; [df/(dxdy) = df/(dydx)]
        self.cross_deriv = second_deriv_x.split()[1]
        # split second derivatives to avoid cross products
        self.second_deriv_x = second_deriv_x.split()[0]
        self.second_deriv_y = second_deriv_y.split()[1]


    def __call__(self, x):
        """
        Returns f(x) as an ad.ADF object if x is an adnumber, else returns f(x)
        """
        if isinstance(x[0], ad.ADF) and isinstance(x[1], ad.ADF):
            ad_funcs = list(map(ad.to_auto_diff,x))
            val = self.f(x)
            variables = ad_funcs[0]._get_variables(ad_funcs)
            lc_wrt_args = self.first_deriv(x)
            qc_wrt_args = numpy.array([self.second_deriv_x(x), self.second_deriv_y(x)])
            cp_wrt_args = self.cross_deriv(x)

            lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = ad._apply_chain_rule(
                    ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args)

            return ad.ADF(val, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
        else:
            return self.f(x)
