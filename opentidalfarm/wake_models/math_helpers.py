import math
import numpy
import ad
import ad.admath

def vector_difference(a, b):
    """
    Returns the difference in two vectors
    """
    if isinstance(a[0], ad.ADF) or isinstance(b[0], ad.ADF):
        return numpy.array([a[0]-b[0], a[1]-b[1]])
    else:
        return a-b if isinstance(a, numpy.ndarray) else\
           numpy.array(a)-numpy.array(b)


def l2_norm(x):
    """
    Return the l2-norm of x
    """
    return (sum([v**2 for v in x]))**0.5


def l2_norm_sq(x):
    """
    Returns the l2-norm of x before rooting
    """
    return sum([v**2 for v in x])


def normalize_vector(x):
    """
    Returns a normalized vector of length 1. Has to copy x and divide values in
    place as this method is sometimes called with numpy.ndarray of ad.ADF
    objects.
    """
    norm = l2_norm(x)
    y = numpy.array([v/norm for v in x])
    return y


def angle_between_vectors(a, b):
    """
    Returns the angle between two vectors
    """
    an = normalize_vector(a)
    bn = normalize_vector(b)
    return math.acos(numpy.dot(an, bn))


def heaviside(x, a=0, k=1):
    """
    Returns the heaviside function of x, centred at a with a sharpness of k.
    Higher k smoothens the step.
    """
    return 1./(1 + ad.admath.exp(-2*k*(x-a))) if isinstance(x, ad.ADF) else\
           1./(1 + math.exp(-2*k*(x-a)))


def box_car(x, a, b, k=1):
    """
    Returns the box_car value at x between limits a and b with a boundary
    sharpness of k.
    """
    return heaviside(x, a, k) - heaviside(x, b, k)
