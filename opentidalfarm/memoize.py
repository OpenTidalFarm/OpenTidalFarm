import numpy
import cPickle
from helpers import cpu0only

def to_tuple(obj):
    if hasattr(obj, '__iter__'):
        return tuple([to_tuple(o) for o in obj])
    else:
        return obj

# Implement a memoization function to avoid duplicated functional (derivative) evaluations
class MemoizeMutable:

    def get_key(self, args, kwds):
        h1 = to_tuple(args)
        h2 = to_tuple(kwds.items())
        h = tuple([h1, h2])
        # h = hash(h)  # We could hash it, but it is often useful to have the
                       # explicit turbine parameter -> functional value mapping
        return h

    def __init__(self, fn):
        self.fn = fn
        self.memo = {}

    def __call__(self, *args, **kwds):
        h = self.get_key(args, kwds)

        if not self.memo.has_key(h):
            self.memo[h] = self.fn(*args, **kwds)
        return self.memo[h]

    def has_cache(self, *args, **kwds):
        h = self.get_key(args, kwds)
        return h in self.memo

    # Insert a function value into the cache manually.
    def __add__(self, value, *args, **kwds):
        h = self.get_key(args, kwds)
        self.memo[h] = value

    @cpu0only
    def save_checkpoint(self, filename):
        cPickle.dump(self.memo, open(filename, "wb"))

    def load_checkpoint(self, filename):
        self.memo = cPickle.load(open(filename, "rb"))
