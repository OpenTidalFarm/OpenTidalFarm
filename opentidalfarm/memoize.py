import cPickle
from helpers import cpu0only


def to_tuple(obj):
    if hasattr(obj, '__iter__'):
        return tuple([to_tuple(o) for o in obj])
    else:
        return obj


class MemoizeMutable:
    ''' Implements a memoization function to avoid duplicated functional (derivative) evaluations '''

    def get_key(self, args, kwds):
        h1 = to_tuple(args)
        h2 = to_tuple(kwds.items())
        h = tuple([h1, h2])
        # Often useful to have a explicit 
        # turbine parameter -> functional value mapping,
        # i.e. no hashing on the key
        if self.hash_keys:
            h = hash(h)  
        return h

    def __init__(self, fn, hash_keys=False):
        self.fn = fn
        self.memo = {}
        self.hash_keys = hash_keys

    def __call__(self, *args, **kwds):
        h = self.get_key(args, kwds)

        if h not in self.memo:
            self.memo[h] = self.fn(*args, **kwds)
        else:
            print "Using checkpoint value."
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
        try:
            self.memo = cPickle.load(open(filename, "rb"))
        except IOError:
            info_red("Checkpoint file not found.")
