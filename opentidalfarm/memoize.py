import os
import signal
import cPickle
from dolfin import log, INFO, WARNING
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
        ''' sigint_save: Create a checkpoint file in case a sigint signal is received. '''
        self.fn = fn
        self.memo = {}
        self.hash_keys = hash_keys

    def __call__(self, *args, **kwds):
        h = self.get_key(args, kwds)

        if h not in self.memo:
            self.memo[h] = self.fn(*args, **kwds)
        else:
            log(INFO, "Use checkpoint value.")
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
        def sig_save(sig, stack):
            print "Received signal %i. Writing final checkpoint to disk before exiting..." % sig
            cPickle.dump(self.memo, open(filename, "wb"))
            print "Checkpoint writing finished. Bye."
            os._exit(sig)

        # Make sure we save successfully, even if the user sends a signal
        print "Save checkpoint."
        old_handler = signal.signal(signal.SIGINT, sig_save)
        cPickle.dump(self.memo, open(filename, "wb"))
        signal.signal(signal.SIGINT, old_handler)

    def load_checkpoint(self, filename):
        try:
            self.memo = cPickle.load(open(filename, "rb"))
        except IOError:
            log(WARNING, "Warning: Checkpoint file '%s' not found." % filename)
        except ValueError:
            log(WARNING, "Error: Checkpoint file '%s' is invalid." % filename)
