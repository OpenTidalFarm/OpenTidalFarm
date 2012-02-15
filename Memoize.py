import numpy
import cPickle

# Implement a memoization function to avoid duplicated functional (derivative) evaluations
class MemoizeMutable:

  def __init__(self, fn):
    self.fn = fn
    self.memo = {}

  def __call__(self, *args, **kwds):
    str = cPickle.dumps(args, 1)+cPickle.dumps(kwds, 1)
    if not self.memo.has_key(str): 
      self.memo[str] = self.fn(*args, **kwds)
    return self.memo[str]

  def has_cache(self, *args, **kwds):
    str = cPickle.dumps(args, 1)+cPickle.dumps(kwds, 1)
    return str in self.memo

  # Insert a function value into the cache manually.
  def __add__(self, value, *args, **kwds):
    str = cPickle.dumps(args, 1)+cPickle.dumps(kwds, 1)
    self.memo[str] = value

