#!/usr/bin/env python

import os, os.path
import sys
import subprocess
import multiprocessing
import time
from optparse import OptionParser

test_cmds = {}

parser = OptionParser()
parser.add_option("-n", type="int", dest="num_procs", default = 1, help = "To run on N cores, use -n N; to use all processors available, run test.py -n 0.")
parser.add_option("-t", type="string", dest="test_name", help = "To run one specific test, use -t TESTNAME. By default all test are run.")
parser.add_option("-s", dest="short_only", default = False, action="store_true", help = "To run the short tests only, use -s. By default all test are run.")
parser.add_option("--timings", dest="timings", default=False, action="store_true", help = "Print timings of tests.")
(options, args) = parser.parse_args(sys.argv)

if options.num_procs <= 0:
  options.num_procs = None

basedir = os.path.dirname(os.path.abspath(sys.argv[0]))
subdirs = [x for x in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, x))]
if options.test_name:
  if not options.test_name in subdirs:
    print "Specified test not found."
    sys.exit(1)
  else:
    subdirs = [options.test_name]

# Keep path variables (for buildbot's sake for instance)
orig_pythonpath = os.getenv('PYTHONPATH', '')
pythonpath = os.pathsep.join([os.path.abspath(os.path.join(basedir, os.path.pardir)), orig_pythonpath])
os.putenv('PYTHONPATH', pythonpath)

timings = {}

def f(subdir):
  test_cmd = test_cmds.get(subdir, 'make')
  if test_cmd is not None:

    print "--------------------------------------------------------"
    print "Running %s " % subdir
    print "--------------------------------------------------------"

    start_time = time.time()
    handle = subprocess.Popen(test_cmd, shell=True, cwd=os.path.join(basedir, subdir))
    exit = handle.wait()
    end_time   = time.time()
    timings[subdir] = end_time - start_time
    if exit != 0:
      print "subdir: ", subdir
      print "exit: ", exit
      return subdir
    else:
      return None

tests = sorted(subdirs)

pool = multiprocessing.Pool(options.num_procs)

fails = pool.map(f, tests)
# Remove Nones
fails = [fail for fail in fails if fail is not None]

if options.timings:
  for subdir in sorted(timings, key=timings.get, reverse=True):
    print "%s : %s s" % (subdir, timings[subdir])

if len(fails) > 0:
  print "Failures: ", set(fails)
  sys.exit(1)
