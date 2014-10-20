from distutils.core import setup
setup(name='OpenTidalFarm',
      version='0.2',
      author='Simon Funke',
      author_email='simon.funke@gmail.com',
      url='http://www.opentidalfarm.com',
      packages = ['opentidalfarm',
                  'opentidalfarm.problems',
                  'opentidalfarm.solvers',
                  'opentidalfarm.domains',
                  'opentidalfarm.functionals',
                  'opentidalfarm.farm',
                  'opentidalfarm.options',
                  'opentidalfarm.turbines']
      )
