from distutils.core import setup
setup(name='OpenTidalFarm',
      version='dev',
      author='Simon Funke',
      author_email='simon.funke@gmail.com',
      url='http://www.opentidalfarm.com',
      packages = ['opentidalfarm',
                  'opentidalfarm.problems',
                  'opentidalfarm.solvers',
                  'opentidalfarm.domains',
                  'opentidalfarm.functionals',
                  'opentidalfarm.farm',
                  'opentidalfarm.turbines'],
      scripts=['scripts/convert_to_new_xml.py']
      )

